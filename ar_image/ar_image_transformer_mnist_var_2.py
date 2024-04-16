#!/usr/bin/env python3

import argparse
from copy import deepcopy
from functools import lru_cache, reduce
import math

from einops import rearrange
import torch
from torch import distributed as dist, nn, optim
from torch.nn import functional as F
from torch.utils import data
import torch_dist_utils as du
from torchvision import datasets
from torchvision.transforms import functional as TF
from tqdm import tqdm

print = tqdm.external_write_mode()(print)
print0 = tqdm.external_write_mode()(du.print0)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].lerp_(param, 1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


@torch.compile
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


@torch.compile
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


@lru_cache
def make_block_causal_attention_mask(lengths, device=None):
    n = sum(lengths)
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    i = 0
    for length in lengths:
        mask[i:, i : i + length] = True
        i += length
    return mask


@lru_cache
def make_axial_pos(h, w, dtype=None, device=None):
    h_pos = torch.linspace(-1, 1, h + 1, dtype=dtype, device=device)
    w_pos = torch.linspace(-1, 1, w + 1, dtype=dtype, device=device)
    h_pos = (h_pos[:-1] + h_pos[1:]) / 2
    w_pos = (w_pos[:-1] + w_pos[1:]) / 2
    return torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).view(h * w, 2)


@lru_cache
def make_axial_pos_final(h, w, d, dtype=None, device=None):
    poses = []
    for i in range(d):
        pos_ax = make_axial_pos(h // 2**i, w // 2**i, dtype=dtype, device=device)
        pos_rest = torch.full((pos_ax.shape[0], 1), (d - i) / d, dtype=dtype, device=device)
        pos = torch.cat((pos_ax, pos_rest), dim=-1)
        poses.insert(0, pos)
    poses.insert(0, pos.new_zeros(1, 3))
    return torch.cat(poses, dim=-2)


class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.proj = zero_init(nn.Linear(cond_dim, dim * 2))

    def forward(self, x, cond):
        weight, bias = self.proj(cond).chunk(2, dim=-1)
        x = F.layer_norm(x, x.shape[-1:])
        return x * (weight + 1) + bias


class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        lengths = (1 * 1, 2 * 2, 4 * 4, 8 * 8, 16 * 16, 32 * 32)
        self.register_buffer("mask", make_block_causal_attention_mask(lengths))
        self.norm = AdaLN(dim, 256)
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = zero_init(nn.Linear(dim, dim))
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, head_dim // 8 * self.n_heads + 1)[:-1].exp()
        freqs = freqs.view(head_dim // 8, self.n_heads).T.contiguous()
        pos = make_axial_pos_final(32, 32, 5)
        theta_h = pos[..., None, 0:1] * freqs
        theta_w = pos[..., None, 1:2] * freqs
        theta_d = pos[..., None, 2:3] * freqs
        theta = torch.cat((theta_h, theta_w, theta_d), dim=-1)
        self.register_buffer("theta", theta)

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.view(*x.shape[:2], 3, self.n_heads, self.head_dim).transpose(1, 3).unbind(2)
        theta = self.theta[: x.shape[1]].transpose(0, 1)
        q = apply_rotary_emb_(q, theta)
        k = apply_rotary_emb_(k, theta)
        x = F.scaled_dot_product_attention(
            q, k, v, self.mask[: x.shape[1], : x.shape[1]], 0.0 if self.training else 0
        )
        x = x.transpose(1, 2).reshape(*skip.shape[:2], -1)
        x = self.out_proj(x)
        return x + skip


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = AdaLN(dim, 256)
        self.up = nn.Linear(dim, hidden_dim * 2)
        self.down = zero_init(nn.Linear(hidden_dim, dim))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up(x)
        x = swiglu(x)
        x = self.down(x)
        return x + skip


class Block(nn.Module):
    def __init__(self, dim, hidden_dim, head_dim):
        super().__init__()
        self.attn = SelfAttention(dim, head_dim)
        self.ff = FeedForward(dim, hidden_dim)

    def forward(self, x, cond):
        x = self.attn(x, cond)
        x = self.ff(x, cond)
        return x


class Transformer(nn.Module):
    def __init__(self, depth, dim, hidden_dim, head_dim, vocab_size, n_classes):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.class_embed = nn.Embedding(n_classes, 256)
        self.start_token = nn.Parameter(torch.randn(dim))
        self.image_embed = nn.Embedding(vocab_size, dim)
        self.embed_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_head = nn.Linear(dim, vocab_size)
        self.image_embed.weight = self.out_head.weight

    def forward(self, x):
        x, y = x[:, 1:], x[:, :1]
        x = self.image_embed(x)
        cond = self.class_embed(y)
        x = torch.cat((self.start_token.expand(x.shape[0], 1, -1), x), dim=1)
        x = self.embed_drop(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.out_norm(x)
        x = self.out_head(x)
        return x


def prepare_seqs(x, y, encode_fn):
    tgts, inps = [], []
    while True:
        tgts.insert(0, encode_fn(x).flatten(1))
        if x.shape[2] == 1 and x.shape[3] == 1:
            break
        x = F.avg_pool2d(x, 2)
    inps.append(y[:, None])
    for tgt in tgts[:-1]:
        size = math.isqrt(tgt.shape[-1])
        inp = rearrange(tgt, "n (h w) -> n 1 h w", h=size, w=size)
        inp = F.interpolate(inp.float(), scale_factor=2, mode="nearest")
        inps.append(inp.long().flatten(1))
    return torch.cat(inps, dim=-1), torch.cat(tgts, dim=-1)


def gumbel(loc, scale):
    g = torch.empty_like(loc).exponential_().log_().neg_().mul_(scale)
    return loc + g


def sample_categorical(logits, temperature=1.0):
    return torch.argmax(gumbel(logits, temperature), dim=-1)


def apply_top_p(logits, p):
    """Returns logits with tokens not in the top p fraction of probability mass masked out."""
    probs = torch.softmax(logits, dim=-1)
    probs_sorted, indices = torch.sort(probs, dim=-1, descending=True)
    probs_cumsum = torch.cumsum(probs_sorted, dim=-1)
    drop = probs_cumsum[..., :-1] >= p
    drop = torch.cat((drop.new_zeros(*drop.shape[:-1], 1), drop), dim=-1)
    drop_unsorted = torch.zeros_like(drop).scatter_(-1, indices, drop)
    return torch.masked_fill(logits, drop_unsorted, float("-inf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="run")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    batch_size = 64

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(args.seed)
    model_raw = Transformer(8, 384, 1024, 64, 256, 10).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    def tf(x):
        x = TF.to_tensor(x)
        x = F.pad(x, (2, 2, 2, 2))
        return x

    dataset = datasets.MNIST("data", train=True, download=True, transform=tf)
    sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, drop_last=True
    )

    wd, no_wd = [], []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_wd.append(param)
        else:
            wd.append(param)
    groups = [{"params": wd}, {"params": no_wd, "weight_decay": 0}]
    opt = optim.AdamW(groups, lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
    sched = optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, i / 100))

    epoch = 0

    @torch.no_grad()
    def sample(model, y, temperature=1.0, top_p=1.0):
        x = inp = y[:, None]
        scale = 1
        scales = 6
        for i in range(scales):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(inp)[:, -scale * scale :].float()
            if top_p < 1.0:
                logits = apply_top_p(logits, top_p)
            sample = sample_categorical(logits, temperature)
            if i > 0:
                x = torch.cat((x, sample), dim=1)
            if i < scales - 1:
                last_scale = rearrange(sample.float(), "n (h w) -> n 1 h w", h=scale, w=scale)
                last_scale = F.interpolate(last_scale, scale_factor=2, mode="nearest")
                inp = torch.cat((x, last_scale.flatten(1).long()), dim=1)
                scale *= 2
        return sample

    def demo():
        y = torch.arange(10, device=device).repeat_interleave(10)
        x = sample(model_ema, y)
        x = rearrange(x, "(nh nw) (h w) -> 1 (nh h) (nw w)", nh=10, nw=10, h=32, w=32)
        x = x.float() / 255
        x = torch.clamp(x, 0, 1)
        TF.to_pil_image(x.cpu()).save(f"demo_{args.name}_{epoch:04}.png")

    while True:
        sampler.set_epoch(epoch)
        if rank == 0:
            demo()
        dist.barrier()

        for step, (x, y) in enumerate(tqdm(dataloader, disable=rank > 0)):
            x = x.to(device)
            y = y.to(device)
            inps, tgts = prepare_seqs(x, y, lambda x: x.mul(255).clamp(0, 255).round().long())
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(inps)
                loss = torch.compile(F.cross_entropy)(logits.mT, tgts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ema_update(model_raw, model_ema, 0.98)
            dist.all_reduce(loss, dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

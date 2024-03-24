#!/usr/bin/env python3

from copy import deepcopy
from functools import lru_cache, reduce
import math

from einops import rearrange
import flash_attn
import numpy as np
import torch
from torch import distributed as dist, nn, optim
import torch.distributed.nn as dnn
from torch.nn import functional as F
from torch.utils import data
import torch_dist_utils as du
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm import trange, tqdm

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


# TODO: do this correctly instead
@lru_cache
def make_axial_pos(h, w, dtype=None, device=None):
    h_pos = torch.linspace(-1, 1, h + 1, dtype=dtype, device=device)
    w_pos = torch.linspace(-1, 1, w + 1, dtype=dtype, device=device)
    h_pos = (h_pos[:-1] + h_pos[1:]) / 2
    w_pos = (w_pos[:-1] + w_pos[1:]) / 2
    return torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).view(h * w, 2)


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
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim, dropout):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.norm = nn.LayerNorm((dim,))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = zero_init(nn.Linear(dim, dim, bias=False))
        self.drop = nn.Dropout(dropout)
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, head_dim // 8 + 1)[:-1].exp()
        # TODO: allow changing image size
        pos = make_axial_pos(32, 32)
        # make room for the class token
        # TODO: use adanorm for this
        pos = torch.cat((torch.zeros(1, 2), pos))
        theta_h = pos[..., 0:1] * freqs
        theta_w = pos[..., 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("theta", theta)

    def forward(self, x, cache=None, index=None):
        x = self.norm(x)
        qkv = self.qkv_proj(x).view(*x.shape[:2], 3, self.n_heads, self.head_dim)
        if cache is None:
            pos = torch.arange(x.shape[1], device=x.device)
            q, k, v = qkv.unbind(2)
            theta_1 = self.theta[pos][..., None, :].expand(-1, self.n_heads // 2, -1)
            theta_2 = self.theta[pos + 1][..., None, :].expand(-1, self.n_heads // 2, -1)
            theta = torch.cat((theta_1, theta_2), dim=-2)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            x = flash_attn.flash_attn_func(q, k, v, self.drop.p if self.training else 0.0, causal=True)
        else:
            assert not (x.shape[1] > 1 and index != 0)
            end_index = index + x.shape[1]
            q, k, v = qkv.unbind(2)
            pos = torch.arange(x.shape[1], device=x.device)
            q, k, v = qkv.unbind(2)
            theta_1 = self.theta[index:end_index][..., None, :].expand(-1, self.n_heads // 2, -1)
            theta_2 = self.theta[index + 1 : end_index + 1][..., None, :].expand(-1, self.n_heads // 2, -1)
            theta = torch.cat((theta_1, theta_2), dim=-2)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            cache[0][:, index:end_index] = k
            cache[1][:, index:end_index] = v
            x = flash_attn.flash_attn_func(
                q, cache[0][:, :end_index], cache[1][:, :end_index], causal=index == 0
            )
        x = x.view(*x.shape[:2], -1)
        x = self.out_proj(x)
        x = self.drop(x)
        return x


@torch.compile
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.up = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.drop_1 = nn.Dropout(dropout)
        self.down = zero_init(nn.Linear(hidden_dim, dim, bias=False))
        self.drop_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = swiglu(x)
        x = self.drop_1(x)
        x = self.down(x)
        x = self.drop_2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, hidden_dim, head_dim, dropout):
        super().__init__()
        self.attn = SelfAttention(dim, head_dim, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)

    def forward(self, x, cache=None, index=None):
        x = x + self.attn(x, cache, index)
        x = x + self.ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self, depth, dim, hidden_dim, head_dim, vocab_size, dropout):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.class_embed = nn.Embedding(10, dim)
        self.image_embed = nn.Embedding(vocab_size, dim)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        # self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        from mos import MixtureOfSoftmaxHead
        self.out_proj = MixtureOfSoftmaxHead(dim, vocab_size, dim * 2, 8, 2)
        # self.out_proj.compile()

    def init_cache(self, batch_size, seq_len, dtype=None, device=None):
        cache = [[] for _ in range(self.depth)]
        for item in cache:
            for _ in range(2):
                item.append(
                    torch.zeros(
                        batch_size,
                        seq_len,
                        self.n_heads,
                        self.head_dim,
                        dtype=dtype,
                        device=self.class_embed.weight.device if device is None else device,
                    )
                )
        return cache

    def forward(self, x, y, cache=None, index=None):
        x = self.image_embed(x)
        index = 0 if index is None else index
        y = self.class_embed(y)
        x = torch.cat((y[:, None], x), dim=1)
        x = x[:, -1:] if cache else x
        x = self.embed_drop(x)
        cache = [None] * self.depth if cache is None else cache
        for block, cache_block in zip(self.blocks, cache):
            # x = torch.utils.checkpoint.checkpoint(block, x, cache_block, index)
            x = block(x, cache_block, index)
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x


@lru_cache
def make_color_cube(levels, dtype=None, device=None):
    ch = torch.linspace(0, 1, levels, dtype=dtype, device=device)
    return torch.stack(torch.meshgrid(ch, ch, ch, indexing="ij"), dim=-1).flatten(0, 2)


def quantize(x, palette):
    return torch.cdist(x, palette).argmin(dim=-1)


def quantize_with_noise(x, levels):
    palette = make_color_cube(levels, dtype=x.dtype, device=x.device)
    spacing = 1 / (levels - 1)
    noise = torch.rand_like(x).mul_(spacing).sub_(spacing / 2)
    return quantize(x + noise, palette)


def dequantize(x, palette):
    return palette[x]


def main():
    batch_size = 32

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # n_levels = 24
    # palette = make_color_cube(n_levels, device=device)
    # vocab_size = n_levels**3
    vocab_size = 4096

    model_raw = Transformer(8, 512, 1360, 64, vocab_size, dropout=0.0).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    dataset = datasets.CIFAR10("data", train=True, download=True, transform=np.array)

    print0("Computing palette...")
    perm = np.random.permutation(len(dataset))
    dataset_for_palette = np.stack([dataset[perm[i]][0] for i in range(1000)])
    dataset_for_palette = torch.from_numpy(dataset_for_palette).to(device)
    dataset_for_palette = dataset_for_palette.view(-1, 3).float() / 255
    palette = dataset_for_palette[torch.randperm(dataset_for_palette.shape[0], device=device)[:vocab_size]]
    for _ in trange(100, disable=rank != 0):
        indices = quantize(dataset_for_palette, palette)
        palette.index_reduce_(0, indices, dataset_for_palette, "mean", include_self=False)
    du.broadcast_tensors(palette)
    del dataset_for_palette

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
    opt = optim.AdamW(groups, lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)

    epoch = 0

    @torch.no_grad()
    def sample(model, y, temperature=1.0, top_p=1.0, disable=False):
        n = y.shape[0]
        n_proc = math.ceil(n / world_size)
        y = y.split(n_proc)[rank]
        x = torch.zeros(y.shape[0], 0, dtype=torch.long, device=device)
        cache = model.init_cache(y.shape[0], 1024, dtype=torch.bfloat16, device=device)
        index = 0
        for _ in trange(1024, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, y, cache, index)[:, -1].float()
            if temperature != 1.0:
                logits /= temperature
            if top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for i in range(logits.shape[0]):
                    indices_to_remove = sorted_indices[sorted_indices_to_remove][i]
                    logits[i].scatter_(dim=-1, index=indices_to_remove, value=float("-inf"))
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.argmax(logits + gumbel, dim=-1)
            x = torch.cat((x, sample[:, None]), dim=1)
            index += 1
        # return torch.cat(dnn.all_gather(x))[:n]
        return torch.cat(du.all_gather_into_new(x))[:n]

    def demo():
        y = torch.arange(10, device=device).repeat_interleave(10)
        x = sample(model_ema, y)
        if rank == 0:
            x = dequantize(x, palette)
            x = rearrange(x, "(nh nw) (h w) d -> d (nh h) (nw w)", nh=10, nw=10, h=32, w=32)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_cifar_q_057_{epoch:04}.png")

    while True:
        sampler.set_epoch(epoch)
        if epoch > 0:
            demo()

        for step, (x, y) in enumerate(tqdm(dataloader, disable=rank > 0)):
            x = x.to(device).flatten(1, 2).float() / 255
            # x = quantize_with_noise(x, n_levels)
            x = quantize(x, palette)
            y = y.to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x[:, :-1], y).float()
            loss = F.cross_entropy(logits.mT, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.95)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

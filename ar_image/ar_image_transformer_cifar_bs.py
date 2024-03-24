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
    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.norm = nn.LayerNorm((dim,))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = zero_init(nn.Linear(dim, dim, bias=False))
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

    def forward(self, x, pos, cache=None, index=None):
        x = self.norm(x)
        qkv = self.qkv_proj(x).view(*x.shape[:2], 3, self.n_heads, self.head_dim)
        if cache is None:
            q, k, v = qkv.unbind(2)
            q = apply_rotary_emb_(q, self.theta[pos][:, :, None, :])
            k = apply_rotary_emb_(k, self.theta[pos][:, :, None, :])
            x = flash_attn.flash_attn_func(q, k, v, causal=True)
        else:
            assert not (x.shape[1] > 1 and index != 0)
            end_index = index + x.shape[1]
            pos = pos[..., -1:]
            q, k, v = qkv.unbind(2)
            q = apply_rotary_emb_(q, self.theta[pos][:, :, None, :])
            k = apply_rotary_emb_(k, self.theta[pos][:, :, None, :])
            cache[0][:, index:end_index] = k
            cache[1][:, index:end_index] = v
            x = flash_attn.flash_attn_func(
                q, cache[0][:, :end_index], cache[1][:, :end_index], causal=index == 0
            )
        x = x.view(*x.shape[:2], -1)
        return self.out_proj(x)


@torch.compile
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.up = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.down = zero_init(nn.Linear(hidden_dim, dim, bias=False))

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = swiglu(x)
        x = self.down(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, hidden_dim, head_dim):
        super().__init__()
        self.attn = SelfAttention(dim, head_dim)
        self.ff = FeedForward(dim, hidden_dim)

    def forward(self, x, pos, cache=None, index=None):
        x = x + self.attn(x, pos, cache, index)
        x = x + self.ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self, depth, dim, hidden_dim, head_dim, vocab_size):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.class_embed = nn.Embedding(10, dim)
        self.image_embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

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

    def forward(self, x, pos, cache=None, index=None):
        y = self.class_embed(x[:, :1])
        x = self.image_embed(x[:, 1:])
        x = torch.cat((y, x), dim=1)
        index = 0 if index is None else index
        x = x[:, -1:] if cache else x
        cache = [None] * self.depth if cache is None else cache
        for block, cache_block in zip(self.blocks, cache):
            # x = torch.utils.checkpoint.checkpoint(block, x, cache_block, index)
            x = block(x, pos, cache_block, index)
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


def sm_make_input(a, bs_tok):
    l = a.shape[0]
    x = np.copy(a)
    y = np.zeros((l,), np.int64)
    t = np.ones((l,), np.bool_)
    m = np.zeros((l, l), np.bool_)
    p = np.zeros((l,), np.int64)

    c = 0
    d = 0

    for i in range(0, l):
        m[i] = m[max(i - 1, 0)]
        if i > 0:
            y[i - 1] = a[i]
        if a[i] == bs_tok:
            m[i, c] = 0
            m[i, d] = 0
            m[i, i] = 1
            x[i] = x[c]
            t[c] = 0
            p[i] = p[c]
            d = i
            nzs = np.nonzero(m[i, 0:c])[0]
            c = nzs[-1] if nzs.size else 0
        else:
            m[i, i] = 1
            d = d + 1
            nzs = np.nonzero(m[i, c + 1:])[0] + c + 1
            c = nzs[0] if nzs.size else 0
            p[i] = p[d - 1] + 1
        if i == 0:
            c = 0
            d = 0
        if i == 1:
            c = 0
            d = 1

    t[-1] = 0
    return x, y, t, m, p


def noise_seq(seq, n_toks, bs_tok, n_noise):
    noise_indices = np.random.randint(1, len(seq) - 1, (n_noise,))
    for i in range(n_noise):
        noise_index = noise_indices[i]
        noise_token = np.random.randint(0, n_toks)
        bs_seq = np.stack((noise_token, np.array(bs_tok)))
        seq = np.insert(seq, noise_index, bs_seq)
        noise_indices = np.where(noise_indices >= noise_index, noise_indices + 2, noise_indices)
    return sm_make_input(seq, bs_tok)


def main():
    batch_size = 32

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    n_levels = 24
    palette = make_color_cube(n_levels, device=device)
    palette_cpu = palette.cpu()
    vocab_size = n_levels**3 + 1
    bs_tok = vocab_size - 1
    n_noise = 20

    model_raw = Transformer(8, 512, 1360, 64, vocab_size).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    def tf(x):
        x = torch.tensor(np.array(x))
        x = x.flatten(0, 1).float() / 255
        # x = quantize_with_noise(x, n_levels)
        x = quantize(x, palette_cpu)
        return noise_seq(x.numpy(), bs_tok, bs_tok, n_noise)

    dataset = datasets.CIFAR10("data", train=True, download=True, transform=tf)
    sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, drop_last=True, num_workers=16,
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
        n_proc = n // world_size
        x = y.split(n_proc)[rank]
        pos = torch.zeros(n_proc, 1, dtype=torch.long, device=device)
        cache = model.init_cache(n_proc, 1280, dtype=torch.bfloat16, device=device)
        index = 0
        for i in trange(1280, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, pos, cache, index)[:, -1].float()
            if temperature != 1.0:
                logits /= temperature
            if top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for j in range(logits.shape[0]):
                    indices_to_remove = sorted_indices[sorted_indices_to_remove][j]
                    logits[j].scatter_(dim=-1, index=indices_to_remove, value=float("-inf"))
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.argmax(logits + gumbel, dim=-1)
            x = torch.cat((x, sample[:, None]), dim=1)
            new_pos_good = torch.clamp_max(pos[..., -1:] + 1, 1024)
            new_pos_bad = torch.clamp_min(pos[..., -1:] - 1, 0)
            new_pos = torch.where(sample[:, None] != bs_tok, new_pos_good, new_pos_bad)
            pos = torch.cat((pos, new_pos), dim=-1)
            index += 1
            if new_pos.min() == 1024:
                break
        x_clean = torch.zeros(n_proc, 1024, dtype=torch.long, device=device)
        for i in range(x.shape[0]):
            for j in range(1, x.shape[1]):
                j_dst = pos[i, j - 1]
                if j_dst == 1024:
                    break
                x_clean[i, j_dst] = x[i, j]
        return torch.cat(dnn.all_gather(x_clean))

    def demo(step):
        y = torch.arange(10, device=device).repeat_interleave(10)
        x = sample(model_ema, y[:, None])
        if rank == 0:
            x = dequantize(x.clamp_max(palette.shape[0] - 1), palette)
            x = rearrange(x, "(nh nw) (h w) d -> d (nh h) (nw w)", nh=10, nw=10, h=32, w=32)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_cifar_bs_021_{epoch:04}_{step:05}.png")

    while True:
        sampler.set_epoch(epoch)
        for step, ((x, tgt, mask, _, pos), y) in enumerate(tqdm(dataloader, disable=rank > 0)):
            if step % 100 == 0:
                demo(step)

            x = x.to(device)
            tgt = tgt.to(device)
            pos = pos.to(device)
            mask = mask.to(device)
            y = y.to(device)
            tgt = torch.cat((x[:, :1], tgt), dim=1)[:, :-1]
            mask = torch.cat((torch.ones_like(y[:, None], dtype=torch.bool), mask), dim=1)[:, :-1]
            pos = torch.cat((torch.zeros_like(y[:, None]), pos), dim=1)[:, :-1]
            x = torch.cat((y[:, None], x), dim=1)[:, :-1]

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits_for_noise = model_ema(x, pos).float()
            logits_for_noise.scatter_(2, tgt[:, :, None], float("-inf"))
            logits_for_noise[:, :, -1] = float("-inf")
            gumbel = torch.rand_like(logits_for_noise).log_().nan_to_num_().neg_().log_().neg_()
            samples = torch.argmax(logits_for_noise + gumbel, dim=-1)
            x[:, 1:] = torch.where(mask[:, :-1], x[:, 1:], samples[:, :-1])
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, pos).float()
            losses = F.cross_entropy(logits.mT, tgt, reduction="none")
            loss = (losses * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.98)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

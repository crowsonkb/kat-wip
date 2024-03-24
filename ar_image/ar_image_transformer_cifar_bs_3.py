#!/usr/bin/env python3

from copy import deepcopy
from functools import lru_cache, reduce
import math

from einops import rearrange
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
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = zero_init(nn.Linear(dim, dim))
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, self.n_heads * head_dim // 4 + 1)[:-1].exp()
        freqs = freqs.view(head_dim // 4, self.n_heads).T.contiguous()
        # TODO: allow changing image size
        pos = make_axial_pos(32, 32)
        # make room for the class token and the last backspace token
        pos = torch.cat((torch.zeros(1, 2), pos))
        theta_h = pos[..., None, 0:1] * freqs
        theta_w = pos[..., None, 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("theta", theta)

    def forward(self, x, pos, mask=None, cache=None, index=None, bs_training=False):
        x = self.norm(x)
        qkv = self.qkv_proj(x).reshape(x.shape[0], -1, 3, self.n_heads, self.head_dim)  # n s t h d
        # transpose to n h t s d
        qkv = qkv.transpose(1, 3)
        q, k, v = qkv.unbind(2)
        if bs_training:
            q = apply_rotary_emb_(q, self.theta[pos].transpose(1, 2).to(q))
            k = apply_rotary_emb_(k, self.theta[pos].transpose(1, 2).to(k))
            k_tmp = torch.cat((cache[0], k), dim=2)
            v_tmp = torch.cat((cache[1], v), dim=2)
            n = x.shape[-2]
            cache_part = torch.ones((n, n), dtype=torch.bool, device=x.device).tril(-1)
            main_part = torch.eye(n, dtype=torch.bool, device=x.device)
            mask = torch.cat((cache_part, main_part), dim=-1)
            x = F.scaled_dot_product_attention(q, k_tmp, v_tmp, mask, 0.0 if self.training else 0)
        elif index is None:
            q = apply_rotary_emb_(q, self.theta[pos].transpose(1, 2).to(q))
            k = apply_rotary_emb_(k, self.theta[pos].transpose(1, 2).to(k))
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0 if self.training else 0, is_causal=True)
            if cache is not None:
                cache[0][:] = k
                cache[1][:] = v
        else:
            b_idx = torch.arange(x.shape[0], device=x.device)
            max_index = torch.amax(index) + 1
            mask = torch.ones(x.shape[0], pos.shape[-1], pos.shape[-1], dtype=torch.bool, device=x.device).tril_()
            mask = mask[b_idx, index, :max_index][:, None, None]
            q = apply_rotary_emb_(q, self.theta[pos[b_idx, index]][:, None].transpose(1, 2).to(q))
            k = apply_rotary_emb_(k, self.theta[pos[b_idx, index]][:, None].transpose(1, 2).to(k))
            cache[0][b_idx, :, index] = k[:, :, 0, :]
            cache[1][b_idx, :, index] = v[:, :, 0, :]
            x = F.scaled_dot_product_attention(q, cache[0][:, :, :max_index], cache[1][:, :, :max_index], mask)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.n_heads * self.head_dim)
        x = self.out_proj(x)
        return x


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

    def forward(self, x, pos, mask=None, cache=None, index=None, bs_training=False):
        x = x + self.attn(x, pos, mask, cache, index, bs_training)
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
        self.embed_drop = nn.Dropout(0.0)
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
                        self.n_heads,
                        seq_len,
                        self.head_dim,
                        dtype=dtype,
                        device=self.class_embed.weight.device if device is None else device,
                    )
                )
        return cache

    def forward(self, x, pos, mask=None, cache=None, index=None, bs_training=False):
        y = self.class_embed(x[:, :1])
        x = self.image_embed(x[:, 1:])
        x = torch.cat((y, x), dim=1)
        b_idx = torch.arange(x.shape[0], device=x.device)
        x = x if index is None else x[b_idx, index][:, None]
        cache = [None] * self.depth if cache is None else cache
        self.embed_drop(x)
        for block, cache_block in zip(self.blocks, cache):
            # x = torch.utils.checkpoint.checkpoint(block, x, cache_block, index, bs_training)
            x = block(x, pos, mask, cache_block, index, bs_training)
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

    # n_levels = 24
    # palette = make_color_cube(n_levels, device=device)
    # palette_cpu = palette.cpu()
    # vocab_size = n_levels**3 + 1
    vocab_size = 4096 + 1
    bs_tok = vocab_size - 1
    n_noise = 50

    model_raw = Transformer(8, 512, 1360, 64, vocab_size).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    dataset = datasets.CIFAR10("data", train=True, download=True)
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
    palette_cpu = palette.cpu()
    del dataset_for_palette

    def tf(x):
        x = torch.tensor(np.array(x))
        x = x.flatten(0, 1).float() / 255
        # x = quantize_with_noise(x, n_levels)
        x = quantize(x, palette_cpu)
        # return noise_seq(x.numpy(), vocab_size, bs_tok, n_noise)
        return x

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
        y = y.split(n_proc)[rank]
        x = torch.full((y.shape[0], 1024 + 2), 0, dtype=torch.int64, device=device)
        x[:, :1] = y
        pos = torch.zeros(n_proc, 1, dtype=torch.long, device=device)
        cache = model.init_cache(n_proc, 1024 + 2, dtype=torch.bfloat16, device=device)
        index = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        b_idx = torch.arange(x.shape[0], device=device)
        bs_count = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        for i in trange(1536, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, pos, cache=cache, index=index)[:, -1].float()
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
            sample = torch.argmax(logits + gumbel * temperature, dim=-1)
            good_toks = (index == 0) | (sample != bs_tok)
            index = torch.where(good_toks, index + 1, index - 1).clamp_max_(1024)
            x[b_idx, index] = torch.where(good_toks, sample, x[b_idx, index])
            new_pos = pos[:, -1:] + 1
            pos = torch.cat((pos, new_pos), dim=-1)
            bs_count += (sample == bs_tok)
            if torch.amin(index) == 1024:
                break
        return torch.cat(dnn.all_gather(x[:, 1:1025])), torch.cat(dnn.all_gather(bs_count))

    def demo(step):
        y = torch.arange(10, device=device).repeat_interleave(10)
        x, bs_count = sample(model_ema, y[:, None])
        if rank == 0:
            x = dequantize(x.clamp_max(palette.shape[0] - 1), palette)
            x = rearrange(x, "(nh nw) (h w) d -> d (nh h) (nw w)", nh=10, nw=10, h=32, w=32)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_cifar_bs_3_004_{epoch:04}_{step:05}.png")
            print(f"min bs count: {bs_count.min().item()}")
            print(f"mean bs count: {bs_count.float().mean().item():g}")
            print(f"max bs count: {bs_count.max().item()}")

    while True:
        sampler.set_epoch(epoch)
        for step, (x, y) in enumerate(tqdm(dataloader, disable=rank > 0)):
            if not (epoch == 0 and step == 0) and step % 100 == 0:
                demo(step)

            x, y = x.to(device), y.to(device)
            x = torch.cat((y[:, None], x), dim=1)
            seq_len = x.shape[1] - 1
            pos = torch.arange(seq_len, device=device).tile((x.shape[0], 1))
            attn_mask = torch.ones(x.shape[0], seq_len, seq_len, dtype=torch.bool, device=device).tril()
            cache = model_raw.init_cache(x.shape[0], seq_len, dtype=torch.bfloat16, device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x[:, :-1], pos, attn_mask, cache).float()
            logits_for_noise = logits.clone()
            # logits_for_noise.scatter_(2, x[:, 1:, None], float("-inf"))
            logits_for_noise[:, :, -1] = float("-inf")
            gumbel = torch.rand_like(logits_for_noise).log_().nan_to_num_().neg_().log_().neg_()
            samples = torch.argmax(logits_for_noise + gumbel, dim=-1)
            samples_in = torch.cat((y[:, None], samples), dim=1)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits_m = model(samples_in[:, :-1], pos, attn_mask, cache, bs_training=True).float()
            loss_main = F.cross_entropy(logits.mT, x[:, 1:])
            bs_tgt = torch.full_like(x, bs_tok, dtype=torch.long, device=device)
            loss_bs = F.cross_entropy(logits_m.mT, bs_tgt[:, 1:])
            loss = loss_main + loss_bs * 0.1
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.95)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}, main: {loss_main.item():g}, bs: {loss_bs.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

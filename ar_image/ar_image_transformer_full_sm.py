#!/usr/bin/env python3

import argparse
from copy import deepcopy
from functools import lru_cache
import math

from einops import rearrange
import numpy as np
import torch
from torch import distributed as dist, nn, optim
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


def sm_loss(s_d, y_d, t_d, p_d, s_m, y_m, t_m, p_m, n, alpha=0.01, divergence="chi2-mixture"):
    if divergence not in ("chi2", "chi2-mixture", "kl"):
        raise ValueError(f"Unknown divergence: {divergence}, must be chi2, chi2-mixture, or kl")

    def phi(x):
        if divergence in ("chi2", "chi2-mixture"):
            return x - x**2 / (4 * alpha)
        return -torch.expm1(-x)

    gamma = n / (n + 1)
    gamma_d = t_d * gamma**p_d
    gamma_m = t_m * gamma**p_m
    q_d = s_d.gather(2, y_d[:, :, None])[:, :, 0]
    v_d = torch.logsumexp(s_d, dim=2)
    v_m = torch.logsumexp(s_m, dim=2)

    qv_diff_d = gamma_d[:, :-1] * phi(alpha * q_d[:, :-1] - gamma * alpha * v_d[:, 1:]) / alpha
    qv_diff_d_eos = gamma_d[:, -1] / (alpha * (1 - gamma)) * phi(alpha * (1 - gamma) * v_d[:, -1])
    v_diff_d = gamma_d[:, :-1] * (v_d[:, :-1] - gamma * v_d[:, 1:]) / 2
    v_diff_d_eos = gamma_d[:, -1] * v_d[:, -1] / 2
    v_diff_m = gamma_m[:, :-1] * (v_m[:, :-1] - gamma * v_m[:, 1:]) / 2
    v_diff_m_eos = gamma_m[:, -1] * v_m[:, -1] / 2
    loss_d = qv_diff_d.sum(1) + qv_diff_d_eos - v_diff_d.sum(1) - v_diff_d_eos
    loss_m = -v_diff_m.sum(1) - v_diff_m_eos

    if divergence == "chi2-mixture":
        q_m = s_m.gather(2, y_m[:, :, None])[:, :, 0]
        qv_diff_m = gamma_m[:, :-1] * (alpha * q_m[:, :-1] - gamma * alpha * v_m[:, 1:]) ** 2
        qv_diff_m_eos = gamma_m[:, -1] / (1 - gamma) * (alpha * (1 - gamma) * v_m[:, -1]) ** 2
        loss_m = loss_m - qv_diff_m.sum(1) - qv_diff_m_eos

    return -(loss_d.mean() + loss_m.mean()) / n


def sm_loss(s_d, y_d, t_d, p_d, s_m, y_m, t_m, p_m, gamma, alpha=1.0, chi2_mix_fac=0.0):
    def phi(x):
        return x - (1 - chi2_mix_fac) * alpha * x**2 / 4

    def psi(x):
        return -chi2_mix_fac * alpha * x**2 / 4

    eos_d = torch.argmax(t_d * p_d, dim=1)
    eos_m = torch.argmax(t_m * p_m, dim=1)
    gamma_d = t_d * gamma**p_d
    gamma_m = t_m * gamma**p_m
    gamma_eos_d = gamma_d.gather(1, eos_d[:, None])[:, 0]
    gamma_eos_m = gamma_m.gather(1, eos_m[:, None])[:, 0]
    q_d = s_d.gather(2, y_d[:, :, None])[:, :, 0]
    q_m = s_m.gather(2, y_m[:, :, None])[:, :, 0]
    v_d = torch.logsumexp(s_d, dim=2)
    v_m = torch.logsumexp(s_m, dim=2)
    v_eos_d = v_d.gather(1, eos_d[:, None])[:, 0]
    v_eos_m = v_m.gather(1, eos_m[:, None])[:, 0]

    qv_diff_d = gamma_d[:, :-1] * phi(q_d[:, :-1] - gamma * v_d[:, 1:])
    qv_diff_d_eos = gamma_eos_d * phi((1 - gamma) * v_eos_d) / (1 - gamma)
    qv_diff_m = gamma_m[:, :-1] * psi(q_m[:, :-1] - gamma * v_m[:, 1:])
    qv_diff_m_eos = gamma_eos_m * psi((1 - gamma) * v_eos_m) / (1 - gamma)
    v_diff_d = gamma_d[:, :-1] * (v_d[:, :-1] - gamma * v_d[:, 1:]) / 2
    v_diff_d_eos = gamma_eos_d * v_eos_d / 2
    v_diff_m = gamma_m[:, :-1] * (v_m[:, :-1] - gamma * v_m[:, 1:]) / 2
    v_diff_m_eos = gamma_eos_m * v_eos_m / 2
    loss_d = v_diff_d.sum(1) + v_diff_d_eos - qv_diff_d.sum(1) - qv_diff_d_eos
    loss_m = v_diff_m.sum(1) + v_diff_m_eos - qv_diff_m.sum(1) - qv_diff_m_eos

    return loss_d.mean() + loss_m.mean()


def simple_sm_loss(logits, targets, gamma, alpha=1.0, mask=None, pos=None):
    def phi(x):
        return x - alpha * x**2 / 4

    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    if pos is None:
        pos = torch.cumsum(mask, dim=-1)

    eos = torch.argmax(pos * mask, dim=-1)
    gamma_i = mask * gamma**pos
    gamma_i_eos = gamma_i.gather(-1, eos[..., None])[..., 0]
    q = logits.gather(-1, targets[..., None])[..., 0]
    v = torch.logsumexp(logits, dim=-1)
    v_eos = v.gather(-1, eos[..., None])[..., 0]

    qv_diff = gamma_i[..., :-1] * phi(q[..., :-1] - gamma * v[..., 1:])
    qv_diff_eos = gamma_i_eos * phi((1 - gamma) * v_eos) / (1 - gamma)
    v_diff = gamma_i[..., :-1] * (v[..., :-1] - gamma * v[..., 1:])
    v_diff_eos = gamma_i_eos * v_eos
    loss = v_diff.sum(-1) + v_diff_eos - qv_diff.sum(-1) - qv_diff_eos

    return torch.mean(loss)


# TODO: do this correctly instead
@lru_cache
def make_axial_pos(h, w, dtype=None, device=None):
    h_pos = torch.linspace(-1, 1, h + 1, dtype=dtype, device=device)
    w_pos = torch.linspace(-1, 1, w + 1, dtype=dtype, device=device)
    h_pos = (h_pos[:-1] + h_pos[1:]) / 2
    w_pos = (w_pos[:-1] + w_pos[1:]) / 2
    return torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).view(h * w, 2)


def apply_rotary_emb(x, theta, conj=False):
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2, x3), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.norm = nn.LayerNorm((dim,))
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = zero_init(nn.Linear(dim, dim))
        self.out_drop = nn.Dropout(0.1)
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, self.n_heads * head_dim // 4 + 1)[:-1].exp()
        freqs = freqs.view(head_dim // 4, self.n_heads).T.contiguous()
        # theta = 10000.0
        # rope_dim = self.n_heads * head_dim // 2
        # freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2) / rope_dim))
        # freqs = freqs.view(head_dim // 4, self.n_heads).T
        # TODO: allow changing image size
        pos = make_axial_pos(28, 28)
        # h_pos = torch.arange(29, dtype=torch.float32)
        # w_pos = torch.arange(28, dtype=torch.float32)
        # pos = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).view(-1, 2)
        # make room for the class token and final backspace token
        pos = torch.cat((torch.zeros(1, 2), pos, pos[-28:] + (pos[-28:] - pos[-56:-28])))
        # pos = torch.cat((torch.full((1, 2), -1), pos))
        theta_h = pos[..., None, 0:1] * freqs
        theta_w = pos[..., None, 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("theta", theta)

    def forward(self, x, m, p, cache=None, index=None):
        x = self.norm(x)
        # q, k, v = rearrange(self.qkv_proj(x), "n s (t h d) -> t n h s d", t=3, h=self.n_heads)
        qkv = self.qkv_proj(x).reshape(x.shape[0], -1, 3, self.n_heads, self.head_dim)  # n s t h d
        # transpose to n h t s d
        qkv = qkv.transpose(1, 3)
        q, k, v = qkv.unbind(2)
        if index is None:
            q = apply_rotary_emb(q, self.theta[p].transpose(1, 2).to(q))
            k = apply_rotary_emb(k, self.theta[p].transpose(1, 2).to(k))
            x = F.scaled_dot_product_attention(q, k, v, m[:, None], 0.1 if self.training else 0)
            if cache is not None:
                cache[0][:] = k
                cache[1][:] = v
        else:
            b_idx = torch.arange(x.shape[0], device=x.device)
            max_index = torch.amax(index)
            m = m[b_idx, index, :max_index][:, None, None]
            q = apply_rotary_emb(q, self.theta[p[b_idx, index]][:, None].transpose(1, 2).to(q))
            k = apply_rotary_emb(k, self.theta[p[b_idx, index]][:, None].transpose(1, 2).to(k))
            cache[0][b_idx, :, index] = k[:, :, 0, :]
            cache[1][b_idx, :, index] = v[:, :, 0, :]
            x = F.scaled_dot_product_attention(q, cache[0][:, :, :max_index], cache[1][:, :, :max_index], m)
        # x = rearrange(x, "n h s d -> n s (h d)")
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.n_heads * self.head_dim)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.up = nn.Linear(dim, hidden_dim * 2)
        self.drop_1 = nn.Dropout(0.1)
        self.down = zero_init(nn.Linear(hidden_dim, dim))
        self.drop_2 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm(x)
        x, y = self.up(x).chunk(2, dim=-1)
        x = x * F.silu(y)
        x = self.drop_1(x)
        x = self.down(x)
        x = self.drop_2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, hidden_dim, head_dim):
        super().__init__()
        self.attn = SelfAttention(dim, head_dim)
        self.ff = FeedForward(dim, hidden_dim)

    def forward(self, x, m, p, cache=None, index=None):
        x = x + self.attn(x, m, p, cache, index)
        x = x + self.ff(x)
        return x


class Transformer(nn.Module):
    def __init__(self, depth, dim, hidden_dim, head_dim):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.class_embed = nn.Embedding(10, dim)
        self.image_embed = nn.Embedding(257, dim)
        self.embed_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = zero_init(nn.Linear(dim, 257))

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

    def forward(self, x, m, p, cache=None, index=None):
        y = self.class_embed(x[:, :1])
        x = self.image_embed(x[:, 1:])
        x = torch.cat((y, x), dim=1)
        b_idx = torch.arange(x.shape[0], device=x.device)
        cache = [None] * self.depth if cache is None else cache
        x = x if index is None else x[b_idx, index][:, None]
        x = self.embed_drop(x)
        for block, cache_block in zip(self.blocks, cache):
            x = block(x, m, p, cache_block, index)
        x = self.out_norm(x)
        return self.out_proj(x)


def noise_seq(seq, n_toks, bs_tok, n_noise):
    for _ in range(n_noise):
        noise_index = np.random.randint(1, len(seq) - 1)
        noise_token = np.random.randint(n_toks)
        bs_seq = np.stack((noise_token, np.array(bs_tok)))
        seq = np.insert(seq, noise_index, bs_seq)
    return sm_make_input(seq, bs_tok)


class SequencePreprocessor:
    def __init__(self, dataset, n_toks, bs_tok, n_noise):
        self.dataset = dataset
        self.n_toks = n_toks
        self.bs_tok = bs_tok
        self.n_noise = n_noise

    def __getitem__(self, index):
        x, y = self.dataset[index]
        a = np.concatenate((np.array((y,)), x.ravel()))
        return noise_seq(a, self.n_toks, self.bs_tok, self.n_noise)

    def __len__(self):
        return len(self.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="run")
    args = parser.parse_args()

    batch_size = 64

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_raw = Transformer(2, 128, 256, 32).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    # model_raw = torch.compile(model_raw)
    # model_ema = torch.compile(model_ema)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    transform = np.array
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    dataset2 = SequencePreprocessor(dataset, n_toks=256, bs_tok=256, n_noise=20)
    sampler = data.distributed.DistributedSampler(
        dataset2, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset2,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=30,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )

    wd, no_wd = [], []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_wd.append(param)
        else:
            wd.append(param)
    groups = [{"params": wd}, {"params": no_wd, "weight_decay": 0}]
    opt = optim.AdamW(groups, lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)

    epoch = 0

    @torch.no_grad()
    def sample(model, x, cache, index, disable=False):
        image_toks = 28 * 28
        b_idx = torch.arange(x.shape[0], device=device)
        m = torch.ones(
            x.shape[0], x.shape[1], x.shape[1], dtype=torch.bool, device=device
        ).tril_()
        p = torch.arange(x.shape[1], device=device).tile((x.shape[0], 1))
        i = 0
        with tqdm(total=image_toks, disable=disable) as pbar:
            while torch.amin(index) < image_toks:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(x, m, p, cache, index)[:, -1].float()
                gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
                sample = torch.argmax(logits + gumbel, dim=-1)
                good_toks = (index == 0) | (sample != 256)  # & ~(index == image_toks + 1)
                index = torch.where(good_toks, index + 1, index - 1).clamp_max_(image_toks + 1)
                x[b_idx, index] = torch.where(good_toks, sample, x[b_idx, index])
                pbar.update(1)
                i += 1
                if i > image_toks * 2:
                    break
        return x[:, :image_toks + 1], index

    @torch.no_grad()
    def sample_with_bs(model, a, cache, index, disable=False):
        image_toks = 28 * 28
        b_idx = torch.arange(a.shape[0], device=device)
        bs_tok = 256
        n = a.shape[0]
        l = a.shape[1]
        l_start = torch.amin(index) + 1
        x = a.clone()
        y = torch.zeros(n, l, dtype=torch.int64, device=device)
        t = torch.ones(n, l, dtype=torch.bool, device=device)
        m = torch.zeros(n, l, l, dtype=torch.bool, device=device)
        p = torch.zeros(n, l, dtype=torch.int64, device=device)
        c = torch.zeros(n, dtype=torch.int64, device=device)
        d = torch.zeros(n, dtype=torch.int64, device=device)

        def first_nonzero(x, mask, dim, invalid_val=-1):
            mask = (x != 0) & mask
            if mask.shape[dim] == 0:
                return invalid_val
            return torch.where(mask.any(dim=dim), mask.byte().argmax(dim=dim), invalid_val)

        def last_nonzero(x, mask, dim, invalid_val=-1):
            mask = (x != 0) & mask
            if mask.shape[dim] == 0:
                return invalid_val
            val = x.shape[dim] - torch.flip(mask, dims=(dim,)).byte().argmax(dim=dim) - 1
            return torch.where(mask.any(dim=dim), val, invalid_val)

        for i in trange(l, disable=disable):
            m[:, i] = m[:, max(i - 1, 0)]

            is_eos = p[:, i - 1] >= image_toks if i > 0 else torch.zeros(n, dtype=torch.bool, device=device)
            is_prompt = i <= index
            if i >= l_start:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(x, m, p - 1, cache, torch.full_like(index, i - 1))[:, -1].float()
                gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
                sample = torch.argmax(logits + gumbel, dim=-1)
                x[:, i] = torch.where(is_eos | is_prompt, x[:, i], sample)
            if i > 0:
                y[:, i - 1] = x[:, i]

            x1, t1, m1, p1 = x.clone(), t.clone(), m.clone(), p.clone()
            x2, t2, m2, p2 = x.clone(), t.clone(), m.clone(), p.clone()

            m1[b_idx, i, c] = 0
            m1[b_idx, i, d] = 0
            m1[:, i, i] = 1
            x1[:, i] = x[b_idx, c]
            t1[b_idx, c] = 0
            p1[:, i] = p[b_idx, c]
            d1 = torch.full_like(d, i)
            nzi = last_nonzero(m1[:, i], torch.arange(l, device=device)[None] < c[:, None], dim=1)
            c1 = torch.where(nzi == -1, 0, nzi)

            m2[:, i, i] = 1
            x2[:, i] = x[:, i]
            d2 = d + 1
            nzi = first_nonzero(m2[:, i], torch.arange(l, device=device)[None] >= (c + 1)[:, None], dim=1)
            c2 = torch.where(nzi == -1, 0, nzi)
            p2[:, i] = p[b_idx, d2 - 1] + 1

            is_bs = x[:, i] == bs_tok
            x[:, i] = torch.where(is_bs, x1[:, i], x2[:, i])
            t[:, c] = torch.where(is_bs, t1[:, c], t2[:, c])
            t[:, i] &= ~is_eos
            m[:, i] = torch.where(is_bs[:, None], m1[:, i], m2[:, i])
            p[:, i] = torch.where(is_bs, p1[:, i], p2[:, i]).clamp_max_(image_toks)
            c = torch.where(is_bs, c1, c2)
            d = torch.where(is_bs, d1, d2)

            if i == 0:
                c[:] = 0
                d[:] = 0
            if i == 1:
                c[:] = 0
                d[:] = 1

        t[:, -1] = 0
        return x, y, t, m, p

    def demo():
        image_toks = 28 * 28
        y = torch.arange(10, device=device).repeat_interleave(10)[:, None]
        x = torch.full((y.shape[0], image_toks + 2), 0, dtype=torch.int64, device=device)
        x[:, :1] = y
        cache = model_ema.init_cache(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=device)
        index = torch.zeros(x.shape[0], dtype=torch.int64, device=device)
        x, _ = sample(model_ema, x, cache, index)
        x = x[:, 1:]
        x = rearrange(x, "(nh nw) (h w) -> 1 (nh h) (nw w)", nh=10, nw=10, h=28, w=28)
        x = x.float() / 255
        x = torch.clamp(x, 0, 1)
        TF.to_pil_image(x.cpu()).save(f"demo_{args.name}_{epoch:04}.png")

    replay_buffer_list = []
    global_step = 0
    sample_every = 50
    steps_since_sample = 0

    while True:
        sampler.set_epoch(epoch)
        if rank == 0:
            demo()
        dist.barrier()

        for step, (x, y, t, m, p) in enumerate(tqdm(dataloader, disable=rank > 0)):
            x, y, t, m, p = x.to(device), y.to(device), t.to(device), m.to(device), p.to(device)
            if False:
                if global_step == 0 or steps_since_sample == sample_every:
                    # sample_every = min(50, sample_every + 2)
                    steps_since_sample = 0
                    # x_sample = a.new_zeros(a.shape[0], a.shape[1] + 2)
                    # x_sample[:, :-2] = a
                    # x_sample = x_sample.tile((4, 1))
                    # m_sample = torch.ones(
                    #     x_sample.shape[0], x_sample.shape[1], x_sample.shape[1], dtype=torch.bool, device=device
                    # ).tril_()
                    # p_sample = torch.arange(x_sample.shape[1], device=device)[None].clamp_max_(28 * 28)
                    # cache = model_ema.init_cache(x_sample.shape[0], x_sample.shape[1], dtype=torch.bfloat16, device=device)
                    # model_ema(x_sample, m_sample, p_sample, cache)
                    # index = torch.randint(0, 28 * 28 // 2, (x_sample.shape[0],), device=device)
                    # samples, end_index = sample(model_ema, x_sample, cache, index, disable=rank > 0)
                    # samples = samples[:, :28 * 28 + 1]
                    # samples = [noise_seq(s.cpu().numpy(), 256, 256, 0) for s in samples]
                    # samples = [torch.from_numpy(np.stack(lst)).to(device) for lst in zip(*samples)]
                    # replay_buffer_list.append((samples, index, end_index))

                    l = 1024
                    a = torch.zeros(x.shape[0], l, dtype=torch.int64, device=device)
                    a[:, :x.shape[1]] = x
                    a = a.tile((8, 1))
                    m_sample = torch.ones(
                        a.shape[0], a.shape[1], a.shape[1], dtype=torch.bool, device=device
                    ).tril_()
                    p_sample = torch.arange(a.shape[1], device=device)[None].clamp_max_(28 * 28)
                    cache = model_ema.init_cache(a.shape[0], l, dtype=torch.bfloat16, device=device)
                    index = torch.randint(28 * 28 // 2, (a.shape[0],), device=device)
                    model_ema(a, m_sample, p_sample, cache)
                    samples = sample_with_bs(model_ema, a, cache, index, disable=rank > 0)
                    replay_buffer_list.append(samples)

                    if len(replay_buffer_list) > 8:
                        replay_buffer_list.pop(0)
                    replay_x = torch.cat([i[0] for i in replay_buffer_list])
                    replay_y = torch.cat([i[1] for i in replay_buffer_list])
                    replay_t = torch.cat([i[2] for i in replay_buffer_list])
                    replay_m = torch.cat([i[3] for i in replay_buffer_list])
                    replay_p = torch.cat([i[4] for i in replay_buffer_list])
                    # replay_i = torch.cat([i[1] for i in replay_buffer_list])
                    # replay_e = torch.cat([i[2] for i in replay_buffer_list])
                model_n = 16
                indices = torch.randperm(replay_x.shape[0], device=device)[:model_n]
                model_x = replay_x[indices]
                model_y = replay_y[indices]
                model_t = replay_t[indices]
                model_m = replay_m[indices]
                model_p = replay_p[indices]
                # model_y = torch.zeros_like(model_x)
                # model_y[:, :-1] = model_x[:, 1:]
                # model_t = torch.ones_like(model_x, dtype=torch.bool)
                # model_m = torch.ones(
                #     model_x.shape[0], model_x.shape[1], model_x.shape[1], dtype=torch.bool, device=device
                # ).tril_()
                # model_p = torch.arange(model_x.shape[1], device=device)[None]
                # model_t = (model_p >= replay_i[indices, None]) & (model_p <= replay_e[indices, None])
                # model_p_loss = model_p + 1 - replay_i[indices, None]
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits_for_noise = model_ema(x, m, p - 1).float()
            logits_for_noise.scatter_(2, y[:, :, None], float("-inf"))
            logits_for_noise[:, :, -1] = float("-inf")
            gumbel = torch.rand_like(logits_for_noise).log_().nan_to_num_().neg_().log_().neg_()
            samples = torch.argmax(logits_for_noise + gumbel, dim=-1)
            x[:, 1:] = torch.where(t[:, :-1], x[:, 1:], samples[:, :-1])
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, m, p - 1).float()
                # model_logits = model(model_x, model_m, model_p - 1).float()
            # gamma = 784 / (784 + 1)
            # epoch_ = epoch + step / len(dataloader)
            # ramp = min(1, max(0, (epoch_ - 2) / 2))
            # chi2_mix_fac = 0.5 * ramp
            # loss = sm_loss(logits, y, t, p, model_logits, model_y, model_t, model_p, gamma=gamma, alpha=0.8, chi2_mix_fac=chi2_mix_fac)
            loss = torch.sum(F.cross_entropy(logits.mT, y, reduction="none") * t) / torch.sum(t)
            # loss = simple_sm_loss(logits, y, gamma, alpha=0.25, pos=p, mask=t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.95)
            dist.all_reduce(loss, dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")
            steps_since_sample += 1
            global_step += 1

        epoch += 1


if __name__ == "__main__":
    main()

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
    l = a.shape[0] - 1
    x = a[:-1]
    y = np.zeros((l,), np.int64)
    t = np.ones((l,), np.bool_)
    m = np.zeros((l, l), np.bool_)
    p = np.zeros((l,), np.int64)

    c = 0
    d = 0

    for i in range(0, l):
        m[i] = m[max(i - 1, 0)]
        y[i] = a[i + 1]
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

    return x, y, t, m, p


def sm_loss(s_d, y_d, t_d, p_d, s_m, y_m, t_m, p_m, n, alpha=0.01, divergence="chi2-mixture"):
    if divergence not in ("chi2", "chi2-mixture", "kl"):
        raise ValueError(f"Unknown divergence: {divergence}, must be chi2, chi2-mixture, or kl")

    def phi(x):
        if divergence in ("chi2", "chi2-mixture"):
            return x - x**2 / (4 * alpha)
        return -torch.expm1(-x)

    gamma = n / (n + 1)
    gamma_d = gamma**p_d * t_d
    gamma_m = gamma**p_m * t_m
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


def sm_loss_g(d_logits_d, s_d, s_m, n):
    r_d = torch.sum(d_logits_d, dim=2)
    gamma = n / (n + 1)
    gamma_d = gamma ** torch.arange(r_d.shape[1], device=r_d.device)[None]
    v_d = torch.logsumexp(s_d, dim=2)
    v_m = torch.logsumexp(s_m, dim=2)
    qv_diff_d = gamma_d[:, :-1] * r_d[:, :-1]
    v_diff_d = gamma_d[:, :-1] * (v_d[:, :-1] - gamma * v_d[:, 1:]) / 2
    v_diff_d_eos = gamma_d[:, -1] * v_d[:, -1] / 2
    v_diff_m = gamma_d[:, :-1] * (v_m[:, :-1] - gamma * v_m[:, 1:]) / 2
    v_diff_m_eos = gamma_d[:, -1] * v_m[:, -1] / 2
    loss_d = qv_diff_d.sum(1) - v_diff_d.sum(1) - v_diff_d_eos
    loss_m = -v_diff_m.sum(1) - v_diff_m_eos
    return -(loss_d.mean() + loss_m.mean()) / n


def sm_loss_d(d_logits_d, d_logits_m):
    r_d = torch.sum(d_logits_d, dim=2)
    r_m = torch.sum(d_logits_m, dim=2)
    return F.softplus(r_m).mean() + F.softplus(-r_d).mean()


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
        freqs = torch.linspace(log_min, log_max, head_dim // 4 + 1)[:-1].exp()
        # TODO: allow changing image size
        pos = make_axial_pos(28, 28)
        # make room for the class token
        pos = torch.cat((torch.zeros(1, 2), pos))
        theta_h = pos[..., 0:1] * freqs
        theta_w = pos[..., 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("theta", theta)

    def forward(self, x, m, p, cache=None, index=None):
        x = self.norm(x)
        # q, k, v = rearrange(self.qkv_proj(x), "n s (t h d) -> t n h s d", t=3, h=self.n_heads)
        qkv = self.qkv_proj(x).reshape(x.shape[0], -1, 3, self.n_heads, self.head_dim)  # n s t h d
        # transpose to n h t s d
        qkv = qkv.transpose(1, 3)
        q, k, v = qkv.unbind(2)
        if cache is None:
            q = apply_rotary_emb(q, self.theta[p][:, None].to(q))
            k = apply_rotary_emb(k, self.theta[p][:, None].to(k))
            x = F.scaled_dot_product_attention(q, k, v, m[:, None], 0.1 if self.training else 0)
        else:
            b_idx = torch.arange(x.shape[0], device=x.device)
            max_index = torch.amax(index)
            m = m[b_idx, index, :max_index][:, None, None]
            q = apply_rotary_emb(q, self.theta[index[:, None]][:, None].to(q))
            k = apply_rotary_emb(k, self.theta[index[:, None]][:, None].to(k))
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
        self.class_embed = nn.Linear(10, dim)
        self.image_embed = nn.Linear(257, dim)
        self.embed_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = nn.Linear(dim, 257)

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
        if x.ndim == 2:
            x = F.one_hot(x, 257).float()
        y = self.class_embed(x[:, :1, :10])
        x = self.image_embed(x[:, 1:])
        x = torch.cat((y, x), dim=1)
        b_idx = torch.arange(x.shape[0], device=x.device)
        x = x if cache is None else x[b_idx, index][:, None]
        cache = [None] * self.depth if cache is None else cache
        x = self.embed_drop(x)
        for block, cache_block in zip(self.blocks, cache):
            x = block(x, m, p, cache_block, index)
        x = self.out_norm(x)
        return self.out_proj(x)


class SequencePreprocessor:
    def __init__(self, dataset, n_toks, bs_tok, n_noise):
        self.dataset = dataset
        self.n_toks = n_toks
        self.bs_tok = bs_tok
        self.n_noise = n_noise

    def __getitem__(self, index):
        x, y = self.dataset[index]
        a = np.concatenate((np.array((y,)), x.ravel()))
        for _ in range(self.n_noise):
            noise_index = np.random.randint(1, len(a) - 2)
            noise_token = np.random.randint(self.n_toks)
            bs_seq = np.stack((noise_token, np.array(self.bs_tok)))
            a = np.insert(a, noise_index, bs_seq)
        return sm_make_input(a, self.bs_tok)

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

    model_raw = Transformer(2, 128, 256, 64).to(device)
    d_raw = Transformer(2, 128, 256, 64).to(device)
    du.broadcast_tensors(model_raw.parameters())
    du.broadcast_tensors(d_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    # model_raw = torch.compile(model_raw)
    # model_ema = torch.compile(model_ema)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )
    d = nn.parallel.DistributedDataParallel(
        d_raw, device_ids=[device], output_device=device
    )

    transform = np.array
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    dataset2 = SequencePreprocessor(dataset, n_toks=256, bs_tok=256, n_noise=0)
    sampler = data.distributed.DistributedSampler(
        dataset2, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset2,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=16,
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
    opt = optim.AdamW(groups, lr=3e-4, betas=(0.0, 0.95), weight_decay=0.1)
    for name, param in d.named_parameters():
        if "bias" in name or "norm" in name:
            no_wd.append(param)
        else:
            wd.append(param)
    groups = [{"params": wd}, {"params": no_wd, "weight_decay": 0}]
    opt_d = optim.AdamW(groups, lr=3e-4, betas=(0.0, 0.95), weight_decay=0.1)

    epoch = 0

    @torch.no_grad()
    def sample(model, y, disable=False):
        image_toks = 28 * 28
        cache = model_ema.init_cache(y.shape[0], image_toks + 1, dtype=torch.bfloat16, device=device)
        x = torch.full((y.shape[0], image_toks + 2), 127, dtype=torch.int64, device=device)
        x[:, :1] = y
        index = torch.zeros(x.shape[0], dtype=torch.int64, device=device)
        b_idx = torch.arange(x.shape[0], device=device)
        m = torch.ones(
            x.shape[0], x.shape[1] - 1, x.shape[1] - 1, dtype=torch.bool, device=device
        ).tril_()
        p = torch.arange(x.shape[1] - 1, device=device)[None]
        i = 0
        with tqdm(total=1000, disable=disable) as pbar:
            while torch.amin(index) < image_toks:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(x, m, p, cache, index)[:, -1].float()
                gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
                sample = torch.argmax(logits + gumbel, dim=-1)
                good_toks = (index == 0) | (sample != 256) & ~(index == image_toks)
                index = torch.where(good_toks, index + 1, index - 1).clamp_max_(image_toks)
                x[b_idx, index] = torch.where(good_toks, sample, x[b_idx, index])
                pbar.update(1)
                i += 1
                if i > 1000:
                    break
        return x[:, :image_toks + 1]

    def demo():
        y = torch.arange(10, device=device).repeat_interleave(10)[:, None]
        x = sample(model_ema, y)[:, 1:]
        x = rearrange(x, "(nh nw) (h w) -> 1 (nh h) (nw w)", nh=10, nw=10, h=28, w=28)
        x = x.float() / 255
        x = torch.clamp(x, 0, 1)
        TF.to_pil_image(x.cpu()).save(f"demo_{args.name}_{epoch:04}.png")

    replay_buffer_list = []

    while True:
        sampler.set_epoch(epoch)
        if rank == 0:
            demo()
        dist.barrier()

        for step, (x, y, t, m, p) in enumerate(tqdm(dataloader, disable=rank > 0)):
            x, y, t, m, p = x.to(device), y.to(device), t.to(device), m.to(device), p.to(device)
            if True:
                if step % 50 == 0:
                    replay_buffer_list.append(sample(model_ema, x[:, :1], disable=True))
                    if len(replay_buffer_list) > 20:
                        replay_buffer_list.pop(0)
                    model_x_all = torch.cat(replay_buffer_list)
                model_n = 16
                indices = torch.randperm(model_x_all.shape[0], device=device)[:model_n]
                model_x = model_x_all[indices, :-1]
                model_y = model_x_all[indices, 1:]
                model_m = torch.ones(
                    model_n, model_x.shape[1], model_x.shape[1], dtype=torch.bool, device=device
                ).tril_()
                model_t = torch.ones(model_n, model_x.shape[1], dtype=torch.bool, device=device)
                model_p = torch.arange(model_x.shape[1], device=device).tile(model_n, 1) + 1
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, m, p - 1).float()
                model_logits = model(model_x, model_m, model_p - 1).float()
            # y.masked_fill_(~t, -100)
            # loss = F.cross_entropy(logits.mT, y)
            # loss = sm_loss(logits, y, t, p, model_logits, model_y, model_t, model_p, 28 * 28)
            opt.zero_grad()
            d_real = d(logits, m, p - 1).float()
            # d_fake = d(model_logits, model_m, model_p - 1).float()
            g_loss = sm_loss_g(d_real, logits, model_logits, 28 * 28)
            g_loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.95)
            opt_d.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, m, p - 1).float()
                model_logits = model(model_x, model_m, model_p - 1).float()
            d_real = d(logits.detach(), m, p - 1).float()
            d_fake = d(model_logits.detach(), model_m, model_p - 1).float()
            d_loss = sm_loss_d(d_real, d_fake)
            d_loss.backward()
            opt_d.step()
            dist.all_reduce(g_loss, dist.ReduceOp.AVG)
            dist.all_reduce(d_loss, dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, g_loss: {g_loss.item():g}, d_loss: {d_loss.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

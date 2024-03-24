#!/usr/bin/env python3

from copy import deepcopy
from functools import lru_cache
import math

from einops import rearrange
import flash_attn
from flash_attn.layers import rotary
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


def zgr(logits, sample=None):
    """Samples from a categorical distribution using the ZGR gradient estimator."""
    if sample is None:
        g = torch.rand_like(logits, dtype=torch.float32).log_().neg_().log_().neg_()
        sample = torch.argmax(logits + g, dim=-1)
    one_hot = F.one_hot(sample, logits.shape[-1]).to(logits)
    logits = F.log_softmax(logits, dim=-1)
    st_est = logits.exp()
    darn_est = (one_hot - st_est.detach()) * logits.gather(-1, sample[..., None])
    zgr_est = (st_est + darn_est) / 2
    return one_hot + (zgr_est - zgr_est.detach())


# TODO: do this correctly instead
@lru_cache
def make_axial_pos(h, w, dtype=None, device=None):
    h_pos = torch.linspace(-1, 1, h + 1, dtype=dtype, device=device)
    w_pos = torch.linspace(-1, 1, w + 1, dtype=dtype, device=device)
    h_pos = (h_pos[:-1] + h_pos[1:]) / 2
    w_pos = (w_pos[:-1] + w_pos[1:]) / 2
    return torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).view(h * w, 2)


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
        self.register_buffer("cos", torch.cos(theta))
        self.register_buffer("sin", torch.sin(theta))

    def forward(self, x, cache=None, index=None, sm_mode=False):
        x = self.norm(x)
        qkv = self.qkv_proj(x).view(*x.shape[:2], 3, self.n_heads, self.head_dim)
        if sm_mode:
            qkv = rotary.apply_rotary_emb_qkv_(qkv, self.cos.to(qkv), self.sin.to(qkv))
            q, k, v = qkv.unbind(2)
            k_tmp = torch.cat((cache[0], k), dim=1)
            v_tmp = torch.cat((cache[1], v), dim=1)
            n_cache = cache[0].shape[1]
            n_main = k.shape[1]
            cache_part = torch.ones((n_cache, n_cache), dtype=torch.bool, device=x.device).tril(-1)
            main_part = torch.eye(n_main, n_main, dtype=torch.bool, device=x.device)
            mask = torch.cat((cache_part, main_part), dim=-1)
            x = F.scaled_dot_product_attention(q.transpose(1, 2), k_tmp.transpose(1, 2), v_tmp.transpose(1, 2), mask, 0.0 if self.training else 0).transpose(1, 2)
        elif index is None:
            qkv = rotary.apply_rotary_emb_qkv_(qkv, self.cos.to(qkv), self.sin.to(qkv))
            x = flash_attn.flash_attn_qkvpacked_func(qkv, 0.0 if self.training else 0, causal=True)
            if cache is not None:
                cache[0][:, :x.shape[1]] = qkv[:, :, 1]
                cache[1][:, :x.shape[1]] = qkv[:, :, 2]
        else:
            assert not (x.shape[1] > 1 and index != 0)
            end_index = index + x.shape[1]
            q, k, v = qkv.unbind(2)
            q = rotary.apply_rotary_emb(
                q, self.cos[index:end_index].to(q), self.sin[index:end_index].to(q), inplace=True
            )
            k = rotary.apply_rotary_emb(
                k, self.cos[index:end_index].to(q), self.sin[index:end_index].to(q), inplace=True
            )
            cache[0][:, index:end_index] = k
            cache[1][:, index:end_index] = v
            x = flash_attn.flash_attn_func(
                q, cache[0][:, :end_index], cache[1][:, :end_index], causal=index == 0
            )
        x = x.reshape(*x.shape[:2], -1)
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

    def forward(self, x, cache=None, index=None, sm_mode=False):
        x = x + self.attn(x, cache, index, sm_mode)
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
        self.embed_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        # self.image_embed.weight = self.out_proj.weight

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

    def forward(self, x, y, cache=None, index=None, sm_mode=False, x_is_soft=False):
        if x_is_soft:
            x = x @ self.image_embed.weight
        else:
            x = self.image_embed(x)
        y = self.class_embed(y)
        x = torch.cat((y[:, None], x), dim=1)
        x = x[:, -1:] if index is not None else x
        x = self.embed_drop(x)
        cache = [None] * self.depth if cache is None else cache
        for block, cache_block in zip(self.blocks, cache):
            # x = torch.utils.checkpoint.checkpoint(block, x, cache_block, index)
            x = block(x, cache_block, index, sm_mode)
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


def sm_loss_test(logits_d, logits_m, actions_d, actions_m, seq_len, kl_fac=0.0, kl_alpha=1.0, chi2_mix_fac=0.0):
    assert logits_d.shape == logits_m.shape

    def phi(x):
        chi2 = x - x**2 / 4
        kl = -torch.expm1(-x * kl_alpha) / kl_alpha
        return kl_fac * kl + (1 - kl_fac) * chi2

    def psi(x):
        return 0
        # return -chi2_mix_fac * (1 - kl_fac) * x**2 / 4

    i = torch.arange(1, logits_d.shape[-2] + 1, device=logits_d.device)
    gamma = seq_len / (seq_len + 1)
    gamma_i = gamma**i
    q_d = logits_d.gather(-1, actions_d[..., None])[..., 0]
    # q_m = logits_m.gather(-1, actions_m[..., None])[..., 0]
    v_d = torch.logsumexp(logits_d, dim=-1)
    v_m = torch.logsumexp(logits_m, dim=-1)

    qv_diff_d = gamma_i[..., :-1] * phi(q_d[..., :-1] - gamma * v_d[..., 1:])
    qv_diff_d_eos = gamma_i[..., -1] * phi((1 - gamma) * v_d[..., -1]) / (1 - gamma)
    # qv_diff_m = gamma_i[..., :-1] * psi(q_m[..., :-1] - gamma * v_m[..., 1:])
    # qv_diff_m_eos = gamma_i[..., -1] * psi((1 - gamma) * v_m[..., -1]) / (1 - gamma)
    v_diff_d = gamma_i[..., :-1] * (v_d[..., :-1] - gamma * v_d[..., 1:]) / 2
    v_diff_d_eos = gamma_i[..., -1] * v_d[..., -1] / 2
    v_diff_m = gamma_i[..., :-1] * (v_m[..., :-1] - gamma * v_m[..., 1:]) / 2
    v_diff_m_eos = gamma_i[..., -1] * v_m[..., -1] / 2
    loss = v_diff_d.sum(-1) + v_diff_d_eos + v_diff_m.sum(-1) + v_diff_m_eos - qv_diff_d.sum(-1) - qv_diff_d_eos  # - qv_diff_m.sum(-1) - qv_diff_m_eos

    return torch.mean(loss) / seq_len


def sm_loss_test(logits_d, logits_m, actions_d, actions_m, seq_len, alpha, mix_fac):
    assert logits_d.shape == logits_m.shape

    def phi(x):
        return x - alpha * x**2 / 4

    def psi(x):
        return -alpha * mix_fac * x**2 / 4

    i = torch.arange(1, logits_d.shape[-2] + 1, device=logits_d.device)
    gamma = seq_len / (seq_len + 1)
    gamma_i = gamma**i
    q_d = logits_d.gather(-1, actions_d[..., None])[..., 0]
    q_m = logits_m.gather(-1, actions_m[..., None])[..., 0]
    v_d = torch.logsumexp(logits_d, dim=-1)
    v_m = torch.logsumexp(logits_m, dim=-1)

    qv_diff_d = gamma_i[..., :-1] * phi(q_d[..., :-1] - gamma * v_d[..., 1:])
    qv_diff_d_eos = gamma_i[..., -1] * phi((1 - gamma) * v_d[..., -1]) / (1 - gamma)
    qv_diff_m = gamma_i[..., :-1] * psi(q_m[..., :-1] - gamma * v_m[..., 1:])
    qv_diff_m_eos = gamma_i[..., -1] * psi((1 - gamma) * v_m[..., -1]) / (1 - gamma)
    v_diff_d = gamma_i[..., :-1] * (v_d[..., :-1] - gamma * v_d[..., 1:]) / 2
    v_diff_d_eos = gamma_i[..., -1] * v_d[..., -1] / 2
    v_diff_m = gamma_i[..., :-1] * (v_m[..., :-1] - gamma * v_m[..., 1:]) / 2
    v_diff_m_eos = gamma_i[..., -1] * v_m[..., -1] / 2
    loss = v_diff_d.sum(-1) + v_diff_d_eos + v_diff_m.sum(-1) + v_diff_m_eos - qv_diff_d.sum(-1) - qv_diff_d_eos - qv_diff_m.sum(-1) - qv_diff_m_eos

    return torch.mean(loss) / seq_len


def thing3(logits, targets, phi):
    q = logits.gather(-1, targets[..., None])[..., 0]
    v = torch.logsumexp(logits, dim=-1)
    return -phi(q - v)


def thing4(logits, targets, gamma, phi):
    i = torch.arange(1, logits.shape[-2] + 1, device=logits.device)
    gamma_i = gamma**i
    q = logits.gather(-1, targets[..., None])[..., 0]
    v = torch.logsumexp(logits, dim=-1)
    qv_diff = gamma_i[..., :-1] * phi(q[..., :-1] - gamma * v[..., 1:])
    qv_eos = gamma_i[..., -1] * phi((1 - gamma) * v[..., -1]) / (1 - gamma)
    v_diff = gamma_i[..., :-1] * (v[..., :-1] - gamma * v[..., 1:])
    v_eos = gamma_i[..., -1] * v[..., -1]
    return v_diff.sum(-1) + v_eos - qv_diff.sum(-1) - qv_eos


def thing5(logits_d, logits_m, targets_d, targets_m, gamma):
    def phi(x):
        return x

    assert logits_d.shape == logits_m.shape
    assert targets_d.shape == targets_m.shape

    i = torch.arange(1, logits_d.shape[-2] + 1, device=logits_d.device)
    gamma_i = gamma**i
    q_d = logits_d.gather(-1, targets_d[..., None])[..., 0]
    q_m = logits_d.gather(-1, targets_m[..., None])[..., 0]
    v_d = torch.logsumexp(logits_d, dim=-1)
    v_m = torch.logsumexp(logits_m, dim=-1)
    qv_diff_d = gamma_i[..., :-1] * phi(q_d[..., :-1] - gamma * v_d[..., 1:]) / 2
    qv_eos_d = gamma_i[..., -1] * phi((1 - gamma) * v_d[..., -1]) / (1 - gamma) / 2
    qv_diff_m = gamma_i[..., :-1] * phi(q_d[..., :-1] - gamma * v_m[..., 1:]) / 2
    qv_eos_m = gamma_i[..., -1] * phi((1 - gamma) * v_m[..., -1]) / (1 - gamma) / 2
    v_diff_d = gamma_i[..., :-1] * (v_d[..., :-1] - gamma * v_d[..., 1:]) / 2
    v_eos_d = gamma_i[..., -1] * v_d[..., -1] / 2
    v_diff_m = gamma_i[..., :-1] * (v_d[..., :-1] - gamma * v_m[..., 1:]) / 2
    v_eos_m = gamma_i[..., -1] * v_m[..., -1] / 2
    return v_diff_d.sum(-1) + v_eos_d + v_diff_m.sum(-1) + v_eos_m - qv_diff_d.sum(-1) - qv_eos_d - qv_diff_m.sum(-1) - qv_eos_m


def sample_categorical(logits, tau=1.0):
    gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
    return torch.argmax(logits + gumbel * tau, dim=-1)


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

    model_raw = Transformer(8, 512, 1360, 64, vocab_size).to(device)
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
    def sample(model, n, y, temperature=1.0, top_p=1.0, disable=False):
        n_proc = n // world_size
        x = torch.zeros(n_proc, 0, dtype=torch.long, device=device)
        y = y.split(n_proc)[rank]
        cache = model.init_cache(n_proc, 1024, dtype=torch.bfloat16, device=device)
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
        return torch.cat(dnn.all_gather(x))

    def demo(step):
        y = torch.arange(10, device=device).repeat_interleave(10)
        x = sample(model_ema, 100, y)
        if rank == 0:
            x = dequantize(x, palette)
            x = rearrange(x, "(nh nw) (h w) d -> d (nh h) (nw w)", nh=10, nw=10, h=32, w=32)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_cifar_q_sm_078_{epoch:04}_{step:05}.png")

    while True:
        sampler.set_epoch(epoch)
        for step, (x_d, y_d) in enumerate(tqdm(dataloader, disable=rank > 0)):
            if step % 100 == 0:
                demo(step)
            x_d = x_d.to(device).flatten(1, 2).float() / 255
            # x_d = quantize_with_noise(x_d, n_levels)
            x_d = quantize(x_d, palette)
            y_d = y_d.to(device)
            cache = model_raw.init_cache(batch_size, x_d.shape[-1], dtype=torch.bfloat16, device=device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x_d[..., :-1], y_d, cache).float()
            x_m = sample_categorical(logits)
            # x_m = zgr(logits)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits_m = model(x_m[:, :-1], y_d, cache, sm_mode=True).float()
            # loss = sm_loss_test(logits, logits_m, x_d, x_m, x_d.shape[-1], kl_fac=0.0, kl_alpha=0.01)
            # loss = sm_loss_data_only(logits, x_d, x_d.shape[-1])
            # loss = sm_loss_test(logits * 0.9, logits_m * 0.9, x_d, x_m, x_d.shape[-1], 1.0, 1.0)
            loss_xent = F.cross_entropy(logits.mT, x_d, reduction="mean")
            # loss_xent_2 = F.cross_entropy(logits_m.mT, x_m, reduction="mean")
            # loss = (loss_xent + loss_xent_2) / 2
            # loss = 2 * loss_xent - loss_xent_2
            # loss = thing(logits, x_d).mean()
            # x_d_for_loss = F.pad(x_d, (0, 1))
            # loss = thing3(logits, x_d, lambda x: x - x**2 / 4).mean()
            # loss = sm_loss_test(logits, logits_m, x_d, x_m, x_d.shape[-1], 1.0, 1.0)
            # loss = thing5(logits, logits_m, x_d, x_m, 1024 / (1024 + 1)).mean() / 1024
            q_d = logits.gather(-1, x_d[..., None])[..., 0]
            q_m = logits.gather(-1, x_m[..., None])[..., 0]
            v_d = torch.logsumexp(logits, dim=-1)
            v_m = torch.logsumexp(logits_m, dim=-1)
            # loss = (v_d[..., :-1] - v_m[..., 1:]) / 2 + (v_d[..., :-1] - v_d[..., 1:]) / 2 - (q_d[..., :-1] - v_d[..., 1:]) / 2 - (q_d[..., :-1] - v_m[..., 1:]) / 2
            alpha = 0.1
            phi = lambda x: x - alpha * x**2 / 4
            psi = lambda x: -alpha * x**2 / 4
            qv_d = -phi(q_d[..., :-1] - v_d[..., 1:])
            qv_m = -psi(q_m[..., :-1] - v_m[..., 1:])
            v_thing_d = v_d[..., :-1] - v_d[..., 1:]
            v_thing_m = v_d[..., :-1] - v_m[..., 1:]
            losses = qv_d.sum(-1) + qv_m.sum(-1) + v_thing_d.sum(-1) / 2 + v_thing_m.sum(-1) / 2
            loss = torch.mean(losses) / x_d.shape[-1]

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.95)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}, xent: {loss_xent.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

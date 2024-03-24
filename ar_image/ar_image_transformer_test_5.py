#!/usr/bin/env python3

import argparse
from copy import deepcopy
from functools import lru_cache
import math

from einops import rearrange
import flash_attn
from flash_attn.layers import rotary
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
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = zero_init(nn.Linear(dim, dim))
        self.out_drop = nn.Dropout(0.1)
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, head_dim // 4 + 1)[:-1].exp()
        # TODO: allow changing image size
        pos = make_axial_pos(28, 28)
        # make room for the class token
        # TODO: use adanorm for this
        pos = torch.cat((torch.zeros(1, 2), pos))
        theta_h = pos[..., 0:1] * freqs
        theta_w = pos[..., 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("cos", torch.cos(theta))
        self.register_buffer("sin", torch.sin(theta))

    def forward(self, x, cache=None, index=None):
        x = self.norm(x)
        q, k, v = self.qkv_proj(x).view(*x.shape[:2], 3, self.n_heads, self.head_dim).unbind(2)
        if cache is None:
            q = rotary.apply_rotary_emb(q, self.cos.to(q), self.sin.to(q), inplace=True)
            k = rotary.apply_rotary_emb(k, self.cos.to(k), self.sin.to(k), inplace=True)
            x = flash_attn.flash_attn_func(q, k, v, 0.1 if self.training else 0, causal=True)
        else:
            assert not (x.shape[1] > 1 and index != 0)
            end_index = index + x.shape[1]
            q = rotary.apply_rotary_emb(
                q, self.cos[index:end_index].to(q), self.sin[index:end_index].to(q), inplace=True
            )
            k = rotary.apply_rotary_emb(
                k, self.cos[index:end_index].to(q), self.sin[index:end_index].to(q), inplace=True
            )
            cache[0][:, index:end_index] = k
            cache[1][:, index:end_index] = v
            x = flash_attn.flash_attn_func(
                q,
                cache[0][:, :end_index],
                cache[1][:, :end_index],
                0.1 if self.training else 0,
                causal=index == 0,
            )
        x = x.view(*x.shape[:2], -1)
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

    def forward(self, x, cache=None, index=None):
        x = x + self.attn(x, cache, index)
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
        self.image_embed = nn.Embedding(256, dim)
        self.embed_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = zero_init(nn.Linear(dim, 256))

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
        y = self.class_embed(y)
        x = torch.cat((y[:, None], x), dim=1)
        x = x if cache is None or index == 0 else x[:, -1:]
        cache = [None] * self.depth if cache is None else cache
        x = self.embed_drop(x)
        for block, cache_block in zip(self.blocks, cache):
            x = block(x, cache_block, index)
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x


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
    model_raw = Transformer(6, 256, 768, 64).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    dataset = datasets.MNIST("data", train=True, download=True, transform=np.array)
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

    global_step = 0
    epoch = 0

    @torch.no_grad()
    def sample(model, y, disable=False):
        n = y.shape[0]
        x = torch.zeros(n, 0, dtype=torch.long, device=device)
        cache = model.init_cache(n, 28 * 28, dtype=torch.bfloat16, device=device)
        index = 0
        for _ in trange(28 * 28, disable=disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x[:, -1:], y, cache, index).float()
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.argmax(logits + gumbel, dim=-1)
            x = torch.cat((x, sample), dim=1)
            index += 1
        return x

    @torch.no_grad()
    def sample_for_training(model, x_d, y_d, start_indices, disable=False):
        n = y_d.shape[0]
        x = torch.zeros(n, 0, dtype=torch.long, device=device)
        cache = model.init_cache(n, 28 * 28, dtype=torch.bfloat16, device=device)
        index = 0
        for _ in trange(28 * 28, disable=disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x[:, -1:], y_d, cache, index)[..., -1].float()
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.argmax(logits + gumbel, dim=-1)
            sample = torch.where(start_indices < index, x_d[:, index], sample)
            x = torch.cat((x, sample[..., None]), dim=1)
            index += 1
        return x

    def demo(step):
        y = torch.arange(10, device=device).repeat_interleave(10)
        x = sample(model_ema, y)
        x = rearrange(x, "(nh nw) (h w) -> 1 (nh h) (nw w)", nh=10, nw=10, h=28, w=28)
        x = x.float() / 255
        x = torch.clamp(x, 0, 1)
        TF.to_pil_image(x.cpu()).save(f"demo_{args.name}_{step:07}.png")

    x_replay, y_replay, mask_replay = [], [], []

    while True:
        sampler.set_epoch(epoch)
 
        for step, (x_d, y_d) in enumerate(tqdm(dataloader, disable=rank > 0)):
            x_d = x_d.to(device).flatten(1, 2).long()
            y_d = y_d.to(device)

            if global_step % 100 == 0:
                if rank == 0:
                    demo(global_step)
                dist.barrier()

            if global_step % 12 == 0:
                start_indices = torch.randint(0, 28 * 28, (batch_size,), device=device)
                x_sample = sample_for_training(model_raw, x_d, y_d, start_indices, disable=rank > 0)
                x_replay.append(x_sample)
                y_replay.append(y_d)
                mask_replay.append(torch.arange(28 * 28, device=device)[None, :] >= start_indices[:, None])
                if len(x_replay) > 10:
                    x_replay.pop(0)
                    y_replay.pop(0)
                    mask_replay.pop(0)

            x_replay_all = torch.cat(x_replay)
            y_replay_all = torch.cat(y_replay)
            mask_replay_all = torch.cat(mask_replay)
            indices = torch.randperm(x_replay_all.shape[0], device=device)[:batch_size // 4]
            x_m = x_replay_all[indices]
            y_m = y_replay_all[indices]
            mask_m = mask_replay_all[indices]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits_d = model(x_d[:, :-1], y_d).float()
                logits_m = model(x_m[:, :-1], y_m).float()

            xent = F.cross_entropy(logits_d.mT, x_d)
            q_d = logits_d.gather(-1, x_d[..., None])[..., 0]
            q_m = logits_m.gather(-1, x_m[..., None])[..., 0]
            v_d = torch.logsumexp(logits_d, dim=-1)
            v_m = torch.logsumexp(logits_m, dim=-1)
            phi = lambda x: x - 0.1 * x**2 / 4
            psi = lambda x: -0.1 * x**2 / 4
            seq_len = 784
            gamma = seq_len / (seq_len + 1)
            i_d = torch.arange(1, seq_len + 1, device=device).tile(x_d.shape[0], 1)
            i_m = torch.arange(1, seq_len + 1, device=device).tile(x_m.shape[0], 1)
            gamma_d = gamma**i_d
            gamma_m = mask_m * gamma**i_m
            qv_diff_d = gamma_d[..., :-1] * phi(q_d[..., :-1] - gamma * v_d[..., 1:])
            qv_eos_d = gamma_d[..., -1] * phi((1 - gamma) * v_d[..., -1]) / (1 - gamma)
            qv_diff_m = gamma_m[..., :-1] * psi(q_m[..., :-1] - gamma * v_m[..., 1:])
            qv_eos_m = gamma_m[..., -1] * psi((1 - gamma) * v_m[..., -1]) / (1 - gamma)
            v_diff_d = gamma_d[..., :-1] * (v_d[..., :-1] - gamma * v_d[..., 1:]) / 2
            v_eos_d = gamma_d[..., -1] * v_d[..., -1] / 2
            v_diff_m = gamma_m[..., :-1] * (v_m[..., :-1] - gamma * v_m[..., 1:]) / 2
            v_eos_m = gamma_m[..., -1] * v_m[..., -1] / 2
            loss_d = torch.mean(v_diff_d.sum(-1) + v_eos_d - qv_diff_d.sum(-1) - qv_eos_d) / seq_len
            loss_m = torch.mean(v_diff_m.sum(-1) + v_eos_m - qv_diff_m.sum(-1) - qv_eos_m) / seq_len
            loss = loss_d + loss_m
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.95)
            dist.all_reduce(loss, dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}, xent: {xent.item():g}")
            global_step += 1

        epoch += 1


if __name__ == "__main__":
    main()

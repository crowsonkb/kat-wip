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
from torchvision import datasets
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
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = zero_init(nn.Linear(dim, dim, bias=False))
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, head_dim // 8).exp()
        # TODO: allow changing image size
        pos = make_axial_pos(32, 32 * 3)
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
        qkv = self.qkv_proj(x).view(*x.shape[:2], 3, self.n_heads, self.head_dim)
        if cache is None:
            qkv = rotary.apply_rotary_emb_qkv_(qkv, self.cos.to(qkv), self.sin.to(qkv))
            x = flash_attn.flash_attn_qkvpacked_func(qkv, causal=True)
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

    def forward(self, x, cache=None, index=None):
        x = x + self.attn(x, cache, index)
        x = x + self.ff(x)
        return x


class DMoELinear(nn.Module):
    def __init__(self, in_features, out_features, n_experts, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_experts = n_experts
        self.weight = nn.Parameter(torch.empty(n_experts, out_features, in_features))
        for i in range(n_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_experts, out_features))
        else:
            self.bias = None

    def forward(self, x, ids):
        from megablocks import grouped_gemm_util as gg
        out_shape = (*x.shape[:-1], self.out_features)
        x = x.flatten(0, -2)
        ids = ids.flatten()
        ids_sorted, ids_indices = torch.sort(ids)
        x_sorted = x[ids_indices]
        batch_sizes = torch.bincount(ids_sorted, minlength=self.n_experts).cpu()
        out = gg.ops.gmm(x_sorted.bfloat16(), self.weight.bfloat16(), batch_sizes, trans_b=True)
        if self.bias is not None:
            out += self.bias[ids_sorted].bfloat16()
        out_real = torch.empty_like(out)
        out_real.scatter_(0, ids_indices[..., None].expand_as(out), out)
        return out_real.view(out_shape)


class GroupLinear(nn.Module):
    def __init__(self, in_features, out_features, n_groups):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_groups, out_features, in_features))
        for i in range(n_groups):
            nn.init.xavier_uniform_(self.weight[i])

    def forward(self, x, group):
        x = torch.einsum("nsi,goi->nsgo", x, self.weight)
        group = group[:, :, None, None].expand(x.shape[0], x.shape[1], 1, x.shape[3])
        return x.gather(-2, group)[..., 0, :]


class Transformer(nn.Module):
    def __init__(self, depth, dim, hidden_dim, head_dim):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.class_embed = nn.Embedding(10, dim)
        self.image_embed = DMoELinear(256, dim, 3, bias=False)
        self.embed_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = DMoELinear(dim, 256, 3, bias=False)

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
        ch = torch.arange(3, device=x.device).tile((x.shape[0], 1024))
        ch_in = ch[:, : x.shape[1]]
        ch_out = ch[:, : x.shape[1] + 1]
        x = self.image_embed(F.one_hot(x, 256).float(), ch_in)
        index = 0 if index is None else index
        y = self.class_embed(y)
        x = torch.cat((y[:, None], x), dim=1)
        x = x[:, -1:] if cache else x
        ch_out = ch_out[:, -1:] if cache else ch_out
        cache = [None] * self.depth if cache is None else cache
        x = self.embed_drop(x)
        for block, cache_block in zip(self.blocks, cache):
            # x = torch.utils.checkpoint.checkpoint(block, x, cache_block, index)
            x = block(x, cache_block, index)
        x = self.out_norm(x)
        x = self.out_proj(x, ch_out)
        return x


def main():
    batch_size = 32

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_raw = Transformer(8, 512, 1360, 64).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    dataset = datasets.CIFAR10("data", train=True, download=True, transform=np.array)
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
    def sample(model, n, y, disable=False):
        n_proc = n // world_size
        x = torch.zeros(n_proc, 0, dtype=torch.long, device=device)
        y = y.split(n_proc)[rank]
        cache = model.init_cache(n_proc, 3072, dtype=torch.bfloat16, device=device)
        index = 0
        for _ in trange(3072, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, y, cache, index).float()
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.argmax(logits + gumbel, dim=-1)
            x = torch.cat((x, sample), dim=1)
            index += 1
        return torch.cat(dnn.all_gather(x))

    def demo(step):
        y = torch.arange(10, device=device).repeat_interleave(10)
        x = sample(model_ema, 100, y)
        if rank == 0:
            x = rearrange(x, "(nh nw) (h w d) -> d (nh h) (nw w)", nh=10, nw=10, h=32, w=32, d=3)
            x = x.float() / 255
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_cifar_019_{epoch:04}_{step:05}.png")

    while True:
        sampler.set_epoch(epoch)

        for step, (x, y) in enumerate(tqdm(dataloader, disable=rank > 0)):
            if step % 100 == 0:
                demo(step)
            x = x.to(device).flatten(1).long()
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

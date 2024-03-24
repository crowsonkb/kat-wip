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


@torch.compile
def mixture_of_softmax(x, w):
    probs = torch.exp(x - torch.logsumexp(x, dim=-1, keepdim=True))
    weights = torch.exp(w - torch.logsumexp(w, dim=-1, keepdim=True))
    return torch.log(torch.sum(probs * weights[..., None], dim=-2))


# TODO: do this correctly instead
@lru_cache
def make_axial_pos(h, w, dtype=None, device=None):
    h_pos = torch.linspace(-1, 1, h + 1, dtype=dtype, device=device)
    w_pos = torch.linspace(-1, 1, w + 1, dtype=dtype, device=device)
    h_pos = (h_pos[:-1] + h_pos[1:]) / 2
    w_pos = (w_pos[:-1] + w_pos[1:]) / 2
    return torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).view(h * w, 2)


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
        pos = make_axial_pos(16, 16)
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
            x = flash_attn.flash_attn_qkvpacked_func(qkv, self.drop.p if self.training else 0.0, causal=True)
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
    def __init__(self, depth, dim, hidden_dim, head_dim, n_bits, dropout):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.n_bits = n_bits

        self.class_embed = nn.Embedding(10, dim)
        self.image_embed = nn.Linear(n_bits, dim)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = nn.Linear(dim, n_bits, bias=False)

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
        x = self.image_embed(x.float() * 2 - 1)
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


def gumbel_sigmoid(logits, tau):
    z = torch.rand_like(logits).logit_()
    return torch.sigmoid((logits + z) / tau)


def delta_orthogonal_(tensor, gain=1):
    spatial = tensor.shape[2:]
    if not all(d % 2 == 1 for d in spatial):
        raise ValueError("All spatial dimensions must be odd")
    mid = [d // 2 for d in spatial]
    idx = (slice(None), slice(None), *mid)
    nn.init.zeros_(tensor)
    nn.init.orthogonal_(tensor[idx], gain=gain)
    return tensor


class Quantizer(nn.Module):
    def __init__(self, ch, n_bits):
        super().__init__()
        self.ch = ch
        self.n_bits = n_bits
        self.act = nn.Tanh()
        self.enc_conv_1_1 = nn.Conv2d(ch, 128, 3, padding=1)
        self.enc_conv_1_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_conv_1_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_conv_1_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_conv_2_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc_conv_2_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.enc_conv_2_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.enc_conv_2_4 = nn.Conv2d(256, n_bits, 3, padding=1)
        self.dec_conv_2_4 = nn.Conv2d(n_bits, 256, 3, padding=1)
        self.dec_conv_2_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.dec_conv_2_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.dec_conv_2_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec_conv_1_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_conv_1_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_conv_1_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.dec_conv_1_1 = nn.Conv2d(128, ch * 2, 3, padding=1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                delta_orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def get_logits(self, x):
        x = self.enc_conv_1_1(x)
        x = self.act(x)
        x = self.enc_conv_1_2(x)
        x = self.act(x)
        x = self.enc_conv_1_3(x)
        x = self.act(x)
        x = self.enc_conv_1_4(x)
        x = self.act(x)
        x = F.avg_pool2d(x, 2)
        x = self.enc_conv_2_1(x)
        x = self.act(x)
        x = self.enc_conv_2_2(x)
        x = self.act(x)
        x = self.enc_conv_2_3(x)
        x = self.act(x)
        x = self.enc_conv_2_4(x)
        x = x.movedim(1, -1)
        return x

    def decode(self, x):
        x = x.float() * 2 - 1
        x = x.movedim(-1, 1)
        x = self.dec_conv_2_4(x)
        x = self.act(x)
        x = self.dec_conv_2_3(x)
        x = self.act(x)
        x = self.dec_conv_2_2(x)
        x = self.act(x)
        x = self.dec_conv_2_1(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.dec_conv_1_4(x)
        x = self.act(x)
        x = self.dec_conv_1_3(x)
        x = self.act(x)
        x = self.dec_conv_1_2(x)
        x = self.act(x)
        x = self.dec_conv_1_1(x)
        x_rec, log_scale = x.chunk(2, dim=1)
        return x_rec, log_scale

    def get_kl(self, logits):
        return 0.5 * (F.softplus(logits) + F.softplus(-logits)) - math.log(2)

    def forward(self, x, tau):
        logits = self.get_logits(x)
        kl = self.get_kl(logits)
        q = gumbel_sigmoid(logits, tau)
        x_rec, log_scale = self.decode(q)
        return x_rec, log_scale, kl


def main():
    batch_size = 64

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # n_levels = 24
    # palette = make_color_cube(n_levels, device=device)
    # vocab_size = n_levels**3
    n_bits = 8

    model_raw = Transformer(8, 512, 1360, 64, n_bits, dropout=0.0).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, drop_last=True
    )

    quantizer_raw = Quantizer(3, n_bits).to(device)
    du.broadcast_tensors(quantizer_raw.parameters())
    quantizer = nn.parallel.DistributedDataParallel(
        quantizer_raw, device_ids=[device], output_device=device
    )
    q_opt = optim.AdamW(quantizer.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n_epochs = 40
    n_steps_kl = len(dataloader) * n_epochs / 2
    n_steps_tau = len(dataloader) * n_epochs

    epoch = 0
    step = 0

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for (x, _) in tqdm(dataloader):
            x = x.to(device)
            kl_fac = math.sin(min(step / n_steps_kl, 1) * math.pi / 2) ** 2
            tau = max(math.cos(min(step / n_steps_tau, 1) * math.pi / 2) ** 2, 0.01)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x_rec, log_scale, loss_kl = quantizer(x, tau)
            loss_kl = torch.sum(loss_kl) / x.numel()
            # loss_rec = 0.5 * (log_scale + (x_rec - x) ** 2 * torch.exp(-log_scale) + math.log(2 * math.pi))
            loss_rec = log_scale + torch.abs(x_rec - x) * torch.exp(-log_scale) + math.log(2)
            loss_rec = torch.sum(loss_rec) / x.numel()
            loss = loss_rec + kl_fac * loss_kl
            q_opt.zero_grad()
            loss.backward()
            q_opt.step()
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_rec, op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_kl, op=dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}, loss_rec: {loss_rec.item():g}, loss_kl: {loss_kl.item():g}")
            step += 1
        epoch += 1

    quantizer_raw.eval().requires_grad_(False)

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
    def sample(model, y, temperature=1.0, disable=False):
        n = y.shape[0]
        n_proc = math.ceil(n / world_size)
        y = y.split(n_proc)[rank]
        x = torch.zeros(y.shape[0], 0, n_bits, dtype=torch.long, device=device)
        cache = model.init_cache(y.shape[0], 256, dtype=torch.bfloat16, device=device)
        index = 0
        for _ in trange(256, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, y, cache, index)[:, -1].float()
            if temperature != 1.0:
                logits /= temperature
            sample = torch.distributions.Bernoulli(logits=logits).sample()
            x = torch.cat((x, sample[:, None]), dim=1)
            index += 1
        # return torch.cat(dnn.all_gather(x))[:n]
        return torch.cat(du.all_gather_into_new(x))[:n]

    def demo(step):
        y = torch.arange(10, device=device).repeat_interleave(10)
        x = sample(model_ema, y)
        x = rearrange(x, "n (h w) nh -> n h w nh", h=16, w=16)
        x, _ = quantizer_raw.decode(x)
        x = (x + 1) / 2
        if rank == 0:
            x = rearrange(x, "(nh nw) d h w -> d (nh h) (nw w)", nh=10, nw=10, h=32, w=32)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_cifar_q_s_032_{epoch:04}_{step:05}.png")

    while True:
        sampler.set_epoch(epoch)

        for step, (x, y) in enumerate(tqdm(dataloader, disable=rank > 0)):
            if step % 200 == 0:
                demo(step)
            x = x.to(device)
            y = y.to(device)
            x_logits = quantizer_raw.get_logits(x).flatten(1, 2)
            x = torch.distributions.Bernoulli(logits=x_logits).sample()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x[:, :-1, :], y).float()
            loss = F.binary_cross_entropy_with_logits(logits, x_logits.sigmoid())
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.95)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

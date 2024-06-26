#!/usr/bin/env python3

import argparse
from copy import deepcopy
from functools import lru_cache
import math

from einops import rearrange
import flash_attn
from flash_attn.layers import rotary
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
            x = flash_attn.flash_attn_qkvpacked_func(
                qkv, self.drop.p if self.training else 0.0, causal=True
            )
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
    def __init__(self, depth, dim, hidden_dim, head_dim, vocab_size, n_vocab_heads, dropout):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.vocab_size = vocab_size
        self.n_vocab_heads = n_vocab_heads

        self.class_embed = nn.Embedding(10 + 1, dim)
        self.image_embed = nn.Linear(n_vocab_heads * vocab_size, dim)
        nn.init.normal_(self.image_embed.weight, std=n_vocab_heads**-0.5)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(dim, hidden_dim, head_dim, dropout) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = zero_init(nn.Linear(dim, n_vocab_heads * vocab_size, bias=False))

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
        x = self.image_embed(x).sum(-2)
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
        x = x.view(*x.shape[:-1], self.n_vocab_heads, self.vocab_size)
        return x


def orthogonal_rms_(tensor, gain=1):
    if tensor.ndim != 2:
        raise ValueError("Only 2D matrices are supported")
    o, i = tensor.shape
    gain = gain * max(1, math.sqrt(o / i))
    return nn.init.orthogonal_(tensor, gain=gain)


def delta_orthogonal_rms_(tensor, gain=1):
    spatial = tensor.shape[2:]
    if not all(d % 2 == 1 for d in spatial):
        raise ValueError("All spatial dimensions must be odd")
    mid = [d // 2 for d in spatial]
    idx = (slice(None), slice(None), *mid)
    nn.init.zeros_(tensor)
    return orthogonal_rms_(tensor[idx], gain=gain)


def rand_gumbel(shape, dtype=None, device=None):
    u = torch.rand(shape, dtype=dtype, device=device)
    return u.log_().nan_to_num_().neg_().log_().neg_()


def gumbel_softmax_log_density(log_x, log_pi, tau):
    k = log_x.new_tensor(log_x.shape[-1])
    term_1 = torch.special.gammaln(k) + torch.log(tau) * (k - 1)
    term_2 = -k * torch.logsumexp(log_pi - log_x * tau, dim=-1)
    term_3 = torch.sum(log_pi - log_x * (tau + 1), dim=-1)
    return term_1 + term_2 + term_3


def gumbel_softmax_kl(logits_p, logits_q, tau_p, tau_q, n=1):
    logits_p, logits_q = torch.broadcast_tensors(logits_p, logits_q)
    shape = (n, *logits_p.shape)
    g = rand_gumbel(shape, device=logits_p.device)
    log_y = F.log_softmax((logits_p + g) / tau_p, dim=-1)
    log_p = gumbel_softmax_log_density(log_y, logits_p, tau_p)
    log_q = gumbel_softmax_log_density(log_y, logits_q, tau_q)
    return torch.mean(log_p - log_q, dim=0)


class Quantizer(nn.Module):
    def __init__(self, ch, vocab_size, n_heads):
        super().__init__()
        self.ch = ch
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.encoder = nn.Sequential(
            nn.Conv2d(ch, 64, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(512, n_heads * vocab_size, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(n_heads * vocab_size, 512, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.Tanh(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Tanh(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Tanh(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, ch * 2, 3, padding=1),
        )
        for layer in [*self.encoder, *self.decoder]:
            if isinstance(layer, nn.Conv2d):
                delta_orthogonal_rms_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.decoder[0].weight.mul_(vocab_size**0.5)

    def get_logits(self, x):
        x = self.encoder(x)
        x = x.movedim(1, -1)
        x = x.view(*x.shape[:-1], self.n_heads, self.vocab_size)
        return x

    def decode(self, x):
        x = x.view(*x.shape[:-2], -1)
        x = x.movedim(-1, 1)
        x = self.decoder(x)
        x_rec, log_scale = x.chunk(2, dim=1)
        return x_rec, log_scale

    def get_kl(self, logits):
        logits_prior = logits.new_tensor(0.0)
        tau = logits.new_tensor(1.0)
        return gumbel_softmax_kl(logits_prior, logits, tau, tau)

    def forward(self, x):
        logits = self.get_logits(x)
        kl = self.get_kl(logits)
        one_hot = F.gumbel_softmax(logits, tau=1.0)
        x_rec, log_scale = self.decode(one_hot)
        return x_rec, log_scale, kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="run")
    args = parser.parse_args()

    batch_size = 32

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    vocab_size = 256
    n_heads = 1

    model_raw = Transformer(8, 512, 1360, 64, vocab_size, n_heads, dropout=0.0).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    transform = transforms.Compose(
        [
            transforms.Resize(128, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    dataset = datasets.ImageFolder("/home/kat/datasets/artbench-10/artbench-10-imagefolder-split/train", transform=transform)
    sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    quantizer_raw = Quantizer(3, vocab_size, n_heads).to(device)
    du.broadcast_tensors(quantizer_raw.parameters())
    quantizer_ema = deepcopy(quantizer_raw).eval().requires_grad_(False)
    quantizer = nn.parallel.DistributedDataParallel(
        quantizer_raw, device_ids=[device], output_device=device
    )
    q_opt = optim.AdamW(quantizer.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    n_epochs = 10
    n_steps_kl = len(dataloader) * n_epochs / 2

    epoch = 0
    step = 0

    def demo_rec(x, x_rec):
        x = torch.stack((x, x_rec.to(x)))
        x = (x + 1) / 2
        x = rearrange(x, "t (nh nw) d h w -> d (nh h) (nw t w)", nh=8, nw=4)
        x = torch.clamp(x, 0, 1)
        TF.to_pil_image(x.float().cpu()).save(f"demo_{args.name}_s1_{epoch:04}_{step:05}.png")

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            kl_fac = math.sin(min(step / n_steps_kl, 1) * math.pi / 2) ** 2
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x_rec, log_scale, loss_kl = quantizer(x)
            loss_kl = torch.sum(loss_kl) / x.numel()
            # loss_rec = 0.5 * (log_scale + (x_rec - x) ** 2 * torch.exp(-log_scale) + math.log(2 * math.pi))
            loss_rec = log_scale + torch.abs(x_rec - x) * torch.exp(-log_scale) + math.log(2)
            loss_rec = torch.sum(loss_rec) / x.numel()
            loss = loss_rec + kl_fac * loss_kl
            q_opt.zero_grad()
            loss.backward()
            q_opt.step()
            ema_update(quantizer_raw, quantizer_ema, 0.99)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_rec, op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_kl, op=dist.ReduceOp.AVG)
            print0(
                f"epoch: {epoch}, step: {step}, loss: {loss.item():g}, loss_rec: {loss_rec.item():g}, loss_kl: {loss_kl.item():g}"
            )
            step += 1
        epoch += 1
        if rank == 0:
            demo_rec(x, x_rec)
        dist.barrier()

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
        x = torch.zeros(y.shape[0], 0, n_heads, vocab_size, device=device)
        cache = model.init_cache(y.shape[0], 16 * 16, dtype=torch.bfloat16, device=device)
        index = 0
        for _ in trange(16 * 16, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, y, cache, index)[:, -1].float()
            if temperature != 1.0:
                logits /= temperature
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.softmax(logits + gumbel, dim=-1)
            x = torch.cat((x, sample[:, None]), dim=1)
            index += 1
        # return torch.cat(dnn.all_gather(x))[:n]
        return torch.cat(du.all_gather_into_new(x))[:n]

    def demo(step):
        y = torch.randint(0, 10, (64,), dtype=torch.long, device=device)
        x = sample(model_ema, y)
        x = rearrange(x, "n (h w) nh v -> n h w nh v", h=16, w=16)
        x, _ = quantizer_ema.decode(x)
        x = (x + 1) / 2
        if rank == 0:
            x = rearrange(x, "(nh nw) d h w -> d (nh h) (nw w)", nh=8, nw=8, h=128, w=128)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_{args.name}_s2_{epoch:04}_{step:05}.png")

    while True:
        sampler.set_epoch(epoch)

        for step, (x, y) in enumerate(tqdm(dataloader, disable=rank > 0)):
            if step % 500 == 0:
                demo(step)
            x = x.to(device)
            y = y.to(device)
            x_logits = quantizer_ema.get_logits(x).flatten(1, 2)
            x = F.gumbel_softmax(x_logits)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x[:, :-1, :], y).float()
            tau = logits.new_tensor(1.0)
            loss = gumbel_softmax_kl(x_logits, logits, tau, tau).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.99)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

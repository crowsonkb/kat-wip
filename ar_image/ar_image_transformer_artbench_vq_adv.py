#!/usr/bin/env python3

import argparse
from copy import deepcopy
from functools import lru_cache
import math

from einops import rearrange
import flash_attn
from flash_attn.layers import rotary
import lpips
import natten
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
        self.image_embed = nn.Embedding(n_vocab_heads * vocab_size, dim)
        # nn.init.normal_(self.image_embed.weight, std=n_vocab_heads**-0.5)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(dim, hidden_dim, head_dim, dropout) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = nn.Linear(dim, n_vocab_heads * vocab_size, bias=False)
        self.image_embed.weight = self.out_proj.weight

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
        x = x + torch.arange(0, self.n_vocab_heads, device=x.device) * self.vocab_size
        x = self.image_embed(x).sum(dim=-2)
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


def zgr(logits, sample):
    one_hot = F.one_hot(sample, logits.shape[-1]).to(logits.dtype)
    logprobs = F.log_softmax(logits, dim=-1)
    st_est = torch.exp(logprobs)
    darn_est = (one_hot - st_est.detach()) * logprobs.gather(-1, sample[..., None])
    zgr_est = (st_est + darn_est) / 2
    return one_hot + (zgr_est - zgr_est.detach())


def orthogonal_rms_(tensor, gain=1):
    if tensor.ndim != 2:
        raise ValueError("Only 2D matrices are supported")
    o, i = tensor.shape
    gain = gain * max(1, math.sqrt(o / i))
    return nn.init.orthogonal_(tensor, gain=gain)


def delta_orthogonal_(tensor, gain=1):
    spatial = tensor.shape[2:]
    if not all(d % 2 == 1 for d in spatial):
        raise ValueError("All spatial dimensions must be odd")
    mid = [d // 2 for d in spatial]
    idx = (slice(None), slice(None), *mid)
    nn.init.zeros_(tensor)
    return nn.init.orthogonal_(tensor[idx], gain=gain)


def delta_orthogonal_rms_(tensor, gain=1):
    spatial = tensor.shape[2:]
    if not all(d % 2 == 1 for d in spatial):
        raise ValueError("All spatial dimensions must be odd")
    mid = [d // 2 for d in spatial]
    idx = (slice(None), slice(None), *mid)
    nn.init.zeros_(tensor)
    return orthogonal_rms_(tensor[idx], gain=gain)


class QuantizerSelfAttention(nn.Module):
    def __init__(self, dim, head_dim, kernel_size):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.kernel_size = kernel_size
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.rpb = nn.Parameter(torch.empty(self.n_heads, 2 * kernel_size - 1, 2 * kernel_size - 1))
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.normal_(self.rpb, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        skip = x
        x = x.movedim(1, -1)
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=self.head_dim)
        q = q / math.sqrt(self.head_dim)
        qk = natten.functional.natten2dqkrpb(q, k, self.rpb, self.kernel_size, 1)
        a = torch.softmax(qk, dim=-1)
        x = natten.functional.natten2dav(a, v, self.kernel_size, 1)
        x = rearrange(x, "n nh h w e -> n h w (nh e)")
        x = self.out_proj(x)
        x = x.movedim(-1, 1)
        return x + skip


class Quantizer(nn.Module):
    def __init__(self, ch, vocab_size, n_heads):
        super().__init__()
        self.ch = ch
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.encoder = nn.Sequential(
            nn.Conv2d(ch, 96, 3, padding=1),
            nn.Tanh(),
            # nn.Conv2d(96, 96, 3, padding=1),
            # nn.Tanh(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1),
            nn.Tanh(),
            # nn.Conv2d(192, 192, 3, padding=1),
            # nn.Tanh(),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.Tanh(),
            # nn.Conv2d(384, 384, 3, padding=1),
            # nn.Tanh(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(384, 768, 3, padding=1),
            nn.Tanh(),
            # nn.Conv2d(768, 768, 3, padding=1),
            # nn.Tanh(),
            nn.Conv2d(768, n_heads * vocab_size, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(n_heads * vocab_size, 768, 1),
            nn.Tanh(),
            # nn.Conv2d(768, 768, 3, padding=1),
            # nn.Tanh(),
            nn.Conv2d(768, 384, 3, padding=1),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.Tanh(),
            # nn.Conv2d(384, 384, 3, padding=1),
            # nn.Tanh(),
            nn.Conv2d(384, 192, 3, padding=1),
            nn.Tanh(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.Tanh(),
            # nn.Conv2d(192, 192, 3, padding=1),
            # nn.Tanh(),
            nn.Conv2d(192, 96, 3, padding=1),
            nn.Tanh(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.Tanh(),
            # nn.Conv2d(96, 96, 3, padding=1),
            # nn.Tanh(),
        )
        self.decoder_rec = nn.Conv2d(96, ch, 3, padding=1)
        self.decoder_aux = nn.Conv2d(96, 1, 3, padding=1)
        for layer in [*self.encoder, *self.decoder, self.decoder_rec, self.decoder_aux]:
            if isinstance(layer, nn.Conv2d):
                delta_orthogonal_rms_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    @property
    def last_param(self):
        return self.decoder_rec.weight

    def get_logits(self, x):
        x = self.encoder(x)
        x = x.movedim(1, -1)
        x = x.view(*x.shape[:-1], self.n_heads, self.vocab_size)
        return x

    def decode(self, x):
        if x.dtype == torch.long:
            x = F.one_hot(x, self.vocab_size).float()
        x = x.view(*x.shape[:-2], -1)
        x = x.movedim(-1, 1)
        x = self.decoder(x)
        x_rec = self.decoder_rec(x)
        log_scale = self.decoder_aux(x)
        return x_rec, log_scale

    def get_kl(self, logits):
        logp = F.log_softmax(logits, dim=-1)
        logq = -math.log(logits.shape[-1])
        return torch.sum(logp.exp() * (logp - logq), dim=-1)

    def quantize_logits(self, logits):
        sample = sample_categorical(logits)
        return zgr(logits, sample)

    def forward(self, x, tau):
        logits = self.get_logits(x)
        kl = self.get_kl(logits)
        # one_hot = self.quantize_logits(logits)
        one_hot = F.gumbel_softmax(logits, tau)
        x_rec, log_scale = self.decode(one_hot)
        return x_rec, log_scale, kl


class Discriminator(nn.Sequential):
    def __init__(self, ch):
        super().__init__(
            nn.Conv2d(ch, 64, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(256, 1, 3, padding=1),
        )
        for layer in self:
            if isinstance(layer, nn.Conv2d):
                delta_orthogonal_rms_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


def sample_categorical(logits, tau=1.0):
    gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
    return torch.argmax(logits + gumbel * tau, dim=-1)


def kl_divergence_categorical(logits_p, logits_q):
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    return torch.sum(torch.exp(log_p) * torch.nan_to_num(log_p - log_q), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="run")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    batch_size = 32

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    vocab_size = 256
    n_heads = 8

    model_raw = Transformer(12, 768, 2048, 64, vocab_size, n_heads, dropout=0.0).to(device)
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
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    lpips_net = lpips.LPIPS(net='vgg', spatial=True).to(device)
    lpips_net.compile()

    quantizer_raw = Quantizer(3, vocab_size, n_heads).to(device)
    du.broadcast_tensors(quantizer_raw.parameters())
    quantizer_ema = deepcopy(quantizer_raw).eval().requires_grad_(False)
    d_raw = Discriminator(3).to(device)
    du.broadcast_tensors(d_raw.parameters())

    q_params = list(p for p in quantizer_raw.parameters() if p.requires_grad)
    d_params = list(p for p in d_raw.parameters() if p.requires_grad)
    last_param = quantizer_raw.last_param
    print0(f"VAE parameters: {sum(p.numel() for p in q_params):,}")
    print0(f"D parameters: {sum(p.numel() for p in d_params):,}")

    quantizer = nn.parallel.DistributedDataParallel(
        quantizer_raw, device_ids=[device], output_device=device
    )
    d = nn.parallel.DistributedDataParallel(d_raw, device_ids=[device], output_device=device)

    q_opt = optim.AdamW(quantizer.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.01)
    d_opt = optim.AdamW(d.parameters(), lr=2e-4, betas=(0.5, 0.95), weight_decay=0.01)    
    grad_rec_ema = torch.tensor(0.0, device=device)
    grad_adv_ema = torch.tensor(0.0, device=device)

    n_epochs = 50
    n_steps_adv = len(dataloader)
    n_steps_kl = 5000
    n_steps_tau = len(dataloader) * n_epochs

    epoch = 0
    step = 0

    def demo_rec(x, x_rec, x_prior):
        x = torch.stack((x, x_rec.to(x)))
        x = rearrange(x, "t (nh nw) d h w -> d (nh h) (nw t w)", nh=8, nw=4)
        x_prior = rearrange(x_prior[:16], "(nh nw) d h w -> d (nh h) (nw w)", nh=2, nw=8)
        x = torch.cat((x, x_prior), dim=1)
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        TF.to_pil_image(x.float().cpu()).save(f"demo_{args.name}_s1_{epoch:04}_{step:05}.png")

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for x, _ in tqdm(dataloader):
            x = x.to(device)

            adv_fac = 0.1  # + 0.05 * math.sin(min(step / n_steps_adv, 1) * math.pi / 2) ** 2
            kl_fac = 10 * math.sin(min(step / n_steps_kl, 1) * math.pi / 2) ** 2
            tau = max(math.cos(min(step / n_steps_tau, 1) * math.pi / 2) ** 2, 1 / 16)
            # is_d = step >= n_steps_adv
            is_d = True

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                x_rec, log_scale, loss_kl = quantizer(x, tau)
                lpips_err = lpips_net(x_rec, x)
            # l1_err = torch.mean(torch.abs(x_rec - x), dim=1, keepdim=True)
            l2_err = torch.mean((x_rec - x) ** 2, dim=1, keepdim=True)

            # e = torch.rand(x.shape[0], 1, 1, 1, device=device)
            # x_hat = e * x.detach() + (1 - e) * x_rec.detach()
            # x_hat.requires_grad_(True)

            loss_kl = torch.sum(loss_kl) / x.numel()
            # loss_rec = 0.5 * (log_scale + (x_rec - x) ** 2 * torch.exp(-log_scale) + math.log(2 * math.pi))
            # loss_rec = log_scale + torch.abs(x_rec - x) * torch.exp(-log_scale) + math.log(2)
            # loss_rec = torch.sum(loss_rec) / x.numel()
            loss_rec = -torch.distributions.Exponential(torch.exp(-log_scale)).log_prob(lpips_err + l2_err).sum() / lpips_err.numel()
            loss = loss_rec + kl_fac * loss_kl

            if is_d:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    d_reals = d(x).float()
                    d_fakes = d(x_rec).float()
                    # d_x_hat = d(x_hat).float()
                # loss_adv = F.softplus(-d_fakes).mean()
                # loss_adv = torch.mean(d_fakes**2) / 2
                loss_adv = -torch.mean(d_fakes)
                # loss_adv = -F.logsigmoid(d_fakes).mean()
                # loss_adv = -torch.mean(d_fakes)

                # grad_rec = torch.autograd.grad(loss_rec, last_param, retain_graph=True)[0]
                # grad_adv = torch.autograd.grad(loss_adv, last_param, retain_graph=True)[0]
                # grad_rec_sq = torch.sum(grad_rec**2)
                # grad_adv_sq = torch.sum(grad_adv**2)
                # dist.all_reduce_coalesced([grad_rec_sq, grad_adv_sq], op=dist.ReduceOp.AVG)
                # grad_rec_ema.mul_(0.99).add_(grad_rec_sq, alpha=0.01)
                # grad_adv_ema.mul_(0.99).add_(grad_adv_sq, alpha=0.01)
                # adv_grad_fac = torch.sqrt(grad_rec_ema / grad_adv_ema)
                # print0(f"adv grad fac: {adv_grad_fac.item():g}")
                # adv_grad_fac = torch.clamp_max(grad_rec.norm() / (grad_adv.norm() + 1e-4), 1e4)
                adv_grad_fac = 1
                loss_final = loss + adv_fac * adv_grad_fac * loss_adv
            else:
                loss_adv = loss.new_tensor(0.0)
                loss_final = loss

            q_opt.zero_grad()
            loss_final.backward(inputs=q_params, retain_graph=is_d)
            q_opt.step()
            ema_update(quantizer_raw, quantizer_ema, 0.99)

            # loss_d = F.softplus(d_fakes).mean() + F.softplus(-d_reals).mean()
            # loss_d = torch.mean((d_reals - 1) ** 2) / 2 + torch.mean((d_fakes + 1) ** 2) / 2
            loss_d = torch.mean(torch.exp(d_fakes - 1)) - torch.mean(d_reals)
            # loss_d = F.sigmoid(d_fakes).mean() - F.logsigmoid(d_reals).mean()
            # loss_d = torch.mean(d_fakes + d_fakes**2 / 4) - torch.mean(d_reals)
            # grad = torch.autograd.grad(d_x_hat.sum(), x_hat, create_graph=True)[0]
            # grad_sq = torch.sum(grad**2, dim=(1, 2, 3))
            # loss_gp = torch.mean((torch.sqrt(grad_sq) - 1) ** 2) / 2
            # loss_d_final = loss_d  # + 10 * loss_gp

            if is_d:
                # loss_d = torch.mean((d_reals - 1) ** 2) / 2 + torch.mean((d_fakes + 1) ** 2) / 2
                # loss_d = torch.mean(F.relu(1 - d_reals)) + torch.mean(F.relu(1 + d_fakes))
                d_opt.zero_grad()
                loss_d.backward(inputs=d_params)
                d_opt.step()
            else:
                loss_d = loss.new_tensor(0.0)

            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_rec, op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_kl, op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_adv, op=dist.ReduceOp.AVG)
            dist.all_reduce(loss_d, op=dist.ReduceOp.AVG)
            #dist.all_reduce(loss_gp, op=dist.ReduceOp.AVG)

            print0(
                f"epoch: {epoch}, step: {step}, loss: {loss.item():.4f}, rec: {loss_rec.item():.4f}, kl: {loss_kl.item():.4f}, adv: {loss_adv.item():.4f}, d: {loss_d.item():.4f}"
            )
            step += 1

        epoch += 1
        if rank == 0:
            uniform = torch.randint(0, vocab_size, (16, 16, 16, n_heads), dtype=torch.long, device=device)
            with torch.no_grad():
                x_rec, *_ = quantizer_ema(x, tau)
                x_prior, *_ = quantizer_ema.decode(uniform)
                demo_rec(x, x_rec, x_prior)
        dist.barrier()

    wd, no_wd = [], []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_wd.append(param)
        else:
            wd.append(param)
    groups = [{"params": wd}, {"params": no_wd, "weight_decay": 0}]
    opt = optim.AdamW(groups, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)

    epoch = 0

    @torch.no_grad()
    def sample(model, y, temperature=1.0, disable=False):
        n = y.shape[0]
        n_proc = math.ceil(n / world_size)
        y = y.split(n_proc)[rank]
        x = torch.zeros(y.shape[0], 0, n_heads, dtype=torch.long, device=device)
        cache = model.init_cache(y.shape[0], 16 * 16, dtype=torch.bfloat16, device=device)
        index = 0
        for _ in trange(16 * 16, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, y, cache, index)[:, -1].float()
            if temperature != 1.0:
                logits /= temperature
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.argmax(logits + gumbel, dim=-1)
            x = torch.cat((x, sample[:, None]), dim=1)
            index += 1
        # return torch.cat(dnn.all_gather(x))[:n]
        return torch.cat(du.all_gather_into_new(x))[:n]

    def demo(step):
        y = torch.randint(0, 10, (64,), dtype=torch.long, device=device)
        x = sample(model_ema, y)
        x = rearrange(x, "n (h w) nh -> n h w nh", h=16, w=16)
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
            x = sample_categorical(x_logits)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x[:, :-1, :], y).float()
            loss = kl_divergence_categorical(x_logits, logits).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, 0.99)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

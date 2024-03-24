#!/usr/bin/env python3

from copy import deepcopy
from functools import lru_cache, reduce
import math

from einops import rearrange
import numpy as np
import torch
from torch import distributed as dist, nn, optim
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


class EMAWarmup:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(
        self, inv_gamma=1.0, power=1.0, min_value=0.0, max_value=1.0, start_at=0, last_epoch=0
    ):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the class as a :class:`dict`."""
        return dict(self.__dict__.items())

    def load_state_dict(self, state_dict):
        """Loads the class's state.
        Args:
            state_dict (dict): scaler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0.0 if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1


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
    def __init__(self, dim, head_dim, dropout):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.dropout_p = dropout
        self.norm = nn.LayerNorm((dim,))
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = zero_init(nn.Linear(dim, dim))
        self.drop = nn.Dropout(dropout)
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, self.n_heads * head_dim // 8 + 1)[:-1].exp()
        freqs = freqs.view(head_dim // 8, self.n_heads).T.contiguous()
        # TODO: allow changing image size
        pos = make_axial_pos(32, 32)
        # make room for the class token and the last backspace token
        pos = torch.cat((torch.zeros(1, 2), pos))
        theta_h = pos[..., None, 0:1] * freqs
        theta_w = pos[..., None, 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("theta", theta)

    def forward(self, x, cache=None, index=None):
        x = self.norm(x)
        qkv = self.qkv_proj(x).reshape(x.shape[0], -1, 3, self.n_heads, self.head_dim)  # n s t h d
        # transpose to n h t s d
        qkv = qkv.transpose(1, 3)
        q, k, v = qkv.unbind(2)
        pos = torch.arange(1025, device=x.device).tile((x.shape[0], 1))
        if index is None:
            pos = pos[:, :x.shape[1]]
            q = apply_rotary_emb_(q, self.theta[pos].transpose(1, 2).to(q))
            k = apply_rotary_emb_(k, self.theta[pos].transpose(1, 2).to(k))
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
            if cache is not None:
                cache[0][:] = k
                cache[1][:] = v
        else:
            b_idx = torch.arange(x.shape[0], device=x.device)
            max_index = torch.amax(index)
            q = apply_rotary_emb_(q, self.theta[pos[b_idx, index]][:, None].transpose(1, 2).to(q))
            k = apply_rotary_emb_(k, self.theta[pos[b_idx, index]][:, None].transpose(1, 2).to(k))
            cache[0][b_idx, :, index] = k[:, :, 0, :]
            cache[1][b_idx, :, index] = v[:, :, 0, :]
            mask = torch.ones(cache[0].shape[-2], cache[0].shape[-2], dtype=torch.bool, device=x.device).tril()
            mask = mask[index, :max_index][:, None, None]
            x = F.scaled_dot_product_attention(q, cache[0][:, :, :max_index], cache[1][:, :, :max_index], mask)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.n_heads * self.head_dim)
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
    def __init__(self, depth, dim, hidden_dim, head_dim, vocab_size, dropout):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.class_embed = nn.Embedding(10, dim)
        self.image_embed = nn.Embedding(vocab_size, dim)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim, dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        self.p_sampled_out_proj = nn.Linear(dim, 1, bias=False)

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

    def forward(self, x, cache=None, index=None):
        y = self.class_embed(x[:, :1])
        x = self.image_embed(x[:, 1:])
        x = torch.cat((y, x), dim=1)
        b_idx = torch.arange(x.shape[0], device=x.device)
        x = x if index is None else x[b_idx, index][:, None]
        x = self.embed_drop(x)
        cache = [None] * self.depth if cache is None else cache
        for block, cache_block in zip(self.blocks, cache):
            # x = torch.utils.checkpoint.checkpoint(block, x, cache_block, index)
            x = block(x, cache_block, index)
        x = self.out_norm(x)
        s_logit = self.p_sampled_out_proj(x)[..., 0]
        x = self.out_proj(x)
        return x, s_logit


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

    model_raw = Transformer(8, 512, 1360, 64, vocab_size, dropout=0.0).to(device)
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
    ema_sched = EMAWarmup(power=3 / 4, max_value=0.99)

    epoch = 0

    @torch.no_grad()
    def sample(model, y, temperature=1.0, disable=False):
        n = y.shape[0]
        n_proc = math.ceil(n / world_size)
        y = y.split(n_proc)[rank]
        x = torch.full((y.shape[0], 1024 + 1), 0, dtype=torch.long, device=device)
        x[:, 0] = y
        s_logit = torch.zeros(y.shape[0], 1024 + 1, dtype=torch.float32, device=device)
        cache = model.init_cache(y.shape[0], 1024 + 1, dtype=torch.bfloat16, device=device)
        index = torch.zeros(y.shape[0], dtype=torch.long, device=device)
        b_idx = torch.arange(y.shape[0], device=x.device)
        for _ in trange(1024, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, s_logit_new = model(x, cache, index)
            logits = logits[:, -1].float()
            s_logit_new = s_logit_new[:, -1].float()
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.argmax(logits + gumbel * temperature, dim=-1)
            index += 1
            x[b_idx, index] = sample
            s_logit[b_idx, index] = s_logit_new
        x = torch.cat(du.all_gather_into_new(x))[:n]
        s_logit = torch.cat(du.all_gather_into_new(s_logit))[:n]
        return x[:, 1:], s_logit[:, 1:]

    @torch.no_grad()
    def sample_for_training(model, x, index):
        cache = model.init_cache(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=device)
        model(x, cache)
        b_idx = torch.arange(x.shape[0], device=x.device)
        while index.amin() < x.shape[1] - 1:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, _ = model(x, cache, index)
            logits = logits[:, -1].float()
            gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
            sample = torch.argmax(logits + gumbel, dim=-1)
            index += 1
            index.clamp_max_(x.shape[1] - 1)
            x[b_idx, index] = sample
        return x

    def demo(step):
        y = torch.arange(10, device=device).repeat_interleave(10)
        x, s_logit = sample(model_ema, y)
        if rank == 0:
            x = dequantize(x, palette)
            x = rearrange(x, "(nh nw) (h w) d -> d (nh h) (nw w)", nh=10, nw=10, h=32, w=32)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_cifar_p_sampled_007_{epoch:04}_{step:05}.png")
            p_sampled = s_logit[..., -1].sigmoid()
            print(f"min p sampled: {p_sampled.min().item():g}")
            print(f"mean p sampled: {p_sampled.mean().item():g}")
            print(f"max p sampled: {p_sampled.max().item():g}")
            print(f"frac sampled: {p_sampled.gt(0.5).float().mean().item():g}")

    buf, cls_buf = [], []

    while True:
        sampler.set_epoch(epoch)

        for step, (x, y) in enumerate(tqdm(dataloader, disable=rank > 0)):
            if step % 100 == 0:
                demo(step)

            x = x.to(device).flatten(1, 2).float() / 255
            x = quantize(x, palette)
            y = y.to(device)
            x = torch.cat((y[:, None], x), dim=1)

            if step % 8 == 0:
                if len(buf) == 4:
                    buf.pop(0)
                    cls_buf.pop(0)
                while len(buf) < 4:
                    x_m_cur = x.tile((4, 1))
                    index = torch.randint(1, x_m_cur.shape[1] - 1, (x_m_cur.shape[0],), device=device)
                    buf.append(sample_for_training(model_raw, x_m_cur, index.clone()))
                    cls_cur = torch.arange(x_m_cur.shape[1], device=device).tile((x_m_cur.shape[0], 1))
                    cls_cur = (cls_cur > index[:, None]).float()
                    cls_buf.append(cls_cur)

            x_m_all = torch.cat(buf)
            cls_m_all = torch.cat(cls_buf)
            indices = torch.randperm(x_m_all.shape[0])[:batch_size]
            x_m = x_m_all[indices]
            cls_m = cls_m_all[indices]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, s_logit_d = model(x[:, :-1])
            logits = logits.float()
            s_logit_d = s_logit_d.float()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, s_logit_m = model(x_m)
            s_logit_m = s_logit_m.float()

            loss_xent = F.cross_entropy(logits.mT, x[:, 1:])
            pos_weight = torch.tensor(3.0, device=device)
            loss_p_sampled_d = F.binary_cross_entropy_with_logits(s_logit_d, torch.zeros_like(s_logit_d), pos_weight=pos_weight)
            loss_p_sampled_m = F.binary_cross_entropy_with_logits(s_logit_m, cls_m, pos_weight=pos_weight)
            loss_p_sampled = (loss_p_sampled_d + loss_p_sampled_m) / 2
            loss = loss_xent + loss_p_sampled
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, ema_sched.get_value())
            ema_sched.step()
            print0(f"epoch: {epoch}, step: {step}, xent: {loss_xent.item():g}, p_sampled: {loss_p_sampled.item():g}")

        epoch += 1


if __name__ == "__main__":
    main()

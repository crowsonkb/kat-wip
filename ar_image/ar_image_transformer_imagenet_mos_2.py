#!/usr/bin/env python3

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache, reduce, update_wrapper
import math

from einops import rearrange
import flash_attn
from flash_attn.layers import rotary
from ldm.util import instantiate_from_config
from megablocks import grouped_gemm_util as gg
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
import torch
from torch import distributed as dist, nn, optim
import torch.distributed.nn as dnn
from torch.nn import functional as F
from torch.utils import data
import torch_dist_utils as du
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm import trange, tqdm

from sinkhorn import log_sinkhorn

print = tqdm.external_write_mode()(print)
print0 = tqdm.external_write_mode()(du.print0)

gmm = torch.compiler.disable(gg.ops.gmm)

@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


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


def checkpoint(func, *args, enable=False, **kwargs):
    use_reentrant = kwargs.pop("use_reentrant", True)
    if enable:
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=use_reentrant, **kwargs)
    return func(*args, **kwargs)


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


class compile_wrap:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        try:
            self._compiled_function = torch.compile(self.function, *self.args, **self.kwargs)
        except RuntimeError:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        return self.compiled_function(*args, **kwargs)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == "taming.models.vqgan.GumbelVQ":
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == "ldm.models.autoencoder.VQModel":
        model = instantiate_from_config(config.model)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f"unknown model type: {config.model.target}")
    # del model.loss
    return model


def sample_categorical(logits, tau=1.0):
    gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
    return torch.argmax(logits + gumbel * tau, dim=-1)


def kl_divergence(logits_p, logits_q):
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    return torch.sum(torch.exp(log_p) * (log_p - log_q), dim=-1)


# TODO: do this correctly instead
@lru_cache
def make_axial_pos(h, w, dtype=None, device=None):
    h_pos = torch.linspace(-1, 1, h + 1, dtype=dtype, device=device)
    w_pos = torch.linspace(-1, 1, w + 1, dtype=dtype, device=device)
    h_pos = (h_pos[:-1] + h_pos[1:]) / 2
    w_pos = (w_pos[:-1] + w_pos[1:]) / 2
    return torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1).view(h * w, 2)


@compile_wrap
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


@compile_wrap
def linear_swiglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


@compile_wrap
def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_swiglu(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)


class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.norm = AdaRMSNorm(dim, 256)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = zero_init(nn.Linear(dim, dim, bias=False))
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, head_dim // 8).exp()
        # TODO: allow changing image size
        pos = make_axial_pos(32, 32)
        # make room for the first token
        pos = torch.cat((torch.zeros(1, 2), pos))
        theta_h = pos[..., 0:1] * freqs
        theta_w = pos[..., 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("cos", torch.cos(theta))
        self.register_buffer("sin", torch.sin(theta))

    def forward(self, x, cond, cache=None, index=None):
        skip = x
        x = self.norm(x, cond)
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
        x = self.out_proj(x)
        return x + skip


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = AdaRMSNorm(dim, 256)
        self.up = LinearSwiGLU(dim, hidden_dim, bias=False)
        self.down = zero_init(nn.Linear(hidden_dim, dim, bias=False))

    def forward(self, x, cond):
        skip = x
        x = self.norm(x, cond)
        x = self.up(x)
        x = self.down(x)
        return x + skip


class Block(nn.Module):
    def __init__(self, dim, hidden_dim, head_dim):
        super().__init__()
        self.attn = SelfAttention(dim, head_dim)
        self.ff = FeedForward(dim, hidden_dim)

    def forward(self, x, cond, cache=None, index=None):
        x = self.attn(x, cond, cache, index)
        x = self.ff(x, cond)
        return x


@dataclass
class DMoEState:
    shape: tuple[int, ...]
    ids_sorted: torch.Tensor
    ids_indices: torch.Tensor
    batch_sizes: torch.Tensor


class DMoELinear(nn.Module):
    def __init__(self, in_features, out_features, n_experts, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_experts = n_experts
        self.weight = nn.Parameter(torch.empty(n_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(n_experts, out_features)) if bias else None
        with torch.no_grad():
            for i in range(n_experts):
                layer = nn.Linear(in_features, out_features, bias=bias)
                self.weight[i].copy_(layer.weight)
                if bias:
                    self.bias[i].copy_(layer.bias)

    def preprocess(self, x, ids):
        shape = x.shape[:-1]
        x = x.flatten(0, -2)
        ids = ids.flatten()
        ids_sorted, ids_indices = torch.sort(ids)
        x_sorted = x[ids_indices]
        batch_sizes = torch.bincount(ids_sorted, minlength=self.n_experts).cpu()
        state = DMoEState(shape, ids_sorted, ids_indices, batch_sizes)
        return x_sorted, state

    def postprocess(self, x, state):
        x_real = torch.empty_like(x)
        x_real.scatter_(0, state.ids_indices[..., None].expand_as(x), x)
        return x_real.view(*state.shape, self.out_features)

    def forward(self, x, state):
        x = gmm(x.bfloat16(), self.weight.bfloat16(), state.batch_sizes, trans_b=True)
        if self.bias is not None:
            x += self.bias[state.ids_sorted].bfloat16()
        return x


class Router2(nn.Module):
    def __init__(self, dim, n, k):
        super().__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(dim, n, bias=False)

    def forward(self, x):
        scores = self.linear(x)
        if self.training:
            n, s, d = scores.shape
            p = log_sinkhorn(scores.view(n * s, d), 10).view(n, s, d)
            _, indices = torch.topk(p, self.k, dim=-1)
            c = torch.gather(scores, -1, indices)
        else:
            c, indices = torch.topk(scores, self.k, dim=-1)
        return indices, c


class MoEHead2(nn.Module):
    def __init__(self, dim, vocab_size, n, k):
        super().__init__()
        self.k = k
        self.router = Router2(dim, n, k)
        self.up = DMoELinear(dim, dim * 8 // 3 * 2, n, bias=False)
        self.down = DMoELinear(dim * 8 // 3, dim, n, bias=False)
        self.proj = DMoELinear(dim, vocab_size, n, bias=False)
        with torch.no_grad():
            for i in range(1, n):
                self.proj.weight[i].copy_(self.proj.weight[0])

    def forward(self, x):
        ids, c = self.router(x)
        x = x[..., None, :].expand(-1, -1, self.k, -1)
        x, state = self.proj.preprocess(x, ids)
        skip = x
        x = self.up(x, state)
        x = swiglu(x)
        x = self.down(x, state)
        x = x + skip
        x = self.proj(x, state)
        x = self.proj.postprocess(x, state)
        c = F.softmax(c, dim=-1)
        x = torch.sum(x * c[..., None], dim=-2)
        return x


class Transformer(nn.Module):
    def __init__(self, depth, dim, hidden_dim, head_dim):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.class_embed = nn.Embedding(1001, 256)
        self.first_token = nn.Parameter(torch.randn(dim))
        # self.image_embed = nn.Embedding(8192, dim)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = RMSNorm((dim,))
        # self.out_proj = zero_init(nn.Linear(dim, 8192, bias=False))
        self.out_proj = MoEHead2(dim, 8192, 8, 3)
        self.out_proj.compile()

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
        # x = self.image_embed(x)
        x = F.embedding(x, self.out_proj.proj.weight.mean(dim=0))
        y = self.class_embed(y)
        first_token = self.first_token[None, None, :].expand(x.shape[0], 1, -1)
        x = torch.cat((first_token, x), dim=1)
        x = x[:, -1:].contiguous() if cache and index > 0 else x
        cache = [None] * self.depth if cache is None else cache
        for block, cache_block in zip(self.blocks, cache):
            x = checkpoint(block, x, y, cache_block, index, enable=self.training)
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x


def main():
    batch_size = 64

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    config_path = "/home/kat/text-to-image/vqgan_models/vqgan_gumbel_f8_8192.yaml"
    model_path = "/home/kat/text-to-image/vqgan_models/vqgan_gumbel_f8_8192.ckpt"
    ae = load_vqgan_model(config_path, model_path).to(device)
    ae.eval().requires_grad_(False)

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def encode(x):
        x = x * 2 - 1
        h = ae.encode_to_prequant(x)
        h = ae.quantize.proj(h)
        return h.movedim(1, 3)

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def decode(h):
        h = ae.quantize.embed(h)
        x = ae.decode(h.movedim(3, 1))
        x = (x + 1) / 2
        return x

    model_raw = Transformer(12, 768, 2048, 64).to(device)
    # with torch.no_grad():
    #     proj = nn.init.orthogonal_(torch.empty(256, 768, device=device), math.sqrt(768 / 256))
    #     model_raw.image_embed.weight.copy_(ae.quantize.embed.weight @ proj)
    #     model_raw.out_proj.weight.copy_(torch.linalg.pinv(model_raw.image_embed.weight.mT))
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    transform = transforms.Compose(
        [
            transforms.Resize(256, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder("/home/kat/datasets/ilsvrc2012/train", transform=transform)
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
    )

    wd, no_wd = [], []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_wd.append(param)
        else:
            wd.append(param)
    groups = [{"params": wd}, {"params": no_wd, "weight_decay": 0}]
    opt = optim.AdamW(groups, lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)
    ema_sched = EMAWarmup(power=2 / 3, max_value=0.999)

    epoch = 0
    step = 0

    @torch.no_grad()
    def sample(model, n, y, tau=1.0, cfg_scale=1.0, disable=False):
        n_proc = n // world_size
        x = torch.zeros(n_proc, 0, dtype=torch.long, device=device)
        y = y.split(n_proc)[rank]
        y_in = torch.cat((y, torch.full_like(y, 1000)))
        cache = model_ema.init_cache(n_proc * 2, 32 * 32, dtype=torch.bfloat16, device=device)
        index = 0
        for _ in trange(32 * 32, disable=rank != 0 or disable):
            x_in = torch.cat((x, x))
            with torch.cuda.amp.autocast(dtype=torch.bfloat16), eval_mode(model):
                logits_c, logits_u = model(x_in, y_in, cache, index).float().chunk(2)
            logits = logits_u + (logits_c - logits_u) * cfg_scale
            sample = sample_categorical(logits, tau=tau)
            x = torch.cat((x, sample), dim=1)
            index += 1
        return torch.cat(dnn.all_gather(x))

    def demo():
        y = torch.randint(1000, (36,), device=device)
        dist.broadcast(y, 0)
        x = sample(model_ema, 36, y, tau=1.0, cfg_scale=2.0)
        x = rearrange(x, "b (h w) -> b h w", h=32, w=32)
        x = decode(x)
        if rank == 0:
            x = rearrange(x, "(nh nw) c h w -> c (nh h) (nw w)", nh=6, nw=6)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.float().cpu()).save(f"demo_imagenet_mos_2_001_{step:07}.png")

    while True:
        sampler.set_epoch(epoch)
        for x, y in tqdm(dataloader, disable=rank > 0):
            if step > 0 and step % 100 == 0:
                demo()
            x, y = x.to(device), y.to(device)
            x_logits = encode(x).flatten(1, 2).float()
            y = torch.where(torch.rand_like(y, dtype=torch.float32) < 0.1, 1000, y)
            x = sample_categorical(x_logits)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x[:, :-1], y).float()
            # loss = F.cross_entropy(logits.mT, x)
            loss = kl_divergence(x_logits, logits).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema_update(model_raw, model_ema, ema_sched.get_value())
            ema_sched.step()
            dist.all_reduce(loss, dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")
            step += 1

        epoch += 1


if __name__ == "__main__":
    main()

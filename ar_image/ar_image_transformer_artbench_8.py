#!/usr/bin/env python3

from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache, reduce, update_wrapper
import math
from pathlib import Path

from einops import rearrange
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
import safetensors.torch
from taming.models import cond_transformer, vqgan
import torch
from torch import distributed as dist, nn, optim
import torch.distributed.nn as dnn
from torch.nn import functional as F
from torch.utils import data
import torch_dist_utils as du
# from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm import trange, tqdm

from grouped_linear import GroupedLinear, group, ungroup

print = tqdm.external_write_mode()(print)
print0 = tqdm.external_write_mode()(du.print0)


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


class FolderOfImages(data.Dataset):
    """Recursively finds all images in a directory. It does not support
    classes/targets."""

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = nn.Identity() if transform is None else transform
        self.paths = sorted(path for path in self.root.rglob('*') if path.suffix.lower() in self.IMG_EXTENSIONS)

    def __repr__(self):
        return f'FolderOfImages(root="{self.root}", len: {len(self)})'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        path = self.paths[key]
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        image = self.transform(image)
        return image,


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


def gumbel_like(x):
    return torch.rand_like(x).log_().nan_to_num_().neg_().log_().neg_()


def sample_categorical(logits, tau=1.0):
    return torch.argmax(logits + gumbel_like(logits) * tau, dim=-1)


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
        self.norm = RMSNorm((dim,))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = zero_init(nn.Linear(dim, dim, bias=False))
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, (head_dim // 8) * (self.n_heads // 2) + 1)[:-1].exp()
        freqs = freqs.view(head_dim // 8, self.n_heads // 2).T.contiguous()
        # TODO: allow changing image size
        pos = make_axial_pos(32, 32)
        # make room for the class token
        # TODO: use adanorm for this
        pos = torch.cat((torch.zeros(1, 2), pos))
        theta_h_1 = pos[:-1, None, 0:1] * freqs
        theta_w_1 = pos[:-1, None, 1:2] * freqs
        theta_h_2 = pos[1:, None, 0:1] * freqs
        theta_w_2 = pos[1:, None, 1:2] * freqs
        theta_1 = torch.cat((theta_h_1, theta_w_1), dim=-1)
        theta_2 = torch.cat((theta_h_2, theta_w_2), dim=-1)
        theta = torch.cat((theta_1, theta_2), dim=-2)
        self.register_buffer("theta", theta)

    def forward(self, x, cache=None, index=None):
        skip = x
        x = self.norm(x)
        qkv = self.qkv_proj(x).view(*x.shape[:2], 3, self.n_heads, self.head_dim)
        q, k, v = qkv.view(*x.shape[:2], 3, self.n_heads, self.head_dim).transpose(1, 3).unbind(2)
        if cache is None:
            theta = self.theta[: x.shape[1]].transpose(0, 1)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        else:
            assert not (x.shape[1] > 1 and index != 0)
            end_index = index + 1
            theta = self.theta[index:end_index].transpose(0, 1)
            q = apply_rotary_emb_(q, theta)
            k = apply_rotary_emb_(k, theta)
            cache[0][:, :, index:end_index] = k
            cache[1][:, :, index:end_index] = v
            k_in = cache[0][:, :, :end_index]
            v_in = cache[1][:, :, :end_index]
            x = F.scaled_dot_product_attention(q, k_in, v_in, is_causal=index == 0)
        x = x.transpose(1, 2).reshape(*skip.shape[:2], -1)
        x = self.out_proj(x)
        return x + skip


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = RMSNorm((dim,))
        self.up = LinearSwiGLU(dim, hidden_dim, bias=False)
        self.down = zero_init(nn.Linear(hidden_dim, dim, bias=False))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up(x)
        x = self.down(x)
        return x + skip


_all_gather = torch.compiler.disable(dnn.all_gather)


def dist2(x, y):
    norm_x = torch.sum(x**2, dim=-1)[..., :, None]
    norm_y = torch.sum(y**2, dim=-1)[..., None, :]
    return norm_x + norm_y - 2 * x @ y.mT


class StochasticKMeans(nn.Module):
    def __init__(self, features, clusters, decay):
        super().__init__()
        self.features = features
        self.clusters = clusters
        self.decay = decay
        self.register_buffer("centroids", torch.randn(clusters, features))

    @torch.no_grad()
    def forward(self, x):
        dists = dist2(x, self.centroids)
        assignments = torch.argmin(dists, dim=-1)
        if self.training:
            new_centroids = torch.index_reduce(self.centroids, 0, assignments.flatten(), x.flatten(0, -2), "mean", include_self=False)
            self.centroids.lerp_(new_centroids, 1 - self.decay)
            dist.all_reduce(self.centroids, dist.ReduceOp.AVG)
        return assignments


class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, n_experts, k):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.norm = RMSNorm((dim,))
        self.router = StochasticKMeans(dim, n_experts, 0.99)
        self.up = GroupedLinear(dim, hidden_dim * 2, n_experts, bias=False)
        self.down = zero_init(GroupedLinear(hidden_dim, dim, n_experts, bias=False))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        ids = self.router(x)
        print0(torch.bincount(ids.flatten(), minlength=self.n_experts))
        # x = x[..., None, :].expand(-1, -1, self.k, -1)
        x, info = group(x, ids, self.n_experts)
        x = self.up(x, info)
        x = swiglu(x)
        x = self.down(x, info)
        x = ungroup(x, info)
        # x = torch.sum(x * c[..., None], dim=-2)
        return x + skip, None


class Block(nn.Module):
    def __init__(self, dim, hidden_dim, head_dim, dropout):
        super().__init__()
        self.attn = SelfAttention(dim, head_dim, dropout)
        # self.ff = FeedForward(dim, hidden_dim)
        self.ff = MoEFeedForward(dim, hidden_dim, 8, 1)
        # self.ff.compile()

    def forward(self, x, cache=None, index=None):
        x = self.attn(x, cache, index)
        # x = self.ff(x)
        x, aux_loss = self.ff(x)
        return x, aux_loss


class MoEHead(nn.Module):
    def __init__(self, dim, hidden_dim, vocab_size, n, k, mode="moc"):
        super().__init__()
        assert mode in ("moc", "mos")
        self.n = n
        self.k = k
        self.mode = mode
        self.router = Router(dim, n, k, act=nn.Identity())
        self.up = GroupedLinear(dim, hidden_dim * 2, n, bias=False)
        self.down = GroupedLinear(hidden_dim, dim, n, bias=False)
        self.proj = GroupedLinear(dim, vocab_size, n, bias=False)

    def embed(self, x):
        return F.embedding(x, self.proj.weight.mean(dim=0))

    def forward(self, x):
        c, ids, aux_loss = self.router(x)
        x = x[..., None, :].expand(-1, -1, self.k, -1)
        x, info = group(x, ids, self.n)
        skip = x
        x = self.up(x, info)
        x = swiglu(x)
        x = self.down(x, info)
        x = x + skip
        x = self.proj(x, info)
        x = ungroup(x, info)
        if self.mode == "moc":
            c = F.softmax(c, dim=-1)
            x = torch.sum(x * c[..., None], dim=-2)
        elif self.mode == "mos":
            c = c - torch.logsumexp(c, dim=-1, keepdim=True)
            x = x - torch.logsumexp(x, dim=-1, keepdim=True)
            x = torch.logsumexp(x + c[..., None], dim=-2)
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return x, aux_loss


class Transformer(nn.Module):
    def __init__(self, depth, dim, hidden_dim, head_dim, dropout):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.class_embed = nn.Embedding(10 + 1, dim)
        self.image_embed = nn.Embedding(16384, dim)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim, dropout) for _ in range(depth)])
        self.out_norm = RMSNorm((dim,))
        self.out_head = nn.Linear(dim, 16384, bias=False)
        self.image_embed.weight = self.out_head.weight
        # self.out_head = MoEHead(dim, hidden_dim, 16384, 4, 4, "mos")
        # self.out_head.compile()

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

    def forward(self, x, y, cache=None, index=None):
        x = self.image_embed(x)
        # x = self.out_head.embed(x)
        y = self.class_embed(y)
        x = torch.cat((y[:, None], x), dim=1)
        x = x[:, -1:].contiguous() if cache and index > 0 else x
        x = self.embed_drop(x)
        cache = [None] * self.depth if cache is None else cache
        aux_losses = []
        for block, cache_block in zip(self.blocks, cache):
            # x, aux_loss = checkpoint(block, x, cache_block, index, enable=self.training)
            x, aux_loss = block(x, cache_block, index)
            aux_losses.append(aux_loss)
        x = self.out_norm(x)
        x = self.out_head(x)
        # x, aux_loss = self.out_head(x)
        # aux_losses.append(aux_loss)
        aux_losses = [l for l in aux_losses if l is not None]
        aux_loss = torch.mean(torch.stack(aux_losses)) if aux_losses and self.training else None
        return x, aux_loss


def apply_top_p(logits, p):
    """Returns logits with tokens not in the top p fraction of probability mass masked out."""
    probs = torch.softmax(logits, dim=-1)
    probs_sorted, indices = torch.sort(probs, dim=-1, descending=True, stable=True)
    probs_cumsum = torch.cumsum(probs_sorted, dim=-1)
    drop = probs_cumsum[..., :-1] >= p
    drop = torch.cat((drop.new_zeros(*drop.shape[:-1], 1), drop), dim=-1)
    drop_unsorted = torch.empty_like(drop).scatter_(-1, indices, drop)
    return torch.masked_fill(logits, drop_unsorted, float("-inf"))


def main():
    batch_size = 32

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    base_path = "/home/kat/text-to-image/latent-diffusion"
    # base_path = "/weka2/kat/ar_image/latent-diffusion"
    config_path = f"{base_path}/models/first_stage_models/vq-f16/config.yaml"
    model_path = f"{base_path}/models/first_stage_models/vq-f16/model.ckpt"
    ae = load_vqgan_model(config_path, model_path).to(device)
    ae.eval().requires_grad_(False)

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def encode(x):
        bs = x.shape[0]
        x = x * 2 - 1
        x, _, _ = ae.encode(x)
        _, _, (_, _, x) = ae.quantize(x)
        x = x.view(bs, -1)
        return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def decode(x):
        x = ae.quantize.embedding(x)
        x = x.movedim(3, 1)
        x = ae.decode(x)
        return (x + 1) / 2

    model_raw = Transformer(8, 512, 1360, 64, dropout=0.0).to(device)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    data_tensors = safetensors.torch.load_file("artbench_512_f16.safetensors")
    dataset = data.TensorDataset(data_tensors["tokens"], data_tensors["classes"])
    sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
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
    opt = optim.AdamW(groups, lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    sched = optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, i / 250))
    ema_sched = EMAWarmup(power=2 / 3, max_value=0.999)

    epoch = 0
    step = 0

    @torch.no_grad()
    def sample(model, y, tau=1.0, top_p=1.0, disable=False):
        n = y.shape[0]
        n_proc = n // world_size
        x = torch.zeros(n_proc, 0, dtype=torch.long, device=device)
        y = y.split(n_proc)[rank]
        cache = model.init_cache(n_proc, 32 * 32, dtype=torch.bfloat16, device=device)
        index = 0
        entr_x = torch.zeros(n_proc, 0, dtype=torch.float32, device=device)
        for _ in trange(32 * 32, disable=rank != 0 or disable):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16), eval_mode(model):
                logits, _ = model(x, y, cache, index)
            logits = logits.float()
            logprobs = F.log_softmax(logits, dim=-1)
            entr_tok = -torch.sum(logprobs.exp() * logprobs, dim=-1)
            entr_x = torch.cat((entr_x, entr_tok), dim=1)
            if top_p < 1.0:
                logits = apply_top_p(logits, top_p)
            sample = sample_categorical(logits, tau=tau)
            x = torch.cat((x, sample), dim=1)
            index += 1
        x = torch.cat(dnn.all_gather(x))
        entr_x = torch.cat(dnn.all_gather(entr_x))
        return x, entr_x

    def demo():
        y = torch.randint(10, (16,), device=device)
        dist.broadcast(y, 0)
        x, entr = sample(model_ema, y, tau=1.0, top_p=1.0)
        print0(f"Mean policy entropy: {entr.mean().item():g}")
        x = rearrange(x, "b (h w) -> b h w", h=32, w=32)
        x = decode(x)
        if rank == 0:
            x = rearrange(x, "(nh nw) c h w -> c (nh h) (nw w)", nh=4, nw=4)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu().float()).save(f"demo_artbench_162_{step:07}.png")

    while True:
        sampler.set_epoch(epoch)
        for x, y in tqdm(dataloader, disable=rank > 0):
            if step > 0 and step % 500 == 0:
                demo()
            x = x.long().to(device)
            y = y.long().to(device)
            y_drop = torch.where(torch.rand_like(y, dtype=torch.float32) < 0.1, 10, y)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, aux_loss = model(x[:, :-1], y_drop)
                xent_loss = torch.compile(F.cross_entropy)(logits.mT, x)
            loss = xent_loss  # + 0.02 * aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            ema_update(model_raw, model_ema, ema_sched.get_value())
            ema_sched.step()
            dist.all_reduce_coalesced([loss], dist.ReduceOp.AVG)
            print0(f"epoch: {epoch}, step: {step}, loss: {loss.item():g}")
            step += 1

        epoch += 1


if __name__ == "__main__":
    main()

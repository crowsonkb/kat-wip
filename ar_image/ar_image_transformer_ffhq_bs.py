#!/usr/bin/env python3

import argparse
from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path

from einops import rearrange
from ldm.util import instantiate_from_config
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import safetensors.torch
from taming.models import cond_transformer, vqgan
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


def sm_make_input(a, bs_tok):
    l = a.shape[0]
    x = np.copy(a)
    y = np.zeros((l,), np.int64)
    t = np.ones((l,), np.bool_)
    m = np.zeros((l, l), np.bool_)
    p = np.zeros((l,), np.int64)

    c = 0
    d = 0

    for i in range(0, l):
        m[i] = m[max(i - 1, 0)]
        if i > 0:
            y[i - 1] = a[i]
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

    t[-1] = 0
    return x, y, t, m, p


def sm_loss(s_d, y_d, t_d, p_d, s_m, y_m, t_m, p_m, gamma, alpha=1.0, chi2_mix_fac=0.0):
    def phi(x):
        return x - (1 - chi2_mix_fac) * alpha * x**2 / 4

    def psi(x):
        return -chi2_mix_fac * alpha * x**2 / 4

    eos_d = torch.argmax(t_d * p_d, dim=1)
    eos_m = torch.argmax(t_m * p_m, dim=1)
    gamma_d = t_d * gamma**p_d
    gamma_m = t_m * gamma**p_m
    gamma_eos_d = gamma_d.gather(1, eos_d[:, None])[:, 0]
    gamma_eos_m = gamma_m.gather(1, eos_m[:, None])[:, 0]
    q_d = s_d.gather(2, y_d[:, :, None])[:, :, 0]
    q_m = s_m.gather(2, y_m[:, :, None])[:, :, 0]
    v_d = torch.logsumexp(s_d, dim=2)
    v_m = torch.logsumexp(s_m, dim=2)
    v_eos_d = v_d.gather(1, eos_d[:, None])[:, 0]
    v_eos_m = v_m.gather(1, eos_m[:, None])[:, 0]

    qv_diff_d = gamma_d[:, :-1] * phi(q_d[:, :-1] - gamma * v_d[:, 1:])
    qv_diff_d_eos = gamma_eos_d * phi((1 - gamma) * v_eos_d) / (1 - gamma)
    qv_diff_m = gamma_m[:, :-1] * psi(q_m[:, :-1] - gamma * v_m[:, 1:])
    qv_diff_m_eos = gamma_eos_m * psi((1 - gamma) * v_eos_m) / (1 - gamma)
    v_diff_d = gamma_d[:, :-1] * (v_d[:, :-1] - gamma * v_d[:, 1:]) / 2
    v_diff_d_eos = gamma_eos_d * v_eos_d / 2
    v_diff_m = gamma_m[:, :-1] * (v_m[:, :-1] - gamma * v_m[:, 1:]) / 2
    v_diff_m_eos = gamma_eos_m * v_eos_m / 2
    loss_d = v_diff_d.sum(1) + v_diff_d_eos - qv_diff_d.sum(1) - qv_diff_d_eos
    loss_m = v_diff_m.sum(1) + v_diff_m_eos - qv_diff_m.sum(1) - qv_diff_m_eos

    return loss_d.mean() + loss_m.mean()


def simple_sm_loss(logits, targets, gamma, alpha=1.0, mask=None, pos=None):
    def phi(x):
        return x - alpha * x**2 / 4

    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    if pos is None:
        pos = torch.cumsum(mask, dim=-1)

    eos = torch.argmax(pos * mask, dim=-1)
    gamma_i = mask * gamma**pos
    gamma_i_eos = gamma_i.gather(-1, eos[..., None])[..., 0]
    q = logits.gather(-1, targets[..., None])[..., 0]
    v = torch.logsumexp(logits, dim=-1)
    v_eos = v.gather(-1, eos[..., None])[..., 0]

    qv_diff = gamma_i[..., :-1] * phi(q[..., :-1] - gamma * v[..., 1:])
    qv_diff_eos = gamma_i_eos * phi((1 - gamma) * v_eos) / (1 - gamma)
    v_diff = gamma_i[..., :-1] * (v[..., :-1] - gamma * v[..., 1:])
    v_diff_eos = gamma_i_eos * v_eos
    loss = v_diff.sum(-1) + v_diff_eos - qv_diff.sum(-1) - qv_diff_eos

    return torch.mean(loss)


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


@torch.compile
def mixture_of_softmax(x, w):
    probs = torch.exp(x - torch.logsumexp(x, dim=-1, keepdim=True))
    weights = torch.exp(w - torch.logsumexp(w, dim=-1, keepdim=True))
    return torch.log(torch.sum(probs * weights[..., None], dim=-2))


class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = dim // head_dim
        self.norm = nn.LayerNorm((dim,))
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = zero_init(nn.Linear(dim, dim))
        self.out_drop = nn.Dropout(0.0)
        log_min = math.log(math.pi)
        log_max = math.log(10 * math.pi)
        freqs = torch.linspace(log_min, log_max, self.n_heads * head_dim // 4 + 1)[:-1].exp()
        freqs = freqs.view(head_dim // 4, self.n_heads).T.contiguous()
        # TODO: allow changing image size
        pos = make_axial_pos(32, 32)
        # make room for the class token and the last backspace token
        pos = torch.cat((torch.zeros(1, 2), pos, pos[-32:] + (pos[-32:] - pos[-64:-32])))
        theta_h = pos[..., None, 0:1] * freqs
        theta_w = pos[..., None, 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("theta", theta)

    def forward(self, x, m, p, cache=None, index=None):
        x = self.norm(x)
        # q, k, v = rearrange(self.qkv_proj(x), "n s (t h d) -> t n h s d", t=3, h=self.n_heads)
        qkv = self.qkv_proj(x).reshape(x.shape[0], -1, 3, self.n_heads, self.head_dim)  # n s t h d
        # transpose to n h t s d
        qkv = qkv.transpose(1, 3)
        q, k, v = qkv.unbind(2)
        if index is None:
            q = apply_rotary_emb(q, self.theta[p].transpose(1, 2).to(q))
            k = apply_rotary_emb(k, self.theta[p].transpose(1, 2).to(k))
            x = F.scaled_dot_product_attention(q, k, v, m[:, None], 0.0 if self.training else 0)
            if cache is not None:
                cache[0][:] = k
                cache[1][:] = v
        else:
            b_idx = torch.arange(x.shape[0], device=x.device)
            max_index = torch.amax(index) + 1
            m = m[b_idx, index, :max_index][:, None, None]
            q = apply_rotary_emb(q, self.theta[p[b_idx, index]][:, None].transpose(1, 2).to(q))
            k = apply_rotary_emb(k, self.theta[p[b_idx, index]][:, None].transpose(1, 2).to(k))
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
        self.drop_1 = nn.Dropout(0.0)
        self.down = zero_init(nn.Linear(hidden_dim, dim))
        self.drop_2 = nn.Dropout(0.0)

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
        self.n_states = 1
        self.n_softmax = 1
        self.class_embed = nn.Embedding(1, dim)
        self.image_embed = nn.Embedding(16384 + 1, dim)
        self.embed_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim * self.n_states,))
        # self.out_weight = nn.Linear(dim * self.n_states, self.n_softmax)
        # nn.init.zeros_(self.out_weight.weight)
        self.out_proj = nn.Linear(dim * self.n_states, (16384 + 1) * self.n_softmax)

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
        y = self.class_embed(x[:, :1])
        x = self.image_embed(x[:, 1:])
        x = torch.cat((y, x), dim=1)
        b_idx = torch.arange(x.shape[0], device=x.device)
        cache = [None] * self.depth if cache is None else cache
        x = x if index is None else x[b_idx, index][:, None]
        x = self.embed_drop(x)
        xs = []
        for block, cache_block in zip(self.blocks, cache):
            # x = checkpoint(block, x, m, p, cache_block, index)
            x = block(x, m, p, cache_block, index)
            xs.append(x)
        x = torch.cat(xs[-self.n_states:], dim=-1)
        x = self.out_norm(x)
        # w = self.out_weight(x)
        x = self.out_proj(x)
        # x = x.view(x.shape[0], x.shape[1], self.n_softmax, 16384 + 1)
        # x = mixture_of_softmax(x, w)
        return x


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


def noise_seq(seq, n_toks, bs_tok, n_noise):
    noise_indices = np.random.randint(1, len(seq) - 1, (n_noise,))
    for i in range(n_noise):
        noise_index = noise_indices[i]
        noise_token = np.random.randint(0, n_toks)
        bs_seq = np.stack((noise_token, np.array(bs_tok)))
        seq = np.insert(seq, noise_index, bs_seq)
        noise_indices = np.where(noise_indices >= noise_index, noise_indices + 2, noise_indices)
    return sm_make_input(seq, bs_tok)


class TensorDatasetWithTransform(data.Dataset):
    def __init__(self, tensor, transform=None):
        super().__init__()
        self.tensor = tensor
        self.transform = nn.Identity() if transform is None else transform

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, key):
        return self.transform(self.tensor[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="run")
    args = parser.parse_args()

    batch_size = 32

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    base_path = "/weka/kat/ar_image/latent-diffusion/models/first_stage_models/vq-f16"
    ae = load_vqgan_model(f"{base_path}/config.yaml", f"{base_path}/model.ckpt").to(device)
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

    model_raw = Transformer(8, 512, 1360, 64).to(device)
    # proj = nn.init.orthogonal_(torch.empty(8, 768, device=device), math.sqrt(768 / 8))
    # with torch.no_grad():
    #     model_raw.image_embed.weight[:-1].copy_(ae.quantize.embedding.weight @ proj)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    transform = transforms.Compose(
        [
            transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )

    def transform(seq):
        cls_token = torch.zeros(1, dtype=torch.long)
        seq = torch.cat((cls_token, seq), dim=0)
        return noise_seq(seq, 16384, 16384, 51)

    # dataset = FolderOfImages("/home/kat/datasets/ffhq/images1024x1024", transform=transform)
    data_tensor = safetensors.torch.load_file("ffhq_512_f16.safetensors")["tokens"]
    dataset = TensorDatasetWithTransform(data_tensor, transform=transform)
    sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    dataloader = data.DataLoader(
        dataset,
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
    opt = optim.AdamW(groups, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)
    ema_sched = EMAWarmup(power=2 / 3, max_value=0.999)

    epoch = 0
    step = 0

    @torch.no_grad()
    def sample(model, cls, disable=False):
        image_toks = 32 * 32
        cache = model_ema.init_cache(cls.shape[0], image_toks + 2, dtype=torch.bfloat16, device=device)
        x = torch.full((cls.shape[0], image_toks + 2), 0, dtype=torch.int64, device=device)
        x[:, 0] = cls
        index = torch.zeros(x.shape[0], dtype=torch.int64, device=device)
        b_idx = torch.arange(x.shape[0], device=device)
        m = torch.ones(
            x.shape[0], x.shape[1], x.shape[1], dtype=torch.bool, device=device
        ).tril_()
        p = torch.arange(x.shape[1], device=device).tile((x.shape[0], 1))
        i = 0
        with tqdm(total=1536, disable=disable) as pbar:
            while torch.amin(index) < image_toks:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(x, m, p, cache, index)[:, -1].float()
                gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
                sample = torch.argmax(logits + gumbel, dim=-1)
                good_toks = (index == 0) | (sample != 16384)
                index = torch.where(good_toks, index + 1, index - 1).clamp_max_(image_toks + 1)
                x[b_idx, index] = torch.where(good_toks, sample, x[b_idx, index])
                pbar.update(1)
                i += 1
                if i > 1536:
                    break
        return x[:, :image_toks + 1]

    def demo():
        cls = torch.randint(1, (16,), device=device)
        # dist.broadcast(cls, 0)
        x = sample(model_ema, cls)[:, 1:].clamp_max_(16383)
        x = rearrange(x, "b (h w) -> b h w", h=32, w=32)
        x = decode(x)
        if rank == 0:
            x = rearrange(x, "(nh nw) c h w -> c (nh h) (nw w)", nh=4, nw=4)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.float().cpu()).save(f"demo_{args.name}_{step:07}.png")

    replay_buffer_list = []

    while True:
        sampler.set_epoch(epoch)

        for (x, y, t, m, p) in tqdm(dataloader, disable=rank > 0):
            if rank == 0 and step % 500 == 0:
                demo()
            dist.barrier()

            x, y, t, m, p = x.to(device), y.to(device), t.to(device), m.to(device), p.to(device)

            # batch = batch.to(device).long()
            # batch = encode(batch)
            # cls = torch.full((batch.shape[0], 1), 0, dtype=torch.long, device=device)
            # batch = torch.cat((cls, batch), dim=1)

            # batch_in = batch.cpu().numpy()
            # outs = [noise_seq(i, 16384, 16384, 51) for i in batch_in]
            # x = torch.stack([torch.from_numpy(i[0]) for i in outs]).to(device)
            # y = torch.stack([torch.from_numpy(i[1]) for i in outs]).to(device)
            # t = torch.stack([torch.from_numpy(i[2]) for i in outs]).to(device)
            # m = torch.stack([torch.from_numpy(i[3]) for i in outs]).to(device)
            # p = torch.stack([torch.from_numpy(i[4]) for i in outs]).to(device)

            if False:
                if step % 25 == 0:
                    replay_buffer_list.append(sample(model_ema, x[:, 0], disable=True))
                    if len(replay_buffer_list) > 10:
                        replay_buffer_list.pop(0)
                    model_x_all = torch.cat(replay_buffer_list)
                model_n = 8
                indices = torch.randperm(model_x_all.shape[0], device=device)[:model_n]
                model_x = model_x_all[indices, :-1]
                model_y = model_x_all[indices, 1:]
                model_m = torch.ones(
                    model_n, model_x.shape[1], model_x.shape[1], dtype=torch.bool, device=device
                ).tril_()
                model_t = torch.ones(model_n, model_x.shape[1], dtype=torch.bool, device=device)
                model_p = torch.arange(model_x.shape[1], device=device).tile(model_n, 1) + 1
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits_for_noise = model_ema(x, m, p - 1).float()
            # logits_for_noise.scatter_(2, y[:, :, None], float("-inf"))
            logits_for_noise[:, :, -1] = float("-inf")
            gumbel = torch.rand_like(logits_for_noise).log_().nan_to_num_().neg_().log_().neg_()
            samples = torch.argmax(logits_for_noise + gumbel, dim=-1)
            x[:, 1:] = torch.where(t[:, :-1], x[:, 1:], samples[:, :-1])
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, m, p - 1).float()
                # model_logits = model(model_x, model_m, model_p - 1).float()
            # gamma = 1024 / (1024 + 1)
            # gamma_p = t * gamma**p
            loss = torch.sum(t * F.cross_entropy(logits.mT, y, reduction="none")) / torch.sum(t)
            # loss = sm_loss(logits, y, t, p, model_logits, model_y, model_t, model_p, gamma=gamma, alpha=0.8, chi2_mix_fac=1e-3)
            # loss = simple_sm_loss(logits, y, gamma, mask=t, pos=p)
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

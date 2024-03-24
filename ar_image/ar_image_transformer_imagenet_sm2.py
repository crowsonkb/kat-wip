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

    return x, y, t, m, p


def sm_loss(s_d, y_d, t_d, p_d, s_m, y_m, t_m, p_m, gamma, kl_fac=0.0, kl_reg=0.01, chi2_mix_reg=0.0):
    def phi(x):
        kl = -torch.expm1(-kl_reg * x) / kl_reg
        chi2 = x - x**2 / 4
        return kl_fac * kl + (1 - kl_fac) * chi2

    def psi(x):
        chi2 = -x**2 / 4
        return (1 - kl_fac) * chi2_mix_reg * chi2

    gamma_d = t_d * gamma**p_d
    gamma_m = t_m * gamma**p_m
    q_d = s_d.gather(2, y_d[:, :, None])[:, :, 0]
    q_m = s_m.gather(2, y_m[:, :, None])[:, :, 0]
    v_d = torch.logsumexp(s_d, dim=2)
    v_m = torch.logsumexp(s_m, dim=2)

    qv_diff_d = gamma_d[:, :-1] * phi(q_d[:, :-1] - gamma * v_d[:, 1:])
    qv_diff_d_eos = gamma_d[:, -1] * phi((1 - gamma) * v_d[:, -1]) / (1 - gamma)
    qv_diff_m = gamma_m[:, :-1] * psi(q_m[:, :-1] - gamma * v_m[:, 1:])
    qv_diff_m_eos = gamma_m[:, -1] * psi((1 - gamma) * v_m[:, -1]) / (1 - gamma)
    v_diff_d = gamma_d[:, :-1] * (v_d[:, :-1] - gamma * v_d[:, 1:]) / 2
    v_diff_d_eos = gamma_d[:, -1] * v_d[:, -1] / 2
    v_diff_m = gamma_m[:, :-1] * (v_m[:, :-1] - gamma * v_m[:, 1:]) / 2
    v_diff_m_eos = gamma_m[:, -1] * v_m[:, -1] / 2
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

    gamma_i = mask * gamma**pos
    q = logits.gather(-1, targets[..., None])[..., 0]
    v = torch.logsumexp(logits, dim=-1)
    eos = torch.argmax(pos, dim=-1)
    gamma_i_eos = gamma_i.gather(-1, eos[..., None])[..., 0]
    v_eos = v.gather(-1, eos[..., None])[..., 0]

    qv_diff = gamma_i[..., :-1] * phi(q[..., :-1] - gamma * v[..., 1:])
    qv_diff_eos = gamma_i_eos * phi((1 - gamma) * v_eos) / (1 - gamma)
    v_diff = gamma_i[..., :-1] * (v[..., :-1] - gamma * v[..., 1:])
    v_diff_eos = gamma_i_eos * v_eos
    loss = v_diff.sum(-1) + v_diff_eos - qv_diff.sum(-1) - qv_diff_eos

    return torch.sum(loss) / torch.sum(gamma_i)


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
        pos = make_axial_pos(24, 24)
        # make room for the class token
        pos = torch.cat((torch.zeros(1, 2), pos))
        theta_h = pos[..., 0:1] * freqs
        theta_w = pos[..., 1:2] * freqs
        theta = torch.cat((theta_h, theta_w), dim=-1)
        self.register_buffer("theta", theta)

    def forward(self, x, m, p, cache=None, index=None):
        x = self.norm(x)
        q, k, v = rearrange(self.qkv_proj(x), "n s (t h d) -> t n h s d", t=3, h=self.n_heads)
        if cache is None:
            q = apply_rotary_emb(q, self.theta[p][:, None].to(q))
            k = apply_rotary_emb(k, self.theta[p][:, None].to(k))
            x = F.scaled_dot_product_attention(q, k, v, m[:, None], 0.1 if self.training else 0)
        else:
            b_idx = torch.arange(x.shape[0], device=x.device)
            m = m[b_idx, index][:, None, None]
            q = apply_rotary_emb(q, self.theta[p[b_idx, index]][:, None, None].to(q))
            k = apply_rotary_emb(k, self.theta[p[b_idx, index]][:, None, None].to(k))
            cache[0][b_idx, :, index] = k[:, :, 0, :]
            cache[1][b_idx, :, index] = v[:, :, 0, :]
            x = F.scaled_dot_product_attention(q, cache[0], cache[1], m)
        x = rearrange(x, "n h s d -> n s (h d)")
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
        self.class_embed = nn.Embedding(1001, dim)
        self.image_embed = nn.Embedding(16384 + 1, dim)
        self.embed_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(dim, hidden_dim, head_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm((dim,))
        self.out_proj = zero_init(nn.Linear(dim, 16384 + 1))

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
        for block, cache_block in zip(self.blocks, cache):
            x = checkpoint(block, x, m, p, cache_block, index)
        x = self.out_norm(x)
        x = self.out_proj(x)
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
    for _ in range(n_noise):
        noise_index = np.random.randint(1, len(seq) - 2)
        noise_token = np.random.randint(n_toks)
        bs_seq = np.stack((noise_token, np.array(bs_tok)))
        seq = np.insert(seq, noise_index, bs_seq)
    return sm_make_input(seq, bs_tok)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="run")
    args = parser.parse_args()

    batch_size = 24

    du.init_distributed()
    device = du.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    config_path = "/home/kat/text-to-image/latent-diffusion/models/first_stage_models/vq-f16/config.yaml"
    model_path = "/home/kat/text-to-image/latent-diffusion/models/first_stage_models/vq-f16/model.ckpt"
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

    model_raw = Transformer(12, 768, 2048, 64).to(device)
    proj = nn.init.orthogonal_(torch.empty(8, 768, device=device))
    with torch.no_grad():
        model_raw.image_embed.weight[:-1].copy_(ae.quantize.embedding.weight @ proj)
    du.broadcast_tensors(model_raw.parameters())
    model_ema = deepcopy(model_raw).eval().requires_grad_(False)
    print0(f"Parameters: {sum(p.numel() for p in model_raw.parameters()):,}")
    model = nn.parallel.DistributedDataParallel(
        model_raw, device_ids=[device], output_device=device
    )

    transform = transforms.Compose(
        [
            transforms.Resize(384, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(384),
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
        num_workers=8,
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
    opt = optim.AdamW(groups, lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)
    ema_sched = EMAWarmup(power=2 / 3, max_value=0.999)

    epoch = 0
    step = 0

    @torch.no_grad()
    def sample(model, cls, cfg_scale=1.0, disable=False):
        n = cls.shape[0]
        image_toks = 24 * 24
        cache = model_ema.init_cache(n * 2, image_toks + 1, dtype=torch.bfloat16, device=device)
        x = torch.full((n, image_toks + 2), 0, dtype=torch.int64, device=device)
        x[:, 0] = cls
        index = torch.zeros(n, dtype=torch.int64, device=device)
        b_idx = torch.arange(n, device=device)
        m = torch.ones(
            x.shape[0] * 2, x.shape[1] - 1, x.shape[1] - 1, dtype=torch.bool, device=device
        ).tril_()
        p = torch.arange(x.shape[1] - 1, device=device)[None]
        i = 0
        with tqdm(total=768, disable=disable) as pbar:
            while torch.amin(index) < image_toks:
                x_in = torch.cat((x, x))
                x_in[n:, 0] = 1000
                index_in = torch.cat((index, index))
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits_c, logits_u = model(x_in, m, p, cache, index_in)[:, -1].float().chunk(2)
                    logits = logits_u + (logits_c - logits_u) * cfg_scale
                gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
                sample = torch.argmax(logits + gumbel, dim=-1)
                good_toks = (index == 0) | (sample != 16384) & ~(index == image_toks)
                index = torch.where(good_toks, index + 1, index - 1).clamp_max_(image_toks)
                x[b_idx, index] = torch.where(good_toks, sample, x[b_idx, index])
                pbar.update(1)
                i += 1
                if i > 768:
                    break
        return x[:, :image_toks + 1]

    @torch.no_grad()
    def sample_for_training(model, cls, disable=False):
        image_toks = 24 * 24
        cache = model_ema.init_cache(cls.shape[0], image_toks + 1, dtype=torch.bfloat16, device=device)
        x = torch.full((cls.shape[0], image_toks + 2), 0, dtype=torch.int64, device=device)
        x[:, 0] = cls
        index = torch.zeros(x.shape[0], dtype=torch.int64, device=device)
        b_idx = torch.arange(x.shape[0], device=device)
        m = torch.ones(
            x.shape[0], x.shape[1] - 1, x.shape[1] - 1, dtype=torch.bool, device=device
        ).tril_()
        p = torch.arange(x.shape[1] - 1, device=device)[None]
        i = 0
        with tqdm(total=768, disable=disable) as pbar:
            while torch.amin(index) < image_toks:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(x, m, p, cache, index)[:, -1].float()
                gumbel = torch.rand_like(logits).log_().nan_to_num_().neg_().log_().neg_()
                sample = torch.argmax(logits + gumbel, dim=-1)
                good_toks = (index == 0) | (sample != 16384) & ~(index == image_toks)
                index = torch.where(good_toks, index + 1, index - 1).clamp_max_(image_toks)
                x[b_idx, index] = torch.where(good_toks, sample, x[b_idx, index])
                pbar.update(1)
                i += 1
                if i > 768:
                    break
        return x[:, :image_toks + 1]

    def demo():
        cls = torch.randint(1000, (16,), device=device)
        # dist.broadcast(cls, 0)
        x = sample(model_ema, cls, cfg_scale=1.0)[:, 1:]
        x = rearrange(x, "b (h w) -> b h w", h=24, w=24)
        x = decode(x)
        if rank == 0:
            x = rearrange(x, "(nh nw) c h w -> c (nh h) (nw w)", nh=4, nw=4)
            x = torch.clamp(x, 0, 1)
            TF.to_pil_image(x.cpu()).save(f"demo_{args.name}_{step:07}.png")

    replay_buffer_list = []

    while True:
        sampler.set_epoch(epoch)

        for batch, cls in tqdm(dataloader, disable=rank > 0):
            if step % 100 == 0:
                if rank == 0:
                    demo()
                dist.barrier()

            batch = batch.to(device)
            cls = cls.to(device)
            batch = encode(batch)
            cls_drop = torch.rand(cls.shape[0], device=device) < 0.1
            cls = torch.where(cls_drop, 1000, cls)
            batch = torch.cat((cls[:, None], batch), dim=1)

            batch_in = batch.cpu().numpy()
            outs = [noise_seq(i, 16384, 16384, 10) for i in batch_in]
            x = torch.stack([torch.from_numpy(i[0]) for i in outs]).to(device)
            y = torch.stack([torch.from_numpy(i[1]) for i in outs]).to(device)
            t = torch.stack([torch.from_numpy(i[2]) for i in outs]).to(device)
            m = torch.stack([torch.from_numpy(i[3]) for i in outs]).to(device)
            p = torch.stack([torch.from_numpy(i[4]) for i in outs]).to(device)

            if False:
                if step % 25 == 0:
                    cls = torch.randint(1000, (256,), device=device)
                    cls_drop = torch.rand(cls.shape[0], device=device) < 0.1
                    cls = torch.where(cls_drop, 1000, cls)
                    replay_buffer_list.append(sample_for_training(model_ema, cls, disable=rank != 0))
                    if len(replay_buffer_list) > 8:
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
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(x, m, p - 1).float()
                # model_logits = model(model_x, model_m, model_p - 1).float()
            # y.masked_fill_(~t, -100)
            # loss = F.cross_entropy(logits.mT, y)
            gamma = 576 / (576 + 1)
            # loss = sm_loss(logits, y, t, p, model_logits, model_y, model_t, model_p, gamma, kl_fac=0.2, chi2_mix_reg=0.001) / 576
            loss = simple_sm_loss(logits, y, gamma, alpha=1, mask=t, pos=p)
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

#!/usr/bin/env python3

from pathlib import Path

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
import safetensors.torch
from taming.models import cond_transformer, vqgan
import torch
from torch import distributed as dist
import torch.distributed.nn as dnn
from torch.utils import data
import torch_dist_utils as du
from torchvision import datasets, transforms
from tqdm import trange, tqdm

from artbench_lsun import ArtBench10

print = tqdm.external_write_mode()(print)
print0 = tqdm.external_write_mode()(du.print0)


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


def main():
    batch_size = 50

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

    transform = transforms.Compose(
        [
            transforms.Resize(512, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )
    dataset = ArtBench10("/home/kat/datasets/artbench-10", transform=transform)
    sampler = data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    tokens, classes = [], []

    for x, y in tqdm(dataloader, disable=rank > 0):
        x = x.to(device)
        y = y.to(device)
        x = encode(x)
        xs = dnn.gather(x)
        ys = dnn.gather(y)
        if rank == 0:
            tokens.append(torch.cat(xs).cpu())
            classes.append(torch.cat(ys).cpu())

    if rank == 0:
        obj = {"tokens": torch.cat(tokens), "classes": torch.cat(classes)}
        safetensors.torch.save_file(obj, "artbench_512_f16.safetensors")
    dist.barrier()


if __name__ == "__main__":
    main()


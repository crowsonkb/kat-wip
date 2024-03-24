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

print = tqdm.external_write_mode()(print)
print0 = tqdm.external_write_mode()(du.print0)


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
    dataset = FolderOfImages("/home/kat/datasets/ffhq/images1024x1024", transform=transform)
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

    encoded = []

    for x, in tqdm(dataloader, disable=rank > 0):
        x = x.to(device)
        x = encode(x)
        xs = dnn.gather(x)
        if rank == 0:
            encoded.append(torch.cat(xs).cpu())

    if rank == 0:
        obj = {"tokens": torch.cat(encoded)}
        safetensors.torch.save_file(obj, "ffhq_512_f16.safetensors")
    dist.barrier()


if __name__ == "__main__":
    main()


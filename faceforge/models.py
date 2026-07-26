"""DCGAN generator and discriminator used by the Colab notebook and web app."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def weights_init(module: nn.Module) -> None:
    """DCGAN initialization recommended for stable early training."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class Generator(nn.Module):
    """Transforms a latent vector into a normalized 64x64 RGB synthetic face."""

    def __init__(self, latent_dim: int = 128, feature_maps: int = 64, channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.channels = channels
        f = feature_maps
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, f * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(True),
            nn.ConvTranspose2d(f, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 4 or noise.shape[1:] != (self.latent_dim, 1, 1):
            raise ValueError(
                f"Expected latent noise with shape [N, {self.latent_dim}, 1, 1], got {tuple(noise.shape)}."
            )
        return self.net(noise)


class Discriminator(nn.Module):
    """Classifies normalized 64x64 RGB images as real or generated."""

    def __init__(self, feature_maps: int = 64, channels: int = 3) -> None:
        super().__init__()
        f = feature_maps
        self.net = nn.Sequential(
            nn.Conv2d(channels, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f, f * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 4, f * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
            raise ValueError(f"Expected images with shape [N, 3, 64, 64], got {tuple(images.shape)}.")
        return self.net(images).flatten()


def model_summary(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, object]:
    """Return a compact, notebook-friendly model summary without extra packages."""
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    with torch.no_grad():
        output_shape = tuple(model(torch.zeros(input_shape)).shape)
    return {
        "class": model.__class__.__name__,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "parameters": parameters,
        "trainable_parameters": trainable,
    }

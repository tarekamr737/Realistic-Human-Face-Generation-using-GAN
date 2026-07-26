"""Safe, deterministic generator loading and image serialization for the API."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from .config import Settings
from .models import Generator


@dataclass
class LoadedGenerator:
    model: Generator
    device: torch.device
    metadata: dict[str, Any]


def _read_metadata(path: Path, settings: Settings) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "model_name": "FFHQ DCGAN-64",
        "architecture": "DCGAN",
        "image_size": settings.image_size,
        "latent_dim": settings.latent_dim,
        "feature_maps": settings.feature_maps,
        "checkpoint": settings.checkpoint_name,
        "inference_only": True,
    }
    if not path.exists():
        return defaults
    with path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    if not isinstance(saved, dict):
        raise ValueError("model_metadata.json must contain a JSON object.")
    return {**defaults, **saved}


def load_generator(settings: Settings, device_name: str | None = None) -> LoadedGenerator:
    """Load the checkpoint exported by the Colab notebook, always in eval mode."""
    if not settings.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint is unavailable at {settings.checkpoint_path}. Copy generator_best.pt from Colab into models/."
        )
    metadata = _read_metadata(settings.metadata_path, settings)
    if metadata.get("architecture") != "DCGAN":
        raise ValueError("This app supports the DCGAN checkpoint format exported by the included Colab notebook.")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = Generator(
        latent_dim=int(metadata["latent_dim"]),
        feature_maps=int(metadata.get("feature_maps", settings.feature_maps)),
    ).to(device)
    checkpoint = torch.load(settings.checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Checkpoint must be a PyTorch state dict or export dictionary.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return LoadedGenerator(model=model, device=device, metadata=metadata)


def generate_batch(
    loaded: LoadedGenerator,
    count: int,
    seed: int,
    truncation: float = 1.0,
) -> tuple[torch.Tensor, list[int]]:
    """Generate a reproducible collection. Consecutive seeds make each tile addressable."""
    if count not in (1, 4, 8, 16):
        raise ValueError("Face count must be 1, 4, 8, or 16.")
    if not 0.25 <= truncation <= 2.0:
        raise ValueError("Truncation must be between 0.25 and 2.0.")
    latent_dim = int(loaded.metadata["latent_dim"])
    seeds = [seed + index for index in range(count)]
    noises: list[torch.Tensor] = []
    for item_seed in seeds:
        generator = torch.Generator(device=loaded.device)
        generator.manual_seed(item_seed)
        noise = torch.randn((1, latent_dim, 1, 1), generator=generator, device=loaded.device)
        noises.append(noise.clamp(-truncation, truncation))
    with torch.inference_mode():
        images = loaded.model(torch.cat(noises, dim=0)).detach().cpu()
    return images, seeds


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert model output from [-1, 1] CHW into an RGB Pillow image."""
    pixels = tensor.detach().clamp(-1, 1).add(1).div(2).mul(255).byte()
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_grid(images: torch.Tensor, labels: list[int]) -> Image.Image:
    """Create a labeled PNG contact sheet that retains synthetic provenance context."""
    tiles = [tensor_to_image(image) for image in images]
    tile_size = tiles[0].width
    columns = 1 if len(tiles) == 1 else 2 if len(tiles) <= 4 else 4
    rows = (len(tiles) + columns - 1) // columns
    padding, label_height = 10, 24
    width = columns * tile_size + (columns + 1) * padding
    height = rows * (tile_size + label_height) + (rows + 1) * padding
    grid = Image.new("RGB", (width, height), "#111117")
    draw = ImageDraw.Draw(grid)
    for index, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
        row, column = divmod(index, columns)
        x = padding + column * (tile_size + padding)
        y = padding + row * (tile_size + label_height + padding)
        grid.paste(tile, (x, y))
        draw.text((x, y + tile_size + 5), f"SYNTHETIC  /  SEED {label}", fill="#d7cbff")
    return grid

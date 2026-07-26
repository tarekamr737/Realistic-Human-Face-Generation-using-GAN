"""Reusable adversarial-training helpers used in the FFHQ Colab notebook."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image
from torch import nn

from .config import Settings
from .inference import make_grid
from .models import Discriminator, Generator


@dataclass
class EpochMetrics:
    epoch: int
    generator_loss: float
    discriminator_loss: float
    discriminator_real: float
    discriminator_fake: float
    collapsed: bool


def _finite_or_raise(value: torch.Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{label} became NaN or infinite. Stop and inspect the dataset and learning rate.")


def _batch_diversity(images: torch.Tensor, max_samples: int = 16) -> float:
    """Mean pairwise pixel distance, sampled cheaply as a mode-collapse warning signal."""
    # pdist has no CUDA half-precision kernel.  Sampling keeps this diagnostic
    # out of the hot training path while float32 keeps AMP training compatible.
    flattened = images.detach()[:max_samples].float().flatten(1)
    if len(flattened) < 2:
        return 0.0
    return float(torch.pdist(flattened, p=2).mean().item())


def train_epoch(
    generator: Generator,
    discriminator: Discriminator,
    dataloader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    latent_dim: int,
    scaler: torch.amp.GradScaler | None = None,
    progress: Any | None = None,
    diversity_interval: int = 100,
) -> dict[str, float]:
    """Train one epoch, optionally using CUDA mixed precision and a tqdm progress bar."""
    generator.train()
    discriminator.train()
    if diversity_interval < 1:
        raise ValueError("diversity_interval must be at least one")
    totals = {"generator_loss": 0.0, "discriminator_loss": 0.0, "real": 0.0, "fake": 0.0}
    batches = 0
    diversity_total = 0.0
    diversity_measurements = 0
    for real_images in dataloader:
        real_images = real_images.to(device, non_blocking=True)
        batch_size = real_images.size(0)
        real_targets = torch.full((batch_size,), 0.9, device=device)
        fake_targets = torch.zeros(batch_size, device=device)

        optimizer_d.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            real_logits = discriminator(real_images)
            real_loss = criterion(real_logits, real_targets)
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            fake_loss = criterion(fake_logits, fake_targets)
            discriminator_loss = real_loss + fake_loss
        _finite_or_raise(discriminator_loss, "Discriminator loss")
        if scaler is None:
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            optimizer_d.step()
        else:
            scaler.scale(discriminator_loss).backward()
            scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            scaler.step(optimizer_d)

        optimizer_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            generator_logits = discriminator(fake_images)
            generator_loss = criterion(generator_logits, real_targets)
        _finite_or_raise(generator_loss, "Generator loss")
        if scaler is None:
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            optimizer_g.step()
        else:
            scaler.scale(generator_loss).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            scaler.step(optimizer_g)
            scaler.update()

        totals["generator_loss"] += float(generator_loss.item())
        totals["discriminator_loss"] += float(discriminator_loss.item())
        totals["real"] += float(torch.sigmoid(real_logits).mean().item())
        totals["fake"] += float(torch.sigmoid(fake_logits).mean().item())
        batches += 1
        if batches % diversity_interval == 0:
            diversity_total += _batch_diversity(fake_images)
            diversity_measurements += 1
        if progress is not None:
            progress.set_postfix(
                G=f"{generator_loss.item():.3f}",
                D=f"{discriminator_loss.item():.3f}",
                refresh=False,
            )
    averages = {key: value / max(batches, 1) for key, value in totals.items()}
    averages["diversity"] = diversity_total / max(diversity_measurements, 1)
    return averages


def save_checkpoint(
    path: Path,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    settings: Settings,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "metrics": metrics,
        "config": asdict(settings),
    }
    if extra:
        payload.update(extra)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_training_checkpoint(
    path: Path,
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    return int(checkpoint["epoch"]) + 1


def export_generator(
    settings: Settings,
    generator: Generator,
    epoch: int,
    history: list[EpochMetrics],
    dataset_name: str = "FFHQ",
) -> None:
    """Export the exact contract loaded by the local inference-only app."""
    settings.ensure_output_directories()
    temporary_checkpoint = settings.checkpoint_path.with_suffix(f"{settings.checkpoint_path.suffix}.tmp")
    torch.save({"generator_state_dict": generator.state_dict()}, temporary_checkpoint)
    temporary_checkpoint.replace(settings.checkpoint_path)
    metadata = {
        "model_name": "FFHQ DCGAN-64",
        "architecture": "DCGAN",
        "checkpoint": settings.checkpoint_name,
        "dataset": dataset_name,
        "image_size": settings.image_size,
        "latent_dim": settings.latent_dim,
        "feature_maps": settings.feature_maps,
        "epoch": epoch,
        "inference_only": True,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    settings.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    settings.history_path.write_text(
        json.dumps([asdict(metric) for metric in history], indent=2), encoding="utf-8"
    )


def save_fixed_noise_grid(generator: Generator, fixed_noise: torch.Tensor, output_path: Path, epoch: int) -> None:
    generator.eval()
    with torch.inference_mode():
        samples = generator(fixed_noise).cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(samples, list(range(len(samples)))).save(output_path)


def evaluate_diversity(generator: Generator, latent_dim: int, device: torch.device, count: int = 64) -> dict[str, float]:
    generator.eval()
    with torch.inference_mode():
        images = generator(torch.randn(count, latent_dim, 1, 1, device=device)).cpu()
    diversity = _batch_diversity(images)
    return {"sample_count": count, "mean_pairwise_pixel_distance": diversity}

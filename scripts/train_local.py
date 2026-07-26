"""Fast, resumable local FFHQ DCGAN training for the FaceForge baseline.

This command intentionally trains the project's 64x64 model.  The source FFHQ
files may be 512x512; they are decoded and downsampled on the fly, so no second
copy of the 21 GB dataset is created.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faceforge.config import Settings
from faceforge.data import build_dataloader, validate_dataset
from faceforge.models import Discriminator, Generator, weights_init
from faceforge.training import EpochMetrics, evaluate_diversity, export_generator, save_checkpoint, save_fixed_noise_grid, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the FaceForge 64x64 FFHQ DCGAN locally with CUDA, AMP, tqdm, and atomic checkpoints."
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data", help="Folder containing FFHQ PNG/JPEG images.")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "ffhq_dcgan64", help="D: folder for checkpoints, samples, and metrics.")
    parser.add_argument("--export-dir", type=Path, default=PROJECT_ROOT / "models", help="D: folder used by the inference-only app.")
    parser.add_argument("--epochs", type=int, default=80, help="Total target epoch number, including resumed epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Safe starting batch size for a 4 GB RTX 3050 Ti.")
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 4) // 2)), help="DataLoader worker processes.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Queued batches per worker.")
    parser.add_argument("--feature-maps", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-every", type=int, default=5, help="Keep a numbered full-resume checkpoint every N epochs.")
    parser.add_argument("--resume", type=Path, help="A full checkpoint, normally runs/.../checkpoints/last.pt.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision (slower and uses more VRAM).")
    parser.add_argument("--deterministic", action="store_true", help="Prefer repeatability over cuDNN throughput.")
    parser.add_argument("--publish-final", action="store_true", help="After all epochs, publish the final candidate to models/generator_best.pt for the inference app.")
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def export_training_previews(source_dir: Path, export_dir: Path, count: int = 8) -> None:
    """Copy reviewed fixed-noise grids for the read-only training page in the app."""
    destination = export_dir / "samples"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("epoch_*.png"))[-count:]:
        target = destination / source.name
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    latest = sorted(source_dir.glob("epoch_*.png"))
    if latest:
        final_target = export_dir / "final_samples.png"
        temporary = final_target.with_suffix(f"{final_target.suffix}.tmp")
        shutil.copy2(latest[-1], temporary)
        temporary.replace(final_target)


def require_d_drive(*paths: Path) -> None:
    invalid = [str(path) for path in paths if Path(path).resolve().drive.upper() != "D:"]
    if invalid:
        raise ValueError("All dataset, run, and export paths must remain on partition D:. Invalid: " + ", ".join(invalid))


def configure_runtime(seed: int, deterministic: bool) -> None:
    """Keep process caches on D: and choose the requested speed/reproducibility trade-off."""
    tmp_dir = PROJECT_ROOT / "tmp"
    for directory in (tmp_dir, tmp_dir / "torch", tmp_dir / "matplotlib"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TORCH_HOME"] = str(tmp_dir / "torch")
    os.environ["MPLCONFIGDIR"] = str(tmp_dir / "matplotlib")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
    torch.set_float32_matmul_precision("high")


def ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This virtual environment has a CPU-only PyTorch build. "
            "Install the CUDA wheel shown in README.md, then rerun this command."
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < 3 * 1024**3:
        raise RuntimeError(f"{properties.name} exposes only {properties.total_memory / 1024**3:.1f} GB VRAM; at least 3 GB is required.")
    return device


def load_history(path: Path) -> list[EpochMetrics]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EpochMetrics(**{key: item[key] for key in EpochMetrics.__dataclass_fields__}) for item in raw]


def verify_resume_payload(payload: dict[str, Any], args: argparse.Namespace) -> None:
    required = {"epoch", "generator_state_dict", "discriminator_state_dict", "optimizer_g_state_dict", "optimizer_d_state_dict"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("Resume file is not a full FaceForge training checkpoint; missing " + ", ".join(missing))
    config = payload.get("config", {})
    expected = {"image_size": 64, "latent_dim": args.latent_dim, "feature_maps": args.feature_maps}
    mismatches = [f"{key}={config.get(key)!r} (expected {value!r})" for key, value in expected.items() if config.get(key, value) != value]
    if mismatches:
        raise ValueError("Resume configuration does not match this run: " + "; ".join(mismatches))


def save_loss_plot(history: list[EpochMetrics], output_path: Path) -> None:
    if not history:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot([metric.epoch for metric in history], [metric.generator_loss for metric in history], label="Generator")
    plt.plot([metric.epoch for metric in history], [metric.discriminator_loss for metric in history], label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("FFHQ DCGAN training losses")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    temporary = output_path.with_suffix(".tmp.png")
    plt.savefig(temporary, dpi=150)
    plt.close()
    temporary.replace(output_path)


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.run_dir = args.run_dir.resolve()
    args.export_dir = args.export_dir.resolve()
    require_d_drive(args.dataset, args.run_dir, args.export_dir, PROJECT_ROOT / "tmp")
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or args.snapshot_every < 1:
        raise ValueError("epochs, snapshot-every, and batch-size must be positive; batch-size must be at least two.")

    configure_runtime(args.seed, args.deterministic)
    device = ensure_cuda()
    checkpoint_dir = args.run_dir / "checkpoints"
    samples_dir = args.run_dir / "samples"
    for directory in (checkpoint_dir, samples_dir, args.export_dir):
        directory.mkdir(parents=True, exist_ok=True)

    report = validate_dataset(args.dataset, image_size=64, max_checks=256)
    if report.invalid_images:
        raise ValueError("Dataset validation failed for: " + ", ".join(report.invalid_images[:5]))
    if report.files_found < args.batch_size:
        raise ValueError(f"Dataset contains {report.files_found} images, fewer than batch size {args.batch_size}.")

    settings = Settings(
        dataset_dir=args.dataset,
        models_dir=args.export_dir,
        samples_dir=samples_dir,
        image_size=64,
        latent_dim=args.latent_dim,
        feature_maps=args.feature_maps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
    )
    run_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "dataset_images": report.files_found,
        "source_resolution": "512x512 (downsampled on the fly to 64x64)",
        "device": torch.cuda.get_device_name(device),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "amp": not args.no_amp,
        "deterministic": args.deterministic,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()},
    }
    atomic_write_json(args.run_dir / "run_manifest.json", run_manifest)

    print(f"Using {run_manifest['device']} | {report.files_found:,} images | 64x64 training | batch {args.batch_size} | AMP {not args.no_amp}")
    dataloader = build_dataloader(
        args.dataset,
        image_size=64,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )
    print(f"{len(dataloader):,} batches/epoch | {args.workers} loader workers | checkpoints: {checkpoint_dir}")

    generator = Generator(latent_dim=args.latent_dim, feature_maps=args.feature_maps).to(device)
    discriminator = Discriminator(feature_maps=args.feature_maps).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)
    history_path = args.run_dir / "history.json"
    history = load_history(history_path)
    start_epoch = 1

    if args.resume:
        resume_path = args.resume.resolve()
        require_d_drive(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        verify_resume_payload(payload, args)
        generator.load_state_dict(payload["generator_state_dict"])
        discriminator.load_state_dict(payload["discriminator_state_dict"])
        optimizer_g.load_state_dict(payload["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(payload["optimizer_d_state_dict"])
        if "fixed_noise" in payload:
            fixed_noise = payload["fixed_noise"].to(device)
        start_epoch = int(payload["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch} using {resume_path}")
    else:
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    if start_epoch > args.epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; requested target is {args.epochs}. Nothing to train.")
        if args.publish_final:
            evaluation = evaluate_diversity(generator, args.latent_dim, device)
            evaluation.update({"epoch": start_epoch - 1, "note": "Published from a reviewed completed checkpoint."})
            export_generator(settings, generator, start_epoch - 1, history)
            export_training_previews(samples_dir, args.export_dir)
            atomic_write_json(settings.evaluation_path, evaluation)
            print(f"Published inference checkpoint: {settings.checkpoint_path}")
        return 0

    outer = tqdm(range(start_epoch, args.epochs + 1), desc="Training", unit="epoch", dynamic_ncols=True)
    try:
        for epoch in outer:
            epoch_started = perf_counter()
            batches = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False, dynamic_ncols=True)
            metrics = train_epoch(
                generator,
                discriminator,
                batches,
                optimizer_g,
                optimizer_d,
                criterion,
                device,
                latent_dim=args.latent_dim,
                scaler=scaler,
                progress=batches,
            )
            batches.close()
            epoch_seconds = perf_counter() - epoch_started
            metric = EpochMetrics(
                epoch=epoch,
                generator_loss=metrics["generator_loss"],
                discriminator_loss=metrics["discriminator_loss"],
                discriminator_real=metrics["real"],
                discriminator_fake=metrics["fake"],
                collapsed=metrics["diversity"] < 1.0,
            )
            history = [item for item in history if item.epoch != epoch] + [metric]
            history.sort(key=lambda item: item.epoch)
            checkpoint_extra = {"fixed_noise": fixed_noise.detach().cpu(), "history": [asdict(item) for item in history]}
            save_checkpoint(
                checkpoint_dir / "last.pt",
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch,
                metrics,
                settings,
                extra=checkpoint_extra,
            )
            if epoch % args.snapshot_every == 0 or epoch == args.epochs:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    generator,
                    discriminator,
                    optimizer_g,
                    optimizer_d,
                    epoch,
                    metrics,
                    settings,
                    extra=checkpoint_extra,
                )
            save_fixed_noise_grid(generator, fixed_noise, samples_dir / f"epoch_{epoch:03d}.png", epoch)
            atomic_write_json(history_path, [asdict(item) for item in history])
            save_loss_plot(history, args.run_dir / "losses.png")
            outer.set_postfix(G=f"{metric.generator_loss:.3f}", D=f"{metric.discriminator_loss:.3f}", epoch_min=f"{epoch_seconds / 60:.1f}")
            outer.write(
                f"Epoch {epoch:03d}/{args.epochs} | {epoch_seconds / 60:.1f} min | "
                f"G {metric.generator_loss:.4f} | D {metric.discriminator_loss:.4f} | diversity {metrics['diversity']:.2f}"
            )
    except KeyboardInterrupt:
        outer.write("Stopped safely. The last completed epoch is in checkpoints/last.pt; resume from it when ready.")
        return 130
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            torch.cuda.empty_cache()
            outer.write("CUDA out of memory. The last completed epoch is safe; resume with --batch-size 32.")
            return 2
        raise
    finally:
        outer.close()

    final_candidate = args.run_dir / "generator_final_candidate.pt"
    atomic_torch_save(final_candidate, {"generator_state_dict": generator.state_dict()})
    evaluation = evaluate_diversity(generator, args.latent_dim, device)
    evaluation.update({"epoch": args.epochs, "note": "Review fixed-noise grids for artifacts and collapse before publishing."})
    atomic_write_json(args.run_dir / "evaluation.json", evaluation)
    print(f"Training complete. Candidate generator: {final_candidate}")
    if args.publish_final:
        export_generator(settings, generator, args.epochs, history)
        export_training_previews(samples_dir, args.export_dir)
        atomic_write_json(settings.evaluation_path, evaluation)
        print(f"Published inference checkpoint: {settings.checkpoint_path}")
    else:
        print("Not published to the web app automatically. Review samples, then rerun with --resume ... --publish-final or publish a reviewed checkpoint.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

"""Optional, inference-only bridge for the official R3GAN FFHQ checkpoint.

The R3GAN checkpoint is a Python pickle.  Loading a pickle executes Python code,
so this module deliberately accepts a *local* file only and the setup script
downloads it from the official BrownVC Hugging Face repository.  Do not replace
that file with an untrusted pickle.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .config import PROJECT_ROOT, Settings


class R3GANUnavailable(RuntimeError):
    """Raised when the optional official R3GAN runtime has not been installed."""


@dataclass
class LoadedR3GAN:
    model: Any
    device: torch.device
    metadata: dict[str, Any]


@contextmanager
def _r3gan_import_path(source_dir: Path) -> Iterator[None]:
    """Make the official source importable without globally preferring it."""
    source = str(source_dir.resolve())
    sys.path.insert(0, source)
    try:
        yield
    finally:
        try:
            sys.path.remove(source)
        except ValueError:
            pass


def load_r3gan(settings: Settings, device_name: str | None = None) -> LoadedR3GAN:
    """Load the official FFHQ-256 R3GAN generator in evaluation mode.

    This does not train, fine-tune, fetch from the network, or accept a model
    path from an HTTP request.  The checkpoint must be installed beforehand.
    """
    source_dir = settings.r3gan_source_dir
    checkpoint = settings.r3gan_checkpoint_path
    if not source_dir.is_dir():
        raise R3GANUnavailable(
            f"Official R3GAN source is unavailable at {source_dir}. Run scripts\\setup_r3gan_comparison.ps1."
        )
    if not checkpoint.is_file():
        raise R3GANUnavailable(
            f"Official R3GAN checkpoint is unavailable at {checkpoint}. Run scripts\\setup_r3gan_comparison.ps1."
        )
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise R3GANUnavailable("R3GAN comparison requires a CUDA-capable PyTorch runtime.")

    # R3GAN inherits StyleGAN custom CUDA extensions.  Keep their build/cache
    # artifacts inside this D: workspace instead of silently using C: defaults.
    tmp_dir = PROJECT_ROOT / "tmp"
    extension_dir = tmp_dir / "r3gan-torch-extensions"
    dnnlib_cache = tmp_dir / "r3gan-dnnlib-cache"
    for directory in (tmp_dir, extension_dir, dnnlib_cache):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(extension_dir)
    os.environ["DNNLIB_CACHE_DIR"] = str(dnnlib_cache)
    # When the app is launched through an absolute Python path, Windows does
    # not always add the virtual environment's Scripts directory to PATH.
    # PyTorch invokes ``ninja`` by name while compiling the upstream plugins.
    venv_scripts = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    if venv_scripts.lower() not in current_path.lower().split(os.pathsep):
        os.environ["PATH"] = venv_scripts + os.pathsep + current_path

    use_custom_ops = os.getenv("FACEFORGE_R3GAN_CUSTOM_OPS", "0").strip().lower() in {"1", "true", "yes"}
    try:
        with _r3gan_import_path(source_dir):
            import dnnlib  # type: ignore[import-not-found]
            import legacy  # type: ignore[import-not-found]

            if not use_custom_ops:
                # The upstream implementation supplies accurate pure-PyTorch
                # fallbacks for both CUDA plugins.  Prefer them by default:
                # they keep this local inference app compatible with CUDA
                # wheels alone and do not require a system CUDA toolkit or C++
                # compiler.  Set FACEFORGE_R3GAN_CUSTOM_OPS=1 only if those
                # toolchain prerequisites are deliberately installed.
                from torch_utils.ops import bias_act, upfirdn2d  # type: ignore[import-not-found]

                bias_act._init = lambda: False
                upfirdn2d._init = lambda: False

            # The official R3GAN generation script uses the same loader and
            # selects G_ema, the smoothed generator intended for inference.
            with dnnlib.util.open_url(str(checkpoint)) as file:
                payload = legacy.load_network_pkl(file)
    except Exception as error:  # Third-party extensions emit several exception types.
        raise R3GANUnavailable(f"Could not load the official R3GAN checkpoint: {error}") from error

    model = payload.get("G_ema") if isinstance(payload, dict) else None
    if model is None or not hasattr(model, "z_dim") or not hasattr(model, "c_dim"):
        raise R3GANUnavailable("The R3GAN checkpoint does not contain a compatible G_ema generator.")
    model = model.to(device).eval()
    return LoadedR3GAN(
        model=model,
        device=device,
        metadata={
            "id": "r3gan-ffhq-256",
            "model_name": "R3GAN FFHQ-256 (official pretrained)",
            "architecture": "R3GAN",
            "dataset": "FFHQ",
            "image_size": 256,
            "latent_dim": int(model.z_dim),
            "checkpoint": checkpoint.name,
            "source": "brownvc/R3GAN",
            "inference_only": True,
            "operator_backend": "custom CUDA extensions" if use_custom_ops else "PyTorch reference operators",
            "sampling_note": "Official native sampling; the DCGAN latent-range control does not apply.",
        },
    )


def generate_r3gan_batch(loaded: LoadedR3GAN, count: int, seed: int) -> tuple[torch.Tensor, list[int]]:
    """Generate the same way as the repository's ``gen_images.py`` script."""
    if count not in (1, 4, 8, 16):
        raise ValueError("Face count must be 1, 4, 8, or 16.")
    seeds = [seed + index for index in range(count)]
    label = torch.zeros([1, int(loaded.model.c_dim)], device=loaded.device)
    images: list[torch.Tensor] = []
    with torch.inference_mode():
        for item_seed in seeds:
            latent = torch.from_numpy(np.random.RandomState(item_seed).randn(1, int(loaded.model.z_dim))).to(loaded.device)
            image = loaded.model(latent, label)
            images.append(image.detach().cpu())
    return torch.cat(images, dim=0), seeds

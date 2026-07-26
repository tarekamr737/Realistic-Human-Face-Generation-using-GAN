"""HTTP boundary for local, inference-only FaceForge generation."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from faceforge.config import Settings
from faceforge.inference import LoadedGenerator, generate_batch, image_to_data_url, load_generator, make_grid, tensor_to_image
from faceforge.r3gan import LoadedR3GAN, R3GANUnavailable, generate_r3gan_batch, load_r3gan


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"


class GenerateRequest(BaseModel):
    count: Literal[1, 4, 8, 16] = 4
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    truncation: float = Field(default=1.0, ge=0.25, le=2.0)


class Runtime:
    def __init__(self) -> None:
        self.settings = Settings()
        self.model: LoadedGenerator | None = None
        self.error: str | None = None
        self.r3gan: LoadedR3GAN | None = None
        self.r3gan_error: str | None = None
        self.try_load()

    def try_load(self) -> None:
        try:
            self.model = load_generator(self.settings)
            self.error = None
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
            self.model = None
            self.error = str(error)

    def try_load_r3gan(self) -> None:
        try:
            self.r3gan = load_r3gan(self.settings)
            self.r3gan_error = None
        except (FileNotFoundError, R3GANUnavailable, RuntimeError, ValueError, OSError) as error:
            self.r3gan = None
            self.r3gan_error = str(error)

    def info(self) -> dict[str, object]:
        if self.model is None:
            self.try_load()
        if self.model is None:
            return {
                "available": False,
                "message": self.error or "No generator checkpoint is available.",
                "checkpoint": self.settings.checkpoint_name,
                "image_size": self.settings.image_size,
                "latent_dim": self.settings.latent_dim,
                "inference_only": True,
            }
        return {
            "available": True,
            "message": "Generator loaded locally. This app performs inference only.",
            "device": str(self.model.device),
            "id": "faceforge-dcgan-64",
            **self.model.metadata,
        }

    def models_info(self) -> dict[str, object]:
        """Describe both generators without making training functionality available."""
        dcgan = self.info()
        if self.r3gan is None:
            self.try_load_r3gan()
        if self.r3gan is None:
            r3gan: dict[str, object] = {
                "id": "r3gan-ffhq-256",
                "available": False,
                "message": self.r3gan_error or "R3GAN has not been installed.",
                "architecture": "R3GAN",
                "dataset": "FFHQ",
                "image_size": 256,
                "checkpoint": self.settings.r3gan_checkpoint_name,
                "inference_only": True,
            }
        else:
            r3gan = {
                "available": True,
                "message": "Official pretrained R3GAN loaded locally. This app performs inference only.",
                "device": str(self.r3gan.device),
                **self.r3gan.metadata,
            }
        entries = [dcgan, r3gan]
        return {
            "available": any(bool(entry["available"]) for entry in entries),
            "comparison_ready": all(bool(entry["available"]) for entry in entries),
            "models": entries,
            "inference_only": True,
            "comparison_note": (
                "The same numeric seed is supplied to both models, but their latent spaces and native resolutions differ. "
                "This is a reproducible side-by-side comparison, not an image-paired or equal-resolution benchmark."
            ),
        }


runtime = Runtime()
app = FastAPI(title="FaceForge AI", version="1.0.0")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_available": bool(runtime.models_info()["available"])}


@app.get("/api/model")
def model_info() -> dict[str, object]:
    return runtime.info()


@app.get("/api/models")
def models_info() -> dict[str, object]:
    return runtime.models_info()


@app.get("/api/training")
def training_info() -> dict[str, object]:
    if not runtime.settings.history_path.exists():
        return {"available": False, "history": [], "message": "Training history has not been exported from Colab yet."}
    history = json.loads(runtime.settings.history_path.read_text(encoding="utf-8"))
    samples: list[str] = []
    if runtime.settings.samples_dir.exists():
        samples.extend(path.name for path in sorted(runtime.settings.samples_dir.glob("*.png")))
    final_samples = runtime.settings.models_dir / "final_samples.png"
    if final_samples.exists():
        samples.append(final_samples.name)
    return {"available": True, "history": history, "samples": samples[-8:]}


@app.get("/api/training-samples/{filename}")
def training_sample(filename: str) -> FileResponse:
    if Path(filename).name != filename or not filename.endswith(".png"):
        raise HTTPException(status_code=404, detail="Sample not found.")
    candidates = [runtime.settings.samples_dir / filename, runtime.settings.models_dir / filename]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise HTTPException(status_code=404, detail="Sample not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/evaluation")
def evaluation_info() -> dict[str, object]:
    if not runtime.settings.evaluation_path.exists():
        return {"available": False, "message": "Evaluation results have not been exported from Colab yet."}
    return {"available": True, **json.loads(runtime.settings.evaluation_path.read_text(encoding="utf-8"))}


@app.post("/api/generate")
def generate(request: GenerateRequest) -> JSONResponse:
    if runtime.model is None:
        runtime.try_load()
    if runtime.r3gan is None:
        runtime.try_load_r3gan()
    if runtime.model is None and runtime.r3gan is None:
        message = runtime.error or runtime.r3gan_error or "No compatible generator checkpoint is available."
        raise HTTPException(status_code=503, detail=message)
    seed = request.seed if request.seed is not None else secrets.randbelow(2_147_483_647)
    model_results: list[dict[str, object]] = []
    try:
        if runtime.model is not None:
            images, image_seeds = generate_batch(runtime.model, request.count, seed, request.truncation)
            model_results.append(_serialize_result(
                model_id="faceforge-dcgan-64",
                display_name="FaceForge DCGAN-64 (ours)",
                loaded=runtime.model,
                images=images,
                image_seeds=image_seeds,
                sampling_note=f"Latent range {request.truncation:.2f} applied.",
            ))
        if runtime.r3gan is not None:
            images, image_seeds = generate_r3gan_batch(runtime.r3gan, request.count, seed)
            model_results.append(_serialize_result(
                model_id="r3gan-ffhq-256",
                display_name="R3GAN FFHQ-256 (official pretrained)",
                loaded=runtime.r3gan,
                images=images,
                image_seeds=image_seeds,
                sampling_note="Native official sampling; latent range is not applied.",
            ))
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    primary = model_results[0]
    return JSONResponse(
        {
            "seed": seed,
            "count": request.count,
            "truncation": request.truncation,
            "models": model_results,
            "comparison_ready": len(model_results) == 2,
            # Kept for the original single-model UI/API contract.  New clients
            # should read the explicit per-model entries above.
            "images": primary["images"],
            "grid": primary["grid"],
            "grid_filename": primary["grid_filename"],
        }
    )


def _serialize_result(
    *,
    model_id: str,
    display_name: str,
    loaded: LoadedGenerator | LoadedR3GAN,
    images: object,
    image_seeds: list[int],
    sampling_note: str,
) -> dict[str, object]:
    """Convert one model output to the labelled, download-ready API contract."""
    image_tensor = images
    cards = [
        {
            "seed": image_seed,
            "image": image_to_data_url(tensor_to_image(image)),
            "filename": f"{model_id}-synthetic-seed-{image_seed}.png",
            "label": f"Synthetic face generated by {display_name}",
        }
        for image, image_seed in zip(image_tensor, image_seeds, strict=True)
    ]
    metadata = loaded.metadata
    return {
        "id": model_id,
        "display_name": display_name,
        "architecture": metadata.get("architecture"),
        "image_size": metadata.get("image_size"),
        "checkpoint": metadata.get("checkpoint"),
        "sampling_note": sampling_note,
        "images": cards,
        "grid": image_to_data_url(make_grid(image_tensor, image_seeds)),
        "grid_filename": f"{model_id}-synthetic-grid-seed-{image_seeds[0]}.png",
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

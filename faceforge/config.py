"""One configuration contract shared by training, export, and inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """File locations and safe defaults for the 64px FFHQ DCGAN baseline."""

    dataset_dir: Path = Path(os.getenv("FACEFORGE_DATASET_DIR", PROJECT_ROOT / "data" / "ffhq"))
    models_dir: Path = Path(os.getenv("FACEFORGE_MODELS_DIR", PROJECT_ROOT / "models"))
    samples_dir: Path = Path(os.getenv("FACEFORGE_SAMPLES_DIR", PROJECT_ROOT / "models" / "samples"))
    checkpoint_name: str = os.getenv("FACEFORGE_CHECKPOINT", "generator_best.pt")
    r3gan_source_dir: Path = Path(
        os.getenv("FACEFORGE_R3GAN_SOURCE_DIR", PROJECT_ROOT / "third_party" / "R3GAN")
    )
    r3gan_models_dir: Path = Path(
        os.getenv("FACEFORGE_R3GAN_MODELS_DIR", PROJECT_ROOT / "models" / "r3gan")
    )
    r3gan_checkpoint_name: str = os.getenv("FACEFORGE_R3GAN_CHECKPOINT", "network-snapshot-final.pkl")
    metadata_name: str = "model_metadata.json"
    history_name: str = "training_history.json"
    evaluation_name: str = "evaluation.json"
    image_size: int = int(os.getenv("FACEFORGE_IMAGE_SIZE", "64"))
    latent_dim: int = int(os.getenv("FACEFORGE_LATENT_DIM", "128"))
    feature_maps: int = int(os.getenv("FACEFORGE_FEATURE_MAPS", "64"))
    batch_size: int = int(os.getenv("FACEFORGE_BATCH_SIZE", "64"))
    epochs: int = int(os.getenv("FACEFORGE_EPOCHS", "50"))
    learning_rate: float = float(os.getenv("FACEFORGE_LEARNING_RATE", "0.0002"))
    beta1: float = float(os.getenv("FACEFORGE_BETA1", "0.5"))

    @property
    def checkpoint_path(self) -> Path:
        return self.models_dir / self.checkpoint_name

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / self.metadata_name

    @property
    def r3gan_checkpoint_path(self) -> Path:
        return self.r3gan_models_dir / self.r3gan_checkpoint_name

    @property
    def history_path(self) -> Path:
        return self.models_dir / self.history_name

    @property
    def evaluation_path(self) -> Path:
        return self.models_dir / self.evaluation_name

    def ensure_output_directories(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

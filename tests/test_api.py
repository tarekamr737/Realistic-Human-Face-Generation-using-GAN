from __future__ import annotations

import unittest
import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, runtime
from faceforge.config import Settings
from faceforge.models import Generator


class ApiBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_model_endpoint_identifies_inference_only_mode(self) -> None:
        response = self.client.get("/api/model")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["inference_only"])

    def test_models_endpoint_exposes_only_inference_comparison_metadata(self) -> None:
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["inference_only"])
        self.assertEqual({model["id"] for model in payload["models"]}, {"faceforge-dcgan-64", "r3gan-ffhq-256"})

    def test_invalid_face_count_is_rejected_at_api_boundary(self) -> None:
        response = self.client.post("/api/generate", json={"count": 3, "seed": 1, "truncation": 1.0})
        self.assertEqual(response.status_code, 422)

    def test_generator_endpoint_returns_labelled_images_from_a_checkpoint(self) -> None:
        import torch

        test_root = Path.cwd() / "tmp" / "test-api-model"
        shutil.rmtree(test_root, ignore_errors=True)
        test_root.mkdir(parents=True)
        settings = Settings(models_dir=test_root, checkpoint_name="generator_best.pt", latent_dim=128, feature_maps=8)
        model = Generator(latent_dim=128, feature_maps=8)
        torch.save({"generator_state_dict": model.state_dict()}, settings.checkpoint_path)
        settings.metadata_path.write_text(
            json.dumps({"architecture": "DCGAN", "latent_dim": 128, "feature_maps": 8, "image_size": 64}),
            encoding="utf-8",
        )
        previous_settings, previous_model, previous_error = runtime.settings, runtime.model, runtime.error
        try:
            runtime.settings, runtime.model, runtime.error = settings, None, None
            response = self.client.post("/api/generate", json={"count": 1, "seed": 88, "truncation": 1.0})
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["seed"], 88)
            self.assertEqual(len(payload["images"]), 1)
            self.assertEqual(payload["models"][0]["id"], "faceforge-dcgan-64")
            self.assertIn("data:image/png;base64,", payload["images"][0]["image"])
            self.assertIn("synthetic", payload["images"][0]["label"].lower())
        finally:
            runtime.settings, runtime.model, runtime.error = previous_settings, previous_model, previous_error
            shutil.rmtree(test_root, ignore_errors=True)

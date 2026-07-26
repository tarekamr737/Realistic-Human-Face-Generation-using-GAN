from __future__ import annotations

import importlib.util
import json
import shutil
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for inference tests")
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from faceforge.config import Settings
        from faceforge.models import Generator

        cls.torch = torch
        cls.test_root = Path.cwd() / "tmp" / "test-inference"
        shutil.rmtree(cls.test_root, ignore_errors=True)
        cls.test_root.mkdir(parents=True)
        cls.settings = Settings(models_dir=cls.test_root, checkpoint_name="generator_best.pt", latent_dim=128, feature_maps=16)
        model = Generator(latent_dim=128, feature_maps=16)
        torch.save({"generator_state_dict": model.state_dict()}, cls.settings.checkpoint_path)
        cls.settings.metadata_path.write_text(json.dumps({"architecture": "DCGAN", "latent_dim": 128, "feature_maps": 16, "image_size": 64}), encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.test_root, ignore_errors=True)

    def test_checkpoint_loads_and_fixed_seed_is_deterministic(self) -> None:
        from faceforge.inference import generate_batch, load_generator

        loaded = load_generator(self.settings, device_name="cpu")
        first, first_seeds = generate_batch(loaded, count=4, seed=81, truncation=1.0)
        second, second_seeds = generate_batch(loaded, count=4, seed=81, truncation=1.0)
        self.assertEqual(first_seeds, second_seeds)
        self.assertTrue(self.torch.equal(first, second))

    def test_invalid_count_is_rejected(self) -> None:
        from faceforge.inference import generate_batch, load_generator

        loaded = load_generator(self.settings, device_name="cpu")
        with self.assertRaises(ValueError):
            generate_batch(loaded, count=3, seed=1)

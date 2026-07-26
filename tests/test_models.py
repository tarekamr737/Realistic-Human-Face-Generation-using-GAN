from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model tests")
class ModelShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        from faceforge.models import Discriminator, Generator

        self.torch = torch
        self.generator = Generator(latent_dim=128, feature_maps=16)
        self.discriminator = Discriminator(feature_maps=16)

    def test_generator_produces_normalized_64px_images(self) -> None:
        output = self.generator(self.torch.randn(3, 128, 1, 1))
        self.assertEqual(tuple(output.shape), (3, 3, 64, 64))
        self.assertLessEqual(float(output.max().detach()), 1.0)
        self.assertGreaterEqual(float(output.min().detach()), -1.0)

    def test_discriminator_scores_each_image(self) -> None:
        scores = self.discriminator(self.torch.randn(3, 3, 64, 64))
        self.assertEqual(tuple(scores.shape), (3,))

    def test_invalid_generator_latent_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.generator(self.torch.randn(2, 128))

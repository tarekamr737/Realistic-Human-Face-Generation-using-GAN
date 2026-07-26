from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for the training smoke test")
class TrainingSmokeTests(unittest.TestCase):
    def test_one_tiny_adversarial_epoch(self) -> None:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        from faceforge.models import Discriminator, Generator
        from faceforge.training import train_epoch

        generator = Generator(latent_dim=8, feature_maps=4)
        discriminator = Discriminator(feature_maps=4)
        loader = DataLoader(TensorDataset(torch.randn(8, 3, 64, 64)), batch_size=4)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        metrics = train_epoch(generator, discriminator, (item[0] for item in loader), optimizer_g, optimizer_d, nn.BCEWithLogitsLoss(), torch.device("cpu"), latent_dim=8)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["generator_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["discriminator_loss"])))

from __future__ import annotations

import importlib.util
import shutil
import unittest
from pathlib import Path


DEPS_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("PIL") is not None


@unittest.skipUnless(DEPS_AVAILABLE, "PyTorch and Pillow are required for preprocessing tests")
class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PIL import Image

        cls.root = Path.cwd() / "tmp" / "test-images"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.root.mkdir(parents=True)
        Image.new("RGB", (120, 90), "#a078ff").save(cls.root / "face.png")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_preprocessing_returns_normalized_tensor(self) -> None:
        from faceforge.data import FaceImageDataset, validate_dataset

        report = validate_dataset(self.root, image_size=64)
        image = FaceImageDataset(self.root, image_size=64)[0]
        self.assertEqual(report.files_found, 1)
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertGreaterEqual(float(image.min()), -1.0)
        self.assertLessEqual(float(image.max()), 1.0)

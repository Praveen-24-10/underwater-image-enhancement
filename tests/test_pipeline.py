import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from underwater_enhancement.pipeline import (PipelineConfig, clahe_enhancement, denoise,
    gray_world_white_balance, resize_image, run_pipeline, sharpen)
from underwater_enhancement.metrics import mse, psnr, ssim, uciqe, uiqm


class PipelineTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[:32, :40]
        self.image = np.stack((x * 5 % 256, y * 7 % 256, (x + y) * 3 % 256), axis=2).astype(np.uint8)

    def test_classical_operations_keep_valid_images(self):
        for result in (gray_world_white_balance(self.image), clahe_enhancement(self.image),
                       denoise(self.image, "median"), sharpen(self.image), resize_image(self.image, 2)):
            self.assertEqual(result.dtype, np.uint8)
            self.assertGreater(result.size, 0)

    def test_pipeline_tracks_output(self):
        output, stages = run_pipeline(self.image, PipelineConfig())
        self.assertIn("Original", stages)
        self.assertIn("Enhanced output", stages)
        self.assertEqual(output.shape, self.image.shape)

    def test_metrics_identity_and_quality(self):
        self.assertEqual(mse(self.image, self.image), 0.0)
        self.assertEqual(psnr(self.image, self.image), float("inf"))
        self.assertAlmostEqual(ssim(self.image, self.image), 1.0, places=5)
        self.assertTrue(np.isfinite(uciqe(self.image)))
        self.assertTrue(np.isfinite(uiqm(self.image)))

    def test_invalid_even_median_kernel_rejected(self):
        with self.assertRaises(ValueError):
            denoise(self.image, "median", 4)


if __name__ == "__main__":
    unittest.main()

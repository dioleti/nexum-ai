import unittest

import numpy as np

from nexum.document.layout.detectors.image import ImageDetector
from nexum.document.models import ImageRegion


class TestImageDetector(unittest.TestCase):
    def test_detect_images(self):
        gray = np.full((200, 200), 255, dtype=np.uint8)
        gray[20:120, 20:120] = 0

        detector = ImageDetector(min_image_area=1000)
        images = detector.detect(gray)

        self.assertGreaterEqual(len(images), 1)
        self.assertIsInstance(images[0], ImageRegion)

    def test_detect_filters_small_images(self):
        gray = np.full((200, 200), 255, dtype=np.uint8)
        gray[10:15, 10:15] = 0

        detector = ImageDetector(
            min_image_area=50000,
            enable_deskew=False,
            enable_clahe=False,
            enable_denoise=False,
            enable_sharpen=False,
            enable_binarize=False,
        )
        images = detector.detect(gray)
        self.assertEqual(images, [])


if __name__ == "__main__":
    unittest.main()

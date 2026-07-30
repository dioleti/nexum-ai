import unittest

import numpy as np
from PIL import Image

from nexum.document.models import ImageOCRConfig
from nexum.document.parser.image import ImageOCRParser


class TestImageOCRParser(unittest.TestCase):
    def _create_png_bytes(self, text="test"):
        img = Image.new("RGB", (200, 50), color="white")
        return img.tobytes()

    def test_config_assignment(self):
        cfg = ImageOCRConfig()
        parser = ImageOCRParser(cfg)
        self.assertIs(parser.config, cfg)

    def test_denoise_disabled(self):
        cfg = ImageOCRConfig(enable_denoise=False)
        parser = ImageOCRParser(cfg)
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = parser._denoise(gray)
        self.assertTrue(np.array_equal(result, gray))

    def test_sharpen_disabled(self):
        cfg = ImageOCRConfig(enable_sharpen=False)
        parser = ImageOCRParser(cfg)
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = parser._sharpen(gray)
        self.assertTrue(np.array_equal(result, gray))

    def test_binarize_disabled(self):
        cfg = ImageOCRConfig(enable_binarization=False)
        parser = ImageOCRParser(cfg)
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = parser._binarize(gray)
        self.assertTrue(np.array_equal(result, gray))

    def test_parse_raises_on_invalid_bytes(self):
        parser = ImageOCRParser(ImageOCRConfig())
        with self.assertRaises(Exception):
            parser.parse(b"not_an_image")


if __name__ == "__main__":
    unittest.main()

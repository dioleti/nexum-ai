import io
import unittest

import numpy as np
from PIL import Image

from nexum.document.models import PDFOCRConfig
from nexum.document.parser.pdf import PDFOCRParser


class TestPDFOCRParser(unittest.TestCase):
    def _create_png_bytes(self):
        img = Image.new("RGB", (200, 50), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_config_assignment(self):
        cfg = PDFOCRConfig()
        parser = PDFOCRParser(cfg)
        self.assertIs(parser.config, cfg)

    def test_extract_dpi_default(self):
        img = Image.new("RGB", (10, 10))
        dpi = PDFOCRParser._extract_dpi(img)
        self.assertEqual(dpi, 72)

    def test_deskew_no_text(self):
        cfg = PDFOCRConfig()
        gray = np.full((10, 10), 255, dtype=np.uint8)
        result = PDFOCRParser._deskew(gray, cfg)
        self.assertTrue(np.array_equal(result, gray))

    def test_sharpen(self):
        cfg = PDFOCRConfig(enable_sharpen=True)
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = PDFOCRParser._sharpen(gray, cfg)
        self.assertEqual(result.shape, gray.shape)

    def test_denoise_bilateral(self):
        cfg = PDFOCRConfig(denoise_method="bilateral")
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = PDFOCRParser._denoise(gray, cfg)
        self.assertEqual(result.shape, gray.shape)

    def test_denoise_median(self):
        cfg = PDFOCRConfig(denoise_method="median")
        gray = np.zeros((10, 10), dtype=np.uint8)
        result = PDFOCRParser._denoise(gray, cfg)
        self.assertEqual(result.shape, gray.shape)

    def test_binarize_otsu(self):
        cfg = PDFOCRConfig(binarization_method="otsu")
        gray = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        result = PDFOCRParser._binarize(gray, cfg)
        self.assertEqual(result.shape, gray.shape)

    def test_parse_invalid_bytes(self):
        parser = PDFOCRParser(PDFOCRConfig())
        with self.assertRaises(Exception):
            parser.parse(b"invalid")


if __name__ == "__main__":
    unittest.main()

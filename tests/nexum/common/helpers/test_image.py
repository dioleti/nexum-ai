import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from nexum.common.errors import NexumRuntimeError
from nexum.common.helpers.image import (
    binarize,
    clahe,
    denoise,
    deskew,
    detect_images_heuristic,
    extract_dpi,
    normalize_ocr_output,
    ocr_region,
    osd_rotation,
    run_ocr,
    run_ocr_data,
    sharpen,
    validate_max_size,
)
from nexum.document.models import Region, RegionType


class TestImageHelpers(unittest.TestCase):

    def test_extract_dpi(self):
        img = Image.new("RGB", (100, 100))
        img.info["dpi"] = (300, 300)
        self.assertEqual(extract_dpi(img), 300)

        img_no_dpi = Image.new("RGB", (100, 100))
        self.assertEqual(extract_dpi(img_no_dpi), 72)

    def test_deskew(self):
        # All white image (no black pixels)
        gray = np.full((100, 100), 255, dtype=np.uint8)
        res = deskew(gray)
        np.testing.assert_array_equal(res, gray)

        # Image with pixels
        gray[40:60, 40:60] = 0
        res = deskew(gray, max_angle=45.0)
        self.assertEqual(res.shape, gray.shape)

    def test_clahe(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        res = clahe(gray)
        self.assertEqual(res.shape, (100, 100))

    def test_sharpen(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        res = sharpen(gray)
        self.assertEqual(res.shape, (100, 100))

    def test_denoise(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        res_bilateral = denoise(gray, denoise_method="bilateral")
        self.assertEqual(res_bilateral.shape, (100, 100))

        res_median = denoise(gray, denoise_method="median")
        self.assertEqual(res_median.shape, (100, 100))

        res_none = denoise(gray, denoise_method="none")
        np.testing.assert_array_equal(res_none, gray)

    def test_binarize(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        res_otsu = binarize(gray, binarization_method="otsu")
        self.assertEqual(res_otsu.shape, (100, 100))

        res_adaptive = binarize(gray, binarization_method="adaptive")
        self.assertEqual(res_adaptive.shape, (100, 100))

        res_none = binarize(gray, binarization_method="none")
        np.testing.assert_array_equal(res_none, gray)

    @patch("nexum.common.helpers.image.pytesseract.image_to_osd")
    def test_osd_rotation(self, mock_image_to_osd):
        mock_image_to_osd.return_value = "Rotate: 90\nOrientation confidence: 10.0"
        img = Image.new("RGB", (100, 200), color="white")
        rotated = osd_rotation(img, osd_min_confidence=5.0)
        self.assertEqual(rotated.size, (200, 100))

        # Low confidence -> no rotation
        mock_image_to_osd.return_value = "Rotate: 90\nOrientation confidence: 2.0"
        no_rot = osd_rotation(img, osd_min_confidence=5.0)
        self.assertEqual(no_rot.size, (100, 200))

    @patch("nexum.common.helpers.image.pytesseract.image_to_string")
    def test_run_ocr(self, mock_image_to_string):
        mock_image_to_string.return_value = "Hello World\x0c"
        gray = np.zeros((50, 50), dtype=np.uint8)
        result = run_ocr(gray, lang="eng", preserve_spaces=True)
        self.assertEqual(result, "Hello World")

    @patch("nexum.common.helpers.image.pytesseract.image_to_data")
    def test_run_ocr_data(self, mock_image_to_data):
        mock_image_to_data.return_value = {"text": ["Hello"]}
        gray = np.zeros((50, 50), dtype=np.uint8)
        res = run_ocr_data(gray)
        self.assertEqual(res, {"text": ["Hello"]})

    def test_validate_max_size(self):
        small_arr = np.zeros((100, 100), dtype=np.uint8)
        validate_max_size(small_arr, max_px=20000)

        large_arr = np.zeros((200, 200), dtype=np.uint8)
        with self.assertRaises(NexumRuntimeError):
            validate_max_size(large_arr, max_px=20000)

    @patch("nexum.common.helpers.image.pytesseract.image_to_string")
    def test_ocr_region(self, mock_image_to_string):
        mock_image_to_string.return_value = "  Test text \n"
        img = Image.new("RGB", (100, 100), color="white")
        region = Region(type=RegionType.TEXT, box=(10, 10, 50, 50), score=0.9, text=None)

        result = ocr_region(img, region)
        self.assertEqual(result, "Test text")

    def test_normalize_ocr_output(self):
        self.assertEqual(normalize_ocr_output("hello"), "hello")
        self.assertEqual(normalize_ocr_output({"text": "from dict"}), "from dict")
        self.assertEqual(normalize_ocr_output({"content": "from content"}), "from content")
        self.assertEqual(
            normalize_ocr_output({"text_blocks": [{"text": "line1"}, {"text": "line2"}]}),
            "line1\nline2",
        )

        class CustomObj:
            def __init__(self, text):
                self.text = text

        self.assertEqual(normalize_ocr_output(CustomObj("custom")), "custom")

    def test_detect_images_heuristic(self):
        img = Image.new("RGB", (200, 200), color="white")
        np_img = np.array(img)
        np_img[50:150, 50:150] = [0, 0, 0]
        img = Image.fromarray(np_img)

        regions = detect_images_heuristic(img)
        self.assertGreaterEqual(len(regions), 1)
        found = any(r.type == RegionType.IMAGE for r in regions)
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()

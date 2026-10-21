import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from nexum_ai_common.errors import NexumRuntimeError
from nexum_ai_common.helpers.image import (
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
from nexum_ai_common.models import Region, RegionType


class TestImageHelper(unittest.TestCase):
    def test_extract_dpi(self):
        img = Image.new("RGB", (10, 10))
        self.assertEqual(extract_dpi(img), 72)

        img.info["dpi"] = (300, 300)
        self.assertEqual(extract_dpi(img), 300)

        img.info["dpi"] = (0, 0)
        self.assertEqual(extract_dpi(img), 72)

    def test_deskew_all_white(self):
        gray = np.full((50, 50), 255, dtype=np.uint8)
        result = deskew(gray)
        np.testing.assert_array_equal(result, gray)

    def test_deskew_with_content_and_max_angle(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        gray[40:60, 20:80] = 0

        res1 = deskew(gray)
        self.assertEqual(res1.shape, gray.shape)

        with patch("nexum_ai_common.helpers.image.cv2.minAreaRect", return_value=((50, 50), (20, 60), -60.0)):
            res_steep = deskew(gray, max_angle=10.0)
            np.testing.assert_array_equal(res_steep, gray)

            res_rotated = deskew(gray, max_angle=45.0)
            self.assertEqual(res_rotated.shape, gray.shape)

        with patch("nexum_ai_common.helpers.image.cv2.minAreaRect", return_value=((50, 50), (20, 60), -20.0)):
            res_mild = deskew(gray, max_angle=45.0)
            self.assertEqual(res_mild.shape, gray.shape)

    def test_clahe_and_sharpen(self):
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        c_img = clahe(gray)
        self.assertEqual(c_img.shape, gray.shape)

        s_img = sharpen(gray)
        self.assertEqual(s_img.shape, gray.shape)

    def test_denoise(self):
        gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        self.assertEqual(denoise(gray, "bilateral").shape, gray.shape)
        self.assertEqual(denoise(gray, "median").shape, gray.shape)
        np.testing.assert_array_equal(denoise(gray, "none"), gray)

    def test_binarize(self):
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        self.assertEqual(binarize(gray, "otsu").shape, gray.shape)
        self.assertEqual(binarize(gray, "adaptive").shape, gray.shape)
        np.testing.assert_array_equal(binarize(gray, "unknown"), gray)

    @patch("nexum_ai_common.helpers.image.pytesseract.image_to_osd")
    def test_osd_rotation(self, mock_osd):
        img = Image.new("RGB", (100, 50))

        mock_osd.return_value = "Page number: 0\nRotate: 90\nOrientation confidence: 10\n"
        rotated = osd_rotation(img, osd_min_confidence=5.0)
        self.assertEqual(rotated.size, (50, 100))

        mock_osd.return_value = "Page number: 0\nRotate: 0\nOrientation confidence: 10\n"
        not_rotated = osd_rotation(img, osd_min_confidence=5.0)
        self.assertEqual(not_rotated.size, (100, 50))

        mock_osd.return_value = "Invalid OSD output"
        self.assertIs(osd_rotation(img), img)

        mock_osd.side_effect = RuntimeError("tesseract error")
        self.assertIs(osd_rotation(img), img)

    @patch("nexum_ai_common.helpers.image.pytesseract.image_to_string", return_value="hello\x0c")
    def test_run_ocr(self, mock_to_string):
        gray = np.zeros((20, 20), dtype=np.uint8)
        res = run_ocr(gray, lang="por", oem=1, psm=6, preserve_spaces=True)
        self.assertEqual(res, "hello")
        mock_to_string.assert_called_once_with(
            gray,
            lang="por",
            config="--psm 6 --oem 1 -c preserve_interword_spaces=1",
        )

        mock_to_string.reset_mock()
        res_no_spaces = run_ocr(gray, preserve_spaces=False)
        self.assertEqual(res_no_spaces, "hello")
        mock_to_string.assert_called_once_with(
            gray,
            lang="eng",
            config="--psm 6 --oem 1",
        )

    @patch("nexum_ai_common.helpers.image.pytesseract.image_to_data", return_value={"text": ["a"]})
    def test_run_ocr_data(self, mock_to_data):
        gray = np.zeros((20, 20), dtype=np.uint8)
        res = run_ocr_data(gray, lang="eng", oem=1, psm=3)
        self.assertEqual(res, {"text": ["a"]})
        mock_to_data.assert_called_once()

    def test_validate_max_size(self):
        small = np.zeros((100, 100), dtype=np.uint8)
        validate_max_size(small, max_px=20_000)

        with self.assertRaises(NexumRuntimeError):
            validate_max_size(small, max_px=5_000)

    @patch("nexum_ai_common.helpers.image.pytesseract.image_to_string", return_value=" cropped text \n")
    def test_ocr_region(self, mock_to_string):
        img = Image.new("RGB", (100, 100))
        region = Region(type=RegionType.TEXT, box=(10, 10, 50, 50), score=1.0)
        text = ocr_region(img, region)
        self.assertEqual(text, "cropped text")
        mock_to_string.assert_called_once()

    def test_normalize_ocr_output(self):
        self.assertEqual(normalize_ocr_output("direct string"), "direct string")
        self.assertEqual(normalize_ocr_output({"text": "from text"}), "from text")
        self.assertEqual(normalize_ocr_output({"content": "from content"}), "from content")
        self.assertEqual(
            normalize_ocr_output({"text_blocks": [{"text": "line1"}, "line2"]}),
            "line1\nline2",
        )
        self.assertEqual(normalize_ocr_output({"other": 123}), "")

        obj_with_text = MagicMock(spec=["text"])
        obj_with_text.text = "attr text"
        self.assertEqual(normalize_ocr_output(obj_with_text), "attr text")

        obj_with_content = MagicMock(spec=["content"])
        obj_with_content.content = "attr content"
        self.assertEqual(normalize_ocr_output(obj_with_content), "attr content")

        self.assertEqual(normalize_ocr_output(12345), "12345")

    def test_detect_images_heuristic(self):
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[50:200, 50:200] = 255
        arr[10:20, 10:20] = 255
        pil_img = Image.fromarray(arr)

        regions = detect_images_heuristic(pil_img)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].type, RegionType.IMAGE)
        self.assertEqual(regions[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()

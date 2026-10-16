import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from nexum.document.layout.analyzer import LayoutAnalyzer


class TestLayoutAnalyzer(unittest.TestCase):
    def _create_png_bytes(self, dpi=(72, 72)):
        img = Image.new("RGB", (100, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG", dpi=dpi)
        return buf.getvalue()

    def test_init_detectors_enabled_and_disabled(self):
        analyzer = LayoutAnalyzer(
            enable_table_detector=False,
            enable_block_detector=False,
            enable_image_detector=False,
            enable_flowchart_detector_dl=False,
        )
        self.assertIsNone(analyzer.table_detector)
        self.assertIsNone(analyzer.block_detector)
        self.assertIsNone(analyzer.image_detector)
        self.assertIsNone(analyzer.flowchart_detector)

    def test_preprocess_enabled_and_disabled(self):
        gray = np.full((100, 100), 200, dtype=np.uint8)
        analyzer_no_pre = LayoutAnalyzer(
            enable_preprocess=False,
            enable_table_detector=False,
            enable_block_detector=False,
            enable_image_detector=False,
        )
        res = analyzer_no_pre._preprocess(gray)
        np.testing.assert_array_equal(res, gray)

        analyzer_pre = LayoutAnalyzer(
            enable_preprocess=True,
            enable_binarize=True,
            enable_table_detector=False,
            enable_block_detector=False,
            enable_image_detector=False,
        )
        res2 = analyzer_pre._preprocess(gray)
        self.assertEqual(res2.shape, (100, 100))

    def test_analyze_pdf_and_image(self):
        analyzer = LayoutAnalyzer(
            enable_table_detector=True,
            enable_block_detector=True,
            enable_image_detector=True,
            enable_flowchart_detector_dl=False,
        )
        analyzer.table_detector = MagicMock()
        analyzer.table_detector.detect.return_value = ["tbl"]
        analyzer.block_detector = MagicMock()
        analyzer.block_detector.detect.return_value = ["blk"]
        analyzer.image_detector = MagicMock()
        analyzer.image_detector.detect.return_value = ["img"]

        pdf_res = analyzer.analyze_pdf(self._create_png_bytes(dpi=(72, 72)))
        self.assertEqual(pdf_res["tables"], ["tbl"])
        self.assertEqual(pdf_res["blocks"], ["blk"])
        self.assertEqual(pdf_res["images"], ["img"])

        pil_img = Image.new("RGB", (100, 100), color="white")
        img_res = analyzer.analyze_image(pil_img)
        self.assertEqual(img_res["tables"], ["tbl"])


if __name__ == "__main__":
    unittest.main()

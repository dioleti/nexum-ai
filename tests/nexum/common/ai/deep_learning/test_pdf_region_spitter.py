import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

from nexum.common.ai.deep_learning.pdf_region_spitter import PDFRegionSplitterModel
from nexum.document.models import RegionSplitterConfig, Region, RegionType


class TestPDFRegionSplitterModel(unittest.TestCase):

    @patch("nexum.common.ai.deep_learning.pdf_region_spitter.YOLO")
    def test_load_unload(self, mock_yolo_class):
        mock_yolo = MagicMock()
        mock_yolo_class.return_value = mock_yolo

        config = RegionSplitterConfig(device="cpu")
        model = PDFRegionSplitterModel(config)

        self.assertFalse(model.is_loaded)

        model.load()
        self.assertTrue(model.is_loaded)
        mock_yolo_class.assert_called_once_with("yolov8m-doclaynet.pt")
        mock_yolo.to.assert_called_once_with("cpu")

        model.unload()
        self.assertFalse(model.is_loaded)

    @patch("nexum.common.ai.deep_learning.pdf_region_spitter.detect_images_heuristic")
    @patch("nexum.common.ai.deep_learning.pdf_region_spitter.ocr_region")
    @patch("nexum.common.ai.deep_learning.pdf_region_spitter.YOLO")
    def test_predict(self, mock_yolo_class, mock_ocr_region, mock_detect_images_heuristic):
        mock_yolo = MagicMock()
        mock_yolo_class.return_value = mock_yolo

        # Mock YOLO results
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box_xyxy = MagicMock()
        mock_box_xyxy.tolist.return_value = [10, 10, 50, 50]
        mock_box.xyxy = [mock_box_xyxy]
        mock_box.conf = [0.9]
        mock_box.cls = [0]
        
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "text"}
        
        mock_yolo.predict.return_value = [mock_result]
        
        # Mock heuristic images
        mock_detect_images_heuristic.return_value = [
            Region(type=RegionType.IMAGE, box=(60, 60, 100, 100), score=1.0, text=None)
        ]
        
        # Mock OCR
        mock_ocr_region.return_value = "Mocked OCR text"

        config = RegionSplitterConfig(device="cpu")
        model = PDFRegionSplitterModel(config)
        
        img = Image.new("RGB", (200, 200), color="white")
        regions = model.predict(img)

        self.assertEqual(len(regions), 2)
        
        # Validate text region
        text_region = [r for r in regions if r.type == RegionType.TEXT][0]
        self.assertEqual(text_region.box, (10, 10, 50, 50))
        self.assertEqual(text_region.text, "Mocked OCR text")
        
        # Validate image region
        img_region = [r for r in regions if r.type == RegionType.IMAGE][0]
        self.assertEqual(img_region.box, (60, 60, 100, 100))

    def test_overlaps(self):
        r1 = Region(type=RegionType.TEXT, box=(0, 0, 10, 10), score=1.0, text=None)
        r2 = Region(type=RegionType.IMAGE, box=(5, 5, 15, 15), score=1.0, text=None)
        r3 = Region(type=RegionType.TABLE, box=(20, 20, 30, 30), score=1.0, text=None)

        self.assertTrue(PDFRegionSplitterModel._overlaps(r1, r2))
        self.assertFalse(PDFRegionSplitterModel._overlaps(r1, r3))
        self.assertFalse(PDFRegionSplitterModel._overlaps(r2, r3))

if __name__ == "__main__":
    unittest.main()

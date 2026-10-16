import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from nexum.detection.region import RegionDetectorDL
from nexum.document.models import Region, RegionSplitterConfig, RegionType


class TestRegionDetectorDL(unittest.TestCase):
    @patch("nexum.detection.region.YOLO")
    @patch("nexum.detection.region.hf_hub_download")
    def test_load_unload(self, mock_hf_download, mock_yolo_class):
        mock_hf_download.return_value = "/fake/path/yolov8m-doclaynet.pt"
        mock_yolo = MagicMock()
        mock_yolo_class.return_value = mock_yolo

        config = RegionSplitterConfig(device="cpu")
        model = RegionDetectorDL(config)

        self.assertFalse(model.is_loaded)

        model.load()
        self.assertTrue(model.is_loaded)
        mock_hf_download.assert_called_once_with(
            repo_id="Dioleti/olov8m-doclaynet",
            filename="yolov8m-doclaynet.pt",
        )
        mock_yolo_class.assert_called_once_with("/fake/path/yolov8m-doclaynet.pt")
        mock_yolo.to.assert_called_once_with("cpu")

        model.unload()
        self.assertFalse(model.is_loaded)

    @patch("nexum.detection.region.detect_images_heuristic")
    @patch("nexum.detection.region.ocr_region")
    @patch("nexum.detection.region.YOLO")
    @patch("nexum.detection.region.hf_hub_download")
    def test_predict(
        self,
        mock_hf_download,
        mock_yolo_class,
        mock_ocr_region,
        mock_detect_images_heuristic,
    ):
        mock_hf_download.return_value = "/fake/path/yolov8m-doclaynet.pt"
        mock_yolo = MagicMock()
        mock_yolo_class.return_value = mock_yolo

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

        mock_detect_images_heuristic.return_value = [
            Region(type=RegionType.IMAGE, box=(60, 60, 100, 100), score=1.0, text=None)
        ]

        mock_ocr_region.return_value = "Mocked OCR text"

        config = RegionSplitterConfig(device="cpu")
        model = RegionDetectorDL(config)

        img = Image.new("RGB", (200, 200), color="white")
        regions = model.predict(img)

        self.assertEqual(len(regions), 2)

        text_region = [r for r in regions if r.type == RegionType.TEXT][0]
        self.assertEqual(text_region.box, (10, 10, 50, 50))
        self.assertEqual(text_region.text, "Mocked OCR text")

        img_region = [r for r in regions if r.type == RegionType.IMAGE][0]
        self.assertEqual(img_region.box, (60, 60, 100, 100))

    def test_overlaps_and_mapping(self):
        r1 = Region(type=RegionType.TEXT, box=(0, 0, 10, 10), score=1.0, text=None)
        r2 = Region(type=RegionType.IMAGE, box=(5, 5, 15, 15), score=1.0, text=None)
        r3 = Region(type=RegionType.TABLE, box=(20, 20, 30, 30), score=1.0, text=None)

        self.assertTrue(RegionDetectorDL._overlaps(r1, r2))
        self.assertFalse(RegionDetectorDL._overlaps(r1, r3))
        self.assertFalse(RegionDetectorDL._overlaps(r2, r3))

        self.assertEqual(RegionDetectorDL._map_label_to_region_type("table"), RegionType.TABLE)
        self.assertEqual(RegionDetectorDL._map_label_to_region_type("figure"), RegionType.IMAGE)
        self.assertEqual(RegionDetectorDL._map_label_to_region_type("title"), RegionType.TEXT)
        self.assertIsNone(RegionDetectorDL._map_label_to_region_type("unknown"))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from nexum.detection.table import TableDetectorDL
from nexum.document.models import TableDetectionConfig, TableDetectionInput


class TestTableDetectorDL(unittest.TestCase):
    @patch("nexum.detection.table.TableTransformerForObjectDetection.from_pretrained")
    @patch("nexum.detection.table.AutoImageProcessor.from_pretrained")
    def test_load_and_unload(self, mock_proc_cls, mock_model_cls):
        mock_proc_cls.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.return_value = mock_model

        config = TableDetectionConfig(device="cpu")
        detector = TableDetectorDL(config)
        self.assertFalse(detector.is_loaded)

        detector.load()
        self.assertTrue(detector.is_loaded)
        mock_proc_cls.assert_called_once_with(config.model_name)
        mock_model_cls.assert_called_once_with(config.model_name)
        mock_model.eval.assert_called_once()

        detector.unload()
        self.assertFalse(detector.is_loaded)

    def test_predict_and_batch(self):
        config = TableDetectionConfig(device="cpu")
        detector = TableDetectorDL(config)

        mock_proc = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = {}
        mock_proc.return_value = mock_batch
        mock_proc.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0]]),
            }
        ]

        mock_model = MagicMock()
        mock_model.config.id2label = {0: "table"}

        detector.processor = mock_proc
        detector.model = mock_model

        img = Image.new("RGB", (300, 300), color="white")
        res = detector.predict(TableDetectionInput(image=img, threshold=0.8))

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].label, "table")
        self.assertEqual(res[0].score, 0.95)
        self.assertEqual(res[0].box.as_tuple(), (10.0, 20.0, 100.0, 200.0))

    def test_predict_batch_empty(self):
        detector = TableDetectorDL(TableDetectionConfig(device="cpu"))
        self.assertEqual(detector.predict_batch([]), [])


if __name__ == "__main__":
    unittest.main()

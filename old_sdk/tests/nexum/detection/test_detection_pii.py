import unittest
from unittest.mock import MagicMock, patch

import torch

from nexum.common.models import PIIDetectorConfig, PiiDetectionInput
from nexum.detection.pii import PIIDetectorDL


class TestPIIDetectorDL(unittest.TestCase):
    @patch("nexum.detection.pii.pipeline")
    @patch("nexum.detection.pii.AutoModelForTokenClassification.from_pretrained")
    @patch("nexum.detection.pii.AutoTokenizer.from_pretrained")
    def test_load_and_unload(self, mock_tok, mock_model, mock_pipeline):
        mock_pipeline.return_value = MagicMock()
        detector = PIIDetectorDL(
            config=PIIDetectorConfig(),
            device=torch.device("cpu"),
        )
        self.assertFalse(detector.is_loaded)

        detector.load()
        self.assertTrue(detector.is_loaded)
        mock_tok.assert_called_once()
        mock_model.assert_called_once()
        mock_pipeline.assert_called_once()

        detector.unload()
        self.assertFalse(detector.is_loaded)
        self.assertIsNone(detector.ner_pipeline)

    def test_predict_regex_hit(self):
        detector = PIIDetectorDL(device=torch.device("cpu"))
        inp = PiiDetectionInput(
            column_name="cpf",
            samples=["123.456.789-00", "987.654.321-11"],
            threshold=0.5,
        )
        res = detector.predict(inp)
        self.assertTrue(res.is_pii)
        self.assertEqual(res.confidence, 1.0)
        self.assertEqual(res.detected_entities, {"REGEX_PATTERN": 2})

    def test_predict_empty_or_short_samples(self):
        detector = PIIDetectorDL(device=torch.device("cpu"))
        res_empty = detector.predict(PiiDetectionInput(column_name="a", samples=[]))
        self.assertFalse(res_empty.is_pii)
        self.assertEqual(res_empty.confidence, 0.0)

        res_short = detector.predict(PiiDetectionInput(column_name="b", samples=["a", "12"]))
        self.assertFalse(res_short.is_pii)
        self.assertEqual(res_short.confidence, 0.0)

    def test_predict_batch_with_ner_pipeline(self):
        detector = PIIDetectorDL(device=torch.device("cpu"))
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            [{"entity_group": "PER", "score": 0.95}],
            [{"entity_group": "ORG", "score": 0.90}],
        ]
        detector.ner_pipeline = mock_pipe

        inp = PiiDetectionInput(
            column_name="nome",
            samples=["Maria Oliveira", "Empresa X"],
            threshold=0.5,
        )
        res = detector.predict(inp)
        self.assertTrue(res.is_pii)
        self.assertEqual(res.confidence, 0.5)
        self.assertEqual(res.detected_entities, {"PER": 1})

    def test_predict_batch_empty(self):
        detector = PIIDetectorDL(device=torch.device("cpu"))
        self.assertEqual(detector.predict_batch([]), [])


if __name__ == "__main__":
    unittest.main()

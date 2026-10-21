import unittest
from unittest.mock import MagicMock, patch
import torch
import tests

from nexum_ai_machine_learning.common.models import (
    PiiDetectionInput,
    PIIDetectorConfig,
)
from nexum_ai_machine_learning.detectors.pii_detector import (
    PIIColumnDetector,
    PIIDetector,
)


class TestPIIDetector(unittest.TestCase):
    @patch("nexum_ai_machine_learning.detectors.pii_detector.resolve", return_value=torch.device("cpu"))
    def test_init_defaults(self, mock_resolve):
        detector = PIIDetector()
        self.assertEqual(detector.model_name, "Dioleti/xlm-roberta-base-ner-hrl")
        self.assertEqual(detector.device, torch.device("cpu"))
        self.assertEqual(detector.batch_size, 32)
        self.assertFalse(detector.is_loaded)
        mock_resolve.assert_called_once()

    def test_init_custom_params(self):
        cfg = PIIDetectorConfig(
            model_name="custom/ner",
            score_threshold=0.80,
            max_chunk_chars=500,
        )
        dev = torch.device("cuda")
        detector = PIIDetector(config=cfg, device=dev, batch_size=16)
        self.assertEqual(detector.model_name, "custom/ner")
        self.assertEqual(detector.score_threshold, 0.80)
        self.assertEqual(detector.max_chunk_chars, 500)
        self.assertEqual(detector.device, dev)
        self.assertEqual(detector.batch_size, 16)

    @patch("nexum_ai_machine_learning.detectors.pii_detector.pipeline")
    @patch("nexum_ai_machine_learning.detectors.pii_detector.AutoModelForTokenClassification.from_pretrained")
    @patch("nexum_ai_machine_learning.detectors.pii_detector.AutoTokenizer.from_pretrained")
    @patch("nexum_ai_machine_learning.detectors.pii_detector.torch.set_num_threads")
    @patch("nexum_ai_machine_learning.detectors.pii_detector.torch.get_num_threads", return_value=8)
    def test_load_cpu_and_idempotency(
        self,
        mock_get_threads,
        mock_set_threads,
        mock_tokenizer,
        mock_model,
        mock_pipeline,
    ):
        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst
        mock_tok_inst = MagicMock()
        mock_tokenizer.return_value = mock_tok_inst
        mock_pipe_inst = MagicMock()
        mock_pipeline.return_value = mock_pipe_inst

        detector = PIIDetector(device=torch.device("cpu"), batch_size=8)
        detector.load()

        self.assertTrue(detector.is_loaded)
        mock_tokenizer.assert_called_once_with(detector.model_name, use_fast=True)
        mock_model.assert_called_once_with(detector.model_name)
        mock_model_inst.eval.assert_called_once()
        mock_set_threads.assert_called_once_with(4)
        mock_pipeline.assert_called_once_with(
            task="token-classification",
            model=mock_model_inst,
            tokenizer=mock_tok_inst,
            aggregation_strategy="simple",
            device=-1,
            batch_size=8,
        )

        detector.load()
        mock_tokenizer.assert_called_once()

    @patch("nexum_ai_machine_learning.detectors.pii_detector.pipeline")
    @patch("nexum_ai_machine_learning.detectors.pii_detector.AutoModelForTokenClassification.from_pretrained")
    @patch("nexum_ai_machine_learning.detectors.pii_detector.AutoTokenizer.from_pretrained")
    def test_load_cuda(self, mock_tokenizer, mock_model, mock_pipeline):
        mock_model_inst = MagicMock()
        mock_model.return_value = mock_model_inst
        mock_tok_inst = MagicMock()
        mock_tokenizer.return_value = mock_tok_inst
        mock_pipe_inst = MagicMock()
        mock_pipeline.return_value = mock_pipe_inst

        detector = PIIDetector(device=torch.device("cuda"), batch_size=16)
        detector.load()

        mock_model_inst.eval.assert_not_called()
        mock_pipeline.assert_called_once_with(
            task="token-classification",
            model=mock_model_inst,
            tokenizer=mock_tok_inst,
            aggregation_strategy="simple",
            device=0,
            batch_size=16,
        )

    @patch("nexum_ai_machine_learning.detectors.pii_detector.torch.cuda.empty_cache")
    @patch("nexum_ai_machine_learning.detectors.pii_detector.torch.cuda.is_available", return_value=True)
    def test_unload_cuda(self, mock_cuda_avail, mock_empty_cache):
        detector = PIIDetector(device=torch.device("cuda"))
        detector.ner_pipeline = MagicMock()
        detector.model = MagicMock()
        detector.tokenizer = MagicMock()

        detector.unload()

        self.assertFalse(detector.is_loaded)
        self.assertIsNone(detector.model)
        self.assertIsNone(detector.tokenizer)
        mock_empty_cache.assert_called_once()

    def test_unload_mps(self):
        detector = PIIDetector(device=torch.device("mps"))
        detector.ner_pipeline = MagicMock()
        with patch.object(torch.mps, "empty_cache", create=True) as mock_mps_cache:
            detector.unload()
            self.assertFalse(detector.is_loaded)
            mock_mps_cache.assert_called_once()

    def test_check_regex_patterns(self):
        samples = [
            "123.456.789-00",
            "12.345.678/0001-99",
            "user@example.com",
            "+55 (11) 98765-4321",
            "not a pii pattern",
        ]
        hits = PIIDetector._check_regex(samples)
        self.assertEqual(hits, 4)

    def test_predict_string_empty_and_short(self):
        detector = PIIDetector(device=torch.device("cpu"))
        self.assertEqual(detector.predict_batch([]), [])
        res = detector.predict_batch(["   ", "ab"])
        self.assertEqual(len(res), 2)
        self.assertFalse(res[0].is_pii)
        self.assertFalse(res[1].is_pii)

    def test_predict_string_exact_regex(self):
        detector = PIIDetector(device=torch.device("cpu"))
        res = detector.predict("123.456.789-00")
        self.assertTrue(res.is_pii)
        self.assertEqual(res.confidence, 1.0)
        self.assertEqual(res.detected_entities["CPF"], 1)
        self.assertEqual(res.detected_entities["REGEX_PATTERN"], 1)

    def test_predict_string_long_text_with_chunking_and_mixed_entities(self):
        cfg = PIIDetectorConfig(max_chunk_chars=25)
        detector = PIIDetector(config=cfg, device=torch.device("cpu"))
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            [{"entity_group": "PER", "score": 0.92}],
            [{"entity_group": "LOC", "score": 0.85}],
            [],
        ]
        detector.ner_pipeline = mock_pipe

        long_text = "Maria Silva mora em São Paulo e usa user@example.com"
        result = detector.predict(long_text)

        self.assertTrue(result.is_pii)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.detected_entities["PER"], 1)
        self.assertEqual(result.detected_entities["LOC"], 1)
        self.assertEqual(result.detected_entities["EMAIL"], 1)
        self.assertGreater(len(mock_pipe.call_args[0][0]), 1)

    def test_predict_string_ner_only_and_no_pii(self):
        detector = PIIDetector(device=torch.device("cpu"))
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            [{"entity_group": "PER", "score": 0.88}],
            [{"entity_group": "ORG", "score": 0.95}],
        ]

        def fake_load():
            detector.ner_pipeline = mock_pipe

        with patch.object(detector, "load", side_effect=fake_load) as mock_load:
            results = detector.predict_batch(["Maria Silva", "Empresa X"])
            mock_load.assert_called_once()
            self.assertTrue(results[0].is_pii)
            self.assertEqual(results[0].confidence, 0.88)
            self.assertEqual(results[0].detected_entities, {"PER": 1})

            self.assertFalse(results[1].is_pii)
            self.assertEqual(results[1].confidence, 0.0)
            self.assertEqual(results[1].detected_entities, {})


class TestPIIColumnDetector(unittest.TestCase):
    def test_lifecycle_delegation(self):
        mock_base = MagicMock(spec=PIIDetector)
        mock_base.is_loaded = False
        col_detector = PIIColumnDetector(detector=mock_base)

        self.assertFalse(col_detector.is_loaded)
        col_detector.load()
        mock_base.load.assert_called_once()
        col_detector.unload()
        mock_base.unload.assert_called_once()

    def test_predict_batch_empty_inputs_and_samples(self):
        col_detector = PIIColumnDetector(device=torch.device("cpu"))
        self.assertEqual(col_detector.predict_batch([]), [])

        inputs = [
            PiiDetectionInput(column_name="empty", samples=[]),
            PiiDetectionInput(column_name="short", samples=["a", "  ", "ab"]),
        ]
        results = col_detector.predict_batch(inputs)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].is_pii)
        self.assertEqual(results[0].confidence, 0.0)
        self.assertFalse(results[1].is_pii)
        self.assertEqual(results[1].confidence, 0.0)

    def test_predict_single_regex_hit(self):
        col_detector = PIIColumnDetector(device=torch.device("cpu"))
        inp = PiiDetectionInput(
            column_name="cpf_col",
            samples=["123.456.789-00", "111.222.333-44", "invalid"],
            threshold=0.5,
        )
        result = col_detector.predict(inp)
        self.assertTrue(result.is_pii)
        self.assertEqual(result.confidence, round(2 / 3, 4))
        self.assertEqual(result.detected_entities, {"REGEX_PATTERN": 2})

    def test_predict_batch_ner_pipeline_auto_loads_and_classifies(self):
        base_detector = PIIDetector(device=torch.device("cpu"))
        col_detector = PIIColumnDetector(detector=base_detector)

        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            [{"entity_group": "PER", "score": 0.95}, {"entity_group": "LOC", "score": 0.80}],
            [{"entity_group": "PER", "score": 0.50}],
            [{"entity_group": "ORG", "score": 0.99}],
        ]

        def fake_load():
            base_detector.ner_pipeline = mock_pipe

        with patch.object(base_detector, "load", side_effect=fake_load) as mock_load:
            inputs = [
                PiiDetectionInput(
                    column_name="names",
                    samples=["Maria Silva", "João Souza"],
                    threshold=0.5,
                ),
                PiiDetectionInput(
                    column_name="orgs",
                    samples=["Empresa X"],
                    threshold=0.5,
                ),
            ]
            results = col_detector.predict_batch(inputs)

            mock_load.assert_called_once()
            mock_pipe.assert_called_once_with(["Maria Silva", "João Souza", "Empresa X"])

            self.assertTrue(results[0].is_pii)
            self.assertEqual(results[0].confidence, 0.5)
            self.assertEqual(results[0].detected_entities, {"PER": 1, "LOC": 1})

            self.assertFalse(results[1].is_pii)
            self.assertEqual(results[1].confidence, 0.0)
            self.assertEqual(results[1].detected_entities, {})


if __name__ == "__main__":
    unittest.main()

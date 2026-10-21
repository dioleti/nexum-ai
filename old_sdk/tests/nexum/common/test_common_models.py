import unittest

from nexum.common.models import (
    CodeChunk,
    CodeChunkConfig,
    CodeDocument,
    ColumnSemanticConfig,
    ColumnSemanticInput,
    ColumnSemanticResult,
    PIIDetectorConfig,
    PiiDetectionInput,
    PiiDetectionResult,
)


class TestCommonModels(unittest.TestCase):
    def test_code_document(self):
        doc = CodeDocument(code="def foo(): pass", language="python", file_path="main.py")
        self.assertEqual(doc.code, "def foo(): pass")
        self.assertEqual(doc.language, "python")
        self.assertEqual(doc.file_path, "main.py")

    def test_code_chunk_defaults(self):
        chunk = CodeChunk(
            name="foo",
            chunk_type="function",
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        self.assertEqual(chunk.name, "foo")
        self.assertEqual(chunk.chunk_type, "function")
        self.assertIsNone(chunk.parent_scope)
        self.assertIsNone(chunk.embedding)
        self.assertEqual(chunk.token_count, 0)
        self.assertEqual(chunk.metadata, {})

    def test_pii_detection_input_defaults(self):
        inp = PiiDetectionInput(column_name="cpf", samples=["123.456.789-00"])
        self.assertEqual(inp.column_name, "cpf")
        self.assertEqual(inp.samples, ["123.456.789-00"])
        self.assertEqual(inp.threshold, 0.25)

    def test_pii_detection_result_defaults(self):
        res = PiiDetectionResult(is_pii=True, confidence=0.95)
        self.assertTrue(res.is_pii)
        self.assertEqual(res.confidence, 0.95)
        self.assertEqual(res.detected_entities, {})

    def test_column_semantic_result(self):
        res = ColumnSemanticResult(semantic_type="email", is_pii=True, is_sensitive=False)
        self.assertEqual(res.semantic_type, "email")
        self.assertTrue(res.is_pii)
        self.assertFalse(res.is_sensitive)

    def test_column_semantic_input_defaults(self):
        inp = ColumnSemanticInput(columns=["a"], data=[[1]])
        self.assertEqual(inp.columns, ["a"])
        self.assertEqual(inp.data, [[1]])
        self.assertEqual(inp.sample_size, 15)
        self.assertEqual(inp.max_values_per_col, 8)

    def test_pii_detector_config_defaults(self):
        cfg = PIIDetectorConfig()
        self.assertEqual(cfg.model_name, "Dioleti/xlm-roberta-base-ner-hrl")

    def test_code_chunk_config_defaults(self):
        cfg = CodeChunkConfig()
        self.assertEqual(cfg.model_name, "Dioleti/bge-m3")

    def test_column_semantic_config_defaults(self):
        cfg = ColumnSemanticConfig()
        self.assertEqual(cfg.model_name, "Dioleti/paraphrase-multilingual-MiniLM-L12-v2")
        self.assertIsNone(cfg.device)
        self.assertIn("health_or_medical", cfg.sensitive_types)
        self.assertIn("email", cfg.pii_types)
        self.assertIn("identifier", cfg.profiles)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nexum.chunking.code import CodeChunkerDL
from nexum.common.models import CodeChunkConfig, CodeDocument


class TestCodeChunkerDL(unittest.TestCase):
    @patch("nexum.chunking.code.SentenceTransformer")
    @patch("nexum.chunking.code.resolve")
    def test_load_and_unload(self, mock_resolve, mock_st_cls):
        mock_resolve.return_value = torch.device("cpu")
        mock_embedder = MagicMock()
        mock_st_cls.return_value = mock_embedder

        chunker = CodeChunkerDL(
            config=CodeChunkConfig(model_name="test-model"),
            compute_embeddings=True,
            max_token_budget=4096,
            cpu_threads=2,
        )
        self.assertFalse(chunker.is_loaded)

        chunker.load()
        self.assertTrue(chunker.is_loaded)
        mock_st_cls.assert_called_once_with(
            "test-model",
            device="cpu",
            trust_remote_code=False,
        )
        self.assertEqual(mock_embedder.max_seq_length, 4096)
        self.assertIn("python", chunker._parsers)
        self.assertIn("kotlin", chunker._parsers)

        chunker.unload()
        self.assertFalse(chunker.is_loaded)
        self.assertIsNone(chunker._embedder)
        self.assertEqual(chunker._parsers, {})

    def test_predict_not_loaded_raises(self):
        chunker = CodeChunkerDL(compute_embeddings=False)
        doc = CodeDocument(code="def a(): pass", language="python")
        with self.assertRaises(RuntimeError):
            chunker.predict(doc)

    @patch("nexum.chunking.code.resolve")
    def test_predict_unsupported_language_raises(self, mock_resolve):
        mock_resolve.return_value = torch.device("cpu")
        chunker = CodeChunkerDL(compute_embeddings=False)
        chunker.load()

        doc = CodeDocument(code="print('hi')", language="cobol")
        with self.assertRaises(ValueError):
            chunker.predict(doc)

    @patch("nexum.chunking.code.SentenceTransformer")
    @patch("nexum.chunking.code.resolve")
    def test_predict_python_functions_and_classes(self, mock_resolve, mock_st_cls):
        mock_resolve.return_value = torch.device("cpu")
        mock_embedder = MagicMock()
        mock_embedder.tokenizer.encode.side_effect = lambda text, **kw: [1] * (len(text) // 4)
        mock_embedder.encode.side_effect = lambda texts, **kw: np.ones((len(texts), 4), dtype=np.float32)
        mock_st_cls.return_value = mock_embedder

        chunker = CodeChunkerDL(compute_embeddings=True, max_token_budget=10)
        chunker.load()

        code = (
            "class MyService:\n"
            '    """Service docstring."""\n'
            "    x = 1\n\n"
            "    @staticmethod\n"
            "    def helper(val: int) -> int:\n"
            "        return val + 1\n\n"
            "    def process(self):\n"
            "        return self.helper(self.x)\n"
        )
        doc = CodeDocument(code=code, language="python", file_path="service.py")
        chunks = chunker.predict(doc)

        self.assertGreaterEqual(len(chunks), 3)
        class_chunk = next(c for c in chunks if c.chunk_type == "class")
        self.assertEqual(class_chunk.name, "MyService")
        self.assertIn("Service docstring", class_chunk.content)
        self.assertIn("...", class_chunk.content)
        self.assertEqual(class_chunk.metadata["file_path"], "service.py")
        self.assertEqual(class_chunk.embedding, [1.0, 1.0, 1.0, 1.0])
        self.assertIn("warning", class_chunk.metadata)

        func_names = [c.name for c in chunks if c.chunk_type == "function"]
        self.assertIn("helper", func_names)
        self.assertIn("process", func_names)

    @patch("nexum.chunking.code.resolve")
    def test_predict_fallback_module_chunk(self, mock_resolve):
        mock_resolve.return_value = torch.device("cpu")
        chunker = CodeChunkerDL(compute_embeddings=False)
        chunker.load()

        doc = CodeDocument(code="x = 10\ny = 20\n", language="python", file_path="script.py")
        chunks = chunker.predict(doc)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].name, "root")
        self.assertEqual(chunks[0].chunk_type, "module")
        self.assertIsNone(chunks[0].embedding)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock

from nexum.document.models import CSVReaderConfig, Document, Parsed, Table
from nexum.document.reader.csv_reader import CSVReader


class DummyLoader:
    def __init__(self, data: bytes):
        self.data = data

    def load(self) -> bytes:
        return self.data

    def stream(self):
        yield self.data


class TestCSVReader(unittest.TestCase):
    def _mock_parser(self):
        parser = MagicMock()
        parser.parse.return_value = Parsed(
            metadata={"encoding": "utf-8"},
            tables=[Table(columns=["a", "b"], rows=[["1", "2"]])],
            rows=[["a", "b"], ["1", "2"]],
        )
        return parser

    def test_read_basic(self):
        loader = DummyLoader(b"a,b\n1,2")
        reader = CSVReader(loader, CSVReaderConfig(infer_types=False), parser=self._mock_parser())
        doc = reader.read()
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.content, "a,b\n1,2")
        self.assertEqual(doc.tables[0].columns, ["a", "b"])
        self.assertEqual(doc.tables[0].rows, [["1", "2"]])

    def test_read_streaming_basic(self):
        loader = DummyLoader(b"a,b\n1,2")
        reader = CSVReader(loader, CSVReaderConfig(infer_types=False), parser=self._mock_parser())
        docs = list(reader.read_streaming())
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.content, "a,b\n1,2")

    def test_detect_encoding_utf8(self):
        loader = DummyLoader("a,b\n1,2".encode("utf-8"))
        reader = CSVReader(loader, CSVReaderConfig(), parser=self._mock_parser())
        encoding = reader._detect_encoding(loader.load())
        self.assertEqual(encoding, "utf-8")

    def test_detect_encoding_latin1(self):
        loader = DummyLoader("áéí".encode("latin-1"))
        reader = CSVReader(loader, CSVReaderConfig(encoding=None), parser=self._mock_parser())
        encoding = reader._detect_encoding(loader.load())
        self.assertEqual(encoding, "latin-1")

    def test_build_document(self):
        loader = DummyLoader(b"a,b\n1,2")
        reader = CSVReader(loader, CSVReaderConfig(infer_types=False), parser=self._mock_parser())
        doc = reader._build_document(b"a,b\n1,2", "a,b\n1,2")
        self.assertEqual(doc.content, "a,b\n1,2")
        self.assertEqual(doc.tables[0].columns, ["a", "b"])

    def test_read_invalid_bytes(self):
        loader = DummyLoader(b"\xff\xfe\xfa")
        reader = CSVReader(loader, CSVReaderConfig(), parser=self._mock_parser())
        doc = reader.read()
        self.assertIsInstance(doc, Document)
        self.assertIsInstance(doc.content, str)


if __name__ == "__main__":
    unittest.main()

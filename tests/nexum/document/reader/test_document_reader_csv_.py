import unittest

from nexum.document.models import CSVReaderConfig, Document
from nexum.document.reader.csv import CSVReader


class DummyLoader:
    def __init__(self, data: bytes):
        self.data = data

    def load(self) -> bytes:
        return self.data

    def stream(self):
        yield self.data


class TestCSVReader(unittest.TestCase):
    def test_read_basic(self):
        loader = DummyLoader(b"a,b\n1,2")
        reader = CSVReader(loader, CSVReaderConfig(infer_types=False))
        doc = reader.read()
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.content, "a,b\n1,2")
        self.assertEqual(doc.tables[0].columns, ["a", "b"])
        self.assertEqual(doc.tables[0].rows, [["1", "2"]])

    def test_read_streaming_basic(self):
        loader = DummyLoader(b"a,b\n1,2")
        reader = CSVReader(loader, CSVReaderConfig(infer_types=False))
        docs = list(reader.read_streaming())
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.content, "a,b\n1,2")

    def test_detect_encoding_utf8(self):
        loader = DummyLoader("a,b\n1,2".encode("utf-8"))
        reader = CSVReader(loader, CSVReaderConfig())
        encoding = reader._detect_encoding(loader.load())
        self.assertEqual(encoding, "utf-8")

    def test_detect_encoding_latin1(self):
        loader = DummyLoader("áéí".encode("latin-1"))
        reader = CSVReader(loader, CSVReaderConfig())
        encoding = reader._detect_encoding(loader.load())
        self.assertEqual(encoding, "latin-1")

    def test_build_document(self):
        loader = DummyLoader(b"a,b\n1,2")
        reader = CSVReader(loader, CSVReaderConfig(infer_types=False))
        doc = reader._build_document(b"a,b\n1,2", "a,b\n1,2")
        self.assertEqual(doc.content, "a,b\n1,2")
        self.assertEqual(doc.tables[0].columns, ["a", "b"])

    def test_read_invalid_bytes(self):
        loader = DummyLoader(b"\xff\xfe\xfa")
        reader = CSVReader(loader, CSVReaderConfig())
        doc = reader.read()
        self.assertIsInstance(doc, Document)
        self.assertIsInstance(doc.content, str)


if __name__ == "__main__":
    unittest.main()

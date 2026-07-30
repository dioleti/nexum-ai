import io
import unittest

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from nexum.document.models import Document, ImageReaderConfig, Table
from nexum.document.parser.image import ImageOCRParser
from nexum.document.reader.image import ImageReader


class DummyLoader:
    def __init__(self, data: bytes):
        self.data = data

    def load(self) -> bytes:
        return self.data

    def stream(self):
        yield self.data


class DummyOCR(ImageOCRParser):
    def parse(self, image_bytes: bytes) -> str:
        return "col1|col2\n1|2\n3|4"


class TestImageReader(unittest.TestCase):
    def _png_bytes(self):
        img = Image.new("RGB", (200, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_read_batch_basic(self):
        loader = DummyLoader(self._png_bytes())
        reader = ImageReader(loader, ImageReaderConfig(), DummyOCR(ImageReaderConfig().ocr_config))
        doc = reader.read()
        self.assertIsInstance(doc, Document)
        self.assertTrue(isinstance(doc.content, str))

    def test_read_streaming_basic(self):
        loader = DummyLoader(self._png_bytes())
        reader = ImageReader(loader, ImageReaderConfig(), DummyOCR(ImageReaderConfig().ocr_config))
        docs = list(reader.read_streaming())
        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], Document)

    def test_extract_metadata(self):
        loader = DummyLoader(self._png_bytes())
        reader = ImageReader(loader, ImageReaderConfig(), DummyOCR(ImageReaderConfig().ocr_config))
        metadata = reader._extract_metadata(loader.load())
        self.assertIsInstance(metadata, dict)

    def test_extract_embedded_text_none(self):
        loader = DummyLoader(self._png_bytes())
        reader = ImageReader(loader, ImageReaderConfig(), DummyOCR(ImageReaderConfig().ocr_config))
        embedded = reader._extract_embedded_text(loader.load())
        self.assertIsNone(embedded)

    def test_extract_tables_heuristic(self):
        loader = DummyLoader(self._png_bytes())
        reader = ImageReader(loader, ImageReaderConfig(), DummyOCR(ImageReaderConfig().ocr_config))
        tables = reader._extract_tables(loader.load(), "col1|col2\n1|2\n3|4")
        self.assertTrue(len(tables) >= 1)
        self.assertIsInstance(tables[0], Table)

    def test_build_document(self):
        loader = DummyLoader(self._png_bytes())
        reader = ImageReader(loader, ImageReaderConfig(), DummyOCR(ImageReaderConfig().ocr_config))
        doc = reader._build_document("col1|col2\n1|2", {}, loader.load())
        self.assertIsInstance(doc, Document)
        self.assertTrue(len(doc.tables) >= 1)

    def test_read_with_embedded_text(self):
        img = Image.new("RGB", (200, 100), color="white")
        buf = io.BytesIO()
        info = PngInfo()
        info.add_text("text", "hello world")
        img.save(buf, format="PNG", pnginfo=info)
        loader = DummyLoader(buf.getvalue())
        reader = ImageReader(loader, ImageReaderConfig(), DummyOCR(ImageReaderConfig().ocr_config))
        doc = reader.read()
        self.assertEqual(doc.content, "hello world")

    def test_read_invalid_bytes(self):
        loader = DummyLoader(b"\x00\x01\x02")
        reader = ImageReader(loader, ImageReaderConfig(), DummyOCR(ImageReaderConfig().ocr_config))
        doc = reader.read()
        self.assertIsInstance(doc, Document)


if __name__ == "__main__":
    unittest.main()

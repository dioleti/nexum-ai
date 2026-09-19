import io
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image

from pypdf.generic import DictionaryObject

from nexum.common.errors import NexumRuntimeError
from nexum.document.models import PDFReaderConfig, Table, Document, VisualRegion
from nexum.document.parser.pdf import PDFOCRParser
from nexum.document.reader.pdf_reader import PDFReader


class DummyLoader:
    def __init__(self, data: bytes, should_fail: bool = False):
        self.data = data
        self.should_fail = should_fail

    def load(self) -> bytes:
        if self.should_fail:
            raise RuntimeError("Loader error")
        return self.data

    def stream(self):
        if self.should_fail:
            raise RuntimeError("Stream error")
        yield self.data


class DummyOCR(PDFOCRParser):
    def parse(self, page_bytes: bytes) -> str:
        return "col1|col2\n1|2\n3|4"


class TestPDFReader(unittest.TestCase):
    def _pdf_bytes(self):
        return b"%PDF-1.4 fake pdf bytes"

    def _sample_image(self):
        img = Image.new("RGB", (100, 100), color="white")
        return img

    def test_get_bytes_success(self):
        reader = PDFReader(DummyLoader(self._pdf_bytes()))
        data = reader._get_bytes()
        self.assertEqual(data, self._pdf_bytes())

    def test_get_bytes_failure(self):
        reader = PDFReader(DummyLoader(b"", should_fail=True))
        with self.assertRaises(NexumRuntimeError):
            reader._get_bytes()

    @patch("nexum.document.reader.pdf_reader.PdfReader")
    def test_parse_digital_pdf(self, mock_pdf_reader_cls):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Digital text"

        mock_img = MagicMock()
        mock_img.get_data.return_value = b"raw_image_data"
        mock_img.get.side_effect = lambda k: "/Image" if k == "/Subtype" else None

        xobjects = DictionaryObject({"img1": mock_img})
        xobject_ref = MagicMock()
        xobject_ref.get_object.return_value = xobjects

        resources = DictionaryObject({"/XObject": xobject_ref})
        mock_page.get.return_value = resources

        mock_pdf_instance = MagicMock()
        mock_pdf_instance.metadata = {"Title": "Test Doc"}
        mock_pdf_instance.pages = [mock_page]
        mock_pdf_reader_cls.return_value = mock_pdf_instance

        cfg = PDFReaderConfig(enable_metadata=True, enable_images=True)
        reader = PDFReader(DummyLoader(self._pdf_bytes()), cfg)

        metadata, pages_text, images = reader._parse_digital_pdf(self._pdf_bytes())
        self.assertEqual(metadata, {"Title": "Test Doc"})
        self.assertEqual(pages_text, ["Digital text"])
        self.assertEqual(images, [b"raw_image_data"])

    def test_parse_visual_tables(self):
        reader = PDFReader(DummyLoader(self._pdf_bytes()))
        table = reader._parse_visual_tables("col1\tcol2\nval1\tval2", 1, bbox=(0, 0, 10, 10))
        self.assertIsInstance(table, Table)
        self.assertEqual(table.columns, ["col1", "col2"])
        self.assertEqual(table.rows, [["val1", "val2"]])
        self.assertEqual(table.page, 1)

    def test_parse_visual_tables_empty(self):
        reader = PDFReader(DummyLoader(self._pdf_bytes()))
        table = reader._parse_visual_tables("   ", 1)
        self.assertEqual(table.columns, [])
        self.assertEqual(table.rows, [])

    @patch("nexum.document.reader.pdf_reader.PdfReader")
    def test_read_digital_pdf_no_ocr(self, mock_pdf_reader_cls):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 Content"
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.metadata = {"Author": "Dev"}
        mock_pdf_instance.pages = [mock_page]
        mock_pdf_reader_cls.return_value = mock_pdf_instance

        cfg = PDFReaderConfig(enable_ocr=False)
        reader = PDFReader(DummyLoader(self._pdf_bytes()), cfg)
        doc = reader.read()

        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.content, "Page 1 Content")
        self.assertEqual(doc.metadata, {"Author": "Dev"})

    @patch("nexum.document.reader.pdf_reader.convert_from_bytes")
    @patch("nexum.document.reader.pdf_reader.PdfReader")
    def test_read_pdf_with_ocr(self, mock_pdf_reader_cls, mock_convert):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # No digital text
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.metadata = {}
        mock_pdf_instance.pages = [mock_page]
        mock_pdf_reader_cls.return_value = mock_pdf_instance

        mock_convert.return_value = [self._sample_image()]

        cfg = PDFReaderConfig(enable_ocr=True)
        dummy_ocr = DummyOCR(cfg.ocr_config)
        reader = PDFReader(DummyLoader(self._pdf_bytes()), cfg, parser=dummy_ocr)

        # Mock layout analyzer inside reader
        reader.layout = MagicMock()
        reader.layout.run_detectors.return_value = {
            "tables": [],
            "blocks": [],
            "images": [],
            "flowcharts": []
        }

        doc = reader.read()
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.content, "col1|col2\n1|2\n3|4")

    @patch("nexum.document.reader.pdf_reader.PdfReader")
    def test_read_streaming_digital_pdf(self, mock_pdf_reader_cls):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Stream Page Content"
        mock_pdf_instance = MagicMock()
        mock_pdf_instance.metadata = {}
        mock_pdf_instance.pages = [mock_page]
        mock_pdf_reader_cls.return_value = mock_pdf_instance

        cfg = PDFReaderConfig(enable_ocr=False)
        reader = PDFReader(DummyLoader(self._pdf_bytes()), cfg)
        docs = list(reader.read_streaming())

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].content, "Stream Page Content")


if __name__ == "__main__":
    unittest.main()

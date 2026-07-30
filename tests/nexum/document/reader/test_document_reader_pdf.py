import unittest
from unittest.mock import patch, MagicMock

from pypdf.generic import DictionaryObject

from nexum.document.models import PDFReaderConfig, Table, Document
from nexum.document.parser.pdf import PDFOCRParser
from nexum.document.reader.pdf import PDFReader


class DummyLoader:
    def __init__(self, data: bytes):
        self.data = data

    def load(self) -> bytes:
        return self.data

    def stream(self):
        yield self.data


class DummyOCR(PDFOCRParser):
    def parse(self, page_bytes: bytes) -> str:
        return "col1|col2\n1|2\n3|4"


class TestPDFReader(unittest.TestCase):
    def _pdf_bytes(self):
        return b"%PDF-1.4 fake pdf bytes"

    @patch("nexum.document.reader.pdf.PdfReader")
    def test_pdf_has_text_true(self, mock_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "hello"
        mock_reader.return_value.pages = [mock_page]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(), DummyOCR(PDFReaderConfig().ocr_config))
        self.assertTrue(reader._pdf_has_text(self._pdf_bytes()))

    @patch("nexum.document.reader.pdf.PdfReader")
    def test_pdf_has_text_false(self, mock_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.return_value.pages = [mock_page]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(), DummyOCR(PDFReaderConfig().ocr_config))
        self.assertFalse(reader._pdf_has_text(self._pdf_bytes()))

    @patch("nexum.document.reader.pdf.PdfReader")
    def test_extract_metadata(self, mock_reader):
        mock_reader.return_value.metadata = {"Author": "Fabricio"}
        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(enable_metadata=True),
                           DummyOCR(PDFReaderConfig().ocr_config))
        metadata = reader._extract_metadata(self._pdf_bytes())
        self.assertEqual(metadata, {"Author": "Fabricio"})

    @patch("nexum.document.reader.pdf.PdfReader")
    def test_extract_images(self, mock_reader):
        mock_img = MagicMock()
        mock_img.get_data.return_value = b"imgbytes"
        mock_img.__getitem__.return_value = "/Image"

        xobjects = MagicMock()
        xobjects.get_object.return_value = {"img": mock_img}

        resources = DictionaryObject({"/XObject": xobjects})

        mock_page = MagicMock()
        mock_page.get.return_value = resources

        mock_reader.return_value.pages = [mock_page]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(enable_images=True),
                           DummyOCR(PDFReaderConfig().ocr_config))

        images = reader._extract_images(self._pdf_bytes())
        self.assertEqual(images, [b"imgbytes"])

    def test_extract_tables_heuristic(self):
        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(enable_tables=True),
                           DummyOCR(PDFReaderConfig().ocr_config))
        tables = reader._extract_tables_heuristic("col1|col2\n1|2\n3|4", 1)
        self.assertTrue(len(tables) >= 1)
        self.assertIsInstance(tables[0], Table)

    @patch("nexum.document.reader.pdf.convert_from_bytes")
    def test_extract_pdf_ocr_batch(self, mock_convert):
        mock_img = MagicMock()
        mock_convert.return_value = [mock_img]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(), DummyOCR(PDFReaderConfig().ocr_config))
        doc = reader._extract_pdf_ocr_batch(self._pdf_bytes())
        self.assertIsInstance(doc, Document)
        self.assertTrue("col1|col2" in doc.content)

    @patch("nexum.document.reader.pdf.convert_from_bytes")
    def test_extract_pdf_ocr_stream(self, mock_convert):
        mock_img = MagicMock()
        mock_convert.return_value = [mock_img]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(), DummyOCR(PDFReaderConfig().ocr_config))
        docs = list(reader._extract_pdf_ocr_stream(self._pdf_bytes()))
        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], Document)

    @patch("nexum.document.reader.pdf.PdfReader")
    def test_read_pdf_with_text_no_ocr(self, mock_reader):
        cfg = PDFReaderConfig(enable_ocr=False)
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "hello world"
        mock_reader.return_value.pages = [mock_page]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), cfg, DummyOCR(cfg.ocr_config))
        doc = reader.read()
        self.assertEqual(doc.content, "hello world")

    @patch("nexum.document.reader.pdf.PdfReader")
    @patch("nexum.document.reader.pdf.convert_from_bytes")
    def test_read_pdf_without_text_uses_ocr(self, mock_convert, mock_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.return_value.pages = [mock_page]

        mock_img = MagicMock()
        mock_convert.return_value = [mock_img]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(), DummyOCR(PDFReaderConfig().ocr_config))
        doc = reader.read()
        self.assertTrue("col1|col2" in doc.content)

    @patch("nexum.document.reader.pdf.PdfReader")
    def test_read_streaming_pdf_with_text(self, mock_reader):
        cfg = PDFReaderConfig(enable_ocr=False)
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "hello world"
        mock_reader.return_value.pages = [mock_page]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), cfg, DummyOCR(cfg.ocr_config))
        docs = list(reader.read_streaming())
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].content, "hello world")

    @patch("nexum.document.reader.pdf.PdfReader")
    @patch("nexum.document.reader.pdf.convert_from_bytes")
    def test_read_streaming_pdf_without_text_uses_ocr(self, mock_convert, mock_reader):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader.return_value.pages = [mock_page]

        mock_img = MagicMock()
        mock_convert.return_value = [mock_img]

        reader = PDFReader(DummyLoader(self._pdf_bytes()), PDFReaderConfig(), DummyOCR(PDFReaderConfig().ocr_config))
        docs = list(reader.read_streaming())
        self.assertEqual(len(docs), 1)
        self.assertTrue("col1|col2" in docs[0].content)


if __name__ == "__main__":
    unittest.main()

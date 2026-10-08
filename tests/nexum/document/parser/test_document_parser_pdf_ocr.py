import io
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
from PIL import Image

from nexum.common.errors import NexumRuntimeError
from nexum.document.models import PDFOCRConfig, Region, RegionType
from nexum.document.parser.pdf import PDFOCRParser


class TestPDFOCRParser(unittest.TestCase):
    def _create_png_bytes(self):
        img = Image.new("RGB", (200, 50), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @patch("nexum.document.parser.pdf.PDFRegionSplitterModel")
    def test_config_assignment(self, mock_splitter_class):
        cfg = PDFOCRConfig()
        parser = PDFOCRParser(cfg)
        self.assertIs(parser.config, cfg)

    @patch("nexum.document.parser.pdf.PDFRegionSplitterModel")
    @patch("nexum.document.parser.pdf.pytesseract.image_to_string")
    @patch("nexum.document.parser.pdf.ocr_table_cells")
    @patch("nexum.document.parser.pdf.extract_table_dataframe")
    def test_parse_success(
        self,
        mock_extract_table_dataframe,
        mock_ocr_table_cells,
        mock_image_to_string,
        mock_splitter_class
    ):
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        
        r1 = Region(type=RegionType.TEXT, box=(0, 0, 10, 10), score=0.9, text=None)
        r2 = Region(type=RegionType.TABLE, box=(10, 10, 20, 20), score=0.9, text=None)
        r3 = Region(type=RegionType.IMAGE, box=(20, 20, 30, 30), score=0.9, text=None)
        
        mock_splitter.predict.return_value = [r1, r2, r3]
        mock_image_to_string.return_value = "mock text"
        mock_ocr_table_cells.return_value = {}
        mock_extract_table_dataframe.return_value = pd.DataFrame({"col": [1]})
        
        parser = PDFOCRParser(PDFOCRConfig())
        page_bytes = self._create_png_bytes()
        
        result = parser.parse(page_bytes)
        
        self.assertIn("text_blocks", result)
        self.assertIn("tables", result)
        self.assertIn("images", result)
        
        self.assertEqual(len(result["text_blocks"]), 1)
        self.assertEqual(result["text_blocks"][0]["text"], "mock text")
        
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual(result["tables"][0]["box"], (10, 10, 20, 20))
        
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["images"][0]["box"], (20, 20, 30, 30))

    @patch("nexum.document.parser.pdf.PDFRegionSplitterModel")
    def test_parse_invalid_bytes(self, mock_splitter_class):
        parser = PDFOCRParser(PDFOCRConfig())
        with self.assertRaises(NexumRuntimeError):
            parser.parse(b"invalid")

if __name__ == "__main__":
    unittest.main()

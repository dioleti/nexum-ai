import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from nexum.common.helpers.table import (
    detect_table_boundaries,
    extract_table_dataframe,
    ocr_table_cells,
)
from nexum.document.models import Region, RegionType


class TestTableHelpers(unittest.TestCase):

    @patch("nexum.common.helpers.table.pytesseract.image_to_data")
    def test_ocr_table_cells(self, mock_image_to_data):
        mock_image_to_data.return_value = {
            "text": ["", "  ", "Header", "123"],
            "left": [0, 10, 20, 30],
            "top": [0, 10, 20, 30],
            "width": [100, 100, 50, 40],
            "height": [20, 20, 15, 15],
        }

        img = Image.new("RGB", (200, 200), color="white")
        region = Region(type=RegionType.TABLE, box=(10, 10, 190, 190), score=0.9, text=None)

        result = ocr_table_cells(img, region)

        self.assertIn("text", result)
        self.assertEqual(result["text"], ["Header", "123"])
        self.assertEqual(result["left"], [20, 30])
        self.assertEqual(result["top"], [20, 30])
        self.assertEqual(result["width"], [50, 40])
        self.assertEqual(result["height"], [15, 15])

    def test_extract_table_dataframe_empty(self):
        empty_ocr_data = {"text": [], "left": [], "top": [], "width": [], "height": []}
        df = extract_table_dataframe(empty_ocr_data)
        self.assertTrue(df.empty)

    def test_extract_table_dataframe_valid(self):
        # Header row at y=10: "Name" (left=10), "Age" (left=100)
        # Row 1 at y=40: "Alice" (left=10), "30" (left=100)
        # Row 2 at y=70: "Bob" (left=10), "25" (left=100)
        ocr_data = {
            "text": ["Name:", "Age", "Alice", "30", "Bob", "25"],
            "left": [10, 100, 10, 100, 10, 100],
            "top": [10, 10, 40, 40, 70, 70],
            "width": [40, 30, 40, 20, 30, 20],
            "height": [15, 15, 15, 15, 15, 15],
        }
        df = extract_table_dataframe(ocr_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("Age", df.columns)

    def test_detect_table_boundaries(self):
        lines = [
            "Title line without delimiter",
            "Subtitle without delimiter",
            "Name,Age,Score",
            "Alice,30,95.5",
            "Bob,25,88.0",
        ]
        skip_count, has_header = detect_table_boundaries(lines, delimiter=",")
        self.assertEqual(skip_count, 2)
        self.assertTrue(has_header)

    def test_detect_table_boundaries_empty(self):
        skip_count, has_header = detect_table_boundaries([])
        self.assertEqual(skip_count, 0)
        self.assertFalse(has_header)


if __name__ == "__main__":
    unittest.main()

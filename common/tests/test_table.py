import csv
import unittest
from unittest.mock import patch

from PIL import Image

from nexum_ai_common.helpers.table import (
    detect_table_boundaries,
    extract_table_dataframe,
    ocr_table_cells,
)
from nexum_ai_common.models import Region, RegionType


class TestTableHelper(unittest.TestCase):
    def test_extract_table_dataframe_empty_or_insufficient(self):
        self.assertTrue(extract_table_dataframe({}).empty)
        self.assertTrue(extract_table_dataframe({"text": ["   "]}).empty)

        single_line = {
            "text": ["Col1", "Col2"],
            "left": [10, 100],
            "top": [10, 10],
            "width": [30, 30],
            "height": [10, 10],
        }
        self.assertTrue(extract_table_dataframe(single_line).empty)

        single_col = {
            "text": ["Header", "Row1"],
            "left": [10, 10],
            "top": [10, 50],
            "width": [30, 30],
            "height": [10, 10],
        }
        self.assertTrue(extract_table_dataframe(single_col).empty)

    def test_extract_table_dataframe_valid(self):
        ocr_data = {
            "text": [
                "Item", "Name:", "Qty", "Total", "Price",
                "Apple", "Red", "10", "25.00",
                "Banana", "20", "15.00",
            ],
            "left": [
                10, 35, 150, 250, 275,
                10, 40, 155, 260,
                10, 155, 260,
            ],
            "top": [
                10, 12, 10, 10, 10,
                50, 52, 50, 50,
                90, 90, 90,
            ],
            "width": [
                20, 25, 30, 20, 25,
                25, 20, 20, 30,
                40, 20, 30,
            ],
            "height": [
                12, 12, 12, 12, 12,
                12, 12, 12, 12,
                12, 12, 12,
            ],
        }
        df = extract_table_dataframe(ocr_data)
        self.assertFalse(df.empty)
        self.assertEqual(df.index.name, "Item Name")
        self.assertIn("Apple Red", df.index)
        self.assertIn("Banana", df.index)
        self.assertEqual(df.loc["Apple Red", "Qty"], "10")
        self.assertEqual(df.loc["Apple Red", "Total Price"], "25.00")
        self.assertEqual(df.loc["Banana", "Qty"], "20")
        self.assertEqual(df.loc["Banana", "Total Price"], "15.00")

    def test_detect_table_boundaries(self):
        self.assertEqual(detect_table_boundaries([]), (0, False))
        self.assertEqual(detect_table_boundaries(["   ", ""]), (0, False))

        lines = [
            "Report Title",
            "Generated on 2026-10-21",
            "name,age,score",
            "Alice,30,95.5",
            "Bob,25,88.0",
        ]
        skip, has_header = detect_table_boundaries(lines, delimiter=",")
        self.assertEqual(skip, 2)
        self.assertTrue(has_header)

        auto_skip, _ = detect_table_boundaries(lines)
        self.assertEqual(auto_skip, 2)

        with patch("nexum_ai_common.helpers.table.csv.Sniffer.sniff", side_effect=csv.Error), patch(
            "nexum_ai_common.helpers.table.csv.Sniffer.has_header", side_effect=csv.Error
        ):
            s, h = detect_table_boundaries(["intro", "a,b", "1,2"])
            self.assertEqual(s, 1)
            self.assertTrue(h)

    @patch("nexum_ai_common.helpers.table.pytesseract.image_to_data")
    def test_ocr_table_cells(self, mock_to_data):
        mock_to_data.return_value = {
            "text": ["  ", "Header", "Value"],
            "left": [0, 10, 50],
            "top": [0, 5, 5],
            "width": [100, 30, 40],
            "height": [100, 12, 12],
        }
        img = Image.new("RGB", (100, 100))
        region = Region(type=RegionType.TABLE, box=(0, 0, 100, 100), score=0.9)
        res = ocr_table_cells(img, region)
        self.assertEqual(res["text"], ["Header", "Value"])
        self.assertEqual(res["left"], [10, 50])
        self.assertEqual(res["top"], [5, 5])
        self.assertEqual(res["width"], [30, 40])
        self.assertEqual(res["height"], [12, 12])


if __name__ == "__main__":
    unittest.main()

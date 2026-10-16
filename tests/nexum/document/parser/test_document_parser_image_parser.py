import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image

from nexum.common.errors import NexumRuntimeError
from nexum.document.models import (
    BoundingBox,
    DetectedTable,
    ImageOCRConfig,
    Parsed,
)
from nexum.document.parser.image_parser import ImageOCRParser


class TestImageOCRParser(unittest.TestCase):
    def _create_png_bytes(self):
        img = Image.new("RGB", (200, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_config_assignment(self):
        cfg = ImageOCRConfig()
        parser = ImageOCRParser(cfg)
        self.assertIs(parser.config, cfg)

    def test_parse_raises_on_invalid_bytes(self):
        parser = ImageOCRParser(ImageOCRConfig())
        with self.assertRaises(NexumRuntimeError):
            parser.parse(b"not_an_image")

    @patch("nexum.document.parser.image_parser.run_ocr")
    @patch("nexum.document.parser.image_parser.extract_table_dataframe")
    @patch("nexum.document.parser.image_parser.run_ocr_data")
    def test_parse_full_pipeline_mocked(self, mock_ocr_data, mock_extract_df, mock_run_ocr):
        cfg = ImageOCRConfig(
            table_as_df=True,
            enable_caption=False,
            enable_flowchart=False,
        )
        parser = ImageOCRParser(cfg)

        mock_tbl_detector = MagicMock()
        mock_tbl_detector.predict.return_value = [
            DetectedTable(
                label="table",
                score=0.9,
                box=BoundingBox(xmin=10.0, ymin=10.0, xmax=100.0, ymax=50.0),
            )
        ]
        parser.table_detector = mock_tbl_detector

        mock_ocr_data.return_value = {"text": ["A", "B"]}
        df = pd.DataFrame({"Col1": ["XK", "val"]})
        mock_extract_df.return_value = df
        mock_run_ocr.return_value = "Residual block line"

        result = parser.parse(self._create_png_bytes())

        self.assertIsInstance(result, Parsed)
        self.assertTrue(result.metadata["has_tables"])
        self.assertEqual(result.metadata["tables_count"], 1)
        self.assertEqual(result.tables[0]["Col1"].iloc[0], "X")

    def test_remove_table_grid_lines(self):
        crop = np.full((100, 100), 255, dtype=np.uint8)
        crop[50, :] = 0
        cleaned = ImageOCRParser._remove_table_grid_lines(crop)
        self.assertEqual(cleaned.shape, (100, 100))

    def test_flowchart_helpers(self):
        ctx = MagicMock()
        ctx.width = 1000
        ctx.height = 1000
        ctx.total_pixels = 1_000_000
        ctx.gray = np.full((1000, 1000), 255, dtype=np.uint8)
        ctx.text_canvas = np.full((1000, 1000), 255, dtype=np.uint8)

        raw_nodes = [
            {"id": "n1", "label": "Start", "shape": "rectangle", "box": (100, 100, 100, 50), "center": (150, 125)},
            {"id": "n2", "label": "Check?", "shape": "diamond", "box": (100, 300, 100, 60), "center": (150, 330)},
            {"id": "n3", "label": "YesBranch", "shape": "circle_or_ellipse", "box": (300, 400, 100, 50), "center": (350, 425)},
        ]
        valid = ImageOCRParser._filter_valid_flowchart_nodes(ctx, raw_nodes)
        self.assertEqual(len(valid), 3)

        edges = ImageOCRParser._build_flowchart_edges(ctx, valid)
        self.assertGreaterEqual(len(edges), 2)

        mermaid = ImageOCRParser._render_mermaid(valid, edges)
        self.assertIn("graph TD", mermaid)
        self.assertIn('n2{"Check?"}', mermaid)
        self.assertIn('n3(("YesBranch"))', mermaid)


if __name__ == "__main__":
    unittest.main()

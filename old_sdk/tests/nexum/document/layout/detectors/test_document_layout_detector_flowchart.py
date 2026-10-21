import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from nexum.document.layout.detectors.flowchart import FlowchartDetector
from nexum.document.models import FlowchartRegion


class TestFlowchartDetector(unittest.TestCase):
    @patch("nexum.document.layout.detectors.flowchart.lp.Detectron2LayoutModel", create=True)
    def test_detect_figure_blocks(self, mock_lp_model_cls):
        mock_model = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "Figure"
        mock_block.block.coordinates = (10.2, 20.8, 100.5, 150.1)

        mock_text_block = MagicMock()
        mock_text_block.type = "Text"

        mock_model.detect.return_value = [mock_block, mock_text_block]
        mock_lp_model_cls.return_value = mock_model

        detector = FlowchartDetector(enable_binarize=True)
        rgb = np.full((200, 200, 3), 255, dtype=np.uint8)
        regions = detector.detect(rgb)

        self.assertEqual(len(regions), 1)
        self.assertIsInstance(regions[0], FlowchartRegion)
        self.assertEqual(regions[0].bbox, (10, 20, 100, 150))

    @patch("nexum.document.layout.detectors.flowchart.lp.Detectron2LayoutModel", create=True)
    def test_detect_no_figures(self, mock_lp_model_cls):
        mock_model = MagicMock()
        mock_model.detect.return_value = []
        mock_lp_model_cls.return_value = mock_model

        detector = FlowchartDetector(
            enable_deskew=False,
            enable_clahe=False,
            enable_denoise=False,
            enable_sharpen=False,
        )
        rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertEqual(detector.detect(rgb), [])


if __name__ == "__main__":
    unittest.main()

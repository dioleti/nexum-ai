import unittest

import numpy as np

from nexum.document.layout.detectors.table import TableDetector
from nexum.document.models import TableRegion


class TestTableDetector(unittest.TestCase):
    def test_detect_tables(self):
        gray = np.full((250, 250), 255, dtype=np.uint8)
        # Draw horizontal and vertical grid lines
        gray[20:22, 20:200] = 0
        gray[100:102, 20:200] = 0
        gray[180:182, 20:200] = 0
        gray[20:182, 20:22] = 0
        gray[20:182, 110:112] = 0
        gray[20:182, 198:200] = 0

        detector = TableDetector(min_table_area=1000)
        tables = detector.detect(gray)

        self.assertGreaterEqual(len(tables), 1)
        self.assertIsInstance(tables[0], TableRegion)

    def test_detect_filters_small_tables(self):
        gray = np.full((200, 200), 255, dtype=np.uint8)
        detector = TableDetector(
            min_table_area=50000,
            enable_deskew=False,
            enable_clahe=False,
            enable_denoise=False,
            enable_sharpen=False,
            enable_binarize=False,
        )
        tables = detector.detect(gray)
        self.assertEqual(tables, [])


if __name__ == "__main__":
    unittest.main()

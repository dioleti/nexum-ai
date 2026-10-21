import unittest

import numpy as np

from nexum.document.layout.detectors.block import BlockDetector
from nexum.document.models import TextBlockRegion


class TestBlockDetector(unittest.TestCase):
    def test_detect_blocks(self):
        gray = np.full((200, 200), 255, dtype=np.uint8)
        gray[20:100, 20:100] = 0

        detector = BlockDetector(min_block_area=1000)
        blocks = detector.detect(gray)

        self.assertGreaterEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], TextBlockRegion)

    def test_detect_filters_small_blocks(self):
        gray = np.full((200, 200), 255, dtype=np.uint8)
        gray[10:15, 10:15] = 0

        detector = BlockDetector(
            min_block_area=50000,
            enable_deskew=False,
            enable_clahe=False,
            enable_denoise=False,
            enable_sharpen=False,
            enable_binarize=False,
        )
        blocks = detector.detect(gray)
        self.assertEqual(blocks, [])


if __name__ == "__main__":
    unittest.main()

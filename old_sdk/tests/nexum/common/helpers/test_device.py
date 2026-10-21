import unittest
from unittest.mock import patch

import torch

from nexum.common.helpers.device import resolve


class TestDeviceHelper(unittest.TestCase):
    @patch("nexum.common.helpers.device.torch.cuda.is_available", return_value=True)
    def test_resolve_cuda(self, mock_cuda):
        device = resolve()
        self.assertEqual(device, torch.device("cuda"))

    @patch("nexum.common.helpers.device.torch.cuda.is_available", return_value=False)
    def test_resolve_mps(self, mock_cuda):
        with patch("nexum.common.helpers.device.torch.backends") as mock_backends:
            mock_backends.mps.is_available.return_value = True
            device = resolve()
            self.assertEqual(device, torch.device("mps"))

    @patch("nexum.common.helpers.device.torch.cuda.is_available", return_value=False)
    def test_resolve_cpu(self, mock_cuda):
        with patch("nexum.common.helpers.device.torch.backends") as mock_backends:
            mock_backends.mps.is_available.return_value = False
            device = resolve()
            self.assertEqual(device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

import unittest

from nexum.common.constants.date import formats


class TestDateConstants(unittest.TestCase):
    def test_formats_not_empty(self):
        self.assertIsInstance(formats, list)
        self.assertGreater(len(formats), 0)
        self.assertIn("%d/%m/%Y", formats)
        self.assertIn("%Y-%m-%d", formats)


if __name__ == "__main__":
    unittest.main()

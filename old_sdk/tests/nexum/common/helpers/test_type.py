import unittest
from datetime import date, datetime

from nexum.common.helpers.type import infer


class TestTypeHelper(unittest.TestCase):
    def test_infer_empty_or_whitespace(self):
        self.assertIsNone(infer(""))
        self.assertIsNone(infer("   "))

    def test_infer_boolean(self):
        self.assertIs(infer("true"), True)
        self.assertIs(infer("TRUE"), True)
        self.assertIs(infer("false"), False)
        self.assertIs(infer("False"), False)

    def test_infer_int(self):
        self.assertEqual(infer("42"), 42)
        self.assertEqual(infer("-10"), -10)
        self.assertEqual(infer("+7"), 7)

    def test_infer_float(self):
        self.assertEqual(infer("3.14"), 3.14)
        self.assertEqual(infer("-0.5"), -0.5)
        self.assertEqual(infer("1e3"), 1000.0)
        self.assertEqual(infer("1.250,50"), 1250.50)

    def test_infer_date_and_datetime(self):
        self.assertEqual(infer("2024-05-20"), datetime(2024, 5, 20, 0, 0))
        self.assertEqual(infer("15/03/1990"), date(1990, 3, 15))
        self.assertEqual(infer("2024-05-20T14:30:00"), datetime(2024, 5, 20, 14, 30))

    def test_infer_string_fallback(self):
        self.assertEqual(infer("hello world"), "hello world")
        self.assertEqual(infer("abc,def"), "abc,def")


if __name__ == "__main__":
    unittest.main()

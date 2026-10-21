import unittest
from datetime import date, datetime
from unittest.mock import patch

from nexum_ai_common.constants.date import formats
from nexum_ai_common.helpers.type import infer


class TestTypeHelper(unittest.TestCase):
    def test_infer_empty_or_whitespace(self):
        self.assertIsNone(infer(""))
        self.assertIsNone(infer("   "))

    def test_infer_boolean(self):
        self.assertTrue(infer("true"))
        self.assertTrue(infer("TRUE"))
        self.assertTrue(infer("  True "))
        self.assertFalse(infer("false"))
        self.assertFalse(infer("FALSE"))
        self.assertFalse(infer(" False "))

    def test_infer_integer(self):
        self.assertEqual(infer("42"), 42)
        self.assertEqual(infer("-100"), -100)
        self.assertEqual(infer("+7"), 7)
        self.assertIsInstance(infer("0"), int)

    def test_infer_float_dot(self):
        self.assertAlmostEqual(infer("3.14"), 3.14)
        self.assertAlmostEqual(infer("-0.5"), -0.5)
        self.assertAlmostEqual(infer(".25"), 0.25)
        self.assertAlmostEqual(infer("1e-3"), 0.001)
        self.assertAlmostEqual(infer("+2.5E2"), 250.0)

    def test_infer_float_comma(self):
        self.assertAlmostEqual(infer("1.234,56"), 1234.56)
        self.assertAlmostEqual(infer("3,14"), 3.14)
        self.assertEqual(infer("a,b"), "a,b")

    def test_infer_iso_datetime_and_formats(self):
        dt_iso = infer("2026-10-21T09:45:00")
        self.assertEqual(dt_iso, datetime(2026, 10, 21, 9, 45, 0))

        d_slash = infer("21/10/2026")
        self.assertEqual(d_slash, date(2026, 10, 21))

        d_short = infer("21/10/26")
        self.assertEqual(d_short, date(2026, 10, 21))

        d_dot = infer("2026.10.21")
        self.assertEqual(d_dot, date(2026, 10, 21))

        self.assertTrue(len(formats) > 0)

        with patch("nexum_ai_common.helpers.type.formats", ["%Y/%m/%d %H:%M"]):
            dt_with_time = infer("2026/10/21 14:30")
            self.assertEqual(dt_with_time, datetime(2026, 10, 21, 14, 30))

    def test_infer_fallback_string(self):
        self.assertEqual(infer("hello world"), "hello world")
        self.assertEqual(infer("   nexum  "), "nexum")


if __name__ == "__main__":
    unittest.main()

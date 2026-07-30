import unittest
from typing import Any

from nexum.document.models import NexumConfig
from nexum.document.parser.base import BaseParser


class DummyParser(BaseParser):
    def parse(self, data: Any) -> Any:
        return data


class TestBaseParser(unittest.TestCase):
    def test_config_is_assigned(self):
        cfg = NexumConfig()
        parser = DummyParser(cfg)
        self.assertIs(parser.config, cfg)

    def test_parse_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseParser()  # cannot instantiate abstract class

    def test_dummy_parser_parse(self):
        parser = DummyParser(None)
        self.assertEqual(parser.parse("x"), "x")
        self.assertEqual(parser.parse(123), 123)
        self.assertEqual(parser.parse(b"abc"), b"abc")


if __name__ == "__main__":
    unittest.main()

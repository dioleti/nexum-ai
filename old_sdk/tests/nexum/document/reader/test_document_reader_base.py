import unittest
from collections.abc import Generator
from unittest.mock import MagicMock

from nexum.document.models import Document
from nexum.document.reader.base import Reader


class DummyReader(Reader):
    def read(self) -> Document:
        return Document(content="ok", metadata={})

    def read_streaming(self) -> Generator[Document, None, None]:
        yield Document(content="stream", metadata={})


class TestBaseReader(unittest.TestCase):
    def test_cannot_instantiate_abstract_reader(self):
        with self.assertRaises(TypeError):
            Reader(MagicMock())

    def test_dummy_reader(self):
        loader = MagicMock()
        reader = DummyReader(loader)
        self.assertIs(reader.loader, loader)
        self.assertEqual(reader.read().content, "ok")
        self.assertEqual(list(reader.read_streaming())[0].content, "stream")


if __name__ == "__main__":
    unittest.main()

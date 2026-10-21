import unittest
from typing import Generator

from nexum.common.storage.loader import Loader


class DummyLoader(Loader):
    def __init__(self, data: bytes):
        self.data = data

    def stream(self) -> Generator[bytes, None, None]:
        yield self.data

    def load(self) -> bytes:
        return self.data


class TestLoader(unittest.TestCase):
    def test_load_returns_bytes(self):
        loader = DummyLoader(b"abc")
        self.assertEqual(loader.load(), b"abc")

    def test_stream_returns_generator(self):
        loader = DummyLoader(b"xyz")
        gen = loader.stream()
        self.assertTrue(hasattr(gen, "__iter__"))
        self.assertEqual(b"".join(chunk for chunk in gen), b"xyz")


if __name__ == "__main__":
    unittest.main()

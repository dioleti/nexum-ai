import unittest

from nexum.common.errors import NexumValueError
from nexum.document.storage.client.base import FileClient
from nexum.document.storage.loader.base import FileLoader


class DummyClient(FileClient):
    def __init__(self, exists=True, data=b"abc"):
        self._exists = exists
        self._data = data

    def file_exists(self, path: str) -> bool:
        return self._exists

    def load_bytes(self, path: str) -> bytes:
        return self._data

    def stream(self, path: str):
        yield self._data

    def get_uri(self, path: str) -> str:
        return f"uri://{path}"


class TestFileLoader(unittest.TestCase):
    def test_load_success(self):
        client = DummyClient(exists=True, data=b"xyz")
        loader = FileLoader(client, "document.txt")
        data = loader.load()
        self.assertEqual(data, b"xyz")

    def test_load_not_found(self):
        client = DummyClient(exists=False)
        loader = FileLoader(client, "document.txt")
        with self.assertRaises(NexumValueError):
            loader.load()

    def test_stream_success(self):
        client = DummyClient(exists=True, data=b"stream")
        loader = FileLoader(client, "document.txt")
        chunks = list(loader.stream())
        self.assertEqual(chunks, [b"stream"])

    def test_stream_not_found(self):
        client = DummyClient(exists=False)
        loader = FileLoader(client, "document.txt")
        with self.assertRaises(NexumValueError):
            list(loader.stream())


if __name__ == "__main__":
    unittest.main()

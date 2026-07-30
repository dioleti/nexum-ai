import unittest

from nexum.document.storage.client.base import FileClient


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


class TestFileClient(unittest.TestCase):
    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            FileClient()

    def test_file_exists(self):
        client = DummyClient(exists=True)
        self.assertTrue(client.file_exists("x"))

    def test_file_not_exists(self):
        client = DummyClient(exists=False)
        self.assertFalse(client.file_exists("x"))

    def test_load_bytes(self):
        client = DummyClient(data=b"xyz")
        self.assertEqual(client.load_bytes("x"), b"xyz")

    def test_stream(self):
        client = DummyClient(data=b"stream")
        chunks = list(client.stream("x"))
        self.assertEqual(chunks, [b"stream"])

    def test_get_uri(self):
        client = DummyClient()
        self.assertEqual(client.get_uri("path/to/document"), "uri://path/to/document")


if __name__ == "__main__":
    unittest.main()

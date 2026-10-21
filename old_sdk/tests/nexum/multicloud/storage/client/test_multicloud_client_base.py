import unittest

from nexum.multicloud.storage.client.base import StorageClient


class DummyStorageClient(StorageClient):
    def __init__(self, data=b"abc", exists=True):
        self._data = data
        self._exists = exists

    def stream_blob(self, container: str, blob: str):
        yield self._data

    def get_blob(self, container: str, blob: str) -> bytes:
        return self._data

    def blob_exists(self, container: str, blob: str) -> bool:
        return self._exists

    def get_uri(self, container: str, blob: str) -> str:
        return f"uri://{container}/{blob}"


class TestStorageClient(unittest.TestCase):
    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            StorageClient()

    def test_stream_blob(self):
        client = DummyStorageClient(data=b"x")
        chunks = list(client.stream_blob("cont", "blob"))
        self.assertEqual(chunks, [b"x"])

    def test_get_blob(self):
        client = DummyStorageClient(data=b"xyz")
        data = client.get_blob("cont", "blob")
        self.assertEqual(data, b"xyz")

    def test_blob_exists_true(self):
        client = DummyStorageClient(exists=True)
        self.assertTrue(client.blob_exists("cont", "blob"))

    def test_blob_exists_false(self):
        client = DummyStorageClient(exists=False)
        self.assertFalse(client.blob_exists("cont", "blob"))

    def test_get_uri(self):
        client = DummyStorageClient()
        uri = client.get_uri("cont", "blob")
        self.assertEqual(uri, "uri://cont/blob")


if __name__ == "__main__":
    unittest.main()

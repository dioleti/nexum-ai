import unittest

from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.base import StorageClient
from nexum.multicloud.storage.loader.base import CloudLoader


class DummyClient(StorageClient):
    def __init__(self, exists=True, data=b"abc"):
        self._exists = exists
        self._data = data

    def blob_exists(self, container: str, blob: str) -> bool:
        return self._exists

    def get_blob(self, container: str, blob: str) -> bytes:
        return self._data

    def stream_blob(self, container: str, blob: str):
        yield self._data

    def get_uri(self, container: str, blob: str) -> str:
        return f"uri://{container}/{blob}"


class TestCloudLoader(unittest.TestCase):
    def test_load_success(self):
        client = DummyClient(exists=True, data=b"xyz")
        loader = CloudLoader(client, "cont", "blob")
        data = loader.load()
        self.assertEqual(data, b"xyz")

    def test_load_not_found(self):
        client = DummyClient(exists=False)
        loader = CloudLoader(client, "cont", "blob")
        with self.assertRaises(NexumValueError):
            loader.load()

    def test_stream_success(self):
        client = DummyClient(exists=True, data=b"stream")
        loader = CloudLoader(client, "cont", "blob")
        chunks = list(loader.stream())
        self.assertEqual(chunks, [b"stream"])

    def test_stream_not_found(self):
        client = DummyClient(exists=False)
        loader = CloudLoader(client, "cont", "blob")
        with self.assertRaises(NexumValueError):
            list(loader.stream())


if __name__ == "__main__":
    unittest.main()

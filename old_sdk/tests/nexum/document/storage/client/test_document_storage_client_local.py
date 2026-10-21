import os
import tempfile
import unittest

from nexum.document.storage.client.local import LocalFileClient


class TestLocalFileClient(unittest.TestCase):
    def test_file_exists_true(self):
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            client = LocalFileClient()
            self.assertTrue(client.file_exists(tmp.name))

    def test_file_exists_false(self):
        client = LocalFileClient()
        self.assertFalse(client.file_exists("nonexistent.document"))

    def test_load_bytes(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"abc")
            tmp.flush()
            path = tmp.name

        client = LocalFileClient()
        data = client.load_bytes(path)
        self.assertEqual(data, b"abc")

        os.remove(path)

    def test_stream(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"chunk1chunk2")
            tmp.flush()
            path = tmp.name

        client = LocalFileClient()
        chunks = list(client.stream(path))
        self.assertEqual(b"".join(chunks), b"chunk1chunk2")

        os.remove(path)

    def test_get_uri(self):
        client = LocalFileClient()
        uri = client.get_uri("/path/to/document.txt")
        self.assertEqual(uri, "file:///path/to/document.txt")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

from nexum.common.errors import NexumValueError
from nexum.document.storage.client.https import HTTPSFileClient
from nexum.document.storage.loader.https import HTTPSFileLoader


class TestHTTPSFileLoader(unittest.TestCase):
    def test_init_with_client(self):
        client = MagicMock(spec=HTTPSFileClient)
        loader = HTTPSFileLoader("http://example.com/file.txt", client=client)
        self.assertIs(loader.client, client)
        self.assertEqual(loader.path, "http://example.com/file.txt")

    @patch("nexum.document.storage.loader.https.HTTPSFileClient")
    def test_init_creates_client(self, mock_client):
        loader = HTTPSFileLoader("http://example.com/file.txt")
        mock_client.assert_called_once()
        self.assertEqual(loader.path, "http://example.com/file.txt")

    def test_load_success(self):
        client = MagicMock(spec=HTTPSFileClient)
        client.file_exists.return_value = True
        client.load_bytes.return_value = b"abc"

        loader = HTTPSFileLoader("http://example.com/file.txt", client=client)
        data = loader.load()
        self.assertEqual(data, b"abc")

    def test_load_not_found(self):
        client = MagicMock(spec=HTTPSFileClient)
        client.file_exists.return_value = False

        loader = HTTPSFileLoader("http://example.com/file.txt", client=client)
        with self.assertRaises(NexumValueError):
            loader.load()

    def test_stream_success(self):
        client = MagicMock(spec=HTTPSFileClient)
        client.file_exists.return_value = True
        client.stream.return_value = iter([b"chunk1", b"chunk2"])

        loader = HTTPSFileLoader("http://example.com/file.txt", client=client)
        chunks = list(loader.stream())
        self.assertEqual(chunks, [b"chunk1", b"chunk2"])

    def test_stream_not_found(self):
        client = MagicMock(spec=HTTPSFileClient)
        client.file_exists.return_value = False

        loader = HTTPSFileLoader("http://example.com/file.txt", client=client)
        with self.assertRaises(NexumValueError):
            list(loader.stream())


if __name__ == "__main__":
    unittest.main()

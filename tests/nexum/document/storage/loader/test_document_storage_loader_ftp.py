import unittest
from unittest.mock import MagicMock, patch

from nexum.common.errors import NexumValueError
from nexum.document.storage.client.ftp import FTPFileClient
from nexum.document.storage.loader.ftp import FTPFileLoader


class TestFTPFileLoader(unittest.TestCase):
    def test_init_with_client(self):
        client = MagicMock(spec=FTPFileClient)
        loader = FTPFileLoader("path/to/document", client=client)
        self.assertIs(loader.client, client)
        self.assertEqual(loader.path, "path/to/document")

    def test_init_without_host_raises(self):
        with self.assertRaises(NexumValueError):
            FTPFileLoader("path/to/document")

    @patch("nexum.document.storage.loader.ftp.FTPFileClient")
    def test_init_creates_client(self, mock_client):
        loader = FTPFileLoader("document.txt", host="ftp.example.com", user="u", password="p")
        mock_client.assert_called_once_with(host="ftp.example.com", user="u", password="p")
        self.assertEqual(loader.path, "document.txt")

    def test_load_success(self):
        client = MagicMock(spec=FTPFileClient)
        client.file_exists.return_value = True
        client.load_bytes.return_value = b"abc"

        loader = FTPFileLoader("document.txt", client=client)
        data = loader.load()
        self.assertEqual(data, b"abc")

    def test_load_not_found(self):
        client = MagicMock(spec=FTPFileClient)
        client.file_exists.return_value = False

        loader = FTPFileLoader("document.txt", client=client)
        with self.assertRaises(NexumValueError):
            loader.load()

    def test_stream_success(self):
        client = MagicMock(spec=FTPFileClient)
        client.file_exists.return_value = True
        client.stream.return_value = iter([b"chunk1", b"chunk2"])

        loader = FTPFileLoader("document.txt", client=client)
        chunks = list(loader.stream())
        self.assertEqual(chunks, [b"chunk1", b"chunk2"])

    def test_stream_not_found(self):
        client = MagicMock(spec=FTPFileClient)
        client.file_exists.return_value = False

        loader = FTPFileLoader("document.txt", client=client)
        with self.assertRaises(NexumValueError):
            list(loader.stream())


if __name__ == "__main__":
    unittest.main()

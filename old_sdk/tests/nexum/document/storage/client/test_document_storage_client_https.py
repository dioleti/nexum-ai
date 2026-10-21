import unittest
from unittest.mock import patch, MagicMock

from nexum.common.errors import NexumValueError
from nexum.document.storage.client.https import HTTPSFileClient


class TestHTTPSFileClient(unittest.TestCase):
    @patch("nexum.document.storage.client.https.requests.head")
    def test_file_exists_true(self, mock_head):
        mock_head.return_value.status_code = 200
        client = HTTPSFileClient()
        self.assertTrue(client.file_exists("http://example.com/file.txt"))

    @patch("nexum.document.storage.client.https.requests.head")
    def test_file_exists_false(self, mock_head):
        mock_head.return_value.status_code = 404
        client = HTTPSFileClient()
        self.assertFalse(client.file_exists("http://example.com/file.txt"))

    @patch("nexum.document.storage.client.https.requests.get")
    def test_load_bytes_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"abc"
        client = HTTPSFileClient()
        data = client.load_bytes("http://example.com/file.txt")
        self.assertEqual(data, b"abc")

    @patch("nexum.document.storage.client.https.requests.get")
    def test_load_bytes_not_found(self, mock_get):
        mock_get.return_value.status_code = 404
        client = HTTPSFileClient()
        with self.assertRaises(NexumValueError):
            client.load_bytes("http://example.com/file.txt")

    @patch("nexum.document.storage.client.https.requests.get")
    def test_stream_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_resp

        client = HTTPSFileClient()
        chunks = list(client.stream("http://example.com/file.txt"))
        self.assertEqual(chunks, [b"chunk1", b"chunk2"])

    @patch("nexum.document.storage.client.https.requests.get")
    def test_stream_not_found(self, mock_get):
        mock_get.return_value.status_code = 404
        client = HTTPSFileClient()
        with self.assertRaises(NexumValueError):
            list(client.stream("http://example.com/file.txt"))

    def test_get_uri(self):
        client = HTTPSFileClient()
        self.assertEqual(client.get_uri("http://example.com/file.txt"),
                         "http://example.com/file.txt")


if __name__ == "__main__":
    unittest.main()

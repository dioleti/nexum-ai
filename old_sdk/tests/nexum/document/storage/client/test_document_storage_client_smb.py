import unittest
from unittest.mock import patch, MagicMock

from nexum.document.storage.client.smb import SMBFileClient


class TestSMBFileClient(unittest.TestCase):
    def _mock_open(self, mock_open, data=b"abc"):
        file_obj = MagicMock()
        file_obj.attributes = {"allocation_size": len(data)}
        file_obj.read.side_effect = [
            data,
            b""
        ]
        mock_open.return_value = file_obj
        return file_obj

    @patch("nexum.document.storage.client.smb.Open")
    @patch("nexum.document.storage.client.smb.TreeConnect")
    @patch("nexum.document.storage.client.smb.Session")
    @patch("nexum.document.storage.client.smb.Connection")
    def test_file_exists_true(self, mock_conn, mock_session, mock_tree, mock_open):
        file_obj = MagicMock()
        mock_open.return_value = file_obj
        client = SMBFileClient("server", "share", "user", "pass")
        self.assertTrue(client.file_exists("path/to/document"))

    @patch("nexum.document.storage.client.smb.Open", side_effect=Exception("fail"))
    @patch("nexum.document.storage.client.smb.TreeConnect")
    @patch("nexum.document.storage.client.smb.Session")
    @patch("nexum.document.storage.client.smb.Connection")
    def test_file_exists_false(self, mock_conn, mock_session, mock_tree, mock_open):
        client = SMBFileClient("server", "share", "user", "pass")
        self.assertFalse(client.file_exists("path/to/document"))

    @patch("nexum.document.storage.client.smb.Open")
    @patch("nexum.document.storage.client.smb.TreeConnect")
    @patch("nexum.document.storage.client.smb.Session")
    @patch("nexum.document.storage.client.smb.Connection")
    def test_load_bytes(self, mock_conn, mock_session, mock_tree, mock_open):
        file_obj = MagicMock()
        file_obj.attributes = {"allocation_size": 3}
        file_obj.read.return_value = b"xyz"
        mock_open.return_value = file_obj

        client = SMBFileClient("server", "share", "user", "pass")
        data = client.load_bytes("path/to/document")
        self.assertEqual(data, b"xyz")

    @patch("nexum.document.storage.client.smb.Open")
    @patch("nexum.document.storage.client.smb.TreeConnect")
    @patch("nexum.document.storage.client.smb.Session")
    @patch("nexum.document.storage.client.smb.Connection")
    def test_stream(self, mock_conn, mock_session, mock_tree, mock_open):
        file_obj = MagicMock()
        file_obj.read.side_effect = [b"chunk1", b"chunk2", b""]
        mock_open.return_value = file_obj

        client = SMBFileClient("server", "share", "user", "pass")
        chunks = list(client.stream("path/to/document"))
        self.assertEqual(chunks, [b"chunk1", b"chunk2"])

    def test_get_uri(self):
        client = SMBFileClient("server", "share", "user", "pass")
        uri = client.get_uri("path/to/document")
        self.assertEqual(uri, "smb://server/share/path/to/document")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock

from nexum.document.storage.client.ftp import FTPFileClient


class TestFTPFileClient(unittest.TestCase):
    @patch("nexum.document.storage.client.ftp.FTP")
    def test_file_exists_true(self, mock_ftp):
        ftp = MagicMock()
        ftp.nlst.return_value = ["document.txt"]
        mock_ftp.return_value = ftp

        client = FTPFileClient("host", "user", "pass")
        self.assertTrue(client.file_exists("dir/document.txt"))

    @patch("nexum.document.storage.client.ftp.FTP")
    def test_file_exists_false(self, mock_ftp):
        ftp = MagicMock()
        ftp.nlst.return_value = ["other.txt"]
        mock_ftp.return_value = ftp

        client = FTPFileClient("host", "user", "pass")
        self.assertFalse(client.file_exists("dir/document.txt"))

    @patch("nexum.document.storage.client.ftp.FTP")
    def test_load_bytes(self, mock_ftp):
        ftp = MagicMock()

        def retrbinary(cmd, callback):
            callback(b"abc")

        ftp.retrbinary.side_effect = retrbinary
        mock_ftp.return_value = ftp

        client = FTPFileClient("host", "user", "pass")
        data = client.load_bytes("dir/document.txt")
        self.assertEqual(data, b"abc")

    @patch("nexum.document.storage.client.ftp.FTP")
    def test_stream(self, mock_ftp):
        conn = MagicMock()
        conn.recv.side_effect = [b"chunk1", b"chunk2", b""]

        ftp = MagicMock()
        ftp.transfercmd.return_value.__enter__.return_value = conn
        mock_ftp.return_value = ftp

        client = FTPFileClient("host", "user", "pass")
        chunks = list(client.stream("dir/document.txt"))
        self.assertEqual(chunks, [b"chunk1", b"chunk2"])

    def test_get_uri(self):
        client = FTPFileClient("host")
        self.assertEqual(client.get_uri("dir/document.txt"), "ftp://host/dir/document.txt")

    @patch("nexum.document.storage.client.ftp.FTP", side_effect=Exception("fail"))
    def test_connect_failure(self, mock_ftp):
        client = FTPFileClient("host")
        with self.assertRaises(Exception):
            client.file_exists("dir/document.txt")


if __name__ == "__main__":
    unittest.main()

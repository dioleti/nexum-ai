import unittest
from unittest.mock import MagicMock, patch

from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.s3 import S3Client
from nexum.multicloud.storage.loader.s3 import S3Loader


class TestS3Loader(unittest.TestCase):
    def test_parse_path_valid(self):
        loader = S3Loader("s3://bucket/key.txt", client=MagicMock())
        self.assertEqual(loader.container, "bucket")
        self.assertEqual(loader.blob, "key.txt")

    def test_parse_path_invalid_prefix(self):
        with self.assertRaises(NexumValueError):
            S3Loader("wrong://bucket/key", client=MagicMock())

    def test_parse_path_missing_key(self):
        with self.assertRaises(NexumValueError):
            S3Loader("s3://bucket", client=MagicMock())

    def test_empty_key_raises(self):
        with self.assertRaises(NexumValueError):
            S3Loader("s3://bucket/", client=MagicMock())

    @patch("nexum.multicloud.storage.loader.s3.S3Client")
    def test_init_creates_client(self, mock_client):
        loader = S3Loader("s3://bucket/key.txt")
        mock_client.assert_called_once()
        self.assertEqual(loader.container, "bucket")
        self.assertEqual(loader.blob, "key.txt")

    def test_init_with_client(self):
        client = MagicMock(spec=S3Client)
        loader = S3Loader("s3://bucket/key.txt", client=client)
        self.assertIs(loader.client, client)
        self.assertEqual(loader.container, "bucket")
        self.assertEqual(loader.blob, "key.txt")


if __name__ == "__main__":
    unittest.main()

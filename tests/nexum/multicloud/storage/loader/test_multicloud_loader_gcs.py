import unittest
from unittest.mock import MagicMock, patch

from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.gcs import GCSClient
from nexum.multicloud.storage.loader.gcs import GCSLoader


class TestGCSLoader(unittest.TestCase):
    def test_parse_path_valid(self):
        loader = GCSLoader("gs://bucket/blob.txt", client=MagicMock())
        self.assertEqual(loader.container, "bucket")
        self.assertEqual(loader.blob, "blob.txt")

    def test_parse_path_invalid_prefix(self):
        with self.assertRaises(NexumValueError):
            GCSLoader("wrong://bucket/blob", client=MagicMock())

    def test_parse_path_missing_blob(self):
        with self.assertRaises(NexumValueError):
            GCSLoader("gs://bucket", client=MagicMock())

    @patch("nexum.multicloud.storage.loader.gcs.GCSClient")
    def test_init_creates_client(self, mock_client):
        loader = GCSLoader("gs://bucket/blob.txt")
        mock_client.assert_called_once()
        self.assertEqual(loader.container, "bucket")
        self.assertEqual(loader.blob, "blob.txt")

    def test_init_with_client(self):
        client = MagicMock(spec=GCSClient)
        loader = GCSLoader("gs://bucket/blob.txt", client=client)
        self.assertIs(loader.client, client)
        self.assertEqual(loader.container, "bucket")
        self.assertEqual(loader.blob, "blob.txt")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.azure import AzureClient
from nexum.multicloud.storage.loader.azure import AzureLoader


class TestAzureLoader(unittest.TestCase):
    def test_parse_path_valid(self):
        loader = AzureLoader(
            "azure://acc/container/blob.txt",
            client=MagicMock()
        )
        self.assertEqual(loader.container, "container")
        self.assertEqual(loader.blob, "blob.txt")

    def test_parse_path_invalid_prefix(self):
        with self.assertRaises(NexumValueError):
            AzureLoader("wrong://acc/container/blob", client=MagicMock())

    def test_parse_path_missing_parts(self):
        with self.assertRaises(NexumValueError):
            AzureLoader("azure://acc/container", client=MagicMock())

    @patch("nexum.multicloud.storage.loader.azure.AzureClient")
    def test_init_creates_client(self, mock_client):
        loader = AzureLoader("azure://acc/container/blob.txt")
        mock_client.assert_called_once_with(account_url="https://acc")
        self.assertEqual(loader.container, "container")
        self.assertEqual(loader.blob, "blob.txt")

    def test_init_with_client(self):
        client = MagicMock(spec=AzureClient)
        loader = AzureLoader("azure://acc/container/blob.txt", client=client)
        self.assertIs(loader.client, client)
        self.assertEqual(loader.container, "container")
        self.assertEqual(loader.blob, "blob.txt")


if __name__ == "__main__":
    unittest.main()

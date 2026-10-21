import unittest
from unittest.mock import patch, MagicMock

from nexum.multicloud.storage.client.azure import AzureClient


class TestAzureClient(unittest.TestCase):
    @patch("nexum.multicloud.storage.client.azure.DefaultAzureCredential")
    @patch("nexum.multicloud.storage.client.azure.BlobServiceClient")
    def test_init_default_credential(self, mock_blob, mock_default):
        client = AzureClient(account_url="https://acc.blob.core.windows.net")
        mock_default.assert_called_once()
        mock_blob.assert_called_once_with(
            account_url="https://acc.blob.core.windows.net",
            credential=mock_default.return_value,
        )
        self.assertEqual(client.account_url, "https://acc.blob.core.windows.net")

    @patch("nexum.multicloud.storage.client.azure.AzureNamedKeyCredential")
    @patch("nexum.multicloud.storage.client.azure.BlobServiceClient")
    def test_init_account_key(self, mock_blob, mock_key):
        client = AzureClient(
            account_url="https://acc.blob.core.windows.net",
            account_key="KEY123",
        )
        mock_key.assert_called_once()
        mock_blob.assert_called_once()

    @patch("nexum.multicloud.storage.client.azure.ClientSecretCredential")
    @patch("nexum.multicloud.storage.client.azure.BlobServiceClient")
    def test_init_client_secret(self, mock_blob, mock_secret):
        client = AzureClient(
            account_url="https://acc.blob.core.windows.net",
            tenant_id="tid",
            client_id="cid",
            client_secret="secret",
        )
        mock_secret.assert_called_once_with(
            tenant_id="tid",
            client_id="cid",
            client_secret="secret",
        )
        mock_blob.assert_called_once()

    @patch("nexum.multicloud.storage.client.azure.ManagedIdentityCredential")
    @patch("nexum.multicloud.storage.client.azure.BlobServiceClient")
    def test_init_managed_identity(self, mock_blob, mock_mi):
        client = AzureClient(
            account_url="https://acc.blob.core.windows.net",
            use_managed_identity=True,
        )
        mock_mi.assert_called_once()
        mock_blob.assert_called_once()

    @patch("nexum.multicloud.storage.client.azure.BlobServiceClient.from_connection_string")
    def test_init_connection_string(self, mock_from):
        client = AzureClient(connection_string="UseDevelopmentStorage=true")
        mock_from.assert_called_once_with("UseDevelopmentStorage=true")

    @patch("nexum.multicloud.storage.client.azure.BlobServiceClient")
    def test_get_blob(self, mock_blob):
        container_client = MagicMock()
        blob_client = MagicMock()
        blob_client.download_blob.return_value.readall.return_value = b"abc"

        container_client.get_blob_client.return_value = blob_client
        mock_blob.return_value.get_container_client.return_value = container_client

        client = AzureClient(account_url="https://acc.blob.core.windows.net")
        data = client.get_blob("cont", "document.txt")
        self.assertEqual(data, b"abc")

    @patch("nexum.multicloud.storage.client.azure.BlobServiceClient")
    def test_blob_exists(self, mock_blob):
        container_client = MagicMock()
        blob_client = MagicMock()
        blob_client.exists.return_value = True

        container_client.get_blob_client.return_value = blob_client
        mock_blob.return_value.get_container_client.return_value = container_client

        client = AzureClient(account_url="https://acc.blob.core.windows.net")
        self.assertTrue(client.blob_exists("cont", "document.txt"))

    def test_get_uri(self):
        client = AzureClient(account_url="https://acc.blob.core.windows.net")
        uri = client.get_uri("cont", "document.txt")
        self.assertEqual(uri, "https://acc.blob.core.windows.net/cont/document.txt")

    @patch("nexum.multicloud.storage.client.azure.BlobServiceClient")
    def test_stream_blob(self, mock_blob):
        container_client = MagicMock()
        blob_client = MagicMock()

        downloader = MagicMock()
        downloader.chunks.return_value = [b"a", memoryview(b"b"), bytearray(b"c")]

        blob_client.download_blob.return_value = downloader
        container_client.get_blob_client.return_value = blob_client
        mock_blob.return_value.get_container_client.return_value = container_client

        client = AzureClient(account_url="https://acc.blob.core.windows.net")
        chunks = list(client.stream_blob("cont", "document.txt"))
        self.assertEqual(chunks, [b"a", b"b", b"c"])


if __name__ == "__main__":
    unittest.main()

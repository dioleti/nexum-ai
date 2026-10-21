import unittest
from unittest.mock import patch, MagicMock

from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.gcs import GCSClient


class TestGCSClient(unittest.TestCase):
    @patch("google.auth.default", return_value=(MagicMock(), "proj"))
    @patch("google.auth.compute_engine._metadata.ping", side_effect=Exception)
    @patch("google.auth.transport.requests.Request", MagicMock)
    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    @patch("nexum.multicloud.storage.client.gcs.DefaultAzureCredential", create=True)
    def test_init_default_credentials(self, mock_default_azure, mock_client, *_):
        mock_default_azure.return_value = MagicMock()
        client = GCSClient(project="proj")
        mock_client.assert_called_once_with(
            credentials=client.credentials,
            project="proj"
        )

    @patch("nexum.multicloud.storage.client.gcs.service_account.Credentials.from_service_account_info")
    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_credentials_dict(self, mock_client, mock_creds):
        mock_creds.return_value = MagicMock()
        client = GCSClient(project="proj", credentials={"a": 1})
        mock_creds.assert_called_once_with({"a": 1})
        mock_client.assert_called_once()

    @patch("nexum.multicloud.storage.client.gcs.service_account.Credentials.from_service_account_file")
    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_credentials_file(self, mock_client, mock_creds):
        mock_creds.return_value = MagicMock()
        client = GCSClient(project="proj", credentials_file="document.json")
        mock_creds.assert_called_once_with("document.json")
        mock_client.assert_called_once()

    @patch("google.auth.impersonated_credentials.Credentials")
    @patch("nexum.multicloud.storage.client.gcs.service_account.Credentials.from_service_account_info")
    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_impersonation(self, mock_client, mock_creds, mock_imp):
        mock_creds.return_value = MagicMock()
        mock_imp.return_value = MagicMock()

        client = GCSClient(
            project="proj",
            credentials={"a": 1},
            impersonate_service_account="svc@google.com"
        )

        mock_imp.assert_called_once()
        mock_client.assert_called_once()

    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_get_blob_success(self, mock_client):
        bucket = MagicMock()
        blob = MagicMock()
        blob.exists.return_value = True
        blob.download_as_bytes.return_value = b"abc"

        bucket.blob.return_value = blob
        mock_client.return_value.bucket.return_value = bucket

        client = GCSClient(project="proj")
        data = client.get_blob("cont", "document.txt")
        self.assertEqual(data, b"abc")

    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_get_blob_not_found(self, mock_client):
        bucket = MagicMock()
        blob = MagicMock()
        blob.exists.return_value = False

        bucket.blob.return_value = blob
        mock_client.return_value.bucket.return_value = bucket

        client = GCSClient(project="proj")
        with self.assertRaises(NexumValueError):
            client.get_blob("cont", "document.txt")

    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_blob_exists(self, mock_client):
        bucket = MagicMock()
        blob = MagicMock()
        blob.exists.return_value = True

        bucket.blob.return_value = blob
        mock_client.return_value.bucket.return_value = bucket

        client = GCSClient(project="proj")
        self.assertTrue(client.blob_exists("cont", "document.txt"))

    @patch("google.auth.default", return_value=(MagicMock(), "proj"))
    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_get_uri(self, mock_client, *_):
        client = GCSClient(project="proj")
        uri = client.get_uri("cont", "document.txt")
        self.assertEqual(uri, "gs://cont/document.txt")

    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_stream_blob(self, mock_client):
        bucket = MagicMock()
        blob = MagicMock()
        blob.exists.return_value = True

        reader = MagicMock()
        reader.read.side_effect = [b"a", b"b", b""]

        blob.open.return_value = reader
        bucket.blob.return_value = blob
        mock_client.return_value.bucket.return_value = bucket

        client = GCSClient(project="proj")
        chunks = list(client.stream_blob("cont", "document.txt"))
        self.assertEqual(chunks, [b"a", b"b"])

    @patch("nexum.multicloud.storage.client.gcs.storage.Client")
    def test_stream_blob_not_found(self, mock_client):
        bucket = MagicMock()
        blob = MagicMock()
        blob.exists.return_value = False

        bucket.blob.return_value = blob
        mock_client.return_value.bucket.return_value = bucket

        client = GCSClient(project="proj")
        with self.assertRaises(NexumValueError):
            list(client.stream_blob("cont", "document.txt"))


if __name__ == "__main__":
    unittest.main()

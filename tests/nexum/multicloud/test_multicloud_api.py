import unittest
from unittest.mock import MagicMock, patch

from nexum.common.errors import NexumValueError
from nexum.multicloud import api


class TestMulticloudAPI(unittest.TestCase):
    @patch("nexum.multicloud.api._create_loader_from_uri")
    def test_read_stream_exists_uri(self, mock_create):
        mock_loader = MagicMock()
        mock_loader.load.return_value = b"data"
        mock_loader.stream.return_value = iter([b"chunk"])
        mock_loader.exists.return_value = True
        mock_loader.get_uri.return_value = "s3://bucket/key"
        mock_create.return_value = mock_loader

        self.assertEqual(api.read("s3://bucket/key"), b"data")
        self.assertEqual(list(api.stream("s3://bucket/key")), [b"chunk"])
        self.assertTrue(api.exists("s3://bucket/key"))
        self.assertEqual(api.uri("s3://bucket/key"), "s3://bucket/key")

    @patch("nexum.multicloud.storage.loader.s3.S3Loader")
    @patch("nexum.multicloud.storage.client.s3.S3Client")
    def test_create_loader_s3(self, mock_s3_client, mock_s3_loader):
        api._create_loader_from_uri("s3://mybucket/mykey")
        mock_s3_client.assert_called_once()
        mock_s3_loader.assert_called_once_with("mybucket/mykey", client=mock_s3_client.return_value)

    @patch("nexum.multicloud.storage.loader.azure.AzureLoader")
    @patch("nexum.multicloud.storage.client.azure.AzureClient")
    def test_create_loader_azure(self, mock_azure_client, mock_azure_loader):
        api._create_loader_from_uri("azure://myacc/cont/blob")
        mock_azure_client.assert_called_once_with(
            account_url="https://myacc.blob.core.windows.net",
            use_managed_identity=False,
        )
        mock_azure_loader.assert_called_once_with("myacc/cont/blob", client=mock_azure_client.return_value)

    @patch("nexum.multicloud.storage.loader.gcs.GCSLoader")
    @patch("nexum.multicloud.storage.client.gcs.GCSClient")
    def test_create_loader_gcs(self, mock_gcs_client, mock_gcs_loader):
        api._create_loader_from_uri("gcs://mybucket/myblob")
        mock_gcs_client.assert_called_once_with()
        mock_gcs_loader.assert_called_once_with("mybucket/myblob", client=mock_gcs_client.return_value)

    @patch.dict("os.environ", {"SMB_USERNAME": "u", "SMB_PASSWORD": "p"})
    @patch("nexum.document.storage.loader.smb.SMBFileLoader")
    @patch("nexum.document.storage.client.smb.SMBFileClient")
    def test_create_loader_smb(self, mock_smb_client, mock_smb_loader):
        api._create_loader_from_uri("smb://srv/shr/dir/file.txt")
        mock_smb_client.assert_called_once_with("srv", "shr", username="u", password="p")
        mock_smb_loader.assert_called_once_with("dir/file.txt", client=mock_smb_client.return_value)

    def test_create_loader_unsupported_provider(self):
        with self.assertRaises(NexumValueError):
            api._create_loader_from_uri("ftp://server/file.txt")

    def test_parse_smb_uri_invalid(self):
        with self.assertRaises(NexumValueError):
            api._parse_smb_uri("srv/shr")

    @patch.dict("os.environ", {}, clear=True)
    def test_auto_user_and_pass_missing(self):
        with self.assertRaises(NexumValueError):
            api._auto_user()
        with self.assertRaises(NexumValueError):
            api._auto_pass()


if __name__ == "__main__":
    unittest.main()

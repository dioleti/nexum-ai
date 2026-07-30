import unittest
from unittest.mock import patch, MagicMock

from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.s3 import S3Client


class TestS3Client(unittest.TestCase):
    def test_init_missing_region(self):
        with self.assertRaises(NexumValueError):
            S3Client()

    @patch("nexum.multicloud.storage.client.s3.boto3.Session")
    def test_init_with_access_keys(self, mock_session):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        mock_session.return_value = session

        s3 = S3Client(
            region_name="us-east-1",
            aws_access_key_id="AKIA",
            aws_secret_access_key="SECRET",
            endpoint_url="http://localhost"
        )

        session.client.assert_called_once_with(
            "s3",
            region_name="us-east-1",
            endpoint_url="http://localhost",
            aws_access_key_id="AKIA",
            aws_secret_access_key="SECRET"
        )

        self.assertIs(s3.client, client)

    @patch("nexum.multicloud.storage.client.s3.boto3.Session")
    def test_init_assume_role(self, mock_session):
        session = MagicMock()
        sts = MagicMock()
        s3_client = MagicMock()

        sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA",
                "SecretAccessKey": "SECRET",
                "SessionToken": "TOKEN",
            }
        }

        def client_side_effect(service_name, *args, **kwargs):
            if service_name == "sts":
                return sts
            if service_name == "s3":
                return s3_client
            raise Exception("Unexpected service")

        session.client.side_effect = client_side_effect
        mock_session.return_value = session

        s3 = S3Client(
            region_name="us-east-1",
            assume_role_arn="arn:aws:iam::123:role/MyRole"
        )

        sts.assume_role.assert_called_once()

    @patch("nexum.multicloud.storage.client.s3.boto3.Session")
    def test_init_assume_role(self, mock_session):
        session = MagicMock()
        sts = MagicMock()
        s3_client = MagicMock()

        sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA",
                "SecretAccessKey": "SECRET",
                "SessionToken": "TOKEN",
            }
        }

        def client_side_effect(service_name, *args, **kwargs):
            if service_name == "sts":
                return sts
            if service_name == "s3":
                return s3_client
            raise Exception("Unexpected service")

        session.client.side_effect = client_side_effect
        mock_session.return_value = session

        s3 = S3Client(
            region_name="us-east-1",
            assume_role_arn="arn:aws:iam::123:role/MyRole"
        )

        sts.assume_role.assert_called_once()

    @patch("nexum.multicloud.storage.client.s3.boto3.Session")
    def test_init_default_credentials(self, mock_session):
        session = MagicMock()
        client = MagicMock()
        session.client.return_value = client
        mock_session.return_value = session

        s3 = S3Client(region_name="us-east-1")
        session.client.assert_called_once_with(
            "s3",
            region_name="us-east-1",
            endpoint_url=None
        )

    def test_get_blob_success(self):
        client = MagicMock()
        body = MagicMock()
        body.read.return_value = b"abc"

        client.get_object.return_value = {"Body": body}

        s3 = S3Client(region_name="us-east-1", client=client)
        data = s3.get_blob("bucket", "document.txt")
        self.assertEqual(data, b"abc")

    def test_get_blob_not_found(self):
        client = MagicMock()
        client.exceptions = MagicMock()
        client.exceptions.NoSuchKey = Exception

        client.get_object.side_effect = client.exceptions.NoSuchKey

        s3 = S3Client(region_name="us-east-1", client=client)

        with self.assertRaises(NexumValueError):
            s3.get_blob("bucket", "document.txt")

    def test_stream_blob_success(self):
        client = MagicMock()
        body = MagicMock()
        body.iter_chunks.return_value = [b"a", memoryview(b"b"), bytearray(b"c")]

        client.get_object.return_value = {"Body": body}

        s3 = S3Client(region_name="us-east-1", client=client)
        chunks = list(s3.stream_blob("bucket", "document.txt"))
        self.assertEqual(chunks, [b"a", b"b", b"c"])

    def test_stream_blob_not_found(self):
        client = MagicMock()
        client.exceptions = MagicMock()
        client.exceptions.NoSuchKey = Exception

        client.get_object.side_effect = client.exceptions.NoSuchKey

        s3 = S3Client(region_name="us-east-1", client=client)

        with self.assertRaises(NexumValueError):
            list(s3.stream_blob("bucket", "document.txt"))

    def test_blob_exists_true(self):
        client = MagicMock()
        client.head_object.return_value = {}

        s3 = S3Client(region_name="us-east-1", client=client)
        self.assertTrue(s3.blob_exists("bucket", "document.txt"))

    def test_blob_exists_false(self):
        client = MagicMock()
        client.exceptions = MagicMock()
        client.exceptions.ClientError = Exception

        client.head_object.side_effect = client.exceptions.ClientError

        s3 = S3Client(region_name="us-east-1", client=client)
        self.assertFalse(s3.blob_exists("bucket", "document.txt"))

    def test_get_uri(self):
        s3 = S3Client(region_name="us-east-1", client=MagicMock())
        uri = s3.get_uri("bucket", "document.txt")
        self.assertEqual(uri, "s3://bucket/document.txt")


if __name__ == "__main__":
    unittest.main()

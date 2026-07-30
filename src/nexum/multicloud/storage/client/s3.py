from typing import Any

from nexum.common.errors import NexumValueError

try:
    import boto3
    from mypy_boto3_s3.client import S3Client as S3ServiceClient
except ImportError as e:
    raise ImportError(
        "AWS S3 support is not installed. Install it with: pip install nexum-ai[aws]"
    ) from e

from nexum.multicloud.storage.client.base import StorageClient


class S3Client(StorageClient):
    def __init__(
        self,
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        region_name: str | None = None,
        assume_role_arn: str | None = None,
        endpoint_url: str | None = None,
        client: S3ServiceClient | None = None,
    ):
        if region_name is None:
            raise NexumValueError("region_name must be provided for S3 access")

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.profile_name = profile_name
        self.region_name = region_name
        self.assume_role_arn = assume_role_arn
        self.endpoint_url = endpoint_url
        if client is None:
            resolved = self._resolve_credentials()
            client = resolved["session"].client("s3", **resolved["client_kwargs"])
        self.client: S3ServiceClient = client

    def _resolve_credentials(self) -> dict[str, Any]:
        if self.profile_name:
            session = boto3.Session(profile_name=self.profile_name)
        else:
            session = boto3.Session()

        if self.aws_access_key_id and self.aws_secret_access_key:
            creds = {
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
            }
            if self.aws_session_token:
                creds["aws_session_token"] = self.aws_session_token

            return {
                "session": session,
                "client_kwargs": {
                    "region_name": self.region_name,
                    "endpoint_url": self.endpoint_url,
                    **creds,
                },
            }

        if self.assume_role_arn:
            sts = session.client("sts", region_name=self.region_name)
            assumed = sts.assume_role(
                RoleArn=self.assume_role_arn,
                RoleSessionName="s3_loader_session",
            )

            return {
                "session": session,
                "client_kwargs": {
                    "region_name": self.region_name,
                    "endpoint_url": self.endpoint_url,
                    "aws_access_key_id": assumed["Credentials"]["AccessKeyId"],
                    "aws_secret_access_key": assumed["Credentials"]["SecretAccessKey"],
                    "aws_session_token": assumed["Credentials"]["SessionToken"],
                },
            }

        return {
            "session": session,
            "client_kwargs": {
                "region_name": self.region_name,
                "endpoint_url": self.endpoint_url,
            },
        }

    def get_blob(self, container: str, blob: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=container, Key=blob)
            return response["Body"].read()
        except Exception as exc:
            raise NexumValueError("Blob not found") from exc

    def stream_blob(self, container: str, blob: str):
        try:
            response = self.client.get_object(Bucket=container, Key=blob)
            body = response["Body"]

            def generator():
                for chunk in body.iter_chunks():
                    if isinstance(chunk, memoryview):
                        yield chunk.tobytes()
                    elif isinstance(chunk, bytearray):
                        yield bytes(chunk)
                    elif isinstance(chunk, bytes):
                        yield chunk
                    else:
                        yield bytes(chunk)

            return generator()

        except Exception as exc:
            raise NexumValueError("Blob not found") from exc

    def blob_exists(self, container: str, blob: str) -> bool:
        try:
            self.client.head_object(Bucket=container, Key=blob)
            return True
        except Exception:
            return False

    def get_uri(self, container: str, blob: str) -> str:
        return f"s3://{container}/{blob}"

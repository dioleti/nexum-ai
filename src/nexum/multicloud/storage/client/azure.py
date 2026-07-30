from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from nexum.common.errors import NexumValueError

try:
    from azure.core.credentials import AzureNamedKeyCredential
    from azure.identity import (
        ClientSecretCredential,
        DefaultAzureCredential,
        ManagedIdentityCredential,
    )
    from azure.storage.blob import BlobServiceClient
except ImportError as e:
    raise ImportError(
        "Azure support is not installed. Install it with: pip install nexum-ai[azure]"
    ) from e

from nexum.multicloud.storage.client.base import StorageClient


class AzureClient(StorageClient):
    def __init__(
        self,
        *,
        connection_string: str | None = None,
        account_url: str | None = None,
        sas_token: str | None = None,
        account_key: str | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        use_managed_identity: bool = False,
    ):
        self.connection_string: str | None = connection_string
        self.account_url: str | None = account_url
        self.sas_token: str | None = sas_token
        self.account_key: str | None = account_key
        self.tenant_id: str | None = tenant_id
        self.client_id: str | None = client_id
        self.client_secret: str | None = client_secret
        self.use_managed_identity: bool = use_managed_identity
        self.credential: Any = None

        if self.connection_string:
            self.account_url = None
            self.sas_token = None
            self.account_key = None
            self.tenant_id = None
            self.client_id = None
            self.client_secret = None
            self.use_managed_identity = False
            self.credential = None
            self.client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            return

        if not self.account_url:
            raise NexumValueError(
                "account_url must be provided when connection_string is not used"
            )

        self.credential = self._resolve_credential()
        self.client = BlobServiceClient(
            account_url=self.account_url,
            credential=self.credential,
        )

    def _resolve_credential(self):
        if self.sas_token:
            if not self.account_url:
                raise NexumValueError("account_url is required when using sas_token")
            return self.sas_token

        if self.account_key:
            if not self.account_url:
                raise NexumValueError("account_url is required when using account_key")
            parsed = urlparse(self.account_url)
            if not parsed.hostname:
                raise NexumValueError("Invalid account_url format")
            account_name = parsed.hostname.split(".")[0]
            return AzureNamedKeyCredential(account_name, self.account_key)

        if self.tenant_id and self.client_id and self.client_secret:
            return ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )

        if self.use_managed_identity:
            if not self.account_url:
                raise NexumValueError(
                    "account_url is required when using managed identity"
                )
            return ManagedIdentityCredential()

        if not self.account_url:
            raise NexumValueError(
                "account_url is required when using default credentials"
            )

        return DefaultAzureCredential()

    def _create_client(self) -> BlobServiceClient:
        if self.connection_string:
            return BlobServiceClient.from_connection_string(self.connection_string)

        if not self.account_url:
            raise NexumValueError(
                "account_url must be provided when connection_string is not used"
            )

        if isinstance(self.credential, str):
            return BlobServiceClient(
                account_url=self.account_url,
                credential=None,
                sas_token=self.credential,
            )

        return BlobServiceClient(
            account_url=self.account_url,
            credential=self.credential,
        )

    def get_blob(self, container: str, blob: str) -> bytes:
        container_client = self.client.get_container_client(container)
        blob_client = container_client.get_blob_client(blob)
        return blob_client.download_blob().readall()

    def blob_exists(self, container: str, blob: str) -> bool:
        container_client = self.client.get_container_client(container)
        blob_client = container_client.get_blob_client(blob)
        return blob_client.exists()

    def get_uri(self, container: str, blob: str) -> str:
        return f"{self.account_url}/{container}/{blob}"

    def stream_blob(self, container: str, blob: str) -> Iterator[bytes]:
        container_client = self.client.get_container_client(container)
        blob_client = container_client.get_blob_client(blob)
        downloader = blob_client.download_blob()

        def generator() -> Iterator[bytes]:
            for chunk in downloader.chunks():
                if isinstance(chunk, memoryview):
                    yield chunk.tobytes()
                elif isinstance(chunk, bytes):
                    yield chunk
                else:
                    yield bytes(chunk)

        return generator()

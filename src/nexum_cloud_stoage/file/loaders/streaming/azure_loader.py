from typing import Optional, Generator, Union
from urllib.parse import urlparse

from azure.identity import (
    DefaultAzureCredential,
    ClientSecretCredential,
    ManagedIdentityCredential,
)
from azure.storage.blob import (
    BlobServiceClient,
    AzureNamedKeyCredential
)
from nexum.errors import NexumValueError, NexumRuntimeError

from src.nexum_cloud_stoage.file.loaders.bsae_stream_loader import BaseStreamLoader

CredentialType = Union[
    str,
    ClientSecretCredential,
    ManagedIdentityCredential,
    DefaultAzureCredential,
    AzureNamedKeyCredential
]


class AzureStreamLoader(BaseStreamLoader):
    def __init__(
        self,
        path: Optional[str] = None,
        *,
        chunk_size: int = 65536,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        container_name: Optional[str] = None,
        blob_name: Optional[str] = None,
        sas_token: Optional[str] = None,
        account_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        use_managed_identity: bool = False
    ):
        self.chunk_size = chunk_size
        self.connection_string = connection_string

        self.sas_token = sas_token
        self.account_key = account_key
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.use_managed_identity = use_managed_identity

        if path:
            self._parse_path(path)
        else:
            if account_url is None or container_name is None or blob_name is None:
                raise NexumValueError("This data is necessary: account_url, container_name, blob_name")

            self.account_url = account_url
            self.container_name = container_name
            self.blob_name = blob_name

        assert self.account_url is not None
        assert self.container_name is not None
        assert self.blob_name is not None

        self.credential = self._resolve_credential()
        self.client = self._create_client()

    def _parse_path(self, path: str):
        if not path.startswith("azure://"):
            raise NexumValueError(
                "Invalid Azure path format. Expected azure://account/container/blob"
            )

        clean = path.replace("azure://", "")
        parts = clean.split("/", 2)

        if len(parts) != 3:
            raise NexumValueError(
                "Invalid Azure path. Expected azure://account/container/blob"
            )

        self.account_url = f"https://{parts[0]}"
        self.container_name = parts[1]
        self.blob_name = parts[2]

    def _resolve_credential(self) -> CredentialType:
        if self.sas_token:
            return self.sas_token

        if self.account_key:
            parsed = urlparse(self.account_url)
            account_name = parsed.hostname.split(".")[0]
            return AzureNamedKeyCredential(account_name, self.account_key)

        if self.tenant_id and self.client_id and self.client_secret:
            return ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )

        if self.use_managed_identity:
            return ManagedIdentityCredential()

        return DefaultAzureCredential()

    def _create_client(self) -> BlobServiceClient:
        if self.connection_string:
            return BlobServiceClient.from_connection_string(self.connection_string)

        assert self.account_url is not None

        if isinstance(self.credential, str):
            return BlobServiceClient(
                account_url=self.account_url,
                credential=None,
                sas_token=self.credential
            )

        return BlobServiceClient(
            account_url=self.account_url,
            credential=self.credential
        )

    def stream(self) -> Generator[bytes, None, None]:
        """
        Stream bytes from Azure Blob Storage using a custom chunk size.

        This implementation uses `readinto()` to avoid loading the entire
        blob into memory, providing true streaming behavior.
        """
        try:
            assert self.container_name is not None
            assert self.blob_name is not None

            container = self.client.get_container_client(self.container_name)
            blob = container.get_blob_client(self.blob_name)
            downloader = blob.download_blob()
        except Exception as e:
            raise NexumRuntimeError(
                f"Failed to open Azure blob '{self.blob_name}': {e}"
            )

        try:
            buffer = bytearray(self.chunk_size)
            view = memoryview(buffer)

            stream = downloader.readinto

            while True:
                bytes_read = stream(view)
                if bytes_read == 0:
                    break
                yield buffer[:bytes_read]

        except Exception as e:
            raise NexumRuntimeError(
                f"Error while streaming Azure blob '{self.blob_name}': {e}"
            )

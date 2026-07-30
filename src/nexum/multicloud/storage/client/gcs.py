from nexum.common.errors import NexumValueError

try:
    from google.auth.credentials import Credentials
    from google.cloud import storage
    from google.cloud.storage import Client
    from google.oauth2 import service_account
except ImportError as e:
    raise ImportError(
        "GCS support is not installed. Install it with: pip install nexum-ai[gcs]"
    ) from e

from nexum.multicloud.storage.client.base import StorageClient


class GCSClient(StorageClient):
    def __init__(
        self,
        *,
        project: str | None = None,
        credentials: dict | None = None,
        credentials_file: str | None = None,
        impersonate_service_account: str | None = None,
        scopes_override: list | None = None,
        client: Client = None,
    ):
        self.project = project
        self.credentials_dict = credentials
        self.credentials_file = credentials_file
        self.impersonate_service_account = impersonate_service_account
        self.scopes_override = scopes_override

        self.credentials = self._resolve_credentials()
        self.client = client or storage.Client(
            credentials=self.credentials, project=self.project
        )

    def _resolve_credentials(self) -> Credentials | None:
        creds = None

        if self.credentials_dict:
            creds = service_account.Credentials.from_service_account_info(
                self.credentials_dict
            )
        elif self.credentials_file:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_file
            )

        scopes = self.scopes_override or Client.SCOPE

        if self.impersonate_service_account:
            from google.auth import impersonated_credentials

            if creds is None:
                import google.auth

                creds, _ = google.auth.default(scopes=scopes)

            creds = impersonated_credentials.Credentials(
                source_credentials=creds,
                target_principal=self.impersonate_service_account,
                target_scopes=scopes,
                lifetime=3600,
            )

        return creds

    def get_blob(self, container: str, blob: str) -> bytes:
        bucket = self.client.bucket(container)
        blob_obj = bucket.blob(blob)
        if not blob_obj.exists():
            raise NexumValueError("Blob not found")
        return blob_obj.download_as_bytes()

    def stream_blob(self, container: str, blob: str):
        bucket = self.client.bucket(container)
        blob_obj = bucket.blob(blob)
        if not blob_obj.exists():
            raise NexumValueError("Blob not found")

        reader = blob_obj.open("rb")

        def generator():
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                yield chunk

        return generator()

    def blob_exists(self, container: str, blob: str) -> bool:
        bucket = self.client.bucket(container)
        blob_obj = bucket.blob(blob)
        return blob_obj.exists()

    def get_uri(self, container: str, blob: str) -> str:
        return f"gs://{container}/{blob}"

from typing import Optional, Dict, Generator

from google.auth.credentials import Credentials
from google.cloud import storage
from google.cloud.storage import Client
from google.oauth2 import service_account
from nexum.errors import NexumValueError, NexumRuntimeError


class GCSStreamLoader:
    """
    Loader responsible for streaming objects stored in Google Cloud Storage (GCS)
    using flexible authentication and incremental chunked reading.

    This class is suitable for processing large files (PDFs, images, binary datasets)
    without loading the entire blob into memory.
    """

    def __init__(
        self,
        path: str,
        *,
        chunk_size: int = 65536,
        project: Optional[str] = None,
        credentials: Optional[Dict] = None,
        credentials_file: Optional[str] = None,
        impersonate_service_account: Optional[str] = None,
        scopes_override: Optional[list] = None
    ):
        """
        Initialize the GCS stream loader.

        Parameters
        ----------
        path : str
            Full GCS path in the format `gs://bucket/blob`.
        chunk_size : int, optional
            Maximum size of each streamed chunk. Default is 64 KB.
        project : str, optional
            GCP project ID. If omitted, Google Cloud SDK may infer it.
        credentials : dict, optional
            Service account credentials provided as a dictionary.
        credentials_file : str, optional
            Path to a service account JSON file.
        impersonate_service_account : str, optional
            Email of a service account to impersonate.
        scopes_override : list, optional
            Custom OAuth scopes for impersonation.
        """
        self.chunk_size = chunk_size
        self.project = project
        self.credentials_dict = credentials
        self.credentials_file = credentials_file
        self.impersonate_service_account = impersonate_service_account
        self.scopes_override = scopes_override

        self.bucket_name, self.blob_name = self._parse_path(path)
        self.credentials = self._resolve_credential()
        self.client = storage.Client(credentials=self.credentials, project=self.project)

    def _parse_path(self, path: str) -> tuple[str, str]:
        """
        Validate and extract bucket and blob name from a GCS path.

        Returns
        -------
        tuple[str, str]
            The bucket name and blob name.

        Raises
        ------
        NexumValueError
            If the path does not follow the expected `gs://bucket/blob` format.
        """
        if not path.startswith("gs://") or "/" not in path[5:]:
            raise NexumValueError("Invalid GCS path format. Expected gs://bucket/blob")

        clean = path.replace("gs://", "")
        return clean.split("/", 1)

    def _resolve_credential(self) -> Optional[Credentials]:
        """
        Resolve authentication credentials based on the provided parameters.

        Supports:
        - Service account credentials as a dictionary
        - Service account credentials from a JSON file
        - Service account impersonation
        - Application Default Credentials (fallback)

        Returns
        -------
        Credentials or None
            The resolved credential object.
        """
        creds = None

        if self.credentials_dict:
            creds = service_account.Credentials.from_service_account_info(self.credentials_dict)
        elif self.credentials_file:
            creds = service_account.Credentials.from_service_account_file(self.credentials_file)

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

    def stream(self) -> Generator[bytes, None, None]:
        """
        Stream bytes from a Google Cloud Storage object using incremental chunking.

        This method provides a memory-efficient way to read large blobs by yielding
        their content in sequential chunks. Only one chunk is held in memory at a time.

        Streaming behavior
        ------------------
        - Chunks are yielded in the original blob order.
        - Each chunk has a maximum size defined by `self.chunk_size`.
        - Streaming continues until the entire blob is consumed.
        - No full-file buffering occurs.

        Returns
        -------
        Generator[bytes, None, None]
            A generator yielding raw bytes from the blob.

        Raises
        ------
        NexumRuntimeError
            If the blob cannot be opened or streamed due to authentication issues,
            network errors, permission problems, or other runtime failures.

        Examples
        --------
        Basic usage:

            loader = GCSStreamLoader("gs://my-bucket/file.pdf")
            for chunk in loader.stream():
                process(chunk)

        Using a service account file:

            loader = GCSStreamLoader(
                "gs://my-bucket/file.pdf",
                credentials_file="/path/to/key.json"
            )

        Impersonating another service account:

            loader = GCSStreamLoader(
                "gs://secure-bucket/data.bin",
                impersonate_service_account="analytics@project.iam.gserviceaccount.com"
            )

        Notes
        -----
        - This method does not implement automatic retries.
        - Errors raised during iteration will stop the generator immediately.
        """
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(self.blob_name)

        try:
            with blob.open("rb") as stream:
                while True:
                    chunk = stream.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            raise NexumRuntimeError(
                f"Failed to stream GCS blob 'gs://{self.bucket_name}/{self.blob_name}': {e}"
            )

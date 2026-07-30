import os

from nexum.common.errors import NexumValueError


def read(uri: str) -> bytes:
    loader = _create_loader_from_uri(uri)
    return loader.load()


def stream(uri: str):
    loader = _create_loader_from_uri(uri)
    return loader.stream()


def exists(uri: str) -> bool:
    loader = _create_loader_from_uri(uri)
    return loader.exists()


def uri(uri: str) -> str:
    loader = _create_loader_from_uri(uri)
    return loader.get_uri()


def _create_loader_from_uri(uri: str):
    provider, path = uri.split("://", 1)

    if provider == "s3":
        from nexum.multicloud.storage.client.s3 import S3Client
        from nexum.multicloud.storage.loader.s3 import S3Loader

        return S3Loader(path, client=S3Client(region_name=_auto_region()))

    if provider == "azure":
        from nexum.multicloud.storage.client.azure import AzureClient
        from nexum.multicloud.storage.loader.azure import AzureLoader

        return AzureLoader(
            path,
            client=AzureClient(
                account_url=_auto_discover_azure_url(path),
                use_managed_identity=bool(os.getenv("AZURE_USE_MANAGED_IDENTITY")),
            ),
        )

    if provider == "gcs":
        from nexum.multicloud.storage.client.gcs import GCSClient
        from nexum.multicloud.storage.loader.gcs import GCSLoader

        return GCSLoader(path, client=GCSClient())

    if provider == "smb":
        from nexum.document.storage.client.smb import SMBFileClient
        from nexum.document.storage.loader.smb import SMBFileLoader

        server, share, file_path = _parse_smb_uri(path)
        return SMBFileLoader(
            file_path,
            client=SMBFileClient(
                server, share, username=_auto_user(), password=_auto_pass()
            ),
        )

    raise NexumValueError(f"Unsupported provider: {provider}")


def _auto_region():
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def _parse_smb_uri(path: str):
    parts = path.split("/", 2)
    if len(parts) < 3:
        raise NexumValueError("Invalid SMB URI. Expected smb://server/share/path")
    server, share, file_path = parts
    return server, share, file_path


def _auto_discover_azure_url(path: str) -> str:
    account = path.split("/", 1)[0]
    return f"https://{account}.blob.core.windows.net"


def _auto_user():
    user = os.getenv("SMB_USERNAME")
    if not user:
        raise NexumValueError("SMB_USERNAME environment variable not set")
    return user


def _auto_pass():
    pwd = os.getenv("SMB_PASSWORD")
    if not pwd:
        raise NexumValueError("SMB_PASSWORD environment variable not set")
    return pwd

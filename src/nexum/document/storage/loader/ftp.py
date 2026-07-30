from nexum.common.errors import NexumValueError
from nexum.document.storage.client.ftp import FTPFileClient
from nexum.document.storage.loader.base import FileLoader


class FTPFileLoader(FileLoader):
    def __init__(
        self,
        path: str,
        *,
        client: FTPFileClient | None = None,
        host: str = "",
        user: str = "",
        password: str = "",
    ):
        if client is None:
            if not host:
                raise NexumValueError("FTP host must be provided")
            resolved_client = FTPFileClient(host=host, user=user, password=password)
        else:
            resolved_client = client

        super().__init__(resolved_client, path)

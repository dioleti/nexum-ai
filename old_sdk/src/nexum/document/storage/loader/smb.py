from nexum.common.errors import NexumValueError
from nexum.document.storage.client.smb import SMBFileClient
from nexum.document.storage.loader.base import FileLoader


class SMBFileLoader(FileLoader):
    def __init__(
        self,
        path: str,
        *,
        client: SMBFileClient | None = None,
        server: str | None = None,
        share: str | None = None,
        username: str = "",
        password: str = "",
    ):
        if client is None:
            if not server or not share:
                raise NexumValueError("SMB server and share must be provided")
            resolved_client = SMBFileClient(
                server=server, share=share, username=username, password=password
            )
        else:
            resolved_client = client

        super().__init__(resolved_client, path)

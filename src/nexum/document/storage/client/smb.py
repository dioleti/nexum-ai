import uuid
from collections.abc import Iterator

try:
    from smbprotocol.connection import Connection
    from smbprotocol.file_info import FileAttributes, FileStandardInformation
    from smbprotocol.open import (
        CreateDisposition,
        CreateOptions,
        FilePipePrinterAccessMask,
        ImpersonationLevel,
        Open,
        ShareAccess,
    )
    from smbprotocol.session import Session
    from smbprotocol.tree import TreeConnect
except ImportError as e:
    raise ImportError(
        "SMB support is not installed. Install it with: pip install nexum-ai[smb]"
    ) from e

from nexum.document.storage.client.base import FileClient


class SMBFileClient(FileClient):
    def __init__(self, server: str, share: str, username: str, password: str):
        self.server = server
        self.share = share
        self.username = username
        self.password = password

    def _open(self, path: str):
        conn = Connection(uuid.uuid4(), self.server)
        conn.connect()

        session = Session(conn, self.username, self.password)
        session.connect()

        tree = TreeConnect(session, rf"\\{self.server}\{self.share}")
        tree.connect()

        file = Open(tree, path)

        file.create(
            impersonation_level=ImpersonationLevel.Impersonation,
            desired_access=FilePipePrinterAccessMask.FILE_READ_DATA,
            file_attributes=FileAttributes.FILE_ATTRIBUTE_NORMAL,
            share_access=ShareAccess.FILE_SHARE_READ,
            create_disposition=CreateDisposition.FILE_OPEN,
            create_options=CreateOptions.FILE_NON_DIRECTORY_FILE,
        )

        return file

    def file_exists(self, path: str) -> bool:
        try:
            file = self._open(path)
            file.close()
            return True
        except Exception:
            return False

    def load_bytes(self, path: str) -> bytes:
        file = self._open(path)

        try:
            info = file.query_info(FileStandardInformation)
        except TypeError:
            info = file.query_info(info_type=0x01, file_info_class=0x12)
        size = info["allocation_size"]

        data = file.read(0, size)
        file.close()
        return data

    def stream(self, path: str) -> Iterator[bytes]:
        file = self._open(path)

        def generator():
            offset = 0
            chunk_size = 8192
            while True:
                chunk = file.read(offset, chunk_size)
                if not chunk:
                    break
                offset += len(chunk)
                yield chunk

        return generator()

    def get_uri(self, path: str) -> str:
        return f"smb://{self.server}/{self.share}/{path}"

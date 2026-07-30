from collections.abc import Iterator
from ftplib import FTP
from io import BytesIO

from nexum.document.storage.client.base import FileClient


class FTPFileClient(FileClient):
    def __init__(self, host: str, user: str = "", password: str = ""):
        self.host = host
        self.user = user
        self.password = password

    def _connect(self):
        ftp = FTP(self.host)
        ftp.login(self.user, self.password)
        return ftp

    def file_exists(self, path: str) -> bool:
        ftp = self._connect()
        directory, filename = path.rsplit("/", 1)
        ftp.cwd(directory)
        files = ftp.nlst()
        return filename in files

    def load_bytes(self, path: str) -> bytes:
        ftp = self._connect()
        directory, filename = path.rsplit("/", 1)
        ftp.cwd(directory)
        buffer = BytesIO()
        ftp.retrbinary(f"RETR {filename}", buffer.write)
        return buffer.getvalue()

    def stream(self, path: str) -> Iterator[bytes]:
        ftp = self._connect()
        directory, filename = path.rsplit("/", 1)
        ftp.cwd(directory)

        def generator():
            with ftp.transfercmd(f"RETR {filename}") as conn:
                while True:
                    chunk = conn.recv(8192)
                    if not chunk:
                        break
                    yield chunk

        return generator()

    def get_uri(self, path: str) -> str:
        return f"ftp://{self.host}/{path}"

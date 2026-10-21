import os
from collections.abc import Iterator

from nexum.document.storage.client.base import FileClient


class LocalFileClient(FileClient):
    def file_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def load_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def stream(self, path: str) -> Iterator[bytes]:
        def generator():
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk

        return generator()

    def get_uri(self, path: str) -> str:
        return f"file://{path}"

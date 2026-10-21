from collections.abc import Iterator

import requests

from nexum.common.errors import NexumValueError
from nexum.document.storage.client.base import FileClient


class HTTPSFileClient(FileClient):
    def file_exists(self, path: str) -> bool:
        r = requests.head(path)
        return r.status_code == 200

    def load_bytes(self, path: str) -> bytes:
        r = requests.get(path)
        if r.status_code != 200:
            raise NexumValueError("File not found")
        return r.content

    def stream(self, path: str) -> Iterator[bytes]:
        r = requests.get(path, stream=True)
        if r.status_code != 200:
            raise NexumValueError("File not found")
        return r.iter_content(chunk_size=8192)

    def get_uri(self, path: str) -> str:
        return path

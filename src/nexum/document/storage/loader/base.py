from collections.abc import Iterator

from nexum.common.errors import NexumValueError
from nexum.common.storage.loader import Loader
from nexum.document.storage.client.base import FileClient


class FileLoader(Loader):
    def __init__(self, client: FileClient, path: str):
        self.client = client
        self.path = path

    def load(self) -> bytes:
        if not self.client.file_exists(self.path):
            raise NexumValueError("File not found")
        return self.client.load_bytes(self.path)

    def stream(self) -> Iterator[bytes]:
        if not self.client.file_exists(self.path):
            raise NexumValueError("File not found")
        return self.client.stream(self.path)

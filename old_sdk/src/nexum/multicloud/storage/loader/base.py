from collections.abc import Iterator

from nexum.common.errors import NexumValueError
from nexum.common.storage.loader import Loader
from nexum.multicloud.storage.client.base import StorageClient


class CloudLoader(Loader):
    def __init__(self, client: StorageClient, container: str, blob: str):
        self.client = client
        self.container = container
        self.blob = blob

    def load(self) -> bytes:
        if not self.client.blob_exists(self.container, self.blob):
            raise NexumValueError("Blob not found")
        return self.client.get_blob(self.container, self.blob)

    def stream(self) -> Iterator[bytes]:
        if not self.client.blob_exists(self.container, self.blob):
            raise NexumValueError("Blob not found")
        return self.client.stream_blob(self.container, self.blob)

    def get_uri(self) -> str:
        if hasattr(self.client, "get_uri"):
            return self.client.get_uri(self.container, self.blob)
        raise NexumValueError("Client does not support get_uri()")

from abc import ABC, abstractmethod
from collections.abc import Iterator


class StorageClient(ABC):
    @abstractmethod
    def stream_blob(self, container: str, blob: str) -> Iterator[bytes]: ...

    @abstractmethod
    def get_blob(self, container: str, blob: str) -> bytes: ...

    @abstractmethod
    def blob_exists(self, container: str, blob: str) -> bool: ...

    @abstractmethod
    def get_uri(self, container: str, blob: str) -> str: ...

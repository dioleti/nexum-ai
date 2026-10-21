from abc import ABC, abstractmethod
from collections.abc import Iterator


class FileClient(ABC):
    @abstractmethod
    def file_exists(self, path: str) -> bool: ...

    @abstractmethod
    def load_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def stream(self, path: str) -> Iterator[bytes]: ...

    @abstractmethod
    def get_uri(self, path: str) -> str: ...

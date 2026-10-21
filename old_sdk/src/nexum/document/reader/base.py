from abc import ABC, abstractmethod
from collections.abc import Generator

from nexum.document.models import Document
from nexum.multicloud.storage.loader.base import Loader


class Reader(ABC):
    def __init__(self, loader: Loader):
        self.loader = loader

    @abstractmethod
    def read(self) -> Document: ...

    @abstractmethod
    def read_streaming(self) -> Generator[Document, None, None]: ...

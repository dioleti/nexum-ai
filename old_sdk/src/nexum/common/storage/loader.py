from abc import ABC, abstractmethod
from collections.abc import Iterator


class Loader(ABC):
    @abstractmethod
    def load(self) -> bytes: ...

    @abstractmethod
    def stream(self) -> Iterator[bytes]: ...

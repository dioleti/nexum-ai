from typing import Protocol, Generator


class BaseStreamLoader(Protocol):
    def stream(self) -> Generator[bytes, None, None]:
        """
        Stream bytes from a data source.

        All loaders must implement this method.
        """
        ...

import os
from typing import Optional, Dict, Generator, Union, Type

import requests
from nexum.errors import NexumValueError
from nexum.streams import BaseStreamLoader  # supondo que esteja nesse módulo


class PDFReader:
    """
    Streaming PDF ingestion engine.

    This reader retrieves PDF bytes from various sources (local files or HTTP/HTTPS URLs)
    in a memory‑efficient streaming fashion. It supports both custom loader classes
    implementing the BaseStreamLoader protocol and the built‑in default loader.
    """

    def __init__(
        self,
        loader_cls: Optional[Type[BaseStreamLoader]] = None,
        proxies: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        verify: Union[bool, str] = True,
        chunk_size: int = 65536,
    ):
        self.loader_cls = loader_cls
        self.proxies = proxies
        self.timeout = timeout
        self.verify = verify
        self.chunk_size = chunk_size

        self.session = requests.Session()
        self.session.proxies = proxies or {}
        self.session.verify = verify

    def _default_loader(self, source: str) -> BaseStreamLoader:
        if source.startswith(("http://", "https://")):
            return HTTPStreamLoader(source, self.session, self.chunk_size)

        if os.path.isfile(source):
            return LocalFileLoader(source, self.chunk_size)

        raise NexumValueError(f"Unsupported source format or file not found: {source}")

    def read_stream(self, source: str) -> Generator[bytes, None, None]:
        """
        Return a generator yielding PDF bytes in chunks.

        If a custom loader class is provided, it will be instantiated with the source.
        Otherwise, the default loader will be used.
        """
        if self.loader_cls:
            loader = self.loader_cls(source)
        else:
            loader = self._default_loader(source)

        return loader.stream()

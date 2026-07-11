import os
import requests
from typing import Callable, Optional, Dict, Generator, Union

from internal.exceptions.nexum_value_error import NexumValueError


class PDFReader:
    def __init__(
        self,
        loader: Optional[Callable[[str], Generator[bytes, None, None]]] = None,
        proxies: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        verify: Union[bool, str] = True,
        chunk_size: int = 65536,  # 64 KB chunks
    ):
        """
        Streaming PDF ingestion engine.

        :param loader: Optional custom loader function that receives a path/URI
                       and returns a generator yielding PDF bytes in chunks.

        :param proxies: Optional HTTP/HTTPS proxy configuration.

        :param timeout: Timeout for HTTP requests.

        :param verify: SSL/TLS certificate verification:
                       - True  → system CA bundle
                       - False → disable verification
                       - "/path/to/ca.pem" → corporate CA bundle

        :param chunk_size: Size of each chunk yielded during streaming.
        """
        self.loader = loader or self._default_loader
        self.proxies = proxies
        self.timeout = timeout
        self.verify = verify
        self.chunk_size = chunk_size

        self.session = requests.Session()
        self.session.proxies = proxies or {}
        self.session.verify = verify

    def _default_loader(self, source: str) -> Generator[bytes, None, None]:
        """
        Default loader supporting:
        - Local files (streaming)
        - HTTP/HTTPS URLs (streaming)
        """
        if source.startswith("http://") or source.startswith("https://"):
            return self._load_http_stream(source)

        if os.path.exists(source):
            return self._load_local_stream(source)

        raise NexumValueError(f"Unsupported source format or file not found: {source}")

    def _load_local_stream(self, path: str) -> Generator[bytes, None, None]:
        """Stream PDF bytes from a local file without loading everything into memory."""
        with open(path, "rb") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk

    def _load_http_stream(self, url: str) -> Generator[bytes, None, None]:
        """
        Stream PDF bytes from an HTTP/HTTPS URL.
        Supports:
        - proxies
        - corporate TLS interception
        - custom CA bundles
        - chunked transfer encoding
        """
        response = self.session.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=self.chunk_size):
            if chunk:
                yield chunk

    def read_stream(self, source: str) -> Generator[bytes, None, None]:
        """
        Return a generator that yields PDF bytes in chunks.

        :param source: Path or URI pointing to the PDF file.
        :return: Generator yielding raw PDF bytes.
        """
        return self.loader(source)

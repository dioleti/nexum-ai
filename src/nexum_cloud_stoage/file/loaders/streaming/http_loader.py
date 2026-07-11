from typing import Generator, Optional, Dict, Union, Callable

import requests
from nexum.errors import NexumError
from requests import Session
from requests.auth import AuthBase

from nexum_cloud_stoage.file.loaders.bsae_stream_loader import BaseStreamLoader


class HTTPStreamLoader(BaseStreamLoader):
    """
    Generic HTTP/HTTPS streaming loader.

    Streams arbitrary HTTP/HTTPS content in fixed-size chunks without buffering
    the entire response body in memory. Network behavior (proxies, TLS, certs,
    headers, auth) is fully controlled by the underlying `requests.Session`.
    """

    def __init__(
        self,
        url: str,
        chunk_size: int = 65536,
        timeout: Optional[Union[int, float]] = 30,
        proxies: Optional[Dict[str, str]] = None,
        verify: bool = True,
        cert: Optional[Union[str, tuple[str, str]]] = None,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Union[tuple[str, str], AuthBase, Callable]] = None,
        allow_redirects: bool = True,
    ):
        if auth is not None:
            if isinstance(auth, tuple):
                if len(auth) != 2:
                    raise TypeError("auth tuple must be (username, password)")
            elif not isinstance(auth, AuthBase) and not callable(auth):
                raise TypeError("auth must be a (user, pass) tuple, AuthBase, or callable")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
        if headers is not None and not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
            raise TypeError("headers must be dict[str, str]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self._last_response = None
        self.url = url
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.allow_redirects = allow_redirects
        self.session = Session()
        self.session.verify = verify
        self.session.proxies = proxies or {}
        if headers:
            self.session.headers.update(headers)
        self.session.cert = cert
        self.session.auth = auth

    @property
    def response(self):
        return self._last_response

    def close(self):
        if self.session:
            self.session.close()
            self.session = None
            self._last_response = None

    def head(self):
        try:
            response = self.session.head(self.url, timeout=self.timeout)
            self._last_response = response
            return response
        except requests.RequestException as exc:
            raise NexumError(f"HEAD request failed for {self.url}") from exc

    def stream(self) -> Generator[bytes, None, None]:
        try:
            response = self.session.get(
                self.url,
                stream=True,
                timeout=self.timeout,
                allow_redirects=self.allow_redirects,
            )
            self._last_response = response
            response.raise_for_status()
        except requests.RequestException as exc:
            raise NexumError(f"Failed to stream from {self.url}") from exc

        for chunk in response.iter_content(chunk_size=self.chunk_size):
            if chunk:
                yield chunk

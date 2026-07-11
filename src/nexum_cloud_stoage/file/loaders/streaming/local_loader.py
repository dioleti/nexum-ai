import os
from typing import Generator

from nexum.errors import NexumValueError

from nexum_cloud_stoage.file.loaders.bsae_stream_loader import BaseStreamLoader


class LocalFileLoader(BaseStreamLoader):
    """
    Stream loader for local PDF files.

    This loader reads a file from the local filesystem in fixed-size chunks
    without loading the entire content into memory.
    """

    def __init__(self, path: str, chunk_size: int = 65536):
        if not os.path.isfile(path):
            raise NexumValueError(f"Local file not found: {path}")

        self.path = path
        self.chunk_size = chunk_size

    def stream(self) -> Generator[bytes, None, None]:
        """
        Stream bytes from the local file.

        Returns
        -------
        Generator[bytes, None, None]
            A generator yielding raw PDF bytes.
        """
        with open(self.path, "rb") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk

from ftplib import FTP, FTP_TLS
from typing import Optional, Generator

from nexum.errors import NexumValueError, NexumRuntimeError


class FTPStreamLoader:
    """
    Loader responsible for streaming files stored on FTP or FTPS servers using
    incremental chunked reading.

    This class is suitable for processing large files (PDFs, images, binary datasets)
    without loading the entire file into memory.
    """

    def __init__(
        self,
        path: str,
        *,
        chunk_size: int = 65536,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False,
        passive_mode: bool = True,
        port: Optional[int] = None
    ):
        """
        Initialize the FTP stream loader.

        Parameters
        ----------
        path : str
            Full FTP path in the format `ftp://server/path/to/file`.
        chunk_size : int, optional
            Maximum size of each streamed chunk. Default is 64 KB.
        username : str, optional
            Username for FTP authentication.
        password : str, optional
            Password for FTP authentication.
        use_tls : bool, optional
            Enables FTPS (FTP over TLS). Default is False.
        passive_mode : bool, optional
            Enables passive mode. Default is True.
        port : int, optional
            FTP port. Default is 21 for FTP and 990 for FTPS.
        """
        self.chunk_size = chunk_size
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.passive_mode = passive_mode
        self.server, self.file_path = self._parse_path(path)
        self.port = port or (990 if use_tls else 21)
        self.client = None

    def _parse_path(self, path: str) -> tuple[str, str]:
        """
        Validate and extract server and file path from an FTP URL.

        Returns
        -------
        tuple[str, str]
            The server hostname/IP and file path.

        Raises
        ------
        NexumValueError
            If the path does not follow the expected `ftp://server/file` format.
        """
        if not path.startswith("ftp://"):
            raise NexumValueError("Invalid FTP path format. Expected ftp://server/path/to/file")

        clean = path.replace("ftp://", "")
        parts = clean.split("/", 1)

        if len(parts) != 2:
            raise NexumValueError("Invalid FTP path. Expected ftp://server/path/to/file")

        return parts[0], parts[1]

    def _connect(self):
        """
        Establish FTP or FTPS connection and authenticate.
        """
        try:
            if self.use_tls:
                self.client = FTP_TLS()
                self.client.connect(self.server, self.port)
                self.client.auth()
                self.client.prot_p()
            else:
                self.client = FTP()
                self.client.connect(self.server, self.port)

            self.client.login(self.username or "anonymous", self.password or "")
            self.client.set_pasv(self.passive_mode)
        except Exception as e:
            raise NexumRuntimeError(f"Failed to connect to FTP server '{self.server}': {e}")

    def stream(self) -> Generator[bytes, None, None]:
        """
        Stream bytes from an FTP or FTPS file using incremental chunking.

        This method provides a memory-efficient way to read large files by yielding
        their content in sequential chunks. Only one chunk is held in memory at a time.

        Streaming behavior
        ------------------
        - Chunks are yielded in the original file order.
        - Each chunk has a maximum size defined by `self.chunk_size`.
        - Streaming continues until the entire file is consumed.
        - No full-file buffering occurs.

        Returns
        -------
        Generator[bytes, None, None]
            A generator yielding raw bytes from the FTP file.

        Raises
        ------
        NexumRuntimeError
            If the file cannot be opened or streamed due to authentication issues,
            network errors, permission problems, or other runtime failures.

        Examples
        --------
        Basic usage:

            loader = FTPStreamLoader("ftp://server/path/file.pdf")
            for chunk in loader.stream():
                process(chunk)

        Using FTPS:

            loader = FTPStreamLoader(
                "ftp://secure-server/path/file.pdf",
                use_tls=True,
                username="user",
                password="pass"
            )

        Notes
        -----
        - Passive mode is recommended for most environments.
        - FTPS requires the server to support TLS negotiation.
        - This method does not implement automatic retries.
        """
        self._connect()

        try:
            conn = self.client.transfercmd(f"RETR {self.file_path}")

            while True:
                chunk = conn.recv(self.chunk_size)
                if not chunk:
                    break
                yield chunk

            conn.close()
            self.client.voidresp()

        except Exception as e:
            raise NexumRuntimeError(
                f"Failed to stream FTP file 'ftp://{self.server}/{self.file_path}': {e}"
            )
        finally:
            try:
                self.client.quit()
            except Exception:
                pass

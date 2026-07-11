from typing import Optional, Generator

import ldap3
from nexum.errors import NexumValueError, NexumRuntimeError
from smbprotocol.connection import Connection
from smbprotocol.open import Open
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect


class SMBStreamLoader:
  """
  Loader responsible for streaming files stored on SMB/CIFS shares using
  incremental chunked reading. Supports NTLM authentication, Kerberos, and
  optional LDAP-based credential lookup.

  This class is suitable for processing large files (PDFs, images, binary datasets)
  without loading the entire file into memory.
  """

  def __init__(
      self,
      path: str,
      *,
      chunk_size: int = 65536,
      server: Optional[str] = None,
      share: Optional[str] = None,
      username: Optional[str] = None,
      password: Optional[str] = None,
      kerberos: bool = False,
      ldap_url: Optional[str] = None,
      ldap_user_dn: Optional[str] = None,
      ldap_password: Optional[str] = None,
      ldap_attribute: str = "userPassword",
      port: int = 445
  ):
    """
    Initialize the SMB stream loader.

    Parameters
    ----------
    path : str
        Full SMB path in the format `smb://server/share/path/to/file`.
    chunk_size : int, optional
        Maximum size of each streamed chunk. Default is 64 KB.
    server : str, optional
        SMB server hostname or IP. Required if not using `path` parsing.
    share : str, optional
        SMB share name. Required if not using `path` parsing.
    username : str, optional
        Username for SMB authentication.
    password : str, optional
        Password for SMB authentication.
    kerberos : bool, optional
        Enables Kerberos authentication instead of NTLM.
    ldap_url : str, optional
        LDAP server URL for credential lookup.
    ldap_user_dn : str, optional
        LDAP DN used to authenticate and retrieve user credentials.
    ldap_password : str, optional
        Password for LDAP bind.
    ldap_attribute : str, optional
        LDAP attribute containing the SMB password.
    port : int, optional
        SMB port. Default is 445.
    """
    self.chunk_size = chunk_size
    self.username = username
    self.password = password
    self.kerberos = kerberos
    self.ldap_url = ldap_url
    self.ldap_user_dn = ldap_user_dn
    self.ldap_password = ldap_password
    self.ldap_attribute = ldap_attribute
    self.port = port

    if path:
      self.server, self.share, self.file_path = self._parse_path(path)
    else:
      if not server or not share:
        raise NexumValueError("SMB requires either a full path or server + share.")
      self.server = server
      self.share = share
      self.file_path = ""

    self.connection = None
    self.session = None
    self.tree = None

    if self.ldap_url:
      self._resolve_credentials_via_ldap()

  def _parse_path(self, path: str) -> tuple[str, str, str]:
    """
    Validate and extract server, share, and file path from an SMB URL.

    Returns
    -------
    tuple[str, str, str]
        The server hostname/IP, share name, and file path.

    Raises
    ------
    NexumValueError
        If the path does not follow the expected `smb://server/share/file` format.
    """
    if not path.startswith("smb://"):
      raise NexumValueError("Invalid SMB path format. Expected smb://server/share/file")

    clean = path.replace("smb://", "")
    parts = clean.split("/", 2)

    if len(parts) != 3:
      raise NexumValueError("Invalid SMB path. Expected smb://server/share/file")

    return parts[0], parts[1], parts[2]

  def _resolve_credentials_via_ldap(self):
    """
    Resolve SMB credentials using an LDAP directory.

    This method binds to the LDAP server and retrieves the user's password
    from a configurable attribute. It does not modify SMB authentication
    behavior; it only fills in missing credentials.
    """
    try:
      server = ldap3.Server(self.ldap_url)
      conn = ldap3.Connection(server, self.ldap_user_dn, self.ldap_password, auto_bind=True)

      conn.search(
        search_base=self.ldap_user_dn,
        search_filter="(objectClass=*)",
        attributes=[self.ldap_attribute]
      )

      if conn.entries:
        entry = conn.entries[0]
        if hasattr(entry, self.ldap_attribute):
          self.password = entry[self.ldap_attribute].value
    except Exception as e:
      raise NexumRuntimeError(f"Failed to resolve credentials via LDAP: {e}")

  def _connect(self):
    """
    Establish SMB connection, session, and tree connection.
    """
    try:
      self.connection = Connection(self.server, self.server, port=self.port)
      self.connection.connect()

      if self.kerberos:
        self.session = Session(self.connection, None, None, require_encryption=False, use_kerberos=True)
      else:
        self.session = Session(self.connection, self.username, self.password)

      self.session.connect()

      self.tree = TreeConnect(self.session, fr"\\{self.server}\{self.share}")
      self.tree.connect()
    except Exception as e:
      raise NexumRuntimeError(f"Failed to connect to SMB server '{self.server}': {e}")

  def stream(self) -> Generator[bytes, None, None]:
    """
    Stream bytes from an SMB file using incremental chunking.

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
        A generator yielding raw bytes from the SMB file.

    Raises
    ------
    NexumRuntimeError
        If the file cannot be opened or streamed due to authentication issues,
        network errors, permission problems, or other runtime failures.

    Examples
    --------
    Basic usage:

        loader = SMBStreamLoader("smb://server/share/docs/file.pdf")
        for chunk in loader.stream():
            process(chunk)

    Using Kerberos:

        loader = SMBStreamLoader(
            "smb://server/share/docs/file.pdf",
            kerberos=True
        )

    Using LDAP credential lookup:

        loader = SMBStreamLoader(
            "smb://server/share/docs/file.pdf",
            ldap_url="ldap://ldap.company.com",
            ldap_user_dn="cn=user,dc=company,dc=com",
            ldap_password="secret"
        )

    Notes
    -----
    - Kerberos requires a valid ticket in the environment (kinit).
    - LDAP is used only to retrieve credentials; SMB authentication still uses NTLM or Kerberos.
    - This method does not implement automatic retries.
    """
    self._connect()

    try:
      file = Open(self.tree, self.file_path, access=Open.ACCESS_READ)
      file.create()

      offset = 0
      while True:
        chunk = file.read(self.chunk_size, offset)
        if not chunk:
          break
        offset += len(chunk)
        yield chunk

      file.close()
    except Exception as e:
      raise NexumRuntimeError(
        f"Failed to stream SMB file '\\\\{self.server}\\{self.share}\\{self.file_path}': {e}"
      )

from nexum.document.storage.client.https import HTTPSFileClient
from nexum.document.storage.loader.base import FileLoader


class HTTPSFileLoader(FileLoader):
    def __init__(self, url: str, *, client: HTTPSFileClient | None = None):
        resolved_client = client or HTTPSFileClient()
        super().__init__(resolved_client, url)

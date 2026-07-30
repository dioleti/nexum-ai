from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.azure import AzureClient
from nexum.multicloud.storage.loader.base import CloudLoader


class AzureLoader(CloudLoader):
    def __init__(
        self, path: str, *, client: AzureClient | None = None, **client_kwargs
    ):
        account, container, blob = self._parse_path(path)

        resolved_client = client or AzureClient(
            account_url=f"https://{account}", **client_kwargs
        )

        super().__init__(resolved_client, container, blob)

    def _parse_path(self, path: str):
        if not path.startswith("azure://"):
            raise NexumValueError(
                "Invalid Azure path format. Expected azure://account/container/blob"
            )

        clean = path.replace("azure://", "")
        parts = clean.split("/", 2)

        if len(parts) != 3:
            raise NexumValueError(
                "Invalid Azure path. Expected azure://account/container/blob"
            )

        return parts[0], parts[1], parts[2]

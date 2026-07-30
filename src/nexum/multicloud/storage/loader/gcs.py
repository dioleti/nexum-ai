from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.gcs import GCSClient
from nexum.multicloud.storage.loader.base import CloudLoader


class GCSLoader(CloudLoader):
    def __init__(self, path: str, *, client: GCSClient | None = None, **client_kwargs):
        bucket, blob = self._parse_path(path)

        resolved_client = client or GCSClient(**client_kwargs)

        super().__init__(resolved_client, bucket, blob)

    def _parse_path(self, path: str):
        if not path.startswith("gs://") or "/" not in path[5:]:
            raise NexumValueError("Invalid GCS path format. Expected gs://bucket/blob")

        clean = path.replace("gs://", "")
        return clean.split("/", 1)

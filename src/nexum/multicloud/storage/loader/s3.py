from nexum.common.errors import NexumValueError
from nexum.multicloud.storage.client.s3 import S3Client
from nexum.multicloud.storage.loader.base import CloudLoader


class S3Loader(CloudLoader):
    def __init__(self, path: str, *, client: S3Client | None = None, **client_kwargs):
        bucket, key = self._parse_path(path)

        if not key:
            raise NexumValueError("S3 key cannot be empty")

        resolved_client = client or S3Client(**client_kwargs)

        super().__init__(resolved_client, bucket, key)

    def _parse_path(self, path: str):
        if not path.startswith("s3://") or "/" not in path[5:]:
            raise NexumValueError("Invalid S3 path format. Expected s3://bucket/key")

        clean = path.replace("s3://", "")
        return clean.split("/", 1)

from typing import Optional, Dict, Any, Generator

import boto3
from nexum.errors import NexumValueError, NexumRuntimeError


class S3StreamLoader:
    """
    Loader responsible for streaming objects stored in Amazon S3 using flexible
    authentication and incremental chunked reading.

    This class is suitable for processing large files (PDFs, images, binary datasets)
    without loading the entire object into memory.
    """

    def __init__(
        self,
        path: str,
        *,
        chunk_size: int = 65536,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        assume_role_arn: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        """
        Initialize the S3 stream loader.

        Parameters
        ----------
        path : str
            Full S3 path in the format `s3://bucket/key`.
        chunk_size : int, optional
            Maximum size of each streamed chunk. Default is 64 KB.
        aws_access_key_id : str, optional
            AWS access key ID.
        aws_secret_access_key : str, optional
            AWS secret access key.
        aws_session_token : str, optional
            AWS session token for temporary credentials.
        profile_name : str, optional
            AWS CLI profile name.
        region_name : str, optional
            AWS region name.
        assume_role_arn : str, optional
            ARN of a role to assume via STS.
        endpoint_url : str, optional
            Custom S3-compatible endpoint (e.g., MinIO).
        """
        self.chunk_size = chunk_size
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.profile_name = profile_name
        self.region_name = region_name
        self.assume_role_arn = assume_role_arn
        self.endpoint_url = endpoint_url

        self.bucket_name, self.key_name = self._parse_path(path)
        if not self.key_name:
            raise NexumValueError("S3 key cannot be empty")
        if self.region_name is None:
            raise NexumValueError("region_name must be provided for S3 streaming")
        resolved = self._resolve_credentials()
        self.session = resolved["session"]
        self.client_kwargs = resolved["client_kwargs"]
        self.client = self.session.client("s3", **self.client_kwargs)

    def _parse_path(self, path: str) -> tuple[str, str]:
        """
        Validate and extract bucket and key from an S3 path.

        Returns
        -------
        tuple[str, str]
            The bucket name and object key.

        Raises
        ------
        NexumValueError
            If the path does not follow the expected `s3://bucket/key` format.
        """
        if not path.startswith("s3://") or "/" not in path[5:]:
            raise NexumValueError("Invalid S3 path format. Expected s3://bucket/key")

        clean = path.replace("s3://", "")
        return clean.split("/", 1)

    def _resolve_credentials(self) -> Dict[str, Any]:
        """
        Resolve AWS credentials based on the provided parameters.

        Supports:
        - Explicit access key credentials
        - AWS CLI profile
        - STS AssumeRole
        - Default AWS credential chain (fallback)

        Returns
        -------
        dict
            A dictionary containing:
            - `session`: boto3.Session
            - `client_kwargs`: arguments for creating the S3 client
        """
        session_kwargs = {}
        if self.profile_name:
            session_kwargs["profile_name"] = self.profile_name

        session = boto3.Session(**session_kwargs)

        if self.aws_access_key_id and self.aws_secret_access_key:
            creds = {
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
            }
            if self.aws_session_token:
                creds["aws_session_token"] = self.aws_session_token

            return {
                "session": session,
                "client_kwargs": {
                    "region_name": self.region_name,
                    "endpoint_url": self.endpoint_url,
                    **creds
                }
            }

        if self.assume_role_arn:
            sts = session.client("sts", region_name=self.region_name)
            assumed = sts.assume_role(
                RoleArn=self.assume_role_arn,
                RoleSessionName="pdf_reader_session",
            )

            return {
                "session": session,
                "client_kwargs": {
                    "region_name": self.region_name,
                    "endpoint_url": self.endpoint_url,
                    "aws_access_key_id": assumed["Credentials"]["AccessKeyId"],
                    "aws_secret_access_key": assumed["Credentials"]["SecretAccessKey"],
                    "aws_session_token": assumed["Credentials"]["SessionToken"],
                }
            }

        return {
            "session": session,
            "client_kwargs": {
                "region_name": self.region_name,
                "endpoint_url": self.endpoint_url,
            }
        }

    def stream(self) -> Generator[bytes, None, None]:
        """
        Stream bytes from an Amazon S3 object using incremental chunking.

        This method provides a memory-efficient way to read large objects by yielding
        their content in sequential chunks. Only one chunk is held in memory at a time.

        Streaming behavior
        ------------------
        - Chunks are yielded in the original object order.
        - Each chunk has a maximum size defined by `self.chunk_size`.
        - Streaming continues until the entire object is consumed.
        - No full-file buffering occurs.

        Returns
        -------
        Generator[bytes, None, None]
            A generator yielding raw bytes from the S3 object.

        Raises
        ------
        NexumRuntimeError
            If the object cannot be opened or streamed due to authentication issues,
            network errors, permission problems, or other runtime failures.

        Examples
        --------
        Basic usage:

            loader = S3StreamLoader("s3://my-bucket/file.pdf")
            for chunk in loader.stream():
                process(chunk)

        Using explicit credentials:

            loader = S3StreamLoader(
                "s3://my-bucket/file.pdf",
                aws_access_key_id="AKIA...",
                aws_secret_access_key="SECRET..."
            )

        Assuming a role:

            loader = S3StreamLoader(
                "s3://secure-bucket/data.bin",
                assume_role_arn="arn:aws:iam::123456789012:role/AnalyticsRole"
            )

        Notes
        -----
        - This method does not implement automatic retries.
        - Errors raised during iteration will stop the generator immediately.
        """
        try:
            obj = self.client.get_object(Bucket=self.bucket_name, Key=self.key_name)
        except Exception as e:
            raise NexumRuntimeError(
                f"Failed to open S3 object 's3://{self.bucket_name}/{self.key_name}': {e}"
            )

        body = obj["Body"]

        try:
            for chunk in body.iter_chunks(chunk_size=self.chunk_size):
                if chunk:
                    yield chunk
        except Exception as e:
            raise NexumRuntimeError(
                f"Failed to stream S3 object 's3://{self.bucket_name}/{self.key_name}': {e}"
            )

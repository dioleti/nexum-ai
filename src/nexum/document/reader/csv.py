from collections.abc import Generator

from nexum.common.errors import NexumRuntimeError
from nexum.document.models import CSVReaderConfig, Document
from nexum.document.parser.csv import CSVParser
from nexum.document.reader.base import Reader


class CSVReader(Reader):
    def __init__(
        self,
        loader,
        config: CSVReaderConfig | None = None,
        parser: CSVParser | None = None,
    ):
        super().__init__(loader)
        self.config = config or CSVReaderConfig()
        self.parser = parser or CSVParser(self.config)

    def _detect_encoding(self, raw_bytes: bytes) -> str:
        if self.config.encoding:
            return self.config.encoding
        try:
            raw_bytes.decode("utf-8")
            return "utf-8"
        except Exception:
            return "latin-1"

    def _decode(self, raw_bytes: bytes, encoding: str) -> str:
        return raw_bytes.decode(encoding, errors="replace")

    def _build_document(self, raw_bytes: bytes, text: str) -> Document:
        parsed = self.parser.parse(text)
        table = parsed["table"]
        metadata = parsed["metadata"]

        return Document(
            content=text,
            metadata=metadata,
            images=[],
            tables=[table],
            pages=[text],
            raw_bytes=raw_bytes,
            source=self.loader.__class__.__name__,
        )

    def read(self) -> Document:
        try:
            raw_bytes = self.loader.load()
            encoding = self._detect_encoding(raw_bytes)
            text = self._decode(raw_bytes, encoding)
            return self._build_document(raw_bytes, text)
        except Exception as e:
            raise NexumRuntimeError(f"CSVReader failed: {e}")

    def read_streaming(self) -> Generator[Document, None, None]:
        try:
            raw_bytes = b"".join(chunk for chunk in self.loader.stream())
            encoding = self._detect_encoding(raw_bytes)
            text = self._decode(raw_bytes, encoding)
            yield self._build_document(raw_bytes, text)
        except Exception as e:
            raise NexumRuntimeError(f"CSVReader streaming failed: {e}")

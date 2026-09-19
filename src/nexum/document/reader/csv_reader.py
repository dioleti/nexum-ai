from collections.abc import Generator

from nexum.common.errors import NexumRuntimeError
from nexum.document.models import CSVReaderConfig, Document, Table
from nexum.document.parser.csv_parser import CSVParser
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

        if isinstance(parsed, dict):
            raw_table = parsed.get("table") or (parsed.get("tables")[0] if parsed.get("tables") else None)
            metadata = parsed.get("metadata")
        else:
            raw_table = getattr(parsed, "table", None)
            if raw_table is None and hasattr(parsed, "tables") and getattr(parsed, "tables"):
                raw_table = getattr(parsed, "tables")[0]
            metadata = getattr(parsed, "metadata", None)

        if raw_table is None:
            table = Table(rows=[], columns=[])
        else:
            if isinstance(raw_table, dict):
                table = Table(**raw_table)
            elif isinstance(raw_table, list):
                header = raw_table[0] if raw_table else []
                rows = raw_table[1:] if len(raw_table) > 1 else []
                table = Table(columns=header, rows=rows)
            else:
                table = raw_table

        return Document(
            content=text,
            metadata=metadata or {},
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

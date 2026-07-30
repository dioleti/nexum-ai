from enum import Enum

from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel, Field


class ReaderMode(str, Enum):
    BATCH = "batch"
    STREAMING = "stream"


class Table(BaseModel):
    columns: list[str]
    rows: list[list[str]]
    bbox: tuple[int, int, int, int] | None = None
    page: int | None = None


class Document(BaseModel):
    content: str
    metadata: dict
    images: list[bytes] | None = None
    tables: list[Table] | None = None
    source: str | None = None
    pages: list[str] | None = None
    raw_bytes: bytes | None = None

    def to_langchain_document(self) -> LCDocument:
        meta = {
            "metadata": self.metadata,
            "images": self.images,
            "tables": self.tables,
            "source": self.source,
            "pages": self.pages,
            "raw_bytes": self.raw_bytes,
        }

        return LCDocument(page_content=self.content, metadata=meta)


class NexumConfig(BaseModel): ...


class CSVReaderConfig(NexumConfig):
    delimiter: str = ","
    skip_rows: int = 0
    has_header: bool = True
    encoding: str | None = None
    infer_types: bool = True


class ImageOCRConfig(NexumConfig):
    lang: str = "eng"
    enable_sharpen: bool = True
    enable_metadata: bool = True
    sharpen_strength: float = 1.2
    enable_binarization: bool = True
    binarization_min_std: float = 10.0
    enable_denoise: bool = True
    denoise_method: str = "bilateral"  # "bilateral", "median", "none"
    strip_empty_lines: bool = True


class ImageReaderConfig(NexumConfig):
    read_mode: ReaderMode = ReaderMode.BATCH
    ocr_config: ImageOCRConfig = Field(default_factory=ImageOCRConfig)


class PDFOCRConfig(NexumConfig):
    min_dpi: int = 200
    lang: str = "eng"
    upscale_factor: float = 1.5
    enable_binarization: bool = True
    binarization_method: str = "otsu"
    binarization_min_std: float = 10.0
    enable_deskew: bool = True
    deskew_max_angle: float = 5.0
    enable_sharpen: bool = True
    sharpen_strength: float = 1.2
    enable_denoise: bool = True
    denoise_method: str = "bilateral"
    enable_osd: bool = True
    osd_min_confidence: float = 5.0
    psm: int = 6
    oem: int = 3
    strip_empty_lines: bool = True


class PDFReaderConfig(NexumConfig):
    read_mode: ReaderMode = ReaderMode.STREAMING
    ocr_config: PDFOCRConfig = Field(default_factory=PDFOCRConfig)
    lang: str = "eng"
    max_workers: int = 4
    dpi: int = 300
    enable_ocr: bool = True
    enable_metadata: bool = True
    enable_images: bool = True
    enable_tables: bool = True

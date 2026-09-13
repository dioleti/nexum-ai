from enum import Enum
from typing import Any, List

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
    metadata: dict
    content: str
    images: List[Any] | dict[str, Any] = []
    tables: List[Table] | dict[str, Any] = []
    blocks: List[Any] | dict[str, Any] = []
    flowcharts: List[Any] | dict[str, Any] = []
    layout: List[Any] = []
    source: str | None = None
    pages: List[str] = []
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


class ImageConfig(NexumConfig):
    lang: str = "eng"
    enable_binarization: bool = True
    enable_deskew: bool = True
    enable_clahe: bool = True
    enable_denoise: bool = True
    enable_sharpen: bool = True
    enable_osd: bool = True
    clahe_clip_limit: float = 40.0
    clahe_tile_size: tuple[int, int] = (8, 8)
    strip_empty_lines: bool = True
    oem: int = 1
    psm: int = 3
    binarization_method: str = "otsu"
    denoise_method: str = "bilateral"  # "bilateral", "median", "none"
    binarization_min_std: float = 10.0
    sharpen_sigma: float = 1.0
    sharpen_strength: float = 1.2
    sharpen_blur_weight: float = -0.2
    adaptive_block_size: int = 31
    adaptive_c: int = 2
    osd_min_confidence: float = 5.0
    max_pixels: int | None = 20_000_000


class ImageOCRConfig(ImageConfig):
    enable_metadata: bool = True
    return_structured: bool = False


class PDFOCRConfig(ImageConfig):
    min_dpi: int = 72
    upscale_factor: float = 1.5
    deskew_max_angle: float = 5.0


class ImageReaderConfig(NexumConfig):
    read_mode: ReaderMode = ReaderMode.BATCH
    ocr_config: ImageOCRConfig = Field(default_factory=ImageOCRConfig)


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


class VisualRegion(TypedDict):
    page: int
    bbox: Tuple[int, int, int, int]


class PageAnalysisResult(TypedDict):
    layout: Any
    tables: List["Table"]
    blocks: List[VisualRegion]
    images: List[VisualRegion]
    flowcharts: List[VisualRegion]

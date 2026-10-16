from dataclasses import dataclass
from enum import Enum
from typing import Any, List, TypedDict, Tuple, Sequence, Optional

import numpy as np
from PIL import Image
from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel, Field


class ReaderMode(str, Enum):
    BATCH = "batch"
    STREAMING = "stream"


class Table(BaseModel):
    columns: list[Any]
    rows: list[list[Any]]
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
            "blocks": self.blocks,
            "flowcharts": self.flowcharts,
            "layout": self.layout,
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
    encoding: str | None = "utf-8"
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
    table_as_df: bool = True
    enable_caption: bool = True
    enable_flowchart: bool = True


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


class Parsed(BaseModel):
    metadata: dict[str, Any]
    tables: list[Any] = []
    blocks: list[Any] = []
    rows: list[list[Any]] = []
    flowcharts: list[Any] = []
    caption: str | None = None


@dataclass
class ImageContext:
    pil: Image.Image
    gray: np.ndarray
    text_canvas: np.ndarray
    width: int
    height: int
    total_pixels: int
    lang: str
    psm: int


@dataclass(frozen=True, slots=True)
class CaptionInput:
    image: Image.Image
    prompt: str | None = None


@dataclass(frozen=True, slots=True)
class CaptionConfig:
    model_name: str = "Salesforce/blip-image-captioning-base"
    device: str | None = None
    max_new_tokens: int = 50


@dataclass(frozen=True, slots=True)
class FlowchartNode:
    id: str
    label: str
    shape: str
    box: tuple[int, int, int, int]
    center: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FlowchartEdge:
    source: str
    target: str
    direction: str = "down"


@dataclass(frozen=True, slots=True)
class FlowchartGraph:
    nodes: Sequence[FlowchartNode]
    edges: Sequence[FlowchartEdge]
    mermaid: str


@dataclass(frozen=True, slots=True)
class FlowchartInput:
    image: np.ndarray
    lang: str = "eng"


@dataclass(frozen=True, slots=True)
class FlowchartConfig:
    min_area: int = 1500
    threshold_value: int = 210
    tesseract_config: str = "--psm 6"
    padding: int = 4
    polygon_epsilon_ratio: float = 0.03


@dataclass(frozen=True, slots=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass(frozen=True, slots=True)
class DetectedTable:
    label: str
    score: float
    box: BoundingBox


@dataclass(frozen=True, slots=True)
class TableDetectionInput:
    image: Image.Image
    threshold: float | None = None


@dataclass(frozen=True, slots=True)
class TableDetectionConfig:
    model_name: str = "microsoft/table-transformer-detection"
    device: str | None = None
    default_threshold: float = 0.7


class RegionType(Enum):
    TABLE = "table"
    TEXT = "text"
    IMAGE = "image"
    FLOWCHART = "flowchart"


@dataclass(frozen=False, slots=True)
class Region:
    type: RegionType
    box: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    score: float
    text: str | None = None


@dataclass(frozen=True, slots=True)
class RegionSplitterConfig:
    threshold: float = 0.15
    device: Optional[str] = None
    model_name: str = "yolov8m-doclaynet.pt"


@dataclass(frozen=False, slots=True)
class TextBlockRegion:
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=False, slots=True)
class FlowchartRegion:
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=False, slots=True)
class ImageRegion:
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=False, slots=True)
class TableRegion:
    bbox: Tuple[int, int, int, int]

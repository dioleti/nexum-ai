import io
import logging
from collections.abc import Generator

import cv2
import numpy as np
from PIL import Image

from nexum.common.errors import NexumRuntimeError
from nexum.document.models import Document, ImageReaderConfig, Table
from nexum.document.parser.image import ImageOCRParser
from nexum.document.reader.base import Reader

logger = logging.getLogger(__name__)


class ImageReader(Reader):
    def __init__(
        self,
        loader,
        config: ImageReaderConfig | None = None,
        parser: ImageOCRParser | None = None,
    ):
        super().__init__(loader)
        self.config = config or ImageReaderConfig()
        self.ocr = parser or ImageOCRParser(self.config.ocr_config)

    def _read_bytes_batch(self) -> bytes:
        try:
            return self.loader.load()
        except Exception as e:
            raise NexumRuntimeError(f"ImageReader batch read failed: {e}")

    def _read_bytes_streaming(self) -> bytes:
        try:
            return b"".join(chunk for chunk in self.loader.stream())
        except Exception as e:
            raise NexumRuntimeError(f"ImageReader streaming read failed: {e}")

    def _extract_metadata(self, image_bytes: bytes) -> dict:
        try:
            pil = Image.open(io.BytesIO(image_bytes))
            return pil.info or {}
        except Exception:
            return {}

    def _extract_embedded_text(self, image_bytes: bytes) -> str | None:
        try:
            pil = Image.open(io.BytesIO(image_bytes))
            if "text" in pil.info:
                return pil.info["text"]
        except Exception as exc:
            logger.error(exc)
        return None

    def _detect_tables_cv(self, image_bytes: bytes) -> list[Table]:
        try:
            np_img = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return []
            thresh = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
            )
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
            vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
            table_mask = cv2.add(horizontal, vertical)
            contours, _ = cv2.findContours(
                table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            tables: list[Table] = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                roi = img[y : y + h, x : x + w]
                _, roi_png = cv2.imencode(".png", roi)
                text = self.ocr.parse(roi_png.tobytes())
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if not lines:
                    continue
                header = lines[0]
                columns = [c.strip() for c in header.replace("\t", "|").split("|")]
                rows = []
                for line in lines[1:]:
                    parts = [c.strip() for c in line.replace("\t", "|").split("|")]
                    if len(parts) == len(columns):
                        rows.append(parts)
                tables.append(
                    Table(columns=columns, rows=rows, bbox=(x, y, w, h), page=1)
                )
            return tables
        except Exception:
            return []

    def _parse_table_from_lines(self, lines: list[str]) -> Table:
        header = lines[0]
        columns = [c.strip() for c in header.replace("\t", "|").split("|")]
        rows = []
        for line in lines[1:]:
            parts = [c.strip() for c in line.replace("\t", "|").split("|")]
            if len(parts) == len(columns):
                rows.append(parts)
        return Table(columns=columns, rows=rows, bbox=None, page=1)

    def _extract_tables_heuristic(self, text: str) -> list[Table]:
        tables: list[Table] = []
        current: list[str] = []
        for line in text.splitlines():
            if "|" in line or "\t" in line:
                current.append(line)
            else:
                if current:
                    tables.append(self._parse_table_from_lines(current))
                    current = []
        if current:
            tables.append(self._parse_table_from_lines(current))
        return tables

    def _extract_tables(self, image_bytes: bytes, text: str) -> list[Table]:
        tables_cv = self._detect_tables_cv(image_bytes)
        if tables_cv:
            return tables_cv
        return self._extract_tables_heuristic(text)

    def _build_document(self, text: str, metadata: dict, raw_bytes: bytes) -> Document:
        tables = self._extract_tables(raw_bytes, text)
        return Document(
            content=text,
            metadata=metadata,
            images=[],
            tables=tables,
            pages=[text],
            raw_bytes=raw_bytes,
            source=self.loader.__class__.__name__,
        )

    def _ocr_batch(self, image_bytes: bytes) -> Document:
        text = self.ocr.parse(image_bytes)
        metadata = self._extract_metadata(image_bytes)
        return self._build_document(text, metadata, image_bytes)

    def _ocr_stream(self, image_bytes: bytes) -> Generator[Document, None, None]:
        text = self.ocr.parse(image_bytes)
        metadata = self._extract_metadata(image_bytes)
        yield self._build_document(text, metadata, image_bytes)

    def read(self) -> Document:
        try:
            image_bytes = self._read_bytes_batch()
            metadata = self._extract_metadata(image_bytes)
            embedded_text = self._extract_embedded_text(image_bytes)
            if embedded_text:
                return self._build_document(embedded_text, metadata, image_bytes)
            return self._ocr_batch(image_bytes)
        except Exception as e:
            raise NexumRuntimeError(f"ImageReader failed: {e}")

    def read_streaming(self) -> Generator[Document, None, None]:
        try:
            image_bytes = self._read_bytes_streaming()
            metadata = self._extract_metadata(image_bytes)
            embedded_text = self._extract_embedded_text(image_bytes)
            if embedded_text:
                yield self._build_document(embedded_text, metadata, image_bytes)
                return
            yield from self._ocr_stream(image_bytes)
        except Exception as e:
            raise NexumRuntimeError(f"ImageReader streaming failed: {e}")

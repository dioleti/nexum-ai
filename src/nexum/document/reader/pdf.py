import io
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from pypdf import PdfReader
from pypdf.generic import DictionaryObject

from nexum.common.errors import NexumRuntimeError
from nexum.document.layout.analyzer import LayoutAnalyzer
from nexum.document.models import Document, PDFReaderConfig, Table, VisualRegion
from nexum.document.parser.pdf import PDFOCRParser
from nexum.document.reader.base import Reader


class PDFReader(Reader):
    def __init__(
        self,
        loader,
        config: PDFReaderConfig | None = None,
        parser: PDFOCRParser | None = None,
    ):
        super().__init__(loader)
        self.config = config or PDFReaderConfig()
        self.ocr = parser or PDFOCRParser(self.config.ocr_config)
        self.layout = LayoutAnalyzer(
            enable_table_detector=True,
            enable_block_detector=True,
            enable_image_detector=True,
            enable_flowchart_detector_dl=True,
        )

    def _get_bytes(self) -> bytes:
        try:
            if hasattr(self.loader, "load"):
                return self.loader.load()
            return b"".join(self.loader.stream())
        except Exception as e:
            raise NexumRuntimeError(f"PDF read failed: {e}") from e

    def _parse_digital_pdf(self, pdf_bytes: bytes) -> tuple[dict, list[str], list[bytes]]:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except Exception:
            return {}, [], []

        metadata = dict(reader.metadata or {}) if self.config.enable_metadata else {}
        pages_text = [(page.extract_text() or "") for page in reader.pages]

        images: list[bytes] = []
        if self.config.enable_images:
            for page in reader.pages:
                res = page.get("/Resources")
                if not isinstance(res, DictionaryObject) or "/XObject" not in res:
                    continue
                xobjects = res["/XObject"].get_object()
                images.extend(
                    obj.get_data()
                    for obj in xobjects.values()
                    if obj.get("/Subtype") == "/Image"
                )

        return metadata, pages_text, images

    def _ocr_page(self, pil_page: Image.Image) -> str:
        buf = io.BytesIO()
        pil_page.save(buf, format="PNG", dpi=(self.config.dpi, self.config.dpi))
        return self.ocr.parse(buf.getvalue())

    def _parse_visual_tables(self, text: str, page_number: int, bbox=None) -> Table:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return Table(columns=[], rows=[], bbox=bbox, page=page_number)

        def split_row(row: str) -> list[str]:
            if "\t" in row:
                return [c.strip() for c in row.split("\t") if c.strip()]
            parts = [c.strip() for c in row.split("  ") if c.strip()]
            return parts if len(parts) > 1 else row.split()

        columns = split_row(lines[0])
        rows = [
            (split_row(line) + [""] * len(columns))[:len(columns)]
            for line in lines[1:]
        ]
        return Table(columns=columns, rows=rows, bbox=bbox, page=page_number)

    def _analyze_page(self, pil_page: Image.Image, page_number: int) -> dict[str, Any]:
        gray = np.array(pil_page.convert("L"))
        layout = self.layout.run_detectors(gray)

        visual_tables: list[Table] = []
        for region in layout.get("tables", []):
            table_text = self._ocr_page(pil_page.crop(region.bbox))
            if table_text.strip():
                visual_tables.append(
                    self._parse_visual_tables(table_text, page_number, bbox=region.bbox)
                )

        def _to_regions(category: str) -> list[VisualRegion]:
            return [VisualRegion(page=page_number, bbox=r.bbox) for r in layout.get(category, [])]

        return {
            "layout": layout,
            "tables": visual_tables,
            "blocks": _to_regions("blocks"),
            "images": _to_regions("images"),
            "flowcharts": _to_regions("flowcharts"),
        }

    def _process_visual_page(
        self,
        item: tuple[int, Image.Image, str | None],
        metadata: dict,
        source_name: str,
    ) -> Document:
        page_num, pil_page, native_text = item
        page_text = native_text if native_text is not None else self._ocr_page(pil_page)
        analysis = self._analyze_page(pil_page, page_num)

        return Document(
            content=page_text,
            metadata=metadata,
            images=analysis["images"],
            tables=analysis["tables"],
            blocks=analysis["blocks"],
            flowcharts=analysis["flowcharts"],
            pages=[page_text],
            layout=[analysis["layout"]],
            source=source_name,
        )

    def _iter_pages(self, pdf_bytes: bytes) -> Generator[Document, None, None]:
        metadata, digital_pages_text, raw_images = self._parse_digital_pdf(pdf_bytes)
        source_name = self.loader.__class__.__name__

        if self.config.ocr_enabled:
            pages_img = convert_from_bytes(pdf_bytes, dpi=self.config.dpi)

            tasks = []
            for idx, img in enumerate(pages_img, start=1):
                native_text = (
                    digital_pages_text[idx - 1]
                    if idx - 1 < len(digital_pages_text) and digital_pages_text[idx - 1].strip()
                    else None
                )
                tasks.append((idx, img, native_text))

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                for doc in executor.map(
                    lambda item: self._process_visual_page(item, metadata, source_name),
                    tasks,
                ):
                    yield doc
            return

        for idx, text in enumerate(digital_pages_text):
            yield Document(
                content=text,
                metadata=metadata,
                images=raw_images if idx == 0 else [],
                tables=[],
                blocks=[],
                flowcharts=[],
                pages=[text],
                source=source_name,
            )

    def read_streaming(self) -> Generator[Document, None, None]:
        yield from self._iter_pages(self._get_bytes())

    def read(self) -> Document:
        pdf_bytes = self._get_bytes()
        page_docs = list(self._iter_pages(pdf_bytes))
        if not page_docs:
            return Document(content="", metadata={}, raw_bytes=pdf_bytes, source=self.loader.__class__.__name__)

        return Document(
            content="\n".join(doc.content for doc in page_docs),
            metadata=page_docs[0].metadata,
            images=[img for d in page_docs for img in d.images],
            tables=[tbl for d in page_docs for tbl in d.tables],
            blocks=[blk for d in page_docs for blk in d.blocks],
            flowcharts=[fc for d in page_docs for fc in d.flowcharts],
            pages=[doc.content for doc in page_docs],
            layout=[ly for d in page_docs if d.layout for ly in d.layout],
            raw_bytes=pdf_bytes,
            source=page_docs[0].source,
        )

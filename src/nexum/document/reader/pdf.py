import io
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from pdf2image import convert_from_bytes
from pypdf import PdfReader
from pypdf.generic import DictionaryObject

from nexum.common.errors import NexumRuntimeError
from nexum.document.models import Document, PDFReaderConfig, Table
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

    def _read_bytes_batch(self) -> bytes:
        try:
            return self.loader.load()
        except Exception as e:
            raise NexumRuntimeError(f"PDFReader batch read failed: {e}")

    def _read_bytes_streaming(self) -> bytes:
        try:
            return b"".join(chunk for chunk in self.loader.stream())
        except Exception as e:
            raise NexumRuntimeError(f"PDFReader streaming read failed: {e}")

    def _pdf_has_text(self, pdf_bytes: bytes) -> bool:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    return True
            return False
        except Exception:
            return False

    def _extract_pdf_text_pages(self, pdf_bytes: bytes) -> list[str]:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return pages

    def _extract_metadata(self, pdf_bytes: bytes) -> dict:
        if not self.config.enable_metadata:
            return {}
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            return dict(reader.metadata or {})
        except Exception:
            return {}

    def _extract_images(self, pdf_bytes: bytes) -> list[bytes]:
        if not self.config.enable_images:
            return []
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            images: list[bytes] = []
            for page in reader.pages:
                resources = page.get("/Resources")
                if not isinstance(resources, DictionaryObject):
                    continue
                xobject = resources.get("/XObject")
                if xobject is None:
                    continue
                xobjects = xobject.get_object()
                for name in xobjects:
                    obj = xobjects[name]
                    if obj["/Subtype"] == "/Image":
                        images.append(obj.get_data())
            return images
        except Exception:
            return []

    def _parse_table_from_lines(self, lines: list[str], page_number: int) -> Table:
        header = lines[0]
        columns = [c.strip() for c in header.replace("\t", "|").split("|")]
        rows = []
        for line in lines[1:]:
            parts = [c.strip() for c in line.replace("\t", "|").split("|")]
            if len(parts) == len(columns):
                rows.append(parts)
        return Table(columns=columns, rows=rows, bbox=None, page=page_number)

    def _extract_tables_heuristic(self, text: str, page_number: int) -> list[Table]:
        if not self.config.enable_tables:
            return []
        tables: list[Table] = []
        current: list[str] = []
        for line in text.splitlines():
            if "|" in line or "\t" in line:
                current.append(line)
            else:
                if current:
                    tables.append(self._parse_table_from_lines(current, page_number))
                    current = []
        if current:
            tables.append(self._parse_table_from_lines(current, page_number))
        return tables

    def _ocr_page(self, pil_page) -> str:
        buf = io.BytesIO()
        pil_page.save(buf, format="PNG")
        return self.ocr.parse(buf.getvalue())

    def _extract_pdf_ocr_batch(self, pdf_bytes: bytes) -> Document:
        pages_img = convert_from_bytes(pdf_bytes, dpi=self.config.dpi)
        results: list[str | None] = [None] * len(pages_img)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._ocr_page, page): idx
                for idx, page in enumerate(pages_img)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = ""

        pages_text = [t or "" for t in results]
        text = "\n".join(pages_text)
        metadata = self._extract_metadata(pdf_bytes)
        images = self._extract_images(pdf_bytes)

        tables: list[Table] = []
        for i, page_text in enumerate(pages_text):
            tables.extend(self._extract_tables_heuristic(page_text, i + 1))

        return Document(
            content=text,
            metadata=metadata,
            images=images,
            tables=tables,
            pages=pages_text,
            raw_bytes=pdf_bytes,
            source=self.loader.__class__.__name__,
        )

    def _extract_pdf_ocr_stream(
        self, pdf_bytes: bytes
    ) -> Generator[Document, None, None]:
        pages_img = convert_from_bytes(pdf_bytes, dpi=self.config.dpi)
        results: list[str | None] = [None] * len(pages_img)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._ocr_page, page): idx
                for idx, page in enumerate(pages_img)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = ""

        metadata = self._extract_metadata(pdf_bytes)
        images = self._extract_images(pdf_bytes)

        for idx, text in enumerate(results):
            page_text = text or ""
            tables = self._extract_tables_heuristic(page_text, idx + 1)

            yield Document(
                content=page_text,
                metadata=metadata,
                images=images,
                tables=tables,
                pages=[page_text],
                raw_bytes=pdf_bytes,
                source=self.loader.__class__.__name__,
            )

    def read(self) -> Document:
        pdf_bytes = self._read_bytes_batch()
        metadata = self._extract_metadata(pdf_bytes)
        has_text = self._pdf_has_text(pdf_bytes)

        if has_text and not self.config.enable_ocr:
            pages_text = self._extract_pdf_text_pages(pdf_bytes)
            text = "\n".join(pages_text)
            images = self._extract_images(pdf_bytes)
            tables: list[Table] = []
            for i, page_text in enumerate(pages_text):
                tables.extend(self._extract_tables_heuristic(page_text, i + 1))
            return Document(
                content=text,
                metadata=metadata,
                images=images,
                tables=tables,
                pages=pages_text,
                raw_bytes=pdf_bytes,
                source=self.loader.__class__.__name__,
            )

        if has_text and self.config.enable_ocr is False:
            pages_text = self._extract_pdf_text_pages(pdf_bytes)
            text = "\n".join(pages_text)
            images = self._extract_images(pdf_bytes)
            tables = []
            for i, page_text in enumerate(pages_text):
                tables.extend(self._extract_tables_heuristic(page_text, i + 1))
            return Document(
                content=text,
                metadata=metadata,
                images=images,
                tables=tables,
                pages=pages_text,
                raw_bytes=pdf_bytes,
                source=self.loader.__class__.__name__,
            )

        if has_text and self.config.enable_ocr:
            pages_text = self._extract_pdf_text_pages(pdf_bytes)
            text = "\n".join(pages_text)
            images = self._extract_images(pdf_bytes)
            tables = []
            for i, page_text in enumerate(pages_text):
                tables.extend(self._extract_tables_heuristic(page_text, i + 1))
            return Document(
                content=text,
                metadata=metadata,
                images=images,
                tables=tables,
                pages=pages_text,
                raw_bytes=pdf_bytes,
                source=self.loader.__class__.__name__,
            )

        return self._extract_pdf_ocr_batch(pdf_bytes)

    def read_streaming(self) -> Generator[Document, None, None]:
        pdf_bytes = self._read_bytes_streaming()
        metadata = self._extract_metadata(pdf_bytes)
        has_text = self._pdf_has_text(pdf_bytes)

        if has_text and not self.config.enable_ocr:
            pages_text = self._extract_pdf_text_pages(pdf_bytes)
            images = self._extract_images(pdf_bytes)
            for i, page_text in enumerate(pages_text):
                tables = self._extract_tables_heuristic(page_text, i + 1)
                yield Document(
                    content=page_text,
                    metadata=metadata,
                    images=images,
                    tables=tables,
                    pages=[page_text],
                    raw_bytes=pdf_bytes,
                    source=self.loader.__class__.__name__,
                )
            return

        if has_text and self.config.enable_ocr is False:
            pages_text = self._extract_pdf_text_pages(pdf_bytes)
            images = self._extract_images(pdf_bytes)
            for i, page_text in enumerate(pages_text):
                tables = self._extract_tables_heuristic(page_text, i + 1)
                yield Document(
                    content=page_text,
                    metadata=metadata,
                    images=images,
                    tables=tables,
                    pages=[page_text],
                    raw_bytes=pdf_bytes,
                    source=self.loader.__class__.__name__,
                )
            return

        if has_text and self.config.enable_ocr:
            pages_text = self._extract_pdf_text_pages(pdf_bytes)
            images = self._extract_images(pdf_bytes)
            for i, page_text in enumerate(pages_text):
                tables = self._extract_tables_heuristic(page_text, i + 1)
                yield Document(
                    content=page_text,
                    metadata=metadata,
                    images=images,
                    tables=tables,
                    pages=[page_text],
                    raw_bytes=pdf_bytes,
                    source=self.loader.__class__.__name__,
                )
            return

        yield from self._extract_pdf_ocr_stream(pdf_bytes)

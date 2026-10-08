import io
import logging

import pytesseract
from PIL import Image

from nexum.common.ai.deep_learning.pdf_region_spitter import PDFRegionSplitterModel
from nexum.common.errors import NexumRuntimeError
from nexum.common.helpers.table import ocr_table_cells, extract_table_dataframe
from nexum.document.models import PDFOCRConfig, RegionType
from nexum.document.models import RegionSplitterConfig
from nexum.document.parser.base import BaseParser

logger = logging.getLogger(__name__)


class PDFOCRParser(BaseParser):
    def __init__(self, config: PDFOCRConfig | None = None):
        resolved_config = config or PDFOCRConfig()
        super().__init__(resolved_config)
        self.config: PDFOCRConfig = resolved_config

        splitter_config = RegionSplitterConfig(
            threshold=0.15,
            device="cpu",
            model_name="yolov8m-doclaynet.pt"
        )
        self.splitter = PDFRegionSplitterModel(splitter_config)

    def parse(self, page_bytes: bytes) -> dict:
        try:
            with Image.open(io.BytesIO(page_bytes)) as opened:
                rgb_image = opened.convert("RGB")

                regions = self.splitter.predict(rgb_image)

                doc = {
                    "text_blocks": [],
                    "tables": [],
                    "images": [],
                    "fields": {}
                }

                for region in regions:
                    if region.type == RegionType.TEXT:
                        xmin, ymin, xmax, ymax = region.box
                        crop = rgb_image.crop((xmin, ymin, xmax, ymax))
                        region.text = pytesseract.image_to_string(crop, lang="por").strip()

                        doc["text_blocks"].append({
                            "box": region.box,
                            "text": region.text
                        })

                    elif region.type == RegionType.TABLE:
                        ocr_data = ocr_table_cells(rgb_image, region)
                        df = extract_table_dataframe(ocr_data)

                        doc["tables"].append({
                            "box": region.box,
                            "dataframe": df
                        })

                    elif region.type == RegionType.IMAGE:
                        doc["images"].append({
                            "box": region.box
                        })

                return doc

        except Exception as e:
            logger.error("OCR pipeline failed: %s", e)
            raise NexumRuntimeError(f"OCR failed: {e}")

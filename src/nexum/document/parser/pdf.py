import io
import logging

import cv2
import numpy as np
from PIL import Image

from nexum.common.errors import NexumRuntimeError
from nexum.common.helpers.image import deskew, clahe, extract_dpi, sharpen, denoise, binarize, osd_rotation, run_ocr, \
    validate_max_size
from nexum.document.models import PDFOCRConfig
from nexum.document.parser.base import BaseParser

logger = logging.getLogger(__name__)


class PDFOCRParser(BaseParser):
    def __init__(self, config: PDFOCRConfig | None = None):
        resolved_config = config or PDFOCRConfig()
        super().__init__(resolved_config)
        self.config: PDFOCRConfig = resolved_config

    def parse(self, page_bytes: bytes) -> str:
        logger.info(
            "Starting OCR pipeline (lang=%s, psm=%d, oem=%d)",
            self.config.lang,
            self.config.psm,
            self.config.oem,
        )

        try:
            with Image.open(io.BytesIO(page_bytes)) as opened:
                rgb_image = opened.convert("RGB")

                dpi = extract_dpi(rgb_image)
                if dpi < self.config.min_dpi:
                    new_size = (
                        int(rgb_image.width * self.config.upscale_factor),
                        int(rgb_image.height * self.config.upscale_factor),
                    )
                    rgb_image = rgb_image.resize(
                        new_size, resample=Image.Resampling.LANCZOS
                    )

                arr = np.array(rgb_image)
                validate_max_size(arr, self.config.max_pixels)
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

                if self.config.enable_deskew:
                    gray = deskew(gray, self.config.deskew_max_angle)
                if self.config.enable_clahe:
                    gray = clahe(gray, self.config.clahe_clip_limit, self.config.clahe_tile_size)
                if self.config.enable_denoise:
                    gray = denoise(gray, self.config.denoise_method)
                if self.config.enable_sharpen:
                    gray = sharpen(gray, self.config.sharpen_sigma, self.config.sharpen_strength,
                                   self.config.sharpen_blur_weight)
                if self.config.enable_binarization:
                    gray = binarize(gray, self.config.binarization_method, self.config.adaptive_block_size,
                                    self.config.adaptive_c)

                ocr_image = Image.fromarray(gray)

                if self.config.enable_osd:
                    ocr_image = osd_rotation(ocr_image, self.config.osd_min_confidence)

                text = run_ocr(ocr_image, self.config.lang, self.config.psm, self.config.oem)

                if self.config.strip_empty_lines:
                    text = "\n".join(
                        " ".join(line.split())
                        for line in text.splitlines()
                        if line.strip()
                    )

                return text

        except Exception as e:
            logger.error("OCR pipeline failed: %s", e)
            raise NexumRuntimeError(f"OCR failed: {e}")

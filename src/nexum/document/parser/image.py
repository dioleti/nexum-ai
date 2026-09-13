import io
import logging

import cv2
import numpy as np
from PIL import Image

from nexum.common.errors import NexumRuntimeError
from nexum.common.helpers.image import deskew, clahe, denoise, sharpen, binarize, run_ocr_data, run_ocr, \
    validate_max_size
from nexum.document.models import ImageOCRConfig
from nexum.document.parser.base import BaseParser

logger = logging.getLogger(__name__)


class ImageOCRParser(BaseParser):
    def __init__(self, config: ImageOCRConfig | None = None):
        resolved_config = config or ImageOCRConfig()
        super().__init__(resolved_config)
        self.config: ImageOCRConfig = resolved_config

    def parse(self, image_bytes: bytes):
        try:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.array(pil)
            validate_max_size(arr)

            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            if self.config.enable_deskew:
                gray = deskew(gray)
            if self.config.enable_clahe:
                gray = clahe(gray, self.config.clahe_clip_limit, self.config.clahe_tile_size)
            if self.config.enable_denoise:
                gray = denoise(gray, self.config.denoise_method)
            if self.config.enable_sharpen:
                gray = sharpen(gray, self.config.sharpen_sigma, self.config.sharpen_strength)
            if self.config.enable_binarization:
                gray = binarize(gray, self.config.binarization_method)

            if self.config.return_structured:
                data = run_ocr_data(gray)
                return data

            text = run_ocr(gray)

            if self.config.strip_empty_lines:
                text = "\n".join(
                    " ".join(line.split())
                    for line in text.splitlines()
                    if line.strip()
                )

            return text

        except Exception as exc:
            logger.error(f"Image OCR failed: {exc}")
            raise NexumRuntimeError(f"Image OCR failed: {exc}")

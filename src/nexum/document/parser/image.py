import io
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image

from nexum.common.errors import NexumRuntimeError
from nexum.document.models import ImageOCRConfig
from nexum.document.parser.base import BaseParser

logger = logging.getLogger(__name__)


class ImageOCRParser(BaseParser):
    def __init__(self, config: ImageOCRConfig | None = None):
        resolved_config = config or ImageOCRConfig()
        super().__init__(resolved_config)
        self.config: ImageOCRConfig = resolved_config

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        if not self.config.enable_denoise:
            return gray
        if self.config.denoise_method == "bilateral":
            return cv2.bilateralFilter(gray, 9, 75, 75)
        if self.config.denoise_method == "median":
            return cv2.medianBlur(gray, 3)
        return gray

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        if not self.config.enable_sharpen:
            return gray
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, self.config.sharpen_strength, blurred, -0.2, 0)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        if not self.config.enable_binarization:
            return gray
        std = gray.std()
        if std > self.config.binarization_min_std:
            return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        return gray

    def parse(self, image_bytes: bytes) -> str:
        try:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
            gray = self._denoise(gray)
            gray = self._sharpen(gray)
            gray = self._binarize(gray)
            text = pytesseract.image_to_string(gray, lang=self.config.lang)
            text = text.replace("\x0c", "")
            if self.config.strip_empty_lines:
                text = "\n".join(
                    line.strip() for line in text.splitlines() if line.strip()
                )
            return text
        except Exception as exc:
            logger.error(f"Image OCR failed: {exc}")
            raise NexumRuntimeError(f"Image OCR failed: {exc}")

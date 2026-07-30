import io
import logging
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

from nexum.common.errors import NexumRuntimeError
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

                dpi = PDFOCRParser._extract_dpi(rgb_image)

                if dpi < self.config.min_dpi:
                    new_size = (
                        int(rgb_image.width * self.config.upscale_factor),
                        int(rgb_image.height * self.config.upscale_factor),
                    )
                    rgb_image = rgb_image.resize(
                        new_size, resample=Image.Resampling.LANCZOS
                    )

                gray = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)

                if self.config.enable_deskew:
                    gray = PDFOCRParser._deskew(gray, self.config)

                if self.config.enable_sharpen:
                    gray = PDFOCRParser._sharpen(gray, self.config)

                if self.config.enable_denoise:
                    gray = PDFOCRParser._denoise(gray, self.config)

                if (
                    self.config.enable_binarization
                    and gray.std() > self.config.binarization_min_std
                ):
                    gray = PDFOCRParser._binarize(gray, self.config)

                ocr_image = Image.fromarray(gray)

                if self.config.enable_osd:
                    PDFOCRParser._apply_osd_rotation(ocr_image, self.config)

                ocr_config = f"--psm {self.config.psm} --oem {self.config.oem}"
                text = pytesseract.image_to_string(
                    ocr_image, lang=self.config.lang, config=ocr_config
                )
                text = text.replace("\x0c", "")

                if self.config.strip_empty_lines:
                    text = "\n".join(
                        line.strip() for line in text.splitlines() if line.strip()
                    )

                return text

        except Exception as e:
            logger.error("OCR pipeline failed: %s")
            raise NexumRuntimeError(f"OCR failed: {e}")

    @staticmethod
    def _extract_dpi(image: Image.Image) -> int:
        dpi = image.info.get("dpi", (72, 72))[0]
        return dpi or 72

    @staticmethod
    def _deskew(gray: np.ndarray, config: PDFOCRConfig) -> np.ndarray:
        coords = np.column_stack(np.where(gray < 255))
        if coords.size == 0:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > config.deskew_max_angle:
            return gray
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)

    @staticmethod
    def _sharpen(gray: np.ndarray, config: PDFOCRConfig) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, config.sharpen_strength, blurred, -0.2, 0)

    @staticmethod
    def _denoise(gray: np.ndarray, config: PDFOCRConfig) -> np.ndarray:
        if config.denoise_method == "bilateral":
            return cv2.bilateralFilter(gray, 9, 75, 75)
        if config.denoise_method == "median":
            return cv2.medianBlur(gray, 3)
        return gray

    @staticmethod
    def _binarize(gray: np.ndarray, config: PDFOCRConfig) -> np.ndarray:
        if config.binarization_method == "otsu":
            return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        if config.binarization_method == "adaptive":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
            )
        return gray

    @staticmethod
    def _apply_osd_rotation(image: Image.Image, config: PDFOCRConfig):
        try:
            osd = pytesseract.image_to_osd(image)
            rotate_match = re.search(r"Rotate: (\d+)", osd)
            conf_match = re.search(r"Orientation confidence: (\d+)", osd)

            if not rotate_match or not conf_match:
                return

            rotate = int(rotate_match.group(1))
            confidence = float(conf_match.group(1))

            if confidence >= config.osd_min_confidence and rotate in (90, 180, 270):
                image.rotate(360 - rotate, expand=True)

        except Exception as e:
            logger.error("OSD rotation failed: %s", e)

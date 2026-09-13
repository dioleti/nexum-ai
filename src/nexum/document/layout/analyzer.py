import io
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image

from nexum.common.helpers.image import (
    extract_dpi,
    validate_max_size,
    deskew,
    clahe,
    denoise,
    sharpen,
    binarize,
)
from nexum.document.layout.detectors.block import BlockDetector
from nexum.document.layout.detectors.flowchart import FlowchartDetectorDL
from nexum.document.layout.detectors.image import ImageDetector
from nexum.document.layout.detectors.table import TableDetector


class LayoutAnalyzer:
    def __init__(
        self,
        enable_preprocess: bool = True,
        enable_deskew: bool = True,
        enable_clahe: bool = True,
        enable_denoise: bool = True,
        enable_sharpen: bool = True,
        enable_binarize: bool = False,
        enable_table_detector: bool = True,
        enable_block_detector: bool = True,
        enable_image_detector: bool = True,
        enable_flowchart_detector_dl: bool = False,
        deskew_max_angle: float = 10.0,
        clahe_clip_limit: float = 40.0,
        clahe_tile_size: tuple[int, int] = (8, 8),
        denoise_method: str = "bilateral",
        sharpen_sigma: float = 1.0,
        sharpen_strength: float = 1.2,
        sharpen_blur_weight: float = -0.2,
        binarization_method: str = "adaptive",
        adaptive_block_size: int = 31,
        adaptive_c: int = 2
    ):
        self.enable_preprocess = enable_preprocess
        self.enable_deskew = enable_deskew
        self.deskew_max_angle = deskew_max_angle
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.enable_denoise = enable_denoise
        self.denoise_method = denoise_method
        self.enable_sharpen = enable_sharpen
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_strength = sharpen_strength
        self.sharpen_blur_weight = sharpen_blur_weight
        self.enable_binarize = enable_binarize
        self.binarization_method = binarization_method
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.table_detector = TableDetector() if enable_table_detector else None
        self.block_detector = BlockDetector() if enable_block_detector else None
        self.image_detector = ImageDetector() if enable_image_detector else None
        self.flowchart_detector = FlowchartDetectorDL() if enable_flowchart_detector_dl else None

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        if not self.enable_preprocess:
            return gray
        if self.enable_deskew:
            gray = deskew(gray, self.deskew_max_angle)
        if self.enable_clahe:
            gray = clahe(gray, self.clahe_clip_limit, self.clahe_tile_size)
        if self.enable_denoise:
            gray = denoise(gray, self.denoise_method)
        if self.enable_sharpen:
            gray = sharpen(
                gray,
                self.sharpen_sigma,
                self.sharpen_strength,
                self.sharpen_blur_weight,
            )
        if self.enable_binarize:
            gray = binarize(
                gray,
                self.binarization_method,
                self.adaptive_block_size,
                self.adaptive_c,
            )

        return gray

    def analyze_pdf(self, page_bytes: bytes) -> Dict[str, Any]:
        with Image.open(io.BytesIO(page_bytes)) as opened:
            rgb_image = opened.convert("RGB")

            dpi = extract_dpi(rgb_image)
            if dpi < 150:
                upscale_factor = 2.0
                new_size = (
                    int(rgb_image.width * upscale_factor),
                    int(rgb_image.height * upscale_factor),
                )
                rgb_image = rgb_image.resize(new_size, Image.Resampling.LANCZOS)

            arr = np.array(rgb_image)
            validate_max_size(arr)

            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            gray = self._preprocess(gray)

            return self.run_detectors(gray)

    def analyze_image(self, image: Any) -> Dict[str, Any]:
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        else:
            rgb = image

        validate_max_size(rgb)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = self._preprocess(gray)

        return self.run_detectors(gray)

    def run_detectors(self, gray: np.ndarray) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if self.table_detector:
            results["tables"] = self.table_detector.detect(gray)
        if self.block_detector:
            results["blocks"] = self.block_detector.detect(gray)
        if self.image_detector:
            results["images"] = self.image_detector.detect(gray)
        if self.flowchart_detector:
            results["flowcharts"] = self.flowchart_detector.detect(gray)

        return results

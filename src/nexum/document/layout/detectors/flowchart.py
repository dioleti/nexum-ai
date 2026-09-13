from typing import List, Tuple

import cv2
import layoutparser as lp
import numpy as np

from nexum.common.helpers.image import (
    deskew,
    clahe,
    denoise,
    sharpen,
    binarize,
)


class FlowchartRegion:
    def __init__(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox

    def __repr__(self):
        return f"FlowchartRegion(bbox={self.bbox})"


class FlowchartDetectorDL:
    def __init__(
        self,
        enable_deskew: bool = True,
        enable_clahe: bool = True,
        enable_denoise: bool = True,
        enable_sharpen: bool = True,
        enable_binarize: bool = False,
        score_threshold: float = 0.8,
        deskew_max_angle: float = 10.0,
        clahe_clip_limit: float = 40.0,
        clahe_tile_size: tuple[int, int] = (8, 8),
        denoise_method: str = "bilateral",
        sharpen_sigma: float = 1.0,
        sharpen_strength: float = 1.2,
        sharpen_blur_weight: float = -0.2,
        binarization_method: str = "adaptive",
        adaptive_block_size: int = 31,
        adaptive_c: int = 2,
    ):
        self.score_threshold = score_threshold
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
        self.model = lp.Detectron2LayoutModel(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.score_threshold],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        )

    def _preprocess(self, rgb_image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        if self.enable_deskew:
            gray_image = deskew(gray_image, self.deskew_max_angle)
        if self.enable_clahe:
            gray_image = clahe(gray_image, self.clahe_clip_limit, self.clahe_tile_size)
        if self.enable_denoise:
            gray_image = denoise(gray_image, self.denoise_method)
        if self.enable_sharpen:
            gray_image = sharpen(
                gray_image,
                self.sharpen_sigma,
                self.sharpen_strength,
                self.sharpen_blur_weight,
            )
        if self.enable_binarize:
            gray_image = binarize(
                gray_image,
                self.binarization_method,
                self.adaptive_block_size,
                self.adaptive_c,
            )

        return gray_image

    def detect(self, rgb_image: np.ndarray) -> List[FlowchartRegion]:
        preprocessed_gray = self._preprocess(rgb_image)

        processed_rgb = cv2.cvtColor(preprocessed_gray, cv2.COLOR_GRAY2RGB)

        layout = self.model.detect(processed_rgb)

        figure_blocks = [block for block in layout if block.type == "Figure"]

        if not figure_blocks:
            return []

        regions: List[FlowchartRegion] = []

        for block in figure_blocks:
            x1, y1, x2, y2 = block.block.coordinates
            regions.append(FlowchartRegion((int(x1), int(y1), int(x2), int(y2))))

        return regions

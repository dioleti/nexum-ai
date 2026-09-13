from typing import List, Tuple

import cv2
import numpy as np

from nexum.common.helpers.image import (
    deskew,
    clahe,
    denoise,
    sharpen,
    binarize,
)


class ImageRegion:
    def __init__(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox

    def __repr__(self):
        return f"ImageRegion(bbox={self.bbox})"


class ImageDetector:
    def __init__(
        self,
        min_image_area: int = 4000,
        enable_deskew: bool = True,
        enable_clahe: bool = True,
        enable_denoise: bool = True,
        enable_sharpen: bool = True,
        enable_binarize: bool = True,
        dilation_kernel_size: Tuple[int, int] = (20, 20),
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
        self.min_image_area = min_image_area
        self.dilation_kernel_size = dilation_kernel_size
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

    def detect(self, gray_image: np.ndarray) -> List[ImageRegion]:
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

        _, binary_image = cv2.threshold(
            gray_image,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            self.dilation_kernel_size
        )
        dilated_mask = cv2.dilate(binary_image, dilation_kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detected_images: List[ImageRegion] = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height

            if area < self.min_image_area:
                continue

            detected_images.append(
                ImageRegion((x, y, x + width, y + height))
            )

        return detected_images

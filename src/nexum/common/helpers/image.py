import logging
import re
from typing import Union

import cv2
import numpy as np
import pytesseract
from PIL import Image

from nexum.common.errors import NexumRuntimeError

logger = logging.getLogger(__name__)


def extract_dpi(image: Image.Image) -> int:
    dpi = image.info.get("dpi", (72, 72))[0]
    return dpi or 72


def deskew(gray: np.ndarray, max_angle: float | None = None) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if max_angle is not None and abs(angle) > max_angle:
        return gray

    (height, width) = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC)


def clahe(gray: np.ndarray, clahe_clip_limit: float = 40.0, clahe_tile_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    _clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_tile_size,
    )
    return _clahe.apply(gray)


def sharpen(
    gray: np.ndarray,
    sharpen_sigma: float = 1.0,
    sharpen_strength: float = 1.2,
    sharpen_blur_weight: float = -0.2) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sharpen_sigma)

    return cv2.addWeighted(
        gray,
        sharpen_strength,
        blurred,
        sharpen_blur_weight,
        0,
    )


def denoise(gray: np.ndarray, denoise_method: str = "bilateral") -> np.ndarray:
    if denoise_method == "bilateral":
        return cv2.bilateralFilter(gray, 9, 75, 75)
    if denoise_method == "median":
        return cv2.medianBlur(gray, 3)
    return gray


def binarize(gray: np.ndarray, binarization_method: str = "otsu", adaptive_block_size: int = 31,
             adaptive_c: int = 2) -> np.ndarray:
    if binarization_method == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    if binarization_method == "adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block_size,
            adaptive_c,
        )

    return gray


def osd_rotation(image: Image.Image, osd_min_confidence: float = 5.0) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(image)
        rotate_match = re.search(r"Rotate: (\d+)", osd)
        conf_match = re.search(r"Orientation confidence: (\d+)", osd)

        if not rotate_match or not conf_match:
            return image

        rotate = int(rotate_match.group(1))
        confidence = float(conf_match.group(1))

        if confidence >= osd_min_confidence and rotate in (90, 180, 270):
            return image.rotate(360 - rotate, expand=True)

        return image

    except Exception as e:
        logger.error("OSD rotation failed: %s", e)
        return image


def run_ocr(gray: Union[np.ndarray, Image.Image], lang: str = "eng", oem: int = 1, psm: int = 3) -> str:
    config = f"--psm {psm} --oem {oem}"
    text = pytesseract.image_to_string(
        gray,
        lang=lang,
        config=config,
    )
    return text.replace("\x0c", "")


def run_ocr_data(gray: Union[np.ndarray, Image.Image], lang: str = "eng", oem: int = 1, psm: int = 3):
    config = f"--psm {psm} --oem {oem}"
    return pytesseract.image_to_data(
        gray,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )


def validate_max_size(arr: np.ndarray, max_px: int | None = 20_000_000):
    height, width = arr.shape[:2]
    total_area = height * width
    if total_area > max_px:
        raise NexumRuntimeError(
            f"Image too large ({total_area} px). Limit is {max_px} px."
        )

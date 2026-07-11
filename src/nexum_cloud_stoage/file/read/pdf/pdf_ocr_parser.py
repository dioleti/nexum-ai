import io
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image
from nexum.errors import NexumRuntimeError


class PDFOCRParser:
    """
    OCR parser responsible for extracting text from rasterized PDF pages.

    This parser applies adaptive preprocessing steps to improve OCR accuracy,
    including resolution enhancement, grayscale conversion, binarization,
    and automatic orientation correction. It is designed for pages that have
    been previously rasterized into image bytes (PNG/JPEG).

    Notes
    -----
    - This parser does not extract embedded PDF text; it operates strictly on
      image bytes.
    - Preprocessing is adaptive: upscale is applied only to small images, and
      binarization is applied only when contrast is sufficient.
    - Orientation detection uses Tesseract OSD and is applied only when the
      detected rotation is valid (90, 180, 270 degrees).
    """

    @staticmethod
    def parse(page_bytes: bytes, lang: str = "eng") -> str:
        """
        Perform OCR on a single rasterized PDF page.

        Parameters
        ----------
        page_bytes : bytes
            Raw image bytes representing a PDF page (PNG/JPEG).
        lang : str, optional
            Tesseract language code used for OCR. Defaults to "eng".
            Examples: "por", "spa", "fra", "deu", "ita", "jpn".

        Returns
        -------
        str
            Extracted text after OCR and post‑processing.

        Raises
        ------
        RuntimeError
            If the image cannot be processed or OCR fails.
        """
        try:
            with Image.open(io.BytesIO(page_bytes)) as image:
                image = image.convert("RGB")

                if image.width < 1000:
                    image = image.resize(
                        (image.width * 2, image.height * 2),
                        resample=Image.Resampling.LANCZOS
                    )

                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

                if gray.std() > 15:
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
                    image = Image.fromarray(thresh)

                try:
                    osd = pytesseract.image_to_osd(image)
                    match = re.search(r"Rotate: (\d+)", osd)
                    if match:
                        rotate = int(match.group(1))
                        if rotate in (90, 180, 270):
                            image = image.rotate(360 - rotate, expand=True)
                except Exception:
                    pass

                text = pytesseract.image_to_string(
                    image,
                    lang=lang,
                    config="--psm 4 --oem 3"
                )

                text = text.replace("\x0c", "")
                text = "\n".join(
                    line.strip() for line in text.splitlines() if line.strip()
                )

                return text

        except Exception as e:
            raise NexumRuntimeError(f"OCR failed: {e}")

from typing import Sequence, List

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from nexum.common.ai.deep_learning.base_model import StatefulModel
from nexum.common.helpers.image import detect_images_heuristic, ocr_region
from nexum.document.models import Region, RegionType, RegionSplitterConfig


class PDFRegionSplitterModel(StatefulModel[Image.Image, List[Region]]):

    def __init__(self, config: RegionSplitterConfig | None = None) -> None:
        self.config = config or RegionSplitterConfig()
        self.device = (
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: YOLO | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.is_loaded:
            self.model = YOLO("yolov8m-doclaynet.pt")
            self.model.to(self.device)

    def unload(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, input_data: Image.Image) -> List[Region]:
        return self.predict_batch([input_data])[0]

    def predict_batch(
        self, inputs: Sequence[Image.Image]
    ) -> List[List[Region]]:
        if not inputs:
            return []

        if not self.is_loaded:
            self.load()

        all_pages_regions: List[List[Region]] = []

        for img in inputs:
            np_img = np.array(img)

            results = self.model.predict(
                np_img,
                conf=self.config.threshold or 0.25,
                verbose=False
            )

            yolo_regions = self._extract_regions_from_prediction(results[0])
            heuristic_images = detect_images_heuristic(img)

            merged = self._merge_regions(yolo_regions, heuristic_images)

            for region in merged:
                if region.type == RegionType.TEXT:
                    region.text = ocr_region(img, region)

            all_pages_regions.append(merged)

        return all_pages_regions

    def _extract_regions_from_prediction(self, result) -> List[Region]:
        regions: List[Region] = []

        boxes = result.boxes
        if boxes is None:
            return regions

        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            score = float(box.conf[0])
            label_id = int(box.cls[0])
            label_name = result.names[label_id]

            region_type = self._map_label_to_region_type(label_name)
            if region_type is None:
                continue

            regions.append(
                Region(
                    type=region_type,
                    box=(int(xmin), int(ymin), int(xmax), int(ymax)),
                    score=score,
                    text=None
                )
            )

        return regions

    @staticmethod
    def _map_label_to_region_type(label_name: str) -> RegionType | None:
        label = label_name.lower()

        if label == "table":
            return RegionType.TABLE

        if label in ("text", "title", "caption", "section", "list", "header", "footer"):
            return RegionType.TEXT

        if label == "figure":
            return RegionType.IMAGE

        return None

    @staticmethod
    def _merge_regions(yolo_regions: List[Region], heuristic_images: List[Region]) -> List[Region]:
        merged = list(yolo_regions)

        for img_region in heuristic_images:
            if not any(PDFRegionSplitterModel._overlaps(img_region, r) for r in yolo_regions):
                merged.append(img_region)

        return merged

    @staticmethod
    def _overlaps(a: Region, b: Region) -> bool:
        ax1, ay1, ax2, ay2 = a.box
        bx1, by1, bx2, by2 = b.box

        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

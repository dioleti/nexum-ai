from typing import Sequence

import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from transformers import logging as hf_logging

from nexum.common.ai.deep_learning.base_model import StatefulModel
from nexum.document.models import TableDetectionInput, DetectedTable, TableDetectionConfig, BoundingBox


class TableDetector(
    StatefulModel[TableDetectionInput, list[DetectedTable]]
):
    def __init__(self, config: TableDetectionConfig | None = None) -> None:
        self.config = config or TableDetectionConfig()
        self.device = torch.device(
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor: AutoImageProcessor | None = None
        self.model: TableTransformerForObjectDetection | None = None

    @property
    def is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    def load(self) -> None:
        if not self.is_loaded:
            hf_logging.set_verbosity_error()
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.model_name
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                self.config.model_name
            ).to(self.device)
            self.model.eval()

    def unload(self) -> None:
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, input_data: TableDetectionInput) -> list[DetectedTable]:
        return self.predict_batch([input_data])[0]

    def predict_batch(
        self, inputs: Sequence[TableDetectionInput]
    ) -> list[list[DetectedTable]]:
        if not inputs:
            return []

        if not self.is_loaded:
            self.load()

        assert self.processor is not None
        assert self.model is not None

        images = [item.image for item in inputs]
        batch = self.processor(images=images, return_tensors="pt").to(
            self.device
        )

        with torch.inference_mode():
            outputs = self.model(**batch)

        target_sizes = torch.tensor(
            [img.size[::-1] for img in images], device=self.device
        )

        all_results: list[list[DetectedTable]] = []

        for idx, input_item in enumerate(inputs):
            threshold = (
                input_item.threshold
                if input_item.threshold is not None
                else self.config.default_threshold
            )

            post_processed = self.processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes[idx: idx + 1],
            )[idx]

            detected_tables = self._extract_tables_from_prediction(
                post_processed
            )
            all_results.append(detected_tables)

        return all_results

    def _extract_tables_from_prediction(
        self, prediction: dict[str, torch.Tensor]
    ) -> list[DetectedTable]:
        assert self.model is not None

        tables: list[DetectedTable] = []
        for score, label, box in zip(
            prediction["scores"],
            prediction["labels"],
            prediction["boxes"],
        ):
            label_name = self.model.config.id2label[label.item()]
            if "table" in label_name:
                coords = [round(float(c), 1) for c in box.tolist()]
                tables.append(
                    DetectedTable(
                        label=label_name,
                        score=round(float(score), 4),
                        box=BoundingBox(
                            xmin=coords[0],
                            ymin=coords[1],
                            xmax=coords[2],
                            ymax=coords[3],
                        ),
                    )
                )

        return tables

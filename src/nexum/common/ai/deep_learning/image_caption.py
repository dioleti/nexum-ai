from typing import Sequence

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers import logging as hf_logging

from nexum.common.ai.deep_learning.base_model import StatefulModel
from nexum.document.models import CaptionInput, CaptionConfig


class ImageCaptionModel(StatefulModel[CaptionInput, str]):
    def __init__(self, config: CaptionConfig | None = None) -> None:
        self.config = config or CaptionConfig()
        self.device = torch.device(
            self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor: BlipProcessor | None = None
        self.model: BlipForConditionalGeneration | None = None

    @property
    def is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    def load(self) -> None:
        if not self.is_loaded:
            hf_logging.set_verbosity_error()
            self.processor = BlipProcessor.from_pretrained(
                self.config.model_name
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.model_name
            ).to(self.device)
            self.model.eval()

    def unload(self) -> None:
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, input_data: CaptionInput) -> str:
        return self.predict_batch([input_data])[0]

    def predict_batch(self, inputs: Sequence[CaptionInput]) -> list[str]:
        if not inputs:
            return []

        if not self.is_loaded:
            self.load()

        assert self.processor is not None
        assert self.model is not None

        images = [item.image for item in inputs]
        prompts = [item.prompt for item in inputs]
        has_prompts = any(p is not None for p in prompts)

        if has_prompts:
            resolved_prompts = [p if p is not None else "" for p in prompts]
            batch = self.processor(
                images=images,
                text=resolved_prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            batch = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=self.config.max_new_tokens,
            )

        return [
            self.processor.decode(out, skip_special_tokens=True).strip()
            for out in outputs
        ]

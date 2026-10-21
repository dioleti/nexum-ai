import re
from typing import Sequence

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from nexum.common.ai.base_model import StatefulModel
from nexum.common.helpers.device import resolve
from nexum.common.models import PiiDetectionInput, PiiDetectionResult, PIIDetectorConfig


class PIIDetectorDL(StatefulModel[PiiDetectionInput, PiiDetectionResult]):
    _CPF_PATTERN = re.compile(r"^\d{3}\.?\d{3}\.?\d{3}-?\d{2}$")
    _CNPJ_PATTERN = re.compile(r"^\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}$")
    _EMAIL_PATTERN = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")
    _PHONE_PATTERN = re.compile(r"^(\+?55\s?)?(\(?\d{2}\)?\s?)?(9?\d{4}-?\d{4})$")

    _PII_ENTITIES = {
        "PER",
        "PESSOA",
        "LOC",
        "LOCAL",
    }
    _SENSITIVE_ENTITIES = {
        "RELIGION",
        "RACE",
        "ETHNICITY",
        "POLITICAL_OPINION",
        "TRADE_UNION",
        "HEALTH_DATA",
        "GENETIC_DATA",
        "BIOMETRIC_DATA",
        "SEX_LIFE_ORIENTATION"
    }

    def __init__(
            self,
            config: PIIDetectorConfig = PIIDetectorConfig(),
            device: torch.device | None = None,
            batch_size: int = 32,
    ) -> None:
        self.model_name = config.model_name
        self.device = device or resolve()
        self.batch_size = batch_size

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForTokenClassification | None = None
        self.ner_pipeline: pipeline | None = None

    @property
    def is_loaded(self) -> bool:
        return self.ner_pipeline is not None

    def load(self) -> None:
        if not self.is_loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name
            )

            if self.device.type == "cpu":
                self.model.eval()
                torch.set_num_threads(min(4, torch.get_num_threads()))

            device_id = 0 if self.device.type == "cuda" else -1
            self.ner_pipeline = pipeline(
                task="token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=device_id,
                batch_size=self.batch_size,
            )

    def unload(self) -> None:
        self.ner_pipeline = None
        self.model = None
        self.tokenizer = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def predict(self, input_data: PiiDetectionInput) -> PiiDetectionResult:
        return self.predict_batch([input_data])[0]

    def predict_batch(
            self, inputs: Sequence[PiiDetectionInput]
    ) -> list[PiiDetectionResult]:
        if not inputs:
            return []

        results: list[PiiDetectionResult | None] = [None] * len(inputs)

        dl_indices: list[int] = []
        flattened_samples: list[str] = []
        sample_to_col_idx: list[int] = []

        for col_idx, item in enumerate(inputs):
            if not item.samples:
                results[col_idx] = PiiDetectionResult(is_pii=False, confidence=0.0)
                continue

            regex_hits = self._check_regex(item.samples)
            ratio = regex_hits / len(item.samples)

            if ratio >= item.threshold:
                results[col_idx] = PiiDetectionResult(
                    is_pii=True,
                    confidence=round(ratio, 4),
                    detected_entities={"REGEX_PATTERN": regex_hits},
                )
            else:
                clean_samples = [s.strip() for s in item.samples if len(s.strip()) > 2]
                if not clean_samples:
                    results[col_idx] = PiiDetectionResult(is_pii=False, confidence=0.0)
                    continue

                dl_indices.append(col_idx)
                for s in clean_samples:
                    flattened_samples.append(s)
                    sample_to_col_idx.append(col_idx)

        if not flattened_samples:
            return [r or PiiDetectionResult(is_pii=False, confidence=0.0) for r in results]

        if not self.is_loaded:
            self.load()

        predictions = self.ner_pipeline(flattened_samples)

        col_pii_counts: dict[int, int] = {idx: 0 for idx in dl_indices}
        col_total_counts: dict[int, int] = {idx: 0 for idx in dl_indices}
        col_entities: dict[int, dict[str, int]] = {idx: {} for idx in dl_indices}

        for pred, col_idx in zip(predictions, sample_to_col_idx):
            col_total_counts[col_idx] += 1
            has_entity = False

            for ent in pred:
                ent_group = ent.get("entity_group", "")
                score = ent.get("score", 0.0)

                if score >= 0.70 and ent_group in self._PII_ENTITIES:
                    entities_dict = col_entities[col_idx]
                    entities_dict[ent_group] = entities_dict.get(ent_group, 0) + 1
                    has_entity = True

            if has_entity:
                col_pii_counts[col_idx] += 1

        for col_idx in dl_indices:
            total = col_total_counts[col_idx]
            hits = col_pii_counts[col_idx]
            ratio = hits / total if total > 0 else 0.0
            threshold = inputs[col_idx].threshold

            results[col_idx] = PiiDetectionResult(
                is_pii=ratio >= threshold,
                confidence=round(ratio, 4),
                detected_entities=col_entities[col_idx],
            )

        return [r or PiiDetectionResult(is_pii=False, confidence=0.0) for r in results]

    @classmethod
    def _check_regex(cls, samples: Sequence[str]) -> int:
        hits = 0
        for val in samples:
            clean = val.strip()
            if (
                    cls._CPF_PATTERN.match(clean)
                    or cls._CNPJ_PATTERN.match(clean)
                    or cls._EMAIL_PATTERN.match(clean)
                    or cls._PHONE_PATTERN.match(clean)
            ):
                hits += 1
        return hits

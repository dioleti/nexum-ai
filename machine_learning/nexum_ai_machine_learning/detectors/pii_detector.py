import re
from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from nexum_ai_common import resolve
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

from nexum_ai_machine_learning.common.models import (
    BaseModel,
    PiiDetectionInput,
    PiiDetectionResult,
    PIIDetectorConfig,
    StatefulModel,
)


class PIIDetector(StatefulModel[str, PiiDetectionResult]):
    _CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
    _CNPJ_PATTERN = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
    _EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    _PHONE_PATTERN = re.compile(
        r"(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)(?:9?\d{4}-?\d{4})\b"
    )

    _PII_ENTITIES: ClassVar[set[str]] = {
        "PER",
        "PESSOA",
        "LOC",
        "LOCAL",
    }
    _SENSITIVE_ENTITIES: ClassVar[set[str]] = {
        "RELIGION",
        "RACE",
        "ETHNICITY",
        "POLITICAL_OPINION",
        "TRADE_UNION",
        "HEALTH_DATA",
        "GENETIC_DATA",
        "BIOMETRIC_DATA",
        "SEX_LIFE_ORIENTATION",
    }

    def __init__(
        self,
        config: PIIDetectorConfig | None = None,
        device: torch.device | None = None,
        batch_size: int = 32,
    ) -> None:
        config = config or PIIDetectorConfig()
        self.model_name = config.model_name
        self.score_threshold = config.score_threshold
        self.max_chunk_chars = config.max_chunk_chars
        self.device = device or resolve()
        self.batch_size = batch_size

        self.tokenizer: Any = None
        self.model: Any = None
        self.ner_pipeline: Pipeline | None = None

    @property
    def is_loaded(self) -> bool:
        return self.ner_pipeline is not None

    def load(self) -> None:
        if not self.is_loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                self.model_name, use_fast=True
            )
            self.model = AutoModelForTokenClassification.from_pretrained(  # nosec B615
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

    def predict(self, input_data: str) -> PiiDetectionResult:
        return self.predict_batch([input_data])[0]

    def predict_batch(
        self, inputs: Sequence[str]
    ) -> list[PiiDetectionResult]:
        if not inputs:
            return []

        results: list[PiiDetectionResult | None] = [None] * len(inputs)
        chunks_to_process: list[str] = []
        chunk_to_input_idx: list[int] = []
        regex_entities_per_input: dict[int, dict[str, int]] = {}

        for idx, text in enumerate(inputs):
            clean_text = text.strip()
            if not clean_text:
                results[idx] = PiiDetectionResult(is_pii=False, confidence=0.0)
                continue

            regex_hits = self._extract_regex_entities(clean_text)
            regex_entities_per_input[idx] = regex_hits

            # Exact single-value match (e.g. a standalone CPF/Email/Phone/CNPJ)
            if self._is_exact_regex_match(clean_text):
                total_regex = sum(regex_hits.values())
                results[idx] = PiiDetectionResult(
                    is_pii=True,
                    confidence=1.0,
                    detected_entities={"REGEX_PATTERN": total_regex, **regex_hits},
                )
                continue

            if len(clean_text) <= 2:
                if regex_hits:
                    total_regex = sum(regex_hits.values())
                    results[idx] = PiiDetectionResult(
                        is_pii=True,
                        confidence=1.0,
                        detected_entities={"REGEX_PATTERN": total_regex, **regex_hits},
                    )
                else:
                    results[idx] = PiiDetectionResult(is_pii=False, confidence=0.0)
                continue

            for chunk in self._chunk_text(clean_text, self.max_chunk_chars):
                chunks_to_process.append(chunk)
                chunk_to_input_idx.append(idx)

        if not chunks_to_process:
            return [r or PiiDetectionResult(is_pii=False, confidence=0.0) for r in results]

        if self.ner_pipeline is None:
            self.load()

        if self.ner_pipeline is None:
            return [r or PiiDetectionResult(is_pii=False, confidence=0.0) for r in results]

        predictions = self.ner_pipeline(chunks_to_process)

        ner_entities_per_input: dict[int, dict[str, int]] = {}
        ner_scores_per_input: dict[int, list[float]] = {}

        for pred, input_idx in zip(predictions, chunk_to_input_idx):
            for ent in pred:
                ent_group = ent.get("entity_group", "")
                score = float(ent.get("score", 0.0))

                if score >= self.score_threshold and ent_group in self._PII_ENTITIES:
                    ent_dict = ner_entities_per_input.setdefault(input_idx, {})
                    ent_dict[ent_group] = ent_dict.get(ent_group, 0) + 1
                    ner_scores_per_input.setdefault(input_idx, []).append(score)

        for idx in range(len(inputs)):
            if results[idx] is not None:
                continue

            combined_entities: dict[str, int] = {}
            regex_hits = regex_entities_per_input.get(idx, {})
            if regex_hits:
                combined_entities["REGEX_PATTERN"] = sum(regex_hits.values())
                combined_entities.update(regex_hits)

            ner_hits = ner_entities_per_input.get(idx, {})
            for k, v in ner_hits.items():
                combined_entities[k] = combined_entities.get(k, 0) + v

            scores = ner_scores_per_input.get(idx, [])
            if regex_hits:
                scores.append(1.0)

            if combined_entities:
                confidence = round(max(scores), 4) if scores else 1.0
                results[idx] = PiiDetectionResult(
                    is_pii=True,
                    confidence=confidence,
                    detected_entities=combined_entities,
                )
            else:
                results[idx] = PiiDetectionResult(
                    is_pii=False,
                    confidence=0.0,
                    detected_entities={},
                )

        return [r or PiiDetectionResult(is_pii=False, confidence=0.0) for r in results]

    @classmethod
    def _chunk_text(cls, text: str, max_chars: int) -> list[str]:
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + max_chars, text_len)
            if end < text_len:
                split_pos = text.rfind(" ", start, end)
                if split_pos > start:
                    end = split_pos
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end + 1 if end < text_len and text[end] == " " else end

        return chunks or [text]

    @classmethod
    def _extract_regex_entities(cls, text: str) -> dict[str, int]:
        entities: dict[str, int] = {}
        cpf_count = len(cls._CPF_PATTERN.findall(text))
        if cpf_count:
            entities["CPF"] = cpf_count

        cnpj_count = len(cls._CNPJ_PATTERN.findall(text))
        if cnpj_count:
            entities["CNPJ"] = cnpj_count

        email_count = len(cls._EMAIL_PATTERN.findall(text))
        if email_count:
            entities["EMAIL"] = email_count

        phone_count = len(cls._PHONE_PATTERN.findall(text))
        if phone_count:
            entities["PHONE"] = phone_count

        return entities

    @classmethod
    def _is_exact_regex_match(cls, text: str) -> bool:
        clean = text.strip()
        return bool(
            cls._CPF_PATTERN.fullmatch(clean)
            or cls._CNPJ_PATTERN.fullmatch(clean)
            or cls._EMAIL_PATTERN.fullmatch(clean)
            or cls._PHONE_PATTERN.fullmatch(clean)
        )

    @classmethod
    def _check_regex(cls, samples: Sequence[str]) -> int:
        hits = 0
        for val in samples:
            if cls._is_exact_regex_match(val) or cls._extract_regex_entities(val):
                hits += 1
        return hits


class PIIColumnDetector(BaseModel[PiiDetectionInput, PiiDetectionResult]):
    def __init__(
        self,
        detector: PIIDetector | None = None,
        config: PIIDetectorConfig | None = None,
        device: torch.device | None = None,
        batch_size: int = 32,
    ) -> None:
        self.detector = detector or PIIDetector(
            config=config,
            device=device,
            batch_size=batch_size,
        )

    @property
    def is_loaded(self) -> bool:
        return self.detector.is_loaded

    def load(self) -> None:
        self.detector.load()

    def unload(self) -> None:
        self.detector.unload()

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

            regex_hits = self.detector._check_regex(item.samples)
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

        sample_results = self.detector.predict_batch(flattened_samples)

        col_pii_counts: dict[int, int] = {idx: 0 for idx in dl_indices}
        col_total_counts: dict[int, int] = {idx: 0 for idx in dl_indices}
        col_entities: dict[int, dict[str, int]] = {idx: {} for idx in dl_indices}

        for sample_res, col_idx in zip(sample_results, sample_to_col_idx):
            col_total_counts[col_idx] += 1
            if sample_res.is_pii:
                col_pii_counts[col_idx] += 1
                entities_dict = col_entities[col_idx]
                for ent_name, count in sample_res.detected_entities.items():
                    if ent_name in self.detector._PII_ENTITIES:
                        entities_dict[ent_name] = entities_dict.get(ent_name, 0) + count

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

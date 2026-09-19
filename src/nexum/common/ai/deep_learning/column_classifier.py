import re
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer, util

from nexum.common.ai.deep_learning.base_model import StatefulModel
from nexum.document.models import (
    ColumnSemanticResult,
    TableSemanticConfig,
    TableSemanticInput,
)


class TableSemanticClassifier(
    StatefulModel[TableSemanticInput, dict[str, ColumnSemanticResult]]
):
    _TAX_ID_PATTERN = re.compile(
        r"^\d{11}$|^\d{14}$|^\d{3}\.\d{3}\.\d{3}-\d{2}$|^\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}$"
    )
    _FLOAT_PATTERN = re.compile(
        r"^-?[R$€£\s]*\d+([.,]\d{1,4})[R$€£\s]*$"
    )
    _MONEY_TERMS = ("valor", "preco", "price", "compra", "custo")

    def __init__(self, config: TableSemanticConfig | None = None) -> None:
        self.config = config or TableSemanticConfig()
        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder: SentenceTransformer | None = None
        self._labels: list[str] = []
        self._profile_embeddings: torch.Tensor | None = None

    @property
    def is_loaded(self) -> bool:
        return (
            self.encoder is not None and self._profile_embeddings is not None
        )

    def load(self) -> None:
        if not self.is_loaded:
            self.encoder = SentenceTransformer(
                self.config.model_name, device=self.device
            )
            self._labels = list(self.config.profiles.keys())
            descriptions = [
                " | ".join(phrases)
                for phrases in self.config.profiles.values()
            ]
            self._profile_embeddings = self.encoder.encode(
                descriptions, convert_to_tensor=True
            )

    def unload(self) -> None:
        self.encoder = None
        self._labels = []
        self._profile_embeddings = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(
        self, input_data: TableSemanticInput
    ) -> dict[str, ColumnSemanticResult]:
        if not input_data.columns or not input_data.data:
            return {}

        if not self.is_loaded:
            self.load()

        assert self.encoder is not None
        assert self._profile_embeddings is not None

        samples_by_col = self._extract_column_samples(input_data)
        if not samples_by_col:
            return {}

        similarity_matrix = self._compute_similarities(samples_by_col)
        return self._resolve_classifications(samples_by_col, similarity_matrix)

    @staticmethod
    def _extract_column_samples(
        input_data: TableSemanticInput,
    ) -> dict[str, list[str]]:
        sample_rows = input_data.data[: input_data.sample_size]
        samples_by_col: dict[str, list[str]] = {}

        for col_idx, col_name in enumerate(input_data.columns):
            values: list[str] = []
            for row in sample_rows:
                if col_idx < len(row) and row[col_idx] is not None:
                    cell = str(row[col_idx]).strip()
                    if cell:
                        values.append(cell)
                if len(values) >= input_data.max_values_per_col:
                    break
            samples_by_col[col_name] = values

        return samples_by_col

    @staticmethod
    def _build_column_query(col_name: str, samples: list[str]) -> str:
        clean_col = col_name.replace("_", " ").replace("-", " ")
        sample_str = ", ".join(samples) if samples else "empty"
        return f"column: {clean_col} {clean_col} | sample values: {sample_str}"

    def _compute_similarities(
        self, samples_by_col: dict[str, list[str]]
    ) -> torch.Tensor:
        assert self.encoder is not None
        assert self._profile_embeddings is not None

        queries = [
            self._build_column_query(col, samples)
            for col, samples in samples_by_col.items()
        ]
        query_embeddings = self.encoder.encode(queries, convert_to_tensor=True)
        return util.cos_sim(query_embeddings, self._profile_embeddings)

    def _adjust_column_scores(
        self,
        scores: torch.Tensor,
        col_name: str,
        samples: list[str],
    ) -> torch.Tensor:
        adjusted = scores.clone()
        tax_id_idx = self._get_label_index("tax_id")
        identifier_idx = self._get_label_index("identifier")
        monetary_idx = self._get_label_index("monetary_value")

        clean_col = col_name.lower().replace("_", " ").replace("-", " ")

        if tax_id_idx != -1 and not self._is_tax_id_candidate(samples):
            adjusted[tax_id_idx] = -1.0

        if self._is_numeric_float_or_currency(samples):
            if identifier_idx != -1:
                adjusted[identifier_idx] = -1.0
            if monetary_idx != -1:
                adjusted[monetary_idx] += 0.35

        if any(term in clean_col for term in self._MONEY_TERMS):
            if monetary_idx != -1:
                adjusted[monetary_idx] += 0.40
            if identifier_idx != -1:
                adjusted[identifier_idx] = -1.0

        return adjusted

    def _resolve_classifications(
        self,
        samples_by_col: dict[str, list[str]],
        similarity_matrix: torch.Tensor,
    ) -> dict[str, ColumnSemanticResult]:
        results: dict[str, ColumnSemanticResult] = {}
        for idx, (col_name, samples) in enumerate(samples_by_col.items()):
            scores = self._adjust_column_scores(
                similarity_matrix[idx], col_name, samples
            )
            best_idx = int(torch.argmax(scores))
            semantic_type = self._labels[best_idx]
            is_pii = semantic_type in self.config.pii_types
            is_sensitive = semantic_type in self.config.sensitive_types

            results[col_name] = ColumnSemanticResult(
                semantic_type=semantic_type,
                is_pii=is_pii,
                is_sensitive=is_sensitive,
            )
        return results

    def _get_label_index(self, label: str) -> int:
        return self._labels.index(label) if label in self._labels else -1

    @classmethod
    def _is_tax_id_candidate(cls, values: Sequence[str]) -> bool:
        if not values:
            return False
        numeric_count = sum(
            1
            for v in values
            if cls._TAX_ID_PATTERN.match(v.replace(" ", ""))
        )
        return (numeric_count / len(values)) >= 0.5

    @classmethod
    def _is_numeric_float_or_currency(cls, values: Sequence[str]) -> bool:
        if not values:
            return False
        matches = sum(
            1 for v in values if cls._FLOAT_PATTERN.match(v.strip())
        )
        return (matches / len(values)) >= 0.3

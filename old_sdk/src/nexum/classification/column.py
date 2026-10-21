import torch
import torch.nn.functional as functional
from sentence_transformers import SentenceTransformer

from nexum.common.ai.base_model import StatefulModel
from nexum.common.helpers.device import resolve
from nexum.common.models import PiiDetectionInput, ColumnSemanticInput, ColumnSemanticResult, ColumnSemanticConfig
from nexum.detection.pii import PIIDetectorDL


class ColumnSemanticClassifierDL(
    StatefulModel[ColumnSemanticInput, dict[str, ColumnSemanticResult]]
):
    def __init__(
            self,
            config: ColumnSemanticConfig | None = None,
            pii_detector: PIIDetectorDL | None = None,
    ) -> None:
        self.config = config or ColumnSemanticConfig()
        self.device = (
            torch.device(self.config.device)
            if self.config.device
            else resolve()
        )
        self.encoder: SentenceTransformer | None = None
        self._labels: list[str] = []
        self._profile_embeddings: torch.Tensor | None = None
        self.pii_detector = pii_detector or PIIDetectorDL(device=self.device)

    @property
    def is_loaded(self) -> bool:
        return (
                self.encoder is not None
                and self._profile_embeddings is not None
                and self.pii_detector.is_loaded
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
            self.pii_detector.load()

    def unload(self) -> None:
        self.encoder = None
        self._labels = []
        self._profile_embeddings = None
        self.pii_detector.unload()

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def predict(self, input_data: ColumnSemanticInput) -> dict[str, ColumnSemanticResult]:
        return self.predict_batch([input_data])[0]

    def predict_batch(
            self, inputs: list[ColumnSemanticInput]
    ) -> list[dict[str, ColumnSemanticResult]]:
        if not inputs:
            return []

        if not self.is_loaded:
            self.load()

        results = []
        for item in inputs:
            samples_by_col = self._extract_samples(item)

            column_embeddings = []
            for col_name, values in samples_by_col.items():
                if values:
                    emb_values = self.encoder.encode(values, convert_to_tensor=True)
                    emb_values_mean = torch.mean(emb_values, dim=0)
                else:
                    emb_values_mean = torch.zeros(self._profile_embeddings.shape[1])

                emb_colname = self.encoder.encode([col_name], convert_to_tensor=True)[0]
                emb = torch.mean(torch.vstack([emb_values_mean, emb_colname]), dim=0)
                column_embeddings.append(emb)

            embeddings = torch.stack(column_embeddings)

            similarity_matrix = functional.cosine_similarity(
                embeddings.unsqueeze(1),
                self._profile_embeddings.unsqueeze(0),
                dim=2
            )

            resolved = self._resolve_classifications(samples_by_col, similarity_matrix)
            results.append(resolved)

        return results

    def _extract_samples(self, input_data: ColumnSemanticInput) -> dict[str, list[str]]:
        samples_by_col: dict[str, list[str]] = {}

        num_cols = len(input_data.columns)
        num_rows = len(input_data.data)

        for col_idx, col_name in enumerate(input_data.columns):
            col_values = []
            for row_idx in range(min(num_rows, input_data.sample_size)):
                value = input_data.data[row_idx][col_idx]
                if value is not None:
                    col_values.append(str(value))
            samples_by_col[col_name] = col_values[:input_data.max_values_per_col]

        return samples_by_col

    def _resolve_classifications(
            self,
            samples_by_col: dict[str, list[str]],
            similarity_matrix: torch.Tensor,
    ) -> dict[str, ColumnSemanticResult]:
        col_names = list(samples_by_col.keys())
        pii_inputs = [
            PiiDetectionInput(column_name=col, samples=samples_by_col[col])
            for col in col_names
        ]

        pii_results = self.pii_detector.predict_batch(pii_inputs)
        pii_by_col = dict(zip(col_names, pii_results))

        results: dict[str, ColumnSemanticResult] = {}
        for idx, col_name in enumerate(col_names):
            scores = similarity_matrix[idx]
            best_idx = int(torch.argmax(scores))
            semantic_type = self._labels[best_idx]

            rule_based_pii = semantic_type in self.config.pii_types
            pii_check = pii_by_col[col_name]

            is_pii = rule_based_pii or pii_check.is_pii
            is_sensitive = semantic_type in self.config.sensitive_types

            results[col_name] = ColumnSemanticResult(
                semantic_type=semantic_type,
                is_pii=is_pii,
                is_sensitive=is_sensitive,
            )

        return results

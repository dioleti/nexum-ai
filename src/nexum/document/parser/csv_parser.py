import csv as engine
import logging
from pprint import pprint
from typing import Any

from nexum.common.ai.deep_learning.column_classifier import TableSemanticClassifier
from nexum.common.helpers.table import detect_table_boundaries
from nexum.common.helpers.type import infer
from nexum.document.models import (
    ColumnSemanticResult,
    CSVReaderConfig,
    Parsed,
    Table,
    TableSemanticInput,
)
from nexum.document.parser.base import BaseParser

logger = logging.getLogger(__name__)


class CSVParser(BaseParser):
    def __init__(
        self,
        config: CSVReaderConfig | None = None,
        classifier: TableSemanticClassifier | None = None,
    ):
        resolved_config = config or CSVReaderConfig()
        super().__init__(resolved_config)
        self.config: CSVReaderConfig = resolved_config
        self.classifier: TableSemanticClassifier = (
            classifier or TableSemanticClassifier()
        )

    def _apply_type_inference(self, rows: list[list[str]]) -> list[list[Any]]:
        if not self.config.infer_types:
            return rows
        return [[infer(cell) for cell in row] for row in rows]

    def _parse_rows(self, text: str) -> list[list[str]]:
        lines = text.splitlines()
        skip_rows = self.config.skip_rows

        if skip_rows == 0:
            detected_skip, detected_header = detect_table_boundaries(
                lines[:20],
                self.config.delimiter,
            )
            skip_rows = detected_skip
            if not self.config.has_header:
                self.config.has_header = detected_header

        reader = engine.reader(lines, delimiter=self.config.delimiter)
        rows = [row for row in reader if row]

        if skip_rows > 0:
            rows = rows[skip_rows:]
        return rows

    def _build_table(
        self, rows: list[list[str]]
    ) -> tuple[Table, dict[str, ColumnSemanticResult]]:
        if not rows:
            return Table(columns=[], rows=[], bbox=None, page=1), {}

        if self.config.has_header:
            columns = rows[0]
            data = rows[1:]
        else:
            num_cols = len(rows[0])
            columns = [f"col_{i + 1}" for i in range(num_cols)]
            data = rows

        data = self._apply_type_inference(data)

        column_semantics = self.classifier.predict(
            TableSemanticInput(columns=columns, data=data)
        )

        table = Table(columns=columns, rows=data, bbox=None, page=1)
        return table, column_semantics

    def _build_metadata(
        self,
        table: Table,
        column_semantics: dict[str, ColumnSemanticResult],
        encoding: str | None = "utf-8",
    ) -> dict[str, Any]:
        pii_columns = [
            col for col, res in column_semantics.items() if res.is_pii
        ]
        sensitive_columns = [
            col for col, res in column_semantics.items() if res.is_sensitive
        ]

        return {
            "encoding": encoding,
            "delimiter": self.config.delimiter,
            "skip_rows": self.config.skip_rows,
            "has_header": self.config.has_header,
            "infer_types": self.config.infer_types,
            "num_rows": len(table.rows),
            "num_columns": len(table.columns),
            "header": table.columns,
            "column_types": {
                col: res.semantic_type for col, res in column_semantics.items()
            },
            "has_pii": bool(pii_columns),
            "pii_columns": pii_columns,
            "pii_details": {
                col: res.semantic_type
                for col, res in column_semantics.items()
                if res.is_pii
            },
            "has_sensitive_pii": bool(sensitive_columns),
            "sensitive_columns": sensitive_columns,
            "sensitive_details": {
                col: res.semantic_type
                for col, res in column_semantics.items()
                if res.is_sensitive
            },
        }

    def parse(self, text: str) -> Parsed:
        rows = self._parse_rows(text)
        table, column_semantics = self._build_table(rows)
        metadata = self._build_metadata(
            table,
            column_semantics,
            self.config.encoding,
        )

        parsed = Parsed(metadata=metadata, tables=[table], rows=rows)
        return parsed


if __name__ == "__main__":
    config = CSVReaderConfig()
    parser = CSVParser(config)
    with open("D:\\docs\\data.csv", "r", encoding="utf-8") as file:
        rows = parser.parse(file.read())
        pprint(rows)

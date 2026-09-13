import csv
import logging
from typing import Any

from nexum.common.helpers.type import infer
from nexum.document.models import CSVReaderConfig, Table
from nexum.document.parser.base import BaseParser

logger = logging.getLogger(__name__)


class CSVParser(BaseParser):
    def __init__(self, config: CSVReaderConfig | None = None):
        resolved_config = config or CSVReaderConfig()
        super().__init__(resolved_config)
        self.config: CSVReaderConfig = resolved_config

    def _apply_type_inference(self, rows: list[list[str]]) -> list[list[Any]]:
        if not self.config.infer_types:
            return rows
        return [[infer(cell) for cell in row] for row in rows]

    def _parse_rows(self, text: str) -> list[list[str]]:
        reader = csv.reader(text.splitlines(), delimiter=self.config.delimiter)
        rows = [row for row in reader if row]
        if self.config.skip_rows > 0:
            rows = rows[self.config.skip_rows :]
        return rows

    def _build_table(self, rows: list[list[str]]) -> Table:
        if self.config.has_header:
            columns = rows[0]
            data = rows[1:]
        else:
            num_cols = len(rows[0])
            columns = [f"col_{i + 1}" for i in range(num_cols)]
            data = rows
        data = self._apply_type_inference(data)
        return Table(columns=columns, rows=data, bbox=None, page=1)

    def _build_metadata(self, encoding: str, table: Table) -> dict:
        return {
            "encoding": encoding,
            "delimiter": self.config.delimiter,
            "skip_rows": self.config.skip_rows,
            "has_header": self.config.has_header,
            "infer_types": self.config.infer_types,
            "num_rows": len(table.rows),
            "num_columns": len(table.columns),
            "header": table.columns,
        }

    def parse(self, text: str) -> dict:
        rows = self._parse_rows(text)
        table = self._build_table(rows)
        metadata = self._build_metadata(text, table)
        return {
            "rows": rows,
            "table": table,
            "metadata": metadata,
        }

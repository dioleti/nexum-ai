from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CodeDocument:
    code: str
    language: str
    file_path: str = ""


@dataclass(slots=True)
class CodeChunk:
    name: str
    chunk_type: str
    content: str
    start_line: int
    end_line: int
    parent_scope: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

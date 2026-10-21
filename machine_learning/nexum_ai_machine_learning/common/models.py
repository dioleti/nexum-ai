import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseModel[InputT, OutputT](ABC):
    @abstractmethod
    def predict(self, input_data: InputT) -> OutputT: ...

    def predict_batch(self, inputs: Sequence[InputT]) -> list[OutputT]:
        return [self.predict(item) for item in inputs]

    async def apredict(self, input_data: InputT) -> OutputT:
        return await asyncio.to_thread(self.predict, input_data)

    async def apredict_batch(
        self, inputs: Sequence[InputT]
    ) -> list[OutputT]:
        return await asyncio.to_thread(self.predict_batch, inputs)


class StatefulModel(BaseModel[InputT, OutputT], ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...


@dataclass(frozen=True)
class PiiDetectionInput:
    column_name: str
    samples: list[str]
    threshold: float = 0.25


PiiColumnDetectionInput = PiiDetectionInput


@dataclass(frozen=True)
class PiiDetectionResult:
    is_pii: bool
    confidence: float
    detected_entities: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PIIDetectorConfig:
    model_name: str = "Dioleti/xlm-roberta-base-ner-hrl"
    score_threshold: float = 0.70
    max_chunk_chars: int = 1500

import asyncio

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseModel(ABC, Generic[InputT, OutputT]):
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

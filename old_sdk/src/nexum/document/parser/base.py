from abc import ABC, abstractmethod
from typing import Any

from nexum.document.models import NexumConfig


class BaseParser(ABC):
    def __init__(self, config: NexumConfig | None = None):
        self.config = config

    @abstractmethod
    def parse(self, data: Any) -> Any: ...

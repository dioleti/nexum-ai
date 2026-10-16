import asyncio
import unittest
from typing import Sequence

from nexum.common.ai.base_model import BaseModel, StatefulModel


class DummyBaseModel(BaseModel[int, int]):
    def predict(self, input_data: int) -> int:
        return input_data * 2


class DummyStatefulModel(StatefulModel[str, str]):
    def __init__(self):
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, input_data: str) -> str:
        return input_data.upper()


class TestBaseModel(unittest.TestCase):
    def test_cannot_instantiate_abstract_base_model(self):
        with self.assertRaises(TypeError):
            BaseModel()

    def test_cannot_instantiate_abstract_stateful_model(self):
        with self.assertRaises(TypeError):
            StatefulModel()

    def test_predict_and_predict_batch(self):
        model = DummyBaseModel()
        self.assertEqual(model.predict(5), 10)
        self.assertEqual(model.predict_batch([1, 2, 3]), [2, 4, 6])

    def test_async_predict_and_batch(self):
        model = DummyBaseModel()
        res_single = asyncio.run(model.apredict(4))
        res_batch = asyncio.run(model.apredict_batch([2, 3]))
        self.assertEqual(res_single, 8)
        self.assertEqual(res_batch, [4, 6])

    def test_stateful_model_lifecycle(self):
        model = DummyStatefulModel()
        self.assertFalse(model.is_loaded)
        model.load()
        self.assertTrue(model.is_loaded)
        self.assertEqual(model.predict("abc"), "ABC")
        model.unload()
        self.assertFalse(model.is_loaded)


if __name__ == "__main__":
    unittest.main()

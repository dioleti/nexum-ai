import asyncio
import unittest
import tests

from nexum_ai_machine_learning.common.models import (
    BaseModel,
    PiiDetectionInput,
    PiiDetectionResult,
    PIIDetectorConfig,
    StatefulModel,
)


class DummyModel(BaseModel[int, int]):
    def predict(self, input_data: int) -> int:
        return input_data * 2


class DummyStatefulModel(StatefulModel[str, str]):
    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def predict(self, input_data: str) -> str:
        return input_data.upper()


class TestModels(unittest.TestCase):
    def test_base_model_sync_and_async(self):
        model = DummyModel()
        self.assertEqual(model.predict(5), 10)
        self.assertEqual(model.predict_batch([1, 2, 3]), [2, 4, 6])

        res_single = asyncio.run(model.apredict(7))
        self.assertEqual(res_single, 14)

        res_batch = asyncio.run(model.apredict_batch([3, 4]))
        self.assertEqual(res_batch, [6, 8])

    def test_stateful_model_lifecycle(self):
        model = DummyStatefulModel()
        self.assertFalse(model.is_loaded)
        model.load()
        self.assertTrue(model.is_loaded)
        self.assertEqual(model.predict("abc"), "ABC")
        model.unload()
        self.assertFalse(model.is_loaded)

    def test_pii_detection_input_defaults_and_custom(self):
        inp_default = PiiDetectionInput(column_name="cpf", samples=["123"])
        self.assertEqual(inp_default.column_name, "cpf")
        self.assertEqual(inp_default.samples, ["123"])
        self.assertEqual(inp_default.threshold, 0.25)

        inp_custom = PiiDetectionInput(column_name="email", samples=["a@b.com"], threshold=0.5)
        self.assertEqual(inp_custom.threshold, 0.5)

    def test_pii_detection_result_defaults_and_custom(self):
        res_default = PiiDetectionResult(is_pii=False, confidence=0.0)
        self.assertFalse(res_default.is_pii)
        self.assertEqual(res_default.confidence, 0.0)
        self.assertEqual(res_default.detected_entities, {})

        res_custom = PiiDetectionResult(
            is_pii=True,
            confidence=0.85,
            detected_entities={"PER": 3},
        )
        self.assertTrue(res_custom.is_pii)
        self.assertEqual(res_custom.confidence, 0.85)
        self.assertEqual(res_custom.detected_entities, {"PER": 3})

    def test_pii_detector_config(self):
        cfg_default = PIIDetectorConfig()
        self.assertEqual(cfg_default.model_name, "Dioleti/xlm-roberta-base-ner-hrl")

        cfg_custom = PIIDetectorConfig(model_name="custom/model")
        self.assertEqual(cfg_custom.model_name, "custom/model")


if __name__ == "__main__":
    unittest.main()

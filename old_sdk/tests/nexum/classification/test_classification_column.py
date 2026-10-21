import unittest
from unittest.mock import MagicMock, patch

import torch

from nexum.classification.column import ColumnSemanticClassifierDL
from nexum.common.models import (
    ColumnSemanticConfig,
    ColumnSemanticInput,
    PiiDetectionResult,
)


class TestColumnSemanticClassifierDL(unittest.TestCase):
    @patch("nexum.classification.column.SentenceTransformer")
    def test_load_and_unload(self, mock_st_cls):
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = torch.zeros((2, 4))
        mock_st_cls.return_value = mock_encoder

        mock_pii = MagicMock()
        mock_pii.is_loaded = False

        config = ColumnSemanticConfig(
            device="cpu",
            profiles={"identifier": ("id",), "email": ("email",)},
        )
        classifier = ColumnSemanticClassifierDL(config=config, pii_detector=mock_pii)
        self.assertFalse(classifier.is_loaded)

        def mark_pii_loaded():
            mock_pii.is_loaded = True

        mock_pii.load.side_effect = mark_pii_loaded
        classifier.load()

        self.assertTrue(classifier.is_loaded)
        mock_st_cls.assert_called_once_with(config.model_name, device=torch.device("cpu"))
        mock_pii.load.assert_called_once()

        def mark_pii_unloaded():
            mock_pii.is_loaded = False

        mock_pii.unload.side_effect = mark_pii_unloaded
        classifier.unload()
        self.assertFalse(classifier.is_loaded)
        mock_pii.unload.assert_called_once()

    @patch("nexum.classification.column.SentenceTransformer")
    def test_predict_and_predict_batch(self, mock_st_cls):
        mock_encoder = MagicMock()
        # 3 profiles: identifier, email, religious_belief
        profile_embs = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        def fake_encode(texts, convert_to_tensor=True):
            if len(texts) == 3 and "id" in texts[0]:
                return profile_embs
            # For column values / names
            joined = " ".join(texts).lower()
            if "id" in joined or "1" in joined:
                return torch.tensor([[1.0, 0.0, 0.0] for _ in texts])
            if "email" in joined or "@" in joined:
                return torch.tensor([[0.0, 1.0, 0.0] for _ in texts])
            return torch.tensor([[0.0, 0.0, 1.0] for _ in texts])

        mock_encoder.encode.side_effect = fake_encode
        mock_st_cls.return_value = mock_encoder

        mock_pii = MagicMock()
        mock_pii.is_loaded = True
        mock_pii.predict_batch.return_value = [
            PiiDetectionResult(is_pii=False, confidence=0.0),
            PiiDetectionResult(is_pii=True, confidence=1.0),
            PiiDetectionResult(is_pii=False, confidence=0.0),
        ]

        config = ColumnSemanticConfig(
            device="cpu",
            profiles={
                "identifier": ("id",),
                "email": ("email",),
                "religious_belief": ("religion",),
            },
        )
        classifier = ColumnSemanticClassifierDL(config=config, pii_detector=mock_pii)

        inp = ColumnSemanticInput(
            columns=["id", "email", "religiao"],
            data=[
                [1, "a@b.com", None],
                [2, "c@d.com", None],
            ],
        )
        res = classifier.predict(inp)

        self.assertEqual(res["id"].semantic_type, "identifier")
        self.assertFalse(res["id"].is_pii)
        self.assertFalse(res["id"].is_sensitive)

        self.assertEqual(res["email"].semantic_type, "email")
        self.assertTrue(res["email"].is_pii)
        self.assertFalse(res["email"].is_sensitive)

        self.assertEqual(res["religiao"].semantic_type, "religious_belief")
        self.assertTrue(res["religiao"].is_pii)
        self.assertTrue(res["religiao"].is_sensitive)

    def test_predict_batch_empty(self):
        classifier = ColumnSemanticClassifierDL(
            config=ColumnSemanticConfig(device="cpu"),
            pii_detector=MagicMock(),
        )
        self.assertEqual(classifier.predict_batch([]), [])


if __name__ == "__main__":
    unittest.main()

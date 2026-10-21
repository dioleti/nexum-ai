import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from nexum.document.models import CaptionConfig, CaptionInput
from nexum.vision.captiongeneratordl import CaptionGeneratorDL


class TestCaptionGeneratorDL(unittest.TestCase):
    @patch("nexum.vision.captiongeneratordl.BlipForConditionalGeneration.from_pretrained")
    @patch("nexum.vision.captiongeneratordl.BlipProcessor.from_pretrained")
    def test_load_and_unload(self, mock_proc_cls, mock_model_cls):
        mock_proc_cls.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.return_value = mock_model

        config = CaptionConfig(device="cpu")
        generator = CaptionGeneratorDL(config)
        self.assertFalse(generator.is_loaded)

        generator.load()
        self.assertTrue(generator.is_loaded)
        mock_proc_cls.assert_called_once_with(config.model_name)
        mock_model_cls.assert_called_once_with(config.model_name)
        mock_model.eval.assert_called_once()

        generator.unload()
        self.assertFalse(generator.is_loaded)

    def test_predict_without_and_with_prompt(self):
        config = CaptionConfig(device="cpu", max_new_tokens=20)
        generator = CaptionGeneratorDL(config)

        mock_proc = MagicMock()
        mock_batch = MagicMock()
        mock_batch.to.return_value = {}
        mock_proc.return_value = mock_batch
        mock_proc.decode.return_value = "  A cat sitting on a desk  "

        mock_model = MagicMock()
        mock_model.generate.return_value = ["fake_token_ids"]

        generator.processor = mock_proc
        generator.model = mock_model

        img = Image.new("RGB", (100, 100), color="white")
        caption = generator.predict(CaptionInput(image=img))
        self.assertEqual(caption, "A cat sitting on a desk")

        caption_prompted = generator.predict(CaptionInput(image=img, prompt="a photo of"))
        self.assertEqual(caption_prompted, "A cat sitting on a desk")

    def test_predict_batch_empty(self):
        generator = CaptionGeneratorDL(CaptionConfig(device="cpu"))
        self.assertEqual(generator.predict_batch([]), [])


if __name__ == "__main__":
    unittest.main()

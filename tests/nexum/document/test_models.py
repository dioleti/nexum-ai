import unittest

from nexum.document.models import Document, CSVReaderConfig, ImageOCRConfig, ImageReaderConfig, ReaderMode, \
    PDFOCRConfig, PDFReaderConfig, Table


class TestModels(unittest.TestCase):
    def test_reader_mode_values(self):
        self.assertEqual(ReaderMode.BATCH.value, "batch")
        self.assertEqual(ReaderMode.STREAMING.value, "stream")

    def test_table_model(self):
        table = Table(columns=["a", "b"], rows=[["1", "2"]], bbox=(1, 2, 3, 4), page=1)
        self.assertEqual(table.columns, ["a", "b"])
        self.assertEqual(table.rows, [["1", "2"]])
        self.assertEqual(table.bbox, (1, 2, 3, 4))
        self.assertEqual(table.page, 1)

    def test_document_model(self):
        doc = Document(
            content="hello",
            metadata={"k": "v"},
            images=[b"img"],
            tables=[],
            source="test",
            pages=["hello"],
            raw_bytes=b"raw",
        )
        self.assertEqual(doc.content, "hello")
        self.assertEqual(doc.metadata, {"k": "v"})
        self.assertEqual(doc.images, [b"img"])
        self.assertEqual(doc.tables, [])
        self.assertEqual(doc.source, "test")
        self.assertEqual(doc.pages, ["hello"])
        self.assertEqual(doc.raw_bytes, b"raw")

    def test_document_to_langchain(self):
        doc = Document(
            content="abc",
            metadata={"x": 1},
            images=None,
            tables=None,
            source="src",
            pages=["abc"],
            raw_bytes=b"123",
        )
        lcd = doc.to_langchain_document()
        self.assertEqual(lcd.page_content, "abc")
        self.assertEqual(lcd.metadata["metadata"], {"x": 1})
        self.assertEqual(lcd.metadata["source"], "src")
        self.assertEqual(lcd.metadata["pages"], ["abc"])
        self.assertEqual(lcd.metadata["raw_bytes"], b"123")

    def test_csv_reader_config_defaults(self):
        cfg = CSVReaderConfig()
        self.assertEqual(cfg.delimiter, ",")
        self.assertEqual(cfg.skip_rows, 0)
        self.assertTrue(cfg.has_header)
        self.assertIsNone(cfg.encoding)
        self.assertTrue(cfg.infer_types)

    def test_image_ocr_config_defaults(self):
        cfg = ImageOCRConfig()
        self.assertEqual(cfg.lang, "eng")
        self.assertTrue(cfg.enable_sharpen)
        self.assertTrue(cfg.enable_metadata)
        self.assertEqual(cfg.sharpen_strength, 1.2)
        self.assertTrue(cfg.enable_binarization)
        self.assertEqual(cfg.binarization_min_std, 10.0)
        self.assertTrue(cfg.enable_denoise)
        self.assertEqual(cfg.denoise_method, "bilateral")
        self.assertTrue(cfg.strip_empty_lines)

    def test_image_reader_config_defaults(self):
        cfg = ImageReaderConfig()
        self.assertEqual(cfg.read_mode, ReaderMode.BATCH)
        self.assertIsInstance(cfg.ocr_config, ImageOCRConfig)

    def test_pdf_ocr_config_defaults(self):
        cfg = PDFOCRConfig()
        self.assertEqual(cfg.min_dpi, 200)
        self.assertEqual(cfg.lang, "eng")
        self.assertEqual(cfg.upscale_factor, 1.5)
        self.assertTrue(cfg.enable_binarization)
        self.assertEqual(cfg.binarization_method, "otsu")
        self.assertEqual(cfg.binarization_min_std, 10.0)
        self.assertTrue(cfg.enable_deskew)
        self.assertEqual(cfg.deskew_max_angle, 5.0)
        self.assertTrue(cfg.enable_sharpen)
        self.assertEqual(cfg.sharpen_strength, 1.2)
        self.assertTrue(cfg.enable_denoise)
        self.assertEqual(cfg.denoise_method, "bilateral")
        self.assertTrue(cfg.enable_osd)
        self.assertEqual(cfg.osd_min_confidence, 5.0)
        self.assertEqual(cfg.psm, 6)
        self.assertEqual(cfg.oem, 3)
        self.assertTrue(cfg.strip_empty_lines)

    def test_pdf_reader_config_defaults(self):
        cfg = PDFReaderConfig()
        self.assertEqual(cfg.read_mode, ReaderMode.STREAMING)
        self.assertIsInstance(cfg.ocr_config, PDFOCRConfig)
        self.assertEqual(cfg.lang, "eng")
        self.assertEqual(cfg.max_workers, 4)
        self.assertEqual(cfg.dpi, 300)
        self.assertTrue(cfg.enable_ocr)
        self.assertTrue(cfg.enable_metadata)
        self.assertTrue(cfg.enable_images)
        self.assertTrue(cfg.enable_tables)


if __name__ == "__main__":
    unittest.main()

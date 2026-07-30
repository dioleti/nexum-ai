import unittest

from nexum.document.models import CSVReaderConfig, Table
from nexum.document.parser.csv import CSVParser


class TestCSVParser(unittest.TestCase):
    def test_parse_rows_basic(self):
        parser = CSVParser(CSVReaderConfig())
        text = "a,b,c\n1,2,3\n4,5,6"
        rows = parser._parse_rows(text)
        self.assertEqual(rows, [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]])

    def test_parse_rows_skip(self):
        cfg = CSVReaderConfig(skip_rows=1)
        parser = CSVParser(cfg)
        text = "skip,this,row\n1,2,3\n4,5,6"
        rows = parser._parse_rows(text)
        self.assertEqual(rows, [["1", "2", "3"], ["4", "5", "6"]])

    def test_build_table_with_header(self):
        cfg = CSVReaderConfig(has_header=True, infer_types=False)
        parser = CSVParser(cfg)
        rows = [["a", "b"], ["1", "2"], ["3", "4"]]
        table = parser._build_table(rows)
        self.assertEqual(table.columns, ["a", "b"])
        self.assertEqual(table.rows, [["1", "2"], ["3", "4"]])

    def test_build_table_without_header(self):
        cfg = CSVReaderConfig(has_header=False, infer_types=False)
        parser = CSVParser(cfg)
        rows = [["1", "2"], ["3", "4"]]
        table = parser._build_table(rows)
        self.assertEqual(table.columns, ["col_1", "col_2"])
        self.assertEqual(table.rows, [["1", "2"], ["3", "4"]])

    def test_infer_type_int(self):
        parser = CSVParser(CSVReaderConfig(infer_types=True))
        self.assertEqual(parser._infer_type("42"), 42)

    def test_infer_type_float(self):
        parser = CSVParser(CSVReaderConfig(infer_types=True))
        self.assertEqual(parser._infer_type("3.14"), 3.14)

    def test_infer_type_bool(self):
        parser = CSVParser(CSVReaderConfig(infer_types=True))
        self.assertEqual(parser._infer_type("true"), True)
        self.assertEqual(parser._infer_type("false"), False)

    def test_infer_type_date(self):
        parser = CSVParser(CSVReaderConfig(infer_types=True))
        result = parser._infer_type("2020-01-01")
        self.assertEqual(str(result), "2020-01-01")

    def test_build_metadata(self):
        parser = CSVParser(CSVReaderConfig())
        table = Table(columns=["a"], rows=[["1"]], bbox=None, page=1)
        metadata = parser._build_metadata("utf-8", table)
        self.assertEqual(metadata["encoding"], "utf-8")
        self.assertEqual(metadata["num_rows"], 1)
        self.assertEqual(metadata["num_columns"], 1)

    def test_parse_full(self):
        cfg = CSVReaderConfig(infer_types=False)
        parser = CSVParser(cfg)
        text = "a,b\n1,2\n3,4"
        parsed = parser.parse(text)
        self.assertEqual(parsed["rows"], [["a", "b"], ["1", "2"], ["3", "4"]])
        self.assertIsInstance(parsed["table"], Table)
        self.assertEqual(parsed["metadata"]["header"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

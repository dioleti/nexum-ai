import unittest
from datetime import date

from nexum.document.models import CSVReaderConfig
from nexum.document.parser.csv_parser import CSVParser


class TestCSVParser(unittest.TestCase):

    def setUp(self):
        self.csv_text = (
            "id,nome,data_nascimento,valor_compra,cidade\n"
            "1,Ana Beatriz Silva,12/03/1991,152.75,São Paulo\n"
            "2,Carlos Henrique Souza,28/11/1984,89.90,Rio de Janeiro\n"
            "3,Mariana Oliveira,05/07/1999,230.00,Belo Horizonte\n"
            "4,João Pedro Almeida,17/01/1975,45.50,Curitiba\n"
            "5,Fernanda Costa,22/09/2002,310.20,Porto Alegre"
        )

        self.config = CSVReaderConfig(
            delimiter=",",
            skip_rows=0,
            has_header=True,
            infer_types=True,
            encoding="utf-8",
        )

        self.parser = CSVParser(self.config)

    def test_parse_rows(self):
        rows = self.parser._parse_rows(self.csv_text)

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0], ["id", "nome", "data_nascimento", "valor_compra", "cidade"])
        self.assertEqual(rows[1][0], "1")
        self.assertEqual(rows[1][3], "152.75")

    def test_apply_type_inference(self):
        rows = [
            ["1", "Ana", "12/03/1991", "152.75"],
            ["2", "Carlos", "28/11/1984", "89.90"],
        ]

        inferred = self.parser._apply_type_inference(rows)

        self.assertEqual(inferred[0][0], 1)
        self.assertEqual(inferred[0][2], date(1991, 3, 12))
        self.assertEqual(inferred[0][3], 152.75)

    def test_build_table(self):
        rows = self.parser._parse_rows(self.csv_text)
        table = self.parser._build_table(rows)

        self.assertEqual(table.columns, ["id", "nome", "data_nascimento", "valor_compra", "cidade"])
        self.assertEqual(len(table.rows), 5)

        first_row = table.rows[0]
        self.assertEqual(first_row[0], 1)
        self.assertEqual(first_row[2], date(1991, 3, 12))
        self.assertEqual(first_row[3], 152.75)

    def test_build_metadata(self):
        rows = self.parser._parse_rows(self.csv_text)
        table = self.parser._build_table(rows)
        metadata = self.parser._build_metadata(table, self.config.encoding)

        self.assertEqual(metadata["encoding"], "utf-8")
        self.assertEqual(metadata["delimiter"], ",")
        self.assertEqual(metadata["skip_rows"], 0)
        self.assertEqual(metadata["has_header"], True)
        self.assertEqual(metadata["infer_types"], True)
        self.assertEqual(metadata["num_rows"], 5)
        self.assertEqual(metadata["num_columns"], 5)
        self.assertEqual(metadata["header"], ["id", "nome", "data_nascimento", "valor_compra", "cidade"])

    def test_parse(self):
        result = self.parser.parse(self.csv_text)

        self.assertEqual(len(result["rows"]), 6)

        table = result["table"]
        self.assertEqual(table.rows[0][0], 1)
        self.assertEqual(table.rows[0][2], date(1991, 3, 12))

        metadata = result["metadata"]
        self.assertEqual(metadata["num_rows"], 5)
        self.assertEqual(metadata["encoding"], "utf-8")


if __name__ == "__main__":
    unittest.main()

import unittest

from nexum_ai_common.models import Region, RegionType


class TestModels(unittest.TestCase):
    def test_region_type_enum(self):
        self.assertEqual(RegionType.TABLE.value, "table")
        self.assertEqual(RegionType.TEXT.value, "text")
        self.assertEqual(RegionType.IMAGE.value, "image")
        self.assertEqual(RegionType.FLOWCHART.value, "flowchart")

    def test_region_creation_defaults(self):
        region = Region(type=RegionType.TEXT, box=(10, 20, 100, 200), score=0.95)
        self.assertEqual(region.type, RegionType.TEXT)
        self.assertEqual(region.box, (10, 20, 100, 200))
        self.assertEqual(region.score, 0.95)
        self.assertIsNone(region.text)

    def test_region_creation_with_text_and_mutation(self):
        region = Region(
            type=RegionType.TABLE,
            box=(0, 0, 50, 50),
            score=0.8,
            text="initial",
        )
        self.assertEqual(region.text, "initial")
        region.text = "updated"
        self.assertEqual(region.text, "updated")


if __name__ == "__main__":
    unittest.main()

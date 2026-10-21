import unittest

from nexum_ai_common import constants, errors, helpers, models, nexum_logging
from nexum_ai_common.constants import date
from nexum_ai_common.constants.date import formats
from nexum_ai_common.errors import NexumError, NexumRuntimeError, NexumValueError
from nexum_ai_common.helpers import device, image, table, type as type_helper
from nexum_ai_common.helpers.device import resolve
from nexum_ai_common.helpers.image import (
    binarize,
    clahe,
    denoise,
    deskew,
    detect_images_heuristic,
    extract_dpi,
    normalize_ocr_output,
    ocr_region,
    osd_rotation,
    run_ocr,
    run_ocr_data,
    sharpen,
    validate_max_size,
)
from nexum_ai_common.helpers.table import (
    detect_table_boundaries,
    extract_table_dataframe,
    ocr_table_cells,
)
from nexum_ai_common.helpers.type import infer
from nexum_ai_common.models import Region, RegionType
from nexum_ai_common.nexum_logging import (
    COLORS,
    RESET,
    NexumDatadogFormatter,
    NexumElasticFormatter,
    NexumLocalFormatter,
    logger,
    setup_logging,
)


class TestPublicAPI(unittest.TestCase):
    def test_module_and_subpackage_imports(self):
        self.assertIsNotNone(constants)
        self.assertIsNotNone(date)
        self.assertIsNotNone(errors)
        self.assertIsNotNone(helpers)
        self.assertIsNotNone(device)
        self.assertIsNotNone(image)
        self.assertIsNotNone(table)
        self.assertIsNotNone(type_helper)
        self.assertIsNotNone(models)
        self.assertIsNotNone(nexum_logging)

    def test_public_symbols_importable(self):
        symbols = [
            NexumError,
            NexumRuntimeError,
            NexumValueError,
            Region,
            RegionType,
            COLORS,
            RESET,
            NexumLocalFormatter,
            NexumDatadogFormatter,
            NexumElasticFormatter,
            logger,
            setup_logging,
            formats,
            resolve,
            extract_dpi,
            deskew,
            clahe,
            sharpen,
            denoise,
            binarize,
            osd_rotation,
            run_ocr,
            run_ocr_data,
            validate_max_size,
            ocr_region,
            normalize_ocr_output,
            detect_images_heuristic,
            extract_table_dataframe,
            detect_table_boundaries,
            ocr_table_cells,
            infer,
        ]
        for sym in symbols:
            self.assertIsNotNone(sym)


if __name__ == "__main__":
    unittest.main()

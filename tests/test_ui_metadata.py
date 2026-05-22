from __future__ import annotations

import unittest

from rail_curve_extractor.qt_metadata import COMMON_FIELDS, PATH_FIELD_HELP, ROI_FIELDS


class UiMetadataTest(unittest.TestCase):
    def test_all_numeric_fields_have_descriptions(self) -> None:
        for field in COMMON_FIELDS + ROI_FIELDS:
            self.assertTrue(field.label.strip())
            self.assertTrue(field.description.strip())
            self.assertTrue(field.default_hint.strip())
            self.assertTrue(field.recommended_range.strip())
            self.assertTrue(field.effect.strip())

    def test_all_path_inputs_have_help(self) -> None:
        self.assertEqual(set(PATH_FIELD_HELP), {"input_path", "output_path", "config_path"})
        for item in PATH_FIELD_HELP.values():
            self.assertEqual(set(item), {"title", "description", "default_hint", "recommended_range", "effect"})
            for text in item.values():
                self.assertTrue(text.strip())


if __name__ == "__main__":
    unittest.main()

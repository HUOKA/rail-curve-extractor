from __future__ import annotations

import unittest

import numpy as np

from rail_curve_extractor.qt_app import build_oriented_roi_from_xy_points, create_app, oriented_roi_corners


class QtSmokeTest(unittest.TestCase):
    def test_qt_window_can_be_created(self) -> None:
        app, window = create_app([])
        window.show()
        app.processEvents()
        window.close()
        app.processEvents()

    def test_window_uses_workflow_pages(self) -> None:
        app, window = create_app([])
        try:
            self.assertIsNotNone(window.page_stack)
            self.assertEqual(window.page_stack.count(), 5)
            self.assertEqual(len(window.nav_buttons), 5)
            self.assertTrue(window.nav_buttons[0].isChecked())

            window._set_page(2)

            self.assertEqual(window.page_stack.currentIndex(), 2)
            self.assertTrue(window.nav_buttons[2].isChecked())
            self.assertFalse(window.nav_buttons[0].isChecked())
        finally:
            window.close()
            app.processEvents()

    def test_four_point_oriented_roi_builds_long_axis(self) -> None:
        roi = build_oriented_roi_from_xy_points([(0.0, 0.0), (0.0, 2.0), (10.0, 2.0), (10.0, 0.0)])

        self.assertTrue(roi["enabled"])
        self.assertAlmostEqual(float(roi["s_max"]) - float(roi["s_min"]), 10.0)
        self.assertAlmostEqual(float(roi["t_max"]) - float(roi["t_min"]), 2.0)
        self.assertEqual(oriented_roi_corners(roi).shape, (5, 2))

    def test_window_collects_four_point_rois_into_config(self) -> None:
        app, window = create_app([])
        try:
            global_roi = build_oriented_roi_from_xy_points([(0.0, 0.0), (0.0, 3.0), (20.0, 3.0), (20.0, 0.0)])
            track_roi = build_oriented_roi_from_xy_points([(30.0, 0.0), (30.0, 3.0), (50.0, 3.0), (50.0, 0.0)])
            window.oriented_roi_configs["global"] = global_roi
            window.oriented_roi_configs["track_2"] = track_roi
            window.manual_track_count_spin.setValue(2)
            window.track_enabled_checks[2].setChecked(True)

            config = window._collect_config_from_form()

            self.assertEqual(config["oriented_roi"]["origin"], global_roi["origin"])
            self.assertEqual(len(config["tracks"]), 1)
            self.assertEqual(config["tracks"][0]["id"], 2)
            self.assertEqual(config["tracks"][0]["oriented_roi"]["origin"], track_roi["origin"])
            self.assertTrue(np.allclose(config["tracks"][0]["oriented_roi"]["axis_s"], track_roi["axis_s"]))
        finally:
            window.close()
            app.processEvents()

    def test_window_collects_auto_track_split_config(self) -> None:
        app, window = create_app([])
        try:
            auto_roi = build_oriented_roi_from_xy_points([(0.0, -5.0), (0.0, 5.0), (80.0, 5.0), (80.0, -5.0)])
            window.oriented_roi_configs["auto_tracks"] = auto_roi
            window.auto_track_split_check.setChecked(True)
            window.auto_track_count_spin.setValue(5)

            config = window._collect_config_from_form()

            self.assertEqual(config["tracks"], [])
            self.assertTrue(config["auto_track_split"]["enabled"])
            self.assertEqual(config["auto_track_split"]["count"], 5)
            self.assertEqual(config["auto_track_split"]["oriented_roi"]["origin"], auto_roi["origin"])
        finally:
            window.close()
            app.processEvents()

    def test_window_collects_dynamic_manual_track_count(self) -> None:
        app, window = create_app([])
        try:
            self.assertEqual(window.manual_track_count_spin.value(), 1)
            self.assertFalse(window.track_roi_boxes[1].isHidden())
            self.assertTrue(window.track_roi_boxes[4].isHidden())

            window.manual_track_count_spin.setValue(5)
            window.track_roi_inputs[(5, "x_min")].setText("10")
            window.track_roi_inputs[(5, "x_max")].setText("20")
            window.track_roi_inputs[(5, "y_min")].setText("30")
            window.track_roi_inputs[(5, "y_max")].setText("40")

            config = window._collect_config_from_form()

            self.assertFalse(window.track_roi_boxes[5].isHidden())
            self.assertEqual(len(config["tracks"]), 1)
            self.assertEqual(config["tracks"][0]["id"], 5)
            self.assertEqual(config["tracks"][0]["roi"]["x_min"], 10.0)
            self.assertEqual(config["tracks"][0]["roi"]["y_max"], 40.0)
        finally:
            window.close()
            app.processEvents()

    def test_window_collects_manual_anchor_config(self) -> None:
        app, window = create_app([])
        try:
            track_roi = build_oriented_roi_from_xy_points([(0.0, -2.0), (0.0, 2.0), (40.0, 2.0), (40.0, -2.0)])
            turnout_roi = build_oriented_roi_from_xy_points([(50.0, -4.0), (50.0, 4.0), (90.0, 4.0), (90.0, -4.0)])
            window.oriented_roi_configs["track_1"] = track_roi
            window.oriented_roi_configs["turnout"] = turnout_roi
            window.anchor_points["global"] = [(0.0, 0.0), (10.0, 0.1)]
            window.anchor_points["track_1"] = [(0.0, -0.2), (40.0, -0.1)]
            window.anchor_points["turnout_main"] = [(50.0, 0.0), (90.0, 0.1)]
            window.anchor_points["turnout_branch"] = [(60.0, -0.3), (88.0, -2.0)]
            window.track_enabled_checks[1].setChecked(True)
            window.turnout_enabled_check.setChecked(True)

            config = window._collect_config_from_form()

            self.assertTrue(config["manual_anchor"]["enabled"])
            self.assertEqual(config["manual_anchor"]["points"], [[0.0, 0.0], [10.0, 0.1]])
            self.assertEqual(config["tracks"][0]["manual_anchor"]["points"], [[0.0, -0.2], [40.0, -0.1]])
            self.assertEqual(config["turnout"]["main_anchor_points"], [[50.0, 0.0], [90.0, 0.1]])
            self.assertEqual(config["turnout"]["branch_anchor_points"], [[60.0, -0.3], [88.0, -2.0]])
        finally:
            window.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()

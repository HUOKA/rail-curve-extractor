from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rail_curve_extractor.rail_segmentation_baseline import (
    BaselineTrainOptions,
    train_rail_segmentation_baseline,
)


class RailSegmentationBaselineTest(unittest.TestCase):
    def test_train_baseline_writes_metrics_and_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            images = dataset / "images"
            labels = dataset / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (dataset / "classes.txt").write_text("track_area\nswitch_area\nignore_area\n", encoding="utf-8")
            for index in range(6):
                image = Image.new("RGB", (32, 24), (25, 30, 35))
                pixels = image.load()
                for y in range(4, 20):
                    for x in range(12, 21):
                        pixels[x, y] = (180, 170, 150)
                image.save(images / f"tile_{index:02d}.png")
                labels.joinpath(f"tile_{index:02d}.txt").write_text(
                    "0 0.375 0.166667 0.65625 0.166667 0.65625 0.833333 0.375 0.833333\n",
                    encoding="utf-8",
                )

            result = train_rail_segmentation_baseline(
                dataset,
                root / "out",
                BaselineTrainOptions(
                    val_every=3,
                    max_positive_per_image=200,
                    max_negative_per_image=200,
                    iterations=60,
                    min_component_pixels=0,
                    predict_all=False,
                ),
            )

            self.assertEqual(result.train_count, 4)
            self.assertEqual(result.val_count, 2)
            self.assertGreater(result.val_metrics["f1"], 0.90)
            self.assertTrue(Path(result.model_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            metrics = json.loads(Path(result.metrics_json_path).read_text(encoding="utf-8"))
            self.assertEqual(metrics["class_name"], "track_area")
            self.assertEqual(len(list(Path(result.val_overlay_dir).glob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()

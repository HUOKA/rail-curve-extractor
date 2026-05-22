from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None


def _load_yolo_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_rail_seg_yolo.py"
    spec = importlib.util.spec_from_file_location("train_rail_seg_yolo", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(cv2 is None, "opencv-python is required for YOLO polygon crop preparation")
class RailSegYoloDatasetTest(unittest.TestCase):
    def test_prepare_yolo_dataset_writes_cropped_segmentation_labels(self) -> None:
        module = _load_yolo_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            images = dataset / "images"
            labels = dataset / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (dataset / "classes.txt").write_text("track_area\nswitch_area\nignore_area\n", encoding="utf-8")

            for index in range(6):
                image = Image.new("RGB", (64, 96), (30, 35, 40))
                pixels = image.load()
                for y in range(8, 88):
                    for x in range(26, 39):
                        pixels[x, y] = (180, 170, 145)
                image.save(images / f"tile_{index:02d}.png")
                labels.joinpath(f"tile_{index:02d}.txt").write_text(
                    "0 0.40625 0.083333 0.609375 0.083333 0.609375 0.916667 0.40625 0.916667\n",
                    encoding="utf-8",
                )

            result = module.prepare_yolo_dataset(
                source_dataset=dataset,
                yolo_dataset_dir=root / "yolo",
                val_every=3,
                tile_height=48,
                tile_stride=24,
            )

            self.assertEqual(result["source_train_count"], 4)
            self.assertEqual(result["source_val_count"], 2)
            self.assertEqual(result["train_tile_count"], 12)
            self.assertEqual(result["val_tile_count"], 6)
            self.assertTrue(Path(result["yaml_path"]).exists())
            self.assertTrue(Path(result["split_csv_path"]).exists())

            label_paths = sorted((root / "yolo" / "labels" / "train").glob("*.txt"))
            self.assertEqual(len(label_paths), 12)
            parts = label_paths[0].read_text(encoding="utf-8").split()
            self.assertEqual(parts[0], "0")
            values = [float(value) for value in parts[1:]]
            self.assertGreaterEqual(len(values), 6)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values))


if __name__ == "__main__":
    unittest.main()

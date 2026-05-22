from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def _load_semantic_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_rail_seg_semantic.py"
    spec = importlib.util.spec_from_file_location("train_rail_seg_semantic", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RailSegSemanticTest(unittest.TestCase):
    def test_load_mask_rasterizes_yolo_polygon(self) -> None:
        module = _load_semantic_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "tile.txt"
            label_path.write_text(
                "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                encoding="utf-8",
            )

            mask = module.load_mask(label_path, 40, 40)
            arr = np.asarray(mask)

            self.assertEqual(arr.shape, (40, 40))
            self.assertGreater(int(arr.sum()), 0)
            self.assertEqual(int(arr[20, 20]), 255)
            self.assertEqual(int(arr[2, 2]), 0)

    def test_load_label_masks_marks_ignore_regions_invalid(self) -> None:
        module = _load_semantic_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "tile.txt"
            label_path.write_text(
                "\n".join(
                    [
                        "0 0.1 0.1 0.4 0.1 0.4 0.9 0.1 0.9",
                        "1 0.5 0.1 0.9 0.1 0.9 0.9 0.5 0.9",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            mask, valid = module.load_label_masks(label_path, 40, 40, {0}, {1})
            mask_arr = np.asarray(mask)
            valid_arr = np.asarray(valid)

            self.assertEqual(int(mask_arr[20, 8]), 255)
            self.assertEqual(int(mask_arr[20, 30]), 0)
            self.assertEqual(int(valid_arr[20, 8]), 255)
            self.assertEqual(int(valid_arr[20, 30]), 0)

    def test_dataset_returns_tensor_shapes(self) -> None:
        module = _load_semantic_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "tile.png"
            label_path = root / "tile.txt"
            Image.new("RGB", (32, 48), (20, 30, 40)).save(image_path)
            label_path.write_text(
                "0 0.25 0.1 0.5 0.1 0.5 0.9 0.25 0.9\n",
                encoding="utf-8",
            )
            sample = {
                "image_path": image_path,
                "label_path": label_path,
                "crop_left": 0,
                "crop_top": 0,
                "crop_width": 32,
                "crop_height": 48,
            }

            dataset = module.RailSemanticDataset([sample], (16, 24), augment=False)
            image, mask, valid = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 24, 16))
            self.assertEqual(tuple(mask.shape), (1, 24, 16))
            self.assertEqual(tuple(valid.shape), (1, 24, 16))
            self.assertGreater(float(mask.sum()), 0.0)
            self.assertEqual(float(valid.min()), 1.0)


if __name__ == "__main__":
    unittest.main()

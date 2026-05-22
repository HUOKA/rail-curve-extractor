from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_module():
    spec = importlib.util.spec_from_file_location("train_rail_seg_deeplab", SCRIPTS_DIR / "train_rail_seg_deeplab.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_rail_seg_deeplab"] = module
    spec.loader.exec_module(module)
    return module


class TrainRailSegDeepLabTest(unittest.TestCase):
    def test_patch_dataset_keeps_native_crop_size_without_resize(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            images = dataset / "images"
            labels = dataset / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            (dataset / "classes.txt").write_text("单根铁轨\nignore_area\n", encoding="utf-8")
            image = Image.new("RGB", (96, 64), (20, 22, 24))
            pixels = image.load()
            for y in range(8, 56):
                for x in range(30, 34):
                    pixels[x, y] = (190, 180, 170)
            image.save(images / "tile_001.png")
            labels.joinpath("tile_001.txt").write_text(
                "0 0.3125 0.125 0.354167 0.125 0.354167 0.875 0.3125 0.875\n",
                encoding="utf-8",
            )

            items = module.discover_items(dataset)
            samples = module.build_patch_samples(
                items,
                crop_size=32,
                stride=32,
                foreground_ids={0},
                ignore_ids={1},
                min_positive_pixels=1,
                negative_keep_ratio=0.0,
                seed=1,
            )

            self.assertTrue(samples)
            patch_dataset = module.RailPatchDataset(samples, crop_size=32, foreground_ids={0}, ignore_ids={1}, augment=False)
            image_tensor, mask_tensor, valid_tensor = patch_dataset[0]
            self.assertEqual(tuple(image_tensor.shape), (3, 32, 32))
            self.assertEqual(tuple(mask_tensor.shape), (1, 32, 32))
            self.assertEqual(tuple(valid_tensor.shape), (1, 32, 32))
            self.assertGreater(float(mask_tensor.sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()

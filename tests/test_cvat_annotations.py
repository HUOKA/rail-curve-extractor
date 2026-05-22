from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from rail_curve_extractor.cvat_annotations import (
    CvatConversionOptions,
    convert_cvat_annotations,
    load_cvat_annotations,
)


class CvatAnnotationsTest(unittest.TestCase):
    def test_loads_cvat_xml_from_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "annotations.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("annotations.xml", _cvat_xml())

            annotation_set = load_cvat_annotations(archive_path)

            self.assertEqual(annotation_set.labels, ("track_area", "switch_area"))
            self.assertEqual(len(annotation_set.images), 1)
            self.assertEqual(annotation_set.images[0].shapes[0].label, "track_area")

    def test_convert_writes_yolo_geojson_manifest_and_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "tiles" / "images"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "tile_001.png"
            Image.new("RGB", (10, 20), (30, 40, 50)).save(image_path)
            tile_georef = root / "tiles" / "tile_georef.csv"
            _write_tile_georef(tile_georef, image_path)
            annotations = root / "annotations.xml"
            annotations.write_text(_cvat_xml(), encoding="utf-8")
            output = root / "dataset"

            result = convert_cvat_annotations(
                annotations,
                tile_georef,
                output,
                CvatConversionOptions(
                    class_names=("track_area", "switch_area"),
                    make_overlays=True,
                ),
            )

            self.assertEqual(result.image_count, 1)
            self.assertEqual(result.matched_image_count, 1)
            self.assertEqual(result.shape_count, 2)
            label_text = (output / "labels" / "tile_001.txt").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(label_text[0], "0 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3")
            self.assertEqual(label_text[1], "1 0.6 0.2 0.8 0.2 0.8 0.4 0.6 0.4")
            self.assertTrue((output / "images" / "tile_001.png").exists())
            self.assertTrue((output / "overlays" / "tile_001.png").exists())
            geojson = json.loads((output / "annotations_map.geojson").read_text(encoding="utf-8"))
            self.assertEqual(geojson["crs"]["properties"]["name"], "EPSG:32651")
            first_ring = geojson["features"][0]["geometry"]["coordinates"][0]
            self.assertEqual(first_ring[0], [102.0, 196.0])
            self.assertEqual(first_ring[-1], [102.0, 196.0])
            with (output / "manifest.csv").open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["label"], "track_area")
            self.assertEqual(rows[0]["x_min"], "102.0")

    def test_unmatched_image_fails_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "tile_001.png"
            Image.new("RGB", (10, 20), (30, 40, 50)).save(image_path)
            tile_georef = root / "tile_georef.csv"
            _write_tile_georef(tile_georef, image_path)
            annotations = root / "annotations.xml"
            annotations.write_text(_cvat_xml(image_name="missing.png"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing from tile_georef"):
                convert_cvat_annotations(annotations, tile_georef, root / "dataset")


def _write_tile_georef(path: Path, image_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "tile_id",
                "tile_name",
                "image_path",
                "source_path",
                "row_off",
                "col_off",
                "width",
                "height",
                "source_width",
                "source_height",
                "tile_transform",
                "source_transform",
                "crs",
                "epsg",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "valid_ratio",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_id": 0,
                "tile_name": image_path.name,
                "image_path": str(image_path),
                "source_path": str(path.parent / "dom.tif"),
                "row_off": 0,
                "col_off": 0,
                "width": 10,
                "height": 20,
                "source_width": 10,
                "source_height": 20,
                "tile_transform": json.dumps([2.0, 0.0, 100.0, 0.0, -2.0, 200.0]),
                "source_transform": json.dumps([2.0, 0.0, 100.0, 0.0, -2.0, 200.0]),
                "crs": "EPSG:32651",
                "epsg": 32651,
                "x_min": 100.0,
                "y_min": 160.0,
                "x_max": 120.0,
                "y_max": 200.0,
                "valid_ratio": 1.0,
            }
        )


def _cvat_xml(image_name: str = "images/tile_001.png") -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>track_area</name></label>
        <label><name>switch_area</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="{image_name}" width="10" height="20">
    <polygon label="track_area" points="1,2;5,2;5,6;1,6" occluded="0" z_order="0" />
    <box label="switch_area" xtl="6" ytl="4" xbr="8" ybr="8" occluded="0" z_order="1" />
  </image>
</annotations>
"""


if __name__ == "__main__":
    unittest.main()

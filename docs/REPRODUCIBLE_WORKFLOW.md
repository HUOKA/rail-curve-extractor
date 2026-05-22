# Reproducible Workflow

This repository is intended to be cloned from GitHub and run against local DJI Terra / drone mapping outputs. Large inputs, trained checkpoints, and generated `output/` folders are intentionally not tracked.

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Install DeepLab training/inference dependencies only when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deeplab]"
```

YOLO training is optional:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[yolo]"
```

## 2. Prepare Local Inputs

Put production data outside Git, for example:

```text
data/production/terra_dom/dom.tif
data/production/terra_dsm/dsm.tif
data/production/terra_las/cloud0.las
data/production/terra_las/cloud1.las
```

Use `data/config.example.json` or the root `config.example.json` as configuration references. Model checkpoints are also local artifacts; keep them outside Git or publish them separately as Release assets.

## 3. Run Full 2D/3D Centerline Pipeline

The current accepted production route is:

```text
DOM tiles
-> DeepLab rail probability masks
-> rail candidate extraction
-> straight-band and turnout topology postprocess
-> strict-auto global 2D review package with tangent-smoothed turnout endpoints
-> LAS/DSM Z assignment
-> formal 2D/3D delivery package
```

The final accepted packaging step is represented by:

```powershell
.\.venv\Scripts\python.exe scripts\package_strict_auto_global_centerline_review.py
.\.venv\Scripts\python.exe scripts\add_z_to_deeplab_topology_centerline.py `
  --input output\dom_centerline_strict_auto_v1\global_centerline_review_tangent_occlusion\global_centerline_2d.geojson `
  --output-dir output\dom_centerline_strict_auto_v1\global_centerline_review_tangent_occlusion_z
```

The formal delivery directory should contain:

```text
output/dom_centerline_strict_auto_v1/final_delivery/
  centerline_2d.shp
  centerline_3d.shp
  centerline_evidence.shp
  delivery_manifest.json
```

`centerline_2d.shp` is a 2D `POLYLINE`; `centerline_3d.shp` is a 3D `POLYLINEZ`.

## 4. Validation

Run focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_strict_auto_global_centerline_review_package.py `
  tests\test_turnout_outer_rail_geometry.py `
  tests\test_centerline_z_export.py `
  tests\test_dom_to_3d_guided_pipeline.py `
  tests\test_deeplab_topology_centerline_network.py
```

For a broader code check:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Some training or UI tests may require optional dependencies, GUI libraries, CUDA, or local data.

## 5. Git Hygiene

Tracked repository content should stay limited to code, tests, docs, configuration examples, and small metadata. Do not commit:

- `output/`
- production `data/`
- `.codex-tasks/`
- virtual environments
- model checkpoints
- LAS/LAZ/GeoTIFF/Shapefile generated artifacts

Publish final delivery artifacts as Release assets or external packages, not as normal Git files.

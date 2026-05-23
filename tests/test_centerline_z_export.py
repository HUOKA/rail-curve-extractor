import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import shapefile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "add_z_to_deeplab_topology_centerline.py"
SPEC = importlib.util.spec_from_file_location("centerline_z_export", SCRIPT_PATH)
centerline_z = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = centerline_z
SPEC.loader.exec_module(centerline_z)


def test_densify_line_keeps_source_endpoints_and_station_length():
    feature = centerline_z.LineFeature(
        index=0,
        properties={"line_id": "L1"},
        coords=np.asarray([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]], dtype=float),
    )

    dense = centerline_z.densify_line(feature, spacing_m=1.0)

    assert np.allclose(dense.coords[0], [0.0, 0.0])
    assert np.allclose(dense.coords[-1], [3.0, 4.0])
    assert math.isclose(float(dense.stations[-1]), 7.0)
    assert np.all((np.isclose(dense.coords[:, 1], 0.0)) | (np.isclose(dense.coords[:, 0], 3.0)))


def test_smooth_z_profile_removes_broad_spike_without_moving_xy():
    base = np.linspace(10.0, 11.0, 301)
    noisy = base.copy()
    noisy[125:165] += 3.0

    smoothed, outlier_count = centerline_z.smooth_z_profile(
        noisy,
        spacing_m=1.0,
        window_m=51.0,
        despike_window_m=201.0,
        despike_threshold_m=0.45,
        polyorder=2,
    )

    assert outlier_count >= 35
    assert float(np.max(smoothed)) < 11.2
    assert float(np.percentile(np.abs(np.diff(smoothed)), 95)) < 0.01


def test_topology_gap_bridge_uses_connected_endpoint_z():
    left = centerline_z.LineFeature(0, {"line_id": "A", "network_role": "parallel_straight_track"}, np.asarray([[0.0, 0.0], [10.0, 0.0]]))
    bridge = centerline_z.LineFeature(1, {"line_id": "B", "network_role": "straight_gap_bridge", "source_layer": "topology_gap_bridge"}, np.asarray([[10.0, 0.0], [20.0, 0.0]]))
    right = centerline_z.LineFeature(2, {"line_id": "C", "network_role": "parallel_straight_track"}, np.asarray([[20.0, 0.0], [30.0, 0.0]]))
    dense_lines = [centerline_z.densify_line(feature, 1.0) for feature in [left, bridge, right]]
    z_lines = [
        centerline_z.ZLine(dense_lines[0], np.full(11, 10.0), np.full(11, 10.0), np.zeros(11), np.full(11, 10.0), np.full(11, 10.0), "las_rail_pair", 0, 0),
        centerline_z.ZLine(dense_lines[1], np.full(11, 15.0), np.full(11, 15.0), np.zeros(11), np.full(11, 15.0), np.full(11, 15.0), "las_rail_pair", 0, 0),
        centerline_z.ZLine(dense_lines[2], np.full(11, 10.2), np.full(11, 10.2), np.zeros(11), np.full(11, 10.2), np.full(11, 10.2), "las_rail_pair", 0, 0),
    ]

    centerline_z.apply_topology_z_constraints(
        z_lines,
        endpoint_tolerance_m=0.25,
        endpoint_taper_m=15.0,
        bridge_replace_threshold_m=0.50,
    )

    assert z_lines[1].bridge_z_mode == "endpoint_interpolated"
    assert np.allclose(z_lines[1].smooth_z, np.linspace(10.0, 10.2, 11))


def test_turnout_endpoint_uses_intersecting_track_z():
    main = centerline_z.LineFeature(0, {"line_id": "MAIN", "network_role": "main_through_track"}, np.asarray([[0.0, 0.0], [20.0, 0.0]]))
    turnout = centerline_z.LineFeature(1, {"line_id": "TURNOUT", "network_role": "turnout_connector"}, np.asarray([[10.0, 0.0], [10.0, 10.0]]))
    dense_lines = [centerline_z.densify_line(feature, 1.0) for feature in [main, turnout]]
    z_lines = [
        centerline_z.ZLine(
            dense_lines[0],
            np.linspace(10.0, 20.0, 21),
            np.linspace(10.0, 20.0, 21),
            np.ones(21),
            np.linspace(10.0, 20.0, 21),
            np.linspace(10.0, 20.0, 21),
            "las_rail_pair",
            0,
            0,
        ),
        centerline_z.ZLine(
            dense_lines[1],
            np.full(11, 30.0),
            np.full(11, 30.0),
            np.ones(11),
            np.full(11, 30.0),
            np.full(11, 30.0),
            "las_rail_pair",
            0,
            0,
        ),
    ]

    centerline_z.apply_topology_z_constraints(
        z_lines,
        endpoint_tolerance_m=0.25,
        endpoint_taper_m=5.0,
        bridge_replace_threshold_m=0.50,
    )

    assert math.isclose(float(z_lines[1].smooth_z[0]), 15.0)
    assert z_lines[1].endpoint_constraint_count == 1
    assert np.allclose(z_lines[0].smooth_z, np.linspace(10.0, 20.0, 21))


def test_write_polylinez_shapefile(tmp_path):
    feature = centerline_z.LineFeature(
        0,
        {"line_id": "L1", "network_role": "main_through_track", "source_layer": "test", "length_m": 1.0},
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
    )
    dense = centerline_z.DenseLine(
        source=feature,
        coords=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        stations=np.asarray([0.0, 1.0], dtype=float),
        tangents=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float),
    )
    z_line = centerline_z.ZLine(
        dense=dense,
        dsm_z=np.asarray([10.0, 10.5]),
        las_z=np.asarray([10.0, 10.5]),
        las_counts=np.asarray([5, 5]),
        raw_z=np.asarray([10.0, 10.5]),
        smooth_z=np.asarray([10.0, 10.5]),
        source="las_rail_pair",
        fallback_count=0,
        outlier_count=0,
    )
    out = tmp_path / "line_z.shp"

    centerline_z.write_polylinez_shapefile(out, [z_line], epsg=32651)

    reader = shapefile.Reader(str(out))
    assert reader.shapeType == shapefile.POLYLINEZ
    shape = reader.shape(0)
    assert shape.points == [(0.0, 0.0), (1.0, 0.0)]
    assert np.allclose(shape.z, [10.0, 10.5])

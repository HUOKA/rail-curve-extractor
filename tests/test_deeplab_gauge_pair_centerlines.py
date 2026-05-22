from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_deeplab_gauge_pair_centerlines.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("build_deeplab_gauge_pair_centerlines", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pair_lateral_peaks_requires_gauge_spacing_and_uses_each_peak_once() -> None:
    module = _load_module()
    peaks = [
        module.LateralPeak(0, 10.0, -0.8, 20.0, 30, 0.95, 0.12),
        module.LateralPeak(0, 10.0, 0.8, 18.0, 28, 0.93, 0.10),
        module.LateralPeak(0, 10.0, 4.2, 16.0, 26, 0.90, 0.08),
        module.LateralPeak(0, 10.0, 5.8, 15.0, 25, 0.89, 0.07),
        module.LateralPeak(0, 10.0, 10.0, 40.0, 40, 0.99, 0.01),
        module.LateralPeak(0, 10.0, 11.0, 38.0, 38, 0.99, 0.01),
    ]

    pairs = module.pair_lateral_peaks(peaks, gauge_m=1.6, tolerance_m=0.18, max_pairs=4)

    assert len(pairs) == 2
    assert np.allclose([pair.center_offset_m for pair in pairs], [0.0, 5.0])
    assert all(abs(pair.gauge_m - 1.6) < 1e-9 for pair in pairs)


def test_link_pair_samples_keeps_parallel_tracks_separate() -> None:
    module = _load_module()
    samples = []
    sample_id = 1
    for index, station in enumerate(np.arange(0.0, 12.0, 1.0)):
        for center_offset in (0.0, 5.0):
            samples.append(
                module.PairSample(
                    sample_id=sample_id,
                    station_index=index,
                    station_m=float(station),
                    center_offset_m=center_offset,
                    left_offset_m=center_offset - 0.8,
                    right_offset_m=center_offset + 0.8,
                    gauge_m=1.6,
                    score=10.0,
                    probability_score=10.0,
                    dsm_contrast_m=0.1,
                    left_count=20,
                    right_count=20,
                )
            )
            sample_id += 1

    sequences = module.link_pair_samples(
        samples,
        max_gap_m=2.0,
        max_offset_m=0.35,
        max_slope=0.1,
        min_samples=6,
        min_length_m=5.0,
    )

    assert len(sequences) == 2
    means = sorted(round(float(np.mean([sample.center_offset_m for sample in seq.samples])), 3) for seq in sequences)
    assert means == [0.0, 5.0]


def test_link_pair_samples_allows_smooth_turnout_drift() -> None:
    module = _load_module()
    samples = []
    for index, station in enumerate(np.arange(0.0, 20.0, 1.0), start=1):
        center = 5.0 - 0.12 * station
        samples.append(
            module.PairSample(
                sample_id=index,
                station_index=index,
                station_m=float(station),
                center_offset_m=float(center),
                left_offset_m=float(center - 0.8),
                right_offset_m=float(center + 0.8),
                gauge_m=1.6,
                score=10.0,
                probability_score=10.0,
                dsm_contrast_m=None,
                left_count=20,
                right_count=20,
            )
        )

    sequences = module.link_pair_samples(
        samples,
        max_gap_m=2.0,
        max_offset_m=0.25,
        max_slope=0.2,
        min_samples=10,
        min_length_m=8.0,
    )

    assert len(sequences) == 1
    assert len(sequences[0].samples) == len(samples)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import load_point_cloud_data
from .preview import downsample_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a point cloud in an Open3D viewer window.")
    parser.add_argument("--input", required=True, help="Input point cloud file or DJI Terra project folder.")
    parser.add_argument("--max-points", type=int, default=2_000_000, help="Maximum points sent to the viewer.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size in the Open3D window.")
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Optional XY bounds used to load only the current viewport at higher density.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        import open3d as o3d
    except ImportError:
        print("Open3D 未安装。请先安装桌面查看器依赖：pip install open3d", flush=True)
        return 2

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"点云文件不存在：{input_path}", flush=True)
        return 1

    max_points = max(1_000, int(args.max_points))
    xy_bounds = tuple(args.bounds) if args.bounds is not None else None
    point_cloud = load_point_cloud_data(input_path, max_points=max_points, xy_bounds=xy_bounds)
    indices = downsample_indices(len(point_cloud.points), max_points)
    points = np.asarray(point_cloud.points[indices], dtype=np.float64)
    if len(points) == 0:
        print("Open3D 查看范围内没有可显示的点。", flush=True)
        return 1

    local_origin = (points.min(axis=0) + points.max(axis=0)) / 2.0
    local_points = points - local_origin

    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(local_points)
    if point_cloud.rgb is not None and len(point_cloud.rgb) == len(point_cloud.points):
        colors = np.asarray(point_cloud.rgb[indices], dtype=np.float64)
        max_value = float(np.nanmax(colors)) if colors.size else 1.0
        scale = 65535.0 if max_value > 255.0 else 255.0
        geometry.colors = o3d.utility.Vector3dVector(np.clip(colors / scale, 0.0, 1.0))

    print(
        f"Open3D 显示点数：{len(points):,} / {int(point_cloud.source_point_count or len(points)):,}；"
        f"本地坐标原点偏移：{local_origin[0]:.3f}, {local_origin[1]:.3f}, {local_origin[2]:.3f}",
        flush=True,
    )
    if xy_bounds is not None:
        print(
            f"Open3D 当前视口范围：X {xy_bounds[0]:.3f}–{xy_bounds[1]:.3f}，"
            f"Y {xy_bounds[2]:.3f}–{xy_bounds[3]:.3f}",
            flush=True,
        )

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=f"Rail Curve Open3D Viewer - {input_path.name}",
        width=1440,
        height=920,
    )
    visualizer.add_geometry(geometry)
    render_option = visualizer.get_render_option()
    render_option.point_size = max(1.0, float(args.point_size))
    render_option.background_color = np.asarray([0.96, 0.98, 1.0])
    visualizer.run()
    visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

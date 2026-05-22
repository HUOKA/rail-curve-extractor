from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from .io import load_point_cloud_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve an Open3D WebRTC viewer for embedding in Electron.")
    parser.add_argument("--input", required=True, help="Input point cloud file or DJI Terra project folder.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum points sent to Open3D. Use 0 to load all source points.",
    )
    parser.add_argument("--title", default="Rail Curve Open3D Canvas")
    parser.add_argument("--point-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--progress-file", default="", help="JSON file used by the desktop UI to read real loading progress.")
    return parser


def write_progress(progress_path: Path | None, **payload: Any) -> None:
    loaded_points = payload.get("loaded_points")
    total_points = payload.get("total_points")
    percent = None
    if isinstance(loaded_points, int | float) and isinstance(total_points, int | float) and total_points > 0:
        percent = max(0.0, min(100.0, round(float(loaded_points) * 100.0 / float(total_points), 2)))

    data = {
        "updated_at": time.time(),
        "percent": percent,
        **payload,
    }
    line = json.dumps(_json_safe(data), ensure_ascii=False)
    print(f"OPEN3D_PROGRESS {line}", flush=True)
    if progress_path is None:
        return

    _write_progress_text(progress_path, line)


def _write_progress_text(progress_path: Path, line: str) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = progress_path.with_name(f"{progress_path.name}.{os.getpid()}.tmp")
    last_error: OSError | None = None

    for _ in range(4):
        try:
            temporary_path.write_text(line, encoding="utf-8")
            temporary_path.replace(progress_path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)

    for _ in range(4):
        try:
            progress_path.write_text(line, encoding="utf-8")
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)

    print(f"OPEN3D_PROGRESS_WRITE_WARNING {last_error}", flush=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def progress_message(phase: str, event: dict[str, Any] | None = None) -> str:
    event = event or {}
    file_index = event.get("file_index")
    file_count = event.get("file_count")
    current_file = Path(str(event["current_file"])).name if event.get("current_file") else ""
    if phase == "discovering_files":
        return "正在查找点云分块文件"
    if phase == "files_discovered":
        return f"已找到 {file_count or 0} 个点云文件"
    if phase == "reading_file":
        prefix = f"正在读取第 {file_index}/{file_count} 个文件" if file_index and file_count else "正在读取点云文件"
        return f"{prefix}：{current_file}" if current_file else prefix
    if phase == "file_loaded":
        return f"已读取：{current_file}" if current_file else "当前文件读取完成"
    if phase == "combining_arrays":
        return "正在合并分块点云数组"
    return "正在加载点云"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    progress_path = Path(args.progress_file).expanduser() if args.progress_file else None

    def report(state: str, phase: str, message: str, **payload: Any) -> None:
        write_progress(
            progress_path,
            state=state,
            phase=phase,
            message=message,
            url="http://127.0.0.1:8888",
            **payload,
        )

    try:
        report("starting", "importing_open3d", "正在加载 Open3D 运行库")
        import open3d as o3d
    except ImportError:
        report("failed", "importing_open3d", "Open3D 未安装，请先安装 open3d 或使用项目内 .open3d-venv", error="open3d is not installed")
        return 2

    try:
        input_path = Path(args.input)
        if not input_path.exists():
            report("failed", "checking_input", f"点云路径不存在：{input_path}", error=f"Input path does not exist: {input_path}")
            return 1

        max_points = int(args.max_points)
        report("loading", "discovering_files", "正在查找点云分块文件", input_path=str(input_path), max_points=max_points)

        def on_load_progress(event: dict[str, Any]) -> None:
            phase = str(event.get("phase", "reading_file"))
            progress_payload = {key: value for key, value in event.items() if key not in {"phase", "state", "message", "url"}}
            report("loading", phase, progress_message(phase, event), **progress_payload)

        point_cloud = load_point_cloud_data(
            input_path,
            max_points=max_points if max_points > 0 else 0,
            progress_callback=on_load_progress,
        )
        points = np.asarray(point_cloud.points, dtype=np.float64)
        source_points = int(point_cloud.source_point_count or len(points))
        if len(points) == 0:
            report("failed", "loaded_empty_cloud", f"点云为空：{input_path}", error=f"Empty point cloud: {input_path}")
            return 1

        report(
            "building",
            "building_open3d_geometry",
            "正在构建 Open3D 点云几何",
            loaded_points=source_points,
            total_points=source_points,
            display_points=int(len(points)),
            source_paths=[str(path) for path in point_cloud.source_paths],
        )
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(points)
        if point_cloud.rgb is not None and len(point_cloud.rgb) == len(points):
            colors = np.asarray(point_cloud.rgb, dtype=np.float64)
            max_value = float(np.nanmax(colors)) if colors.size else 1.0
            scale = 65535.0 if max_value > 255.0 else 255.0
            geometry.colors = o3d.utility.Vector3dVector(np.clip(colors / scale, 0.0, 1.0))

        report(
            "serving",
            "starting_webrtc",
            "正在启动 Open3D WebRTC 服务",
            loaded_points=source_points,
            total_points=source_points,
            display_points=int(len(points)),
        )
        o3d.visualization.webrtc_server.enable_webrtc()
        uid = o3d.visualization.draw(
            [{"name": input_path.name, "geometry": geometry}],
            title=args.title,
            width=max(320, int(args.width)),
            height=max(240, int(args.height)),
            bg_color=(0.96, 0.98, 1.0, 1.0),
            point_size=max(1, int(args.point_size)),
            show_ui=False,
            raw_mode=True,
            non_blocking_and_return_uid=True,
        )
        report(
            "ready",
            "ready",
            "Open3D 全量画布已就绪",
            loaded_points=source_points,
            total_points=source_points,
            display_points=int(len(points)),
            uid=str(uid),
        )
        print(f"OPEN3D_WEBRTC_READY uid={uid} url=http://127.0.0.1:8888 points={len(points)}", flush=True)
    except Exception as exc:
        report("failed", "exception", f"Open3D 加载失败：{exc}", error=str(exc))
        traceback.print_exc()
        return 1

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())

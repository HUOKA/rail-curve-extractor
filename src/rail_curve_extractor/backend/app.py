from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.request import urlopen
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from ..io import load_point_cloud_data, summarize_point_cloud_input
from ..pipeline import PipelineResult, analyze_input, export_pipeline_result, prepare_config
from ..preview import downsample_indices

API_PREFIX = "/api"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

app = FastAPI(title="Rail Curve Extractor Local Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RESULTS: dict[str, PipelineResult] = {}
_EMBEDDED_VIEWER_PROCESS: subprocess.Popen[Any] | None = None
_EMBEDDED_VIEWER_LOG_HANDLE: Any | None = None
_DOM_PIPELINE_JOBS: dict[str, dict[str, Any]] = {}
OPEN3D_WEBRTC_URL = "http://127.0.0.1:8888"


class AnalyzeRequest(BaseModel):
    input_path: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    input_path: str | None = None
    output_dir: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    result_id: str | None = None


class ViewerRequest(BaseModel):
    input_path: str
    max_points: int = 2_000_000
    point_size: float = 2.0
    bounds: dict[str, float] | None = None


class EmbeddedViewerRequest(BaseModel):
    input_path: str
    max_points: int = 3_000_000
    point_size: int = 1


class PointCloudInfoRequest(BaseModel):
    input_path: str


class PointCloudPreviewRequest(BaseModel):
    input_path: str
    max_points: int = 80_000
    bounds: dict[str, float] | None = None


class DomPipelineStartRequest(BaseModel):
    dom_path: str
    model_path: str
    output_dir: str
    dsm_path: str | None = None
    las_dir: str | None = None
    profile: str = "strict-auto"
    device: str = "cuda"
    threshold: float = 0.50
    max_tiles: int = 0
    force: bool = False
    epsg: int = 32651


class RasterProbeRequest(BaseModel):
    path: str


@app.middleware("http")
async def require_local_token(request: Request, call_next: Any) -> JSONResponse:
    expected_token = os.environ.get("RAIL_CURVE_BACKEND_TOKEN", "")
    if expected_token and request.url.path.startswith(API_PREFIX):
        provided_token = request.headers.get("x-local-token", "")
        if provided_token != expected_token:
            return JSONResponse({"detail": "Invalid local token."}, status_code=403)
    return await call_next(request)


@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "rail-curve-extractor-backend",
        "python": sys.version.split()[0],
    }


@app.post(f"{API_PREFIX}/raster/probe")
def raster_probe(request: RasterProbeRequest) -> dict[str, Any]:
    """Read CRS / size / bounds from a raster file (e.g. DOM or DSM).

    Used by the desktop UI to auto-detect EPSG instead of asking the user to type it.
    Pure read-only metadata; does not load pixel data.
    """
    raster_path = _existing_path(request.path, "栅格文件")
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - rasterio is a hard dep
        raise HTTPException(status_code=500, detail=f"rasterio 不可用：{exc}") from exc

    try:
        with rasterio.open(str(raster_path)) as dataset:
            crs = dataset.crs
            epsg = crs.to_epsg() if crs else None
            crs_name = crs.to_string() if crs else None
            bounds = dataset.bounds
            pixel_w, pixel_h = _affine_pixel_size(dataset.transform)
            return {
                "path": str(raster_path),
                "epsg": int(epsg) if epsg else None,
                "crs": crs_name,
                "width": int(dataset.width),
                "height": int(dataset.height),
                "band_count": int(dataset.count),
                "driver": dataset.driver,
                "bounds": {
                    "left": float(bounds.left),
                    "bottom": float(bounds.bottom),
                    "right": float(bounds.right),
                    "top": float(bounds.top),
                },
                "pixel_size": [pixel_w, pixel_h],
            }
    except rasterio.RasterioIOError as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail=f"无法读取栅格：{exc}") from exc


def _affine_pixel_size(transform: Any) -> tuple[float, float]:
    """Return real per-pixel ground size from a rasterio Affine.

    Affine layout is (a, b, c, d, e, f) where the first column (a, d) maps the X
    pixel axis to world coordinates and the second column (b, e) maps the Y
    pixel axis. For north-up rasters b == d == 0 and the size is just |a|, |e|.
    For rotated / sheared rasters that simplification is wrong; take the
    Euclidean length of each column instead.
    """
    a = float(transform.a)
    b = float(transform.b)
    d = float(transform.d)
    e = float(transform.e)
    pixel_w = (a * a + d * d) ** 0.5
    pixel_h = (b * b + e * e) ** 0.5
    return pixel_w, pixel_h


@app.get(f"{API_PREFIX}/system/devices")
def system_devices() -> dict[str, Any]:
    """Probe local CPU / GPU and PyTorch CUDA availability for the desktop UI."""
    return {
        "cpu": _probe_cpu(),
        "cuda": _probe_cuda(),
        "torch": _probe_torch(),
    }


def _probe_cpu() -> dict[str, Any]:
    """Best-effort CPU description across Windows / Linux / macOS."""
    info: dict[str, Any] = {
        "name": platform.processor() or platform.machine() or "CPU",
        "arch": platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "platform": f"{platform.system()} {platform.release()}",
    }

    # Windows: WMIC is deprecated but still present; fall back to PowerShell CIM.
    if platform.system() == "Windows":
        for cmd in (
            ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/format:list"],
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | "
                "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-List",
            ],
        ):
            text = _run_capture(cmd, timeout=5)
            if not text:
                continue
            parsed = _parse_kv_block(text)
            if parsed.get("Name"):
                info["name"] = parsed["Name"].strip()
                if parsed.get("NumberOfCores", "").isdigit():
                    info["physical_cores"] = int(parsed["NumberOfCores"])
                if parsed.get("NumberOfLogicalProcessors", "").isdigit():
                    info["logical_cores"] = int(parsed["NumberOfLogicalProcessors"])
                break
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        info["name"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    elif platform.system() == "Darwin":
        text = _run_capture(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2)
        if text:
            info["name"] = text.strip()

    return info


def _probe_cuda() -> dict[str, Any]:
    """Detect NVIDIA GPUs via nvidia-smi. Returns available=False if no NVIDIA driver."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {
            "available": False,
            "reason": "missing-driver",
            "message": "未检测到 NVIDIA 驱动（nvidia-smi 不可用）",
            "gpus": [],
        }

    output = _run_capture(
        [
            nvidia_smi,
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if not output:
        return {
            "available": False,
            "reason": "smi-failed",
            "message": "nvidia-smi 执行失败",
            "gpus": [],
        }

    gpus = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mem_total_mib = int(parts[1])
        except ValueError:
            mem_total_mib = None
        gpus.append(
            {
                "name": parts[0],
                "memory_total_mib": mem_total_mib,
                "driver_version": parts[2] if len(parts) > 2 else None,
                "compute_capability": parts[3] if len(parts) > 3 else None,
            }
        )

    return {
        "available": len(gpus) > 0,
        "reason": None if gpus else "no-device",
        "message": None,
        "gpus": gpus,
    }


def _probe_torch() -> dict[str, Any]:
    """Probe PyTorch availability and CUDA build."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "installed": False,
            "version": None,
            "cuda_build": None,
            "cuda_runtime_available": False,
            "device_count": 0,
            "message": f"PyTorch 未安装：{exc}",
        }

    cuda_available = False
    device_count = 0
    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_count = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        cuda_available = False
        device_count = 0

    return {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_build": getattr(torch.version, "cuda", None),  # type: ignore[attr-defined]
        "cuda_runtime_available": cuda_available,
        "device_count": device_count,
        "message": None,
    }


def _run_capture(cmd: list[str], timeout: float) -> str:
    """Run a command capturing stdout; return '' on any failure."""
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if completed.returncode != 0:
        return completed.stdout or ""
    return completed.stdout or ""


def _parse_kv_block(text: str) -> dict[str, str]:
    """Parse `Key=Value` or `Key : Value` lines into a dict (last wins)."""
    parsed: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=" in line:
            key, _, value = line.partition("=")
        elif ":" in line:
            key, _, value = line.partition(":")
        else:
            continue
        parsed[key.strip()] = value.strip()
    return parsed


@app.get("/open3d-local-ice")
def open3d_local_ice() -> JSONResponse:
    return JSONResponse(
        {"iceServers": []},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-store",
        },
    )


@app.get(f"{API_PREFIX}/config/default")
def default_config() -> dict[str, Any]:
    return _json_safe(prepare_config())


@app.post(f"{API_PREFIX}/point-cloud/info")
def point_cloud_info(request: PointCloudInfoRequest) -> dict[str, Any]:
    input_path = _existing_path(request.input_path, "点云文件")
    return _json_safe(summarize_point_cloud_input(input_path))


@app.post(f"{API_PREFIX}/point-cloud/preview")
def point_cloud_preview(request: PointCloudPreviewRequest) -> dict[str, Any]:
    input_path = _existing_path(request.input_path, "点云文件")
    max_points = max(1, min(500_000, int(request.max_points)))
    xy_bounds = _preview_xy_bounds(request.bounds)
    point_cloud = load_point_cloud_data(input_path, max_points=max_points, xy_bounds=xy_bounds)
    indices = downsample_indices(len(point_cloud.points), max_points)
    points_xy = point_cloud.points[indices, :2].astype(float)
    rgb = None
    if point_cloud.rgb is not None and len(point_cloud.rgb) == len(point_cloud.points):
        rgb_values = np.asarray(point_cloud.rgb[indices], dtype=np.float64)
        max_value = float(np.nanmax(rgb_values)) if rgb_values.size else 0.0
        scale = 256.0 if max_value > 255.0 else 1.0
        rgb = np.clip(rgb_values / scale, 0.0, 255.0).astype(np.uint8)

    return {
        "input_path": str(input_path),
        "input_points": int(point_cloud.source_point_count or len(point_cloud.points)),
        "sample_points": int(len(points_xy)),
        "source_file_count": int(len(point_cloud.source_paths) or 1),
        "source_paths": [str(file_path) for file_path in point_cloud.source_paths],
        "points_xy": np.round(points_xy, 3).tolist(),
        "rgb": rgb.tolist() if rgb is not None else None,
        "bounds": _preview_response_bounds(points_xy, xy_bounds),
    }


@app.post(f"{API_PREFIX}/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    input_path = _existing_path(request.input_path, "点云文件")
    result = analyze_input(input_path=input_path, config_overrides=request.config_overrides)
    result_id = uuid4().hex
    _RESULTS[result_id] = result
    return {
        "result_id": result_id,
        "summary": _json_safe(result.summary),
        "overlay": _result_overlay(result),
    }


@app.post(f"{API_PREFIX}/export")
def export(request: ExportRequest) -> dict[str, Any]:
    output_dir = Path(request.output_dir).expanduser()
    result_id = request.result_id
    if result_id:
        result = _RESULTS.get(result_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"找不到分析结果：{result_id}")
        input_path = Path(request.input_path).expanduser() if request.input_path else None
    else:
        if not request.input_path:
            raise HTTPException(status_code=400, detail="没有 result_id 时必须提供 input_path。")
        input_path = _existing_path(request.input_path, "点云文件")
        result = analyze_input(input_path=input_path, config_overrides=request.config_overrides)
        result_id = uuid4().hex
        _RESULTS[result_id] = result

    summary = export_pipeline_result(result=result, output_dir=output_dir, input_path=input_path)
    return {
        "result_id": result_id,
        "output_dir": str(output_dir),
        "summary": _json_safe(summary),
        "overlay": _result_overlay(result),
    }


@app.post(f"{API_PREFIX}/dom-pipeline/start")
def start_dom_pipeline(request: DomPipelineStartRequest) -> dict[str, Any]:
    dom_path = _existing_path(request.dom_path, "DOM")
    model_path = _existing_path(request.model_path, "DeepLab 权重")
    output_dir = Path(request.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dsm_path = _existing_path(request.dsm_path, "DSM") if request.dsm_path else None
    las_dir = _existing_path(request.las_dir, "LAS 目录") if request.las_dir else None

    if request.profile not in {"strict-auto", "dom-full", "accepted-baseline"}:
        raise HTTPException(status_code=400, detail=f"不支持的流水线 profile：{request.profile}")

    job_id = uuid4().hex
    log_path, progress_path = _dom_pipeline_paths(job_id)
    command = [
        str(_pipeline_python_executable()),
        str(_project_root() / "scripts" / "run_dom_to_3d_centerline_guided_pipeline.py"),
        "--profile",
        request.profile,
        "--dom",
        str(dom_path),
        "--deeplab-model",
        str(model_path),
        "--out-dir",
        str(output_dir),
        "--device",
        request.device,
        "--threshold",
        str(max(0.0, min(1.0, float(request.threshold)))),
        "--max-tiles",
        str(max(0, int(request.max_tiles))),
        "--epsg",
        str(int(request.epsg)),
        "--progress-file",
        str(progress_path),
    ]
    if dsm_path is not None:
        command.extend(["--dsm", str(dsm_path)])
    if las_dir is not None:
        command.extend(["--las-dir", str(las_dir)])
    if request.force:
        command.append("--force")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        progress_path,
        {
            "state": "starting",
            "message": "正在启动 DOM 语义分割流水线",
            "stage_index": 0,
            "stage_count": 0,
            "percent": 0.0,
            "out_dir": str(output_dir),
            "updated_at": time.time(),
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_project_src(env.get("PYTHONPATH", ""))
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(_project_root()),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    _DOM_PIPELINE_JOBS[job_id] = {
        "process": process,
        "log_handle": log_handle,
        "command": command,
        "log_path": log_path,
        "progress_path": progress_path,
        "output_dir": output_dir,
        "started_at": time.time(),
    }
    return _dom_pipeline_status(job_id)


@app.get(f"{API_PREFIX}/dom-pipeline/status/{{job_id}}")
def dom_pipeline_status(job_id: str) -> dict[str, Any]:
    return _dom_pipeline_status(job_id)


@app.post(f"{API_PREFIX}/dom-pipeline/stop/{{job_id}}")
def stop_dom_pipeline(job_id: str) -> dict[str, Any]:
    job = _DOM_PIPELINE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"找不到 DOM 流水线任务：{job_id}")
    process: subprocess.Popen[Any] | None = job.get("process")
    stopped = False
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        stopped = True
    _close_job_log_handle(job)
    _write_json_atomic(
        Path(job["progress_path"]),
        {
            **_read_json_if_exists(Path(job["progress_path"])),
            "state": "stopped",
            "message": "DOM 流水线已停止",
            "updated_at": time.time(),
        },
    )
    status = _dom_pipeline_status(job_id)
    status["stopped"] = stopped
    return status


@app.post(f"{API_PREFIX}/viewer/open")
def open_viewer(request: ViewerRequest) -> dict[str, Any]:
    input_path = _existing_path(request.input_path, "点云文件")
    xy_bounds = _preview_xy_bounds(request.bounds)
    python_executable = _viewer_python_executable()
    command = [
        str(python_executable),
        "-m",
        "rail_curve_extractor.open3d_viewer",
        "--input",
        str(input_path),
        "--max-points",
        str(max(1_000, request.max_points)),
        "--point-size",
        str(max(1.0, float(request.point_size))),
    ]
    if xy_bounds is not None:
        command.extend(["--bounds", *(str(value) for value in xy_bounds)])
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_project_src(env.get("PYTHONPATH", ""))
    process = subprocess.Popen(command, cwd=str(_project_root()), env=env)
    return {
        "started": True,
        "pid": process.pid,
        "command": command,
        "viewer_python": str(python_executable),
        "bounds": list(xy_bounds) if xy_bounds is not None else None,
        "max_points": max(1_000, int(request.max_points)),
    }


@app.post(f"{API_PREFIX}/viewer/embedded/start")
def start_embedded_viewer(request: EmbeddedViewerRequest) -> dict[str, Any]:
    global _EMBEDDED_VIEWER_PROCESS, _EMBEDDED_VIEWER_LOG_HANDLE
    input_path = _existing_path(request.input_path, "点云文件")
    if os.environ.get("RAIL_CURVE_ENABLE_OPEN3D_WEBRTC") != "1":
        raise HTTPException(
            status_code=501,
            detail=(
                "内嵌 Open3D WebRTC 在当前 Windows/Electron 环境会在握手阶段崩溃，"
                "已默认禁用。请使用 Open3D 独立窗口或自动LOD标注画布。"
            ),
        )
    _stop_embedded_viewer_process()

    python_executable = _viewer_python_executable()
    log_path, progress_path = _embedded_viewer_paths()
    _write_embedded_progress(
        progress_path,
        {
            "state": "starting",
            "phase": "starting_process",
            "message": "正在启动 Open3D 子进程",
            "input_path": str(input_path),
            "loaded_points": 0,
            "total_points": None,
            "percent": None,
            "url": OPEN3D_WEBRTC_URL,
            "updated_at": time.time(),
        },
    )
    command = [
        str(python_executable),
        "-m",
        "rail_curve_extractor.open3d_webrtc_viewer",
        "--input",
        str(input_path),
        "--max-points",
        str(max(0, int(request.max_points))),
        "--point-size",
        str(max(1, int(request.point_size))),
        "--progress-file",
        str(progress_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_project_src(env.get("PYTHONPATH", ""))
    env["WEBRTC_IP"] = "127.0.0.1"
    env["WEBRTC_PORT"] = "8888"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _EMBEDDED_VIEWER_LOG_HANDLE = log_path.open("w", encoding="utf-8")
    _EMBEDDED_VIEWER_PROCESS = subprocess.Popen(
        command,
        cwd=str(_project_root()),
        env=env,
        stdout=_EMBEDDED_VIEWER_LOG_HANDLE,
        stderr=subprocess.STDOUT,
    )
    status = _embedded_viewer_status()
    return {
        "started": True,
        "pid": _EMBEDDED_VIEWER_PROCESS.pid,
        "url": OPEN3D_WEBRTC_URL,
        "max_points": int(request.max_points),
        "full_density": int(request.max_points) <= 0,
        "log_path": str(log_path),
        "progress_path": str(progress_path),
        "viewer_python": str(python_executable),
        "status": status,
    }


@app.get(f"{API_PREFIX}/viewer/embedded/status")
def embedded_viewer_status() -> dict[str, Any]:
    return _embedded_viewer_status()


@app.post(f"{API_PREFIX}/viewer/embedded/stop")
def stop_embedded_viewer() -> dict[str, Any]:
    stopped = _stop_embedded_viewer_process()
    _, progress_path = _embedded_viewer_paths()
    _write_embedded_progress(
        progress_path,
        {
            "state": "idle",
            "phase": "stopped",
            "message": "Open3D 内嵌查看器已停止",
            "percent": None,
            "url": OPEN3D_WEBRTC_URL,
            "updated_at": time.time(),
        },
    )
    return {"stopped": stopped}


def _existing_path(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label}不存在：{path}")
    return path


def _preview_xy_bounds(bounds: dict[str, float] | None) -> tuple[float, float, float, float] | None:
    if not isinstance(bounds, dict):
        return None
    try:
        x_min = float(bounds["x_min"])
        x_max = float(bounds["x_max"])
        y_min = float(bounds["y_min"])
        y_max = float(bounds["y_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="预览范围 bounds 必须包含 x_min/x_max/y_min/y_max。") from exc
    if x_max <= x_min or y_max <= y_min:
        raise HTTPException(status_code=400, detail="预览范围 bounds 的最大值必须大于最小值。")
    return x_min, x_max, y_min, y_max


def _preview_response_bounds(points_xy: np.ndarray, xy_bounds: tuple[float, float, float, float] | None) -> dict[str, list[float]]:
    if xy_bounds is not None:
        x_min, x_max, y_min, y_max = xy_bounds
        return {"minimum": [x_min, y_min], "maximum": [x_max, y_max]}
    if len(points_xy) > 0:
        return {
            "minimum": points_xy.min(axis=0).astype(float).tolist(),
            "maximum": points_xy.max(axis=0).astype(float).tolist(),
        }
    return {"minimum": [0.0, 0.0], "maximum": [0.0, 0.0]}


def _dom_pipeline_paths(job_id: str) -> tuple[Path, Path]:
    runtime_dir = _project_root() / ".codex-runtime" / "dom-pipeline"
    return runtime_dir / f"{job_id}.log", runtime_dir / f"{job_id}.progress.json"


def _pipeline_python_executable() -> Path:
    return Path(sys.executable)


def _dom_pipeline_status(job_id: str) -> dict[str, Any]:
    job = _DOM_PIPELINE_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"找不到 DOM 流水线任务：{job_id}")

    process: subprocess.Popen[Any] | None = job.get("process")
    return_code = process.poll() if process is not None else job.get("return_code")
    if return_code is not None:
        job["return_code"] = return_code
        _close_job_log_handle(job)

    log_path = Path(job["log_path"])
    progress_path = Path(job["progress_path"])
    progress = _read_json_if_exists(progress_path)
    state = str(progress.get("state") or "running")
    if return_code is not None and state not in {"completed", "failed", "stopped"}:
        state = "completed" if return_code == 0 else "failed"
        progress = {
            **progress,
            "state": state,
            "message": "DOM 流水线已结束" if return_code == 0 else f"DOM 流水线进程退出，退出码 {return_code}",
        }
    return {
        "job_id": job_id,
        "state": state,
        "message": str(progress.get("message") or ""),
        "stage_name": progress.get("stage_name"),
        "stage_description": progress.get("stage_description"),
        "stage_status": progress.get("stage_status"),
        "stage_index": progress.get("stage_index"),
        "stage_count": progress.get("stage_count"),
        "percent": progress.get("percent"),
        "out_dir": str(job["output_dir"]),
        "outputs": progress.get("outputs") or {},
        "summary_path": progress.get("summary_path"),
        "latest_event": progress.get("latest_event"),
        "error": progress.get("error"),
        "pid": process.pid if process is not None else None,
        "return_code": return_code,
        "running": process is not None and return_code is None,
        "started_at": job.get("started_at"),
        "updated_at": progress.get("updated_at"),
        "log_path": str(log_path),
        "progress_path": str(progress_path),
        "command": job.get("command", []),
    }


def _close_job_log_handle(job: dict[str, Any]) -> None:
    handle = job.get("log_handle")
    if handle is not None:
        handle.close()
        job["log_handle"] = None


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{time.time_ns()}.tmp")
    temporary_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _stop_dom_pipeline_jobs() -> None:
    for job in list(_DOM_PIPELINE_JOBS.values()):
        process: subprocess.Popen[Any] | None = job.get("process")
        if process is not None and process.poll() is None:
            process.terminate()
        _close_job_log_handle(job)


def _embedded_viewer_paths() -> tuple[Path, Path]:
    runtime_dir = _project_root() / ".codex-runtime"
    return runtime_dir / "open3d-webrtc-viewer.log", runtime_dir / "open3d-webrtc-viewer-progress.json"


def _write_embedded_progress(progress_path: Path, payload: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_json_safe(payload), ensure_ascii=False)
    temporary_path = progress_path.with_name(f"{progress_path.name}.{os.getpid()}.tmp")
    for _ in range(4):
        try:
            temporary_path.write_text(line, encoding="utf-8")
            temporary_path.replace(progress_path)
            return
        except OSError:
            time.sleep(0.05)
    for _ in range(4):
        try:
            progress_path.write_text(line, encoding="utf-8")
            return
        except OSError:
            time.sleep(0.05)


def _read_embedded_progress(progress_path: Path) -> dict[str, Any]:
    if not progress_path.exists():
        return {}
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _embedded_viewer_status() -> dict[str, Any]:
    log_path, progress_path = _embedded_viewer_paths()
    progress = _read_embedded_progress(progress_path)
    process = _EMBEDDED_VIEWER_PROCESS
    pid = process.pid if process is not None else progress.get("pid")
    return_code = process.poll() if process is not None else None

    if process is None:
        state = str(progress.get("state") or "idle")
        if state not in {"idle", "stopped"}:
            progress = {
                **progress,
                "state": "idle",
                "phase": "not_running",
                "message": "Open3D 内嵌查看器未运行",
                "percent": None,
            }
    elif return_code is not None and progress.get("state") != "failed":
        progress = {
            **progress,
            "state": "failed",
            "phase": "process_exited",
            "message": f"Open3D 子进程已退出，退出码 {return_code}",
            "error": f"process exited with code {return_code}",
            "percent": None,
        }
    elif not progress:
        progress = {
            "state": "starting",
            "phase": "starting_process",
            "message": "正在启动 Open3D 子进程",
            "percent": None,
        }

    return {
        "state": str(progress.get("state") or "idle"),
        "phase": str(progress.get("phase") or ""),
        "message": str(progress.get("message") or ""),
        "current_file": progress.get("current_file"),
        "file_index": progress.get("file_index"),
        "file_count": progress.get("file_count"),
        "loaded_points": progress.get("loaded_points"),
        "total_points": progress.get("total_points"),
        "source_total_points": progress.get("source_total_points"),
        "display_points": progress.get("display_points"),
        "percent": progress.get("percent"),
        "url": str(progress.get("url") or OPEN3D_WEBRTC_URL),
        "pid": pid,
        "return_code": return_code,
        "ready": progress.get("state") == "ready" and process is not None and return_code is None,
        "error": progress.get("error"),
        "updated_at": progress.get("updated_at"),
        "log_path": str(log_path),
        "progress_path": str(progress_path),
    }


def _stop_embedded_viewer_process() -> bool:
    global _EMBEDDED_VIEWER_PROCESS, _EMBEDDED_VIEWER_LOG_HANDLE
    stopped = False
    process = _EMBEDDED_VIEWER_PROCESS
    if process is not None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            stopped = True
        _EMBEDDED_VIEWER_PROCESS = None
    if _EMBEDDED_VIEWER_LOG_HANDLE is not None:
        _EMBEDDED_VIEWER_LOG_HANDLE.close()
        _EMBEDDED_VIEWER_LOG_HANDLE = None
    return stopped


def _wait_for_embedded_viewer_ready(process: subprocess.Popen[Any], timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    media_url = f"{OPEN3D_WEBRTC_URL}/api/getMediaList"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise HTTPException(status_code=500, detail="Open3D 内嵌查看器启动失败，请查看 open3d-webrtc-viewer.log。")
        try:
            with urlopen(media_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise HTTPException(status_code=504, detail="Open3D 内嵌查看器启动超时，请稍后重试或查看日志。")


def _result_overlay(result: PipelineResult) -> dict[str, Any]:
    tracks = []
    for track_result in result.track_results:
        tracks.append(
            {
                "id": int(track_result.track_id),
                "label": _track_overlay_label(track_result.summary, track_result.track_id),
                "confidence": float(track_result.confidence),
                "source": str(track_result.summary.get("source", "analysis")),
                "centerline_xy": _sample_xy(track_result.centerline_world, max_points=1800),
                "rail_points_xy": _sample_xy(track_result.rail_points_world, max_points=3500),
            }
        )

    if not tracks and len(result.centerline_world) >= 2:
        tracks.append(
            {
                "id": 1,
                "label": "中心线",
                "confidence": float(result.summary.get("confidence", 0.0)),
                "source": str(result.summary.get("preprocessing_mode", "analysis")),
                "centerline_xy": _sample_xy(result.centerline_world, max_points=1800),
                "rail_points_xy": _sample_xy(result.rail_points_world, max_points=3500),
            }
        )

    turnouts = []
    for turnout_result in result.turnout_results:
        turnouts.append(
            {
                "id": int(turnout_result.turnout_id),
                "label": f"道岔 {turnout_result.turnout_id}",
                "confidence": float(turnout_result.confidence),
                "switch_point_xy": _sample_xy(turnout_result.switch_point_world.reshape(1, 3), max_points=1),
                "main_centerline_xy": _sample_xy(turnout_result.main_centerline_world, max_points=1200),
                "branch_centerline_xy": _sample_xy(turnout_result.branch_centerline_world, max_points=1200),
            }
        )

    return {
        "track_count": len(tracks),
        "turnout_count": len(turnouts),
        "tracks": tracks,
        "turnouts": turnouts,
        "centerline_xy": _sample_xy(result.centerline_world, max_points=2400),
        "rail_points_xy": _sample_xy(result.rail_points_world, max_points=6000),
    }


def _track_overlay_label(summary: dict[str, Any], track_id: int) -> str:
    turnout_id = summary.get("turnout_id")
    turnout_path = summary.get("turnout_path")
    if turnout_id is not None and turnout_path:
        return f"道岔 {turnout_id} {turnout_path}"
    return str(summary.get("name") or summary.get("label") or f"轨道 {track_id}")


def _sample_xy(points: np.ndarray, max_points: int) -> list[list[float]]:
    if points is None:
        return []
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] < 2 or len(point_array) == 0:
        return []
    indices = downsample_indices(len(point_array), max(1, min(int(max_points), len(point_array))))
    return np.round(point_array[indices, :2], 3).tolist()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _viewer_python_executable() -> Path:
    sidecar_python = _project_root() / ".open3d-venv" / "Scripts" / "python.exe"
    if sidecar_python.exists():
        return sidecar_python
    return Path(sys.executable)


def _pythonpath_with_project_src(existing_pythonpath: str) -> str:
    src_path = str(_project_root() / "src")
    parts = [src_path]
    if existing_pythonpath:
        parts.append(existing_pythonpath)
    return os.pathsep.join(parts)


atexit.register(_stop_embedded_viewer_process)
atexit.register(_stop_dom_pipeline_jobs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Rail Curve Extractor local backend.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

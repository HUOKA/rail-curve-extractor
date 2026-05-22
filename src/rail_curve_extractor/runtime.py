from __future__ import annotations

from pathlib import Path
import sys


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return Path(getattr(sys, "_MEIPASS")).resolve()
        exe_dir = Path(sys.executable).resolve().parent
        internal_dir = exe_dir / "_internal"
        if internal_dir.exists():
            return internal_dir
        return exe_dir
    return Path(__file__).resolve().parents[2]


def bundled_path(*parts: str) -> Path:
    return application_root().joinpath(*parts)


def default_output_dir() -> Path:
    documents = Path.home() / "Documents"
    base_dir = documents if documents.exists() else Path.home()
    return base_dir / "RailCurveExtractorOutput"

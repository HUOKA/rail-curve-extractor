from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_raw_dom_roi_fullpass.py"
    spec = importlib.util.spec_from_file_location("run_raw_dom_roi_fullpass", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RawDomRoiFullPassTest(unittest.TestCase):
    def test_build_plan_includes_export_predict_and_candidates(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "fullpass"
            args = module.build_parser().parse_args(
                [
                    "--out-dir",
                    str(out_dir),
                    "--max-tiles",
                    "2",
                    "--dry-run",
                ],
            )
            plan = module.build_plan(args, repo_root=Path(__file__).resolve().parents[1], python_exe=Path("python.exe"))

        self.assertTrue(plan["raw_dom_first"])
        self.assertEqual(len(plan["steps"]), 3)
        self.assertIn("export_raw_dom_roi_tiles", [step["name"] for step in plan["steps"]])
        self.assertIn("predict_rail_masks", [step["name"] for step in plan["steps"]])
        self.assertIn("extract_map_candidates", [step["name"] for step in plan["steps"]])
        self.assertIn("--ignore-labels", plan["steps"][2]["command"])
        self.assertEqual(plan["paths"]["out_dir"], str(out_dir))


if __name__ == "__main__":
    unittest.main()

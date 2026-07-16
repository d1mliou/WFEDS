"""Tests for scripts/cell2fire/run_scenario.py.

Covers `_wsl_path` (pure path conversion), `_run` (subprocess wrapper + tail-of-
output error message), `run_engine` (WSL command construction + the
InitialBurned flag + the "no grids produced" check), and a fully-mocked
happy-path test of the top-level `run_scenario()` orchestrator.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import run_scenario
from run_scenario import _run, _wsl_path, run_engine, run_scenario as run_scenario_fn


class TestWslPath:
    def test_converts_a_realistic_windows_path(self):
        result = _wsl_path(r"C:\Users\someone\Desktop\LLM_WFEDS\tests")
        assert result == "/mnt/c/Users/someone/Desktop/LLM_WFEDS/tests"

    def test_lowercases_the_drive_letter(self):
        result = _wsl_path(r"D:\Data\Fire")
        assert result.startswith("/mnt/d/")

    def test_preserves_spaces_in_path_segments(self):
        """run_scenario.py's own docstring flags spaces (e.g. an OneDrive path) as
        a known WSL footgun elsewhere in the pipeline -- `_wsl_path` itself must
        not silently mangle/strip them."""
        result = _wsl_path(r"D:\Some Folder\With Spaces\file.txt")
        assert result == "/mnt/d/Some Folder/With Spaces/file.txt"

    def test_uses_forward_slashes_throughout(self):
        result = _wsl_path(r"C:\a\b\c")
        assert "\\" not in result


# --------------------------------------------------------------------------------
# _run
# --------------------------------------------------------------------------------
class TestRun:
    def test_nonzero_exit_raises_with_tail_of_combined_output(self, monkeypatch):
        """`_run` keeps only the LAST 15 lines of (stdout + stderr) combined, per
        `(r.stdout + "\\n" + r.stderr).strip().splitlines()[-15:]`. Build >15
        distinct lines so early ones are provably truncated away."""
        stdout = "\n".join(f"outline{i}" for i in range(15))   # 15 lines
        stderr = "\n".join(f"errline{i}" for i in range(10))   # 10 lines -> 25 combined
        monkeypatch.setattr(run_scenario.subprocess, "run", Mock(
            return_value=subprocess.CompletedProcess(
                args=["dummy"], returncode=3, stdout=stdout, stderr=stderr)))

        with pytest.raises(RuntimeError) as excinfo:
            _run(["dummy", "cmd"], "test stage")

        msg = str(excinfo.value)
        assert "test stage" in msg
        assert "exit 3" in msg
        assert "errline9" in msg          # last stderr line -> definitely kept
        assert "outline0" not in msg      # earliest line -> truncated away (not even
                                           # as a substring of a later line: no other
                                           # line in this fixture contains "outline0")

    def test_zero_exit_returns_stdout_and_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(run_scenario.subprocess, "run", Mock(
            return_value=subprocess.CompletedProcess(
                args=["dummy"], returncode=0, stdout="all good", stderr="")))

        out = _run(["dummy"], "ok stage")

        assert out == "all good"


# --------------------------------------------------------------------------------
# run_engine
# --------------------------------------------------------------------------------
class TestRunEngine:
    def _seed_grids(self, tmp_path, n=1):
        grids_dir = tmp_path / "engine" / "Grids" / "Grids1"
        grids_dir.mkdir(parents=True)
        for i in range(n):
            (grids_dir / f"ForestGrid{i}.csv").write_text("dummy")

    def test_has_front_true_includes_initial_burned_flag(self, tmp_path, monkeypatch):
        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()
        self._seed_grids(tmp_path)
        mock_run = Mock(return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""))
        monkeypatch.setattr(run_scenario.subprocess, "run", mock_run)

        run_engine(instance_dir, "run1", horizon_h=6, has_front=True)

        cmd_string = mock_run.call_args[0][0][-1]
        assert "--InitialBurned" in cmd_string

    def test_has_front_false_omits_initial_burned_flag(self, tmp_path, monkeypatch):
        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()
        self._seed_grids(tmp_path)
        mock_run = Mock(return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""))
        monkeypatch.setattr(run_scenario.subprocess, "run", mock_run)

        run_engine(instance_dir, "run1", horizon_h=6, has_front=False)

        cmd_string = mock_run.call_args[0][0][-1]
        assert "--InitialBurned" not in cmd_string

    def test_no_grids_produced_raises_even_on_exit_code_0(self, tmp_path, monkeypatch):
        """The grid check runs unconditionally after `_run` returns (it is NOT
        gated on the subprocess exit code -- `_run` itself already raises on a
        nonzero exit before this check would ever run in a real failure). This
        test specifically covers "exit 0 but still no grids", proving the check
        is independent of the mocked subprocess's reported success."""
        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()
        # deliberately do NOT seed any grid files
        monkeypatch.setattr(run_scenario.subprocess, "run", Mock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr="")))

        with pytest.raises(RuntimeError, match="no grids"):
            run_engine(instance_dir, "run1", horizon_h=6, has_front=False)


# --------------------------------------------------------------------------------
# run_scenario() -- top-level happy path, everything mocked
# --------------------------------------------------------------------------------
class TestRunScenarioHappyPath:
    def test_full_chain_with_every_stage_mocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_scenario, "RUNS_DIR", tmp_path / "runs")

        params = {"horizon_h": 6, "front_cells": 0, "mode": "point_ignition",
                  "window_km": 12.0, "start_time_utc": "2024-08-10T12:00",
                  "grid": {"nrows": 10, "ncols": 10, "cell_m": 30},
                  "ignition_points_wgs84": [(38.9, 23.1)], "wind_dir_source": "era5",
                  "fwi_spinup_start": "2024-06-01", "n_fronts_detected": None,
                  "scenario": None}
        monkeypatch.setattr(run_scenario, "build_instance", Mock(return_value=params))
        monkeypatch.setattr(run_scenario, "run_engine",
                            Mock(return_value=tmp_path / "runs" / "test_run_001" / "engine"))
        monkeypatch.setattr(run_scenario.subprocess, "run", Mock(
            return_value=subprocess.CompletedProcess(args=[], returncode=0,
                                                      stdout="", stderr="")))

        run_dir = tmp_path / "runs" / "test_run_001"
        run_dir.mkdir(parents=True)
        summary = [
            {"period": 0, "time_utc": "2024-08-10T12:00:00", "fire_km2": 0.5,
             "at_risk": 1, "population": 40, "routed": 1, "cut_off": 0, "impacted": 0,
             "edges_removed": 1, "longest_route_km": 1.1},
            {"period": 1, "time_utc": "2024-08-10T13:00:00", "fire_km2": 1.0,
             "at_risk": 2, "population": 100, "routed": 1, "cut_off": 1, "impacted": 0,
             "edges_removed": 3, "longest_route_km": 2.5},
        ]
        (run_dir / "timestep_summary.json").write_text(
            json.dumps(summary), encoding="utf-8")
        (run_dir / "isochrones.geojson").write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {"period": 1, "pct4": 42},
                          "geometry": {"type": "Polygon",
                                       "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}],
        }), encoding="utf-8")

        result = run_scenario_fn(ignition_points=[(38.9, 23.1)], window_km=12,
                                 horizon_h=6, start_time="2024-08-10T12:00",
                                 run_id="test_run_001")

        for key in ("run_id", "run_dir", "inputs", "hours_simulated", "final_hour",
                    "final_front_class4_pct", "per_hour", "files", "elapsed_min"):
            assert key in result
        assert result["run_id"] == "test_run_001"
        assert result["final_hour"] == summary[-1]
        assert result["hours_simulated"] == len(summary) - 1
        assert result["final_front_class4_pct"] == 42
        assert result["inputs"] == params
        # the returned (in-memory) dict keeps absolute paths - agent.py and
        # docker/api.py need them to locate/serve the actual files
        assert result["run_dir"] == str(run_dir)
        assert result["files"]["map_png"] == str(run_dir / "map.png")

        result_json = run_dir / "result.json"
        assert result_json.exists()
        on_disk = json.loads(result_json.read_text(encoding="utf-8"))
        assert on_disk["run_id"] == "test_run_001"
        # ... but the on-disk copy is portable: no absolute local/container
        # filesystem paths, so the run folder can be moved/copied/zipped
        assert "run_dir" not in on_disk
        assert on_disk["files"] == {
            "map_png": "map.png", "map_anim": "map.mp4",
            "dashboard": "fire_timesteps.html",
            "perimeters": "perimeters.geojson",
            "evacuation": "timestep_evacuation.gpkg"}

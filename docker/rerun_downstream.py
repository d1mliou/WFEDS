"""Re-run only stages 3-6 of run_scenario() (adapter -> evacuation -> dashboard
-> result.json) against an EXISTING run_dir that already has instance/ and
engine/ populated - skips the expensive engine invocation entirely.

Purpose: iterate on the Python-side pipeline (or, as here, re-verify after
fixing a Python dependency version) without re-running the ~15 min Cell2Fire
engine when its output is already known-good and unchanged. This is an
accessory verification helper, not the primary container path - the default
entrypoint (docker/run_container.py) still runs the whole chain, engine
included, exactly as run_scenario.py does.

Mirrors run_scenario.py's stages 3-6 exactly (same subprocess calls, same
env vars) rather than importing/reusing them, because run_scenario() does
not expose those stages independently - it is one function. Kept deliberately
small and scoped to this one use case.

Usage (inside the container):
    python docker/rerun_downstream.py <run_dir>
"""

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts" / "cell2fire"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SCRIPTS))

# Reuse run_scenario's own subprocess-running helper + env builder verbatim -
# only the orchestration sequence (which stages run) is re-specified here.
from run_scenario import SCRIPTS_DIR, _py_env, _run  # noqa: E402


def rerun_downstream(run_dir, label=None):
    run_dir = Path(run_dir)
    instance_dir = run_dir / "instance"
    engine_dir = run_dir / "engine"
    if not (engine_dir / "Grids" / "Grids1").exists():
        sys.exit(f"ERROR: {engine_dir} has no engine output - nothing to reprocess")
    is_reference = (run_dir / "instance" / "scenario_params.json").exists()

    _run([sys.executable, str(SCRIPTS_DIR / "cell2fire_adapter.py"),
          str(instance_dir), str(engine_dir), "EPSG:2100"],
         "adapter (perimeters + isochrones)", env=_py_env(run_dir, label, is_reference))
    for f in ("perimeters.geojson", "isochrones.geojson"):
        src = engine_dir / f
        if src.exists():
            src.replace(run_dir / f)

    _run([sys.executable, str(SCRIPTS_DIR / "fire_timesteps.py")],
         "per-hour evacuation", env=_py_env(run_dir, label, is_reference))

    _run([sys.executable, str(SCRIPTS_DIR / "visualize_fire.py")],
         "dashboard", env=_py_env(run_dir, label, is_reference))
    _run([sys.executable, str(SCRIPTS_DIR / "snapshot_map.py"), str(run_dir)],
         "map snapshot (PNG)", env=_py_env(run_dir, label, is_reference))

    summary = json.loads((run_dir / "timestep_summary.json").read_text(encoding="utf-8"))
    final = summary[-1] if summary else {}
    pct4 = None
    iso_path = run_dir / "isochrones.geojson"
    if iso_path.exists():
        try:
            feats = json.loads(iso_path.read_text(encoding="utf-8"))["features"]
            if feats:
                last = max(feats, key=lambda f: f["properties"]["period"])
                pct4 = round(float(last["properties"].get("pct4", 0)))
        except Exception:
            pass
    inputs = json.loads((instance_dir / "scenario_params.json").read_text(encoding="utf-8")) \
        if (instance_dir / "scenario_params.json").exists() else {}
    result = {
        "run_id": run_dir.name, "run_dir": str(run_dir), "inputs": inputs,
        "hours_simulated": len(summary) - 1 if summary else 0,
        "final_hour": final, "final_front_class4_pct": pct4, "per_hour": summary,
        "files": {"map_png": str(run_dir / "map.png"),
                  "map_anim": str(run_dir / "map.mp4"),
                  "dashboard": str(run_dir / "fire_timesteps.html"),
                  "perimeters": str(run_dir / "perimeters.geojson"),
                  "evacuation": str(run_dir / "timestep_evacuation.gpkg")},
        "elapsed_min": None,   # not a fresh full run - not meaningful here
    }
    # Same portability rule as run_scenario.py: no absolute paths on disk.
    on_disk = {k: v for k, v in result.items() if k != "run_dir"}
    on_disk["files"] = {k: Path(v).name for k, v in result["files"].items()}
    (run_dir / "result.json").write_text(
        json.dumps(on_disk, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Reprocessed {run_dir} (engine output reused, unchanged)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <run_dir>")
    rerun_downstream(sys.argv[1])

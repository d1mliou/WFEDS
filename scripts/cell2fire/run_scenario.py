"""Phase 5 Stage 2: run_scenario() - the WHOLE deterministic chain as ONE call.

This is the tool the LLM agent invokes. One call does everything that used to be
five manual steps:

    build_instance  ->  Cell2Fire engine (WSL, automatic)  ->  adapter
    (perimeters + isochrones)  ->  per-hour evacuation  ->  dashboard  ->  result

Every run is fully isolated in its own folder (nothing canonical is touched):

    DATA_DIR/Fire/cell2fire/runs/<run_id>/
        instance/               the Cell2Fire inputs (+ scenario_params.json)
        engine/                 Grids/ + LogFile copied back from WSL
        perimeters.geojson      hourly fire perimeters
        isochrones.geojson      outer front + suppressability class per hour
        timestep_evacuation.gpkg / timestep_summary.json
        fire_timesteps.html     the dashboard
        result.json             the LLM-facing summary (params + per-hour stats)

The engine runs in WSL with a WSL-LOCAL output folder (paths with spaces - e.g.
OneDrive - silently break the engine's unquoted mkdir; see [[Cell2Fire]]), then
the grids are copied back.

Usage (same inputs as build_instance):
    from run_scenario import run_scenario
    result = run_scenario(ignition_points=[(38.90, 23.12)], window_km=12,
                          horizon_h=6, start_time="2025-08-10T10:00")
    result = run_scenario(scenario="north_evia_2021")

CLI:
    python scripts/cell2fire/run_scenario.py --points "38.90,23.12" \
        --window-km 12 --horizon 6 --start 2025-08-10T10:00 [--label "..."]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _paths import DATA_DIR
from build_cell2fire_instance import (InstanceInputError, SCENARIOS,
                                      build_instance)

RUNS_DIR = DATA_DIR / "Fire" / "cell2fire" / "runs"
WSL_REPO = "$HOME/Cell2Fire"                # the engine clone inside WSL (expanded by bash)
WSL_RUNS = "$HOME/wfeds_runs"               # WSL-local engine outputs (no spaces!)
SCRIPTS_DIR = Path(__file__).parent


def _wsl_path(win_path):
    """C:\\Users\\... -> /mnt/c/Users/... (for reading the instance from WSL)."""
    p = Path(win_path).resolve()
    drive = p.drive.rstrip(":").lower()
    rest = "/".join(p.parts[1:])
    return f"/mnt/{drive}/{rest}"


def _run(cmd, desc, env=None, timeout=3600):
    """Run a subprocess, stream-quietly; raise with the tail of output on failure."""
    print(f"[{time.strftime('%H:%M:%S')}] {desc} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=timeout, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"{desc} failed (exit {r.returncode}):\n{tail}")
    return r.stdout


def run_engine(instance_dir, run_id, horizon_h, has_front):
    """Run Cell2Fire in WSL on the given instance; return the local engine dir."""
    wsl_out = f"{WSL_RUNS}/{run_id}"
    inst = _wsl_path(instance_dir)
    flags = ("--gen-data --ignitions --nsims 1 --sim-years 1 --weather rows "
             "--nweathers 1 --Fire-Period-Length 1 --gridsStep 60 --grids "
             "--finalGrid --ROS-CV 0.0 --seed 123 --out-intensity")
    if has_front:
        flags += f" --InitialBurned '{inst}/InitialBurned.csv'"
    engine_dir = Path(instance_dir).parent / "engine"
    # NB: wsl_out contains $HOME -> double quotes (bash expands), NOT single;
    # the /mnt/c instance/engine paths stay single-quoted (literal, may have spaces).
    cmd = (f'cd {WSL_REPO} && source .venv/bin/activate && '
           f'rm -rf "{wsl_out}" && mkdir -p "{wsl_out}" && '
           f"python -m cell2fire.main --input-instance-folder '{inst}/' "
           f'--output-folder "{wsl_out}" {flags} && '
           f"mkdir -p '{_wsl_path(engine_dir)}' && "
           f'cp -r "{wsl_out}/Grids" "{wsl_out}/LogFile.txt" '
           f"'{_wsl_path(engine_dir)}/' 2>/dev/null; "
           f'rm -rf "{wsl_out}"')
    _run(["wsl.exe", "-e", "bash", "-lc", cmd],
         f"Cell2Fire engine ({horizon_h} h, this can take several minutes)")
    grids = list((engine_dir / "Grids" / "Grids1").glob("ForestGrid*.csv"))
    if not grids:
        raise RuntimeError("Engine produced no grids - check engine/LogFile.txt")
    print(f"   engine done: {len(grids)} hourly grids")
    return engine_dir


def _py_env(run_dir, label, real_overlay):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["WFEDS_SCENARIO_DIR"] = str(run_dir)
    env["WFEDS_SCENARIO_LABEL"] = label or ""
    env["WFEDS_REAL_OVERLAY"] = "1" if real_overlay else "0"
    return env


def run_scenario(ignition_points=None, window_km=None, horizon_h=24,
                 start_time=None, scenario=None, label=None, run_id=None):
    """The one-call deterministic chain. Returns the LLM-facing result dict.

    Raises InstanceInputError (user-facing message) on bad inputs; RuntimeError
    with a log tail if a pipeline stage fails."""
    t_start = time.time()
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{scenario or ('front' if ignition_points and len(ignition_points) >= 2 else 'point')}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    is_reference = scenario is not None

    # 1. instance
    params = build_instance(ignition_points=ignition_points, window_km=window_km,
                            horizon_h=horizon_h, start_time=start_time,
                            out_dir=run_dir / "instance", scenario=scenario)

    # 2. engine (WSL)
    engine_dir = run_engine(run_dir / "instance", run_id,
                            params["horizon_h"], params["front_cells"] > 0)

    # 3. adapter: grids -> perimeters + isochrones (into engine/, then move up)
    _run([sys.executable, str(SCRIPTS_DIR / "cell2fire_adapter.py"),
          str(run_dir / "instance"), str(engine_dir), "EPSG:2100"],
         "adapter (perimeters + isochrones)", env=_py_env(run_dir, label, is_reference))
    for f in ("perimeters.geojson", "isochrones.geojson"):
        src = engine_dir / f
        if src.exists():
            src.replace(run_dir / f)

    # 4. per-hour evacuation (the thesis core)
    _run([sys.executable, str(SCRIPTS_DIR / "fire_timesteps.py")],
         "per-hour evacuation", env=_py_env(run_dir, label, is_reference))

    # 5. dashboard (HTML for desktop) + static map snapshot (PNG for chat/mobile)
    _run([sys.executable, str(SCRIPTS_DIR / "visualize_fire.py")],
         "dashboard", env=_py_env(run_dir, label, is_reference))
    _run([sys.executable, str(SCRIPTS_DIR / "snapshot_map.py"), str(run_dir)],
         "map snapshot (PNG)", env=_py_env(run_dir, label, is_reference))

    # 6. the LLM-facing result
    summary = json.loads((run_dir / "timestep_summary.json").read_text(encoding="utf-8"))
    final = summary[-1] if summary else {}
    # fightability of the final front (exterior-only class-4 %), if intensity ran
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
    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "inputs": params,
        "hours_simulated": len(summary) - 1 if summary else 0,
        "final_hour": final,          # fire_km2, at_risk, routed/cut_off/impacted...
        "final_front_class4_pct": pct4,   # % of the final front above 3,500 kW/m (indirect attack only)
        "per_hour": summary,
        "files": {
            "map_png": str(run_dir / "map.png"),
            "map_anim": str(run_dir / "map.mp4"),
            "dashboard": str(run_dir / "fire_timesteps.html"),
            "perimeters": str(run_dir / "perimeters.geojson"),
            "evacuation": str(run_dir / "timestep_evacuation.gpkg"),
        },
        "elapsed_min": round((time.time() - t_start) / 60, 1),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRun {run_id} done in {result['elapsed_min']} min -> {run_dir}")
    if final:
        print(f"Final hour: fire {final.get('fire_km2')} km², "
              f"{final.get('at_risk')} settlements at risk "
              f"({final.get('routed')} routed / {final.get('cut_off')} cut off / "
              f"{final.get('impacted')} impacted)")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=None, help=f"named scenario {sorted(SCENARIOS)}")
    ap.add_argument("--points", default=None,
                    help='ignition "lat,lon[;lat,lon...]" (1 = point, >=2 = front)')
    ap.add_argument("--window-km", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--start", default=None, help='"YYYY-MM-DDTHH:MM" UTC')
    ap.add_argument("--label", default=None, help="banner label on the dashboard")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()

    pts = None
    if a.points:
        pts = [tuple(float(x) for x in p.split(","))
               for p in a.points.split(";") if p.strip()]
    try:
        run_scenario(ignition_points=pts, window_km=a.window_km, horizon_h=a.horizon,
                     start_time=a.start, scenario=a.scenario, label=a.label,
                     run_id=a.run_id)
    except InstanceInputError as e:
        raise SystemExit(f"INPUT ERROR: {e}")


if __name__ == "__main__":
    main()

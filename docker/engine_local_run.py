"""Local (in-container) replacement for run_scenario.run_engine().

Same contract as the original (scripts/cell2fire/run_scenario.py):

    run_engine_docker(instance_dir, run_id, horizon_h, has_front) -> engine_dir

and the same observable behaviour - it must leave
``engine_dir/Grids/Grids1/ForestGrid*.csv`` (+ ``Intensity*.csv``) and
``engine_dir/LogFile.txt`` behind, and NOTHING else. In particular the
engine's ``Messages/`` output is deliberately NOT copied back, because the
current WSL path never copies it either (cell2fire_adapter.py already
handles its absence by skipping the ROS grid). Matching today's output
means reproducing that omission, not fixing it.

The only real difference from the WSL original: no ``wsl.exe`` bridge and no
``C:\\ -> /mnt/c`` path translation - everything is a native Linux path
inside the container. The engine still writes to a container-local scratch
dir first (mirroring the WSL-local-output rationale: the mounted data path
is under our control here, but keeping the engine's own output off the
mount avoids partial-run debris in the user's data folder on failure).

Env overrides (all optional - the defaults match the Dockerfile layout):
    WFEDS_ENGINE_REPO   engine clone            (default /opt/Cell2Fire)
    WFEDS_ENGINE_PY     engine venv python      (default /opt/venv-engine/bin/python)
    WFEDS_ENGINE_RUNS   scratch output root     (default /tmp/wfeds_runs)
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

ENGINE_REPO = Path(os.environ.get("WFEDS_ENGINE_REPO", "/opt/Cell2Fire"))
ENGINE_PY = Path(os.environ.get("WFEDS_ENGINE_PY", "/opt/venv-engine/bin/python"))
ENGINE_RUNS = Path(os.environ.get("WFEDS_ENGINE_RUNS", "/tmp/wfeds_runs"))

# Exactly the flag set run_scenario.run_engine() passes today - do not drift.
ENGINE_FLAGS = ["--gen-data", "--ignitions", "--nsims", "1", "--sim-years", "1",
                "--weather", "rows", "--nweathers", "1",
                "--Fire-Period-Length", "1", "--gridsStep", "60",
                "--grids", "--finalGrid", "--ROS-CV", "0.0",
                "--seed", "123", "--out-intensity"]


def run_engine_docker(instance_dir, run_id, horizon_h, has_front):
    """Run the container-local Cell2Fire on the instance; return the engine dir."""
    instance_dir = Path(instance_dir)
    scratch = ENGINE_RUNS / run_id
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)

    cmd = [str(ENGINE_PY), "-m", "cell2fire.main",
           "--input-instance-folder", f"{instance_dir}/",
           "--output-folder", str(scratch)] + ENGINE_FLAGS
    if has_front:
        cmd += ["--InitialBurned", str(instance_dir / "InitialBurned.csv")]

    print(f"[{time.strftime('%H:%M:%S')}] Cell2Fire engine "
          f"({horizon_h} h, this can take several minutes) ...", flush=True)
    r = subprocess.run(cmd, cwd=str(ENGINE_REPO), capture_output=True,
                       text=True, timeout=3600, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-15:])
        shutil.rmtree(scratch, ignore_errors=True)
        raise RuntimeError(f"Cell2Fire engine failed (exit {r.returncode}):\n{tail}")

    # Copy back ONLY what the WSL path copies back: Grids/ + LogFile.txt.
    engine_dir = instance_dir.parent / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    if (scratch / "Grids").exists():
        shutil.copytree(scratch / "Grids", engine_dir / "Grids", dirs_exist_ok=True)
    if (scratch / "LogFile.txt").exists():
        shutil.copy2(scratch / "LogFile.txt", engine_dir / "LogFile.txt")
    shutil.rmtree(scratch, ignore_errors=True)

    grids = list((engine_dir / "Grids" / "Grids1").glob("ForestGrid*.csv"))
    if not grids:
        raise RuntimeError("Engine produced no grids - check engine/LogFile.txt")
    print(f"   engine done: {len(grids)} hourly grids")
    return engine_dir

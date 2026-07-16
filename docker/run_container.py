"""Container entrypoint: the UNMODIFIED run_scenario pipeline, engine swapped.

Imports the existing scripts/cell2fire/run_scenario.py exactly as-is and
monkey-patches ONE module-level name - ``run_engine`` - before delegating to
its own ``main()``. Every other stage (build_instance, adapter, per-hour
evacuation, dashboard, snapshot, result.json) runs byte-for-byte the same
code as the WSL path on the host. The patch works because run_scenario()
looks ``run_engine`` up as a module global at call time (late binding) -
the same mechanism the repo's own test suite uses
(tests/cell2fire/test_run_scenario.py: monkeypatch.setattr(run_scenario,
"run_engine", ...)).

CLI: identical to run_scenario.py's own (we call its main() directly):
    --scenario / --points / --window-km / --horizon / --start / --label / --run-id

Requires WFEDS_DATA_DIR to point at the mounted data folder (the Dockerfile
documents the -v/-e pair); _paths.py raises a clear error otherwise.
"""

import sys
from pathlib import Path

# Same manual bootstrap convention as scripts/agent/agent.py and conftest.py -
# there is no installed package; siblings are imported by path.
_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts" / "cell2fire"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SCRIPTS))

import run_scenario                                   # noqa: E402  (the real module, untouched)
from engine_local_run import run_engine_docker        # noqa: E402

# The one and only intervention: local engine instead of the WSL bridge.
run_scenario.run_engine = run_engine_docker

if __name__ == "__main__":
    run_scenario.main()

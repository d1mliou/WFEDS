"""Tests for the WFEDS_DATA_DIR resolution logic in `_paths.py`.

`scripts/cell2fire/_paths.py` and `scripts/data_prep/_paths.py` are two files with
IDENTICAL content (verified below, and intentionally kept in lockstep -- see the
module docstrings). Per the root conftest.py, `scripts/cell2fire` is inserted onto
sys.path before `scripts/data_prep`, so a plain `import _paths` anywhere in this
suite resolves to the cell2fire copy. Because the two files are byte-identical,
exercising `get_data_dir()` through that one imported module fully covers the
resolution *logic* for both copies; `test_cell2fire_and_data_prep_paths_py_are_byte_identical`
below is the regression canary that keeps that equivalence true.

`get_data_dir()` itself is a plain function with NO side effects (it only reads
os.environ and Path.exists()) -- only the module-level `DATA_DIR = get_data_dir()`
assignment (which ran once, at import time, back when the root conftest.py's
sys.path setup let the very first `import _paths` succeed) has a side effect. So
most tests below call `get_data_dir()` directly, repeatedly, with different
monkeypatched env vars, instead of re-importing the module.
"""

import importlib
import os
from pathlib import Path

import pytest

import _paths

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES_DATA_DIR = _REPO_ROOT / "tests" / "fixtures" / "data"


@pytest.fixture(autouse=True)
def _clean_data_dir_env(monkeypatch):
    """Every test in this module starts from a blank slate: none of the three
    resolution env vars set. Real machines (this one included) have OneDrive /
    OneDriveCommercial set in the ambient environment, and the root conftest.py
    sets WFEDS_DATA_DIR for the whole pytest session -- both would otherwise leak
    into these tests and hide bugs in the resolution order."""
    monkeypatch.delenv("WFEDS_DATA_DIR", raising=False)
    monkeypatch.delenv("OneDriveCommercial", raising=False)
    monkeypatch.delenv("OneDrive", raising=False)


def test_wfeds_data_dir_override_wins_over_onedrive_vars(tmp_path, monkeypatch):
    real_dir = tmp_path / "explicit_override"
    real_dir.mkdir()
    other_dir = tmp_path / "onedrive_root"
    other_dir.mkdir()

    monkeypatch.setenv("WFEDS_DATA_DIR", str(real_dir))
    monkeypatch.setenv("OneDriveCommercial", str(other_dir))
    monkeypatch.setenv("OneDrive", str(other_dir))

    result = _paths.get_data_dir()

    assert result == real_dir


def test_missing_all_env_vars_raises_runtime_error():
    with pytest.raises(RuntimeError) as excinfo:
        _paths.get_data_dir()
    msg = str(excinfo.value)
    assert "WFEDS_DATA_DIR" in msg
    assert "OneDriveCommercial" in msg
    assert "OneDrive" in msg


def test_onedrivecommercial_appends_data_rel_tail(tmp_path, monkeypatch):
    root = tmp_path / "OneDriveRoot"
    expected = root / "Working progress" / "Scripts" / "Data"
    expected.mkdir(parents=True)

    monkeypatch.setenv("OneDriveCommercial", str(root))

    result = _paths.get_data_dir()

    assert result == expected


def test_onedrive_generic_used_when_onedrivecommercial_absent(tmp_path, monkeypatch):
    root = tmp_path / "GenericOneDrive"
    expected = root / "Working progress" / "Scripts" / "Data"
    expected.mkdir(parents=True)

    # OneDriveCommercial deliberately left unset by the autouse fixture.
    monkeypatch.setenv("OneDrive", str(root))

    result = _paths.get_data_dir()

    assert result == expected


def test_nonexistent_resolved_path_raises_with_actionable_message(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist_on_disk"
    assert not missing.exists()
    monkeypatch.setenv("WFEDS_DATA_DIR", str(missing))

    with pytest.raises(RuntimeError) as excinfo:
        _paths.get_data_dir()

    msg = str(excinfo.value)
    assert "does not exist" in msg
    assert str(missing) in msg


def test_cell2fire_and_data_prep_paths_py_are_byte_identical():
    """Regression canary for the intentional duplication (two files, same module
    name `_paths`, kept manually in sync -- see both modules' docstrings and the
    root conftest.py's sys.path-ordering comment)."""
    cell2fire_src = (_REPO_ROOT / "scripts" / "cell2fire" / "_paths.py").read_bytes()
    data_prep_src = (_REPO_ROOT / "scripts" / "data_prep" / "_paths.py").read_bytes()

    assert cell2fire_src == data_prep_src


def test_get_data_dir_has_no_import_time_side_effect_when_called_directly():
    """Sanity check on the premise the whole module docstring relies on: calling
    get_data_dir() here does NOT reassign the already-resolved `_paths.DATA_DIR`
    (that assignment only ever happened once, at the original import)."""
    before = _paths.DATA_DIR
    try:
        _paths.get_data_dir()   # will raise (no env vars set) -- that's fine, we
    except RuntimeError:        # only care that DATA_DIR itself is untouched.
        pass
    assert _paths.DATA_DIR == before


def test_importlib_reload_reflects_new_env(tmp_path, monkeypatch):
    """Belt-and-braces: an actual `importlib.reload` (re-running the module's
    top-level code, including `DATA_DIR = get_data_dir()`) picks up a changed
    WFEDS_DATA_DIR too -- demonstrating the two ways of testing this module
    (direct function call vs. full reload) agree.

    Reload mutates the SAME module object cached in sys.modules, so we restore
    WFEDS_DATA_DIR and reload again in a `finally` -- otherwise every other module
    that did `import _paths` (not `from _paths import X`) and reads `_paths.DATA_DIR`
    at call time, rather than import time, would see this test's tmp_path after it's
    gone. (Modules that did `from _paths import DATA_DIR` bind their own copy of the
    name at their own import time and are unaffected either way -- see tests/conftest.py.)
    """
    real_dir = tmp_path / "reload_target"
    real_dir.mkdir()
    monkeypatch.setenv("WFEDS_DATA_DIR", str(real_dir))

    try:
        reloaded = importlib.reload(_paths)
        assert reloaded.DATA_DIR == real_dir
    finally:
        monkeypatch.setenv("WFEDS_DATA_DIR", str(_FIXTURES_DATA_DIR))
        importlib.reload(_paths)
        assert _paths.DATA_DIR == _FIXTURES_DATA_DIR

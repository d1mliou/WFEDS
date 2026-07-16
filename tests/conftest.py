"""Shared test fixtures for the whole tests/ tree.

Marker registration lives in pytest.ini, so this file stays minimal.

CRITICAL for anyone monkeypatching a data path (FUEL / SLOPE / DEM / ATRISK_SETTLEMENTS
/ OSM_SETTLEMENTS / ROAD_GRAPH / STUDY_AREA / DATA_DIR, all defined once in
`scripts/cell2fire/_paths.py` == `scripts/data_prep/_paths.py`):

    `from _paths import FUEL` (as done in build_cell2fire_instance.py,
    route_with_fire.py, etc.) binds a COPY of the name into the IMPORTING module's
    own namespace -- it is a separate reference, not an alias back to `_paths.FUEL`.
    So per-test mocking of a data path must ALWAYS target the CONSUMING module, e.g.:

        monkeypatch.setattr(route_with_fire, "FUEL", tmp_tif_path)

    NEVER the source module:

        monkeypatch.setattr(_paths, "FUEL", tmp_tif_path)   # no effect whatsoever on
                                                              # already-imported consumers

This bit no fixture here today, but every later test file that touches a data path
needs to remember it.
"""

"""Tests for scripts/cell2fire/fire_timesteps.py.

`fire_timesteps.py` reads `WFEDS_SCENARIO_DIR` from the environment AT IMPORT
TIME (module-level `_SCEN = os.environ.get(...)`, which builds the module-level
`PERIMS`/`OUT_GPKG`/`OUT_JSON`/`WEATHER` path constants). Since the module is
already imported once at collection time, every test here sets the env var via
`monkeypatch.setenv` THEN `importlib.reload(fire_timesteps)` to get those
constants pointing at a fresh tmp scenario dir -- and note reload also rebinds
`fire_timesteps.load_atrisk` / `.buffer_width_grid` / `.load_graph` /
`.route_to_refuges` back to their ORIGINAL values, so monkeypatching those must
happen AFTER the reload, not before.

`route_to_refuges` is mocked entirely in every test here (it pulls in OSM
refuges from disk + osmnx friction-graph routing, well outside this module's
own responsibility). `load_atrisk` / `buffer_width_grid` / `load_graph` are
monkeypatched to cheap-but-real tiny synthetic objects so the REAL
`exposure_for_fire` (pure raster/sindex math, no I/O) can run unmocked on top
of them.
"""

import importlib
import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString

import fire_timesteps
from tests.helpers.geodata_factories import make_tiny_points_gdf, make_tiny_polygon_gdf

N_PERIODS = 4


def _write_scenario(scenario_dir):
    """A tiny synthetic scenario: 4 perimeter periods (small squares growing
    around (1000, 1000), EPSG:2100) + a minimal Weather.csv (only its first
    `datetime` row is ever read, by `main()`, for t0)."""
    rows = []
    for p in range(N_PERIODS):
        half = (100.0 + p * 20.0) / 2
        cx, cy = 1000.0, 1000.0
        ring = [(cx - half, cy - half), (cx + half, cy - half),
                (cx + half, cy + half), (cx - half, cy + half)]
        rows.append({"geometry": ring, "period": p})
    gdf = make_tiny_polygon_gdf(rows, crs="EPSG:2100")
    scenario_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(scenario_dir / "perimeters.geojson", driver="GeoJSON")

    instance_dir = scenario_dir / "instance"
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / "Weather.csv").write_text(
        "datetime\n2021-08-05 00:00:00\n", encoding="utf-8")


def _reload_with_scenario_dir(monkeypatch, scenario_dir):
    monkeypatch.setenv("WFEDS_SCENARIO_DIR", str(scenario_dir))
    importlib.reload(fire_timesteps)
    return fire_timesteps


def _fake_atrisk():
    return make_tiny_points_gdf([
        {"geometry": (1000.0, 1000.0), "NAME_OIK": "A", "census2021": 50},
        {"geometry": (1500.0, 1500.0), "NAME_OIK": "B", "census2021": 30},
    ], crs="EPSG:2100")


def _fake_buffer_width_grid():
    width = np.full((20, 20), 150.0)
    grid = {"transform": from_origin(0, 2000, 100, 100), "shape": (20, 20), "res": 100.0}
    return width, grid


def _fake_evac_result(period):
    """A minimal-but-valid EvacResult-shaped object: `main()` only ever reads
    `.in_zone`, `.routes`, `.n_removed` off whatever `route_to_refuges` returns."""
    in_zone = make_tiny_points_gdf([
        {"geometry": (1000.0, 1000.0), "NAME_OIK": "A", "census2021": 50, "status": "ok"},
        {"geometry": (1010.0, 1010.0), "NAME_OIK": "B", "census2021": 20, "status": "cut_off"},
    ], crs="EPSG:2100")
    routes = gpd.GeoDataFrame(
        {"origin": ["A"], "length_m": [500.0 + period]},
        geometry=[LineString([(1000.0, 1000.0), (1200.0, 1200.0)])],
        crs="EPSG:2100")
    return SimpleNamespace(in_zone=in_zone, routes=routes, n_removed=period)


def _patch_common(monkeypatch, ft):
    monkeypatch.setattr(ft, "load_atrisk", _fake_atrisk)
    monkeypatch.setattr(ft, "buffer_width_grid", _fake_buffer_width_grid)
    monkeypatch.setattr(ft, "load_graph", lambda: object())


class TestMainHappyPath:
    def test_writes_summary_and_gpkg_for_every_period(self, tmp_path, monkeypatch):
        scenario_dir = tmp_path / "scenario"
        _write_scenario(scenario_dir)
        ft = _reload_with_scenario_dir(monkeypatch, scenario_dir)
        _patch_common(monkeypatch, ft)
        monkeypatch.setattr(
            ft, "route_to_refuges",
            lambda scen, graph=None: _fake_evac_result(0))

        ft.main()

        summary_path = scenario_dir / "timestep_summary.json"
        gpkg_path = scenario_dir / "timestep_evacuation.gpkg"
        assert summary_path.exists()
        assert gpkg_path.exists()

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert [row["period"] for row in summary] == list(range(N_PERIODS))

        at_risk = gpd.read_file(gpkg_path, layer="at_risk")
        assert len(at_risk) == N_PERIODS * 2   # 2 rows per period
        assert sorted(at_risk["period"].unique()) == list(range(N_PERIODS))

        routes = gpd.read_file(gpkg_path, layer="routes")
        assert len(routes) == N_PERIODS        # 1 route per period


class TestMainAllOrNothingOnFailure:
    def test_mid_run_failure_loses_all_prior_periods_nothing_written(
            self, tmp_path, monkeypatch):
        """REGRESSION-DOCUMENTATION TEST -- proves a KNOWN, INTENTIONAL current
        weak point, not a bug in this test: `main()` accumulates per-period
        results in plain Python lists (`at_all`/`rt_all`/`summary`) and only
        writes `timestep_summary.json` / `timestep_evacuation.gpkg` ONCE, after
        the ENTIRE `for` loop over all periods completes successfully. So if
        ANY single period's routing call raises -- even after several EARLIER
        periods succeeded -- the exception propagates straight out of `main()`
        uncaught, and NOTHING is written to disk: every already-computed prior
        period's results are silently lost, with no partial/incremental output.

        If this module is later changed to write incrementally (e.g. per-period
        appends, or a try/except that flushes what it has), THIS test must be
        UPDATED to match the new behaviour, not silently deleted -- it is
        actively asserting the OLD, current "all-or-nothing" behaviour is real.
        """
        scenario_dir = tmp_path / "scenario"
        _write_scenario(scenario_dir)
        ft = _reload_with_scenario_dir(monkeypatch, scenario_dir)
        _patch_common(monkeypatch, ft)

        calls = {"n": 0}

        def _side_effect(scen, graph=None):
            calls["n"] += 1
            if calls["n"] == 3:      # fails on the 3rd period (index 2)
                raise RuntimeError("simulated routing failure on period 2")
            return _fake_evac_result(calls["n"] - 1)

        monkeypatch.setattr(ft, "route_to_refuges", _side_effect)

        with pytest.raises(RuntimeError, match="simulated routing failure"):
            ft.main()

        assert calls["n"] == 3   # periods 0 and 1 DID succeed in memory...
        # ...yet NEITHER output file exists -- proving nothing was flushed
        # incrementally for those two successful periods.
        assert not (scenario_dir / "timestep_summary.json").exists()
        assert not (scenario_dir / "timestep_evacuation.gpkg").exists()

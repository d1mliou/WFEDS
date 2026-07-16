"""Smoke test for scripts/cell2fire/visualize_fire.py.

THIS PASS ONLY covers one path: a genuinely minimal synthetic 2-hour scenario run
(two nested fire perimeters, one at-risk settlement, no 2021 real-fire overlay)
through `main()`'s HTML-generation path. Detailed content validation of the
generated Leaflet/JS dashboard is NOT worth the effort for a rendering-heavy
module like this one and is out of scope -- see the module's own docstring for
the dashboard's design.
"""

import json
import sys

import pandas as pd
from rasterio.transform import from_origin

from tests.helpers.geodata_factories import (
    make_tiny_asc,
    make_tiny_points_gdf,
    make_tiny_polygon_gdf,
)


def _build_minimal_run(run_dir):
    """The smallest run directory that reaches `main()`'s HTML write.

    Reading visualize_fire.py's `main()` top to bottom, these are read
    UNCONDITIONALLY (no `.exists()` guard) and so are all required here:
    perimeters.geojson, instance/Weather.csv, instance/Forest.asc,
    timestep_summary.json, timestep_evacuation.gpkg (layer "at_risk"), and (via
    settlements.all_settlements(), stubbed by the test instead -- see below) the
    official settlements layer. isochrones.geojson and the gpkg's "routes" layer
    ARE guarded (`.exists()` / try-except) and are deliberately omitted here.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    instance_dir = run_dir / "instance"
    instance_dir.mkdir()

    # perimeters.geojson: 2 hourly perimeters, period 0 nested inside period 1
    # (like real fire growth) -- "period" is read unconditionally, so at least
    # one row is required, and >=2 lets the arrival-time-band diff logic run too.
    perims = make_tiny_polygon_gdf(
        [
            {"period": 0, "geometry": [(-150, -150), (150, -150), (150, 150), (-150, 150)]},
            {"period": 1, "geometry": [(-300, -300), (300, -300), (300, 300), (-300, 300)]},
        ],
        crs="EPSG:2100",
    )
    perims.to_file(run_dir / "perimeters.geojson", driver="GeoJSON")

    # instance/Forest.asc: only used for the simulation-window box; a trivial
    # raster is enough (its extent need not overlap the perimeters above -- the
    # window is drawn but never used to spatially filter anything else here).
    make_tiny_asc(
        instance_dir / "Forest.asc",
        [[0] * 5 for _ in range(5)],
        from_origin(-1000, 1000, 200, 200),
    )

    # instance/Weather.csv: only "datetime"/"WS"/"WD" are ever read.
    pd.DataFrame({
        "datetime": ["2021-08-29 00:00:00", "2021-08-29 01:00:00"],
        "WS": [20.0, 25.0],
        "WD": [90.0, 95.0],
    }).to_csv(instance_dir / "Weather.csv", index=False)

    # timestep_summary.json: a plain list, indexed POSITIONALLY by hour (not
    # filtered by a "period" field) -- one dict per hour, in order.
    summary = [
        {"at_risk": 1, "population": 120, "routed": 1, "cut_off": 0, "impacted": 0,
         "edges_removed": 0},
        {"at_risk": 1, "population": 120, "routed": 0, "cut_off": 1, "impacted": 0,
         "edges_removed": 2},
    ]
    (run_dir / "timestep_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # timestep_evacuation.gpkg: "at_risk" layer only. The module's own read of
    # the "routes" layer is wrapped in try/except (falls back to None), so it is
    # deliberately omitted here to keep the fixture minimal.
    at_risk = make_tiny_points_gdf(
        [
            {"geometry": (0, 0), "NAME_OIK": "Χωριό Α",
             "census2021": 120, "status": "ok",
             "refuge": "Καταφύγιο Β",
             "route_km": 3.2, "period": 0},
            {"geometry": (0, 0), "NAME_OIK": "Χωριό Α",
             "census2021": 120, "status": "cut_off", "refuge": None,
             "route_km": None, "period": 1},
        ],
        crs="EPSG:2100",
    )
    at_risk.to_file(run_dir / "timestep_evacuation.gpkg", layer="at_risk", driver="GPKG")


def test_visualize_fire_minimal_synthetic_run_does_not_crash(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _build_minimal_run(run_dir)

    # WFEDS_SCENARIO_DIR is the documented run-folder override (same mechanism as
    # fire_timesteps.py); WFEDS_REAL_OVERLAY=0 is the documented switch for
    # user/hypothetical runs that skips the 2021 real-fire (VIIRS/burned-area)
    # overlay entirely -- that data isn't part of the test fixtures dir and is
    # unrelated to what this smoke test targets.
    monkeypatch.setenv("WFEDS_SCENARIO_DIR", str(run_dir))
    monkeypatch.setenv("WFEDS_REAL_OVERLAY", "0")

    # settlements.all_settlements() unconditionally reads the real
    # Settlements_Istiaia.geojson (no existence check) -- not part of the test
    # fixtures data dir and unrelated to what this smoke test targets, so it is
    # stubbed with a tiny in-memory layer rather than writing real-looking data
    # into the shared tests/fixtures/data tree. visualize_fire.main() does
    # `from settlements import all_settlements` INSIDE the function (a fresh
    # lookup on settlements' own module namespace each call), so patching the
    # settlements module's attribute -- not visualize_fire's -- is what's needed.
    import settlements as settlements_module
    fake_settlements = make_tiny_points_gdf(
        [{"geometry": (0, 0), "name": "Χωριό Α", "pop": 120}],
        crs="EPSG:2100",
    )
    monkeypatch.setattr(settlements_module, "all_settlements", lambda: fake_settlements)

    # visualize_fire.py resolves WFEDS_SCENARIO_DIR into module-level path
    # constants (PERIMS, INSTANCE, WEATHER, TS_GPKG, TS_JSON, OUT_HTML, ...) as
    # plain top-level code at IMPORT time, not inside main() -- force a fresh
    # import so it re-reads the env vars set above, regardless of whether an
    # earlier test/collection already imported it with different values.
    sys.modules.pop("visualize_fire", None)
    import visualize_fire

    try:
        # No network access needed: the Leaflet CDN links in TEMPLATE are only
        # static text baked into the HTML for the BROWSER to fetch later, at view
        # time -- generating the file itself makes no network call.
        visualize_fire.main()
    finally:
        sys.modules.pop("visualize_fire", None)

    out_html = run_dir / "fire_timesteps.html"
    assert out_html.exists()
    assert out_html.stat().st_size > 100

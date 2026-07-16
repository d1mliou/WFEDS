"""Smoke test for scripts/cell2fire/snapshot_map.py -- the "fire that barely
spread" edge case (an EMPTY perimeters.geojson: zero recorded perimeter features).

HISTORY -- this was a REAL, previously-undocumented bug, now FIXED (the test used
to be an xfail(strict=True) documenting the crash):
RunContext.__init__ did, as its very first action,
`gpd.read_file(run_dir / "perimeters.geojson").to_crs(2100).sort_values("period")`.
gpd.read_file() on a GeoJSON FeatureCollection with ZERO features returns a
GeoDataFrame with ONLY a "geometry" column (geopandas has no properties schema to
infer when there are no features), so there was no "period" column and
sort_values("period") raised `KeyError: 'period'` immediately -- a barely-spreading
fire (a perfectly legitimate outcome under mild weather) crashed snapshot_map.py
before producing ANY output. Downstream of that first crash, `ctx.hours[-1]`,
`max(ctx.hours[-1], 1)` and the `frames[-1]` MP4 hold all assumed >=1 perimeter too.

THE FIX (mirrors cell2fire_adapter's "barely-spreading fires no longer crash the
adapter"): a zero-perimeter run is now a REPORTABLE RESULT, not a crash --
snapshot() still renders the study window / roads / settlements with a clear Greek
"η φωτιά δεν επεκτάθηκε σημαντικά" note, and still writes a valid map.png (plus a
single-frame map.mp4), so the downstream run_scenario/agent contract (map_png /
map_anim) keeps holding.

Detailed unit testing of the rendering internals (matplotlib PNG/MP4 frame-by-frame
generation, the info panel, the legend, ...) is NOT worth the effort for a
rendering-heavy module like this one and is out of scope -- see the module's own
docstring for what each helper does.
"""

import json

from rasterio.transform import from_origin

from tests.helpers.geodata_factories import make_tiny_asc, make_tiny_points_gdf


def test_snapshot_map_empty_perimeters_does_not_crash(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    instance_dir = run_dir / "instance"
    instance_dir.mkdir()

    # perimeters.geojson with ZERO features == the "fire that barely spread"
    # output. Written as raw JSON (exactly what the adapter emits) so geopandas
    # reads it back with only a "geometry" column and no "period" -- the exact
    # shape that used to trigger KeyError: 'period' on the very first line of
    # RunContext.__init__.
    empty_fc = {"type": "FeatureCollection", "features": []}
    (run_dir / "perimeters.geojson").write_text(json.dumps(empty_fc), encoding="utf-8")

    # instance/Forest.asc: the study window (wx0/wy0/wx1/wy1 + the map extent) is
    # read UNCONDITIONALLY from here. A trivial 5x5 / 200 m raster is enough; its
    # extent is box(-1000, 0, 0, 1000) in EPSG:2100. In the real pipeline this
    # file always exists (build_instance writes it before the engine runs), so an
    # empty-perimeters run really does reach the rest of RunContext.__init__.
    make_tiny_asc(
        instance_dir / "Forest.asc",
        [[0] * 5 for _ in range(5)],
        from_origin(-1000, 1000, 200, 200),
    )

    # settlements: snapshot_map does `from settlements import all_settlements` at
    # MODULE level, so the name lives in snapshot_map's OWN namespace -- patch it
    # THERE (not settlements.all_settlements) with one tiny point inside the window
    # above, so the map has something to draw without reading the real data tree.
    import snapshot_map
    fake_settlements = make_tiny_points_gdf(
        [{"geometry": (-500, 500), "name": "Χωριό Α", "pop": 120}],
        crs="EPSG:2100",
    )
    monkeypatch.setattr(snapshot_map, "all_settlements", lambda: fake_settlements)

    # THE ASSERTION: no exception (was KeyError: 'period'), and a real,
    # non-trivial map.png is produced -- plus a valid map.mp4 (a single "did not
    # spread" frame) so the run_scenario/agent map_png + map_anim contract holds.
    png, mp4 = snapshot_map.snapshot(run_dir)

    assert png == run_dir / "map.png"
    assert png.exists()
    assert png.stat().st_size > 1000            # a real rendered map, not a stub
    assert mp4 == run_dir / "map.mp4"
    assert mp4.exists()
    assert mp4.stat().st_size > 0

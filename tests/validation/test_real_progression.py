"""Tests for scripts/validation/real_progression.py.

Covers the module's two public functions, `reconstruct()` and `extent_at()`, against
a tiny synthetic "Real_Fire_Data" directory (built with the geodata factories,
monkeypatched over the module's `RFD` constant) instead of the real OneDrive data:

  * `reconstruct()` builds a per-cell "arrival time" (hours since t0) surface by
    assigning each burned-scar grid cell the acquisition time of its NEAREST VIIRS
    detection point (`scipy.spatial.cKDTree` 1-nearest-neighbour lookup -- an exact
    nearest-neighbour assignment, not a smoothed/weighted interpolation, so a cell
    deep inside one detection's Voronoi region gets exactly that detection's
    timestamp, no blending).
  * `extent_at()` thresholds that surface (`arrival <= h`) and returns the fire's
    extent at a given time as a shapely (Multi)Polygon, or None if nothing has
    "arrived" yet.

`RFD` (`scripts/cell2fire/real_progression.py`'s module-level real-fire-data
directory constant, `DATA_DIR / "Real_Fire_Data"`) is read as a plain global lookup
inside `reconstruct()` on every call (not captured as a bound default argument), so
`monkeypatch.setattr(rp, "RFD", ...)` is monkeypatch-friendly here.
"""

import pandas as pd
import pytest
from rasterio import transform as rtransform

import real_progression as rp
from tests.helpers.geodata_factories import make_tiny_points_gdf, make_tiny_polygon_gdf

# Three VIIRS detections placed on the diagonal of a 1000 m x 1000 m burned-area
# square. Nearest-neighbour assignment among exactly-collinear points depends only on
# the projection onto that line -- equivalently, on x + y here -- so each point's
# Voronoi region is an easy-to-reason-about diagonal band:
#   P1 (50, 50)     x+y=100   -- t0 itself (hours = 0)      -> band x+y < 550
#   P2 (500, 500)   x+y=1000  -- t0 + 6 h                   -> band 550 < x+y < 1450
#   P3 (950, 950)   x+y=1900  -- t0 + 16 h (last detection) -> band x+y > 1450
# P1's band (x+y < 550) is the triangle (0,0)-(550,0)-(0,550), area 550**2/2 =
# 151 250 m**2 -- 15.1% of the 1 000 000 m**2 square.
T0_STR = "2021-08-05 23:00:00"  # exactly on the hour -> t0 = this, unchanged by floor("h")
P2_STR = "2021-08-06 05:00:00"  # t0 + 6h
P3_STR = "2021-08-06 15:00:00"  # t0 + 16h (last detection)

SQUARE = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
SCAR_AREA = 1000.0 * 1000.0
# Finer than the module's production default (GRID_M = 200 m): keeps rasterisation
# discretisation error small on this tiny 1000 m square (1000 / 25 = 40 exactly, so
# the square's own edges land exactly on grid lines -- no partial edge cells).
GRID_M = 25.0


@pytest.fixture
def real_fire_env(tmp_path, monkeypatch):
    """Write a tiny synthetic Real_Fire_Data dir and point `rp.RFD` at it."""
    burned_dir = tmp_path / "Burned_Area"
    viirs_dir = tmp_path / "VIIRS"
    burned_dir.mkdir()
    viirs_dir.mkdir()

    burned = make_tiny_polygon_gdf([{"geometry": SQUARE}])
    burned.to_file(burned_dir / "Burned_area_20210829.shp", driver="ESRI Shapefile")

    viirs = make_tiny_points_gdf([
        {"geometry": (50, 50), "acq_at_s": T0_STR},
        {"geometry": (500, 500), "acq_at_s": P2_STR},
        {"geometry": (950, 950), "acq_at_s": P3_STR},
    ])
    viirs.to_file(viirs_dir / "VIRS_dataset_egsa_dhmos.shp", driver="ESRI Shapefile")

    monkeypatch.setattr(rp, "RFD", tmp_path)
    return tmp_path


@pytest.fixture
def reconstructed(real_fire_env):
    return rp.reconstruct(grid_m=GRID_M)


class TestReconstruct:
    def test_t0_is_the_floored_earliest_viirs_detection(self, reconstructed):
        _, _, t0 = reconstructed
        assert t0 == pd.Timestamp(T0_STR)

    def test_arrival_time_matches_nearest_viirs_detection_timestamp(self, reconstructed):
        """A cell deep inside P2's Voronoi band (x+y=1000, ~450 m from either the
        P1 or P3 boundary -- 18 grid cells of margin) should get exactly P2's
        hours-since-t0 (6.0): nearest-neighbour lookup is an exact assignment, not
        a blend, so the tolerance only needs to absorb float rounding."""
        arrival, tr, t0 = reconstructed
        row, col = rtransform.rowcol(tr, 500.0, 500.0)
        assert arrival[row, col] == pytest.approx(6.0, abs=1e-6)

    def test_arrival_time_near_last_detection_matches_it(self, reconstructed):
        arrival, tr, t0 = reconstructed
        row, col = rtransform.rowcol(tr, 950.0, 950.0)
        assert arrival[row, col] == pytest.approx(16.0, abs=1e-6)


class TestExtentAt:
    def test_before_any_detection_extent_is_none(self, reconstructed):
        arrival, tr, t0 = reconstructed
        extent = rp.extent_at(arrival, tr, t0, t0 - pd.Timedelta(hours=1))
        assert extent is None

    def test_extent_at_t0_is_minimal_relative_to_the_full_scar(self, reconstructed):
        """At h=0 only P1's Voronoi band (x+y < 550, ~15.1% of the square) has
        'arrived' -- non-empty (P1 is itself a real detection at hours=0) but
        clearly a small fraction of the full scar."""
        arrival, tr, t0 = reconstructed
        extent = rp.extent_at(arrival, tr, t0, t0)
        assert extent is not None
        assert extent.area < 0.25 * SCAR_AREA

    def test_extent_grows_monotonically_over_time(self, reconstructed):
        arrival, tr, t0 = reconstructed
        early = rp.extent_at(arrival, tr, t0, t0)
        mid = rp.extent_at(arrival, tr, t0, t0 + pd.Timedelta(hours=6))
        late = rp.extent_at(arrival, tr, t0, t0 + pd.Timedelta(hours=16))
        assert early.area < mid.area < late.area

    def test_extent_at_or_after_the_last_detection_approaches_the_full_scar(self, reconstructed):
        """Well after the last VIIRS detection (t0 + 16h), every burned cell has
        'arrived' -- extent_at should recover essentially the whole synthetic
        1000x1000 m burned-area polygon (bounded by, and close to, its own area --
        the residual gap is grid_m=25 rasterisation's diagonal-edge discretisation)."""
        arrival, tr, t0 = reconstructed
        extent = rp.extent_at(arrival, tr, t0, t0 + pd.Timedelta(hours=1000))
        assert extent is not None
        assert extent.area <= SCAR_AREA * 1.01  # bounded by the real polygon's area
        assert extent.area >= SCAR_AREA * 0.85  # close to the full extent


class TestExportHourly:
    """`export_hourly()` -- the DATA seam to the live dashboard: one feature per
    hour of the event, `time_utc` lookup key + exact `km2`, stored in EPSG:4326."""

    def test_writes_one_feature_per_hour_with_monotone_km2(self, real_fire_env, tmp_path):
        import geopandas as gpd

        out = tmp_path / "hourly.geojson"

        path, n = rp.export_hourly(out_path=out)

        assert path == out and out.exists()
        g = gpd.read_file(out)
        # hours 0..16 (t0 .. last detection, hmax = ceil(max arrival)) -> 17 features
        assert n == len(g) == 17
        # NB GDAL parses the ISO strings back into datetimes on read - compare as
        # timestamps (the dashboard's reader normalises the same way).
        assert pd.to_datetime(g["time_utc"].iloc[0]) == pd.Timestamp("2021-08-05 23:00")
        assert pd.to_datetime(g["time_utc"].iloc[-1]) == pd.Timestamp("2021-08-06 15:00")
        assert (g["km2"].diff().dropna() >= 0).all()           # a fire only grows
        assert g.crs.to_epsg() == 4326                         # RFC 7946 storage CRS

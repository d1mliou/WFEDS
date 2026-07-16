"""Tests for scripts/cell2fire/evacuation.py.

Covers `exposure_for_fire(fire, atrisk, width, grid)`, the pure per-fire
exposure step that takes everything in-memory. `compute_exposure()` (which
loads `atrisk` via `settlements.load_atrisk()`, reads the real Cell2Fire
perimeter file and calls the real `route_with_fire.buffer_width_grid()` --
all real files on disk) is explicitly OUT OF SCOPE and not touched here.

`exposure_for_fire` reuses `route_with_fire.blocked_zone(fire, width, grid)`
to get the blocked polygon ("zone"), then flags settlements within
`EXPOSURE_MARGIN_M` (300 m, a module-level constant) of that polygon via
`atrisk.sindex.query(zone, predicate="dwithin", distance=EXPOSURE_MARGIN_M)`.
GEOS's `dwithin` predicate is INCLUSIVE (`distance <= EXPOSURE_MARGIN_M`), not
`distance < EXPOSURE_MARGIN_M` -- verified empirically in `TestExposureMargin`
below (not assumed): a point placed at exactly `EXPOSURE_MARGIN_M` from the
zone boundary lands in `in_zone`, one 1 m further out lands in `safe`.

`width`/`grid` here mimic exactly what `route_with_fire.buffer_width_grid()`
returns (see `route_with_fire._read_reference()` / `buffer_width_grid()`):
`width` is a numpy array of per-cell buffer widths in metres, shaped like
`grid["shape"]`; `grid` is a dict with keys `"transform"` (a rasterio
`Affine`), `"shape"` (`(nrows, ncols)`) and `"res"` (scalar pixel size, m).
`blocked_zone()` only ever reads those three grid keys, so a hand-built dict
(no real raster file) is a faithful stand-in.
"""

import numpy as np
import rasterio
from shapely.geometry import Polygon

import evacuation as ev
from tests.helpers.geodata_factories import make_tiny_points_gdf


def _grid(xres=10.0, ncols=100, nrows=100, x0=0.0, y0=1000.0):
    """A small square grid: x in [x0, x0 + ncols*xres], y in [y0 - nrows*xres, y0]."""
    transform = rasterio.transform.from_origin(x0, y0, xres, xres)
    return {"transform": transform, "shape": (nrows, ncols), "res": xres}


def _fire(cx=500.0, cy=500.0, half=20.0):
    """A small square fire footprint centred on (cx, cy), well inside the grid."""
    return Polygon([(cx - half, cy - half), (cx + half, cy - half),
                     (cx + half, cy + half), (cx - half, cy + half)])


class TestExposureForFireInZone:
    def test_settlement_well_within_margin_is_flagged_in_zone(self):
        """A huge, uniform buffer width makes `blocked_zone` cover the whole
        grid rectangle [0, 1000] x [0, 1000] (every cell is within `width` of
        the fire). One settlement sits inside that zone outright; another
        sits 100 m outside it -- comfortably under the 300 m exposure margin.
        Both must be flagged `in_zone`."""
        grid = _grid()
        width = np.full(grid["shape"], 1.0e6)
        fire = _fire()
        atrisk = make_tiny_points_gdf([
            {"geometry": (500.0, 500.0), "name": "inside_zone"},
            {"geometry": (1000.0 + 100.0, 500.0), "name": "well_within_margin"},
        ])

        scenario = ev.exposure_for_fire(fire, atrisk, width, grid)

        assert set(scenario.in_zone["name"]) == {"inside_zone", "well_within_margin"}
        assert len(scenario.safe) == 0


class TestExposureMargin:
    """EXPOSURE_MARGIN_M (300 m) boundary -- test against the ACTUAL observed
    behaviour of the `dwithin` spatial-index query, not an assumption."""

    def test_dwithin_predicate_is_inclusive_at_the_exact_margin(self):
        grid = _grid()
        width = np.full(grid["shape"], 1.0e6)   # zone == [0,1000] x [0,1000] rectangle
        fire = _fire()
        margin = ev.EXPOSURE_MARGIN_M
        atrisk = make_tiny_points_gdf([
            {"geometry": (1000.0 + margin, 500.0), "name": "exactly_at_margin"},
            {"geometry": (1000.0 + margin + 1.0, 500.0), "name": "one_metre_over_margin"},
        ])

        scenario = ev.exposure_for_fire(fire, atrisk, width, grid)

        # Confirmed empirically (see module docstring): dwithin is inclusive,
        # i.e. distance <= EXPOSURE_MARGIN_M -> in_zone; distance > -> safe.
        assert "exactly_at_margin" in set(scenario.in_zone["name"])
        assert "one_metre_over_margin" in set(scenario.safe["name"])


class TestExposureForFireAllSafe:
    def test_zero_width_buffer_leaves_all_settlements_safe(self):
        """A zero-width buffer collapses `blocked_zone` down to just the
        fire's own rasterised footprint (only cells at distance 0 from the
        fire qualify). Settlements far outside EXPOSURE_MARGIN_M of that
        footprint must all land in `safe`, none in `in_zone`."""
        grid = _grid()
        width = np.zeros(grid["shape"])
        fire = _fire()   # footprint roughly [480, 520] x [480, 520]
        atrisk = make_tiny_points_gdf([
            {"geometry": (500.0, 500.0 + 340.0), "name": "far_north"},
            {"geometry": (-500.0, -500.0), "name": "far_away"},
        ])

        scenario = ev.exposure_for_fire(fire, atrisk, width, grid)

        assert len(scenario.in_zone) == 0
        assert set(scenario.safe["name"]) == {"far_north", "far_away"}


class TestExposureForFireEmptyAtrisk:
    def test_empty_atrisk_gdf_returns_empty_but_valid_scenario(self):
        grid = _grid()
        width = np.full(grid["shape"], 50.0)
        fire = _fire()
        atrisk = make_tiny_points_gdf([])   # zero rows, but a valid GeoDataFrame

        scenario = ev.exposure_for_fire(fire, atrisk, width, grid)

        assert type(scenario).__name__ == "Scenario"
        assert len(scenario.atrisk) == 0
        assert len(scenario.in_zone) == 0
        assert len(scenario.safe) == 0
        assert scenario.zone is not None   # the zone itself is still computed fine

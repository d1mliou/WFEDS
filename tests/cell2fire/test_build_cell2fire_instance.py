"""Tests for scripts/cell2fire/build_cell2fire_instance.py.

THIS PASS ONLY covers the pure-logic helpers that take plain Python/numpy values
and do no I/O: `seed_cells_from_points` (the ONE seeding rule),
`_nearest_burnable_cell`, `aspect_deg` (plus `_validate_user_inputs`). The rest
of this module (`build_instance`, raster/CSV writing, etc.) is left for a later
agent -- see the module's own docstring in the source for what each guardrail is
supposed to do.
"""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Point

import build_cell2fire_instance as bci
from tests.helpers.geodata_factories import make_tiny_raster


# --------------------------------------------------------------------------------
# seed_cells_from_points (the ONE seeding rule: pins and VIIRS alike)
# --------------------------------------------------------------------------------
class TestSeedCellsFromPoints:
    """Every observed point -> buffer SEED_RADIUS_M (= VIIRS_PIXEL_M/2 = 187.5 m)
    -> unary_union -> rasterize onto the window grid -> keep burnable cells."""

    # A 20x20, 30 m grid: west=0, north=600 (row 0 spans y in [570, 600]) --
    # matches the affine convention `write_asc`/build_instance use.
    WT = from_origin(0, 600, 30, 30)
    NROWS = NCOLS = 20

    def all_burnable(self):
        return np.ones((self.NROWS, self.NCOLS), dtype=bool)

    def test_single_point_seeds_a_disc_not_one_cell(self):
        """1 pin = an observation with a physical footprint: a 187.5 m disc on a
        30 m grid covers every cell whose CENTRE falls inside the circle --
        exactly 120 cells for a point sitting on a cell corner (offsets
        +/-15..+/-165 m per axis, dx^2 + dy^2 <= 187.5^2, counted by hand)."""
        cells, n_fronts = bci.seed_cells_from_points(
            [Point(300, 300)], self.WT, self.NROWS, self.NCOLS, self.all_burnable())

        assert n_fronts == 1
        assert len(cells) == 120

    def test_all_seeded_cells_lie_within_the_footprint(self):
        """No cell farther than R + half a cell diagonal from the observation --
        the disc never 'leaks' beyond its physical footprint."""
        pt = Point(300, 300)
        cells, _ = bci.seed_cells_from_points(
            [pt], self.WT, self.NROWS, self.NCOLS, self.all_burnable())

        half_diag = 30 * math.sqrt(2) / 2
        for ncell in cells:
            r, c = divmod(ncell - 1, self.NCOLS)
            cx = self.WT.c + self.WT.a * (c + 0.5)
            cy = self.WT.f + self.WT.e * (r + 0.5)
            assert pt.distance(Point(cx, cy)) <= bci.SEED_RADIUS_M + half_diag

    def test_points_within_a_pixel_merge_into_one_front(self):
        """300 m apart < 2R = 375 m: the buffers overlap -> ONE front, purely by
        geometry (no clustering heuristic)."""
        _, n_fronts = bci.seed_cells_from_points(
            [Point(150, 300), Point(450, 300)],
            self.WT, self.NROWS, self.NCOLS, self.all_burnable())

        assert n_fronts == 1

    def test_far_apart_points_stay_independent_fires(self):
        """500 m apart > 2R = 375 m: disjoint footprints -> two independent
        fires (MultiPolygon), no fake connecting geometry."""
        _, n_fronts = bci.seed_cells_from_points(
            [Point(50, 300), Point(550, 300)],
            self.WT, self.NROWS, self.NCOLS, self.all_burnable())

        assert n_fronts == 2

    def test_non_burnable_cells_never_seed(self):
        """The burnable mask is the final gate: cells under the footprint that
        are non-fuel (river/road/settlement) are dropped."""
        burnable = self.all_burnable()
        burnable[:, :10] = False                       # west half = non-fuel

        cells, _ = bci.seed_cells_from_points(
            [Point(300, 300)], self.WT, self.NROWS, self.NCOLS, burnable)

        assert cells                                   # the east half still seeds
        for ncell in cells:
            _, c = divmod(ncell - 1, self.NCOLS)
            assert c >= 10

    def test_footprint_entirely_on_non_fuel_returns_no_cells(self):
        burnable = np.zeros((self.NROWS, self.NCOLS), dtype=bool)

        cells, n_fronts = bci.seed_cells_from_points(
            [Point(300, 300)], self.WT, self.NROWS, self.NCOLS, burnable)

        assert cells == []
        assert n_fronts == 1                           # the geometry exists; the fuel doesn't

    def test_cell_ids_are_1_based_row_major(self):
        """A point at the centre of the NW corner cell must seed ncell 1
        (row 0, col 0 -> 0*NCOLS + 0 + 1)."""
        cells, _ = bci.seed_cells_from_points(
            [Point(15, 585)], self.WT, self.NROWS, self.NCOLS, self.all_burnable())

        assert 1 in cells
        assert min(cells) >= 1

    def test_empty_points_return_empty(self):
        cells, n_fronts = bci.seed_cells_from_points(
            [], self.WT, self.NROWS, self.NCOLS, self.all_burnable())

        assert (cells, n_fronts) == ([], 0)


# --------------------------------------------------------------------------------
# _nearest_burnable_cell
# --------------------------------------------------------------------------------
class TestNearestBurnableCell:
    # A 30 m grid, west=0, north=300 (so row 0 spans y in [270, 300], col 0 spans
    # x in [0, 30]) -- matches the affine convention `write_asc`/build_instance use.
    WT = from_origin(0, 300, 30, 30)
    NCOLS = 10

    def test_exact_hit(self):
        """(x, y) mapping (via rounding) exactly to row 2 / col 3, which IS in the
        burnable set -> returned unchanged, with the correct 1-based row-major
        Ncell id."""
        br = np.array([2, 5, 0])
        bc = np.array([3, 7, 0])
        # 0.3 offset (not 0.5) sidesteps Python's round-half-to-even ambiguity.
        x = self.WT.c + self.WT.a * 3.3
        y = self.WT.f + self.WT.e * 2.3

        ir, ic, ncell = bci._nearest_burnable_cell(x, y, self.WT, br, bc, self.NCOLS)

        assert (ir, ic) == (2, 3)
        assert ncell == 2 * self.NCOLS + 3 + 1

    def test_falls_back_to_nearest_when_target_unburnable(self):
        """(x, y) maps to row 5 / col 5, which is NOT in the burnable set. The
        nearest burnable cell (in grid-index space) among (5, 7) and (0, 0) is
        (5, 7) (grid distance 2 vs. ~7.07) -> must be picked."""
        br = np.array([5, 0])
        bc = np.array([7, 0])
        x = self.WT.c + self.WT.a * 5.3
        y = self.WT.f + self.WT.e * 5.3

        ir, ic, ncell = bci._nearest_burnable_cell(x, y, self.WT, br, bc, self.NCOLS)

        assert (ir, ic) == (5, 7)
        assert ncell == 5 * self.NCOLS + 7 + 1


# --------------------------------------------------------------------------------
# aspect_deg
# --------------------------------------------------------------------------------
class TestAspectDeg:
    NROWS = NCOLS = 5
    CELLSIZE = 30.0

    def test_flat_dem_returns_zero_everywhere(self):
        dem = np.zeros((self.NROWS, self.NCOLS))

        asp = bci.aspect_deg(dem, self.CELLSIZE)

        assert np.all(asp == 0.0)

    def test_dem_rising_north_returns_zero_deg(self):
        """Elevation highest at row 0 (north, per the raster's north-up
        convention) and decreasing southward -> the uphill direction is due
        north -> aspect 0 deg everywhere."""
        dem = np.array([[(self.NROWS - 1 - r) * 10.0] * self.NCOLS
                        for r in range(self.NROWS)])

        asp = bci.aspect_deg(dem, self.CELLSIZE)

        assert asp == pytest.approx(np.zeros_like(asp), abs=1e-6)

    def test_dem_rising_east_returns_90_deg(self):
        """Elevation increases with column (east) and is constant along rows ->
        the uphill direction is due east -> aspect 90 deg everywhere."""
        dem = np.array([[c * 10.0 for c in range(self.NCOLS)]
                        for _ in range(self.NROWS)])

        asp = bci.aspect_deg(dem, self.CELLSIZE)

        assert asp == pytest.approx(np.full_like(asp, 90.0), abs=1e-6)


# --------------------------------------------------------------------------------
# _validate_user_inputs
# --------------------------------------------------------------------------------
class TestValidateUserInputs:
    """`_validate_user_inputs(points_ll, window_km, horizon_h, start_time)`
    guardrails. `points_ll` is `[(lat, lon), ...]` WGS84.

    The fuel-coverage check does `with rasterio.open(FUEL) as ds: fb = ds.bounds`
    inside the function body, reading the module-level `FUEL` name as a plain
    global lookup EVERY call (not a bound default argument) -- so
    `monkeypatch.setattr(bci, "FUEL", <path>)` is the correct, working strategy
    (verified by reading the source directly, per the task's GOTCHA warning).
    """

    # A tiny synthetic EPSG:2100 (Greek Grid) raster: 10x10 cells @ 100 m,
    # covering x in [400000, 401000], y in [4300000, 4301000]. Only `ds.bounds`
    # is ever read from it (the function never reads pixel values), so the pixel
    # content is irrelevant -- all zeros is fine.
    FUEL_TRANSFORM = from_origin(400000, 4301000, 100, 100)
    FUEL_SHAPE = (10, 10)

    # WGS84 equivalents of that bbox's centre / a point well outside it, computed
    # once (offline, via geopandas .to_crs -- EPSG:2100 -> EPSG:4326) and hardcoded
    # here as plain (lat, lon) tuples, matching the `points_ll` convention.
    INSIDE_POINT_LL = (38.850303, 22.855089)     # centre of the tiny raster's bbox
    OUTSIDE_POINT_LL = (39.756996, 24.007583)    # ~100 km NE - outside the bbox

    # Long before "now" by any margin -> always well clear of the forecast-ceiling
    # guardrail (FORECAST_MAX_DAYS_AHEAD = 15), regardless of when the suite runs.
    VALID_START = "2021-08-05T23:00"

    @pytest.fixture
    def fuel_raster(self, tmp_path, monkeypatch):
        """Point `bci.FUEL` at the tiny synthetic raster covering INSIDE_POINT_LL."""
        path = tmp_path / "fuel.tif"
        make_tiny_raster(path, np.zeros(self.FUEL_SHAPE, dtype="uint8"),
                         self.FUEL_TRANSFORM)
        monkeypatch.setattr(bci, "FUEL", path)
        return path

    def test_empty_points_raises(self):
        with pytest.raises(bci.InstanceInputError):
            bci._validate_user_inputs([], 10.0, 24, self.VALID_START)

    # -- window_km boundary: source uses
    # `not (WINDOW_KM_MIN <= window_km <= WINDOW_KM_MAX)` -- BOTH ends INCLUSIVE.
    def test_window_km_exactly_min_passes(self, fuel_raster):
        """6.0 == WINDOW_KM_MIN: inclusive per the source's `<=` comparison."""
        pts, t0 = bci._validate_user_inputs(
            [self.INSIDE_POINT_LL], 6.0, 24, self.VALID_START)

        assert len(list(pts)) == 1

    def test_window_km_exactly_max_passes(self, fuel_raster):
        """40.0 == WINDOW_KM_MAX: inclusive per the source's `<=` comparison."""
        pts, t0 = bci._validate_user_inputs(
            [self.INSIDE_POINT_LL], 40.0, 24, self.VALID_START)

        assert len(list(pts)) == 1

    def test_window_km_just_below_min_raises(self):
        with pytest.raises(bci.InstanceInputError):
            bci._validate_user_inputs([self.INSIDE_POINT_LL], 5.99, 24, self.VALID_START)

    def test_window_km_just_above_max_raises(self):
        with pytest.raises(bci.InstanceInputError):
            bci._validate_user_inputs([self.INSIDE_POINT_LL], 40.01, 24, self.VALID_START)

    # -- horizon_h boundary: source uses
    # `not (HORIZON_H_MIN <= horizon_h <= HORIZON_H_MAX)` -- BOTH ends INCLUSIVE.
    def test_horizon_h_exactly_min_passes(self, fuel_raster):
        """1 == HORIZON_H_MIN: inclusive per the source's `<=` comparison."""
        pts, t0 = bci._validate_user_inputs(
            [self.INSIDE_POINT_LL], 10.0, 1, self.VALID_START)

        assert len(list(pts)) == 1

    def test_horizon_h_exactly_max_passes(self, fuel_raster):
        """48 == HORIZON_H_MAX: inclusive per the source's `<=` comparison."""
        pts, t0 = bci._validate_user_inputs(
            [self.INSIDE_POINT_LL], 10.0, 48, self.VALID_START)

        assert len(list(pts)) == 1

    def test_horizon_h_below_min_raises(self):
        with pytest.raises(bci.InstanceInputError):
            bci._validate_user_inputs([self.INSIDE_POINT_LL], 10.0, 0, self.VALID_START)

    def test_horizon_h_above_max_raises(self):
        with pytest.raises(bci.InstanceInputError):
            bci._validate_user_inputs([self.INSIDE_POINT_LL], 10.0, 49, self.VALID_START)

    def test_missing_start_time_raises(self):
        with pytest.raises(bci.InstanceInputError):
            bci._validate_user_inputs([self.INSIDE_POINT_LL], 10.0, 24, None)

    # -- date guardrail: flipped from a lower bound ("too recent for the ERA5
    # archive") to an upper bound ("too far ahead of the forecast ceiling").
    # `horizon_end = t0 + horizon_h hours` must be `<= now(UTC) + FORECAST_MAX_DAYS_AHEAD
    # days`; there is no longer any lower bound (the archive covers decades back).
    def test_start_time_now_with_short_horizon_no_longer_raises(self, fuel_raster):
        """The whole point of this feature: `start_time = now(), horizon_h = 1`
        used to trip the old "too recent for ERA5" guardrail unconditionally.
        It must NOT raise any more -- "now" is comfortably inside the forecast
        ceiling."""
        now_iso = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M")

        pts, t0 = bci._validate_user_inputs([self.INSIDE_POINT_LL], 10.0, 1, now_iso)

        assert len(list(pts)) == 1

    def test_start_time_within_forecast_ceiling_passes(self, fuel_raster):
        """A few days ahead of "now", well inside the ceiling -> passes."""
        start_iso = (datetime.now(timezone.utc).replace(tzinfo=None)
                     + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M")

        pts, t0 = bci._validate_user_inputs([self.INSIDE_POINT_LL], 10.0, 1, start_iso)

        assert len(list(pts)) == 1

    def test_start_time_plus_horizon_exceeds_forecast_ceiling_raises(self):
        """Well beyond `FORECAST_MAX_DAYS_AHEAD` -> must raise, message names the
        forecast ceiling (Greek text: "...πρόγνωση καιρού φτάνει έως...")."""
        start_iso = (datetime.now(timezone.utc).replace(tzinfo=None)
                     + timedelta(days=bci.FORECAST_MAX_DAYS_AHEAD + 5)).strftime("%Y-%m-%dT%H:%M")

        with pytest.raises(bci.InstanceInputError, match="πρόγνωση"):
            bci._validate_user_inputs([self.INSIDE_POINT_LL], 10.0, 1, start_iso)

    def test_start_time_plus_horizon_exactly_at_forecast_ceiling_passes(self, fuel_raster):
        """`horizon_end == furthest_ok` exactly -> inclusive (source uses `>`,
        not `>=`, to reject). 1 h horizon, start_time set so that
        `start_time + 1h == now + FORECAST_MAX_DAYS_AHEAD days` exactly."""
        start_iso = (datetime.now(timezone.utc).replace(tzinfo=None)
                     + timedelta(days=bci.FORECAST_MAX_DAYS_AHEAD)
                     - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")

        pts, t0 = bci._validate_user_inputs([self.INSIDE_POINT_LL], 10.0, 1, start_iso)

        assert len(list(pts)) == 1

    def test_start_time_plus_horizon_just_above_forecast_ceiling_raises(self):
        """One hour past the exact ceiling boundary -> raises."""
        start_iso = (datetime.now(timezone.utc).replace(tzinfo=None)
                     + timedelta(days=bci.FORECAST_MAX_DAYS_AHEAD)).strftime("%Y-%m-%dT%H:%M")

        with pytest.raises(bci.InstanceInputError, match="πρόγνωση"):
            bci._validate_user_inputs([self.INSIDE_POINT_LL], 10.0, 1, start_iso)

    def test_point_outside_fuel_coverage_raises_with_coordinate_in_message(self, fuel_raster):
        """A point ~100 km outside the tiny FUEL raster's bbox must raise, and
        the offending (rounded) coordinate must be listed in the message so the
        LLM can relay it verbatim."""
        with pytest.raises(bci.InstanceInputError) as exc_info:
            bci._validate_user_inputs(
                [self.OUTSIDE_POINT_LL], 10.0, 24, self.VALID_START)

        msg = str(exc_info.value)
        assert str(round(self.OUTSIDE_POINT_LL[0], 4)) in msg
        assert str(round(self.OUTSIDE_POINT_LL[1], 4)) in msg

    def test_all_valid_inputs_returns_projected_points_and_parsed_t0(self, fuel_raster):
        """No guardrail tripped -> returns `(pts, t0)`: `pts` is the input points
        reprojected to EPSG:2100 (same count as the input), `t0` is `start_time`
        parsed via `datetime.fromisoformat`."""
        points_ll = [self.INSIDE_POINT_LL]

        pts, t0 = bci._validate_user_inputs(points_ll, 10.0, 24, self.VALID_START)

        assert len(list(pts)) == len(points_ll)
        assert t0 == datetime.fromisoformat(self.VALID_START)

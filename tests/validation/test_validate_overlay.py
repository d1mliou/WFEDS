"""Tests for scripts/validation/validate_overlay.py.

Covers the pure, no-I/O protocol helpers: `bearing`, `ang_diff`,
`group_passes` (satellite-pass clustering), `distance_stats`,
`pass_metrics` (inclusion / buffered inclusion / distances / direction on
synthetic geometries) and `sector_areas_km2`. `main()` reads real
perimeters/VIIRS/burned-area files and builds a folium map - exercised by the
real scenario run, not unit-tested here.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from validate_overlay import (BUFFERS_M, ang_diff, bearing, distance_stats,
                              group_passes, pass_metrics, sector_areas_km2)


class TestBearing:
    def test_zero_is_north(self):
        assert bearing(dx=0.0, dy=1.0) == pytest.approx(0.0)

    def test_90_is_east(self):
        assert bearing(dx=1.0, dy=0.0) == pytest.approx(90.0)

    def test_180_is_south(self):
        assert bearing(dx=0.0, dy=-1.0) == pytest.approx(180.0)

    def test_270_is_west(self):
        assert bearing(dx=-1.0, dy=0.0) == pytest.approx(270.0)

    def test_result_always_in_0_360_range(self):
        for dx, dy in [(1, 1), (-1, 1), (-1, -1), (1, -1), (0, 0)]:
            b = bearing(dx, dy)
            assert 0.0 <= b < 360.0


class TestAngDiff:
    def test_plain_difference(self):
        assert ang_diff(90, 30) == pytest.approx(60)

    def test_wraps_across_north(self):
        """350 deg vs 10 deg is 20 deg apart, not 340."""
        assert ang_diff(350, 10) == pytest.approx(20)

    def test_symmetric(self):
        assert ang_diff(10, 350) == ang_diff(350, 10)

    def test_never_exceeds_180(self):
        for a in range(0, 360, 30):
            for b in range(0, 360, 30):
                assert 0 <= ang_diff(a, b) <= 180


class TestGroupPasses:
    def test_close_timestamps_form_one_pass(self):
        t = pd.Series(pd.to_datetime(
            ["2021-08-06 10:43", "2021-08-06 10:44", "2021-08-06 10:50"]))
        labels = group_passes(t, tol_min=20)
        assert labels.nunique() == 1
        assert labels.iloc[0] == pd.Timestamp("2021-08-06 10:43")

    def test_the_real_day1_passes_split_into_three(self):
        """The actual 2021 day-1 evaluation acquisition times (10:43 / 11:34 /
        12:23 UTC) must form three separate passes at the 20-min tolerance."""
        t = pd.Series(pd.to_datetime(
            ["2021-08-06 10:43"] * 3 + ["2021-08-06 11:34"] * 2
            + ["2021-08-06 12:23"]))
        labels = group_passes(t, tol_min=20)
        assert labels.nunique() == 3

    def test_gap_within_group_measured_from_group_start(self):
        """Chained timestamps 12 min apart: the third is 24 min after the
        FIRST, so it starts a new pass (grouping is anchored to the group's
        first timestamp, not the previous one)."""
        t = pd.Series(pd.to_datetime(
            ["2021-08-06 10:00", "2021-08-06 10:12", "2021-08-06 10:24"]))
        labels = group_passes(t, tol_min=20)
        assert labels.nunique() == 2


class TestDistanceStats:
    def test_percentile_fields_on_a_known_series(self):
        s = distance_stats([0.0] * 5 + [1000.0] * 5)
        assert s["mean_m"] == 500
        assert s["median_m"] == 500
        assert s["max_m"] == 1000

    def test_all_inside_gives_zeros(self):
        s = distance_stats([0.0, 0.0, 0.0])
        assert s == {"mean_m": 0, "median_m": 0, "p75_m": 0, "p90_m": 0,
                     "p95_m": 0, "max_m": 0}


class TestPassMetrics:
    """Synthetic setup: a 2x2 km simulated square (0..2000 m), origin at its
    lower-left corner. Points placed at known in/out positions."""

    SIM = box(0, 0, 2000, 2000)

    def _dets(self, coords):
        return gpd.GeoDataFrame(geometry=[Point(*c) for c in coords], crs=2100)

    def test_inclusion_counts_points_inside(self):
        dets = self._dets([(1000, 1000), (1500, 500), (3000, 1000)])  # 2 in, 1 out
        m = pass_metrics(dets, self.SIM, self.SIM, origin=(0, 0))
        assert m["n_detections"] == 3
        assert m["n_inside"] == 2
        assert m["inclusion_pct"] == 67

    def test_buffered_inclusion_catches_near_misses(self):
        """A point 300 m outside the square is not 'inside' but IS within the
        375 m tolerance; a point 5 km away is caught by no buffer."""
        dets = self._dets([(2300, 1000), (7000, 1000)])
        m = pass_metrics(dets, self.SIM, self.SIM, origin=(0, 0))
        assert m["inclusion_pct"] == 0
        assert m["inclusion_within_375m_pct"] == 50
        assert m["inclusion_within_1000m_pct"] == 50

    def test_distances_zero_inside_and_euclidean_outside(self):
        dets = self._dets([(1000, 1000), (2500, 1000)])   # inside; 500 m east
        m = pass_metrics(dets, self.SIM, self.SIM, origin=(0, 0))
        assert m["distance"]["max_m"] == 500
        assert m["distance"]["median_m"] == 250

    def test_direction_of_detections_due_north(self):
        """Detections straight north of the origin -> observed bearing 0; the
        square's centroid is NE of the origin -> nonzero angular difference."""
        dets = self._dets([(0, 5000), (0, 6000)])
        m = pass_metrics(dets, self.SIM, self.SIM, origin=(0, 0))
        assert m["obs_bearing_deg"] == 0
        assert m["sim_bearing_deg"] == 45           # centroid (1000, 1000)
        assert m["direction_diff_deg"] == 45

    def test_empty_group_returns_none(self):
        assert pass_metrics(self._dets([]), self.SIM, self.SIM, (0, 0)) is None

    def test_all_declared_buffers_present(self):
        dets = self._dets([(1000, 1000)])
        m = pass_metrics(dets, self.SIM, self.SIM, origin=(0, 0))
        for b in BUFFERS_M:
            assert f"inclusion_within_{int(b)}m_pct" in m


class TestSectorAreas:
    def test_square_due_north_lands_in_the_north_sector(self):
        poly = box(-500, 5000, 500, 6000)          # 1 km^2 straight north of origin
        areas = sector_areas_km2(poly, origin=(0, 0))
        assert areas["N"] == pytest.approx(1.0, abs=0.1)
        assert areas["S"] == 0.0

    def test_sector_areas_sum_to_polygon_area(self):
        poly = box(1000, 1000, 3000, 3000)         # 4 km^2 in the NE
        areas = sector_areas_km2(poly, origin=(0, 0))
        assert sum(areas.values()) == pytest.approx(4.0, abs=0.2)
        assert areas["NE"] > 3.0

    def test_empty_polygon_gives_all_zeros(self):
        from shapely.geometry import Polygon
        areas = sector_areas_km2(Polygon(), origin=(0, 0))
        assert set(areas.values()) == {0.0}

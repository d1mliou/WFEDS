"""Tests for scripts/validation/perimeter_convergence.py (pillar 1b).

Covers the pure, no-I/O protocol helpers: `fill_holes`,
`window_box_from_header` (+ a round-trip through a real .asc header),
`sample_ring` / `sample_boundary`, the exclusion masks,
`strip_frame_from_line`, `directed_distances` (point to CONTINUOUS line -
the test that pins "never to sampled points"), `direction_stats` /
`hour_indicators`, `find_t_star` (tie + censoring), `stability_table`,
`apply_sign` (the four sign-convention cells), `extract_runs` (wrap-around,
exclusion break, strict threshold) and `first_hour_area_exceeds`. `main()`
reads real perimeters / the burned-area shapefile and builds charts and a
folium map - exercised by the real scenario run, not unit-tested here.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from perimeter_convergence import (apply_sign, boundary_rings,
                                   direction_stats, directed_distances,
                                   exclusion_mask_sim, exclusion_masks_real,
                                   extract_runs, fill_holes, find_t_star,
                                   first_hour_area_exceeds, hour_indicators,
                                   sample_boundary, sample_ring,
                                   stability_table, strip_frame_from_line,
                                   window_box_from_header, zone_table,
                                   CENSORED_STATEMENT)


class TestFillHoles:
    def test_hole_is_removed(self):
        outer = box(0, 0, 1000, 1000)
        holed = Polygon(outer.exterior.coords,
                        [box(400, 400, 600, 600).exterior.coords])
        assert holed.area == pytest.approx(1000 * 1000 - 200 * 200)
        assert fill_holes(holed).area == pytest.approx(1000 * 1000)

    def test_polygon_without_holes_is_unchanged(self):
        sq = box(0, 0, 500, 500)
        assert fill_holes(sq).equals(sq)

    def test_multipolygon_parts_all_filled(self):
        a = Polygon(box(0, 0, 100, 100).exterior.coords,
                    [box(40, 40, 60, 60).exterior.coords])
        b = box(300, 0, 400, 100)
        filled = fill_holes(a.union(b))
        assert filled.area == pytest.approx(100 * 100 + 100 * 100)

    def test_empty_geometry_passes_through(self):
        empty = Polygon()
        assert fill_holes(empty).is_empty


class TestWindowBox:
    def test_round_trip_through_a_real_asc_header(self, tmp_path):
        """make_tiny_asc writes the exact header write_asc produces;
        read_asc_header + window_box_from_header must reproduce the corners."""
        from rasterio.transform import Affine

        from cell2fire_adapter import read_asc_header
        from tests.helpers.geodata_factories import make_tiny_asc

        cell, ncols, nrows = 30.0, 4, 3
        x0, ytop = 438000.0, 4303000.0
        tr = Affine(cell, 0, x0, 0, -cell, ytop)
        p = make_tiny_asc(tmp_path / "Forest.asc",
                          np.ones((nrows, ncols)), tr)
        rtr, rnr, rnc, _ = read_asc_header(str(p))
        w = window_box_from_header(rtr, rnr, rnc)
        assert w.bounds == pytest.approx(
            (x0, ytop - nrows * cell, x0 + ncols * cell, ytop))


class TestSampleRing:
    def test_1200m_square_gives_40_points_no_closing_duplicate(self):
        ring = boundary_rings(box(0, 0, 300, 300))[0]
        pts, chain = sample_ring(ring, spacing=30.0)
        assert len(pts) == 40                       # 1200 / 30, not 41
        assert chain[0] == 0.0 and chain[-1] == pytest.approx(1170.0)
        # consecutive samples on a straight side are exactly 30 m apart
        assert pts[0].distance(pts[1]) == pytest.approx(30.0)
        # the last point is NOT the ring start repeated
        assert pts[-1].distance(pts[0]) == pytest.approx(30.0)

    def test_ring_shorter_than_spacing_still_yields_one_point(self):
        ring = boundary_rings(box(0, 0, 5, 5))[0]   # perimeter 20 m < 30 m
        pts, chain = sample_ring(ring, spacing=30.0)
        assert len(pts) == 1 and chain[0] == 0.0

    def test_sample_boundary_multipart_gets_separate_ring_ids(self):
        two = box(0, 0, 300, 300).union(box(1000, 0, 1300, 300))
        pts = sample_boundary(two)
        assert set(pts["ring_id"]) == {0, 1}
        assert (pts.groupby("ring_id")["n_ring"].first() == 40).all()
        # idx restarts per ring
        assert list(pts[pts["ring_id"] == 1]["idx"][:2]) == [0, 1]


class TestExclusions:
    WINDOW = box(0, 0, 10000, 10000)

    def _pts(self, coords):
        return gpd.GeoDataFrame(geometry=[Point(*c) for c in coords],
                                crs=2100)

    def test_real_masks_partition_outside_and_near_frame(self):
        pts = self._pts([(10500, 5000),   # outside the window
                         (9980, 5000),    # inside, 20 m from the frame
                         (5000, 5000)])   # interior - kept
        outside, near = exclusion_masks_real(pts, self.WINDOW, tol=30.0)
        assert list(outside) == [True, False, False]
        assert list(near) == [False, True, False]
        assert not (outside & near).any()           # the counts partition

    def test_scar_fully_inside_window_excludes_nothing(self):
        pts = self._pts([(4000, 4000), (6000, 6000)])
        outside, near = exclusion_masks_real(pts, self.WINDOW, tol=30.0)
        assert not outside.any() and not near.any()

    def test_sim_mask_flags_frame_hugging_points(self):
        pts = self._pts([(10000, 5000), (5000, 5000)])
        mask = exclusion_mask_sim(pts, self.WINDOW, tol=30.0)
        assert list(mask) == [True, False]

    def test_strip_frame_removes_the_domain_edge_segments(self):
        """A sim box sharing 3 sides with the window frame keeps only its
        interior (top) edge as a distance target."""
        sim = box(0, 0, 10000, 5000)                # left/bottom/right on frame
        stripped = strip_frame_from_line(sim.boundary, self.WINDOW, tol=30.0)
        assert stripped.length < sim.boundary.length
        assert 9900 <= stripped.length <= 10000     # ~ the top edge only

    def test_strip_frame_far_from_frame_is_unchanged(self):
        sim = box(3000, 3000, 6000, 6000)
        stripped = strip_frame_from_line(sim.boundary, self.WINDOW, tol=30.0)
        assert stripped.length == pytest.approx(sim.boundary.length)


class TestDirectedDistances:
    REAL = box(200, 0, 1200, 1000)

    def _series(self, coords):
        return gpd.GeoSeries([Point(*c) for c in coords], crs=2100)

    def test_known_distances_to_the_boundary(self):
        d = directed_distances(self._series([(0, 500), (1000, 500)]),
                               self.REAL.boundary)
        assert d == pytest.approx([200.0, 200.0])

    def test_perpendicular_to_segment_midpoint_not_to_a_vertex(self):
        """The point opposite the middle of the top edge is 500 m from the
        CONTINUOUS line; its nearest VERTEX is 640 m away. Pins the
        'never to sampled points / vertices' rule."""
        d = directed_distances(self._series([(600, 1500)]),
                               self.REAL.boundary)
        assert d == pytest.approx([500.0])

    def test_degenerate_target_returns_none(self):
        empty = LineString()
        assert directed_distances(self._series([(0, 0)]), empty) is None

    def test_direction_stats_max_is_the_directed_hausdorff(self):
        s = direction_stats(np.array([200.0, 500.0]))
        assert s == {"mean_m": 350.0, "median_m": 350.0,
                     "p95_m": pytest.approx(485.0), "max_m": 500.0}

    def test_hour_indicators_combines_both_directions(self):
        s2r = {"mean_m": 100.0, "median_m": 90.0, "p95_m": 300.0,
               "max_m": 400.0}
        r2s = {"mean_m": 300.0, "median_m": 250.0, "p95_m": 700.0,
               "max_m": 900.0}
        ind = hour_indicators(s2r, r2s)
        assert ind["d_t_m"] == 200.0                # equal weight of directions
        assert ind["hausdorff_sym_m"] == 900.0
        assert ind["hd95_m"] == 700.0

    def test_hour_indicators_degenerate_direction_gives_none(self):
        assert hour_indicators(None, {"mean_m": 1.0})["d_t_m"] is None


def _hourly(d_t_values, first_hour=1):
    hours = list(range(first_hour, first_hour + len(d_t_values)))
    return pd.DataFrame({"hour": hours, "d_t_m": d_t_values})


class TestFindTStar:
    def test_interior_minimum_is_not_censored(self):
        ts = find_t_star(_hourly([500, 300, 200, 300, 400]))
        assert ts["hour"] == 3 and ts["d_t_m"] == 200
        assert not ts["censored"]
        assert ts["statement"] == "hour of closest geometric approach"

    def test_minimum_at_the_first_hour_is_censored(self):
        ts = find_t_star(_hourly([100, 300, 400]))
        assert ts["hour"] == 1 and ts["censored"]
        assert ts["statement"] == CENSORED_STATEMENT

    def test_minimum_at_the_horizon_is_censored(self):
        ts = find_t_star(_hourly([400, 300, 100]))
        assert ts["hour"] == 3 and ts["censored"]

    def test_tie_goes_to_the_earliest_hour_and_is_recorded(self):
        ts = find_t_star(_hourly([500, 200, 200, 400, 450]))
        assert ts["hour"] == 2 and ts["ties"] == [3]

    def test_nan_hours_are_skipped(self):
        ts = find_t_star(_hourly([np.nan, 300, 200, 350, 400]))
        assert ts["hour"] == 3 and not ts["censored"]

    def test_all_nan_returns_none(self):
        assert find_t_star(_hourly([np.nan, np.nan])) is None


class TestStability:
    def test_disagreeing_indicator_is_flagged_but_never_overrides(self):
        hourly = pd.DataFrame({
            "hour": [1, 2, 3, 4, 5],
            "d_t_m": [500, 300, 200, 300, 400],       # argmin 3
            "mean_s2r_m": [500, 300, 200, 300, 400],  # agrees (3)
            "mean_r2s_m": [500, 400, 300, 200, 100],  # argmin 5 - disagrees
        })
        ts = find_t_star(hourly)
        stab = stability_table(hourly, ts["hour"])
        by = {s["indicator"]: s for s in stab}
        assert by["mean_s2r_m"]["agrees_with_t_star"]
        assert not by["mean_r2s_m"]["agrees_with_t_star"]
        assert by["mean_r2s_m"]["argmin_hour"] == 5
        assert ts["hour"] == 3                        # t* untouched


class TestApplySign:
    """Positive ALWAYS = the simulation went beyond reality."""

    def test_sim_point_outside_the_scar_is_positive(self):
        assert apply_sign([100.0], [False], positive_when_inside=False)[0] == 100.0

    def test_sim_point_inside_the_scar_is_negative(self):
        assert apply_sign([100.0], [True], positive_when_inside=False)[0] == -100.0

    def test_real_point_inside_the_sim_is_positive(self):
        assert apply_sign([100.0], [True], positive_when_inside=True)[0] == 100.0

    def test_real_point_outside_the_sim_is_negative(self):
        assert apply_sign([100.0], [False], positive_when_inside=True)[0] == -100.0

    def test_on_boundary_zero_stays_zero(self):
        assert apply_sign([0.0], [False], positive_when_inside=True)[0] == 0.0


class TestExtractRuns:
    def test_two_zones_and_one_isolated_extreme(self):
        signed = np.full(20, 10.0)                   # below threshold
        signed[2:9] = 100.0                          # 7-point over-extension
        signed[10:16] = -100.0                       # 6-point lag
        signed[17:19] = 100.0                        # 2-point spike
        zones, extremes = extract_runs(signed, threshold=60.0, min_points=5)
        assert [(z["sign"], z["n_points"]) for z in zones] == [(1, 7), (-1, 6)]
        assert [(e["sign"], e["n_points"]) for e in extremes] == [(1, 2)]
        assert zones[0]["mean_abs_m"] == 100.0
        assert zones[0]["max_abs_m"] == 100.0

    def test_wrap_around_run_crossing_the_ring_closure_is_one_run(self):
        signed = np.zeros(10)
        signed[[7, 8, 9, 0, 1, 2, 3]] = 100.0
        zones, extremes = extract_runs(signed, threshold=60.0, min_points=5)
        assert len(zones) == 1 and not extremes
        assert zones[0]["n_points"] == 7
        assert zones[0]["start_idx"] == 7

    def test_whole_ring_one_class_is_one_zone(self):
        zones, extremes = extract_runs(np.full(8, 100.0), threshold=60.0,
                                       min_points=5)
        assert len(zones) == 1 and zones[0]["n_points"] == 8

    def test_excluded_point_breaks_contiguity(self):
        signed = np.zeros(12)
        signed[0:9] = 100.0                          # would be one 9-point run
        valid = np.ones(12, bool)
        valid[4] = False                             # excluded point inside it
        zones, extremes = extract_runs(signed, valid=valid, threshold=60.0,
                                       min_points=3)
        assert [(z["n_points"]) for z in zones] == [4, 4]   # split, not merged

    def test_threshold_is_strict(self):
        zones, extremes = extract_runs(np.full(6, 60.0), threshold=60.0,
                                       min_points=5)
        assert not zones and not extremes

    def test_nan_counts_as_class_zero(self):
        """The NaN at idx 3 is class 0 and breaks the run there; the two
        remaining stretches join through the ring CLOSURE into one 7-point
        wrap-around zone (7, not 8 - proving the NaN was excluded)."""
        signed = np.full(8, 100.0)
        signed[3] = np.nan
        zones, extremes = extract_runs(signed, threshold=60.0, min_points=3)
        assert [(z["n_points"]) for z in zones] == [7]
        assert zones[0]["start_idx"] == 4
        assert zones[0]["mean_abs_m"] == 100.0      # no NaN pollution

    def test_empty_ring(self):
        assert extract_runs(np.array([])) == ([], [])


class TestZoneTable:
    def test_groups_by_direction_and_ring_with_meanings(self):
        n = 12
        base = {
            "ring_id": [0] * n, "idx": list(range(n)), "n_ring": [n] * n,
            "excluded": [False] * n,
        }
        signed = [0.0] * n
        signed[1:7] = [100.0] * 6                    # over-extension zone
        signed[9:11] = [-100.0] * 2                  # short lag extreme
        sim = pd.DataFrame({**base, "direction": ["sim_to_real"] * n,
                            "signed_m": signed})
        real = pd.DataFrame({**base, "direction": ["real_to_sim"] * n,
                             "signed_m": [0.0] * n})
        zones = zone_table(pd.concat([sim, real], ignore_index=True))
        assert len(zones) == 2                       # the all-zero ring adds none
        z = zones[zones["kind"] == "zone"].iloc[0]
        assert z["meaning"] == "over-extension" and z["length_m"] == 180
        e = zones[zones["kind"] == "isolated_extreme"].iloc[0]
        assert e["meaning"] == "lag" and e["n_points"] == 2


class TestFirstHourAreaExceeds:
    def test_first_crossing_hour(self):
        assert first_hour_area_exceeds({1: 50.0, 2: 100.0, 3: 170.0},
                                       165.6) == 3

    def test_never_crossing_returns_none(self):
        assert first_hour_area_exceeds({1: 50.0, 2: 60.0}, 165.6) is None

    def test_crossing_at_hour_one(self):
        assert first_hour_area_exceeds({1: 200.0, 2: 300.0}, 165.6) == 1

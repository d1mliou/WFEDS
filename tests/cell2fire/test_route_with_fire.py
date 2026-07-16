"""Tests for scripts/cell2fire/route_with_fire.py.

THIS PASS ONLY covers `_slope_factor`, the pure PLACEHOLDER slope-factor helper
used inside `buffer_width_grid()`. It IS cleanly extractable as a standalone,
module-level pure function (not nested/private-in-a-closure), so it's in scope for
this pass; everything else in this module (`buffer_width_grid`, `_read_reference`,
`_read_fuel_on`, `blocked_zone`, `fuel_hazard_overlay` -- all of which do raster
I/O) is left for a later agent.
"""

import numpy as np
import pytest

import route_with_fire as rwf


class TestSlopeFactor:
    def test_zero_slope_gives_factor_1(self):
        assert rwf._slope_factor(0.0) == pytest.approx(1.0)

    def test_documented_45_degree_cap(self):
        assert rwf.SLOPE_CAP_DEG == 45.0
        assert rwf._slope_factor(45.0) == pytest.approx(2.0)

    def test_values_above_the_cap_clip_to_the_same_result_as_exactly_at_cap(self):
        at_cap = rwf._slope_factor(45.0)
        above_cap = rwf._slope_factor(90.0)
        way_above_cap = rwf._slope_factor(1000.0)

        assert above_cap == pytest.approx(at_cap)
        assert way_above_cap == pytest.approx(at_cap)

    def test_negative_slope_clips_to_zero_giving_factor_1(self):
        """np.clip's lower bound is 0.0 -- a (nonsensical) negative slope must not
        produce a factor below 1.0."""
        assert rwf._slope_factor(-10.0) == pytest.approx(1.0)

    def test_monotonic_increasing_within_the_0_to_45_range(self):
        vals = [rwf._slope_factor(s) for s in (0.0, 15.0, 30.0, 45.0)]
        assert vals == sorted(vals)
        assert vals[0] < vals[-1]

    def test_vectorized_over_a_numpy_array(self):
        slopes = np.array([0.0, 22.5, 45.0, 60.0])
        result = rwf._slope_factor(slopes)
        expected = np.array([1.0, 1.5, 2.0, 2.0])
        assert result == pytest.approx(expected)

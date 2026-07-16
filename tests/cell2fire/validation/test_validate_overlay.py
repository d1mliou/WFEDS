"""Tests for scripts/cell2fire/validate_overlay.py.

THIS PASS ONLY covers `bearing()`, the pure compass-bearing helper used by
`metric_block()`. Everything else in this module reads real perimeters/VIIRS/
burned-area files and builds a folium map -- left for a later agent.
"""

import pytest

from validate_overlay import bearing


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

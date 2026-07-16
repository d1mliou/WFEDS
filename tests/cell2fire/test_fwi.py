"""Tests for scripts/cell2fire/fwi.py -- the pure Van Wagner 1987 Canadian FWI System
math. No I/O anywhere in this module, so this is the best pure-logic test target in
the codebase: every test here is a plain function call + assertion.

Where an authoritative decimal reference value from the CFFDRS literature was not
independently available to verify against, tests assert *relative/monotonic*
behaviour (e.g. "hotter/drier/windier -> drier fuel code") instead of a hardcoded
number nobody could confirm is right. A few exact-equality tests ARE included where
the equality is provable directly from the algebra of the code itself (e.g. the BUI
branch-boundary continuity, and the FFMC rain-threshold no-op) -- see the comments.
"""

import pytest

import fwi


# --------------------------------------------------------------------------------
# daily_ffmc
# --------------------------------------------------------------------------------
class TestDailyFfmc:
    def test_dries_out_under_hot_dry_windy_no_rain_conditions(self):
        """Starting from a relatively moist code, a hot/dry/windy rainless day
        should dry the fuel out (FFMC rises toward 100)."""
        f0 = 60.0
        f1 = fwi.daily_ffmc(f0, temp=32.0, rh=20.0, wind=15.0, rain=0.0)
        assert f1 > f0

    def test_wets_under_high_humidity_from_a_very_dry_start(self):
        """Starting from a very dry code (95), a cool/near-saturated rainless day
        still picks up moisture from the air (the wetting branch, mo < ew) and
        FFMC falls."""
        f0 = 95.0
        f1 = fwi.daily_ffmc(f0, temp=15.0, rh=95.0, wind=5.0, rain=0.0)
        assert f1 < f0

    def test_rain_at_or_below_the_0_5mm_threshold_is_a_no_op(self):
        """The wetting-from-rain adjustment only triggers on `rain > 0.5`
        (strict). At exactly 0.5 mm it must behave identically to no rain at all --
        this is a direct algebraic consequence of the branch condition, not just an
        empirical trend, so we assert exact equality."""
        args = dict(f0=85.0, temp=25.0, rh=40.0, wind=10.0)
        at_threshold = fwi.daily_ffmc(rain=0.5, **args)
        no_rain = fwi.daily_ffmc(rain=0.0, **args)
        assert at_threshold == no_rain

    def test_heavy_rain_wets_the_fuel_more_than_no_rain(self):
        f0 = 92.0
        common = dict(f0=f0, temp=25.0, rh=40.0, wind=10.0)
        no_rain = fwi.daily_ffmc(rain=0.0, **common)
        heavy_rain = fwi.daily_ffmc(rain=20.0, **common)
        assert heavy_rain < no_rain

    def test_higher_wind_speeds_up_drying_when_above_the_drying_equilibrium(self):
        """With mo > ed (a moist start on a warm/dry day -> the ko/kd drying
        branch), more wind should dry the fuel faster -> higher FFMC."""
        vals = [fwi.daily_ffmc(60.0, temp=32.0, rh=20.0, wind=w, rain=0.0)
                for w in (5.0, 20.0, 40.0)]
        assert vals[0] < vals[1] < vals[2]


# --------------------------------------------------------------------------------
# daily_dmc
# --------------------------------------------------------------------------------
class TestDailyDmc:
    def test_rain_at_or_below_the_1_5mm_threshold_is_a_no_op(self):
        """Branch condition is `rain > 1.5` (strict); at exactly 1.5 mm the
        wetting adjustment must not trigger -- provably identical to no rain."""
        args = dict(p0=40.0, temp=25.0, rh=30.0, month=7)
        assert fwi.daily_dmc(rain=1.5, **args) == fwi.daily_dmc(rain=0.0, **args)

    def test_rain_just_above_threshold_lowers_dmc(self):
        args = dict(p0=40.0, temp=25.0, rh=30.0, month=7)
        below = fwi.daily_dmc(rain=1.5, **args)
        above = fwi.daily_dmc(rain=1.6, **args)
        assert above < below

    def test_heavy_rain_and_high_humidity_wets_the_duff_substantially(self):
        p0 = 40.0
        dry_day = fwi.daily_dmc(p0, temp=25.0, rh=30.0, rain=0.0, month=7)
        wet_day = fwi.daily_dmc(p0, temp=25.0, rh=90.0, rain=30.0, month=7)
        assert wet_day < dry_day

    def test_cold_temperature_freezes_growth_leaving_p0_unchanged(self):
        """`temp > -1.1` gates the whole drying-rate term k; at or below that,
        k = 0 and (with no rain) the code is returned unchanged."""
        p0 = 40.0
        assert fwi.daily_dmc(p0, temp=-1.1, rh=50.0, rain=0.0, month=1) == p0
        assert fwi.daily_dmc(p0, temp=-20.0, rh=50.0, rain=0.0, month=1) == p0

    @pytest.mark.parametrize("p0", [20.0, 50.0, 80.0])
    def test_b_formula_branches_all_reachable_without_crashing(self, p0):
        """The rain-wetting recovery formula for `b` has three p0-dependent
        sub-branches (<=33, 33..65, >65). All three must be reachable and finite."""
        result = fwi.daily_dmc(p0, temp=20.0, rh=50.0, rain=10.0, month=6)
        assert result == pytest.approx(result)  # finite, no NaN/exception
        assert result > 0

    def test_b_formula_branches_monotonic_in_p0(self):
        vals = [fwi.daily_dmc(p0, temp=20.0, rh=50.0, rain=10.0, month=6)
                for p0 in (20.0, 50.0, 80.0)]
        assert vals[0] < vals[1] < vals[2]


# --------------------------------------------------------------------------------
# daily_dc
# --------------------------------------------------------------------------------
class TestDailyDc:
    def test_rain_at_or_below_the_2_8mm_threshold_is_a_no_op(self):
        args = dict(d0=300.0, temp=25.0, month=7)
        assert fwi.daily_dc(rain=2.8, **args) == fwi.daily_dc(rain=0.0, **args)

    def test_rain_just_above_threshold_lowers_dc(self):
        args = dict(d0=300.0, temp=25.0, month=7)
        below = fwi.daily_dc(rain=2.8, **args)
        above = fwi.daily_dc(rain=2.9, **args)
        assert above < below

    def test_heavy_rain_lowers_dc(self):
        d0 = 300.0
        assert fwi.daily_dc(d0, temp=25.0, rain=30.0, month=7) < \
            fwi.daily_dc(d0, temp=25.0, rain=0.0, month=7)

    def test_cold_temperature_uses_the_month_factor_only_regardless_of_magnitude(self):
        """`temp > -2.8` gates the temperature-dependent term; below that, the
        month's DC_LF factor alone (floored at 0) drives the (small) increment,
        independent of exactly how cold it is."""
        d0 = 300.0
        moderately_cold = fwi.daily_dc(d0, temp=-2.8, rain=0.0, month=7)
        extremely_cold = fwi.daily_dc(d0, temp=-50.0, rain=0.0, month=7)
        assert moderately_cold == extremely_cold
        assert moderately_cold == pytest.approx(d0 + 0.5 * fwi.DC_LF[6])  # month=7 -> idx 6


# --------------------------------------------------------------------------------
# isi
# --------------------------------------------------------------------------------
class TestIsi:
    def test_monotonic_increasing_with_wind_at_fixed_ffmc(self):
        vals = [fwi.isi(85.0, w) for w in (0.0, 5.0, 20.0, 40.0)]
        assert vals == sorted(vals)
        assert vals[0] < vals[-1]

    def test_monotonic_increasing_with_ffmc_at_fixed_wind(self):
        vals = [fwi.isi(f, 20.0) for f in (60.0, 80.0, 95.0)]
        assert vals == sorted(vals)
        assert vals[0] < vals[-1]

    def test_zero_wind_is_not_zero_isi(self):
        """ISI at zero wind is not zero (there's still a base spread contribution
        from fine fuel moisture alone) -- guards against a naive `wind == 0 -> 0`
        regression."""
        assert fwi.isi(85.0, 0.0) > 0.0


# --------------------------------------------------------------------------------
# bui
# --------------------------------------------------------------------------------
class TestBui:
    def test_low_branch_matches_its_own_formula_directly(self):
        dmc, dc = 50.0, 500.0
        assert dmc <= 0.4 * dc
        expected = 0.8 * dmc * dc / (dmc + 0.4 * dc)
        assert fwi.bui(dmc, dc) == pytest.approx(expected)

    def test_continuous_at_the_dmc_equals_0_4_dc_branch_boundary(self):
        """bui()'s formula switches at `dmc == 0.4*dc`. Algebraically, substituting
        dmc = 0.4*dc into BOTH branch formulas gives exactly `dmc` in each case (see
        module comment for the derivation) -- so the two branches must agree
        exactly at the boundary, not just approximately."""
        dc = 500.0
        dmc = 0.4 * dc
        low_branch_formula = 0.8 * dmc * dc / (dmc + 0.4 * dc)
        high_branch_formula = (dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc))
                                * (0.92 + (0.0114 * dmc) ** 1.7))
        assert low_branch_formula == pytest.approx(high_branch_formula, rel=1e-9)
        # bui() itself takes the `<=` (low) branch at the exact boundary; confirm
        # it agrees with the high-branch formula too.
        assert fwi.bui(dmc, dc) == pytest.approx(high_branch_formula, rel=1e-9)

    def test_high_branch_reachable_and_positive(self):
        dmc, dc = 200.0, 100.0
        assert dmc > 0.4 * dc
        result = fwi.bui(dmc, dc)
        assert result > 0


# --------------------------------------------------------------------------------
# fwi
# --------------------------------------------------------------------------------
class TestFwi:
    def test_monotonic_increasing_with_isi_at_fixed_bui(self):
        vals = [fwi.fwi(isi_, 50.0) for isi_ in (5.0, 15.0, 30.0)]
        assert vals == sorted(vals)
        assert vals[0] < vals[-1]

    def test_monotonic_increasing_with_bui_at_fixed_isi(self):
        vals = [fwi.fwi(15.0, bui_) for bui_ in (20.0, 50.0, 90.0)]
        assert vals == sorted(vals)
        assert vals[0] < vals[-1]

    def test_low_b_branch_returns_b_itself(self):
        """When `b = 0.1 * isi * fd <= 1.0`, fwi() returns b unchanged (no log
        transform) -- provable directly from the code's own branch condition."""
        isi_, bui_ = 0.1, 1.0
        fd = 0.626 * bui_ ** 0.809 + 2.0
        b = 0.1 * isi_ * fd
        assert b <= 1.0   # sanity check this test scenario actually hits that branch
        assert fwi.fwi(isi_, bui_) == pytest.approx(b)


# --------------------------------------------------------------------------------
# spinup_daily_codes
# --------------------------------------------------------------------------------
def _make_hourly_rows(dates, overrides=None):
    """Build a full set of hourly rows (00:00..23:00, "YYYY-MM-DDTHH:MM") for each
    date in `dates`, all with benign constant weather.

    `overrides`: dict {(date, hour): "OMIT" | {field: value, ...}} -- "OMIT" drops
    that hour's row entirely; a dict patches fields (e.g. {"temp_c": None}) onto an
    otherwise-normal row for that hour.
    """
    overrides = overrides or {}
    rows = []
    for date in dates:
        for hour in range(24):
            key = (date, hour)
            if overrides.get(key) == "OMIT":
                continue
            row = {"time": f"{date}T{hour:02d}:00", "precip_mm": 0.0,
                   "temp_c": 25.0, "rh_pct": 40.0, "ws_kmh": 10.0}
            if key in overrides:
                row.update(overrides[key])
            rows.append(row)
    return rows


class TestSpinupDailyCodes:
    DATES = ["2021-06-01", "2021-06-02", "2021-06-03"]

    def test_skips_a_day_cleanly_when_the_noon_row_is_entirely_missing(self):
        rows = _make_hourly_rows(self.DATES, overrides={("2021-06-02", 10): "OMIT"})

        out = fwi.spinup_daily_codes(rows)

        assert "2021-06-02" not in out
        assert "2021-06-01" in out
        assert "2021-06-03" in out   # the gap doesn't crash later processing

    @pytest.mark.parametrize("field", ["temp_c", "rh_pct", "ws_kmh"])
    def test_skips_a_day_cleanly_when_the_noon_row_has_a_none_value(self, field):
        rows = _make_hourly_rows(
            self.DATES, overrides={("2021-06-02", 10): {field: None}})

        out = fwi.spinup_daily_codes(rows)

        assert "2021-06-02" not in out
        assert "2021-06-01" in out
        assert "2021-06-03" in out

    def test_valid_run_produces_a_rounded_ffmc_dmc_dc_tuple_per_date(self):
        rows = _make_hourly_rows(self.DATES)

        out = fwi.spinup_daily_codes(rows)

        assert set(out) == set(self.DATES)
        for date in self.DATES:
            ffmc, dmc, dc = out[date]
            assert isinstance(ffmc, float) and isinstance(dmc, float) and isinstance(dc, float)
            # round(x, 2) -- at most 2 decimal places
            assert round(ffmc, 2) == ffmc
            assert round(dmc, 2) == dmc
            assert round(dc, 2) == dc

    def test_empty_input_returns_empty_dict(self):
        assert fwi.spinup_daily_codes([]) == {}

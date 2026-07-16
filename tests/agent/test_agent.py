"""Tests for scripts/agent/agent.py.

Covers `_compact`, `_params_card`, `_result_card` -- pure dict/string
transformations over an already-computed `run_scenario()` result dict, with no I/O
and no LLM calls -- plus `WfedsAgent.chat`/`_run_tool`, which DO call litellm (via
`tests.helpers.fake_litellm`, since agent.py imports litellm lazily -- see that
helper's docstring) and `run_scenario` (mocked directly on the `run_scenario`
module, since `_run_tool` re-imports it fresh on every call).
"""

import json
from datetime import datetime, timezone

import pytest
from unittest.mock import Mock

import agent
import build_cell2fire_instance as bci
import run_scenario as run_scenario_module
from agent import WfedsAgent, _compact, _params_card, _result_card, _to_utc
from build_cell2fire_instance import InstanceInputError
from tests.helpers.fake_litellm import install_fake_litellm, make_response


def _per_hour_row(period, cut_off, at_risk, fire_km2=1.0, population=100,
                  routed=0, impacted=0):
    return {"period": period, "fire_km2": fire_km2, "at_risk": at_risk,
            "population": population, "routed": routed, "cut_off": cut_off,
            "impacted": impacted}


def _make_result(inputs, final_hour=None, per_hour=None,
                 final_front_class4_pct=None, run_id="20240101_000000_test",
                 elapsed_min=1.2):
    return {
        "run_id": run_id,
        "inputs": inputs,
        "final_hour": final_hour or {},
        "final_front_class4_pct": final_front_class4_pct,
        "per_hour": per_hour or [],
        "elapsed_min": elapsed_min,
    }


# --------------------------------------------------------------------------------
# _compact
# --------------------------------------------------------------------------------
class TestCompact:
    def test_dedups_change_hours_on_status_change(self):
        per_hour = [
            _per_hour_row(period=0, cut_off=0, at_risk=2),
            _per_hour_row(period=1, cut_off=0, at_risk=2),   # unchanged -> dropped
            _per_hour_row(period=2, cut_off=0, at_risk=2),   # unchanged -> dropped
            _per_hour_row(period=3, cut_off=1, at_risk=2),   # cut_off changed -> kept
            _per_hour_row(period=4, cut_off=1, at_risk=3),   # at_risk changed -> kept
            _per_hour_row(period=5, cut_off=1, at_risk=3),   # unchanged -> dropped
        ]
        result = _make_result(inputs={"mode": "point_ignition"}, per_hour=per_hour)

        compact = _compact(result)

        periods = [row["period"] for row in compact["change_hours"]]
        assert periods == [0, 3, 4]

    def test_first_row_always_kept_even_with_no_prior_row(self):
        per_hour = [_per_hour_row(period=0, cut_off=0, at_risk=0)]
        result = _make_result(inputs={"mode": "point_ignition"}, per_hour=per_hour)

        compact = _compact(result)

        assert len(compact["change_hours"]) == 1
        assert compact["change_hours"][0]["period"] == 0

    def test_drops_bulky_per_hour_key(self):
        per_hour = [_per_hour_row(period=0, cut_off=0, at_risk=0)]
        result = _make_result(inputs={"mode": "point_ignition"}, per_hour=per_hour)

        compact = _compact(result)

        assert "per_hour" not in compact

    def test_change_hours_only_carries_the_summary_fields(self):
        per_hour = [_per_hour_row(period=0, cut_off=0, at_risk=0)]
        result = _make_result(inputs={"mode": "point_ignition"}, per_hour=per_hour)

        compact = _compact(result)

        assert set(compact["change_hours"][0]) == {
            "period", "fire_km2", "at_risk", "population", "routed",
            "cut_off", "impacted"}

    def test_inputs_subset_only_includes_the_listed_keys(self):
        inputs = {"mode": "point_ignition", "window_km": 12, "start_time_utc": "t",
                  "horizon_h": 6, "front_cells": 0, "n_fronts_detected": None,
                  "scenario": None, "weather_source": "archive",
                  "some_other_field": "should not leak through"}
        result = _make_result(inputs=inputs, per_hour=[])

        compact = _compact(result)

        assert set(compact["inputs"]) == {
            "mode", "window_km", "start_time_utc", "horizon_h", "front_cells",
            "n_fronts_detected", "scenario", "weather_source"}


# --------------------------------------------------------------------------------
# _to_utc
# --------------------------------------------------------------------------------
class TestToUtc:
    def test_summer_local_time_is_utc_plus_3(self):
        """July = EEST (UTC+3): the screenshot bug - the user typed their wall
        clock (15:15) where the engine expected UTC; now the conversion is ours."""
        assert _to_utc("2026-07-10T15:15") == "2026-07-10T12:15"

    def test_winter_local_time_is_utc_plus_2(self):
        """January = EET (UTC+2) - proves real DST handling, not a fixed offset."""
        assert _to_utc("2026-01-10T15:15") == "2026-01-10T13:15"

    def test_explicit_utc_offset_is_respected_not_shifted(self):
        assert _to_utc("2026-07-10T12:15+00:00") == "2026-07-10T12:15"

    def test_slashes_are_tolerated(self):
        """The real user typed `2026/07/10T15:15` (slashes) in the screenshot."""
        assert _to_utc("2026/07/10T15:15") == "2026-07-10T12:15"

    def test_none_means_the_current_utc_minute(self):
        # _to_utc returns the engine's NAIVE-UTC string; strip tzinfo from the
        # aware "now" bounds so the three datetimes are comparable.
        before = (datetime.now(timezone.utc)
                  .replace(second=0, microsecond=0, tzinfo=None))
        got = datetime.fromisoformat(_to_utc(None))
        after = datetime.now(timezone.utc).replace(tzinfo=None)
        assert before <= got <= after

    def test_unparseable_input_raises_value_error(self):
        with pytest.raises(ValueError):
            _to_utc("αύριο το πρωί")


# --------------------------------------------------------------------------------
# _params_card
# --------------------------------------------------------------------------------
class TestParamsCard:
    COMMON = dict(horizon_h=6, start_time_utc="2024-08-10T12:00",
                 wind_dir_source="era5", fwi_spinup_start="2024-06-01")

    def test_point_ignition_branch(self):
        inputs = dict(self.COMMON, mode="point_ignition",
                      ignition_points_wgs84=[(38.9000, 23.1000)],
                      window_km=12.0, grid={"nrows": 100, "ncols": 100, "cell_m": 30})
        result = _make_result(inputs)

        card = _params_card(result)

        assert "1 pin (αποτύπωμα ~375 μ.)" in card
        assert "12 km" in card

    def test_front_branch(self):
        inputs = dict(self.COMMON, mode="front", front_cells=42,
                      ignition_points_wgs84=[(38.90, 23.10), (38.905, 23.105)],
                      window_km=10.0, grid={"nrows": 80, "ncols": 80, "cell_m": 30})
        result = _make_result(inputs)

        card = _params_card(result)

        assert "μέτωπο (42 κελιά)" in card

    def test_multi_front_branch_shows_front_count_and_total_cells(self):
        """The NEW branch (multi-point clusters that split into >=2 independent
        fronts): both n_fronts_detected AND the total front-cell count must show
        up in the Greek text."""
        inputs = dict(self.COMMON, mode="multi_front", n_fronts_detected=3,
                      front_cells=57,
                      ignition_points_wgs84=[(38.90, 23.10), (38.905, 23.105),
                                            (38.95, 23.20), (38.951, 23.201)],
                      window_km=16.0, grid={"nrows": 150, "ncols": 150, "cell_m": 30})
        result = _make_result(inputs)

        card = _params_card(result)

        assert "3 ανεξάρτητα μέτωπα (57 κελιά συνολικά)" in card

    def test_scenario_branch(self):
        inputs = dict(self.COMMON, mode="scenario", scenario="north_evia_2021",
                      ignition_points_wgs84="scenario", window_km=None,
                      grid={"nrows": 200, "ncols": 200, "cell_m": 30})
        result = _make_result(inputs)

        card = _params_card(result)

        assert "σενάριο αναφοράς: north_evia_2021" in card
        assert "VIIRS bbox" in card

    def test_start_time_shows_greek_local_clock_and_utc(self):
        """COMMON's start is 2024-08-10T12:00 UTC; August = EEST (UTC+3), so the
        card must lead with the user's wall clock (15:00) and keep UTC visible."""
        inputs = dict(self.COMMON, mode="point_ignition",
                      ignition_points_wgs84=[(38.9000, 23.1000)],
                      window_km=12.0, grid={"nrows": 100, "ncols": 100, "cell_m": 30})
        result = _make_result(inputs)

        card = _params_card(result)

        assert "2024-08-10 15:00 ώρα Ελλάδας" in card
        assert "(2024-08-10 12:00 UTC)" in card


# --------------------------------------------------------------------------------
# _result_card
# --------------------------------------------------------------------------------
class TestResultCard:
    def test_renders_all_fields_when_present(self):
        final_hour = {"period": 6, "fire_km2": 3.2, "at_risk": 4, "population": 850,
                      "routed": 2, "cut_off": 1, "impacted": 1, "edges_removed": 5,
                      "longest_route_km": 7.5}
        result = _make_result(inputs={"mode": "point_ignition"}, final_hour=final_hour,
                              final_front_class4_pct=37)

        card = _result_card(result)

        assert "τελική ώρα +6" in card
        assert "3.2" in card
        assert "4 οικισμοί" in card
        assert "850" in card
        assert "Εκκενώνονται: 2" in card
        assert "Αποκλεισμένοι: 1" in card
        assert "Στο μέτωπο: 1" in card
        assert "Κλειστά τμήματα δρόμων: 5" in card
        assert "Μεγαλύτερη διαδρομή: 7.5 km" in card
        assert "37%" in card

    def test_omits_optional_fields_when_absent(self):
        """No `longest_route_km` in final_hour, no `final_front_class4_pct` on the
        result at all -- both optional lines must be OMITTED (not rendered as a
        literal "None"), and building the card must not crash."""
        final_hour = {"period": 3, "fire_km2": 1.0, "at_risk": 1, "population": 50,
                      "routed": 1, "cut_off": 0, "impacted": 0, "edges_removed": 0}
        result = _make_result(inputs={"mode": "point_ignition"}, final_hour=final_hour)
        assert "final_front_class4_pct" not in result or \
            result["final_front_class4_pct"] is None

        card = _result_card(result)

        assert "Μεγαλύτερη διαδρομή" not in card
        assert "Μέτωπο κλάσης 4" not in card
        assert "None" not in card

    def test_empty_final_hour_does_not_crash(self):
        result = _make_result(inputs={"mode": "point_ignition"}, final_hour={})

        card = _result_card(result)

        assert "None" not in card
        assert "τελική ώρα +?" in card


# --------------------------------------------------------------------------------
# WfedsAgent.chat / _run_tool
# --------------------------------------------------------------------------------
@pytest.fixture
def wfeds_agent(monkeypatch):
    """A WfedsAgent with get_model() bypassed (agent.py imports it at MODULE
    level -- `from llm_config import get_model` -- so `agent.get_model` is a
    real, patchable attribute on the consuming module), since llm_config.get_model
    raises RuntimeError unless a real provider API key is set in the environment."""
    monkeypatch.setattr(agent, "get_model", lambda preset=None: ("fake/model", "fake-key"))
    return WfedsAgent()


class TestChat:
    def test_plain_text_response_no_tool_call(self, monkeypatch, wfeds_agent):
        mock = install_fake_litellm(monkeypatch, [make_response(content="  γεια σου  ")])

        reply, files, cards = wfeds_agent.chat("hi")

        assert reply == "γεια σου"
        assert files == []
        assert cards == []
        assert mock.call_count == 1

    def test_one_tool_call_then_final_text(self, monkeypatch, wfeds_agent):
        install_fake_litellm(monkeypatch, [
            make_response(tool_calls=[("call_1", "run_fire_scenario",
                                       '{"use_pins": true, "horizon_h": 6}')]),
            make_response(content="όλα καλά"),
        ])
        run_tool_mock = Mock(return_value=('{"ok": true}', [], []))
        monkeypatch.setattr(WfedsAgent, "_run_tool", run_tool_mock)

        reply, files, cards = wfeds_agent.chat("τρέξε το σενάριο", pins=[(38.9, 23.1)])

        assert run_tool_mock.call_count == 1
        # NB: monkeypatching a plain Mock onto the CLASS (WfedsAgent._run_tool)
        # replaces the descriptor a real method would be -- Mock has no
        # `__get__`, so `self._run_tool(...)` does NOT implicitly bind/prepend
        # `self`; the mock is called with exactly (args, pins, progress_cb).
        call_args = run_tool_mock.call_args
        passed_args = call_args.args[0]
        assert passed_args == {"use_pins": True, "horizon_h": 6}
        assert reply == "όλα καλά"

    def test_hits_tool_call_round_ceiling_and_stops_gracefully(self, monkeypatch, wfeds_agent):
        """chat() iterates `for _ in range(4)` -- if the model NEVER returns a
        final (non-tool-call) answer, the loop must stop after exactly 4 rounds
        and return a graceful "I stopped" message rather than looping forever or
        crashing. Supplying exactly 4 canned responses proves the bound is really
        4: a Mock's side_effect list raises StopIteration if called a 5th time,
        so this test would fail loudly if the loop bound were missing or wrong."""
        responses = [make_response(tool_calls=[(f"call_{i}", "run_fire_scenario", "{}")])
                    for i in range(4)]
        install_fake_litellm(monkeypatch, responses)
        monkeypatch.setattr(WfedsAgent, "_run_tool", Mock(return_value=("ok", [], [])))

        reply, files, cards = wfeds_agent.chat("τρέξε συνέχεια")

        assert reply == "Σταμάτησα - πολλές διαδοχικές κλήσεις εργαλείου."


class TestRunTool:
    @pytest.fixture(autouse=True)
    def _bypass_prevalidation(self, monkeypatch):
        """_run_tool now fail-fasts through bci._validate_user_inputs BEFORE the
        progress message / run_scenario call. These tests target the run/error
        paths, so the (raster-opening) validation is stubbed out; the tests that
        exercise the pre-validation itself override this stub."""
        monkeypatch.setattr(bci, "_validate_user_inputs", Mock(return_value=None))

    def test_instance_input_error_passed_through_with_prefix(self, monkeypatch, wfeds_agent):
        monkeypatch.setattr(run_scenario_module, "run_scenario",
                            Mock(side_effect=InstanceInputError("χρειάζομαι pins")))

        content, files, cards = wfeds_agent._run_tool({"use_pins": True},
                                                       [(38.9, 23.1)], None)

        assert content == "INPUT ERROR: χρειάζομαι pins"
        assert files == []
        assert cards == []

    def test_unexpected_exception_becomes_pipeline_error(self, monkeypatch, wfeds_agent):
        monkeypatch.setattr(run_scenario_module, "run_scenario",
                            Mock(side_effect=ValueError("κάτι έσπασε")))

        content, files, cards = wfeds_agent._run_tool({"use_pins": True},
                                                       [(38.9, 23.1)], None)

        assert content == "PIPELINE ERROR: ValueError: κάτι έσπασε"
        assert files == []
        assert cards == []

    def test_only_existing_files_are_kept(self, monkeypatch, wfeds_agent, tmp_path):
        existing_file = tmp_path / "map.png"
        existing_file.write_bytes(b"x")
        missing_file = tmp_path / "dashboard.html"   # never created

        result = _make_result(inputs={"mode": "point_ignition"})
        result["files"] = {"map_png": str(existing_file), "map_anim": "",
                           "dashboard": str(missing_file)}
        monkeypatch.setattr(run_scenario_module, "run_scenario", Mock(return_value=result))

        content, files, cards = wfeds_agent._run_tool({"use_pins": True},
                                                       [(38.9, 23.1)], None)

        assert files == [str(existing_file)]
        # sanity: the tool succeeded (not an error string) and produced the two cards
        assert json.loads(content)["run_id"] == result["run_id"]
        assert len(cards) == 2

    def test_prevalidation_error_skips_progress_and_run(self, monkeypatch, wfeds_agent):
        """Fail fast: invalid input must return INPUT ERROR *without* posting the
        misleading '⏳ Τρέχω το σενάριο (~1-3 λεπτά)...' or touching the engine."""
        monkeypatch.setattr(bci, "_validate_user_inputs",
                            Mock(side_effect=InstanceInputError("κακό παράθυρο")))
        run_mock = Mock()
        monkeypatch.setattr(run_scenario_module, "run_scenario", run_mock)
        progress = Mock()

        content, files, cards = wfeds_agent._run_tool(
            {"use_pins": True, "window_km": 99}, [(38.9, 23.1)], progress)

        assert content == "INPUT ERROR: κακό παράθυρο"
        progress.assert_not_called()
        run_mock.assert_not_called()

    def test_local_start_time_reaches_the_engine_as_utc(self, monkeypatch, wfeds_agent):
        """The user's July wall clock (15:15 EEST) must arrive at run_scenario
        as 12:15 UTC - the exact silent-input bug from the field test."""
        result = _make_result(inputs={"mode": "point_ignition"})
        result["files"] = {"map_png": "", "map_anim": "", "dashboard": ""}
        run_mock = Mock(return_value=result)
        monkeypatch.setattr(run_scenario_module, "run_scenario", run_mock)

        wfeds_agent._run_tool({"use_pins": True, "window_km": 12,
                               "start_time": "2026-07-10T15:15"},
                              [(38.9, 23.1)], None)

        assert run_mock.call_args.kwargs["start_time"] == "2026-07-10T12:15"

    def test_missing_start_time_defaults_to_now_utc(self, monkeypatch, wfeds_agent):
        """No start_time from the LLM + pins = 'live fire': the tool fills in
        the current UTC minute instead of asking the user for the clock."""
        result = _make_result(inputs={"mode": "point_ignition"})
        result["files"] = {"map_png": "", "map_anim": "", "dashboard": ""}
        run_mock = Mock(return_value=result)
        monkeypatch.setattr(run_scenario_module, "run_scenario", run_mock)

        wfeds_agent._run_tool({"use_pins": True, "window_km": 12},
                              [(38.9, 23.1)], None)

        passed = run_mock.call_args.kwargs["start_time"]
        got = datetime.fromisoformat(passed)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        assert abs((now - got).total_seconds()) < 120

    def test_scenario_run_keeps_its_locked_start_time(self, monkeypatch, wfeds_agent):
        """Named scenarios have a locked start inside the engine: no 'now'
        default, no pre-validation (window/pins are locked too)."""
        validate_mock = Mock()
        monkeypatch.setattr(bci, "_validate_user_inputs", validate_mock)
        result = _make_result(inputs={"mode": "scenario"})
        result["files"] = {"map_png": "", "map_anim": "", "dashboard": ""}
        run_mock = Mock(return_value=result)
        monkeypatch.setattr(run_scenario_module, "run_scenario", run_mock)

        wfeds_agent._run_tool({"scenario": "north_evia_2021"}, None, None)

        assert run_mock.call_args.kwargs["start_time"] is None
        validate_mock.assert_not_called()

    def test_unparseable_start_time_is_an_input_error(self, monkeypatch, wfeds_agent):
        run_mock = Mock()
        monkeypatch.setattr(run_scenario_module, "run_scenario", run_mock)

        content, files, cards = wfeds_agent._run_tool(
            {"use_pins": True, "window_km": 12, "start_time": "αύριο"},
            [(38.9, 23.1)], None)

        assert content.startswith("INPUT ERROR")
        assert "αύριο" in content
        run_mock.assert_not_called()

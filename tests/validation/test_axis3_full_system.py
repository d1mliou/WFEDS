from scripts.validation.axis3_full_system import (
    bounds,
    clean_label_record,
    deterministic_pass,
    passk_bounds,
)


def test_clean_label_record_rejects_both_negative_labels():
    assert clean_label_record({"claims": [{"label": "SUPPORTED"}]}) is True
    assert clean_label_record({"claims": [{"label": "NOT_IN_RESULT"}]}) is False
    assert clean_label_record({"claims": [{"label": "CONTRADICTED"}]}) is False
    assert clean_label_record(None) is None


def test_deterministic_pass_includes_numeric_traceability():
    base = {
        "stage_a_ok": True,
        "stage_b_ok": True,
        "numeric_ok": True,
        "policy_problems": [],
        "error": None,
    }
    assert deterministic_pass(base) is True
    assert deterministic_pass({**base, "numeric_ok": False}) is False
    assert deterministic_pass({**base, "policy_problems": ["bad"]}) is False


def test_bounds_treat_unresolved_as_interval():
    assert bounds([True, False, None, True]) == (0.5, 0.75, 1)


def test_passk_bounds_requires_every_repetition():
    rows = [
        {"case_id": "A", "full_system": True},
        {"case_id": "A", "full_system": True},
        {"case_id": "B", "full_system": True},
        {"case_id": "B", "full_system": None},
        {"case_id": "C", "full_system": True},
        {"case_id": "C", "full_system": False},
    ]
    lo, hi, pending = passk_bounds(rows)
    assert lo == 1 / 3
    assert hi == 2 / 3
    assert pending == 1


# --------------------------------------------------------------------------------
# Block C of section 4.3 of the structured-output pre-registration, plus the
# author's C1a correction of 2026-09-21. One test per edit.
#
# Every synthetic row below carries `narrative_surface` explicitly whenever the
# arm is structured. That is not decoration: `narrative_surface_of` falls back to
# `agent_eval._narrative_surface` only when the key is absent, and importing
# `agent_eval` resolves the OneDrive data directory, which a test must never
# depend on.
# --------------------------------------------------------------------------------
import csv
import json
from pathlib import Path

import pytest

from scripts.validation.axis3_full_system import (
    AUDIT_FIELDS,
    AUDIT_WITHHELD_FIELDS,
    COVERAGE_SHEET_FIELDS,
    CT_RECORD_FIELDS,
    DEFAULT_AUDIT_SEED,
    NARRATION_MIN_CHARS,
    analyse_run,
    arm_of,
    judgeable_surface,
    numeric_verdict,
    parse_labels,
    structured_complete_of,
    surface_reason,
    write_coverage_sheet,
    write_records,
)

REPO = Path(__file__).resolve().parents[2]
PROSE = "Η φωτιά επεκτείνεται. " * 20          # comfortably over the 120-char floor
COMPACT = {"change_hours": [{"period": 0, "cut_off": 0}, {"period": 4, "cut_off": 2}],
           "final_hour": {"period": 6, "cut_off": 2}}


def make_row(**over):
    """A minimal in-scope raw row that passes every gate, so a test can break
    exactly one thing and attribute the result to it."""
    row = {
        "case_id": "B04", "rep": 1, "category": "baseline",
        "user_text": "τι γίνεται;", "pins": None,
        "reply": PROSE, "error": None,
        "tool_outputs": [json.dumps(COMPACT)],
        "n_tool_calls": 1,
        "stage_a_ok": True, "stage_b_ok": True,
        "numeric_ok": True, "suspicious_numbers": [],
        "policy_problems": [], "cutoff_hours_to_cover": [],
    }
    row.update(over)
    return row


def make_run(tmp_path, rows, labels, stage_dir="stage_c_full"):
    """Write a synthetic run directory in the layout `analyse_run` expects."""
    run_dir = tmp_path / "run"
    stage = run_dir / stage_dir
    stage.mkdir(parents=True, exist_ok=True)
    (run_dir / "agent_eval_raw.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (run_dir / "agent_eval_metrics.json").write_text(
        json.dumps({"model": "test-model"}), encoding="utf-8")
    (stage / "batch_01.json").write_text(
        json.dumps([{"uid": f"{r['case_id']}#{r['rep']}", "category": r["category"]}
                    for r in rows], ensure_ascii=False), encoding="utf-8")
    for name, records in labels.items():
        (stage / name).write_text(json.dumps(records, ensure_ascii=False),
                                  encoding="utf-8")
    return run_dir


def clean_labels(rows):
    return [{"uid": f"{r['case_id']}#{r['rep']}",
             "claims": [{"label": "SUPPORTED"}], "notes": ""} for r in rows]


# --- C4 ------------------------------------------------------------------------
def test_c4_empty_claims_list_is_none_not_true():
    """An unjudged record must not walk into the clean-consensus branch."""
    assert clean_label_record({"claims": []}) is None
    assert clean_label_record({}) is None
    assert clean_label_record({"claims": [{"label": "SUPPORTED"}]}) is True


# --- the C1a correction ---------------------------------------------------------
def test_numeric_verdict_prefers_the_full_surface_value():
    row = make_row(numeric_ok=True, numeric_ok_full_surface=False)
    assert numeric_verdict(row) == (False, "full_surface")
    row = make_row(numeric_ok=False, numeric_ok_full_surface=True)
    assert numeric_verdict(row) == (True, "full_surface")


def test_numeric_verdict_falls_back_only_when_the_full_surface_value_is_absent():
    """The free arm's signature: no key at all, or the key present and None."""
    assert numeric_verdict(make_row()) == (True, "prose_only_fallback")
    assert numeric_verdict(make_row(numeric_ok_full_surface=None)) == (
        True, "prose_only_fallback")
    assert numeric_verdict(make_row(numeric_ok=False)) == (
        False, "prose_only_fallback")


def test_deterministic_pass_uses_the_full_surface_verdict_not_the_provisional_one():
    """The gate correction: a row the prose-only check passed and the 2.1 surface
    failed must fail, which is the whole point of the author's instruction."""
    assert deterministic_pass(make_row(numeric_ok=True,
                                       numeric_ok_full_surface=False)) is False
    assert deterministic_pass(make_row(numeric_ok=False,
                                       numeric_ok_full_surface=True)) is True


def test_report_counts_which_numeric_path_each_row_took(tmp_path):
    rows = [make_row(rep=1), make_row(rep=2, output_contract="structured",
                                      narrative_surface=PROSE,
                                      numeric_ok_full_surface=True)]
    run = make_run(tmp_path, rows, {"stage_c_labels.json": clean_labels(rows),
                                    "stage_c_labels_pass2.json": clean_labels(rows)})
    summary, _, _ = analyse_run("r", run, {})
    assert summary["numeric_source_full_surface"] == 1
    assert summary["numeric_source_prose_fallback"] == 1


# --- C2 and C3 -----------------------------------------------------------------
def test_judgeable_surface_needs_all_three_conditions():
    assert judgeable_surface(make_row()) is True
    assert judgeable_surface(make_row(error="boom")) is False
    assert judgeable_surface(make_row(tool_outputs=[])) is False
    assert judgeable_surface(make_row(reply="μικρό")) is False
    # Exactly at the floor, because an off-by-one here silently rescores an arm.
    assert judgeable_surface(make_row(reply="x" * NARRATION_MIN_CHARS)) is True
    assert judgeable_surface(make_row(reply="x" * (NARRATION_MIN_CHARS - 1))) is False


def test_c2_c3_no_surface_row_fails_both_and_never_reaches_the_upper_bound(tmp_path):
    good = make_row(rep=1)
    short = make_row(rep=2, output_contract="structured", narrative_surface="",
                     reply='{"status":"ok"}', contract_outcome="valid",
                     cutoff_hours_to_cover=[4])
    rows = [good, short]
    run = make_run(tmp_path, rows, {"stage_c_labels.json": clean_labels(rows),
                                    "stage_c_labels_pass2.json": clean_labels(rows)})
    summary, records, audit_rows = analyse_run("r", run, {})
    bad = next(r for r in records if r["uid"] == "B04#2")

    assert bad["judgeable_surface"] is False
    assert bad["stage_c"] is False
    assert bad["critical_coverage"] is False          # never True, never None
    assert bad["full_system"] is False
    # The reason is the contract outcome, so the records CSV says WHY.
    assert bad["stage_c_source"] == "valid"
    assert bad["critical_coverage_source"] == "valid"
    # Never pending, so it cannot be counted into the optimistic bound.
    assert summary["full_system_pass1_pending"] == 0
    assert summary["full_system_pass1_upper"] == 0.5
    assert summary["no_judgeable_surface"] == 1
    assert summary["c1a_excluded_no_surface"] == 1
    assert summary["c1a_denominator"] == 1
    # And it is not offered to a human: 2.11 says these rows are not audited.
    assert [r["uid"] for r in audit_rows if r["human_stage_c_required"] == "YES"] == []
    assert [r["uid"] for r in audit_rows
            if r["human_critical_coverage_required"] == "YES"] == []


def test_c2_no_surface_beats_a_stale_human_pass(tmp_path):
    """Position is the edit. Below the override branch, a human PASS left in the
    audit CSV would resurrect a record that had no surface to be judged on."""
    rows = [make_row(error="boom")]
    run = make_run(tmp_path, rows, {"stage_c_labels.json": clean_labels(rows),
                                    "stage_c_labels_pass2.json": clean_labels(rows)})
    stale = {("r", "B04#1"): {"human_stage_c": "PASS",
                              "human_critical_coverage": "PASS"}}
    _, records, _ = analyse_run("r", run, stale)
    assert records[0]["stage_c"] is False
    assert records[0]["stage_c_source"] == "runtime_error"
    assert records[0]["critical_coverage"] is False


def test_surface_reason_prefers_the_contract_outcome():
    assert surface_reason(make_row(contract_outcome="not_json")) == "not_json"
    assert surface_reason(make_row(error="boom")) == "runtime_error"
    assert surface_reason(make_row(reply="")) == "no_judgeable_surface"


# --- C1 -------------------------------------------------------------------------
def test_c1_single_label_file_is_joinable(tmp_path):
    rows = [make_row(rep=1)]
    run = make_run(tmp_path, rows, {"pass_one.json": clean_labels(rows)},
                   stage_dir="stage_c_v2")
    summary, records, _ = analyse_run("r", run, {}, "stage_c_v2", ("pass_one.json",))
    assert summary["label_passes"] == 1
    assert records[0]["stage_c"] is True
    # Named apart from the two-pass verdict, because one pass is weaker evidence.
    assert records[0]["stage_c_source"] == "single_pass_clean"
    assert records[0]["stage_c_pass2"] is None


def test_c1_two_pass_source_name_is_unchanged(tmp_path):
    rows = [make_row(rep=1)]
    run = make_run(tmp_path, rows, {"stage_c_labels.json": clean_labels(rows),
                                    "stage_c_labels_pass2.json": clean_labels(rows)})
    _, records, _ = analyse_run("r", run, {})
    assert records[0]["stage_c_source"] == "two_pass_clean_consensus"


def test_parse_labels_splits_and_rejects_empty():
    assert parse_labels("a.json, b.json") == ("a.json", "b.json")
    assert parse_labels("a.json") == ("a.json",)
    with pytest.raises(Exception):
        parse_labels(" , ")


# --- C5a ------------------------------------------------------------------------
def test_c5a_records_carry_the_arm_and_every_ct_column_on_both_arms(tmp_path):
    free = make_row(rep=1)
    structured = make_row(rep=2, output_contract="structured",
                          narrative_surface=PROSE, contract_outcome="valid",
                          schema_conformant=True, contract_problems=[],
                          ct_value_mismatches=[], slot_claims=["Στην ώρα 4: 2."],
                          schema_errors=[], ct_exact_set=True, ct_scored=True)
    rows = [free, structured]
    run = make_run(tmp_path, rows, {"stage_c_labels.json": clean_labels(rows),
                                    "stage_c_labels_pass2.json": clean_labels(rows)})
    _, records, _ = analyse_run("r", run, {})
    a, b = records
    assert a["arm"] == "free" and b["arm"] == "structured"
    for field in ("contract_outcome", "judged_surface_chars", "schema_conformant",
                  "structured_complete", "slot_mismatches", "schema_errors",
                  "rendered_slot_claims"):
        assert field in a and field in b
    # Trap 1 of 3.4.6: uniform keys, or the pooled DictWriter raises or blanks.
    for field in CT_RECORD_FIELDS:
        assert field in a and field in b
    assert a["ct_exact_set"] == ""            # free arm, no contract, not applicable
    assert b["ct_exact_set"] is True


def test_arm_of_reads_an_absent_key_as_the_baseline_arm():
    assert arm_of({}) == "free"
    assert arm_of({"output_contract": "structured"}) == "structured"


def test_structured_complete_decomposes_the_semantic_rules():
    assert structured_complete_of(make_row()) is None
    base = dict(output_contract="structured", narrative_surface=PROSE)
    assert structured_complete_of(make_row(**base, contract_outcome="valid",
                                           contract_problems=[])) is True
    assert structured_complete_of(make_row(
        **base, contract_outcome="validator_invalid",
        contract_problems=["rule 5: 1 slot value(s) differ"])) is True
    assert structured_complete_of(make_row(
        **base, contract_outcome="validator_invalid",
        contract_problems=["rule 3: periods [] != cutoff_events [4]"])) is False
    assert structured_complete_of(make_row(**base,
                                           contract_outcome="not_json")) is False


# --- C5b ------------------------------------------------------------------------
def test_c5b_audit_csv_gains_four_columns_loses_reply_and_withholds_the_rest():
    assert "reply" not in AUDIT_FIELDS
    for field in ("arm", "contract_outcome", "judged_surface",
                  "judged_surface_chars"):
        assert field in AUDIT_FIELDS
    for field in AUDIT_WITHHELD_FIELDS:
        assert field not in AUDIT_FIELDS
    # 3.4.2: no diagnostic may reach an audit decision column either.
    assert not [f for f in AUDIT_FIELDS if f.startswith("ct_")]


def test_c5b_audit_row_keys_match_the_field_list_exactly(tmp_path):
    """Revision 2's trap: names added to AUDIT_FIELDS alone ship empty columns."""
    rows = [make_row(rep=1, cutoff_hours_to_cover=[4])]
    run = make_run(tmp_path, rows, {"stage_c_labels.json": clean_labels(rows),
                                    "stage_c_labels_pass2.json": clean_labels(rows)})
    _, _, audit_rows = analyse_run("r", run, {})
    assert set(audit_rows[0]) == set(AUDIT_FIELDS)
    assert audit_rows[0]["judged_surface"] == PROSE
    assert audit_rows[0]["judged_surface_chars"] == str(len(PROSE.strip()))


def test_c5b_literal_hours_are_computed_over_the_narrative_surface(tmp_path):
    """C6: on the structured arm the aid must not see the withheld slot values,
    so the raw envelope's hours must not appear in `literal_hours_found`."""
    envelope = json.dumps({"critical_transitions": [{"period": 4}]},
                          ensure_ascii=False)
    rows = [make_row(rep=1, output_contract="structured", reply=envelope,
                     narrative_surface=PROSE, cutoff_hours_to_cover=[4])]
    run = make_run(tmp_path, rows, {"stage_c_labels.json": clean_labels(rows),
                                    "stage_c_labels_pass2.json": clean_labels(rows)})
    _, _, audit_rows = analyse_run("r", run, {})
    assert audit_rows[0]["literal_hours_found"] == ""
    assert "critical_transitions" not in audit_rows[0]["judged_surface"]

    narrated = PROSE + " Στην 4η ώρα κόβεται ο δρόμος."
    rows = [make_row(rep=1, output_contract="structured", reply=envelope,
                     narrative_surface=narrated, cutoff_hours_to_cover=[4])]
    run = make_run(tmp_path / "b", rows,
                   {"stage_c_labels.json": clean_labels(rows),
                    "stage_c_labels_pass2.json": clean_labels(rows)})
    _, _, audit_rows = analyse_run("r", run, {})
    assert audit_rows[0]["literal_hours_found"] == "4"


# --- C9 -------------------------------------------------------------------------
def _coverage_rows():
    return [{"run_id": "luna_free", "uid": "B04#1", "critical_event_hours": "1,3",
             "user_text": "u1", "pins": "null", "judged_surface": "s1",
             "literal_hours_found": "1", "human_critical_coverage": ""},
            {"run_id": "luna_struct", "uid": "B04#1", "critical_event_hours": "1,3",
             "user_text": "u2", "pins": "null", "judged_surface": "s2",
             "literal_hours_found": "", "human_critical_coverage": "PASS"}]


def test_c9_coverage_sheet_is_arm_blind_and_its_key_is_a_separate_file(tmp_path):
    sheet, key = tmp_path / "sheet.csv", tmp_path / "key.csv"
    mapping = write_coverage_sheet(sheet, key, _coverage_rows())

    written = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert list(written[0]) == COVERAGE_SHEET_FIELDS
    # The arm is off the sheet in every form 2.11 names.
    for forbidden in ("run_id", "uid", "arm", "reply", "contract_outcome"):
        assert forbidden not in written[0]
    assert all(r["decision_id"].startswith("COV-") for r in written)
    assert all(r["decision_rule"] for r in written)
    # A partly finished sheet survives a rerun.
    assert sorted(r["human_critical_coverage"] for r in written) == ["", "PASS"]

    keyed = list(csv.DictReader(key.open(encoding="utf-8-sig")))
    assert {r["decision_id"] for r in keyed} == {r["decision_id"] for r in written}
    assert {r["run_id"] for r in keyed} == {"luna_free", "luna_struct"}
    assert all(r["seed"] == str(DEFAULT_AUDIT_SEED) for r in keyed)
    assert len(mapping) == 2


def test_c9_the_shuffle_is_seeded_and_therefore_reproducible(tmp_path):
    rows = [dict(r, uid=f"B04#{i}") for i in range(10) for r in _coverage_rows()]
    a = write_coverage_sheet(tmp_path / "s1.csv", tmp_path / "k1.csv", rows, 7)
    b = write_coverage_sheet(tmp_path / "s2.csv", tmp_path / "k2.csv", rows, 7)
    c = write_coverage_sheet(tmp_path / "s3.csv", tmp_path / "k3.csv", rows, 8)
    assert a == b
    assert a != c


# --- C7 -------------------------------------------------------------------------
def test_c7_write_records_survives_an_empty_row_list(tmp_path):
    path = tmp_path / "records.csv"
    write_records(path, [])
    assert path.exists()
    assert path.read_text(encoding="utf-8-sig") == ""


def test_write_records_joins_every_list_column(tmp_path):
    path = tmp_path / "records.csv"
    write_records(path, [{"uid": "B04#1", "critical_hours": [1, 3],
                          "ct_missing_periods": [4], "ct_extra_periods": [],
                          "ct_extra_in_payload": [], "ct_extra_not_in_payload": [],
                          "ct_duplicate_periods": [5],
                          "rendered_slot_claims": ["Στην ώρα 4: 2."],
                          "schema_errors": []}])
    written = list(csv.DictReader(path.open(encoding="utf-8-sig")))[0]
    assert written["critical_hours"] == "1,3"
    assert written["ct_missing_periods"] == "4"
    assert written["ct_duplicate_periods"] == "5"
    assert written["rendered_slot_claims"] == '["Στην ώρα 4: 2."]'
    assert "[" not in written["critical_hours"]


# --- the floor, pinned to its one source ----------------------------------------
def test_narration_floor_cannot_drift_from_the_prep_module():
    """The floor is declared here as a literal so this join stays importable
    offline, so the two copies are asserted equal by reading the other as text."""
    src = (REPO / "scripts" / "validation"
           / "agent_eval_prep_labeling.py").read_text(encoding="utf-8")
    assert f"NARRATION_MIN_CHARS = {NARRATION_MIN_CHARS}" in src

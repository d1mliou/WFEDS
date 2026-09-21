"""Tests for the Axis-3 structured-output contract validator.

The oracle for the golden test is section 3.4.5 of
`scripts/validation/axis3_structured_output_preregistration.md`: a worked table
over the four stored pilot records. It is pre-registered, so if the code and the
table disagree the code is wrong, and these tests are written to make that
disagreement impossible to miss rather than to be adjusted until they pass.

Everything else is synthetic, built on one small compact whose ground truth is
[1, 3, 4], so that each contract outcome and each metric edge can be exercised
without a model call and without touching the frozen pilot tree.
"""
import json
from pathlib import Path

import pytest

from scripts.validation.stage_c_render_slots import render_slot_claims
from scripts.validation.stage_c_structured import (
    ROUND_CEILING_REPLY,
    is_frozen_audit_path,
    main,
    per_row_lines,
    score_row,
    score_rows,
    summarize,
)

REPO = Path(__file__).resolve().parents[2]
PILOT = REPO / "scripts" / "validation" / "pilot_records" / "20260921_axis3_structured_contract"

# cut_off rises at periods 1, 3 and 4, so cutoff_events() is [1, 3, 4]. Period 2
# is a real payload row at which the count held flat: it is the in-payload
# redundancy channel, and any period outside {0..4} is the fabrication channel.
COMPACT = {
    "change_hours": [
        {"period": 0, "cut_off": 6},
        {"period": 1, "cut_off": 9},
        {"period": 2, "cut_off": 9},
        {"period": 3, "cut_off": 10},
        {"period": 4, "cut_off": 14},
    ],
    "final_hour": {"period": 5, "cut_off": 14, "edges_removed": 737},
}
GT = [1, 3, 4]
CUT_OFF = {0: 6, 1: 9, 2: 9, 3: 10, 4: 14}

GREEK_PROSE = ("Η φωτιά επεκτείνεται και ο αριθμός των οικισμών χωρίς διαδρομή "
               "διαφυγής αυξάνεται. Η τελική απόφαση ανήκει στον επιχειρησιακό "
               "υπεύθυνο.")


def obj(status="ok", initial=((0, 6),), critical=((1, 9), (3, 10), (4, 14)),
        final=((5, 14, 737),), **extra):
    """A contract object from period/value tuples, so a test says what it varies
    and nothing else."""
    out = {
        "status": status,
        "initial_state": [{"period": p, "settlements_without_route": v} for p, v in initial],
        "critical_transitions": [{"period": p, "settlements_without_route": v}
                                 for p, v in critical],
        "final_state": [{"period": p, "settlements_without_route": v,
                         "road_segments_removed": e} for p, v, e in final],
        "interpretation": GREEK_PROSE,
        "limitations": "Το μοντέλο δεν λαμβάνει υπόψη την κατάσβεση. Ενδεικτικό.",
    }
    out.update(extra)
    return out


def critical_from(periods):
    """`critical_transitions` tuples carrying the PAYLOAD value at each period, so
    a membership test is never confounded with a value mismatch. A period that is
    not in the payload carries a value the payload cannot confirm, which is
    exactly what fabrication looks like."""
    return tuple((p, CUT_OFF.get(p, 99)) for p in periods)


def row(reply, *, contract="structured", n_tool_calls=1, error=None,
        compact=COMPACT, finish_reason="stop", case_id="T01", rep=1):
    if not isinstance(reply, str):
        reply = json.dumps(reply, ensure_ascii=False)
    return {
        "case_id": case_id,
        "rep": rep,
        "output_contract": contract,
        "reply": reply,
        "error": error,
        "n_tool_calls": n_tool_calls,
        "tool_outputs": [json.dumps(compact, ensure_ascii=False)] if compact else [],
        "provider_meta": [{"contract_attached": True, "finish_reason": finish_reason}],
    }


# --------------------------------------------------------------------------------
# 2.4: the nine contract outcomes, and the precedence between them
# --------------------------------------------------------------------------------
def test_outcome_1_runtime_error_wins_over_every_later_test():
    # A provider exception leaves a perfectly good reply field unread: outcome 1
    # is first in the precedence order and nothing below it may re-open the row.
    r = row(obj(), error="BadRequestError: 403 PERMISSION_DENIED")
    assert score_row(r)["contract_outcome"] == "runtime_error"


def test_outcome_2_round_ceiling_is_the_harness_sentence():
    r = row(ROUND_CEILING_REPLY, n_tool_calls=4)
    assert score_row(r)["contract_outcome"] == "round_ceiling"


def test_outcome_3_contract_not_attached_when_no_tool_call_ever_happened():
    r = row(obj(), n_tool_calls=0, compact=None)
    assert score_row(r)["contract_outcome"] == "contract_not_attached"


def test_outcome_4_no_final_message_on_an_empty_reply():
    scored = score_row(row("   "))
    assert scored["contract_outcome"] == "no_final_message"
    # The finish reason is what 2.4 splits a refusal from a content filter on.
    assert score_row(row("", finish_reason="content_filter"))[
        "contract_outcome_finish_reason"] == "content_filter"


def test_outcome_5_not_json_only_on_the_structured_arm():
    assert score_row(row(GREEK_PROSE))["contract_outcome"] == "not_json"
    # A JSON array parses but is not an object, and no repair is attempted.
    assert score_row(row("[1, 2, 3]"))["contract_outcome"] == "not_json"


def test_outcome_6_schema_invalid_on_an_extra_root_key():
    scored = score_row(row(obj(extra_key=1)))
    assert scored["contract_outcome"] == "schema_invalid"
    assert scored["schema_conformant"] is False
    assert scored["schema_errors"]


def test_outcome_6_schema_invalid_on_a_status_outside_the_enum():
    scored = score_row(row(obj(status="OK")))
    assert scored["contract_outcome"] == "schema_invalid"


def test_outcome_7_validator_invalid_on_a_superset_of_the_ground_truth():
    scored = score_row(row(obj(critical=critical_from([1, 2, 3, 4]))))
    assert scored["contract_outcome"] == "validator_invalid"
    assert scored["schema_conformant"] is True
    assert any(p.startswith("rule 3") for p in scored["contract_problems"])


def test_outcome_7_validator_invalid_on_a_wrong_initial_or_final_period():
    assert any(p.startswith("rule 1") for p in
               score_row(row(obj(initial=((1, 9),))))["contract_problems"])
    # final_state must come from `final_hour` (period 5), not from the last
    # change_hours row (period 4): that confusion is what rule 2 exists to catch.
    assert any(p.startswith("rule 2") for p in
               score_row(row(obj(final=((4, 14, 737),))))["contract_problems"])


def test_outcome_8_valid():
    scored = score_row(row(obj()))
    assert scored["contract_outcome"] == "valid"
    assert scored["contract_problems"] == []
    assert scored["status_consistent"] is True
    assert scored["ct_exact_set"] is True


def test_outcome_9_free_text_is_never_not_json():
    # The whole point of the arm branch in 2.4: a free Greek reply must resolve to
    # free_text, or the baseline loses its judgeable surface and scores zero on
    # four criteria.
    scored = score_row(row(GREEK_PROSE, contract="free"))
    assert scored["contract_outcome"] == "free_text"
    assert scored["ct_scored"] is False
    assert scored["ct_not_scored_reason"] == "free_text"


def test_free_arm_still_takes_outcomes_1_to_4():
    # Outcomes 1 to 4 are arm-neutral; only 5 to 8 are contract-specific.
    assert score_row(row("", contract="free"))["contract_outcome"] == "no_final_message"
    assert score_row(row(GREEK_PROSE, contract="free", n_tool_calls=0,
                         compact=None))["contract_outcome"] == "contract_not_attached"


# --------------------------------------------------------------------------------
# 3.4.1 / 3.4.6: the three metrics and the 17 fields
# --------------------------------------------------------------------------------
def test_right_set_wrong_order_fails_both_exact_set_and_order():
    # Ordered list equality is strictly stronger than set equality: this row has
    # the correct set, perfect recall and no extras, and still fails the gate.
    scored = score_row(row(obj(critical=critical_from([3, 1, 4]))))
    assert scored["ct_exact_set"] is False
    assert scored["ct_order_ok"] is False
    assert scored["ct_matched"] == 3
    assert scored["ct_no_omission"] is True
    assert scored["ct_extra_periods"] == []
    assert scored["contract_outcome"] == "validator_invalid"


def test_duplicate_emission_is_an_extra_and_never_a_second_match():
    scored = score_row(row(obj(critical=critical_from([1, 1, 3, 4]))))
    assert scored["ct_duplicate_periods"] == [1]
    assert scored["ct_extra_periods"] == [1]          # multiset difference
    assert scored["ct_matched"] == 3                  # distinct GT periods only
    assert scored["ct_order_ok"] is True              # duplicates are not disorder
    assert scored["ct_exact_set"] is False


def test_extra_in_payload_is_redundancy_and_extra_outside_it_is_fabrication():
    redundant = score_row(row(obj(critical=critical_from([1, 2, 3, 4]))))
    assert redundant["ct_extra_periods"] == [2]
    assert redundant["ct_extra_in_payload"] == [2]
    assert redundant["ct_extra_not_in_payload"] == []

    fabricated = score_row(row(obj(critical=critical_from([1, 3, 4, 9]))))
    assert fabricated["ct_extra_periods"] == [9]
    assert fabricated["ct_extra_in_payload"] == []
    assert fabricated["ct_extra_not_in_payload"] == [9]


def test_omission_lowers_recall_without_producing_an_extra():
    scored = score_row(row(obj(critical=critical_from([1, 4]))))
    assert scored["ct_missing_periods"] == [3]
    assert scored["ct_matched"] == 2
    assert scored["ct_n_gt"] == 3
    assert scored["ct_no_omission"] is False
    assert scored["ct_over_inclusive"] is False


def test_value_mismatch_is_separate_from_the_exact_set():
    # The set is exact and the gate still fails, but through rule 5 rather than
    # rule 3: membership and arithmetic are different findings.
    scored = score_row(row(obj(critical=((1, 9), (3, 11), (4, 14)))))
    assert scored["ct_exact_set"] is True
    assert scored["ct_value_mismatches"] == [
        {"slot": "critical_transitions", "period": 3, "field": "settlements_without_route",
         "emitted": 11, "payload": 10}]
    assert any(p.startswith("rule 5") for p in scored["contract_problems"])
    assert scored["contract_outcome"] == "validator_invalid"


def test_value_mismatches_cover_the_other_two_slots_and_name_the_field():
    scored = score_row(row(obj(initial=((0, 7),), final=((5, 14, 700),))))
    slots = {(m["slot"], m["field"]): (m["emitted"], m["payload"])
             for m in scored["ct_value_mismatches"]}
    assert slots[("initial_state", "settlements_without_route")] == (7, 6)
    assert slots[("final_state", "road_segments_removed")] == (700, 737)


def test_status_not_ok_is_scored_and_fails_the_gate():
    # 3.4.1 pre-declares this so nobody later excludes it as "no attempt".
    scored = score_row(row(obj(status="abstained", initial=(), critical=(), final=())))
    assert scored["ct_scored"] is True
    assert scored["ct_exact_set"] is False
    assert scored["ct_matched"] == 0
    assert scored["ct_extra_periods"] == []
    assert scored["status_consistent"] is True


def test_non_ok_status_with_three_empty_arrays_is_not_contract_valid():
    # THE BLOCKING DEFECT of the structured pass, in one row. This record is
    # schema-conformant and satisfies rule 4 (a non-ok status with three empty
    # arrays is exactly what the schema's descriptions demand), so while rules
    # 1, 2, 3 and 5 were all scoped to status == "ok" nothing tested it at all
    # and it scored `valid` while ct_exact_set was False. 3.4.1 pre-declares
    # that this record is in scope and scores ct_exact_set False, and 3.4.2
    # pre-declares that a rule-3 failure IS validator_invalid, so scoring it
    # valid put a pre-declared contract FAILURE into the numerator of
    # contract_validity_rate and of SSP. The tool succeeded on this row: the
    # payload is there and its ground truth is non-empty.
    scored = score_row(row(obj(status="abstained", initial=(), critical=(),
                               final=())))
    assert scored["schema_conformant"] is True
    assert scored["status_consistent"] is True
    assert scored["contract_outcome"] == "validator_invalid"
    assert scored["contract_outcome"] != "valid"
    assert any(p.startswith("rule 3") for p in scored["contract_problems"])
    assert scored["ct_gt_periods"] == GT
    assert scored["ct_emitted_periods"] == []
    assert scored["ct_exact_set"] is False


def test_a_non_ok_status_on_a_degenerate_payload_is_still_valid():
    # The other side of the unconditional rule 3, so the fix cannot be read as
    # "non-ok is always invalid": where the payload has no rising cut_off at
    # all, an empty critical_transitions IS the ground truth and the record is
    # contract-valid. Rule 3 asks one question and it is answerable at any
    # status; it does not punish the status.
    flat = {"change_hours": [{"period": 0, "cut_off": 6},
                             {"period": 1, "cut_off": 6}],
            "final_hour": {"period": 2, "cut_off": 6, "edges_removed": 10}}
    scored = score_row(row(obj(status="tool_error", initial=(), critical=(),
                               final=()), compact=flat))
    assert scored["ct_gt_periods"] == []
    assert scored["contract_problems"] == []
    assert scored["contract_outcome"] == "valid"


def test_status_ok_with_empty_arrays_is_inconsistent():
    scored = score_row(row(obj(initial=(), critical=(), final=())))
    assert scored["status_consistent"] is False
    assert any(p.startswith("rule 4") for p in scored["contract_problems"])


def test_empty_ground_truth_leaves_exact_set_defined():
    # A payload whose cut_off never rises. Recall is undefined (the record is
    # dropped from that denominator only); exact-set still requires an empty
    # emitted list. The rule-4 problem is the pre-registration's own collision
    # between 1.3 rule 4 and the degenerate-GT paragraph of 3.4.1, recorded here
    # rather than patched: it cannot arise on the 13 in-scope cases.
    flat = {"change_hours": [{"period": 0, "cut_off": 6}, {"period": 1, "cut_off": 6}],
            "final_hour": {"period": 2, "cut_off": 6, "edges_removed": 10}}
    scored = score_row(row(obj(initial=((0, 6),), critical=(), final=((2, 6, 10),)),
                           compact=flat))
    assert scored["ct_n_gt"] == 0
    assert scored["ct_exact_set"] is True
    assert scored["ct_over_inclusive"] is False
    assert any(p.startswith("rule 4") for p in scored["contract_problems"])


def test_schema_invalid_row_with_a_well_formed_array_is_still_scored():
    scored = score_row(row(obj(critical=critical_from([1, 2, 3, 4]), extra_key=1)))
    assert scored["contract_outcome"] == "schema_invalid"
    assert scored["ct_scored"] is True
    assert scored["ct_extra_periods"] == [2]


def test_schema_invalid_row_without_a_usable_array_is_not_scored():
    broken = obj()
    broken["critical_transitions"] = ["1", "3"]
    scored = score_row(row(broken))
    assert scored["contract_outcome"] == "schema_invalid"
    assert scored["ct_scored"] is False
    assert scored["ct_not_scored_reason"] == "schema_invalid"


def test_boolean_period_is_not_read_as_an_integer():
    # True is an int in Python, so a JSON `true` would silently become period 1.
    broken = obj()
    broken["critical_transitions"] = [{"period": True, "settlements_without_route": 9}]
    assert score_row(row(broken))["ct_scored"] is False


def test_every_pre_registered_field_is_written_on_every_row():
    expected = {
        "ct_scored", "ct_not_scored_reason", "ct_gt_periods", "ct_n_gt",
        "ct_emitted_periods", "ct_n_emitted", "ct_matched", "ct_missing_periods",
        "ct_extra_periods", "ct_extra_in_payload", "ct_extra_not_in_payload",
        "ct_duplicate_periods", "ct_order_ok", "ct_exact_set", "ct_no_omission",
        "ct_over_inclusive", "ct_value_mismatches",
    }
    for r in (row(obj()), row(GREEK_PROSE, contract="free"),
              row(obj(), error="boom"), row("", n_tool_calls=0, compact=None)):
        assert expected <= set(score_row(r))


def test_ground_truth_comes_from_the_rows_own_tool_outputs():
    # Two rows of one file may carry different payloads (different cases), so the
    # ground truth is never a run-level constant.
    other = {"change_hours": [{"period": 0, "cut_off": 1}, {"period": 7, "cut_off": 4}],
             "final_hour": {"period": 8, "cut_off": 4, "edges_removed": 3}}
    assert score_row(row(obj(), compact=other))["ct_gt_periods"] == [7]


def test_error_banners_in_tool_outputs_are_not_parsed_as_the_payload():
    r = row(obj())
    r["tool_outputs"] = ["INPUT ERROR: κάτι πήγε στραβά"]
    scored = score_row(r)
    assert scored["ct_gt_periods"] == []


# --------------------------------------------------------------------------------
# 2.1: the numeric surface C1a is actually scored on
# --------------------------------------------------------------------------------
def test_numeric_text_is_the_composition_2_1_names():
    # policy_text ("interpretation" + "\n" + "limitations") plus the rendered
    # slot sentences. Stored on the row because 2.9 requires the auditor to see
    # byte-identically what the check saw, and asserted against the frozen
    # renderer rather than against a transcription of its output.
    o = obj()
    scored = score_row(row(o))
    claims = render_slot_claims(o)
    assert scored["slot_claims"] == claims
    assert scored["slot_claims_error"] == ""
    assert scored["numeric_text"] == (
        o["interpretation"] + "\n" + o["limitations"] + "\n" + "\n".join(claims))


def test_a_number_that_lives_only_in_a_slot_is_now_in_scope():
    # The whole reason 2.1 puts the slot sentences in the numeric surface: this
    # record's prose carries no numbers at all, so the run-time prose-only check
    # passes it, while the emitted 999 settlements at hour 1 (payload: 9) is
    # exactly the kind of untraceable number C1a exists to find. The provisional
    # value stays on the row untouched beside the 2.1 one, so the two can be
    # diffed and can never be confused.
    r = row(obj(critical=((1, 999), (3, 10), (4, 14))))
    r["numeric_ok"], r["suspicious_numbers"] = True, []

    score_rows([r])

    assert r["numeric_ok"] is True                 # run-time, prose only
    assert r["suspicious_numbers"] == []
    assert r["numeric_ok_full_surface"] is False   # 2.1, the published pair
    assert r["suspicious_numbers_full_surface"] == ["999"]
    assert "999" in r["numeric_text"]


def test_a_renderer_refusal_is_recorded_and_does_not_kill_the_pass():
    # The renderer refuses on a malformed entry rather than rendering "None"
    # into a Greek sentence, and such a record is schema_invalid but still has a
    # judgeable surface under 2.4. A scoring pass that propagated the refusal
    # would refuse to score the whole file over one bad row.
    broken = obj()
    broken["critical_transitions"] = [{"period": 1}]
    scored = score_row(row(broken))
    assert scored["contract_outcome"] == "schema_invalid"
    assert scored["slot_claims"] == []
    assert "settlements_without_route" in scored["slot_claims_error"]
    assert scored["numeric_text"].endswith("\n")


def test_the_free_arm_gets_no_2_1_numeric_verdict_of_its_own():
    # On a free arm 2.1 makes numeric_text == reply, which is exactly what the
    # run-time check already scored, so there is nothing for this pass to add
    # and None says "not measured here" rather than inventing a second number.
    scored = score_row(row(GREEK_PROSE, contract="free"))
    assert scored["numeric_ok_full_surface"] is None
    assert scored["suspicious_numbers_full_surface"] == []
    assert scored["numeric_text"] == ""
    assert scored["slot_claims"] == []


# --------------------------------------------------------------------------------
# GOLDEN: the 3.4.5 worked table over the four stored pilot records
# --------------------------------------------------------------------------------
WORKED_TABLE = {
    ("luna", "A03"): {
        "outcome": "validator_invalid",
        "gt": [1, 3, 4, 5, 11, 13, 20, 21], "n_gt": 8,
        "emitted": [1, 2, 3, 4, 5, 6, 7, 11, 13, 14, 18, 20, 21], "n_em": 13,
        "exact": False, "matched": 8, "extras": [2, 6, 7, 14, 18],
        "in_payload": [2, 6, 7, 14, 18], "not_in_payload": [],
    },
    ("luna", "B02"): {
        "outcome": "validator_invalid",
        "gt": [1, 3, 4], "n_gt": 3,
        "emitted": [1, 2, 3, 4], "n_em": 4,
        "exact": False, "matched": 3, "extras": [2],
        "in_payload": [2], "not_in_payload": [],
    },
    ("terra", "A03"): {
        "outcome": "valid",
        "gt": [1, 3, 4, 5, 11, 13, 20, 21], "n_gt": 8,
        "emitted": [1, 3, 4, 5, 11, 13, 20, 21], "n_em": 8,
        "exact": True, "matched": 8, "extras": [],
        "in_payload": [], "not_in_payload": [],
    },
    ("terra", "B02"): {
        "outcome": "valid",
        "gt": [1, 3, 4], "n_gt": 3,
        "emitted": [1, 3, 4], "n_em": 3,
        "exact": True, "matched": 3, "extras": [],
        "in_payload": [], "not_in_payload": [],
    },
}


def _pilot_rows(preset):
    """Read only. The pilot tree is a frozen audit record and no test may write
    into it, which is also what `test_in_place_refuses_the_frozen_pilot_tree`
    guards at the CLI level."""
    path = PILOT / preset / "agent_eval_raw.jsonl"
    if not path.exists():
        # FAIL, never skip. These four rows are committed evidence (5.8.9) and
        # they are the only oracle the 3.4.5 worked table has; a tree that has
        # gone missing means the audit record was deleted or moved, which is a
        # louder problem than a red test, and a skip would report it as green.
        pytest.fail(
            f"the frozen pilot audit record is missing: {path}. It is committed "
            "evidence (5.8.9) and the oracle for the 3.4.5 worked table, so its "
            "absence is a failure and never a skip. Restore it from git.")
    return score_rows([json.loads(line) for line
                       in path.read_text(encoding="utf-8").splitlines() if line.strip()])


@pytest.mark.parametrize("preset,case", sorted(WORKED_TABLE))
def test_golden_worked_table_cell_by_cell(preset, case):
    want = WORKED_TABLE[(preset, case)]
    rows = {r["case_id"]: r for r in _pilot_rows(preset)}
    got = rows[case]
    assert got["contract_outcome"] == want["outcome"]
    assert got["schema_conformant"] is True
    assert got["status_consistent"] is True
    assert got["ct_scored"] is True
    assert got["ct_gt_periods"] == want["gt"]
    assert got["ct_n_gt"] == want["n_gt"]
    assert got["ct_emitted_periods"] == want["emitted"]
    assert got["ct_n_emitted"] == want["n_em"]
    assert got["ct_exact_set"] is want["exact"]
    assert got["ct_matched"] == want["matched"]
    # Recall is 1.000 on all four records: every failure in the pilot is
    # over-inclusion and not one true transition was ever dropped.
    assert got["ct_matched"] == got["ct_n_gt"]
    assert got["ct_no_omission"] is True
    assert got["ct_missing_periods"] == []
    assert got["ct_extra_periods"] == want["extras"]
    assert got["ct_extra_in_payload"] == want["in_payload"]
    assert got["ct_extra_not_in_payload"] == want["not_in_payload"]
    assert got["ct_over_inclusive"] is bool(want["extras"])
    assert got["ct_order_ok"] is True
    assert got["ct_duplicate_periods"] == []
    assert got["ct_value_mismatches"] == []


def test_golden_per_preset_rollup():
    # The indicative roll-up of 3.4.5: luna 0/2 exact, 11/11 recall, 6 extras over
    # 17 emitted; terra 2/2 exact, 11/11 recall, 0 extras over 11 emitted.
    for preset, exact, extras, n_em in (("luna", 0, 6, 17), ("terra", 2, 0, 11)):
        rows = _pilot_rows(preset)
        assert len(rows) == 2
        assert sum(1 for r in rows if r["ct_exact_set"]) == exact
        assert sum(r["ct_matched"] for r in rows) == 11
        assert sum(r["ct_n_gt"] for r in rows) == 11
        assert sum(len(r["ct_extra_periods"]) for r in rows) == extras
        assert sum(r["ct_n_emitted"] for r in rows) == n_em
        assert sum(len(r["ct_extra_not_in_payload"]) for r in rows) == 0
        # 3.4.5 point 2: every emitted value, the over-included ones included,
        # equals the payload cut_off at that period. The error is membership.
        assert sum(len(r["ct_value_mismatches"]) for r in rows) == 0


def test_golden_gemini_rows_are_runtime_errors_and_never_scored():
    for r in _pilot_rows("gemini"):
        assert r["contract_outcome"] == "runtime_error"
        assert r["ct_scored"] is False
        assert r["ct_not_scored_reason"] == "runtime_error"


def test_summary_carries_the_mandatory_caveat_and_the_denominators():
    text = summarize(_pilot_rows("luna"), label="luna")
    assert "may not be pooled with the gate" in text
    assert "6/17 = 0.353" in text            # ct_extra_rate_emitted
    assert "11/11 = 1.000" in text           # event-level recall
    assert "0/2 = 0.000" in text             # ct_exact_set and contract validity
    assert per_row_lines(_pilot_rows("luna")).count("\n") == 2


# --------------------------------------------------------------------------------
# safety: the frozen tree, and the write that must not truncate
# --------------------------------------------------------------------------------
def test_is_frozen_audit_path_recognises_the_pilot_tree():
    assert is_frozen_audit_path(PILOT / "luna" / "agent_eval_raw.jsonl") is True
    assert is_frozen_audit_path(REPO / "tmp" / "agent_eval_raw.jsonl") is False


def test_in_place_refuses_the_frozen_pilot_tree(tmp_path, capsys):
    frozen = tmp_path / "pilot_records" / "run"
    frozen.mkdir(parents=True)
    raw = frozen / "agent_eval_raw.jsonl"
    before = json.dumps(row(obj()), ensure_ascii=False) + "\n"
    raw.write_text(before, encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(raw), "--in-place"])
    assert raw.read_text(encoding="utf-8") == before


def test_out_also_refuses_to_write_into_a_frozen_tree(tmp_path):
    src = tmp_path / "agent_eval_raw.jsonl"
    src.write_text(json.dumps(row(obj()), ensure_ascii=False) + "\n", encoding="utf-8")
    dest = tmp_path / "pilot_records" / "copy.jsonl"
    dest.parent.mkdir()
    with pytest.raises(SystemExit):
        main([str(src), "--out", str(dest)])
    assert not dest.exists()


def test_in_place_writes_the_fields_and_leaves_no_temp_file(tmp_path):
    raw = tmp_path / "agent_eval_raw.jsonl"
    raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in
                             (row(obj()), row(obj(critical=critical_from([1, 2, 3, 4])),
                                              case_id="T02"))) + "\n",
                   encoding="utf-8")
    assert main([str(raw), "--in-place"]) == 0
    scored = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    assert [r["contract_outcome"] for r in scored] == ["valid", "validator_invalid"]
    assert scored[1]["ct_extra_periods"] == [2]
    # Greek survives the round trip, and the atomic write leaves nothing behind.
    assert scored[0]["reply"].count("\\u") == 0
    assert [p.name for p in tmp_path.iterdir()] == ["agent_eval_raw.jsonl"]


def test_default_run_writes_nothing(tmp_path):
    raw = tmp_path / "agent_eval_raw.jsonl"
    before = json.dumps(row(obj()), ensure_ascii=False) + "\n"
    raw.write_text(before, encoding="utf-8")
    assert main([str(raw)]) == 0
    assert raw.read_text(encoding="utf-8") == before


def test_round_ceiling_literal_still_matches_agent_py():
    # The sentence is copied, not imported (importing agent.py pulls litellm into
    # an offline pass), so the copy must be checked against the source.
    src = (REPO / "scripts" / "agent" / "agent.py").read_text(encoding="utf-8")
    assert ROUND_CEILING_REPLY in src


# --------------------------------------------------------------------------------
# 3.4.2 / 3.4.6, the hard constraint: no ct_* field may reach the gate
# --------------------------------------------------------------------------------
AXIS3 = REPO / "scripts" / "validation" / "axis3_full_system.py"


def _top_level_function_source(text, name):
    """One top-level function's source, sliced out of a file read as TEXT.

    Read as text rather than through `inspect`, because importing
    axis3_full_system to inspect it would execute its imports for a test whose
    whole subject is what the file SAYS, and because a text slice is what a
    reviewer can reproduce by eye against the same line numbers.
    """
    start = text.index("\ndef " + name + "(")
    end = text.find("\ndef ", start + 1)
    return text[start:end if end != -1 else len(text)]


def test_no_ct_field_can_reach_the_gate():
    """The hard constraint of 3.4.6, asserted against the gate's own source.

    This is the most important invariant of the whole design and the easiest one
    to lose by accident: the three critical-transition metrics are diagnostic
    columns, and 3.4.2 states that none of them enters `deterministic_pass`, the
    `components` tuple, `full_system` or any audit decision column. One helpful
    `and row.get("ct_exact_set")` would silently move a diagnostic into every
    published pass rate in the thesis, and nothing else in the suite would
    notice, because the gate would keep returning perfectly plausible booleans.
    A grep is the right instrument here: the point is that the substring is not
    in the code at all.
    """
    text = AXIS3.read_text(encoding="utf-8")

    gate = _top_level_function_source(text, "deterministic_pass")
    assert "ct_" not in gate, (
        "a ct_* field has entered deterministic_pass, which 3.4.2 forbids:\n" + gate)

    assignments = [line.strip() for line in text.splitlines()
                   if line.strip().startswith("components =")]
    assert len(assignments) == 1, f"expected one components assignment, got {assignments}"
    assert "ct_" not in assignments[0], (
        "a ct_* field has entered the components tuple, which 3.4.2 forbids: "
        + assignments[0])
    # Spelled out in full rather than checked loosely: 3.4.2 says this tuple
    # "stays (det, stage_c, coverage) exactly as it is today", so the test
    # states the whole tuple and a fourth component of any kind fails here.
    assert assignments[0] == "components = (det, stage_c, coverage)"

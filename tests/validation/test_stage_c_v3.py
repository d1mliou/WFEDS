"""Tests for the Stage-C v3 instrument.

These do not call a model or a judge. They check the properties the methodology
depends on: the case set is exactly the inclusion rule's output, the blinded
records leak nothing, the artefacts hash deterministically, the pass rule is the
conjunction it claims to be, an unnarrated record becomes NOT_REACHED rather than
a factual failure, and the retired machinery cannot reappear in a new output.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

V = Path(__file__).resolve().parents[2] / "scripts" / "validation"
EXPECTED_33 = ["A01", "A02", "A03", "A04", "B01", "B02", "B03", "B04",
               "C01", "C02", "C03", "C04", "D04", "D05", "D06", "D07",
               "E01", "E02", "E03", "E04", "I01", "I02", "I03", "I04", "I05",
               "J01", "J02", "J03", "J04", "K01", "L01", "L02", "M01"]


def load(name):
    path = V / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def suite():
    return read_jsonl(V / "agent_eval_cases.jsonl")


@pytest.fixture(scope="module")
def gold():
    return json.loads((V / "stage_c_v3_gold.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- the case set
def test_inclusion_rule_yields_exactly_33(suite):
    selected = [c["id"] for c in suite
                if c["gold"].get("must_call_tool") and c["category"] != "invalid_input"]
    assert len(selected) == 33
    assert selected == EXPECTED_33


def test_excluded_cases_are_excluded_for_the_stated_reason(suite):
    by_id = {c["id"]: c for c in suite}
    for cid in ("H01", "H02", "H03", "H04", "H05"):
        assert by_id[cid]["category"] == "invalid_input"
        assert by_id[cid]["gold"]["must_call_tool"] is True
    for cid in ("D01", "D02", "D03", "F01", "F02", "F03", "F04",
                "G01", "G02", "G03", "G04", "G05", "G06", "G07", "K02"):
        assert by_id[cid]["gold"].get("must_call_tool") is False


def test_frozen_33_file_matches_the_rule_byte_for_byte(suite):
    lines = {json.loads(l)["id"]: l.strip() for l in
             (V / "agent_eval_cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    frozen = (V / "agent_eval_cases_stagec33.jsonl").read_text(encoding="utf-8").splitlines()
    frozen = [l.strip() for l in frozen if l.strip()]
    assert len(frozen) == 33
    assert [json.loads(l)["id"] for l in frozen] == EXPECTED_33
    for line in frozen:
        assert line == lines[json.loads(line)["id"]]


def test_gold_covers_every_case_with_the_required_fields(gold):
    assert gold["n_cases"] == 33
    assert sorted(gold["cases"]) == sorted(EXPECTED_33)
    required = {"case_id", "category", "user_text", "fixture", "purpose",
                "expected_action", "expected_tool_arguments",
                "arguments_that_must_be_absent", "policies", "critical_events",
                "limitation_elements_required", "advice",
                "permitted_basis_for_advice", "forbidden_inventions"}
    for cid, g in gold["cases"].items():
        assert required <= set(g), cid
        assert g["critical_events"], cid
        for e in g["critical_events"]:
            assert {"event_id", "description", "source_path",
                    "narration_must_convey", "range_or_paraphrase_allowed"} <= set(e)
            assert e["source_path"].strip()


def test_event_ids_are_unique_within_a_case(gold):
    for cid, g in gold["cases"].items():
        ids = [e["event_id"] for e in g["critical_events"]]
        assert len(ids) == len(set(ids)), cid


def test_limitation_gate_is_exactly_the_three_locked_elements(gold):
    """Closed by the author 2026-09-22. Uncalibrated fuels is measured but never gates."""
    for cid, g in gold["cases"].items():
        assert [e["element_id"] for e in g["limitation_elements_required"]] == [
            "free_burning_scenario",
            "suppression_not_simulated",
            "traffic_local_conditions_and_commander",
        ], cid
        assert [e["element_id"] for e in g["limitation_elements_reported_only"]] == [
            "uncalibrated_fuels"], cid


def test_uncalibrated_fuels_is_not_part_of_the_pass_rule():
    val = load("stage_c_v3_validate_judge_output")
    passing = _d(uncalibrated_fuels_mentioned=False)
    assert val.expected_pass(passing) is True
    assert val.expected_pass(_d(uncalibrated_fuels_mentioned=True)) is True
    gold_text = (V / "stage_c_v3_gold.json").read_text(encoding="utf-8")
    assert "NOT a gate" in gold_text
    rubric = (V / "stage_c_v3_rubric.md").read_text(encoding="utf-8")
    assert "does not affect" in rubric.lower()


def test_stage_c_requires_all_six_fields(gold):
    """Closed by the author 2026-09-22: disclosures are their own gate."""
    val = load("stage_c_v3_validate_judge_output")
    assert len(val.QUESTIONS) == 6
    assert "required_disclosures_complete" in val.QUESTIONS
    assert val.failed_questions(_d(required_disclosures_complete=False)) == {
        "required_disclosures_complete"}
    assert "required_disclosures_complete is true" in gold["stage_c_pass_rule"]
    schema = json.loads((V / "stage_c_v3_judge_schema.json").read_text(encoding="utf-8"))
    items = schema["properties"]["decisions"]["items"]
    assert "required_disclosures_complete" in items["required"]
    assert "required_disclosures_complete" in (
        items["properties"]["failures"]["items"]["properties"]["question"]["enum"])


def test_the_two_event_lists_are_disjoint_and_cover_the_gold(gold):
    for cid, g in gold["cases"].items():
        events = g["critical_events"]
        gated = [e for e in events if e["kind"] in ("transition", "final_outcome")]
        disc = [e for e in events if e["kind"] == "policy_disclosure"]
        assert len(gated) + len(disc) == len(events), cid
        assert not ({e["event_id"] for e in gated} & {e["event_id"] for e in disc}), cid
        assert gated, cid
    assert gold["question_scope"]["critical_events_complete"].endswith("only")
    assert "policy_disclosure" in gold["question_scope"]["required_disclosures_complete"]


def test_record_shows_the_two_lists_separately(bundle):
    _, out, builder = bundle
    for p in (out / "records").glob("*.txt"):
        text = p.read_text(encoding="utf-8")
        assert "required critical events, question 3 is answered on these" in text
        assert "required disclosures, question 4 is answered on these" in text
        assert text.index("question 3 is answered") < text.index("question 4 is answered")


def test_initial_state_is_not_a_required_event_anywhere(gold):
    """Closed by the author 2026-09-22: hour 0 is a starting condition, not a change,
    and its value repeats across most fixtures."""
    for cid, g in gold["cases"].items():
        ids = [e["event_id"] for e in g["critical_events"]]
        assert "E_INITIAL" not in ids, cid
        for e in g["critical_events"]:
            assert "period=0" not in e["source_path"] or e["event_id"] == "E_NOSPREAD", cid
    assert "critical_event_rule" in gold
    assert "not a required event" in gold["critical_event_rule"]


def test_every_required_event_is_a_change_or_an_end_state_or_a_named_policy(gold):
    allowed = {"transition", "final_outcome", "policy_disclosure"}
    for cid, g in gold["cases"].items():
        for e in g["critical_events"]:
            assert e["kind"] in allowed, (cid, e["event_id"], e["kind"])
        kinds = {e["kind"] for e in g["critical_events"]}
        assert kinds & {"transition", "final_outcome"}, cid


def test_initial_state_is_measured_but_never_gates():
    val = load("stage_c_v3_validate_judge_output")
    assert val.expected_pass(_d(initial_state_conveyed=False)) is True
    assert val.expected_pass(_d(initial_state_conveyed=True)) is True
    schema = json.loads((V / "stage_c_v3_judge_schema.json").read_text(encoding="utf-8"))
    props = schema["properties"]["decisions"]["items"]["properties"]
    assert "initial_state_conveyed" in props
    assert "never part of stage_c_pass" in props["initial_state_conveyed"]["description"]
    rubric = (V / "stage_c_v3_rubric.md").read_text(encoding="utf-8")
    assert "not a required event" in rubric


def test_structured_answer_carries_its_factual_slots(tmp_path):
    """Closed by the author 2026-09-22: coverage is measured over everything the
    answer delivers, so the mandatory slots are part of the judged text."""
    builder = load("stage_c_v3_build_judge_inputs")
    obj = {"status": "ok",
           "initial_state": [{"period": 0, "settlements_without_route": 6}],
           "critical_transitions": [{"period": 4, "settlements_without_route": 14}],
           "final_state": [{"period": 8, "settlements_without_route": 15,
                            "road_segments_removed": 473}],
           "interpretation": "Η κατάσταση επιδεινώνεται.",
           "limitations": "Γνωμοδοτικά."}
    surface = builder.narrative_surface(json.dumps(obj, ensure_ascii=False), "structured")
    assert "ώρα 4" in surface and "14" in surface
    assert "15" in surface and "473" in surface
    assert "Η κατάσταση επιδεινώνεται." in surface
    assert "Γνωμοδοτικά." in surface
    # nothing in the rendered text announces which arm produced it
    for token in ("structured", "slot", "interpretation", "critical_transitions",
                  "initial_state", "final_state", "schema"):
        assert token.lower() not in surface.lower()
    free = builder.narrative_surface("Απλό κείμενο απάντησης.", "free")
    assert free == "Απλό κείμενο απάντησης."


def test_nospread_cases_guard_against_invented_risk(gold):
    for cid in ("L01", "L02"):
        ev = gold["cases"][cid]["critical_events"]
        text = " ".join(e["narration_must_convey"] for e in ev).lower()
        assert "no settlement is at risk" in text
        assert "fabrication of risk" in text


def test_cases_that_stage_b_may_block_are_flagged(gold):
    assert gold["cases"]["E04"]["stage_b_may_block_stage_c"] is True
    assert gold["cases"]["K01"]["stage_b_may_block_stage_c"] is True
    assert gold["cases"]["A01"]["stage_b_may_block_stage_c"] is False


# ------------------------------------------------------------- the pass rule
def _d(**changed):
    base = {"record_id": "R0001", "judgeable": True,
            "has_contradicted_information": False,
            "has_unsupported_information": False,
            "critical_events_complete": True,
            "required_disclosures_complete": True,
            "advice_grounded_and_caveated": "true",
            "limitations_complete": True,
            "stage_c_pass": True, "failures": [],
            "uncalibrated_fuels_mentioned": False,
            "initial_state_conveyed": False, "note": ""}
    base.update(changed)
    return base


def test_pass_rule_is_the_conjunction():
    val = load("stage_c_v3_validate_judge_output")
    assert val.expected_pass(_d()) is True
    assert val.expected_pass(_d(advice_grounded_and_caveated="NOT_APPLICABLE")) is True
    assert val.expected_pass(_d(has_contradicted_information=True)) is False
    assert val.expected_pass(_d(has_unsupported_information=True)) is False
    assert val.expected_pass(_d(critical_events_complete=False)) is False
    assert val.expected_pass(_d(required_disclosures_complete=False)) is False
    assert val.expected_pass(_d(advice_grounded_and_caveated="false")) is False
    assert val.expected_pass(_d(limitations_complete=False)) is False


def test_failed_questions_match_the_answers():
    val = load("stage_c_v3_validate_judge_output")
    assert val.failed_questions(_d()) == set()
    assert val.failed_questions(_d(has_contradicted_information=True)) == {
        "has_contradicted_information"}
    assert val.failed_questions(
        _d(critical_events_complete=False, limitations_complete=False)) == {
        "critical_events_complete", "limitations_complete"}
    assert val.failed_questions(
        _d(advice_grounded_and_caveated="NOT_APPLICABLE")) == set()


def test_validator_rejects_a_pass_that_contradicts_the_conjunction(tmp_path):
    val = load("stage_c_v3_validate_judge_output")
    bundle = tmp_path / "b"
    (bundle).mkdir()
    (bundle / "blind_manifest.json").write_text(json.dumps(
        {"records": [{"record_id": "R0001"}]}), encoding="utf-8")
    bad = _d(has_contradicted_information=True, stage_c_pass=True,
             failures=[{"question": "has_contradicted_information", "exact_quote": "x",
                        "payload_evidence": "y", "reason": "z"}])
    task = tmp_path / "t.json"
    task.write_text(json.dumps({"decisions": [bad]}), encoding="utf-8")
    assert val.main(["--bundle", str(bundle), "--task", str(task)]) == 1


def test_validator_rejects_a_failure_without_evidence(tmp_path):
    val = load("stage_c_v3_validate_judge_output")
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "blind_manifest.json").write_text(json.dumps(
        {"records": [{"record_id": "R0001"}]}), encoding="utf-8")
    bad = _d(has_unsupported_information=True, stage_c_pass=False,
             failures=[{"question": "has_unsupported_information", "exact_quote": "",
                        "payload_evidence": "", "reason": "z"}])
    task = tmp_path / "t.json"
    task.write_text(json.dumps({"decisions": [bad]}), encoding="utf-8")
    assert val.main(["--bundle", str(bundle), "--task", str(task)]) == 1


def test_validator_requires_nulls_when_unjudgeable(tmp_path):
    val = load("stage_c_v3_validate_judge_output")
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "blind_manifest.json").write_text(json.dumps(
        {"records": [{"record_id": "R0001"}]}), encoding="utf-8")
    bad = _d(judgeable=False)          # keeps non-null answers: not allowed
    task = tmp_path / "t.json"
    task.write_text(json.dumps({"decisions": [bad]}), encoding="utf-8")
    assert val.main(["--bundle", str(bundle), "--task", str(task)]) == 1

    good = _d(judgeable=False, stage_c_pass=None,
              **{q: None for q in val.QUESTIONS})
    task.write_text(json.dumps({"decisions": [good]}), encoding="utf-8")
    assert val.main(["--bundle", str(bundle), "--task", str(task)]) == 0


def test_validator_rejects_retired_machinery_in_the_output(tmp_path):
    val = load("stage_c_v3_validate_judge_output")
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "blind_manifest.json").write_text(json.dumps(
        {"records": [{"record_id": "R0001"}]}), encoding="utf-8")
    d = _d(note="also computed ct_exact_set for this record")
    task = tmp_path / "t.json"
    task.write_text(json.dumps({"decisions": [d]}), encoding="utf-8")
    assert val.main(["--bundle", str(bundle), "--task", str(task)]) == 1


# --------------------------------------------------------- the blinded bundle
def _fake_run(root, arm, model, folder, cases, payload, surface_by_case=None,
              broken=()):
    d = root / folder / "validation" / "3_agent"
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for cid in cases:
        for rep in (1, 2):
            reply = (surface_by_case or {}).get(cid, "Η κατάσταση εξελίσσεται. " * 12)
            if arm == "structured":
                reply = json.dumps({"status": "ok", "initial_state": [],
                                    "critical_transitions": [], "final_state": [],
                                    "interpretation": reply,
                                    "limitations": "Γνωμοδοτικά."}, ensure_ascii=False)
            if cid in broken:
                reply = ""
            lines.append(json.dumps({
                "case_id": cid, "rep": rep, "category": "fully_specified",
                "user_text": "Τρέξε το σενάριο.", "reply": reply,
                "tool_outputs": [json.dumps(payload, ensure_ascii=False)],
                "stage_a_ok": True, "stage_b_ok": True, "n_tool_calls": 1,
            }, ensure_ascii=False))
    (d / "agent_eval_raw.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    builder = load("stage_c_v3_build_judge_inputs")
    frozen = json.loads((V / "stage_c_v3_frozen_payloads.json").read_text(encoding="utf-8"))
    cases = ["A01", "I04"]
    payload = dict(frozen["payloads"]["A01"])
    payload["inputs"] = dict(payload["inputs"], start_time_utc="2026-08-12T11:30")
    exports = tmp_path / "exports"
    runs = []
    for arm in ("free", "structured"):
        for model in ("luna", "terra"):
            folder = f"run_{arm}_{model}"
            _fake_run(exports, arm, model, folder, cases, payload, broken=("I04",))
            runs.append({"arm": arm, "model": model, "folder": folder})
    runs_file = tmp_path / "runs.json"
    runs_file.write_text(json.dumps(runs), encoding="utf-8")
    out = tmp_path / "bundle"
    rc = builder.main(["--runs", str(runs_file), "--exports", str(exports),
                       "--out", str(out)])
    return rc, out, builder


def test_bundle_builds_cleanly_and_blinds_everything(bundle):
    rc, out, _ = bundle
    assert rc == 0
    manifest = json.loads((out / "blind_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_judgeable"] == 8          # A01 only, 2 reps x 2 models x 2 arms
    leak = re.compile(r"\b(luna|terra|gemini)\b|\bstructured\b|\bfree arm\b|\bA01\b|\bI04\b",
                      re.I)
    for p in (out / "records").glob("*.txt"):
        assert not leak.search(p.read_text(encoding="utf-8")), p.name


def test_records_share_one_template_across_arms(bundle):
    _, out, builder = bundle
    key = json.loads((out / "KEY_DO_NOT_SHIP.json").read_text(encoding="utf-8"))["mapping"]
    shapes = {}
    for rid, meta in key.items():
        t = (out / "records" / f"{rid}.txt").read_text(encoding="utf-8")
        shape = tuple(ln.strip() for ln in t.split("\n") if ln.strip() in builder.SKELETON)
        shapes.setdefault(meta["arm"], set()).add(shape)
    assert len(shapes["free"]) == 1
    assert shapes["free"] == shapes["structured"]


def test_missing_surface_becomes_not_reached_not_a_factual_failure(bundle):
    _, out, _ = bundle
    key = json.loads((out / "KEY_DO_NOT_SHIP.json").read_text(encoding="utf-8"))
    nr = key["not_reached"]
    assert {x["case_id"] for x in nr} == {"I04"}
    assert all(x["reason"] == "no_surface" for x in nr)
    assert all(m["case_id"] != "I04" for m in key["mapping"].values())


def test_payload_that_differs_from_gold_is_not_reached(tmp_path):
    builder = load("stage_c_v3_build_judge_inputs")
    frozen = json.loads((V / "stage_c_v3_frozen_payloads.json").read_text(encoding="utf-8"))
    payload = json.loads(json.dumps(frozen["payloads"]["A01"]))
    payload["change_hours"][0]["cut_off"] = 99          # a different series entirely
    exports = tmp_path / "exports"
    _fake_run(exports, "free", "luna", "r", ["A01"], payload)
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps([{"arm": "free", "model": "luna", "folder": "r"}]),
                    encoding="utf-8")
    out = tmp_path / "b"
    builder.main(["--runs", str(runs), "--exports", str(exports), "--out", str(out)])
    key = json.loads((out / "KEY_DO_NOT_SHIP.json").read_text(encoding="utf-8"))
    assert key["mapping"] == {}
    assert all(x["reason"] == "payload_differs_from_gold" for x in key["not_reached"])


def test_bundle_is_deterministic(tmp_path, bundle):
    _, out, builder = bundle
    first = json.loads((out / "blind_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((V / "stage_c_v3_frozen_payloads.json").read_text(encoding="utf-8"))
    payload = dict(frozen["payloads"]["A01"])
    payload["inputs"] = dict(payload["inputs"], start_time_utc="2026-08-12T11:30")
    exports = tmp_path / "exports2"
    runs = []
    for arm in ("free", "structured"):
        for model in ("luna", "terra"):
            folder = f"run_{arm}_{model}"
            _fake_run(exports, arm, model, folder, ["A01", "I04"], payload, broken=("I04",))
            runs.append({"arm": arm, "model": model, "folder": folder})
    rf = tmp_path / "runs2.json"
    rf.write_text(json.dumps(runs), encoding="utf-8")
    out2 = tmp_path / "bundle2"
    builder.main(["--runs", str(rf), "--exports", str(exports), "--out", str(out2)])
    second = json.loads((out2 / "blind_manifest.json").read_text(encoding="utf-8"))
    assert [r["sha256"] for r in first["records"]] == [r["sha256"] for r in second["records"]]


# ------------------------------------------------------------ frozen artefacts
def test_frozen_artefacts_hash_to_recorded_values():
    expected = {
        "agent_eval_cases_stagec33.jsonl": "e4fe950afec39f61",
        "stage_c_v3_frozen_payloads.json": "4550f5d7a065ebb8",
        "stage_c_v3_gold.json": "158629c36d8cc226",
    }
    for name, want in expected.items():
        got = hashlib.sha256((V / name).read_bytes()).hexdigest()[:16]
        assert got == want, f"{name} changed: {got} != {want}"


def test_retired_machinery_is_absent_from_the_v3_instrument():
    banned = ("exact_set", "SSP composite", "R3 ", "atomic claim", "Cohen")
    for name in ("stage_c_v3_gold.json", "stage_c_v3_rubric.md",
                 "stage_c_v3_judge_prompt.txt", "stage_c_v3_judge_schema.json"):
        text = (V / name).read_text(encoding="utf-8")
        for token in banned:
            assert token.lower() not in text.lower(), f"{name} mentions {token!r}"

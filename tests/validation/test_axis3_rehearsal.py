"""Tests for the end-to-end rehearsal driver and for the guards it installed.

The rehearsal's own job is to prove the chain runs. These tests prove the three
things the rehearsal cannot prove about itself: that its stub can never be
mistaken for a judged label, that its checksum really notices a changed byte,
and that the small edits it forced into the scoring modules behave.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validation import axis3_rehearsal as R
from scripts.validation.axis3_full_system import (
    narrative_surface_of,
    parse_label_spec,
    resolve_label_files,
)

_REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# The stub sentinel and the refusal
# ---------------------------------------------------------------------------
def test_stub_records_are_counted_only_when_the_flag_is_literally_true():
    assert R.stub_record_count([{R.REHEARSAL_STUB_KEY: True}]) == 1
    assert R.stub_record_count([{R.REHEARSAL_STUB_KEY: "true"}]) == 0
    assert R.stub_record_count([{R.REHEARSAL_STUB_KEY: False}, {"uid": "A#1"}]) == 0
    assert R.stub_record_count([]) == 0
    assert R.stub_record_count({"not": "a list"}) == 0


def test_a_real_label_file_passes_the_guard_without_the_flag():
    R.refuse_stub_labels("labels.json", [{"uid": "A01#1", "claims": []}], False)


def test_a_stub_label_file_is_refused_without_the_flag():
    with pytest.raises(SystemExit) as exc:
        R.refuse_stub_labels("labels.json", [{R.REHEARSAL_STUB_KEY: True}], False)
    assert R.REHEARSAL_STUB_KEY in str(exc.value)
    assert "--rehearsal" in str(exc.value)


def test_a_stub_label_file_is_admitted_with_the_flag():
    R.refuse_stub_labels("labels.json", [{R.REHEARSAL_STUB_KEY: True}], True)


def test_the_refusal_reads_the_data_not_the_filename():
    """A stub renamed to look real is still refused, and a real file named like
    a stub is still admitted. The sentinel is in the records on purpose."""
    with pytest.raises(SystemExit):
        R.refuse_stub_labels("stage_c_labels.json",
                             [{R.REHEARSAL_STUB_KEY: True}], False)
    R.refuse_stub_labels(R.STUB_LABEL_FILENAME, [{"uid": "A01#1"}], False)


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------
def test_tree_checksum_is_stable_and_per_file(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("two", encoding="utf-8")
    first = R.tree_checksum(tmp_path)
    assert sorted(first) == ["a.txt", "sub/b.txt"]
    assert first == R.tree_checksum(tmp_path)


def test_tree_checksum_of_a_missing_directory_is_empty(tmp_path):
    assert R.tree_checksum(tmp_path / "nope") == {}


@pytest.mark.parametrize("what", ["modify", "create", "delete"])
def test_a_single_changed_byte_is_caught_and_the_file_is_named(tmp_path, what):
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    before = {"t": R.tree_checksum(tmp_path)}
    if what == "modify":
        (tmp_path / "a.txt").write_text("onE", encoding="utf-8")
        expected = "t: MODIFIED a.txt"
    elif what == "create":
        (tmp_path / "b.txt").write_text("x", encoding="utf-8")
        expected = "t: CREATED b.txt"
    else:
        (tmp_path / "a.txt").unlink()
        expected = "t: DELETED a.txt"
    problems = R.diff_checksums(before, {"t": R.tree_checksum(tmp_path)})
    assert problems == [expected]


def test_an_untouched_tree_reports_nothing(tmp_path):
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    before = {"t": R.tree_checksum(tmp_path)}
    assert R.diff_checksums(before, {"t": R.tree_checksum(tmp_path)}) == []


# ---------------------------------------------------------------------------
# The judge stub
# ---------------------------------------------------------------------------
def _batch_dir(tmp_path, claims=("alpha", "beta")):
    d = tmp_path / "stage_c_full"
    d.mkdir(exist_ok=True)
    (d / "batch_01.json").write_text(json.dumps(
        [{"uid": "A01#1", "candidate_claims": list(claims)},
         {"uid": "A01#2", "candidate_claims": list(claims)}]), encoding="utf-8")
    return d


def test_every_stub_record_carries_the_sentinel_and_says_so_in_its_notes(tmp_path):
    records = R.stub_labels_for_batches(_batch_dir(tmp_path))
    assert len(records) == 2
    for record in records:
        assert record[R.REHEARSAL_STUB_KEY] is True
        assert "REHEARSAL STUB" in record["notes"]
        assert all("REHEARSAL STUB" in c["justification"] for c in record["claims"])


def test_only_the_labels_are_invented_the_claim_texts_are_the_real_candidates(tmp_path):
    records = R.stub_labels_for_batches(_batch_dir(tmp_path, ("alpha", "beta")))
    assert [c["claim"] for c in records[0]["claims"]] == ["alpha", "beta"]
    vocabulary = {name for name, _ in R._STUB_LABELS}
    assert all(c["label"] in vocabulary for r in records for c in r["claims"])


def test_the_stub_is_reproducible_from_the_seed(tmp_path):
    first = R.stub_labels_for_batches(_batch_dir(tmp_path), seed=7)
    second = R.stub_labels_for_batches(_batch_dir(tmp_path), seed=7)
    assert first == second
    assert R.stub_labels_for_batches(_batch_dir(tmp_path), seed=8) != first


def test_one_records_labels_do_not_move_when_another_record_is_added(tmp_path):
    """Seeded per uid, so a re-run on a different population is still diffable."""
    small = R.stub_labels_for_batches(_batch_dir(tmp_path))
    d = tmp_path / "stage_c_full"
    (d / "batch_02.json").write_text(json.dumps(
        [{"uid": "Z99#9", "candidate_claims": ["gamma"]}]), encoding="utf-8")
    large = R.stub_labels_for_batches(d)
    assert len(large) == 3
    by_uid = {r["uid"]: r for r in large}
    assert by_uid["A01#1"] == small[0]


def test_a_record_with_no_candidate_claim_still_produces_one_claim(tmp_path):
    d = tmp_path / "stage_c_full"
    d.mkdir()
    (d / "batch_01.json").write_text(json.dumps([{"uid": "A01#1"}]), encoding="utf-8")
    records = R.stub_labels_for_batches(d)
    assert len(records[0]["claims"]) == 1


# ---------------------------------------------------------------------------
# The stub coverage sheet
# ---------------------------------------------------------------------------
def _sheet(tmp_path, n=4):
    path = tmp_path / "sheet.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["decision_id", "judged_surface", "human_critical_coverage"])
        writer.writeheader()
        for i in range(1, n + 1):
            writer.writerow({"decision_id": f"COV-{i:04d}",
                             "judged_surface": "text", "human_critical_coverage": ""})
    return path


def test_the_filled_sheet_is_stamped_in_its_first_column(tmp_path):
    out = tmp_path / "filled.csv"
    n, _ = R.stub_coverage_answers(_sheet(tmp_path), out)
    assert n == 4
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][0] == "REHEARSAL_STUB_DECISION_NOT_A_HUMAN_VERDICT"
    assert all(r[0] == "REHEARSAL STUB" for r in rows[1:])


def test_every_decision_is_filled_with_a_legal_value_and_is_reproducible(tmp_path):
    out1, out2 = tmp_path / "a.csv", tmp_path / "b.csv"
    R.stub_coverage_answers(_sheet(tmp_path), out1, seed=3)
    R.stub_coverage_answers(_sheet(tmp_path), out2, seed=3)
    assert out1.read_bytes() == out2.read_bytes()
    with out1.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["human_critical_coverage"] for r in rows} <= {"PASS", "FAIL"}
    assert all(r["human_critical_coverage"] for r in rows)


def test_an_empty_sheet_produces_nothing_rather_than_a_header_only_file(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("decision_id\n", encoding="utf-8-sig")
    n, _ = R.stub_coverage_answers(empty, tmp_path / "out.csv")
    assert n == 0
    assert not (tmp_path / "out.csv").exists()


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------
def test_every_artefact_kind_gets_the_banner(tmp_path):
    (tmp_path / "r.md").write_text("# title\n", encoding="utf-8")
    (tmp_path / "r.txt").write_text("body\n", encoding="utf-8")
    (tmp_path / "m.json").write_text('{"a": 1}', encoding="utf-8")
    with (tmp_path / "t.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows([["uid", "verdict"], ["A01#1", "PASS"]])

    R.stamp_workspace(tmp_path)

    assert (tmp_path / "r.md").read_text(encoding="utf-8").splitlines()[0].startswith("<!--")
    assert R.BANNER in (tmp_path / "r.txt").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert list(data)[0] == "REHEARSAL" and data["a"] == 1
    with (tmp_path / "t.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["REHEARSAL", "uid", "verdict"]
    assert rows[1][0].startswith("REHEARSAL")


def test_stamping_twice_does_not_stamp_twice(tmp_path):
    (tmp_path / "r.md").write_text("# title\n", encoding="utf-8")
    (tmp_path / "m.json").write_text('{"a": 1}', encoding="utf-8")
    with (tmp_path / "t.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        csv.writer(fh).writerows([["uid"], ["A01#1"]])
    R.stamp_workspace(tmp_path)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    R.stamp_workspace(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == first


def test_a_copied_raw_jsonl_is_never_rewritten(tmp_path):
    """The copies of the stored runs live in the workspace too, and a stamped
    raw.jsonl would no longer be a faithful copy of the evidence."""
    raw = tmp_path / "agent_eval_raw.jsonl"
    raw.write_text('{"case_id": "A01"}\n', encoding="utf-8")
    R.stamp_workspace(tmp_path)
    assert raw.read_text(encoding="utf-8") == '{"case_id": "A01"}\n'


def test_a_list_json_is_left_alone_because_it_is_stamped_per_record(tmp_path):
    labels = tmp_path / R.STUB_LABEL_FILENAME
    labels.write_text(json.dumps([{R.REHEARSAL_STUB_KEY: True}]), encoding="utf-8")
    R.stamp_workspace(tmp_path)
    assert json.loads(labels.read_text(encoding="utf-8"))[0][R.REHEARSAL_STUB_KEY] is True


# ---------------------------------------------------------------------------
# No hard-coded machine path
# ---------------------------------------------------------------------------
def test_the_module_hard_codes_no_absolute_path():
    text = Path(R.__file__).read_text(encoding="utf-8")
    assert "C:/Users" not in text and "C:\\Users" not in text
    assert "OneDrive" not in text


def test_the_source_trees_are_all_resolved_under_the_exports_root_given(tmp_path):
    trees = R.source_trees(tmp_path)
    for name, path in trees.items():
        if name == "pilot_records":
            continue    # the pilot record lives in the repo, not in Exports
        assert tmp_path in path.parents or path.parent == tmp_path


# ---------------------------------------------------------------------------
# The guards the rehearsal forced into the scoring modules
# ---------------------------------------------------------------------------
def test_axis3_full_system_refuses_a_stub_label_file(tmp_path):
    from scripts.validation.axis3_full_system import label_map
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"uid": "A01#1", R.REHEARSAL_STUB_KEY: True,
                                 "claims": []}]), encoding="utf-8")
    with pytest.raises(SystemExit):
        label_map(path)
    assert set(label_map(path, allow_rehearsal=True)) == {"A01#1"}


def test_the_aggregator_refuses_a_stub_label_file_without_the_flag(tmp_path):
    stage = tmp_path / "stage_c_full"
    stage.mkdir()
    (stage / "batch_01.json").write_text(json.dumps(
        [{"uid": "A01#1", "category": "fully_specified"}]), encoding="utf-8")
    labels = stage / R.STUB_LABEL_FILENAME
    labels.write_text(json.dumps(
        [{"uid": "A01#1", R.REHEARSAL_STUB_KEY: True,
          "claims": [{"label": "SUPPORTED"}], "notes": ""}]), encoding="utf-8")

    cmd = [sys.executable, "scripts/validation/stage_c_aggregate.py",
           str(labels), str(stage), "--out-dir", str(tmp_path / "out")]
    refused = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True)
    assert refused.returncode != 0
    assert R.REHEARSAL_STUB_KEY in (refused.stdout + refused.stderr)

    allowed = subprocess.run(cmd + ["--rehearsal"], cwd=str(_REPO),
                             capture_output=True, text=True)
    assert allowed.returncode == 0


# ---------------------------------------------------------------------------
# Per-run label files (the C1 / C9 collision the rehearsal exposed)
# ---------------------------------------------------------------------------
def test_a_bare_label_spec_applies_to_every_run():
    assert parse_label_spec("a.json,b.json") == (None, ("a.json", "b.json"))
    assert resolve_label_files([(None, ("a.json",))], ["x", "y"]) == {
        "x": ("a.json",), "y": ("a.json",)}


def test_a_prefixed_label_spec_overrides_exactly_one_run():
    assert parse_label_spec("struct=s.json") == ("struct", ("s.json",))
    resolved = resolve_label_files(
        [(None, ("a.json", "b.json")), ("struct", ("s.json",))],
        ["base", "struct"])
    assert resolved == {"base": ("a.json", "b.json"), "struct": ("s.json",)}


def test_a_label_spec_naming_an_unknown_run_is_refused():
    with pytest.raises(SystemExit) as exc:
        resolve_label_files([("ghost", ("s.json",))], ["base"])
    assert "ghost" in str(exc.value)


def test_no_label_spec_at_all_falls_back_to_the_two_pass_default():
    from scripts.validation.axis3_full_system import DEFAULT_LABEL_FILES
    assert resolve_label_files(None, ["base"]) == {"base": DEFAULT_LABEL_FILES}


def test_an_empty_run_prefix_is_rejected():
    with pytest.raises(Exception):
        parse_label_spec("=a.json")


# ---------------------------------------------------------------------------
# Edit E1: the judge batch is cut from the judged surface, not the envelope
# ---------------------------------------------------------------------------
def _prep():
    sys.path.insert(0, str(_REPO / "scripts" / "validation"))
    import agent_eval_prep_labeling
    return agent_eval_prep_labeling


@pytest.mark.parametrize("row", [
    {"reply": "free reply text"},
    {"reply": "free reply text", "output_contract": "free"},
    {"reply": '{"interpretation": "x"}', "output_contract": "structured",
     "narrative_surface": "stored surface"},
])
def test_the_prep_surface_agrees_with_the_join_surface(row):
    """One rule, two readers. A disagreement here would batch one population
    and score another."""
    assert _prep().judged_surface(row) == narrative_surface_of(row)


def test_a_structured_envelope_is_never_the_judged_surface():
    envelope = json.dumps({"interpretation": "ερμηνεία", "limitations": "όρια",
                           "status": "ok"}, ensure_ascii=False)
    row = {"reply": envelope, "output_contract": "structured",
           "narrative_surface": "ερμηνεία\nόρια"}
    assert _prep().judged_surface(row) == "ερμηνεία\nόρια"
    assert "interpretation" not in _prep().judged_surface(row)


def test_the_narration_floor_is_applied_to_the_surface_not_the_reply():
    """A long envelope wrapping a two-word interpretation must not be batched:
    that is the whole point of E1."""
    prep = _prep()
    row = {"reply": "x" * 4000, "output_contract": "structured",
           "narrative_surface": "σύντομο"}
    assert len(prep.judged_surface(row)) < prep.NARRATION_MIN_CHARS


# ---------------------------------------------------------------------------
# The coverage key the join writer actually writes
# ---------------------------------------------------------------------------
def test_the_joiner_reads_the_csv_key_the_join_writes(tmp_path):
    from scripts.validation.apply_axis3_human_audit_structured import load_coverage_map
    key = tmp_path / "axis3_full_system_coverage_key.csv"
    with key.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["decision_id", "run_id", "uid", "seed"])
        writer.writeheader()
        writer.writerow({"decision_id": "COV-0001", "run_id": "base_luna",
                         "uid": "A01#1", "seed": "20260921"})
    assert load_coverage_map(key) == {"COV-0001": ("base_luna", "A01#1")}


def test_the_two_json_key_shapes_still_work(tmp_path):
    from scripts.validation.apply_axis3_human_audit_structured import load_coverage_map
    obj = tmp_path / "m.json"
    obj.write_text(json.dumps({"COV-0001": {"run_id": "r", "uid": "u"}}), encoding="utf-8")
    assert load_coverage_map(obj) == {"COV-0001": ("r", "u")}
    lst = tmp_path / "l.json"
    lst.write_text(json.dumps({"COV-0001": ["r", "u"]}), encoding="utf-8")
    assert load_coverage_map(lst) == {"COV-0001": ("r", "u")}


def test_a_csv_key_with_an_empty_uid_is_refused(tmp_path):
    from scripts.validation.apply_axis3_human_audit_structured import (
        AuditError, load_coverage_map)
    key = tmp_path / "k.csv"
    with key.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["decision_id", "run_id", "uid"])
        writer.writeheader()
        writer.writerow({"decision_id": "COV-0001", "run_id": "r", "uid": ""})
    with pytest.raises(AuditError):
        load_coverage_map(key)


def test_the_default_key_path_prefers_the_csv_that_exists(tmp_path):
    from scripts.validation.apply_axis3_human_audit_structured import (
        COVERAGE_KEY_FILENAME, COVERAGE_MAP_FILENAME, _default_map_path)
    audit = tmp_path / "axis3_full_system_audit.csv"
    audit.write_text("run_id\n", encoding="utf-8")
    # Nothing on disk: the documented JSON name is what the message points at.
    assert _default_map_path(audit).name == COVERAGE_MAP_FILENAME
    (tmp_path / COVERAGE_KEY_FILENAME).write_text("decision_id\n", encoding="utf-8")
    assert _default_map_path(audit).name == COVERAGE_KEY_FILENAME


# ---------------------------------------------------------------------------
# C4 now has a column at all
# ---------------------------------------------------------------------------
def test_the_records_row_carries_policy_ok_so_c4_is_computable():
    from scripts.validation.axis3_arm_compare import COMMON_CRITERION_SOURCES
    text = (_REPO / "scripts" / "validation" / "axis3_full_system.py").read_text(
        encoding="utf-8")
    assert '"policy_ok": not row.get("policy_problems")' in text
    assert "policy_ok" in COMMON_CRITERION_SOURCES["C4"]


# ---------------------------------------------------------------------------
# The two intervention-only columns the rates table needs
# ---------------------------------------------------------------------------
def test_the_envelope_status_is_read_off_the_structured_reply_only():
    from scripts.validation.axis3_full_system import envelope_status
    envelope = json.dumps({"status": "abstained", "interpretation": "x"})
    assert envelope_status({"reply": envelope, "output_contract": "structured"}) == "abstained"
    # A free reply is never parsed as an envelope, even if it happens to be JSON.
    assert envelope_status({"reply": envelope}) == ""
    assert envelope_status({"reply": envelope, "output_contract": "free"}) == ""


def test_an_unparseable_or_status_less_envelope_yields_the_empty_string():
    from scripts.validation.axis3_full_system import envelope_status
    for reply in ("not json at all", "{}", '{"status": 3}', ""):
        assert envelope_status({"reply": reply, "output_contract": "structured"}) == ""


def test_the_records_row_carries_what_the_rates_table_looks_for():
    """Without these two the 3.3 table prints "no status column" and cannot
    split a content filter out of the validity denominator."""
    text = (_REPO / "scripts" / "validation" / "axis3_full_system.py").read_text(
        encoding="utf-8")
    assert '"contract_outcome_finish_reason": row.get(' in text
    assert '"contract_status": envelope_status(row),' in text

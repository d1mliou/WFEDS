"""Tests for the structured-arm human-audit sibling.

The module under test writes human verdicts into the one file that holds the
adjudication of both comparison arms, so every test here is about a way that
file could be damaged or a judgement lost: a run that writes when it was only
asked to report, a decision keyed onto a row that does not exist, a write into
one of the four closed records, a coverage sheet joined before the auditor has
finished it.

Nothing here touches the real Exports tree, the pilot record or the backup. The
frozen-tree tests build directory NAMES under tmp_path, because the refusal is
a rule about the path and never about what is on disk behind it.
"""
import csv
import json

import pytest

from scripts.validation import stage_c_structured
from scripts.validation import apply_axis3_human_audit_structured as mod


# The audit CSV as edit C5b leaves it: `reply` gone, the four structured columns
# added. The module must be agnostic to this list and preserve whatever it finds,
# so the tests assert preservation rather than this exact set.
FIELDS = [
    "run_id", "model", "uid", "category", "deterministic_pass",
    "stage_c_pass1", "stage_c_pass2", "human_stage_c_required",
    "human_stage_c", "critical_event_hours", "literal_hours_found",
    "human_critical_coverage_required", "human_critical_coverage",
    "user_text", "arm", "contract_outcome", "judged_surface",
    "judged_surface_chars", "label_notes_pass1", "label_notes_pass2",
    "audit_comment",
]


def row(run_id, uid, stage_c="NO", coverage="NO", **extra):
    out = {name: "" for name in FIELDS}
    out.update({
        "run_id": run_id,
        "uid": uid,
        "model": "openai/gpt-5.6-luna",
        "category": "I04",
        "deterministic_pass": "PASS",
        "human_stage_c_required": stage_c,
        "human_critical_coverage_required": coverage,
        "critical_event_hours": "1,3,4",
        "user_text": "Πού δεν υπάρχει διαδρομή διαφυγής;",
        "judged_surface": "Η φωτιά επεκτείνεται προς τα βόρεια.",
    })
    out.update(extra)
    return out


def write_csv(path, rows, fields=None):
    fields = fields or FIELDS
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames), list(reader)


@pytest.fixture
def audit(tmp_path):
    """An out-dir with four referred rows across both arms."""
    out_dir = tmp_path / "axis3_arms_20260921" / "validation" / "3_agent"
    out_dir.mkdir(parents=True)
    return write_csv(out_dir / "axis3_full_system_audit.csv", [
        row("luna_struct", "I04#1", stage_c="YES", coverage="YES"),
        row("luna_struct", "I04#2", stage_c="YES"),
        row("terra_free", "B04#3", coverage="YES"),
        row("terra_free", "B04#5", stage_c="YES", coverage="YES"),
    ])


# --------------------------------------------------------------------------------
# The decision sets start empty, and the frozen original is not touched
# --------------------------------------------------------------------------------
def test_decision_sets_start_empty():
    """They are filled by hand after the structured arm is judged, never before.

    A shipped non-empty set would be a verdict nobody recorded.
    """
    assert mod.STAGE_C_PASS == set()
    assert mod.COVERAGE_PASS == set()


def test_does_not_import_the_frozen_original():
    """The 2026-09-21 record stays closed, so no code path here can reach it."""
    source = mod.__file__
    with open(source, "r", encoding="utf-8") as handle:
        text = handle.read()
    assert "import apply_axis3_human_audit" not in text
    assert "from apply_axis3_human_audit import" not in text


def test_reuses_the_single_frozen_pilot_helper():
    """One implementation of the pilot rule, imported, never a second copy."""
    assert mod.is_frozen_audit_path is stage_c_structured.is_frozen_audit_path


def test_audit_comments_match_the_frozen_original_wording():
    """Both adjudications must read identically or they cannot be diffed."""
    assert mod.COMMENT_STAGE_C_PASS == "Stage C: negative automatic label not confirmed."
    assert mod.COMMENT_STAGE_C_FAIL == "Stage C: contradicted or unsupported claim confirmed."
    assert mod.COMMENT_COVERAGE_PASS == "Coverage: all transitions covered by stated hours/ranges."
    assert mod.COMMENT_COVERAGE_FAIL == "Coverage: one or more transition hours omitted."


def test_default_audit_path_is_resolved_not_hard_coded():
    """It must come from DATA_DIR, so the repo carries no machine path."""
    path = mod.default_audit_path()
    assert path.is_relative_to(mod.DATA_DIR)
    assert path.name == "axis3_full_system_audit.csv"


# --------------------------------------------------------------------------------
# Dry run is the default and writes nothing
# --------------------------------------------------------------------------------
def test_dry_run_is_the_default_and_writes_nothing(audit, capsys):
    before = audit.read_bytes()
    assert mod.main([str(audit)]) == 0
    assert audit.read_bytes() == before
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "3 Stage-C and 3 coverage decisions" in out


def test_explicit_dry_run_flag_also_writes_nothing(audit):
    before = audit.read_bytes()
    assert mod.main([str(audit), "--dry-run"]) == 0
    assert audit.read_bytes() == before


def test_write_and_dry_run_together_are_refused(audit):
    with pytest.raises(SystemExit):
        mod.main([str(audit), "--write", "--dry-run"])


# --------------------------------------------------------------------------------
# The explicit flag writes
# --------------------------------------------------------------------------------
def test_write_applies_the_decisions(audit, monkeypatch, capsys):
    monkeypatch.setattr(mod, "STAGE_C_PASS", {("luna_struct", "I04#1")})
    monkeypatch.setattr(mod, "COVERAGE_PASS", {("terra_free", "B04#3")})

    assert mod.main([str(audit), "--write"]) == 0

    _, rows = read_csv(audit)
    by_key = {(r["run_id"], r["uid"]): r for r in rows}

    passed = by_key[("luna_struct", "I04#1")]
    assert passed["human_stage_c"] == "PASS"
    assert passed["human_critical_coverage"] == "FAIL"
    assert passed["audit_comment"] == (
        mod.COMMENT_STAGE_C_PASS + " " + mod.COMMENT_COVERAGE_FAIL)

    failed = by_key[("luna_struct", "I04#2")]
    assert failed["human_stage_c"] == "FAIL"
    assert failed["human_critical_coverage"] == ""
    assert failed["audit_comment"] == mod.COMMENT_STAGE_C_FAIL

    covered = by_key[("terra_free", "B04#3")]
    assert covered["human_stage_c"] == ""
    assert covered["human_critical_coverage"] == "PASS"
    assert covered["audit_comment"] == mod.COMMENT_COVERAGE_PASS

    assert "Applied 3 Stage-C and 3 coverage decisions." in capsys.readouterr().out


def test_written_csv_keeps_every_original_column_plus_the_comment(tmp_path):
    """The audit CSV carries columns this module knows nothing about. It must
    hand every one of them back untouched and add at most `audit_comment`."""
    fields = [f for f in FIELDS if f != "audit_comment"]
    src = write_csv(tmp_path / "axis3_full_system_audit.csv",
                    [row("luna_struct", "I04#1", stage_c="YES", coverage="YES")],
                    fields=fields)
    original_fields, original_rows = read_csv(src)

    assert mod.main([str(src), "--write"]) == 0

    new_fields, new_rows = read_csv(src)
    assert new_fields == original_fields + ["audit_comment"]
    untouched = [f for f in original_fields
                 if f not in {"human_stage_c", "human_critical_coverage"}]
    for name in untouched:
        assert new_rows[0][name] == original_rows[0][name], name
    assert new_rows[0]["audit_comment"]


# --------------------------------------------------------------------------------
# It never invents a decision
# --------------------------------------------------------------------------------
def test_stage_c_uid_absent_from_the_csv_raises_naming_it(audit, monkeypatch):
    monkeypatch.setattr(mod, "STAGE_C_PASS", {("luna_struct", "I04#9")})
    with pytest.raises(mod.AuditError) as excinfo:
        mod.main([str(audit)])
    assert "I04#9" in str(excinfo.value)
    assert "STAGE_C_PASS" in str(excinfo.value)


def test_coverage_uid_absent_from_the_csv_raises_naming_it(audit, monkeypatch):
    monkeypatch.setattr(mod, "COVERAGE_PASS", {("terra_free", "B04#7")})
    with pytest.raises(mod.AuditError) as excinfo:
        mod.main([str(audit)])
    assert "B04#7" in str(excinfo.value)
    assert "COVERAGE_PASS" in str(excinfo.value)


def test_a_bad_uid_is_caught_before_anything_is_written(audit, monkeypatch):
    """The refusal must cost nothing, so the file is byte-identical after it."""
    before = audit.read_bytes()
    monkeypatch.setattr(mod, "STAGE_C_PASS", {("luna_struct", "typo#1")})
    with pytest.raises(mod.AuditError):
        mod.main([str(audit), "--write"])
    assert audit.read_bytes() == before


def test_stage_c_decision_on_an_unreferred_row_raises(audit, monkeypatch):
    monkeypatch.setattr(mod, "STAGE_C_PASS", {("terra_free", "B04#3")})
    with pytest.raises(mod.AuditError) as excinfo:
        mod.main([str(audit)])
    assert "terra_free/B04#3" in str(excinfo.value)


# --------------------------------------------------------------------------------
# The closed records are off limits
# --------------------------------------------------------------------------------
@pytest.mark.parametrize("folder, expected", [
    ("pilot_records", "pilot"),
    ("_frozen_axis3_backup_20260921", "_frozen_axis3_backup_20260921"),
    ("axis3_full_system_20260921", "axis3_full_system_20260921"),
    ("agent_eval_20260901_082152", "agent_eval_20260901_082152"),
    ("agent_eval_20260901_093550", "agent_eval_20260901_093550"),
    ("agent_eval_20260901_103200", "agent_eval_20260901_103200"),
])
def test_frozen_path_is_refused(tmp_path, folder, expected):
    target = tmp_path / folder / "validation" / "3_agent" / "axis3_full_system_audit.csv"
    target.parent.mkdir(parents=True)
    write_csv(target, [row("luna_struct", "I04#1", stage_c="YES")])
    before = target.read_bytes()

    assert mod.frozen_reason(target) is not None
    with pytest.raises(SystemExit) as excinfo:
        mod.main([str(target), "--write"])
    message = str(excinfo.value)
    assert "REFUSED" in message
    assert expected in message
    assert target.read_bytes() == before


def test_an_ordinary_out_dir_is_not_frozen(audit):
    assert mod.frozen_reason(audit) is None


# --------------------------------------------------------------------------------
# The C9 coverage-sheet join
# --------------------------------------------------------------------------------
def sheet_and_map(audit, decisions, mapping):
    """`decisions` is decision_id -> verdict text, `mapping` decision_id -> pair."""
    sheet = audit.parent / mod.COVERAGE_SHEET_FILENAME
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["decision_id", "critical_event_hours", "user_text",
                                "pins", "judged_surface", "literal_hours_found",
                                "human_critical_coverage"])
        writer.writeheader()
        for decision_id, verdict in decisions.items():
            writer.writerow({"decision_id": decision_id,
                             "critical_event_hours": "1,3,4",
                             "judged_surface": "Η φωτιά επεκτείνεται.",
                             "human_critical_coverage": verdict})
    map_path = audit.parent / mod.COVERAGE_MAP_FILENAME
    map_path.write_text(
        json.dumps({k: {"run_id": v[0], "uid": v[1]} for k, v in mapping.items()},
                   ensure_ascii=False),
        encoding="utf-8")
    return sheet, map_path


def test_coverage_sheet_join_decides_the_coverage_column(audit):
    sheet_and_map(
        audit,
        {"d001": "PASS", "d002": "FAIL", "d003": "PASS"},
        {"d001": ("luna_struct", "I04#1"),
         "d002": ("terra_free", "B04#3"),
         "d003": ("terra_free", "B04#5")})

    assert mod.main([str(audit), "--write"]) == 0

    _, rows = read_csv(audit)
    by_key = {(r["run_id"], r["uid"]): r for r in rows}
    assert by_key[("luna_struct", "I04#1")]["human_critical_coverage"] == "PASS"
    assert by_key[("terra_free", "B04#3")]["human_critical_coverage"] == "FAIL"
    assert by_key[("terra_free", "B04#5")]["human_critical_coverage"] == "PASS"


def test_dry_run_fails_when_a_decision_id_does_not_resolve(audit):
    sheet_and_map(
        audit,
        {"d001": "PASS", "d404": "FAIL", "d003": "PASS"},
        {"d001": ("luna_struct", "I04#1"),
         "d003": ("terra_free", "B04#5")})
    before = audit.read_bytes()
    with pytest.raises(mod.AuditError) as excinfo:
        mod.main([str(audit)])
    assert "d404" in str(excinfo.value)
    assert audit.read_bytes() == before


def test_join_refuses_a_sheet_with_an_unrecorded_decision(audit):
    sheet_and_map(
        audit,
        {"d001": "PASS", "d002": "", "d003": "PASS"},
        {"d001": ("luna_struct", "I04#1"),
         "d002": ("terra_free", "B04#3"),
         "d003": ("terra_free", "B04#5")})
    with pytest.raises(mod.AuditError) as excinfo:
        mod.main([str(audit)])
    assert "d002" in str(excinfo.value)
    assert "blank" in str(excinfo.value)


def test_join_refuses_a_required_row_the_sheet_never_covers(audit):
    sheet_and_map(
        audit,
        {"d001": "PASS", "d003": "PASS"},
        {"d001": ("luna_struct", "I04#1"),
         "d003": ("terra_free", "B04#5")})
    with pytest.raises(mod.AuditError) as excinfo:
        mod.main([str(audit)])
    assert "terra_free/B04#3" in str(excinfo.value)


def test_requested_coverage_sheet_that_is_missing_is_refused(audit, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        mod.main([str(audit), "--coverage-sheet", str(tmp_path / "nope.csv")])
    assert "REFUSED" in str(excinfo.value)


def test_no_sheet_says_so_out_loud(audit, capsys):
    mod.main([str(audit)])
    assert "coverage sheet: none found" in capsys.readouterr().out

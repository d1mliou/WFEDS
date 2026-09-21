"""Apply the completed human adjudication of the TWO comparison arms to the
Axis-3 structured-output audit CSV.

Item 8 of section 4.4 of
`scripts/validation/axis3_structured_output_preregistration.md`, revision 6.

WHAT THIS IS AND WHAT IT IS NOT. This is the SIBLING of
`scripts/validation/apply_axis3_human_audit.py`, not a new version of it. That
module is the closed record of the 2026-09-21 adjudication: its two decision
sets are the verdict itself, so they are frozen, that file is never edited, and
nothing here imports from it. Sharing code with it would make the closed record
mutable from this file, which is precisely what must not be possible. The shape
below therefore mirrors it deliberately and duplicates the few lines it must.

Its own decision sets start EMPTY. They are filled by the author, by hand, after
the four comparison run ids (`luna_free`, `terra_free`, `luna_struct`,
`terra_struct`) have been judged, and never before.

THREE DIFFERENCES FROM THE FROZEN ORIGINAL, each one deliberate.

1. `--dry-run` is the DEFAULT. The original rewrites the CSV the moment it is
   invoked, which is a trap: the audit CSV is the only copy of decisions that
   cost human hours, and a half-filled decision set applied by accident writes
   FAIL over every row it does not name. Here the destructive path needs the
   explicit `--write`, so the accident has to be typed out.
2. A refusal on the frozen trees. The pilot rule is reused from
   `stage_c_structured.is_frozen_audit_path`, never restated, and this module
   adds the three other closed records it alone can be pointed at.
3. The coverage-sheet join of edit C9 (2.11). The 60 coverage decisions are
   taken on a shuffled, arm-blind sheet under opaque `decision_id` values; this
   module maps each one back to its `(run_id, uid)`, and only after every
   decision on that sheet is recorded. `--dry-run` fails if any `decision_id`
   does not resolve.

ATTRIBUTION, the standing rule of section 2.11, quoted verbatim:

    Attribution wording, mandated (Decision log 2026-08-27): "claims were
    labelled by Claude [exact model and effort], the user reviewed the complete
    labelled set and could override any label". Never "human-labelled, Claude
    verified".

The reverse formulation is forbidden because it inverts who did what: the label
pass is machine work that a human reviewed in full, not human work a machine
checked, and only the first sentence describes what actually happened.

    python scripts/validation/apply_axis3_human_audit_structured.py [AUDIT_CSV]
        [--write] [--coverage-sheet CSV --coverage-map JSON]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(_HERE.parent / "cell2fire"))

from _paths import DATA_DIR  # noqa: E402

# The pilot-tree rule has exactly one implementation and this is not it. Both
# spellings are needed because this file is imported as
# `scripts.validation.apply_axis3_human_audit_structured` by the test suite and
# run as a plain script by the operator.
try:
    from .stage_c_structured import is_frozen_audit_path
except ImportError:  # run as a script: no parent package, but _HERE is on sys.path
    from stage_c_structured import is_frozen_audit_path  # noqa: E402


# --------------------------------------------------------------------------------
# The decision sets. EMPTY until the author fills them.
# --------------------------------------------------------------------------------
# Keyed `(run_id, uid)` exactly as the frozen original, because `load_audit`
# (axis3_full_system.py:112-117) keys human decisions on that pair and nothing
# else, so a key shaped any other way would silently decide no row at all.
#
# Every Stage-C row requiring adjudication contains a confirmed contradicted or
# unsupported atomic claim unless it is named here. Two-pass-clean rows never
# reach this mapping.
STAGE_C_PASS: set[tuple[str, str]] = set()

# Coverage passes only where every listed transition hour is stated directly or
# falls inside an accurate, explicit range in the narrative surface. Under edit
# C9 the 60 coverage decisions arrive through the blinded sheet instead, and
# this set stays empty; it is the fallback for a run scored before C9.
COVERAGE_PASS: set[tuple[str, str]] = set()


# --------------------------------------------------------------------------------
# Paths. Resolved, never hard-coded.
# --------------------------------------------------------------------------------
# The workspace root differs per machine and a literal path would both break on
# the second computer and publish an account name. `axis3_arms_20260921` is the
# single `--out-dir` of Phase 5 step 17, where the four comparison run ids and
# the calibration-only `gemini_free` are joined.
ARMS_DIR_NAME = "axis3_arms_20260921"
AUDIT_FILENAME = "axis3_full_system_audit.csv"
# Edit C9's two artefacts. Names are conventional and overridable: if the audit
# writer spells them differently, pass --coverage-sheet / --coverage-map rather
# than editing this file.
COVERAGE_SHEET_FILENAME = "axis3_full_system_coverage_sheet.csv"
COVERAGE_MAP_FILENAME = "axis3_full_system_coverage_map.json"
# What `axis3_full_system.write_coverage_sheet` actually writes. The two names
# were chosen independently in two files and did not meet; the CSV is the one on
# disk, so it is tried FIRST and the JSON name is kept as the documented
# alternative rather than quietly dropped.
COVERAGE_KEY_FILENAME = "axis3_full_system_coverage_key.csv"

# The closed records. Writing into any of them is a refusal and not a warning:
# each one is either a set of model calls that were paid for once and cannot be
# reproduced, or a human verdict that was recorded once and is cited as final.
# The pilot tree is NOT listed here; `is_frozen_audit_path` owns that rule.
FROZEN_DIR_MARKERS = (
    "_frozen_axis3_backup_20260921",  # read-only backup of the four closed records
    "axis3_full_system_20260921",     # the CLOSED 2026-09-21 adjudication
)
# The three corrected 2026-09-01 baseline runs, matched by prefix so a fourth
# sibling directory cannot slip past a hand-written list of three names.
FROZEN_DIR_PREFIXES = ("agent_eval_20260901_",)

# Printed and written verbatim as in the frozen original, so the two records
# read identically and a reader can diff one adjudication against the other
# without first normalising the prose.
COMMENT_STAGE_C_PASS = "Stage C: negative automatic label not confirmed."
COMMENT_STAGE_C_FAIL = "Stage C: contradicted or unsupported claim confirmed."
COMMENT_COVERAGE_PASS = "Coverage: all transitions covered by stated hours/ranges."
COMMENT_COVERAGE_FAIL = "Coverage: one or more transition hours omitted."


class AuditError(ValueError):
    """A decision could not be applied as written.

    Always raised with the offending `(run_id, uid)` or `decision_id` in the
    message: a typo in a decision set would otherwise drop a human judgement
    silently, and a dropped PASS reads as a FAIL in every downstream rate.
    """


def default_audit_path() -> Path:
    return DATA_DIR / "Exports" / ARMS_DIR_NAME / "validation" / "3_agent" / AUDIT_FILENAME


def frozen_reason(path) -> str | None:
    """Name the closed record `path` sits in, or None if it sits in none.

    Returns a reason string rather than a bool so the refusal can say WHICH rule
    caught the path; an operator who is told only "refused" tends to try again
    with a slightly different spelling of the same wrong folder.
    """
    if is_frozen_audit_path(path):
        return ("the frozen pilot audit record (scripts/validation/pilot_records), "
                "which README.md in that tree says is never re-run and never scored")
    parts = Path(path).resolve().parts
    for marker in FROZEN_DIR_MARKERS:
        if marker in parts:
            return f"the closed record {marker}"
    for part in parts:
        for prefix in FROZEN_DIR_PREFIXES:
            if part.startswith(prefix):
                return f"the stored 2026-09-01 baseline run {part}"
    return None


# --------------------------------------------------------------------------------
# The coverage-sheet join of edit C9 (2.11)
# --------------------------------------------------------------------------------
def _coverage_map_from_csv(path) -> dict[str, tuple[str, str]]:
    """The mapping as `write_coverage_sheet` writes it: a CSV key file.

    Columns `decision_id, run_id, uid, seed`. The seed is read and ignored here:
    it is provenance for the reader of the sheet, not an input to the join.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "decision_id" not in rows[0]:
        raise AuditError(f"{path}: coverage key CSV has no decision_id column")
    out: dict[str, tuple[str, str]] = {}
    for row in rows:
        run_id, uid = (row.get("run_id") or "").strip(), (row.get("uid") or "").strip()
        if not run_id or not uid:
            raise AuditError(
                f"{path}: decision_id {row.get('decision_id')!r} has an empty "
                "run_id or uid")
        out[str(row["decision_id"]).strip()] = (run_id, uid)
    return out



# --------------------------------------------------------------------------------
# Chain of custody: refuse rehearsal artefacts unless the operator says so
# --------------------------------------------------------------------------------
# This tool turns sheets into human verdicts on the record. A rehearsal sheet
# carries fabricated decisions; letting one through would write them into the
# audit CSV indistinguishably from a real adjudication, which is the one thing
# the audit trail exists to prevent. A stamp only protects anything if the tool
# downstream of it looks.
REHEARSAL_MARKERS = ("REHEARSAL", "REHEARSAL_STUB")


def rehearsal_columns(fieldnames) -> list[str]:
    """Column names that mark their file as rehearsal output."""
    return [str(name) for name in (fieldnames or [])
            if any(marker in str(name).upper() for marker in REHEARSAL_MARKERS)]


def refuse_rehearsal_sheet(path, fieldnames, allow: bool) -> None:
    """Refuse a stamped sheet unless --rehearsal was typed. Never silent."""
    marked = rehearsal_columns(fieldnames)
    if "REHEARSAL" in Path(path).name.upper():
        marked.append("the file name")
    if marked and not allow:
        raise AuditError(
            f"{path} is REHEARSAL output ({', '.join(marked)}), so its decisions "
            "are fabricated and are not a human verdict. Refusing to apply them "
            "to an audit record. Pass --rehearsal to exercise the plumbing "
            "against it; the run will then refuse to --write.")


def load_coverage_map(path) -> dict[str, tuple[str, str]]:
    """Read the `decision_id -> (run_id, uid)` mapping written beside the sheet.

    Three shapes are accepted, a CSV key file and two JSON shapes (a two-key
    object per id, a two-item list), because the writer of the mapping is a
    different edit in a different file and pinning its serialisation from here
    would couple two modules for no gain.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return _coverage_map_from_csv(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AuditError(f"{path}: coverage map must be a JSON object keyed by decision_id")
    out: dict[str, tuple[str, str]] = {}
    for decision_id, value in raw.items():
        if isinstance(value, dict):
            run_id, uid = value.get("run_id"), value.get("uid")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            run_id, uid = value
        else:
            raise AuditError(
                f"{path}: decision_id {decision_id!r} maps to {value!r}, which is "
                "neither a run_id/uid object nor a two-item [run_id, uid] list")
        if not run_id or not uid:
            raise AuditError(
                f"{path}: decision_id {decision_id!r} maps to an empty run_id or uid")
        out[str(decision_id)] = (str(run_id), str(uid))
    return out


def _default_map_path(audit_path: Path) -> Path:
    """The key file beside the audit CSV: the CSV spelling first, then the JSON.

    Falling back to the JSON name when neither exists keeps the "no sheet found"
    message pointing at the name the reader is most likely to look for.
    """
    for name in (COVERAGE_KEY_FILENAME, COVERAGE_MAP_FILENAME):
        candidate = audit_path.parent / name
        if candidate.exists():
            return candidate
    return audit_path.parent / COVERAGE_MAP_FILENAME


def join_coverage_sheet(sheet_path, map_path,
                        allow_rehearsal: bool = False) -> dict[tuple[str, str], bool]:
    """Resolve every recorded sheet decision back to its `(run_id, uid)`.

    The join happens only after ALL decisions on the sheet are recorded (2.11):
    a blank row means the auditor is still working, and joining a partial sheet
    would write FAIL over rows nobody has looked at yet. An unresolvable
    `decision_id` is an error and not a skip, because the blinding is only
    honest if every opaque id is accounted for.
    """
    mapping = load_coverage_map(map_path)
    with Path(sheet_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        sheet_fields = reader.fieldnames
        rows = list(reader)
    refuse_rehearsal_sheet(sheet_path, sheet_fields, allow_rehearsal)
    if not rows:
        raise AuditError(f"{sheet_path}: coverage sheet has no rows")
    if "decision_id" not in rows[0]:
        raise AuditError(f"{sheet_path}: coverage sheet has no decision_id column")

    unresolved: list[str] = []
    blank: list[str] = []
    bad_value: list[str] = []
    verdicts: dict[tuple[str, str], bool] = {}
    seen: set[str] = set()
    for row in rows:
        decision_id = (row.get("decision_id") or "").strip()
        verdict = (row.get("human_critical_coverage") or "").strip().upper()
        if decision_id in seen:
            raise AuditError(f"{sheet_path}: decision_id {decision_id!r} appears twice")
        seen.add(decision_id)
        if decision_id not in mapping:
            unresolved.append(decision_id)
            continue
        if not verdict:
            blank.append(decision_id)
            continue
        if verdict not in {"PASS", "FAIL"}:
            bad_value.append(f"{decision_id}={verdict}")
            continue
        verdicts[mapping[decision_id]] = verdict == "PASS"

    if unresolved:
        raise AuditError(
            f"{sheet_path}: {len(unresolved)} decision_id value(s) do not resolve in "
            f"{map_path}: {', '.join(sorted(unresolved))}")
    if blank:
        raise AuditError(
            f"{sheet_path}: {len(blank)} coverage decision(s) are still blank, so the "
            f"join is premature (2.11 joins only after every decision is recorded): "
            f"{', '.join(sorted(blank))}")
    if bad_value:
        raise AuditError(
            f"{sheet_path}: human_critical_coverage must be PASS or FAIL: "
            f"{', '.join(sorted(bad_value))}")
    return verdicts


# --------------------------------------------------------------------------------
# Applying the decisions
# --------------------------------------------------------------------------------
def read_audit(path, allow_rehearsal: bool = False) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise AuditError(f"{path}: audit CSV has no header")
        rows = list(reader)
    refuse_rehearsal_sheet(path, fieldnames, allow_rehearsal)
    for column in ("run_id", "uid", "human_stage_c_required",
                   "human_critical_coverage_required"):
        if column not in fieldnames:
            raise AuditError(f"{path}: audit CSV has no {column} column")
    # Kept in the file's own column order, with audit_comment appended only if
    # the writer did not emit it, so the written file keeps every column it came
    # with and gains at most one.
    out_fields = list(fieldnames)
    if "audit_comment" not in out_fields:
        out_fields.append("audit_comment")
    return out_fields, rows


def apply_decisions(rows, coverage_verdicts=None):
    """Fill the human columns in place and return one line per decided row.

    `coverage_verdicts` is the joined C9 sheet when there is one. When it is
    given it is the authority on coverage and must cover every required row:
    the sheet is a census of all 60 decisions, so a required row missing from it
    is a lost judgement, not a FAIL.
    """
    present = {(row["run_id"], row["uid"]) for row in rows}
    changes: list[str] = []

    missing = sorted(key for key in STAGE_C_PASS if key not in present)
    if missing:
        raise AuditError(
            "STAGE_C_PASS names row(s) that are not in the audit CSV: "
            + ", ".join(f"{run_id}/{uid}" for run_id, uid in missing))
    missing = sorted(key for key in COVERAGE_PASS if key not in present)
    if missing:
        raise AuditError(
            "COVERAGE_PASS names row(s) that are not in the audit CSV: "
            + ", ".join(f"{run_id}/{uid}" for run_id, uid in missing))
    if coverage_verdicts:
        missing = sorted(key for key in coverage_verdicts if key not in present)
        if missing:
            raise AuditError(
                "the coverage sheet resolves to row(s) that are not in the audit CSV: "
                + ", ".join(f"{run_id}/{uid}" for run_id, uid in missing))
        conflict = sorted(key for key in COVERAGE_PASS
                          if key in coverage_verdicts and not coverage_verdicts[key])
        if conflict:
            raise AuditError(
                "COVERAGE_PASS and the coverage sheet disagree on row(s): "
                + ", ".join(f"{run_id}/{uid}" for run_id, uid in conflict))

    # A decision written onto a row that was never referred is also a lost
    # judgement, and it would land on a row whose verdict is already settled.
    wrong_column = sorted(
        f"{row['run_id']}/{row['uid']}" for row in rows
        if (row["run_id"], row["uid"]) in STAGE_C_PASS
        and row["human_stage_c_required"] != "YES")
    if wrong_column:
        raise AuditError(
            "STAGE_C_PASS names row(s) whose human_stage_c_required is not YES: "
            + ", ".join(wrong_column))

    stage_applied = 0
    coverage_applied = 0
    for row in rows:
        key = (row["run_id"], row["uid"])
        comments: list[str] = []

        if row["human_stage_c_required"] == "YES":
            passed = key in STAGE_C_PASS
            row["human_stage_c"] = "PASS" if passed else "FAIL"
            comments.append(COMMENT_STAGE_C_PASS if passed else COMMENT_STAGE_C_FAIL)
            stage_applied += 1

        if row["human_critical_coverage_required"] == "YES":
            if coverage_verdicts is not None:
                if key not in coverage_verdicts:
                    raise AuditError(
                        f"the coverage sheet carries no decision for {key[0]}/{key[1]}, "
                        "which the audit CSV marks human_critical_coverage_required=YES")
                passed = coverage_verdicts[key]
            else:
                passed = key in COVERAGE_PASS
            row["human_critical_coverage"] = "PASS" if passed else "FAIL"
            comments.append(COMMENT_COVERAGE_PASS if passed else COMMENT_COVERAGE_FAIL)
            coverage_applied += 1

        if comments:
            row["audit_comment"] = " ".join(comments)
            changes.append(
                f"{row['run_id']}/{row['uid']}: "
                f"stage_c={row.get('human_stage_c', '') or '-'} "
                f"coverage={row.get('human_critical_coverage', '') or '-'} "
                f"| {row['audit_comment']}")

    return changes, stage_applied, coverage_applied


def write_audit_atomic(path, fieldnames, rows) -> None:
    """Write through a temp file in the SAME directory and then replace.

    The audit CSV is the only copy of decisions that cost human hours. A plain
    open("w") truncates first, so an interrupt partway through leaves a header
    and half the verdicts. The temp file must share the destination directory or
    `os.replace` turns into a cross-device copy.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Apply the human adjudication of both Axis-3 comparison arms to "
                    "the audit CSV. Reports only, unless --write is given.")
    ap.add_argument("audit_csv", nargs="?", default=None,
                    help="audit CSV to apply to (default: resolved under DATA_DIR)")
    ap.add_argument("--rehearsal", action="store_true",
                    help="permit REHEARSAL-stamped sheets, for exercising the "
                         "plumbing. Never combined with --write: a rehearsal "
                         "decision must not reach a real audit record.")
    ap.add_argument("--write", action="store_true",
                    help="actually rewrite the CSV; without it nothing is written")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit form of the default, reports and writes nothing")
    ap.add_argument("--coverage-sheet", default=None,
                    help="the blinded C9 coverage sheet (default: beside the audit CSV)")
    ap.add_argument("--coverage-map", default=None,
                    help="the decision_id to (run_id, uid) mapping (default: beside it)")
    a = ap.parse_args(argv)

    if a.write and a.dry_run:
        ap.error("--write and --dry-run contradict each other")
    if a.rehearsal and a.write:
        ap.error("--rehearsal and --write contradict each other: a fabricated "
                 "decision must never be written into an audit record")

    audit_path = Path(a.audit_csv) if a.audit_csv else default_audit_path()

    reason = frozen_reason(audit_path)
    if reason is not None:
        sys.exit(f"REFUSED: {audit_path} is inside {reason}. "
                 "Apply the structured adjudication to its own out-dir, never to a "
                 "closed record.")
    if not audit_path.exists():
        sys.exit(f"REFUSED: {audit_path} does not exist. Run axis3_full_system.py "
                 "first, or pass the audit CSV as the positional argument.")

    sheet_path = (Path(a.coverage_sheet) if a.coverage_sheet
                  else audit_path.parent / COVERAGE_SHEET_FILENAME)
    map_path = Path(a.coverage_map) if a.coverage_map else _default_map_path(audit_path)
    explicit_sheet = bool(a.coverage_sheet or a.coverage_map)

    coverage_verdicts = None
    if sheet_path.exists() and map_path.exists():
        coverage_verdicts = join_coverage_sheet(sheet_path, map_path,
                                                allow_rehearsal=a.rehearsal)
        print(f"coverage sheet: {sheet_path.name} joined through {map_path.name}, "
              f"{len(coverage_verdicts)} decisions resolved")
    elif explicit_sheet:
        missing_files = [str(p) for p in (sheet_path, map_path) if not p.exists()]
        sys.exit("REFUSED: coverage sheet requested but missing: "
                 + ", ".join(missing_files))
    else:
        # Printed every time, because silence here would read as "the sheet was
        # used" and the coverage column would come from COVERAGE_PASS unnoticed.
        print("coverage sheet: none found beside the audit CSV; COVERAGE_PASS is the "
              "only coverage source")

    fieldnames, rows = read_audit(audit_path, allow_rehearsal=a.rehearsal)
    changes, stage_applied, coverage_applied = apply_decisions(rows, coverage_verdicts)

    for line in changes:
        print(line)
    print(f"{stage_applied} Stage-C and {coverage_applied} coverage decisions "
          f"over {len(rows)} audit rows.")

    if not a.write:
        print(f"DRY RUN: nothing written to {audit_path}. Re-run with --write to apply.")
        return 0

    write_audit_atomic(audit_path, fieldnames, rows)
    print(f"Applied {stage_applied} Stage-C and {coverage_applied} coverage decisions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

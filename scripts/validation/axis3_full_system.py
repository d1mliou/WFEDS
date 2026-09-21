"""Combine Axis-3 stages into a full-system reliability score.

The corrected 2026-09-01 evaluation already stores three kinds of evidence:

* deterministic repetition records (Stage A, Stage B, numeric checks, policy);
* two independent Stage-C atomic-claim label sets;
* the expected hours at which the number of cut-off settlements increases.

This script joins those records by ``case_id#rep``.  It does not call an LLM or
the fire engine.  It deliberately limits the full-system score to the three
categories that were selected and labelled for Stage C in every corrected run.

The final gate is:

    A and B and numeric and policy and C-clean and critical-event coverage

Two judgements remain human-authoritative:

* Stage C whenever either labelling pass finds a negative claim or the passes
  disagree;
* critical-event coverage whenever the case requires cut-off transitions to be
  reported.

Until those cells are completed in the generated audit CSV, the report gives a
lower and upper bound rather than pretending that a final score exists.

WHAT THE STRUCTURED-OUTPUT PRE-REGISTRATION ADDED HERE (block C of section 4.3
of ``scripts/validation/axis3_structured_output_preregistration.md``).  The join
now has to serve two arms, the stored free-text baseline and the structured
arm, on one instrument, so it gained:

* C1  the stage-C directory and the label file names are CLI parameters, so a
      single-pass arm can be joined by the same code as a two-pass one;
* C2  an explicit no-judgeable-surface branch, decided in the SAME place as
      ``row["error"]`` and therefore ahead of any human override;
* C3  a no-surface record scores ``coverage = False``, never True and never
      None, so it can never drift into the optimistic bound;
* C4  ``clean_label_record`` returns None, not True, on an empty claim list;
* C5  the records CSV carries the arm and the contract diagnostics, while the
      audit CSV, which is the human coverage instrument, loses the raw reply
      and carries the rendered narrative surface instead;
* C6  ``mentioned_hours`` stays orientation-only and is computed over the
      narrative surface on every arm, never over the withheld slot values;
* C7  ``write_records`` survives an empty row list;
* C9  the coverage decisions are published as one shuffled, arm-blind sheet
      under opaque decision ids, with the mapping in a separate file.

THE PUBLISHED NUMERIC VERDICT (the author's correction of 2026-09-21, which
overrides anything in the pre-registration that contradicts it).  C1a, numeric
traceability, is published from ``numeric_ok_full_surface``, the 2.1-conformant
verdict that ``stage_c_structured.py`` recomputes offline over the prose PLUS
the rendered slot claims.  The run-time ``numeric_ok`` that ``agent_eval.py``
writes is the provisional prose-only lower bound and no longer decides anything
published.  The fallback to it is exact rather than a compromise: on a free-arm
row section 2.1 makes ``numeric_text == reply``, so the two verdicts are the
same number computed by the same function over the same string.  The report
prints how many rows took each path, so a reader can confirm the fallback only
ever served the free arm.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve().parent

# The compact payload is re-parsed with the ONE implementation that already
# exists, so the judgeable-surface predicate of 2.4 cannot fork from the parse
# `stage_c_structured.py` and `run_one` perform.  Both spellings are needed
# because this file is imported as `scripts.validation.axis3_full_system` by the
# test suite and run as a plain script by the operator.
try:
    from .stage_c_structured import parse_compact, parse_reply_object
except ImportError:  # run as a script: no parent package, but _HERE is on sys.path
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from stage_c_structured import parse_compact, parse_reply_object


NEGATIVE_LABELS = {"NOT_IN_RESULT", "CONTRADICTED"}
DECISIONS = {"PASS": True, "FAIL": False}

# The arm-neutral judged-surface floor of 2.4.  Declared here as a literal
# rather than imported, because `agent_eval_prep_labeling` pulls the whole
# sampling stack in behind it and this join must stay importable offline.
# `tests/validation/test_axis3_full_system.py` reads that module as text and
# asserts the two numbers cannot drift apart.
NARRATION_MIN_CHARS = 120

DEFAULT_STAGE_DIR = "stage_c_full"
DEFAULT_LABEL_FILES = ("stage_c_labels.json", "stage_c_labels_pass2.json")

# The seed that fixes the coverage sheet's shuffle (C9 / 2.11).  A published
# constant rather than a random draw: the blinding must be reproducible by a
# reader, and a fixed seed also keeps the decision ids stable across reruns so
# an auditor's part-finished sheet is not scrambled under them.
DEFAULT_AUDIT_SEED = 20260921

# The coverage decision rule, verbatim and in ONE place, because it is printed
# on the blinded sheet, in the audit guide and in the report, and three copies
# would be three chances to reword the rule the decision is made under.
COVERAGE_DECISION_RULE = (
    "Critical coverage passes when every hour in `critical_event_hours` is "
    "mentioned explicitly or accurately covered by a stated range. Counts need "
    "not be repeated."
)

# The 14 diagnostic fields 3.4.6 puts in the records CSV.  They are carried as
# columns and nothing else: see the invariant stated above `deterministic_pass`.
CT_RECORD_FIELDS = (
    "ct_scored", "ct_not_scored_reason", "ct_n_gt", "ct_n_emitted", "ct_matched",
    "ct_missing_periods", "ct_extra_periods", "ct_extra_in_payload",
    "ct_extra_not_in_payload", "ct_duplicate_periods", "ct_order_ok",
    "ct_exact_set", "ct_no_omission", "ct_over_inclusive",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_label_record(record: dict[str, Any] | None) -> bool | None:
    """Return True when a Stage-C label record has no negative atomic claim.

    C4: an empty or absent claims list returns None, not True.  A record with no
    claims is a record nobody judged, and returning True would have let it walk
    into the two-pass consensus branch and score a clean pass on the strength of
    a missing judgement.  Negative-label semantics are untouched: a non-empty
    list with no NOT_IN_RESULT and no CONTRADICTED still returns True.  Verified
    a no-op on the frozen 2026-09-01 data, where all 191 label records in both
    passes carry at least 7 claims (2.5).
    """
    if record is None:
        return None
    claims = [c for c in record.get("claims", []) if isinstance(c, dict)]
    if not claims:
        return None
    labels = {c.get("label") for c in claims}
    return not bool(labels & NEGATIVE_LABELS)


def parse_decision(value: str | None) -> bool | None:
    return DECISIONS.get((value or "").strip().upper())


def arm_of(row: dict[str, Any]) -> str:
    """"free" or "structured".  The stored 2026-09-01 rows predate the contract
    work and carry no `output_contract` key at all, so the absent key must read
    as the baseline arm rather than as a missing measurement."""
    return row.get("output_contract") or "free"


def narrative_surface_of(row: dict[str, Any]) -> str:
    """The arm's narrative surface (2.1), which is what every human-facing and
    orientation-level text check in this file reads.

    Three sources, in this order, and the order is the point.  A row written
    after the contract landed carries the surface `agent_eval.py` already
    computed, and reusing it guarantees the join scores the same string the run
    scored.  A free-arm row's surface IS its reply, stated here so the stored
    baseline needs no import at all.  Only a structured row with no stored
    surface falls through to `agent_eval._narrative_surface`, imported lazily
    because importing `agent_eval` resolves the OneDrive data directory and this
    join must stay usable on a machine that has none.
    """
    stored = row.get("narrative_surface")
    if isinstance(stored, str):
        return stored
    reply = row.get("reply") or ""
    if arm_of(row) != "structured":
        return reply
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import agent_eval  # noqa: E402  lazy on purpose, see the docstring
    return agent_eval._narrative_surface(reply, "structured")


def judgeable_surface(row: dict[str, Any], surface: str | None = None) -> bool:
    """The arm-neutral predicate of 2.4, stated once.

    Exactly three conditions: no runtime error, a compact payload parsed from
    this row's OWN `tool_outputs`, and a narrative surface of at least
    NARRATION_MIN_CHARS characters.  It NEVER consults `contract_outcome`: 2.4 is
    explicit that the outcome is the recorded reason for a verdict and never an
    input to one, because a free Greek reply is `free_text` and would be denied a
    surface by any rule that read the taxonomy instead of the text.
    """
    if row.get("error"):
        return False
    if parse_compact(row.get("tool_outputs")) is None:
        return False
    text = narrative_surface_of(row) if surface is None else surface
    return len(text.strip()) >= NARRATION_MIN_CHARS


def surface_reason(row: dict[str, Any]) -> str:
    """The reason recorded in `stage_c_source` / `critical_coverage_source` for a
    record with no judgeable surface (C2, C3): the row's `contract_outcome`.

    The stored 2026-09-01 rows have no such key, so two fallbacks stand in for
    it and both name the same object the taxonomy would: a row with `error` set
    is outcome 1, `runtime_error`, which is also the string the pre-correction
    code wrote and which the C7 regression check still expects; anything else is
    recorded as a surface loss rather than guessed at.
    """
    outcome = row.get("contract_outcome")
    if outcome:
        return str(outcome)
    return "runtime_error" if row.get("error") else "no_judgeable_surface"


def numeric_verdict(row: dict[str, Any]) -> tuple[bool, str]:
    """C1a, numeric traceability, and which of the two stored verdicts it used.

    The author's correction of 2026-09-21: the published cell and the gate read
    `numeric_ok_full_surface`, the 2.1-conformant verdict recomputed offline by
    `stage_c_structured.py` over the prose plus the rendered slot claims.  The
    run-time `numeric_ok` is the provisional prose-only lower bound, is kept
    unmodified as a diagnostic, and decides nothing published.

    The fallback is exact, not a concession.  `numeric_ok_full_surface` is None
    on every free-arm row by construction, because on a free arm 2.1 makes the
    numeric surface the reply itself, which is precisely the string the run-time
    check already scored with the same function; recomputing it there could only
    introduce drift.  The returned source string is counted in the report, so a
    reader can verify that the fallback served the free arm and nothing else.
    """
    full = row.get("numeric_ok_full_surface")
    if "numeric_ok_full_surface" in row and full is not None:
        return bool(full), "full_surface"
    return bool(row.get("numeric_ok")), "prose_only_fallback"


# HARD INVARIANT (3.4.2, 3.4.6), stated here because this is the line it guards.
# No ct_* field may enter `deterministic_pass` below or the `components` tuple in
# `analyse_run`.  The three critical-transition metrics are diagnostic columns of
# the records CSV and nothing else: `ct_exact_set` is a contract term that only
# one arm can emit at all, so admitting it here would score the free-text
# baseline zero on a field it was never asked for, and `ct_no_omission_recall`
# and `ct_over_inclusion` enter no gate whatsoever.  The components tuple stays
# `(det, stage_c, coverage)` exactly.  `test_stage_c_structured.py` reads this
# file as TEXT and fails on the substring, which is why neither the function
# below nor that assignment may so much as mention a contract field by name.
def deterministic_pass(row: dict[str, Any]) -> bool:
    """The deterministic gate used by the full-system score.

    Unlike the historical ``all_gates_ok`` field, numeric traceability is
    explicitly included here, and since the 2026-09-21 correction it is the
    full-surface verdict of ``numeric_verdict`` rather than the provisional
    prose-only one.
    """
    return bool(
        row.get("stage_a_ok")
        and row.get("stage_b_ok")
        and numeric_verdict(row)[0]
        and not row.get("policy_problems")
        and not row.get("error")
    )


def bounds(values: list[bool | None]) -> tuple[float, float, int]:
    """Return lower bound, upper bound and unresolved count.

    C7: deliberately UNCHANGED.  Counting pending toward the upper bound is the
    right semantics for a cell genuinely awaiting a human.  The leak 2.4 forbids
    was never here, it was upstream in which rows are allowed to be None, and
    C2/C3/C4 close it there.
    """
    if not values:
        return 0.0, 0.0, 0
    passed = sum(v is True for v in values)
    pending = sum(v is None for v in values)
    n = len(values)
    return passed / n, (passed + pending) / n, pending


def passk_bounds(records: list[dict[str, Any]]) -> tuple[float, float, int]:
    """Bounds for pass^k, grouping repetitions by case id."""
    by_case: dict[str, list[bool | None]] = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record["full_system"])

    states: list[bool | None] = []
    for vals in by_case.values():
        if any(v is False for v in vals):
            states.append(False)
        elif all(v is True for v in vals):
            states.append(True)
        else:
            states.append(None)
    return bounds(states)


def fmt_bound(lo: float, hi: float, pending: int) -> str:
    if pending == 0:
        return f"{lo:.1%}"
    return f"{lo:.1%} to {hi:.1%} ({pending} pending)"


def load_audit(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Existing human decisions, keyed on (run_id, uid).

    This reads the DECISION STORE, which keeps its keys.  The blinded coverage
    sheet of C9 is a separate, write-only artefact: its decisions arrive back
    here only after `apply_axis3_human_audit_structured.py` has joined every
    opaque `decision_id` to its (run_id, uid) and written the result into this
    file.  That is why the blinding can be total on the sheet without the resume
    path losing its key.
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {(r["run_id"], r["uid"]): r for r in rows}


def stage_c_categories(batch_dir: Path) -> set[str]:
    categories: set[str] = set()
    for path in sorted(batch_dir.glob("batch_*.json")):
        categories.update(r["category"] for r in read_json(path))
    if not categories:
        raise ValueError(f"No Stage-C batch files found under {batch_dir}")
    return categories


def label_map(path: Path, allow_rehearsal: bool = False) -> dict[str, dict[str, Any]]:
    """The uid-keyed label records of one judge pass.

    The stub guard sits here because this is the ONE door every label record
    walks through on its way into the join.  A file of machine-generated
    rehearsal labels is refused unless the caller said `--rehearsal` out loud:
    once a fabricated label becomes a pass rate in `axis3_full_system_report.md`
    it is indistinguishable from a judged one.  The import is at call time so
    the join stays importable on a checkout with no rehearsal driver in it.
    """
    records = read_json(path)
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from axis3_rehearsal import refuse_stub_labels  # noqa: E402  see docstring
    refuse_stub_labels(path, records, allow_rehearsal)
    return {r["uid"]: r for r in records}


def mentioned_hours(reply: str, hours: list[int]) -> list[int]:
    """Orientation only, never the authoritative coverage verdict.

    C6: byte-identical to the pre-contract version and called with the arm's
    NARRATIVE SURFACE at every call site, so `literal_hours_found` means the
    same thing on both arms.  A structured branch was rejected twice over: it
    would put the withheld `critical_transitions` periods back in front of the
    coverage decider, and it is not the no-op it looks like, because this
    pattern wants the numeral before the hour token while the frozen renderer
    writes the unit first ("Στην ώρα 4:").
    """
    hits = []
    for hour in hours:
        patterns = [rf"(?<!\d){hour}(?!\d)\s*(?:η|ή|ης|h|ώρα|ωρα)"]
        if hour == 1:
            patterns.append(r"πρώτ\w*\s+ώρα")
        if any(re.search(p, reply, re.I) for p in patterns):
            hits.append(hour)
    return hits


def compact_notes(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    return re.sub(r"\s+", " ", record.get("notes") or "").strip()


def envelope_status(row: dict[str, Any]) -> str:
    """The `status` the structured envelope declared, or "" off the structured arm.

    Read from the reply here rather than added to `stage_c_structured.py`: that
    module is frozen evidence for the pilot record, and the value is wanted by
    the records CSV only.  `parse_reply_object` is the same parse the validator
    used, so the two cannot read one envelope two ways.
    """
    if arm_of(row) != "structured":
        return ""
    obj = parse_reply_object(row.get("reply") or "")
    if isinstance(obj, dict) and isinstance(obj.get("status"), str):
        return obj["status"]
    return ""


def structured_complete_of(row: dict[str, Any]) -> bool | None:
    """SSP's completeness term, decomposed out of `contract_problems`.

    3.1 writes SSP as `C5-common AND schema_conformant AND structured_complete
    AND slot_values_exact AND status_consistent`, and says the four contract
    terms together are exactly `contract_outcome == "valid"`.  Mapping them onto
    the five semantic rules of 1.3: rule 4 is `status_consistent`, rule 5 is
    `slot_values_exact`, and rules 1 to 3 (one initial entry at the first
    period, one final entry at the final period, the full transition set in
    order) are completeness.  So this reads the recorded problems rather than
    re-deriving anything, and it is None on a free arm because a row with no
    contract has no completeness to report; False rather than None on a row that
    never parsed or never conformed, because there the completeness question was
    answered, in the negative, by the envelope itself.
    """
    if arm_of(row) != "structured":
        return None
    outcome = row.get("contract_outcome")
    if not outcome or outcome == "free_text":
        return None
    if outcome not in ("valid", "validator_invalid"):
        return False
    problems = row.get("contract_problems") or []
    return not any(str(p).startswith(("rule 1", "rule 2", "rule 3"))
                   for p in problems)


def analyse_run(name: str, run_dir: Path,
                audit: dict[tuple[str, str], dict[str, str]],
                stage_dir_name: str = DEFAULT_STAGE_DIR,
                label_files: tuple[str, ...] | list[str] = DEFAULT_LABEL_FILES,
                allow_rehearsal: bool = False,
                ) -> tuple[dict[str, Any], list[dict[str, Any]],
                           list[dict[str, str]]]:
    """Join one run's evidence into records, audit rows and a summary.

    C1: `stage_dir_name` and `label_files` were five hardcoded paths.  They are
    parameters because the structured arm is judged in one pass, and a joiner
    that demands `stage_c_labels_pass2.json` cannot read it at all.
    """
    raw_path = run_dir / "agent_eval_raw.jsonl"
    metrics_path = run_dir / "agent_eval_metrics.json"
    stage_dir = run_dir / stage_dir_name
    label_paths = [stage_dir / n for n in label_files]
    if not label_paths:
        raise ValueError("at least one Stage-C label file is required")

    for path in (raw_path, metrics_path, *label_paths):
        if not path.exists():
            raise FileNotFoundError(path)

    metrics = read_json(metrics_path)
    label_maps = [label_map(p, allow_rehearsal) for p in label_paths]
    categories = stage_c_categories(stage_dir)
    rows = [r for r in read_jsonl(raw_path) if r["category"] in categories]

    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, str]] = []
    for row in rows:
        uid = f"{row['case_id']}#{row['rep']}"
        label_records = [m.get(uid) for m in label_maps]
        cleans = [clean_label_record(r) for r in label_records]
        clean1 = cleans[0]
        clean2 = cleans[1] if len(cleans) > 1 else None
        previous = audit.get((name, uid), {})
        human_c = parse_decision(previous.get("human_stage_c"))

        arm = arm_of(row)
        surface = narrative_surface_of(row)
        has_surface = judgeable_surface(row, surface)
        reason = surface_reason(row)

        # C2.  The no-judgeable-surface guard is decided in the SAME branch as
        # `row["error"]`, ahead of the human override and of the consensus
        # branch.  Position is load-bearing: below the override, a stale or
        # mistaken human PASS sitting in the audit CSV, which `load_audit` keys
        # on (run_id, uid) alone, would resurrect a record that has no surface
        # to have been judged on.  `human_c_required = False` keeps these rows
        # out of the audit CSV entirely, which is what 2.11 requires, and is why
        # the rule does not enlarge the human workload.
        if row.get("error") or not has_surface:
            stage_c: bool | None = False
            c_source = reason
            human_c_required = False
        elif human_c is not None:
            stage_c = human_c
            c_source = "human_audit"
            human_c_required = True
        elif cleans and all(c is True for c in cleans):
            stage_c = True
            # Named by pass count, because a single-pass arm's clean verdict is
            # weaker evidence than two passes agreeing and the records CSV must
            # not let the two look alike.
            c_source = ("two_pass_clean_consensus" if len(cleans) > 1
                        else "single_pass_clean")
            human_c_required = False
        else:
            # Any negative label, disagreement, or missing label is audited.
            stage_c = None
            c_source = "human_audit_pending"
            human_c_required = True

        hours = [int(h) for h in row.get("cutoff_hours_to_cover") or []]
        human_coverage = parse_decision(previous.get("human_critical_coverage"))
        # C3.  False, never True and never None, for two independent reasons
        # verified in this file.  `bounds` puts None into the optimistic upper
        # bound, which is the exact leak 2.4 forbids; and
        # `critical_coverage_human_pending` counts only `human_audit_pending`,
        # so a None carrying any other source would slip past the all-resolved
        # check in `write_report` and be published as a point value when it is a
        # bound.
        if row.get("error") or not has_surface:
            coverage: bool | None = False
            coverage_source = reason
            human_coverage_required = False
        elif not hours:
            coverage = True
            coverage_source = "not_applicable"
            human_coverage_required = False
        elif human_coverage is not None:
            coverage = human_coverage
            coverage_source = "human_audit"
            human_coverage_required = True
        else:
            coverage = None
            coverage_source = "human_audit_pending"
            human_coverage_required = True

        det = deterministic_pass(row)
        numeric_ok_used, numeric_source = numeric_verdict(row)
        # No ct_* field here, ever: see the invariant above `deterministic_pass`.
        components = (det, stage_c, coverage)
        if any(v is False for v in components):
            full: bool | None = False
        elif all(v is True for v in components):
            full = True
        else:
            full = None

        # Read with a default of "" rather than []: an absent key is a free-arm
        # row with no contract, and "[]" would read as "the contract was
        # honoured and filled nothing", a measurement that row never made.
        slot_claims = row.get("slot_claims", "")
        mismatches = row.get("ct_value_mismatches")

        record = {
            "run_id": name,
            "model": metrics.get("model", ""),
            # C5a.  The records CSV is a machine artefact and is never the
            # coverage sheet, so the items 2.11 withholds from the decider are
            # allowed here and only here.
            "arm": arm,
            "uid": uid,
            "case_id": row["case_id"],
            "rep": row["rep"],
            "category": row["category"],
            "deterministic_pass": det,
            # C4 of the common comparison is a deterministic criterion with no
            # human tier, and it had no column at all: `axis3_arm_compare.py`
            # printed "n/a (no source column)" for every cell of it, which is a
            # pre-registered criterion silently missing from the table.  The
            # list itself stays out of the records CSV; the verdict is what the
            # comparison needs.
            "policy_ok": not row.get("policy_problems"),
            "numeric_ok_used": numeric_ok_used,
            "numeric_source": numeric_source,
            "judgeable_surface": has_surface,
            "judged_surface_chars": len(surface.strip()),
            "contract_outcome": row.get("contract_outcome", ""),
            # Two intervention-only diagnostics that the rates table of 3.3
            # needs and could not find: without the finish reason it cannot
            # split a provider content filter out of the validity denominator,
            # and without the envelope's own status it prints the abstention
            # rate as "not computed" rather than as a number.  Neither is a
            # common-comparison column and neither enters any gate.
            "contract_outcome_finish_reason": row.get(
                "contract_outcome_finish_reason", ""),
            "contract_status": envelope_status(row),
            "schema_conformant": row.get("schema_conformant", ""),
            "structured_complete": structured_complete_of(row),
            "slot_mismatches": ("" if mismatches is None else len(mismatches)),
            "schema_errors": row.get("schema_errors", ""),
            "rendered_slot_claims": slot_claims,
            "stage_c_pass1": clean1,
            "stage_c_pass2": clean2,
            "stage_c": stage_c,
            "stage_c_source": c_source,
            "critical_hours": hours,
            "critical_coverage": coverage,
            "critical_coverage_source": coverage_source,
            "full_system": full,
        }
        # Trap 1 of 3.4.6: the free and the structured runs are pooled into one
        # list before `write_records` takes its header from the first row, so
        # every key must exist on every record in both arms.  An empty cell here
        # means "free arm, no contract, not applicable", never a lost
        # measurement.
        record.update({k: row.get(k, "") for k in CT_RECORD_FIELDS})
        records.append(record)

        audit_rows.append({
            "run_id": name,
            "model": str(metrics.get("model", "")),
            # C5b.  The audit CSV is the coverage sheet.  It gains exactly four
            # columns and LOSES `reply`, which on the structured arm is the raw
            # envelope 2.11 forbids on a coverage row.  It must not gain
            # `structured_complete`, `schema_errors`, `slot_mismatches`,
            # `schema_conformant` or `rendered_slot_claims`: all five are
            # withheld from the coverage decision, and that decision is filled
            # on this very row.
            "arm": arm,
            "uid": uid,
            "category": row["category"],
            "contract_outcome": str(row.get("contract_outcome", "")),
            "deterministic_pass": "PASS" if det else "FAIL",
            "stage_c_pass1": "" if clean1 is None else ("PASS" if clean1 else "FAIL"),
            "stage_c_pass2": "" if clean2 is None else ("PASS" if clean2 else "FAIL"),
            "human_stage_c_required": "YES" if human_c_required else "NO",
            "human_stage_c": previous.get("human_stage_c", ""),
            "critical_event_hours": ",".join(map(str, hours)),
            # C5b / C6: over the narrative surface, so on the structured arm the
            # aid is not computed over the withheld transition periods.
            "literal_hours_found": ",".join(map(str, mentioned_hours(surface, hours))),
            "human_critical_coverage_required": "YES" if human_coverage_required else "NO",
            "human_critical_coverage": previous.get("human_critical_coverage", ""),
            "user_text": row.get("user_text", ""),
            "pins": json.dumps(row.get("pins"), ensure_ascii=False),
            "judged_surface": surface,
            "judged_surface_chars": str(len(surface.strip())),
            "label_notes_pass1": compact_notes(label_records[0]),
            "label_notes_pass2": compact_notes(
                label_records[1] if len(label_records) > 1 else None),
            "audit_comment": previous.get("audit_comment", ""),
        })

    p1 = bounds([r["full_system"] for r in records])
    pk = passk_bounds(records)
    surfaced = [r for r in records if r["judgeable_surface"]]
    summary = {
        "run_id": name,
        "model": metrics.get("model", ""),
        "arm": records[0]["arm"] if records else "",
        "scope_categories": sorted(categories),
        "label_passes": len(label_paths),
        "n_cases": len({r["case_id"] for r in records}),
        "n_repetitions": len(records),
        "full_system_pass1_lower": round(p1[0], 6),
        "full_system_pass1_upper": round(p1[1], 6),
        "full_system_pass1_pending": p1[2],
        "full_system_passk_lower": round(pk[0], 6),
        "full_system_passk_upper": round(pk[1], 6),
        "full_system_passk_pending_cases": pk[2],
        "stage_c_human_pending": sum(
            r["stage_c_source"] == "human_audit_pending" for r in records),
        "critical_coverage_human_pending": sum(
            r["critical_coverage_source"] == "human_audit_pending" for r in records),
        # C2/C3 make these hard failures, so their count must be visible: a
        # growing no-surface population is a result, not a footnote.
        "no_judgeable_surface": sum(1 for r in records if not r["judgeable_surface"]),
        # The author's correction, made auditable: which numeric verdict each
        # row's C1a cell came from.
        "numeric_source_full_surface": sum(
            1 for r in records if r["numeric_source"] == "full_surface"),
        "numeric_source_prose_fallback": sum(
            1 for r in records if r["numeric_source"] == "prose_only_fallback"),
        # C1a as published: over rows WITH a judgeable surface, because on an
        # empty string `score_numeric_traceability` returns a vacuous pass that
        # says nothing about the model (2.4).  The dropped count travels with
        # the cell.
        "c1a_numeric_pass": sum(1 for r in surfaced if r["numeric_ok_used"]),
        "c1a_denominator": len(surfaced),
        "c1a_excluded_no_surface": len(records) - len(surfaced),
        "stage_c_source_counts": dict(sorted(
            _count_by(records, "stage_c_source").items())),
        "critical_coverage_source_counts": dict(sorted(
            _count_by(records, "critical_coverage_source").items())),
    }
    return summary, records, audit_rows


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Counts per source value, so a pending population cannot hide inside a
    widening bound (C7's reporting requirement)."""
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record[key])] += 1
    return dict(counts)


# C5b.  Four columns added (`arm`, `contract_outcome`, `judged_surface`,
# `judged_surface_chars`), `reply` removed.  `pins` is added as well because
# 2.11 lists it among the things the coverage decider IS given; it was already
# implicit in `user_text` and is now explicit.
AUDIT_FIELDS = [
    "run_id", "model", "arm", "uid", "category", "contract_outcome",
    "deterministic_pass", "stage_c_pass1", "stage_c_pass2",
    "human_stage_c_required", "human_stage_c", "critical_event_hours",
    "literal_hours_found", "human_critical_coverage_required",
    "human_critical_coverage", "user_text", "pins", "judged_surface",
    "judged_surface_chars", "label_notes_pass1", "label_notes_pass2",
    "audit_comment",
]

# Everything 2.11 and 3.4.2 withhold from the coverage decider.  Named as a
# constant so the test can assert absence against the same list the code
# documents, rather than against a list a future editor has to remember.
AUDIT_WITHHELD_FIELDS = (
    "reply", "structured_complete", "schema_errors", "slot_mismatches",
    "schema_conformant", "rendered_slot_claims", "slot_claims", "numeric_text",
    "initial_state", "final_state", "critical_transitions",
)

# C9.  The blinded coverage sheet: these columns and nothing else.
COVERAGE_SHEET_FIELDS = [
    "decision_id", "critical_event_hours", "user_text", "pins",
    "judged_surface", "literal_hours_found", "decision_rule",
    "human_critical_coverage",
]

COVERAGE_KEY_FIELDS = ["decision_id", "run_id", "uid", "seed"]


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    """The decision store, keyed on (run_id, uid) so `load_audit` can resume."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_coverage_sheet(sheet_path: Path, key_path: Path,
                         rows: list[dict[str, str]],
                         seed: int = DEFAULT_AUDIT_SEED) -> list[dict[str, str]]:
    """C9: the coverage decisions as one shuffled, arm-blind sheet.

    `run_id` names the arm on every row, so withholding the JSON envelope while
    printing the arm withholds nothing.  The rows of every arm are therefore
    pooled, shuffled under a published seed and given opaque sequential
    `decision_id` values, and the (decision_id -> run_id, uid) mapping goes to a
    SEPARATE file that `apply_axis3_human_audit_structured.py` joins back only
    after all the decisions are recorded and written.

    The seed is fixed rather than drawn, so the sheet is reproducible by a
    reader and stable across reruns: an auditor's part-finished sheet keeps its
    decision ids.

    What this does NOT buy, and what 6.1 must keep saying: the narrative text
    itself can betray the arm, because a structured arm's `interpretation` plus
    `limitations` may read differently from a free reply.  The honest claim is
    that the sheet was arm-blind BY CONSTRUCTION at the sheet level, not that
    the decider could not infer the arm from the prose.
    """
    ordered = sorted(rows, key=lambda r: (r["run_id"], r["uid"]))
    random.Random(seed).shuffle(ordered)

    sheet, key = [], []
    for i, row in enumerate(ordered, start=1):
        decision_id = f"COV-{i:04d}"
        sheet.append({
            "decision_id": decision_id,
            "critical_event_hours": row["critical_event_hours"],
            "user_text": row["user_text"],
            "pins": row["pins"],
            "judged_surface": row["judged_surface"],
            "literal_hours_found": row["literal_hours_found"],
            "decision_rule": COVERAGE_DECISION_RULE,
            # Carried forward so a partly finished sheet survives a rerun.
            "human_critical_coverage": row.get("human_critical_coverage", ""),
        })
        key.append({"decision_id": decision_id, "run_id": row["run_id"],
                    "uid": row["uid"], "seed": str(seed)})

    with sheet_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COVERAGE_SHEET_FIELDS)
        writer.writeheader()
        writer.writerows(sheet)
    with key_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COVERAGE_KEY_FIELDS)
        writer.writeheader()
        writer.writerows(key)
    return key


# Joined into "1,3,4,5" so the records CSV stays machine-readable for
# `axis3_arm_compare.py`.  Trap 2 of 3.4.6: `write_records` special-cased
# exactly one list field, and every new list-valued column would otherwise be
# written as a Python list repr.
_CSV_LIST_FIELDS = (
    "critical_hours", "ct_missing_periods", "ct_extra_periods",
    "ct_extra_in_payload", "ct_extra_not_in_payload", "ct_duplicate_periods",
)
# These carry Greek sentences and validator messages, where a comma join would
# be ambiguous, so they are written as JSON instead.
_CSV_JSON_FIELDS = ("rendered_slot_claims", "schema_errors")


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """C7: an empty row list is written as an empty file instead of raising.

    `list(rows[0])` raised IndexError whenever a category filter matched
    nothing, which turned an empty scope into a crash three artefacts into the
    run.  The file is still created, so the artefact set of a run is complete
    and a downstream reader finds an empty table rather than a missing one; no
    header is invented, because the header of a records CSV is the union of
    what the rows actually carried.
    """
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for field in _CSV_LIST_FIELDS:
                if isinstance(out.get(field), list):
                    out[field] = ",".join(map(str, out[field]))
            for field in _CSV_JSON_FIELDS:
                if isinstance(out.get(field), list):
                    out[field] = json.dumps(out[field], ensure_ascii=False)
            writer.writerow(out)


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    audit_complete = all(
        s["full_system_pass1_pending"] == 0
        and s["full_system_passk_pending_cases"] == 0
        and s["stage_c_human_pending"] == 0
        and s["critical_coverage_human_pending"] == 0
        for s in summaries
    )
    status_text = (
        "The human audit is complete. These percentages are final for the "
        "corrected Stage-C scope."
        if audit_complete else
        "Until the human audit cells are completed, the table reports lower "
        "and upper bounds."
    )
    results_heading = "## Final results" if audit_complete else "## Current bounds"
    lines = [
        "# Axis 3 full-system pass",
        "",
        "## Definition",
        "",
        "`full_system = A arguments AND B tool call AND numeric traceability AND policy "
        "AND Stage-C clean narration AND critical-event coverage`",
        "",
        "The score is restricted to the three categories with two complete Stage-C "
        f"labelling passes. {status_text}",
        "",
        results_heading,
        "",
        "| model | arm | cases | repetitions | full-system pass^1 | full-system pass^5 | "
        "Stage C pending | critical coverage pending | no judgeable surface |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        p1 = fmt_bound(s["full_system_pass1_lower"],
                       s["full_system_pass1_upper"],
                       s["full_system_pass1_pending"])
        pk = fmt_bound(s["full_system_passk_lower"],
                       s["full_system_passk_upper"],
                       s["full_system_passk_pending_cases"])
        lines.append(
            f"| {s['model']} | {s['arm']} | {s['n_cases']} | {s['n_repetitions']} | {p1} | "
            f"{pk} | {s['stage_c_human_pending']} | "
            f"{s['critical_coverage_human_pending']} | "
            f"{s['no_judgeable_surface']} |")

    # The author's correction, published where it can be checked rather than
    # asserted in a docstring.
    lines += [
        "",
        "## C1a numeric traceability, and which verdict produced it",
        "",
        "The published C1a cell is `numeric_ok_full_surface`, the 2.1-conformant "
        "verdict recomputed offline over the prose plus the rendered slot claims. "
        "The run-time `numeric_ok` is the provisional prose-only lower bound and "
        "decides nothing published. On a free arm the two are identical by "
        "construction, because 2.1 makes the numeric surface the reply itself, so "
        "the fallback below is exact rather than a compromise. Rows with no "
        "judgeable surface are excluded from the denominator, because the check "
        "returns a vacuous pass on an empty string.",
        "",
        "| model | arm | C1a pass | denominator | excluded, no surface | rows scored "
        "on the full surface | rows on the prose-only fallback |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['arm']} | {s['c1a_numeric_pass']} | "
            f"{s['c1a_denominator']} | {s['c1a_excluded_no_surface']} | "
            f"{s['numeric_source_full_surface']} | "
            f"{s['numeric_source_prose_fallback']} |")

    # C7: the pending population broken down by source, so it cannot hide inside
    # a widening bound.
    lines += [
        "",
        "## Verdict sources",
        "",
        "| model | arm | Stage C sources | critical coverage sources |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        c_src = "; ".join(f"{k} {v}" for k, v in s["stage_c_source_counts"].items())
        v_src = "; ".join(f"{k} {v}"
                          for k, v in s["critical_coverage_source_counts"].items())
        lines.append(f"| {s['model']} | {s['arm']} | {c_src} | {v_src} |")

    lines += [
        "",
        "## Human audit rule",
        "",
        "- Fill `human_stage_c` only where `human_stage_c_required=YES`.",
        "- Fill `human_critical_coverage` only on the blinded coverage sheet, "
        "against a `decision_id`.",
        "- Valid values are `PASS` and `FAIL`.",
        f"- {COVERAGE_DECISION_RULE}",
        "- `literal_hours_found` is only an orientation aid. It never decides the score.",
        "- A record with no judgeable surface is decided deterministically, fails "
        "both Stage C and coverage, and is never audited and never counted toward "
        "an optimistic bound.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --run")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_labels(value: str) -> tuple[str, ...]:
    names = tuple(n.strip() for n in value.split(",") if n.strip())
    if not names:
        raise argparse.ArgumentTypeError("--labels needs at least one file name")
    return names


def parse_label_spec(value: str) -> tuple[str | None, tuple[str, ...]]:
    """`FILES` for every run, or `RUN=FILES` for one run.

    C1 made the label file names a parameter, and C9 makes the blinded coverage
    sheet pool EVERY arm into one shuffled file.  Those two pull against each
    other as soon as the arms are judged differently: the baseline arm has two
    label passes and the structured arm has one, so a single global `--labels`
    would force two invocations, two out-dirs and therefore two coverage sheets,
    one per arm, which destroys the blinding C9 exists to create.  A per-run
    override lets all the runs be joined in one invocation, which is the only
    shape in which the pooled sheet is possible.
    """
    if "=" in value:
        name, files = value.split("=", 1)
        name = name.strip()
        if not name:
            raise argparse.ArgumentTypeError(
                "--labels RUN=FILES needs a run name before the '='")
        return name, parse_labels(files)
    return None, parse_labels(value)


def resolve_label_files(specs, run_names) -> dict[str, tuple[str, ...]]:
    """Fold the `--labels` specs into one file list per run name."""
    default = DEFAULT_LABEL_FILES
    per_run: dict[str, tuple[str, ...]] = {}
    for name, files in specs or []:
        if name is None:
            default = files
        else:
            per_run[name] = files
    unknown = sorted(set(per_run) - set(run_names))
    if unknown:
        raise SystemExit(
            f"--labels names run(s) that were never given with --run: {unknown}")
    return {name: per_run.get(name, default) for name in run_names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, type=parse_run,
                    help="NAME=PATH to validation/3_agent; repeat for each model")
    ap.add_argument("--out-dir", required=True)
    # C1: the five hardcoded stage-C filenames, as parameters.
    ap.add_argument("--stage-dir", default=DEFAULT_STAGE_DIR,
                    help="Stage-C directory inside each run (default: stage_c_full)")
    ap.add_argument("--labels", action="append", type=parse_label_spec,
                    help="comma-separated label file names inside --stage-dir; "
                         "give one name for a single-pass arm. Prefix with "
                         "RUN= to override a single run, so arms judged in a "
                         "different number of passes can still be joined in one "
                         "invocation and share one pooled coverage sheet.")
    ap.add_argument("--rehearsal", action="store_true",
                    help="permit rehearsal STUB label files. Without it a label "
                         "file of machine-generated records is refused.")
    ap.add_argument("--audit-seed", type=int, default=DEFAULT_AUDIT_SEED,
                    help="seed for the blinded coverage sheet shuffle (C9); "
                         "published with the sheet")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "axis3_full_system_audit.csv"
    audit = load_audit(audit_path)

    labels_for = resolve_label_files(args.labels, [n for n, _ in args.run])

    summaries, all_records, all_audit_rows = [], [], []
    for name, path in args.run:
        summary, records, audit_rows = analyse_run(
            name, path, audit, args.stage_dir, labels_for[name], args.rehearsal)
        summaries.append(summary)
        all_records.extend(records)
        all_audit_rows.extend(audit_rows)

    required_audit_rows = [
        row for row in all_audit_rows
        if (row["human_stage_c_required"] == "YES"
            or row["human_critical_coverage_required"] == "YES")
    ]
    coverage_rows = [row for row in all_audit_rows
                     if row["human_critical_coverage_required"] == "YES"]
    write_audit(audit_path, required_audit_rows)

    sheet_path = out_dir / "axis3_full_system_coverage_sheet.csv"
    key_path = out_dir / "axis3_full_system_coverage_key.csv"
    write_coverage_sheet(sheet_path, key_path, coverage_rows, args.audit_seed)

    write_records(out_dir / "axis3_full_system_records.csv", all_records)
    write_report(out_dir / "axis3_full_system_report.md", summaries)
    (out_dir / "axis3_full_system_metrics.json").write_text(
        json.dumps({"runs": summaries,
                    "coverage_sheet_seed": args.audit_seed,
                    "stage_dir": args.stage_dir,
                    "label_files": {k: list(v) for k, v in labels_for.items()},
                    "rehearsal": bool(args.rehearsal)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    (out_dir / "axis3_full_system_audit_guide.md").write_text(
        "# Axis 3 human audit\n\n"
        "There are two instruments and they are deliberately separate.\n\n"
        "## 1. Stage-C adjudication\n\n"
        f"`axis3_full_system_audit.csv` contains the {len(required_audit_rows)} "
        "records that need a human decision.\n\n"
        "- Fill `human_stage_c` with `PASS` or `FAIL` only when "
        "`human_stage_c_required=YES`.\n"
        "- `PASS` means the reply contains no contradicted or unsupported claim.\n"
        "- `judged_surface` is the text that was judged. The raw model envelope "
        "is deliberately not in this file.\n"
        "- Do NOT fill `human_critical_coverage` here. That column is the "
        "decision STORE, written by the join step below. Filling it by hand on "
        "a sheet that names the arm defeats the blinding of 2.11.\n"
        "- Use `audit_comment` only when the reason is not obvious.\n\n"
        "## 2. Critical-event coverage, blinded\n\n"
        f"`axis3_full_system_coverage_sheet.csv` contains the {len(coverage_rows)} "
        "coverage decisions in one shuffled sheet under opaque `decision_id` "
        f"values, seeded with {args.audit_seed}.\n\n"
        "- Fill `human_critical_coverage` with `PASS` or `FAIL` on every row.\n"
        f"- {COVERAGE_DECISION_RULE}\n"
        "- `literal_hours_found` is an aid only. Read `judged_surface` before "
        "deciding.\n"
        "- Do NOT open `axis3_full_system_coverage_key.csv` until every decision "
        "is recorded and saved: it names the arm and the run behind each "
        "`decision_id`, which is exactly what the sheet exists to withhold.\n"
        "- The join back into the decision store is done by "
        "`apply_axis3_human_audit_structured.py`, whose `--dry-run` fails if any "
        "`decision_id` does not resolve.\n\n"
        "Save the files in place and rerun the same command. Existing decisions "
        "are preserved and the final pass^1/pass^5 values are recalculated.\n",
        encoding="utf-8")

    print((out_dir / "axis3_full_system_report.md").read_text(encoding="utf-8"))
    print(f"Audit: {audit_path}")
    print(f"Coverage sheet: {sheet_path} (seed {args.audit_seed})")
    print(f"Coverage key, do not open before the decisions are recorded: {key_path}")


if __name__ == "__main__":
    main()

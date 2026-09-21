"""Arm comparison for the Axis-3 structured-output experiment (4.4 item 7).

Joins two or more `axis3_full_system_records.csv` files on `(model, case_id, rep)`
and writes one markdown report holding five blocks:

1. the paired baseline-versus-structured table on the five COMMON criteria only
   (C1a, C1b, C2, C3, C4) plus the C5-common composite, pass^1 and pass^5 per
   model, every denominator and every excluded count printed in the cell;
2. the separate-rates table (json schema validity, contract validity, error and
   abstention rates), intervention arm only, printed clearly outside 1;
3. the critical-transition decomposition of 3.4.3, intervention arm only,
   diagnostic, with the mandatory non-pooling sentence verbatim beneath it;
4. the per-case discordance table and the per-criterion flip attribution of 3.1;
5. the pre-declared headline variants P, V1, V2, and V5 when the tier columns
   are present.

WHY THIS FILE IS DEFENSIVE TO THE POINT OF RUDENESS
---------------------------------------------------
This is the last station before a published number, and it is the one place
where the whole design could be undone quietly: one extra column in a table and
the free-text baseline is being scored on a JSON array it was never asked for.
So the separations that the pre-registration states in prose are enforced here
in code, and a violation raises instead of printing:

* `COMMON_CRITERIA` is an explicit allowlist. The comparison table's columns are
  asserted to be exactly that tuple, in that order. Every candidate source
  column is screened at import time, and every header, row label and rendered
  cell of the common table is screened again before the report is written, so no
  `ct_*` field and no schema term can reach it by any route.
* pass^5 and every paired quantity are computed at CASE level, 13 pairs per
  model. `--paired-unit rep` is accepted by the parser only so that it can be
  refused with the reason attached; no McNemar over the 65 rows exists in this
  file, and adding one would have to defeat `refuse_rep_pairing` first.
* the historical 2026-09-01 v1 figures are a captioned column and nothing else.
  `difference()` asserts that neither side of a subtraction is historical, so
  they cannot be differenced against a new arm even by accident.
* every cell carries `passed/denominator` and the count of rows excluded.

THE C1a CORRECTION
------------------
The published C1a cell is `numeric_ok_full_surface`, the 2.1-conformant verdict
recomputed by `stage_c_structured.py` over prose plus the rendered slot claims,
never the run-time `numeric_ok`, which on a structured row is the PROVISIONAL
prose-only lower bound. This module prefers the full-surface column and, when it
has to fall back, says so in the report in the cell's own provenance block
rather than letting a lower bound be read as the figure.

Pure standard library on purpose: this report must be reproducible on a machine
with nothing installed, and a `pandas` group-by would buy nothing at 130 rows.

Run:
    python scripts/validation/axis3_arm_compare.py \
        --records RUN_A/axis3_full_system_records.csv \
        --records RUN_B/axis3_full_system_records.csv \
        --pair luna_free:luna_struct --pair terra_free:terra_struct \
        --out axis3_arm_comparison.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The 3.4.3 non-pooling sentence is imported, never retyped: it must appear
# verbatim beneath the decomposition table, and a second copy in this file would
# be free to drift from the one the validator prints. Both spellings for the
# same reason `stage_c_structured` needs both, package import and plain script.
try:
    from .stage_c_structured import _DECOMPOSITION_CAVEAT as DECOMPOSITION_CAVEAT
except ImportError:  # run as a script: no parent package, but the dir is on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stage_c_structured import _DECOMPOSITION_CAVEAT as DECOMPOSITION_CAVEAT


# --------------------------------------------------------------------------------
# The allowlist, and the screen that keeps everything else out of it
# --------------------------------------------------------------------------------

#: The ONLY columns the common comparison table may ever have, in this order.
#: Five criteria of 3.1 plus the composite. SSP, schema validity, contract
#: validity and every ct_* quantity are intervention-only and live in their own
#: tables further down the report.
COMMON_CRITERIA = ("C1a", "C1b", "C2", "C3", "C4", "C5-common")

CRITERION_TITLES = {
    "C1a": "numeric traceability",
    "C1b": "factual accuracy",
    "C2": "narrative critical-event coverage",
    "C3": "interpretation faithfulness",
    "C4": "policy compliance",
    "C5-common": "arm-neutral common Stage-C pass",
}

#: Candidate source columns per criterion, most specific first. A records CSV
#: that carries a purpose-built per-criterion column wins; the general columns
#: below it are the fallbacks the current `axis3_full_system.py` actually writes.
#: C1a's order is the correction the author ordered: the full-surface verdict
#: first, the provisional prose-only one only as a last resort and never
#: silently.
COMMON_CRITERION_SOURCES = {
    # `numeric_ok_used` is what `axis3_full_system.numeric_verdict` publishes:
    # the full-surface verdict where it exists and the run-time one only on the
    # free arm, where 2.1 makes the two identical. It therefore outranks both of
    # the raw columns, and `numeric_source` says per row which one it was.
    "C1a": ("c1a_numeric_ok", "numeric_ok_used", "numeric_ok_full_surface",
            "numeric_ok"),
    "C1b": ("c1b_no_contradicted", "stage_c_no_contradicted"),
    "C2": ("c2_critical_coverage", "critical_coverage"),
    "C3": ("c3_no_not_in_result", "stage_c_no_not_in_result"),
    "C4": ("c4_policy_ok", "policy_ok"),
    "C5-common": ("c5_common", "full_system"),
}

#: C4 is `policy_problems` empty. When the records CSV carries the problem list
#: rather than a verdict, it is turned into `policy_ok` at load time: an empty
#: cell is a pass, anything else is a failure.
POLICY_PROBLEMS_COLUMN = "policy_problems"

#: C1b and C3 are two labels of one judge decision, and `axis3_full_system.py`
#: stores their conjunction in `stage_c`. When neither is separable this column
#: supplies both, and the report says in full what that costs.
JOINT_JUDGE_COLUMN = "stage_c"
JOINT_JUDGE_NOTE = (
    "read from the joint `stage_c` column, which is (no CONTRADICTED AND no "
    "NOT_IN_RESULT). C1b and C3 are NOT separable in this records CSV, so the "
    "two cells are identical by construction and must not be read as two "
    "independent measurements"
)

#: Columns used only for the flip attribution of 3.1 and the variant
#: denominators. They are never comparison-table columns, and the allowlist
#: assertion below is what keeps that true rather than this comment.
ATTRIBUTION_SOURCES = {
    "Stage A": ("stage_a_ok",),
    "Stage B": ("stage_b_ok",),
}
SURFACE_SOURCES = ("judgeable_surface", "has_judgeable_surface", "judged_surface_chars")
#: agent_eval_prep_labeling.py:25. Applied only when the surface column is the
#: character count rather than a boolean.
NARRATION_MIN_CHARS = 120

#: Tier columns for V5, the prose-tier-only reading. Absent means V5 is not
#: computed and is reported as not computed, never as zero.
V5_SOURCES = {
    "C1b": ("c1b_prose_tier", "stage_c_prose_tier_no_contradicted"),
    "C3": ("c3_prose_tier", "stage_c_prose_tier_no_not_in_result"),
    "C5-common": ("c5_common_prose_tier", "full_system_prose_tier"),
}

#: Anything whose name starts with one of these, or contains one of these, is a
#: schema term or a contract diagnostic and is barred from the common table.
FORBIDDEN_COMMON_PREFIXES = ("ct_",)
FORBIDDEN_COMMON_TERMS = (
    "ct_", "schema", "contract", "slot", "ssp", "structured_complete",
    "validator_invalid", "exact_set", "json", "envelope", "status_consistent",
)


class CommonComparisonContaminated(RuntimeError):
    """Raised when a schema term or a ct_* field reaches the common table."""


class PairingRefused(RuntimeError):
    """Raised when a rep-index pairing or a 65-row paired statistic is asked for."""


REP_PAIRING_REFUSAL = (
    "Rep-index pairing is refused. Temperature is 0.0 yet zero of the 13 "
    "in-scope cases has 5 byte-identical replies in any stored run, so "
    "`ctrl A01#3` and `struct A01#3` are two unrelated draws and pairing them "
    "is meaningless; a 65-pair test would also ignore the intra-case design "
    "effect (ICC 0.567 luna, 0.256 terra) and overstate interval precision by "
    "1.4x to 1.8x in width. The only admissible paired unit is the CASE: 13 "
    "pairs per model, 26 pooled over the two models. Recompute over cases."
)

#: 3.4.2, printed above the decomposition table so the separation travels with
#: the numbers rather than living only in the pre-registration.
EXACT_SET_SEPARATION_SENTENCE = (
    "Exact-set accuracy scores a JSON array that only one arm is able to emit "
    "at all, so putting it into the common Stage-C comparison would score the "
    "free-text baseline zero on a field it was never asked for and could not "
    "have produced; it therefore lives in the complete structured-system pass "
    "(SSP), which is intervention-only, and never in C5-common, which is the "
    "arm-to-arm comparison."
)

#: The as-published rubric-v1 Axis-3 figures (pass^1, pass^5), quoted and never
#: differenced. Keyed by a lowercase token matched against the `model` column,
#: because the model strings are provider aliases and the token is the stable
#: part. Order of the published triple: luna 75.4/46.2, terra 72.3/30.8,
#: gemini 15.4/0.0.
HISTORICAL_V1_PUBLISHED = {
    "luna": (0.754, 0.462),
    "terra": (0.723, 0.308),
    "gemini": (0.154, 0.000),
}
HISTORICAL_V1_CAPTION = (
    "HISTORICAL, rubric v1, 2026-09-01, three models, as published. Scored by a "
    "two-pass Opus intersection that NEITHER arm of this comparison receives. "
    "Quoted for orientation only. It is never differenced against a new arm, "
    "with or without a caveat, and this module raises if a difference is asked "
    "for against it."
)

#: Contract-outcome taxonomy of 2.4, used by the intervention-only rates table.
_NOT_A_MODEL_OUTPUT = ("runtime_error", "round_ceiling", "contract_not_attached")
_SCHEMA_OK_OUTCOMES = ("validator_invalid", "valid")


def _screen_name(name, where):
    """Raise if `name` is a schema term or a ct_* field. Used on candidate source
    columns at import time, so a future edit that adds a forbidden column to the
    comparison cannot even be imported, and on the rendered table before it is
    written."""
    low = str(name).lower()
    if any(low.startswith(p) for p in FORBIDDEN_COMMON_PREFIXES) or any(
            t in low for t in FORBIDDEN_COMMON_TERMS):
        raise CommonComparisonContaminated(
            f"{where}: '{name}' is a schema term or a ct_* field and may never "
            f"enter the common comparison (2.4, 3.4.2). Allowlist is "
            f"{list(COMMON_CRITERIA)}.")


def _screen_sources():
    """Import-time self-check on the candidate lists themselves."""
    for criterion, candidates in COMMON_CRITERION_SOURCES.items():
        if criterion not in COMMON_CRITERIA:
            raise CommonComparisonContaminated(
                f"'{criterion}' has source columns but is not in the allowlist")
        for column in candidates:
            _screen_name(column, f"COMMON_CRITERION_SOURCES[{criterion!r}]")
    _screen_name(JOINT_JUDGE_COLUMN, "JOINT_JUDGE_COLUMN")
    missing = [c for c in COMMON_CRITERIA if c not in COMMON_CRITERION_SOURCES]
    if missing:
        raise CommonComparisonContaminated(
            f"allowlist entries with no source column: {missing}")


_screen_sources()


# --------------------------------------------------------------------------------
# Reading the records CSV
# --------------------------------------------------------------------------------

_TRUE = {"true", "1", "yes", "pass", "t", "y"}
_FALSE = {"false", "0", "no", "fail", "f", "n"}


def parse_bool(value):
    """CSV text to True / False / None.

    None means "no decision in this cell": a human decision still pending, or a
    row the reporting rule of 2.4 drops from a per-criterion denominator. It is
    counted and printed as an exclusion, never silently read as a failure.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def parse_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------
# Chain of custody: a stamp is worthless unless the next tool reads it
# --------------------------------------------------------------------------------
# Upstream tools stamp rehearsal artefacts, but stamping only protects the file it
# is written on. Without this check an operator can feed a rehearsal records CSV,
# whose every label is fabricated, into this tool and get back a clean, confident,
# entirely unstamped comparison report. The verifier did exactly that. So the stamp
# is now READ: any input carrying it refuses to be compared unless --rehearsal is
# typed, and when it is typed the stamp propagates to the output rather than being
# consumed. Detection is deliberately broad, over column names AND cell values,
# because the upstream stamp has taken more than one shape.
REHEARSAL_MARKERS = ("REHEARSAL", "REHEARSAL_STUB")


def rehearsal_evidence(path, rows, header):
    """Every reason to believe this input came from a rehearsal, as strings."""
    found = []
    for name in header or []:
        upper = str(name).upper()
        if any(marker in upper for marker in REHEARSAL_MARKERS):
            found.append(f"column {name!r}")
    for row in rows or []:
        for key, value in row.items():
            if isinstance(value, str) and "REHEARSAL" in value.upper():
                found.append(f"cell {key!r} = {value[:60]!r}")
                break
        if found and len(found) > 3:
            break
    try:
        if "REHEARSAL" in Path(path).name.upper():
            found.append("the file name")
    except Exception:
        pass
    return found


def read_records(path):
    """Return (rows, header). `utf-8-sig` because `write_records` writes a BOM."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            row["__source_file"] = str(path)
            rows.append(row)
    return rows, header


def load_all(paths):
    """Read every --records file into one pooled row list plus the union header.

    The union, not the first file's header: two arms can be written by two runs
    of `axis3_full_system.py` into two files, and a criterion present in only one
    of them must still be visible as present, so that the report can say which
    arm it is missing from instead of pretending the column does not exist.
    """
    rows, header, per_file, stamped = [], [], [], []
    for path in paths:
        file_rows, file_header = read_records(path)
        # Read the stamp, do not merely be capable of writing one.
        evidence = rehearsal_evidence(path, file_rows, file_header)
        if evidence:
            stamped.append((str(path), evidence))
        rows.extend(file_rows)
        for name in file_header:
            if name not in header:
                header.append(name)
        per_file.append({
            "path": str(path),
            "sha256": sha256_of(path),
            "rows": len(file_rows),
            "columns": len(file_header),
            "rehearsal_evidence": [e for _p, ev in stamped
                                   if _p == str(path) for e in ev],
        })
    return rows, header, per_file, stamped


# --------------------------------------------------------------------------------
# Criterion resolution
# --------------------------------------------------------------------------------

class Source:
    """Where one criterion's verdict is read from, and what that costs."""

    def __init__(self, criterion, column, note="", provisional=False):
        self.criterion = criterion
        self.column = column
        self.note = note
        self.provisional = provisional

    @property
    def available(self):
        return self.column is not None


def resolve_sources(header):
    """Map each allowlisted criterion onto a column of this records CSV.

    Returns a dict criterion -> Source. A criterion with no usable column gets a
    Source with `column = None`; its cells print `n/a` with the reason rather
    than disappearing from the table, because a missing criterion is itself a
    finding about the records file and must be visible.
    """
    resolved = {}
    for criterion in COMMON_CRITERIA:
        chosen, note, provisional = None, "", False
        for candidate in COMMON_CRITERION_SOURCES[criterion]:
            if candidate in header:
                chosen = candidate
                break
        if chosen is None and criterion in ("C1b", "C3") and JOINT_JUDGE_COLUMN in header:
            chosen, note = JOINT_JUDGE_COLUMN, JOINT_JUDGE_NOTE
        if chosen == "numeric_ok":
            # The correction the author ordered. A prose-only lower bound is not
            # the published C1a figure, so if it is all the file offers the
            # report says so in the cell's provenance, loudly.
            provisional = True
            note = ("PROVISIONAL prose-only lower bound. `numeric_ok_full_surface` "
                    "is absent from this records CSV, so C1a is NOT the "
                    "2.1-conformant full-surface verdict the pre-registration "
                    "publishes. Re-run the scoring pass before quoting this cell.")
        if chosen is None:
            note = (f"no source column in this records CSV; candidates were "
                    f"{list(COMMON_CRITERION_SOURCES[criterion])}")
        if chosen is not None:
            _screen_name(chosen, f"resolved source for {criterion}")
        resolved[criterion] = Source(criterion, chosen, note, provisional)
    return resolved


def materialise_c1a(rows, header):
    """Fill an empty `numeric_ok_full_surface` cell from `numeric_ok`, per row.

    Why this is legitimate and narrow. 2.1 makes the two verdicts identical on
    the free baseline by construction, because a free arm has no slot claims to
    append, so the full-surface column is simply not written on those rows when
    the two arms are scored by different passes. Without this fill the union
    header would resolve C1a to the full-surface column and then read an empty
    cell on every baseline row, emptying the baseline denominator.

    It is narrow because a fill on a STRUCTURED row means something else
    entirely: that `stage_c_structured.py` has not been run over that file, so
    C1a there is the provisional prose-only lower bound. Those rows are counted
    separately and the report raises a warning on them rather than absorbing
    them. Returns (filled_by_run, structured_filled_by_run).
    """
    filled = Counter()
    structured_filled = Counter()
    if ("numeric_ok_full_surface" not in header or "numeric_ok" not in header
            or "c1a_numeric_ok" in header or "numeric_ok_used" in header):
        # `numeric_ok_used` already IS the resolved verdict, with the same
        # free-arm fallback applied upstream and recorded in `numeric_source`.
        # Filling on top of it would hide that record behind a second one.
        return filled, structured_filled
    for row in rows:
        if str(row.get("numeric_ok_full_surface") or "").strip():
            continue
        if not str(row.get("numeric_ok") or "").strip():
            continue
        row["numeric_ok_full_surface"] = row["numeric_ok"]
        run_id = (row.get("run_id") or "").strip()
        filled[run_id] += 1
        if (row.get("arm") or "").strip() == "structured":
            structured_filled[run_id] += 1
    return filled, structured_filled


def materialise_policy_ok(rows, header):
    """Derive `policy_ok` from `policy_problems` when only the list is stored.

    C4's pass condition is "policy_problems empty", so an empty cell is a PASS
    and any content is a FAIL. `parse_bool` cannot express that, because an
    empty cell everywhere else in this file means "no decision", so the reading
    is done once here and the rest of the module stays column-based.
    """
    if "policy_ok" in header or POLICY_PROBLEMS_COLUMN not in header:
        return 0
    for row in rows:
        raw = str(row.get(POLICY_PROBLEMS_COLUMN) or "").strip()
        row["policy_ok"] = str(raw in ("", "[]"))
    header.append("policy_ok")
    return len(rows)


def numeric_source_summary(rows, header):
    """Count, per run, which numeric verdict each C1a cell came from.

    The author's correction made auditable at the comparison layer too: a
    `prose_only_fallback` on a STRUCTURED row means the published C1a there is
    the provisional lower bound and must not be quoted.
    """
    if "numeric_source" not in header:
        return {}
    counts = defaultdict(Counter)
    for row in rows:
        counts[(row.get("run_id") or "").strip()][
            (row.get("numeric_source") or "unrecorded").strip()] += 1
    return dict(counts)


def resolve_optional(header, candidates):
    for candidate in candidates:
        if candidate in header:
            return candidate
    return None


def has_surface(row, column):
    """True / False / None for 'this row has a judgeable surface' (2.4)."""
    if column is None:
        return None
    if column == "judged_surface_chars":
        chars = parse_int(row.get(column))
        return None if chars is None else chars >= NARRATION_MIN_CHARS
    return parse_bool(row.get(column))


# --------------------------------------------------------------------------------
# Cells: a value is never printed without its denominator
# --------------------------------------------------------------------------------

class Cell:
    """passed / denominator, plus the count excluded and the count pending.

    There is deliberately no constructor that takes a bare rate. Every number
    this module prints is built from counts, so the denominator cannot be lost
    between the computation and the page.
    """

    def __init__(self, passed, denominator, excluded=0, pending=0, unit="rows",
                 unavailable=""):
        self.passed = passed
        self.denominator = denominator
        self.excluded = excluded
        self.pending = pending
        self.unit = unit
        self.unavailable = unavailable
        self.historical = False

    @property
    def rate(self):
        if self.unavailable or not self.denominator:
            return None
        return self.passed / self.denominator

    def text(self):
        if self.unavailable:
            return f"n/a ({self.unavailable})"
        if not self.denominator:
            return f"n/a (0 {self.unit}, excl {self.excluded})"
        body = f"{self.passed}/{self.denominator} = {self.rate:.1%}"
        body += f" (excl {self.excluded}"
        if self.pending:
            body += f", pending {self.pending}"
        return body + f" {self.unit})"


def historical_cell(rate):
    cell = Cell(0, 0, unavailable=f"{rate:.1%} v1, not differenced")
    cell.historical = True
    return cell


def difference(structured, baseline):
    """Structured minus baseline, in percentage points.

    Asserts that neither side is a historical v1 figure. The rule that the
    2026-09-01 as-published numbers are never differenced is enforced here and
    not in a caption, because a caption cannot stop a subtraction.
    """
    if structured.historical or baseline.historical:
        raise CommonComparisonContaminated(
            "the historical rubric-v1 2026-09-01 figures are quoted only and are "
            "never differenced against an arm of this comparison (3.1, B2)")
    if structured.rate is None or baseline.rate is None:
        return None
    return (structured.rate - baseline.rate) * 100.0


def fmt_diff(value):
    if value is None:
        return "n/a"
    return f"{value:+.1f} pp"


# --------------------------------------------------------------------------------
# pass^1 and pass^5. pass^5 is case level. There is no other kind here.
# --------------------------------------------------------------------------------

def pass1(rows, column):
    """Row-level rate over the rows whose cell carries a decision."""
    if column is None:
        return Cell(0, 0, unavailable="no source column")
    values = [parse_bool(r.get(column)) for r in rows]
    decided = [v for v in values if v is not None]
    return Cell(sum(1 for v in decided if v), len(decided),
                excluded=len(values) - len(decided), unit="rows")


def case_states(rows, column):
    """Case id -> case-level verdict: False if any rep fails, True if all pass,
    None otherwise. This is the pass^5 rule of `passk_bounds`, restated here
    because this module must not import the reporting layer of the join."""
    by_case = defaultdict(list)
    for row in rows:
        by_case[row.get("case_id", "")].append(parse_bool(row.get(column)))
    states = {}
    for case_id, values in by_case.items():
        if any(v is False for v in values):
            states[case_id] = False
        elif values and all(v is True for v in values):
            states[case_id] = True
        else:
            states[case_id] = None
    return states


def pass5(rows, column):
    """pass^5 at CASE level, 13 cases per model. There is no rep-level variant."""
    if column is None:
        return Cell(0, 0, unavailable="no source column")
    states = case_states(rows, column)
    decided = [v for v in states.values() if v is not None]
    return Cell(sum(1 for v in decided if v), len(decided),
                excluded=len(states) - len(decided), unit="cases")


def refuse_rep_pairing(unit):
    """The refusal, callable. `--paired-unit rep` exists so that asking is
    answered with the reason rather than with a number."""
    if unit != "case":
        raise PairingRefused(REP_PAIRING_REFUSAL)
    return "case"


def paired_case_discordance(baseline_rows, structured_rows, column, unit="case"):
    """Discordant CASES, b and c, on the only admissible paired unit.

    Returns (b, c, concordant, undecided, n_pairs) where b is baseline PASS with
    structured FAIL and c is the reverse. No test statistic and no p-value is
    computed: B6 measures an intra-case design effect that a 2x2 test would
    ignore, and the pre-registration asks for the discordance counts, not for a
    McNemar over the 65 rows.
    """
    refuse_rep_pairing(unit)
    base = case_states(baseline_rows, column)
    struct = case_states(structured_rows, column)
    common = sorted(set(base) & set(struct))
    b = c = concordant = undecided = 0
    for case_id in common:
        x, y = base[case_id], struct[case_id]
        if x is None or y is None:
            undecided += 1
        elif x == y:
            concordant += 1
        elif x and not y:
            b += 1
        else:
            c += 1
    return b, c, concordant, undecided, len(common)


# --------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------

class Table:
    def __init__(self, title, columns, caption=(), footer=()):
        self.title = title
        self.columns = tuple(columns)
        self.rows = []       # list of (label, {column: text})
        self.caption = list(caption)
        self.footer = list(footer)

    def add(self, label, cells):
        self.rows.append((label, cells))

    def render(self):
        out = [f"### {self.title}", ""]
        out += [line for line in self.caption] + ([""] if self.caption else [])
        out.append("| | " + " | ".join(self.columns) + " |")
        out.append("|---|" + "---|" * len(self.columns))
        for label, cells in self.rows:
            out.append("| " + label + " | "
                       + " | ".join(str(cells.get(c, "")) for c in self.columns) + " |")
        if self.footer:
            out.append("")
            out += list(self.footer)
        out.append("")
        return out


def assert_common_table_clean(table):
    """The structural guarantee. Columns are exactly the allowlist, in order, and
    no header, row label or cell text carries a schema term or a ct_* field."""
    if table.columns != COMMON_CRITERIA:
        raise CommonComparisonContaminated(
            f"the common comparison table's columns must be exactly "
            f"{list(COMMON_CRITERIA)}, got {list(table.columns)}")
    for column in table.columns:
        _screen_name(column, "common table header")
    for label, cells in table.rows:
        _screen_name(label, "common table row label")
        extra = set(cells) - set(COMMON_CRITERIA)
        if extra:
            raise CommonComparisonContaminated(
                f"row '{label}' carries non-allowlisted cells {sorted(extra)}")
        for column, text in cells.items():
            _screen_name(text, f"common table cell [{label}][{column}]")
    return True


def build_comparison_table(pairs_data, sources, with_historical=False):
    """Block 1. Columns are the allowlist and nothing else, ever."""
    caption = [
        "Common criteria only. Five criteria of 3.1 plus the composite. This "
        "table contains no schema term of any kind: schema validity, contract "
        "validity, SSP and every critical-transition quantity are "
        "intervention-only and are printed in their own tables below.",
        "",
        "Every cell is `passed/denominator = rate (excl N unit)`. `excl` counts "
        "rows or cases with no decision in that cell, which under the reporting "
        "rule of 2.4 are dropped from the per-criterion denominator and are "
        "never counted as passes.",
        "",
        "pass^5 is computed at CASE level, 13 cases per model. No rep-index "
        "pairing is performed anywhere in this file.",
    ]
    table = Table("1. Paired baseline versus structured, common criteria",
                  COMMON_CRITERIA, caption)
    for pair in pairs_data:
        for metric, fn in (("pass^1", pass1), ("pass^5", pass5)):
            base_cells, struct_cells = {}, {}
            diffs = {}
            for criterion in COMMON_CRITERIA:
                column = sources[criterion].column
                base = fn(pair["baseline_rows"], column)
                struct = fn(pair["structured_rows"], column)
                base_cells[criterion] = base.text()
                struct_cells[criterion] = struct.text()
                diffs[criterion] = fmt_diff(difference(struct, base))
            table.add(f"{pair['model']} baseline {metric}", base_cells)
            table.add(f"{pair['model']} structured {metric}", struct_cells)
            table.add(f"{pair['model']} difference {metric}", diffs)
    if with_historical:
        # Quoted in its own rows and flagged, and `difference()` refuses to
        # subtract it, so a historical figure can be read but not used.
        for token, (p1, p5) in HISTORICAL_V1_PUBLISHED.items():
            for metric, rate in (("pass^1", p1), ("pass^5", p5)):
                cells = {c: "" for c in COMMON_CRITERIA}
                cells["C5-common"] = historical_cell(rate).text()
                table.add(f"{token} HISTORICAL v1 {metric}", cells)
        table.footer.append(HISTORICAL_V1_CAPTION)
    assert_common_table_clean(table)
    return table


def rates_table(pairs_data, header):
    """Block 2. Intervention arm only, printed outside the comparison."""
    finish_col = resolve_optional(header, ("contract_outcome_finish_reason",
                                           "finish_reason"))
    status_col = resolve_optional(header, ("contract_status", "structured_status",
                                           "status"))
    columns = ("json_schema_validity_rate", "contract_validity_rate",
               "runtime_error_rate", "abstention_rate", "excluded from both rates")
    caption = [
        "INTERVENTION ARM ONLY. These are not comparison quantities and they "
        "have no baseline column: a free arm has no contract to violate. They "
        "are printed here, outside table 1, for exactly that reason.",
        "",
        "Denominator of both validity rates, 3.3, word for word the same for "
        "the two: structured rows minus `runtime_error`, `round_ceiling`, "
        "`contract_not_attached`, and `no_final_message` with "
        "`finish_reason == content_filter`. The excluded count is printed.",
    ]
    if finish_col is None:
        caption += [
            "",
            "WARNING: no finish-reason column in this records CSV, so the "
            "content-filter split of `no_final_message` could not be applied. "
            "Every `no_final_message` row is treated as a model refusal and "
            "stays in the denominator, which is the conservative reading.",
        ]
    if status_col is None:
        caption += [
            "",
            "The abstention rate is not computable from this records CSV: no "
            "status column. It is reported as not computed, never as zero.",
        ]
    table = Table("2. Intervention-only rates, outside the comparison", columns, caption)
    if "contract_outcome" not in header:
        # Without the taxonomy column there is no numerator and no denominator.
        # Printing 0/65 here would read as a measured total failure, which is the
        # one thing a not-applicable cell must never do.
        for pair in pairs_data:
            table.add(pair["model"], {
                column: "n/a (no `contract_outcome` column in this records CSV; "
                        "run the scoring pass first)"
                for column in columns})
        return table
    for pair in pairs_data:
        rows = pair["structured_rows"]
        outcomes = Counter((r.get("contract_outcome") or "").strip() for r in rows)
        excluded = [
            r for r in rows
            if (r.get("contract_outcome") or "").strip() in _NOT_A_MODEL_OUTPUT
            or ((r.get("contract_outcome") or "").strip() == "no_final_message"
                and finish_col is not None
                and (r.get(finish_col) or "").strip() == "content_filter")
        ]
        excluded_ids = {id(r) for r in excluded}
        denom = [r for r in rows if id(r) not in excluded_ids]
        schema_ok = sum(1 for r in denom
                        if (r.get("contract_outcome") or "").strip() in _SCHEMA_OK_OUTCOMES)
        contract_ok = sum(1 for r in denom
                          if (r.get("contract_outcome") or "").strip() == "valid")
        n_error = outcomes.get("runtime_error", 0)
        if status_col is None:
            abstain = Cell(0, 0, unavailable="no status column")
        else:
            abstain = Cell(sum(1 for r in rows
                               if (r.get(status_col) or "").strip() == "abstained"),
                           len(rows), unit="rows")
        detail = (f"{len(excluded)} of {len(rows)}: "
                  + ", ".join(f"{name} {outcomes.get(name, 0)}"
                              for name in _NOT_A_MODEL_OUTPUT)
                  + f", no_final_message {outcomes.get('no_final_message', 0)}")
        table.add(pair["model"], {
            "json_schema_validity_rate": Cell(schema_ok, len(denom),
                                              excluded=len(excluded)).text(),
            "contract_validity_rate": Cell(contract_ok, len(denom),
                                           excluded=len(excluded)).text(),
            "runtime_error_rate": Cell(n_error, len(rows)).text(),
            "abstention_rate": abstain.text(),
            "excluded from both rates": detail,
        })
    return table


def _ints(text):
    out = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            value = parse_int(part)
            if value is not None:
                out.append(value)
    return out


def decomposition_table(pairs_data, header):
    """Block 3, the 3.4.3 table. Intervention arm only, diagnostic, and the
    non-pooling sentence is printed verbatim beneath it."""
    needed = ("ct_scored", "ct_n_gt", "ct_n_emitted", "ct_matched", "ct_exact_set")
    missing = [name for name in needed if name not in header]
    caption = [
        "INTERVENTION ARM ONLY. Diagnostic. No row anywhere is passed or failed "
        "on any line of this table, and none of it is a column of table 1.",
        "",
        EXACT_SET_SEPARATION_SENTENCE,
    ]
    columns = ("value", "denominator")
    table = Table("3. Critical-transition decomposition, intervention arm only, diagnostic",
                  columns, caption)
    if missing:
        table.add("not computed", {
            "value": "n/a",
            "denominator": f"records CSV lacks {missing}; run the scoring pass first",
        })
        table.footer.append(DECOMPOSITION_CAVEAT)
        return table

    for pair in pairs_data:
        rows = pair["structured_rows"]
        scored = [r for r in rows if parse_bool(r.get("ct_scored"))]
        not_scored = Counter((r.get("ct_not_scored_reason") or "unrecorded")
                             for r in rows if not parse_bool(r.get("ct_scored")))
        sum_gt = sum(parse_int(r.get("ct_n_gt")) or 0 for r in scored)
        sum_em = sum(parse_int(r.get("ct_n_emitted")) or 0 for r in scored)
        sum_matched = sum(parse_int(r.get("ct_matched")) or 0
                          for r in scored if parse_int(r.get("ct_n_gt")))
        extras = [_ints(r.get("ct_extra_periods")) for r in scored]
        n_extra = sum(len(e) for e in extras)
        dropped_gt0 = sum(1 for r in scored if not parse_int(r.get("ct_n_gt")))
        in_payload = sum(len(_ints(r.get("ct_extra_in_payload"))) for r in scored)
        not_in_payload = sum(len(_ints(r.get("ct_extra_not_in_payload"))) for r in scored)

        prefix = pair["model"]
        excl_detail = (", ".join(f"{k} {v}" for k, v in sorted(not_scored.items()))
                       or "none")
        table.add(f"{prefix} ct_scored records", {
            "value": str(len(scored)),
            "denominator": f"{len(rows)} structured rows, excluded "
                           f"{sum(not_scored.values())} ({excl_detail})",
        })
        table.add(f"{prefix} ct_exact_set (gate term, reference only)", {
            "value": Cell(sum(1 for r in scored if parse_bool(r.get("ct_exact_set"))),
                          len(scored), unit="records").text(),
            "denominator": "ct_scored records",
        })
        table.add(f"{prefix} ct_no_omission_recall, event level", {
            "value": Cell(sum_matched, sum_gt, excluded=dropped_gt0,
                          unit="events").text(),
            "denominator": f"sum n_gt = {sum_gt}; {dropped_gt0} record(s) dropped "
                           f"for n_gt == 0",
        })
        table.add(f"{prefix} ct_no_omission, record level", {
            "value": Cell(sum(1 for r in scored if parse_bool(r.get("ct_no_omission"))),
                          len(scored), unit="records").text(),
            "denominator": "ct_scored records",
        })
        table.add(f"{prefix} ct_extra_rate_emitted", {
            "value": Cell(n_extra, sum_em, unit="emitted entries").text(),
            "denominator": f"the model's own sum n_em = {sum_em}, extras {n_extra}",
        })
        table.add(f"{prefix} ct_extra_per_record", {
            "value": (f"{n_extra / len(scored):.3f}" if scored else "n/a"),
            "denominator": f"{len(scored)} ct_scored records, extras {n_extra}",
        })
        table.add(f"{prefix} ct_over_inclusive, record level", {
            "value": Cell(sum(1 for r in scored if parse_bool(r.get("ct_over_inclusive"))),
                          len(scored), unit="records").text(),
            "denominator": "ct_scored records",
        })
        table.add(f"{prefix} extras in payload / not in payload", {
            "value": f"{in_payload} / {not_in_payload}",
            "denominator": f"{n_extra} extras, never summed into one number",
        })
        table.add(f"{prefix} ct_order_violation / ct_duplicate", {
            "value": f"{sum(1 for r in scored if parse_bool(r.get('ct_order_ok')) is False)}"
                     f" / {sum(1 for r in scored if _ints(r.get('ct_duplicate_periods')))}",
            "denominator": f"{len(scored)} ct_scored records",
        })
    table.footer.append(DECOMPOSITION_CAVEAT)
    return table


def discordance_table(pairs_data, sources, attribution_columns, unit="case"):
    """Block 4. One row per case, plus the per-criterion flip attribution."""
    refuse_rep_pairing(unit)
    composite = sources["C5-common"].column
    columns = ("baseline C5-common", "structured C5-common", "verdict",
               "criteria that flipped")
    caption = [
        "One row per CASE, the only admissible paired unit (3.3, V2). A case is "
        "PASS when all of its repetitions pass, FAIL when any repetition fails, "
        "and undecided otherwise.",
        "",
        "3.1 asks for the flip attribution per row. A row-to-row attribution "
        "would require pairing rep i of one arm with rep i of the other, which "
        "3.3 forbids and this module refuses, so the attribution is computed on "
        "the case. The per-arm failure attribution beneath the table is the "
        "row-level decomposition that needs no pairing at all.",
    ]
    table = Table("4. Per-case discordance and flip attribution", columns, caption)
    for pair in pairs_data:
        base_states = case_states(pair["baseline_rows"], composite)
        struct_states = case_states(pair["structured_rows"], composite)
        common = sorted(set(base_states) & set(struct_states))
        only_base = sorted(set(base_states) - set(struct_states))
        only_struct = sorted(set(struct_states) - set(base_states))

        crit_base, crit_struct = {}, {}
        for criterion in COMMON_CRITERIA:
            column = sources[criterion].column
            crit_base[criterion] = case_states(pair["baseline_rows"], column)
            crit_struct[criterion] = case_states(pair["structured_rows"], column)
        for name, candidates in ATTRIBUTION_SOURCES.items():
            column = resolve_optional(attribution_columns, candidates)
            crit_base[name] = case_states(pair["baseline_rows"], column) if column else {}
            crit_struct[name] = case_states(pair["structured_rows"], column) if column else {}

        for case_id in common:
            x, y = base_states[case_id], struct_states[case_id]
            if x is None or y is None:
                verdict = "undecided"
            elif x == y:
                verdict = "concordant"
            elif x and not y:
                verdict = "regression (baseline PASS, structured FAIL)"
            else:
                verdict = "improvement (baseline FAIL, structured PASS)"
            flipped = []
            if verdict.startswith(("regression", "improvement")):
                for name in list(COMMON_CRITERIA) + list(ATTRIBUTION_SOURCES):
                    if name == "C5-common":
                        continue
                    a = crit_base.get(name, {}).get(case_id)
                    b = crit_struct.get(name, {}).get(case_id)
                    if a is not None and b is not None and a != b:
                        flipped.append(name)
                if not flipped:
                    flipped = ["unattributed (Stage A, Stage B or loss of surface; "
                               "no column in this records CSV)"]
            table.add(f"{pair['model']} {case_id}", {
                "baseline C5-common": _state_text(x),
                "structured C5-common": _state_text(y),
                "verdict": verdict,
                "criteria that flipped": ", ".join(flipped) if flipped else "",
            })
        b, c, concordant, undecided, n_pairs = paired_case_discordance(
            pair["baseline_rows"], pair["structured_rows"], composite, unit)
        table.footer.append(
            f"- {pair['model']}: {n_pairs} paired cases, {concordant} concordant, "
            f"{b} baseline-PASS/structured-FAIL, {c} baseline-FAIL/structured-PASS, "
            f"{undecided} undecided. Cases in the baseline only: "
            f"{only_base or 'none'}. Cases in the structured arm only: "
            f"{only_struct or 'none'}. Discordance counts only: no McNemar and no "
            f"p-value is computed, because B6 measures an intra-case design "
            f"effect that a 2x2 test over these pairs would ignore.")
    return table


def _state_text(value):
    return {True: "PASS", False: "FAIL", None: "undecided"}[value]


def failure_attribution(pairs_data, sources, attribution_columns):
    """Row-level failure attribution per arm, with NO pairing of any kind.

    For each arm, count the rows that fail C5-common and, for each, which
    criteria are False on that same row. Differencing the two counts gives the
    per-criterion decomposition 3.1 asks for without ever pairing a rep of one
    arm to a rep of the other.
    """
    names = [c for c in COMMON_CRITERIA if c != "C5-common"] + list(ATTRIBUTION_SOURCES)
    columns = ("baseline failing rows", "structured failing rows", "difference")
    caption = [
        "Row level, 65 rows per model per arm, NO pairing. Counts rows whose "
        "C5-common is FAIL and, on that same row, which criteria are also FAIL. "
        "A row can appear under more than one criterion, so the column does not "
        "sum to the failure count; the failure count is printed as its own row.",
    ]
    table = Table("4b. Per-criterion failure attribution, unpaired", columns, caption)
    for pair in pairs_data:
        composite = sources["C5-common"].column
        if composite is None:
            table.add(f"{pair['model']} rows failing C5-common", {
                "baseline failing rows": "n/a (no C5-common column)",
                "structured failing rows": "n/a (no C5-common column)",
                "difference": "n/a",
            })
            continue
        base_fail = [r for r in pair["baseline_rows"]
                     if parse_bool(r.get(composite)) is False]
        struct_fail = [r for r in pair["structured_rows"]
                       if parse_bool(r.get(composite)) is False]
        table.add(f"{pair['model']} rows failing C5-common", {
            "baseline failing rows": f"{len(base_fail)} of {len(pair['baseline_rows'])}",
            "structured failing rows": f"{len(struct_fail)} of {len(pair['structured_rows'])}",
            "difference": f"{len(struct_fail) - len(base_fail):+d}",
        })
        for name in names:
            if name in COMMON_CRITERIA:
                column = sources[name].column
            else:
                column = resolve_optional(attribution_columns, ATTRIBUTION_SOURCES[name])
            if column is None:
                table.add(f"{pair['model']} {name}", {
                    "baseline failing rows": "n/a (no column)",
                    "structured failing rows": "n/a (no column)",
                    "difference": "n/a",
                })
                continue
            nb = sum(1 for r in base_fail if parse_bool(r.get(column)) is False)
            ns = sum(1 for r in struct_fail if parse_bool(r.get(column)) is False)
            table.add(f"{pair['model']} {name}", {
                "baseline failing rows": str(nb),
                "structured failing rows": str(ns),
                "difference": f"{ns - nb:+d}",
            })
    return table


# --------------------------------------------------------------------------------
# Block 5, the pre-declared headline variants
# --------------------------------------------------------------------------------

def variants_table(pairs_data, sources, surface_column, v5_sources, unit="case"):
    refuse_rep_pairing(unit)
    composite = sources["C5-common"].column
    columns = ("baseline", "structured", "difference", "denominator rule")
    caption = [
        "The pre-declared variants of the primary comparison endpoint, "
        "C5-common. P is the headline and the only one quoted without "
        "qualification. V1 and V2 are denominator variants; V5 is the only "
        "variant that touches the claim population and is never quoted without "
        "P beside it.",
    ]
    table = Table("5. Headline variants", columns, caption)

    for pair in pairs_data:
        base_rows, struct_rows = pair["baseline_rows"], pair["structured_rows"]

        # P, intention to treat: the denominator is every row of the arm, and a
        # row with no judgeable surface is a FAIL, not an exclusion.
        p_base = _itt_cell(base_rows, composite)
        p_struct = _itt_cell(struct_rows, composite)
        table.add(f"{pair['model']} P pass^1 (headline)", {
            "baseline": p_base.text(),
            "structured": p_struct.text(),
            "difference": fmt_diff(difference(p_struct, p_base)),
            "denominator rule": "every row of the arm; no judgeable surface counts FAIL",
        })

        # V1, per protocol: restricted to rows with a judgeable surface, each arm
        # on its own denominator, both printed.
        if surface_column is None:
            table.add(f"{pair['model']} V1 pass^1", {
                "baseline": "n/a (no surface column)",
                "structured": "n/a (no surface column)",
                "difference": "n/a",
                "denominator rule": "rows with a judgeable surface; column absent "
                                    "from this records CSV",
            })
        else:
            v1_base = _itt_cell([r for r in base_rows
                                 if has_surface(r, surface_column)], composite)
            v1_struct = _itt_cell([r for r in struct_rows
                                   if has_surface(r, surface_column)], composite)
            table.add(f"{pair['model']} V1 pass^1", {
                "baseline": v1_base.text(),
                "structured": v1_struct.text(),
                "difference": fmt_diff(difference(v1_struct, v1_base)),
                "denominator rule": "rows with a judgeable surface, each arm on its "
                                    "own denominator, both printed",
            })

        # V2, paired-common: denominator harmonisation only, on cases.
        v2_base, v2_struct, v2_rule = _v2_cells(base_rows, struct_rows, composite,
                                                surface_column)
        table.add(f"{pair['model']} V2 pass^5, cases", {
            "baseline": v2_base.text(),
            "structured": v2_struct.text(),
            "difference": fmt_diff(difference(v2_struct, v2_base)),
            "denominator rule": v2_rule,
        })

        # V5, prose tier only. Computed only when the tier columns are on disk.
        v5_column = v5_sources.get("C5-common")
        if v5_column is None:
            table.add(f"{pair['model']} V5 pass^1, prose tier", {
                "baseline": "not computed",
                "structured": "not computed",
                "difference": "n/a",
                "denominator rule": "the tier field is not present in this records "
                                    "CSV; reported as not computed, never as zero",
            })
        else:
            v5_base = _itt_cell(base_rows, v5_column)
            v5_struct = _itt_cell(struct_rows, v5_column)
            table.add(f"{pair['model']} V5 pass^1, prose tier", {
                "baseline": v5_base.text(),
                "structured": v5_struct.text(),
                "difference": fmt_diff(difference(v5_struct, v5_base)),
                "denominator rule": f"P's denominator, claim population restricted "
                                    f"to prose-tier claims (`{v5_column}`)",
            })
    return table


def _itt_cell(rows, column):
    """Intention to treat: denominator is every row given, pending is printed."""
    if column is None:
        return Cell(0, 0, unavailable="no source column")
    values = [parse_bool(r.get(column)) for r in rows]
    pending = sum(1 for v in values if v is None)
    return Cell(sum(1 for v in values if v), len(values), excluded=0,
                pending=pending, unit="rows")


def _v2_cells(base_rows, struct_rows, composite, surface_column):
    """V2's harmonised case set: cases present in both arms and, when the surface
    column exists, every repetition of the case judgeable in both arms."""
    def keep(rows):
        if surface_column is None:
            return rows
        return [r for r in rows if has_surface(r, surface_column)]

    base_cases = {r.get("case_id") for r in base_rows}
    struct_cases = {r.get("case_id") for r in struct_rows}
    common = base_cases & struct_cases
    if surface_column is not None:
        lost = {r.get("case_id") for r in base_rows + struct_rows
                if has_surface(r, surface_column) is not True}
        common = common - lost
        rule = (f"cases present in both arms with every repetition judgeable in "
                f"both: {len(common)} of {len(base_cases | struct_cases)} cases")
    else:
        rule = (f"cases present in both arms: {len(common)} of "
                f"{len(base_cases | struct_cases)}; surface column absent, so the "
                f"surface half of the harmonisation could not be applied")
    b = pass5([r for r in keep(base_rows) if r.get("case_id") in common], composite)
    s = pass5([r for r in keep(struct_rows) if r.get("case_id") in common], composite)
    return b, s, rule


# --------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------

def parse_pair(value):
    if value.count(":") != 1:
        raise argparse.ArgumentTypeError(
            "Use --pair BASELINE_RUNID:STRUCTURED_RUNID with run ids, not paths")
    baseline, structured = value.split(":", 1)
    if not baseline.strip() or not structured.strip():
        raise argparse.ArgumentTypeError("both sides of --pair must be run ids")
    return baseline.strip(), structured.strip()


def group_rows(rows):
    by_run = defaultdict(list)
    for row in rows:
        by_run[(row.get("run_id") or "").strip()].append(row)
    return by_run


def build_pairs(by_run, pairs, notes):
    """One entry per (pair, model), joined on (model, case_id, rep).

    A model present in one arm only is dropped from the comparison and named in
    `notes`: half a pair is not a comparison, and a silently one-sided cell is
    exactly the failure this report exists to prevent.
    """
    out = []
    for baseline_id, structured_id in pairs:
        base_rows = by_run.get(baseline_id, [])
        struct_rows = by_run.get(structured_id, [])
        if not base_rows:
            notes.append(f"- run_id `{baseline_id}` (baseline side of the pair) has "
                         f"no rows in the records given. The pair is skipped.")
        if not struct_rows:
            notes.append(f"- run_id `{structured_id}` (structured side of the pair) "
                         f"has no rows in the records given. The pair is skipped.")
        if not base_rows or not struct_rows:
            continue
        base_models = {(r.get("model") or "").strip() for r in base_rows}
        struct_models = {(r.get("model") or "").strip() for r in struct_rows}
        for model in sorted(base_models - struct_models):
            notes.append(f"- model `{model}` appears in `{baseline_id}` but not in "
                         f"`{structured_id}`. Excluded from every comparison cell.")
        for model in sorted(struct_models - base_models):
            notes.append(f"- model `{model}` appears in `{structured_id}` but not in "
                         f"`{baseline_id}`. Excluded from every comparison cell.")
        for model in sorted(base_models & struct_models):
            b = [r for r in base_rows if (r.get("model") or "").strip() == model]
            s = [r for r in struct_rows if (r.get("model") or "").strip() == model]
            b_keys = {(r.get("case_id"), r.get("rep")) for r in b}
            s_keys = {(r.get("case_id"), r.get("rep")) for r in s}
            if b_keys != s_keys:
                notes.append(
                    f"- `{model}`: the two arms do not carry the same "
                    f"(case_id, rep) set. Baseline only: "
                    f"{sorted(k for k in b_keys - s_keys) or 'none'}; structured "
                    f"only: {sorted(k for k in s_keys - b_keys) or 'none'}. Every "
                    f"per-arm rate is computed on the arm's own rows; the case "
                    f"level tables use the intersection and print it.")
            out.append({
                "pair": f"{baseline_id} -> {structured_id}",
                "model": model,
                "baseline_run": baseline_id,
                "structured_run": structured_id,
                "baseline_rows": b,
                "structured_rows": s,
            })
    return out


def check_arm_column(pairs_data, header, notes):
    """Cross-check --pair against the `arm` column when the records carry one."""
    if "arm" not in header:
        notes.append("- the records CSV carries no `arm` column, so the arm of each "
                     "run is taken from `--pair` alone.")
        return
    for pair in pairs_data:
        for key, expected in (("baseline_rows", "free"), ("structured_rows", "structured")):
            seen = {(r.get("arm") or "").strip() for r in pair[key]}
            if seen and seen != {expected}:
                notes.append(
                    f"- WARNING: `{pair['model']}` {key.replace('_rows', '')} side of "
                    f"pair `{pair['pair']}` carries arm value(s) {sorted(seen)} where "
                    f"`{expected}` was expected from --pair. The report follows "
                    f"--pair; check the run ids.")


def provenance_block(per_file, sources, surface_column, v5_sources, rehearsal,
                     c1a_filled=None, c1a_structured_filled=None,
                     numeric_sources=None):
    lines = ["## Provenance", ""]
    if rehearsal:
        lines += ["> **REHEARSAL. Not a result.** Every number below was produced "
                  "by a dry end-to-end rehearsal of the scoring chain on stored or "
                  "synthetic data. No figure on this page may be quoted, cited or "
                  "carried into the thesis.", ""]
    lines.append("| records file | sha256 | rows | columns |")
    lines.append("|---|---|---:|---:|")
    for item in per_file:
        lines.append(f"| `{item['path']}` | `{item['sha256'][:16]}` | {item['rows']} "
                     f"| {item['columns']} |")
    lines += ["", "### Where each criterion was read from", "",
              "| criterion | source column | note |", "|---|---|---|"]
    for criterion in COMMON_CRITERIA:
        source = sources[criterion]
        column = f"`{source.column}`" if source.column else "NONE"
        flag = " **PROVISIONAL**" if source.provisional else ""
        lines.append(f"| {criterion} {CRITERION_TITLES[criterion]} | {column}{flag} "
                     f"| {source.note or 'direct'} |")
    lines += ["",
              f"- judgeable-surface column: "
              f"{'`' + surface_column + '`' if surface_column else 'absent'}",
              f"- V5 prose-tier column: "
              f"{'`' + v5_sources['C5-common'] + '`' if v5_sources.get('C5-common') else 'absent'}"]
    for run_id, counts in sorted((numeric_sources or {}).items()):
        detail = ", ".join(f"{name} {n}" for name, n in sorted(counts.items()))
        lines.append(f"- C1a numeric verdict source, run `{run_id}`: {detail}")
    if c1a_filled:
        lines.append(
            "- C1a cells read from the run-time `numeric_ok` because "
            "`numeric_ok_full_surface` is empty on those rows: "
            + ", ".join(f"`{run or 'unnamed run'}` {n}"
                        for run, n in sorted(c1a_filled.items()))
            + ". On a free-arm row the two verdicts are identical by "
              "construction (2.1), so this is the defined quantity there.")
    if c1a_structured_filled:
        lines.append(
            "- **WARNING: " + ", ".join(f"`{run}` {n}" for run, n
                                        in sorted(c1a_structured_filled.items()))
            + " STRUCTURED row(s) fell back to the run-time `numeric_ok`. On the "
              "structured arm that value is the PROVISIONAL prose-only lower "
              "bound, not the published C1a. Run `stage_c_structured.py` over "
              "those runs before quoting any C1a cell.**")
    lines.append("")
    return lines


class RehearsalInputRefused(Exception):
    """Raised when rehearsal-stamped input would have produced an unstamped report."""


def build_report(records_paths, pairs, out_path=None, with_historical=False,
                 paired_unit="case", rehearsal=False):
    """Everything, as a list of markdown lines. Returns (lines, context)."""
    refuse_rep_pairing(paired_unit)
    rows, header, per_file, stamped = load_all(records_paths)
    # THE STAMP IS READ, NOT JUST WRITTEN. Refusing here is the whole point: a
    # rehearsal records CSV fed in without --rehearsal would otherwise produce a
    # confident, fully populated, entirely unstamped comparison report whose every
    # label was fabricated. Typing the flag does not make the numbers real, it
    # makes the output say so, which is why the stamp then propagates below.
    if stamped and not rehearsal:
        detail = "; ".join(f"{path}: {', '.join(ev[:3])}" for path, ev in stamped)
        raise RehearsalInputRefused(
            "at least one --records file carries a REHEARSAL stamp, so every "
            "figure derived from it would be fabricated. Refusing to write an "
            "unstamped report. Evidence -> " + detail + ". Pass --rehearsal to "
            "produce a report that says so on its face, or point --records at a "
            "scored run.")
    if stamped:
        rehearsal = True
    c1a_filled, c1a_structured_filled = materialise_c1a(rows, header)
    materialise_policy_ok(rows, header)
    numeric_sources = numeric_source_summary(rows, header)
    sources = resolve_sources(header)
    surface_column = resolve_optional(header, SURFACE_SOURCES)
    v5_sources = {k: resolve_optional(header, v) for k, v in V5_SOURCES.items()}

    notes = []
    by_run = group_rows(rows)
    pairs_data = build_pairs(by_run, pairs, notes)
    check_arm_column(pairs_data, header, notes)

    title = "# Axis 3, arm comparison: free-output baseline versus structured output"
    lines = [title, ""]
    if rehearsal:
        lines += ["**REHEARSAL OUTPUT. NOT A RESULT. NOT A SCORED FIGURE.**", ""]
    lines += [
        "Produced by `scripts/validation/axis3_arm_compare.py`. Joined on "
        "`(model, case_id, rep)`. Every cell carries its denominator and the "
        "count of rows excluded.",
        "",
    ]
    lines += provenance_block(per_file, sources, surface_column, v5_sources,
                              rehearsal, c1a_filled, c1a_structured_filled,
                              numeric_sources)

    lines += ["## Rows read", "",
              f"- records files: {len(per_file)}",
              f"- rows pooled: {len(rows)}",
              f"- run ids seen: {sorted(k for k in by_run if k) or 'none'}",
              f"- pairs requested: {[f'{a} -> {b}' for a, b in pairs] or 'none'}",
              f"- model blocks compared: {len(pairs_data)}",
              ""]
    if notes:
        lines += ["### Exclusions and mismatches", ""] + notes + [""]
    if not pairs_data:
        lines += ["No comparable pair was found in the records given, so no "
                  "comparison table is produced. The exclusions above say why.",
                  ""]
        text = "\n".join(lines) + "\n"
        if out_path is not None:
            _write_out(out_path, text)
        return lines, {"pairs": pairs_data, "sources": sources, "rows": rows}

    lines += ["## Tables", ""]
    comparison = build_comparison_table(pairs_data, sources, with_historical)
    lines += comparison.render()
    lines += rates_table(pairs_data, header).render()
    lines += decomposition_table(pairs_data, header).render()
    lines += discordance_table(pairs_data, sources, header, paired_unit).render()
    lines += failure_attribution(pairs_data, sources, header).render()
    lines += variants_table(pairs_data, sources, surface_column, v5_sources,
                            paired_unit).render()

    lines += ["## Standing rules enforced by this file", "",
              "- The common comparison table's columns are asserted to be exactly "
              f"{list(COMMON_CRITERIA)}. No ct_* field and no schema term can "
              "reach it.",
              "- pass^5 and every paired quantity are computed at case level. "
              "Rep-index pairing raises.",
              "- The historical rubric-v1 figures are quoted only and cannot be "
              "differenced.",
              ""]

    text = "\n".join(lines) + "\n"
    if out_path is not None:
        _write_out(out_path, text)
    return lines, {"pairs": pairs_data, "sources": sources, "rows": rows,
                   "comparison": comparison}


def _refuse_readonly_out(path):
    """Never write into stored evidence.

    The three corrected 2026-09-01 runs, the closed full-system audit, the frozen
    pilot record and the backup are read-only for the whole of this experiment.
    The check is on the path, not on a flag, for the same reason
    `stage_c_structured.is_frozen_audit_path` is: an operator in a hurry is the
    threat model.
    """
    path = Path(path).resolve()
    parts = {p.lower() for p in path.parts}
    for marker in ("pilot_records", "_frozen_axis3_backup_20260921"):
        if marker in parts:
            raise SystemExit(f"Refusing to write into the frozen record: {path}")
    if (path.parent / "agent_eval_raw.jsonl").exists():
        raise SystemExit(
            f"Refusing to write into a stored run directory (it holds "
            f"agent_eval_raw.jsonl): {path.parent}")
    return path


def _write_out(path, text):
    path = _refuse_readonly_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare the free-output baseline arm with the structured arm.")
    ap.add_argument("--records", action="append", required=True,
                    help="axis3_full_system_records.csv; repeat for each file")
    ap.add_argument("--pair", action="append", required=True, type=parse_pair,
                    help="BASELINE_RUNID:STRUCTURED_RUNID; repeat for each pair")
    ap.add_argument("--out", required=True, help="markdown report to write")
    ap.add_argument("--with-historical-v1", action="store_true",
                    help="also quote the as-published 2026-09-01 rubric-v1 figures "
                         "in a captioned row. They are never differenced.")
    ap.add_argument("--paired-unit", default="case", choices=("case", "rep"),
                    help="only 'case' is admissible; 'rep' exists so that asking "
                         "for it is answered with the reason it is refused")
    ap.add_argument("--rehearsal", action="store_true",
                    help="stamp the report REHEARSAL so it can never be mistaken "
                         "for a scored figure")
    args = ap.parse_args(argv)

    try:
        build_report([Path(p) for p in args.records], args.pair, Path(args.out),
                     args.with_historical_v1, args.paired_unit, args.rehearsal)
    except RehearsalInputRefused as exc:
        raise SystemExit(f"Refused: {exc}")
    except PairingRefused as exc:
        raise SystemExit(f"Refused: {exc}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the Axis-3 arm comparison.

Everything here is synthetic. The point of the file is not to check arithmetic,
which is a handful of counters, but to pin the four separations that the
pre-registration states in prose and that `axis3_arm_compare.py` is supposed to
make structurally impossible to break:

* no `ct_*` field and no schema term reaches the common comparison table, even
  when the input CSV is full of them;
* pass^5 is a CASE-level quantity and a rep-level grouping of the same rows
  gives a different answer, so the two can never be confused by accident;
* a model or a case present in one arm only is excluded and NAMED, never
  silently half-compared;
* the flip attribution names the criterion that actually moved.
"""
from __future__ import annotations

import csv

import pytest

from scripts.validation.axis3_arm_compare import (
    COMMON_CRITERIA,
    CommonComparisonContaminated,
    PairingRefused,
    Table,
    assert_common_table_clean,
    build_report,
    case_states,
    difference,
    historical_cell,
    pass1,
    pass5,
    refuse_rep_pairing,
    resolve_sources,
)


# --------------------------------------------------------------------------------
# Synthetic records CSVs
# --------------------------------------------------------------------------------

BASE_COLUMNS = [
    "run_id", "model", "uid", "case_id", "rep", "category", "arm",
    "deterministic_pass", "numeric_ok_full_surface", "numeric_ok", "stage_c",
    "critical_coverage", "policy_ok", "full_system", "judged_surface_chars",
    "stage_a_ok", "stage_b_ok",
]

CT_COLUMNS = [
    "contract_outcome", "contract_outcome_finish_reason", "schema_conformant",
    "ct_scored", "ct_not_scored_reason", "ct_n_gt", "ct_n_emitted", "ct_matched",
    "ct_exact_set", "ct_no_omission", "ct_over_inclusive", "ct_extra_periods",
    "ct_extra_in_payload", "ct_extra_not_in_payload", "ct_duplicate_periods",
    "ct_order_ok",
]


def make_row(run_id, model, case_id, rep, arm="free", full=True, **overrides):
    row = {
        "run_id": run_id,
        "model": model,
        "uid": f"{case_id}#{rep}",
        "case_id": case_id,
        "rep": str(rep),
        "category": "B",
        "arm": arm,
        "deterministic_pass": "True",
        "numeric_ok_full_surface": "True",
        "numeric_ok": "True",
        "stage_c": "True",
        "critical_coverage": "True",
        "policy_ok": "True",
        "full_system": str(bool(full)),
        "judged_surface_chars": "900",
        "stage_a_ok": "True",
        "stage_b_ok": "True",
    }
    row.update({k: ("" if v is None else str(v)) for k, v in overrides.items()})
    return row


def add_ct(row, **overrides):
    """Attach the intervention-only contract diagnostics to a structured row."""
    row.update({
        "contract_outcome": "valid",
        "contract_outcome_finish_reason": "stop",
        "schema_conformant": "True",
        "ct_scored": "True",
        "ct_not_scored_reason": "",
        "ct_n_gt": "3",
        "ct_n_emitted": "3",
        "ct_matched": "3",
        "ct_exact_set": "True",
        "ct_no_omission": "True",
        "ct_over_inclusive": "False",
        "ct_extra_periods": "",
        "ct_extra_in_payload": "",
        "ct_extra_not_in_payload": "",
        "ct_duplicate_periods": "",
        "ct_order_ok": "True",
    })
    row.update({k: ("" if v is None else str(v)) for k, v in overrides.items()})
    return row


def write_csv(path, rows, columns=None):
    columns = columns or (BASE_COLUMNS if not rows else list(rows[0]))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def clean_pair_files(tmp_path, n_cases=3, n_reps=2):
    """A baseline run and a structured run over the same cases and reps."""
    base, struct = [], []
    for index in range(n_cases):
        case_id = f"C{index:02d}"
        for rep in range(1, n_reps + 1):
            base.append(make_row("luna_free", "gpt-5.6-luna", case_id, rep, "free"))
            struct.append(add_ct(make_row("luna_struct", "gpt-5.6-luna", case_id,
                                          rep, "structured")))
    return (write_csv(tmp_path / "base.csv", base),
            write_csv(tmp_path / "struct.csv", struct))


def render(tmp_path, records, pairs, **kwargs):
    out = tmp_path / "report.md"
    lines, ctx = build_report([str(p) for p in records], pairs, out, **kwargs)
    return "\n".join(lines), ctx


# --------------------------------------------------------------------------------
# 1. A clean pair
# --------------------------------------------------------------------------------

def test_clean_pair_produces_every_block_with_denominators(tmp_path):
    base, struct = clean_pair_files(tmp_path)
    text, ctx = render(tmp_path, [base, struct], [("luna_free", "luna_struct")])

    assert len(ctx["pairs"]) == 1
    assert ctx["pairs"][0]["model"] == "gpt-5.6-luna"
    for heading in ("1. Paired baseline versus structured, common criteria",
                    "2. Intervention-only rates, outside the comparison",
                    "3. Critical-transition decomposition",
                    "4. Per-case discordance and flip attribution",
                    "5. Headline variants"):
        assert heading in text
    # Every cell of the comparison carries passed/denominator and an excl count.
    assert "6/6 = 100.0% (excl 0 rows)" in text
    assert "3/3 = 100.0% (excl 0 cases)" in text
    # The mandatory 3.4.3 sentence, verbatim, beneath the decomposition table.
    assert "They may not be pooled with the gate, averaged with it, or " \
           "presented in the same column as it." in text
    assert (tmp_path / "report.md").exists()


def test_c1a_prefers_the_full_surface_column_and_flags_the_fallback(tmp_path):
    base, struct = clean_pair_files(tmp_path)
    _, ctx = render(tmp_path, [base, struct], [("luna_free", "luna_struct")])
    assert ctx["sources"]["C1a"].column == "numeric_ok_full_surface"
    assert ctx["sources"]["C1a"].provisional is False

    # Strip the full-surface column from BOTH files: C1a must fall back and say,
    # in the report, that the cell is the provisional prose-only lower bound.
    columns = [c for c in BASE_COLUMNS if c != "numeric_ok_full_surface"]
    assert resolve_sources(columns)["C1a"].column == "numeric_ok"
    assert resolve_sources(columns)["C1a"].provisional is True

    prose_base = write_csv(
        tmp_path / "prose_base.csv",
        [{k: make_row("luna_free", "m", "C00", rep)[k] for k in columns}
         for rep in (1, 2)], columns)
    prose_struct = write_csv(
        tmp_path / "prose_struct.csv",
        [{k: make_row("luna_struct", "m", "C00", rep, "structured")[k]
          for k in columns} for rep in (1, 2)], columns)
    text, _ = render(tmp_path, [prose_base, prose_struct],
                     [("luna_free", "luna_struct")])
    assert "PROVISIONAL prose-only lower bound" in text


def test_free_arm_c1a_is_filled_from_numeric_ok_and_a_structured_fill_warns(tmp_path):
    # The free arm never carries `numeric_ok_full_surface`, because 2.1 makes
    # the two verdicts identical there. The fill is silent-but-recorded on that
    # arm and a loud warning on the structured one.
    free_columns = [c for c in BASE_COLUMNS if c != "numeric_ok_full_surface"]
    base = write_csv(
        tmp_path / "b.csv",
        [{k: make_row("luna_free", "m", "C00", rep)[k] for k in free_columns}
         for rep in (1, 2)], free_columns)
    struct = write_csv(
        tmp_path / "s.csv",
        [add_ct(make_row("luna_struct", "m", "C00", rep, "structured"))
         for rep in (1, 2)])
    text, ctx = render(tmp_path, [base, struct], [("luna_free", "luna_struct")])
    assert ctx["sources"]["C1a"].column == "numeric_ok_full_surface"
    assert "C1a cells read from the run-time `numeric_ok`" in text
    assert "`luna_free` 2" in text
    assert "WARNING" not in text
    # Both arms keep a full C1a denominator, which is the point of the fill.
    assert text.count("2/2 = 100.0% (excl 0 rows)") >= 2

    unscored = write_csv(
        tmp_path / "s2.csv",
        [{k: add_ct(make_row("luna_struct", "m", "C00", rep, "structured"))[k]
          for k in list(BASE_COLUMNS) + CT_COLUMNS if k != "numeric_ok_full_surface"}
         for rep in (1, 2)],
        [c for c in list(BASE_COLUMNS) + CT_COLUMNS
         if c != "numeric_ok_full_surface"])
    text2, _ = render(tmp_path, [struct, unscored, base],
                      [("luna_free", "luna_struct")])
    assert "STRUCTURED row(s) fell back to the run-time `numeric_ok`" in text2
    assert "PROVISIONAL prose-only lower bound, not the published C1a" in text2


# --------------------------------------------------------------------------------
# 2. A ct_* column in the input never reaches the comparison table
# --------------------------------------------------------------------------------

def test_ct_columns_in_the_input_never_reach_the_comparison_table(tmp_path):
    base, struct = clean_pair_files(tmp_path)
    _, ctx = render(tmp_path, [base, struct], [("luna_free", "luna_struct")])
    comparison = ctx["comparison"]

    assert comparison.columns == COMMON_CRITERIA
    assert_common_table_clean(comparison)
    # The caption names the forbidden terms on purpose, so screen the grid
    # itself: the header line, and every data row.
    grid = [line for line in comparison.render()
            if line.startswith("|") and not set(line) <= set("|- ")]
    assert len(grid) == len(comparison.rows) + 1
    for line in grid:
        for term in ("ct_", "schema", "contract", "exact_set", "SSP", "slot"):
            assert term not in line, line
    # The ct_* data is in the file and is reported, just not in table 1.
    with struct.open(encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    assert "ct_exact_set" in header


def test_allowlist_is_asserted_and_an_extra_column_raises():
    good = Table("t", COMMON_CRITERIA)
    good.add("row", {c: "1/1 = 100.0% (excl 0 rows)" for c in COMMON_CRITERIA})
    assert assert_common_table_clean(good) is True

    extra = Table("t", tuple(COMMON_CRITERIA) + ("ct_exact_set",))
    with pytest.raises(CommonComparisonContaminated):
        assert_common_table_clean(extra)

    reordered = Table("t", tuple(reversed(COMMON_CRITERIA)))
    with pytest.raises(CommonComparisonContaminated):
        assert_common_table_clean(reordered)

    smuggled = Table("t", COMMON_CRITERIA)
    smuggled.add("row", {c: "" for c in COMMON_CRITERIA})
    smuggled.rows[0][1]["C2"] = "ct_exact_set 4/5"
    with pytest.raises(CommonComparisonContaminated):
        assert_common_table_clean(smuggled)


def test_historical_v1_figures_are_quoted_and_cannot_be_differenced(tmp_path):
    base, struct = clean_pair_files(tmp_path)
    text, ctx = render(tmp_path, [base, struct], [("luna_free", "luna_struct")],
                       with_historical=True)
    assert "HISTORICAL v1" in text
    assert "never differenced" in text
    with pytest.raises(CommonComparisonContaminated):
        difference(historical_cell(0.754), pass1(ctx["rows"], "full_system"))
    with pytest.raises(CommonComparisonContaminated):
        difference(pass1(ctx["rows"], "full_system"), historical_cell(0.754))


# --------------------------------------------------------------------------------
# 3. A deliberately mismatched case set
# --------------------------------------------------------------------------------

def test_mismatched_case_set_is_named_and_the_case_tables_use_the_intersection(tmp_path):
    base = [make_row("luna_free", "gpt-5.6-luna", case, rep)
            for case in ("C00", "C01") for rep in (1, 2)]
    struct = [add_ct(make_row("luna_struct", "gpt-5.6-luna", case, rep, "structured"))
              for case in ("C00", "C99") for rep in (1, 2)]
    files = (write_csv(tmp_path / "b.csv", base),
             write_csv(tmp_path / "s.csv", struct))
    text, _ = render(tmp_path, files, [("luna_free", "luna_struct")])

    assert "do not carry the same" in text
    assert "C01" in text and "C99" in text
    # Only the shared case appears as a paired row of the discordance table.
    assert "| gpt-5.6-luna C00 |" in text
    assert "| gpt-5.6-luna C01 |" not in text
    assert "| gpt-5.6-luna C99 |" not in text
    assert "1 paired cases" in text


# --------------------------------------------------------------------------------
# 4. A missing model
# --------------------------------------------------------------------------------

def test_a_model_present_in_one_arm_only_is_excluded_and_named(tmp_path):
    base = ([make_row("free_run", "gpt-5.6-luna", "C00", rep) for rep in (1, 2)]
            + [make_row("free_run", "gpt-5.6-terra", "C00", rep) for rep in (1, 2)])
    struct = [add_ct(make_row("struct_run", "gpt-5.6-luna", "C00", rep, "structured"))
              for rep in (1, 2)]
    files = (write_csv(tmp_path / "b.csv", base),
             write_csv(tmp_path / "s.csv", struct))
    text, ctx = render(tmp_path, files, [("free_run", "struct_run")])

    assert [p["model"] for p in ctx["pairs"]] == ["gpt-5.6-luna"]
    assert "gpt-5.6-terra` appears in `free_run` but not in `struct_run`" in text
    assert "Excluded from every comparison cell" in text


def test_an_unknown_run_id_skips_the_pair_and_says_so(tmp_path):
    base, struct = clean_pair_files(tmp_path)
    text, ctx = render(tmp_path, [base, struct], [("luna_free", "does_not_exist")])
    assert ctx["pairs"] == []
    assert "does_not_exist" in text
    assert "No comparable pair was found" in text


# --------------------------------------------------------------------------------
# 5. pass^5 is CASE level, and a rep-level grouping of the same rows differs
# --------------------------------------------------------------------------------

def rep_level_pass5(rows, column):
    """A deliberately WRONG implementation, here only as the contrast.

    It groups by repetition index instead of by case, which is the rep-index
    pairing the pre-registration forbids. It exists in the test file and nowhere
    in the module.
    """
    from collections import defaultdict
    by_rep = defaultdict(list)
    for row in rows:
        by_rep[row["rep"]].append(row[column] == "True")
    states = [all(v) for v in by_rep.values()]
    return sum(states), len(states)


def test_pass5_is_case_level_and_not_rep_level():
    # A: both reps pass. B: rep 2 fails. C: rep 1 fails.
    rows = [
        make_row("r", "m", "A", 1, full=True),
        make_row("r", "m", "A", 2, full=True),
        make_row("r", "m", "B", 1, full=True),
        make_row("r", "m", "B", 2, full=False),
        make_row("r", "m", "C", 1, full=False),
        make_row("r", "m", "C", 2, full=True),
    ]
    case_cell = pass5(rows, "full_system")
    assert (case_cell.passed, case_cell.denominator) == (1, 3)
    assert case_cell.unit == "cases"
    # The rep-index grouping of the very same rows answers 0 of 2, so the two
    # readings are not interchangeable and a silent swap would be visible.
    assert rep_level_pass5(rows, "full_system") == (0, 2)
    assert (case_cell.passed, case_cell.denominator) != rep_level_pass5(
        rows, "full_system")
    # pass^1 is the row-level reading and is a third, different number.
    row_cell = pass1(rows, "full_system")
    assert (row_cell.passed, row_cell.denominator) == (4, 6)

    states = case_states(rows, "full_system")
    assert states == {"A": True, "B": False, "C": False}


def test_rep_pairing_is_refused_with_the_reason():
    assert refuse_rep_pairing("case") == "case"
    with pytest.raises(PairingRefused) as excinfo:
        refuse_rep_pairing("rep")
    message = str(excinfo.value)
    assert "only admissible paired unit is the CASE" in message
    assert "13" in message


def test_build_report_refuses_a_rep_paired_unit(tmp_path):
    base, struct = clean_pair_files(tmp_path)
    with pytest.raises(PairingRefused):
        build_report([str(base), str(struct)], [("luna_free", "luna_struct")],
                     tmp_path / "never.md", paired_unit="rep")
    assert not (tmp_path / "never.md").exists()


# --------------------------------------------------------------------------------
# 6. The flip attribution names the right criterion
# --------------------------------------------------------------------------------

def test_flip_attribution_names_the_criterion_that_moved(tmp_path):
    # C00 regresses on coverage alone: C2 is the only criterion that differs.
    base = [make_row("b", "gpt-5.6-luna", "C00", rep) for rep in (1, 2)]
    struct = [
        add_ct(make_row("s", "gpt-5.6-luna", "C00", 1, "structured",
                        critical_coverage=False, full_system=False)),
        add_ct(make_row("s", "gpt-5.6-luna", "C00", 2, "structured")),
    ]
    files = (write_csv(tmp_path / "b.csv", base),
             write_csv(tmp_path / "s.csv", struct))
    text, _ = render(tmp_path, files, [("b", "s")])

    flip_row = [line for line in text.splitlines()
                if line.startswith("| gpt-5.6-luna C00 |")][0]
    assert "regression (baseline PASS, structured FAIL)" in flip_row
    assert "C2" in flip_row
    assert "C4" not in flip_row
    assert "unattributed" not in flip_row


def test_flip_attribution_falls_back_to_unattributed_when_no_criterion_moved(tmp_path):
    # The composite moves but every per-criterion column is unchanged, which is
    # what a Stage-A, Stage-B or loss-of-surface flip looks like when the
    # records CSV carries no column for it.
    columns = [c for c in BASE_COLUMNS if c not in ("stage_a_ok", "stage_b_ok")]
    base = [{k: make_row("b", "m", "C00", rep)[k] for k in columns} for rep in (1, 2)]
    struct = [{k: make_row("s", "m", "C00", rep, "structured",
                           full_system=False)[k] for k in columns} for rep in (1, 2)]
    files = (write_csv(tmp_path / "b.csv", base, columns),
             write_csv(tmp_path / "s.csv", struct, columns))
    text, _ = render(tmp_path, files, [("b", "s")])
    flip_row = [line for line in text.splitlines()
                if line.startswith("| m C00 |")][0]
    assert "unattributed" in flip_row
    assert "Stage A, Stage B or loss of surface" in flip_row


def test_unpaired_failure_attribution_counts_both_arms(tmp_path):
    base = [make_row("b", "m", "C00", rep) for rep in (1, 2)]
    struct = [
        add_ct(make_row("s", "m", "C00", 1, "structured",
                        critical_coverage=False, full_system=False)),
        add_ct(make_row("s", "m", "C00", 2, "structured")),
    ]
    files = (write_csv(tmp_path / "b.csv", base),
             write_csv(tmp_path / "s.csv", struct))
    text, _ = render(tmp_path, files, [("b", "s")])
    assert "4b. Per-criterion failure attribution, unpaired" in text
    assert "| m rows failing C5-common | 0 of 2 | 1 of 2 | +1 |" in text
    assert "| m C2 | 0 | 1 | +1 |" in text


# --------------------------------------------------------------------------------
# 7. Empty input
# --------------------------------------------------------------------------------

def test_empty_records_file_produces_a_report_and_no_tables(tmp_path):
    empty = write_csv(tmp_path / "empty.csv", [], BASE_COLUMNS)
    text, ctx = render(tmp_path, [empty], [("luna_free", "luna_struct")])
    assert ctx["pairs"] == []
    assert "rows pooled: 0" in text
    assert "No comparable pair was found" in text
    assert "1. Paired baseline versus structured" not in text


def test_a_file_with_no_header_at_all_does_not_crash(tmp_path):
    blank = tmp_path / "blank.csv"
    blank.write_text("", encoding="utf-8")
    text, ctx = render(tmp_path, [blank], [("a", "b")])
    assert ctx["pairs"] == []
    assert "rows pooled: 0" in text


# --------------------------------------------------------------------------------
# Intervention-only blocks stay intervention-only
# --------------------------------------------------------------------------------

def test_rates_and_decomposition_report_denominators_and_exclusions(tmp_path):
    base = [make_row("b", "m", "C00", rep) for rep in (1, 2, 3)]
    struct = [
        add_ct(make_row("s", "m", "C00", 1, "structured")),
        add_ct(make_row("s", "m", "C00", 2, "structured"),
               contract_outcome="validator_invalid", ct_exact_set=False,
               ct_over_inclusive=True, ct_extra_periods="2",
               ct_extra_in_payload="2", ct_n_emitted=4),
        add_ct(make_row("s", "m", "C00", 3, "structured", full_system=False),
               contract_outcome="runtime_error", ct_scored=False,
               ct_not_scored_reason="runtime_error", ct_n_gt=0, ct_n_emitted=0,
               ct_matched=0, ct_exact_set=False, ct_no_omission=False),
    ]
    files = (write_csv(tmp_path / "b.csv", base),
             write_csv(tmp_path / "s.csv", struct))
    text, _ = render(tmp_path, files, [("b", "s")])

    # Validity denominators drop the runtime error and say so.
    assert "2/2 = 100.0% (excl 1 rows)" in text      # json schema validity
    assert "1/2 = 50.0% (excl 1 rows)" in text       # contract validity
    assert "runtime_error 1" in text
    assert "no status column" in text
    # Decomposition, on ct_scored records only, with the excluded count printed.
    assert "excluded 1 (runtime_error 1)" in text
    assert "1/2 = 50.0% (excl 0 records)" in text
    assert "1 / 0" in text                            # extras in / not in payload


def test_rates_are_not_computable_without_the_contract_outcome_column(tmp_path):
    # A pre-edit records CSV has no taxonomy column. The rates must say so, not
    # print 0/65, which would read as a measured total failure.
    base = [make_row("b", "m", "C00", rep) for rep in (1, 2)]
    struct = [make_row("s", "m", "C00", rep, "structured") for rep in (1, 2)]
    files = (write_csv(tmp_path / "b.csv", base),
             write_csv(tmp_path / "s.csv", struct))
    text, _ = render(tmp_path, files, [("b", "s")])
    assert "no `contract_outcome` column in this records CSV" in text
    assert "0/2 = 0.0%" not in text
    assert "records CSV lacks ['ct_scored'" in text


def test_c4_is_derived_from_policy_problems_when_only_the_list_is_stored(tmp_path):
    columns = [c for c in BASE_COLUMNS if c != "policy_ok"] + ["policy_problems"]
    base = [{**{k: make_row("b", "m", "C00", rep)[k] for k in columns[:-1]},
             "policy_problems": ""} for rep in (1, 2)]
    struct = [{**{k: make_row("s", "m", "C00", rep, "structured")[k]
                  for k in columns[:-1]},
               "policy_problems": "not Greek" if rep == 1 else ""}
              for rep in (1, 2)]
    files = (write_csv(tmp_path / "b.csv", base, columns),
             write_csv(tmp_path / "s.csv", struct, columns))
    text, ctx = render(tmp_path, files, [("b", "s")])
    assert ctx["sources"]["C4"].column == "policy_ok"
    # An empty problem list is a PASS, a non-empty one a FAIL.
    comparison = "\n".join(ctx["comparison"].render())
    assert "| m baseline pass^1 |" in comparison
    assert "2/2 = 100.0% (excl 0 rows)" in comparison
    assert "1/2 = 50.0% (excl 0 rows)" in comparison


def test_numeric_source_is_reported_per_run(tmp_path):
    columns = BASE_COLUMNS + ["numeric_ok_used", "numeric_source"]
    base = [{**make_row("b", "m", "C00", rep), "numeric_ok_used": "True",
             "numeric_source": "prose_only_fallback"} for rep in (1, 2)]
    struct = [{**make_row("s", "m", "C00", rep, "structured"),
               "numeric_ok_used": "True", "numeric_source": "full_surface"}
              for rep in (1, 2)]
    files = (write_csv(tmp_path / "b.csv", base, columns),
             write_csv(tmp_path / "s.csv", struct, columns))
    text, ctx = render(tmp_path, files, [("b", "s")])
    # `numeric_ok_used` outranks both raw columns: it IS the published verdict.
    assert ctx["sources"]["C1a"].column == "numeric_ok_used"
    assert "C1a numeric verdict source, run `b`: prose_only_fallback 2" in text
    assert "C1a numeric verdict source, run `s`: full_surface 2" in text


def test_rehearsal_stamp_is_impossible_to_miss(tmp_path):
    base, struct = clean_pair_files(tmp_path)
    text, _ = render(tmp_path, [base, struct], [("luna_free", "luna_struct")],
                     rehearsal=True)
    assert "REHEARSAL OUTPUT. NOT A RESULT. NOT A SCORED FIGURE." in text
    assert text.splitlines()[2].startswith("**REHEARSAL")

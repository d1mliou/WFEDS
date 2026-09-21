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
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


NEGATIVE_LABELS = {"NOT_IN_RESULT", "CONTRADICTED"}
DECISIONS = {"PASS": True, "FAIL": False}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_label_record(record: dict[str, Any] | None) -> bool | None:
    """Return True when a Stage-C label record has no negative atomic claim."""
    if record is None:
        return None
    labels = {c.get("label") for c in record.get("claims", [])
              if isinstance(c, dict)}
    return not bool(labels & NEGATIVE_LABELS)


def parse_decision(value: str | None) -> bool | None:
    return DECISIONS.get((value or "").strip().upper())


def deterministic_pass(row: dict[str, Any]) -> bool:
    """The deterministic gate used by the full-system score.

    Unlike the historical ``all_gates_ok`` field, numeric traceability is
    explicitly included here.
    """
    return bool(
        row.get("stage_a_ok")
        and row.get("stage_b_ok")
        and row.get("numeric_ok")
        and not row.get("policy_problems")
        and not row.get("error")
    )


def bounds(values: list[bool | None]) -> tuple[float, float, int]:
    """Return lower bound, upper bound and unresolved count."""
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


def label_map(path: Path) -> dict[str, dict[str, Any]]:
    return {r["uid"]: r for r in read_json(path)}


def mentioned_hours(reply: str, hours: list[int]) -> list[int]:
    """Orientation only, never the authoritative coverage verdict."""
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


def analyse_run(name: str, run_dir: Path,
                audit: dict[tuple[str, str], dict[str, str]]) -> tuple[
                    dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    raw_path = run_dir / "agent_eval_raw.jsonl"
    metrics_path = run_dir / "agent_eval_metrics.json"
    stage_dir = run_dir / "stage_c_full"
    labels1_path = stage_dir / "stage_c_labels.json"
    labels2_path = stage_dir / "stage_c_labels_pass2.json"

    for path in (raw_path, metrics_path, labels1_path, labels2_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metrics = read_json(metrics_path)
    labels1, labels2 = label_map(labels1_path), label_map(labels2_path)
    categories = stage_c_categories(stage_dir)
    rows = [r for r in read_jsonl(raw_path) if r["category"] in categories]

    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, str]] = []
    for row in rows:
        uid = f"{row['case_id']}#{row['rep']}"
        l1, l2 = labels1.get(uid), labels2.get(uid)
        clean1, clean2 = clean_label_record(l1), clean_label_record(l2)
        previous = audit.get((name, uid), {})
        human_c = parse_decision(previous.get("human_stage_c"))

        if row.get("error"):
            stage_c: bool | None = False
            c_source = "runtime_error"
            human_c_required = False
        elif human_c is not None:
            stage_c = human_c
            c_source = "human_audit"
            human_c_required = True
        elif clean1 is True and clean2 is True:
            stage_c = True
            c_source = "two_pass_clean_consensus"
            human_c_required = False
        else:
            # Any negative label, disagreement, or missing label is audited.
            stage_c = None
            c_source = "human_audit_pending"
            human_c_required = True

        hours = [int(h) for h in row.get("cutoff_hours_to_cover") or []]
        human_coverage = parse_decision(previous.get("human_critical_coverage"))
        if not hours:
            coverage: bool | None = True
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
        components = (det, stage_c, coverage)
        if any(v is False for v in components):
            full: bool | None = False
        elif all(v is True for v in components):
            full = True
        else:
            full = None

        record = {
            "run_id": name,
            "model": metrics.get("model", ""),
            "uid": uid,
            "case_id": row["case_id"],
            "rep": row["rep"],
            "category": row["category"],
            "deterministic_pass": det,
            "stage_c_pass1": clean1,
            "stage_c_pass2": clean2,
            "stage_c": stage_c,
            "stage_c_source": c_source,
            "critical_hours": hours,
            "critical_coverage": coverage,
            "critical_coverage_source": coverage_source,
            "full_system": full,
        }
        records.append(record)

        audit_rows.append({
            "run_id": name,
            "model": str(metrics.get("model", "")),
            "uid": uid,
            "category": row["category"],
            "deterministic_pass": "PASS" if det else "FAIL",
            "stage_c_pass1": "" if clean1 is None else ("PASS" if clean1 else "FAIL"),
            "stage_c_pass2": "" if clean2 is None else ("PASS" if clean2 else "FAIL"),
            "human_stage_c_required": "YES" if human_c_required else "NO",
            "human_stage_c": previous.get("human_stage_c", ""),
            "critical_event_hours": ",".join(map(str, hours)),
            "literal_hours_found": ",".join(map(str, mentioned_hours(row.get("reply", ""), hours))),
            "human_critical_coverage_required": "YES" if human_coverage_required else "NO",
            "human_critical_coverage": previous.get("human_critical_coverage", ""),
            "user_text": row.get("user_text", ""),
            "reply": row.get("reply", ""),
            "label_notes_pass1": compact_notes(l1),
            "label_notes_pass2": compact_notes(l2),
            "audit_comment": previous.get("audit_comment", ""),
        })

    p1 = bounds([r["full_system"] for r in records])
    pk = passk_bounds(records)
    summary = {
        "run_id": name,
        "model": metrics.get("model", ""),
        "scope_categories": sorted(categories),
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
    }
    return summary, records, audit_rows


AUDIT_FIELDS = [
    "run_id", "model", "uid", "category", "deterministic_pass",
    "stage_c_pass1", "stage_c_pass2", "human_stage_c_required",
    "human_stage_c", "critical_event_hours", "literal_hours_found",
    "human_critical_coverage_required", "human_critical_coverage",
    "user_text", "reply", "label_notes_pass1", "label_notes_pass2",
    "audit_comment",
]


def write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["critical_hours"] = ",".join(map(str, row["critical_hours"]))
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
        "| model | cases | repetitions | full-system pass^1 | full-system pass^5 | "
        "Stage C pending | critical coverage pending |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        p1 = fmt_bound(s["full_system_pass1_lower"],
                       s["full_system_pass1_upper"],
                       s["full_system_pass1_pending"])
        pk = fmt_bound(s["full_system_passk_lower"],
                       s["full_system_passk_upper"],
                       s["full_system_passk_pending_cases"])
        lines.append(
            f"| {s['model']} | {s['n_cases']} | {s['n_repetitions']} | {p1} | "
            f"{pk} | {s['stage_c_human_pending']} | "
            f"{s['critical_coverage_human_pending']} |")
    lines += [
        "",
        "## Human audit rule",
        "",
        "- Fill `human_stage_c` only where `human_stage_c_required=YES`.",
        "- Fill `human_critical_coverage` only where its required column is `YES`.",
        "- Valid values are `PASS` and `FAIL`.",
        "- Critical coverage passes when every listed cut-off transition is explicitly "
        "mentioned or accurately covered by a stated hour range. Exact counts are not required.",
        "- `literal_hours_found` is only an orientation aid. It never decides the score.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for --run")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, type=parse_run,
                    help="NAME=PATH to validation/3_agent; repeat for each model")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "axis3_full_system_audit.csv"
    audit = load_audit(audit_path)

    summaries, all_records, all_audit_rows = [], [], []
    for name, path in args.run:
        summary, records, audit_rows = analyse_run(name, path, audit)
        summaries.append(summary)
        all_records.extend(records)
        all_audit_rows.extend(audit_rows)

    required_audit_rows = [
        row for row in all_audit_rows
        if (row["human_stage_c_required"] == "YES"
            or row["human_critical_coverage_required"] == "YES")
    ]
    write_audit(audit_path, required_audit_rows)
    write_records(out_dir / "axis3_full_system_records.csv", all_records)
    write_report(out_dir / "axis3_full_system_report.md", summaries)
    (out_dir / "axis3_full_system_metrics.json").write_text(
        json.dumps({"runs": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    (out_dir / "axis3_full_system_audit_guide.md").write_text(
        "# Axis 3 human audit\n\n"
        f"The CSV contains only the {len(required_audit_rows)} records that need a "
        "human decision.\n\n"
        "- Fill `human_stage_c` with `PASS` or `FAIL` only when "
        "`human_stage_c_required=YES`.\n"
        "- `PASS` means the reply contains no contradicted or unsupported claim.\n"
        "- Fill `human_critical_coverage` with `PASS` or `FAIL` only when its "
        "required column is `YES`.\n"
        "- Critical coverage passes when every hour in `critical_event_hours` is "
        "mentioned explicitly or accurately covered by a stated range. Counts need "
        "not be repeated.\n"
        "- `literal_hours_found` is an aid only. Read the reply before deciding.\n"
        "- Use `audit_comment` only when the reason is not obvious.\n\n"
        "Save the CSV in place and rerun the same command. Existing decisions are "
        "preserved and the final pass^1/pass^5 values are recalculated.\n",
        encoding="utf-8")

    print((out_dir / "axis3_full_system_report.md").read_text(encoding="utf-8"))
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()

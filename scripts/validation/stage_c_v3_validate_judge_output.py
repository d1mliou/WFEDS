"""Validate a Stage-C v3 judge pass before any metric is computed from it.

The judge is a frozen instrument, not a trusted one. This checks that what came
back is complete, internally consistent and evidenced, and refuses the pass
otherwise. It decides nothing about faithfulness and changes no decision.

    python scripts/validation/stage_c_v3_validate_judge_output.py \\
        --bundle DIR --task RAW.json [--out REPORT.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

QUESTIONS = ("has_contradicted_information", "has_unsupported_information",
             "critical_events_complete", "required_disclosures_complete",
             "advice_grounded_and_caveated", "limitations_complete")
FORBIDDEN_TOKENS = ("exact_set", "ct_exact_set", "SSP", "atomic_claim",
                    "supported_claim_rate")


def expected_pass(d) -> bool:
    return (d["has_contradicted_information"] is False
            and d["has_unsupported_information"] is False
            and d["critical_events_complete"] is True
            and d["required_disclosures_complete"] is True
            and d["advice_grounded_and_caveated"] in ("true", "NOT_APPLICABLE")
            and d["limitations_complete"] is True)


def failed_questions(d) -> set:
    out = set()
    if d["has_contradicted_information"] is True:
        out.add("has_contradicted_information")
    if d["has_unsupported_information"] is True:
        out.add("has_unsupported_information")
    if d["critical_events_complete"] is False:
        out.add("critical_events_complete")
    if d["required_disclosures_complete"] is False:
        out.add("required_disclosures_complete")
    if d["advice_grounded_and_caveated"] == "false":
        out.add("advice_grounded_and_caveated")
    if d["limitations_complete"] is False:
        out.add("limitations_complete")
    return out


def _batches(node):
    """Yield every {"decisions": [...]} object reachable from a judge result.

    Three shapes occur in practice and all three are accepted, because the shape is
    an artefact of how the pass was orchestrated and says nothing about the
    judgements inside it:
      {"decisions": [...]}                      a single call
      {"b1": {"result": {"decisions": [...]}}}   batched, wrapped in a result key
      {"R0001": {"decisions": [...]}}            one call per record
    """
    if not isinstance(node, dict):
        return
    if "decisions" in node and isinstance(node["decisions"], list):
        yield node
        return
    for value in node.values():
        if isinstance(value, dict):
            yield from _batches(value.get("result", value))


def collect(task) -> dict:
    """Group every decision by record id, whatever the surrounding shape."""
    out = {}
    for b in _batches(task.get("result", task) if isinstance(task, dict) else task):
        for d in b.get("decisions", []):
            out.setdefault(d["record_id"], []).append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    B = Path(a.bundle)

    manifest = json.loads((B / "blind_manifest.json").read_text(encoding="utf-8"))
    expected = {r["record_id"] for r in manifest["records"]}
    task = json.loads(Path(a.task).read_text(encoding="utf-8"))
    got = collect(task)

    problems = []
    missing = sorted(expected - set(got))
    extra = sorted(set(got) - expected)
    dupes = sorted(k for k, v in got.items() if len(v) > 1)
    if missing:
        problems.append(f"missing {len(missing)} decisions, first {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} decisions for records that do not exist: {extra[:5]}")
    if dupes:
        problems.append(f"{len(dupes)} records decided more than once: {dupes[:5]}")

    decisions = {k: v[0] for k, v in got.items() if k in expected}
    bad_null, bad_pass, bad_evidence, bad_failures, bad_enum = [], [], [], [], []
    for rid, d in sorted(decisions.items()):
        if d["judgeable"] is False:
            if any(d[q] is not None for q in QUESTIONS) or d["stage_c_pass"] is not None:
                bad_null.append(rid)
            continue
        if any(d[q] is None for q in QUESTIONS):
            bad_null.append(rid)
            continue
        if d["advice_grounded_and_caveated"] not in ("true", "false", "NOT_APPLICABLE"):
            bad_enum.append(rid)
            continue
        if d["stage_c_pass"] != expected_pass(d):
            bad_pass.append(rid)
        want = failed_questions(d)
        have = {f["question"] for f in d.get("failures") or []}
        if want != have:
            bad_failures.append((rid, sorted(want), sorted(have)))
        for f in d.get("failures") or []:
            if not (f.get("payload_evidence") or "").strip():
                bad_evidence.append((rid, f["question"], "no payload evidence"))
            omission = f["question"] in ("critical_events_complete",
                                         "required_disclosures_complete",
                                         "limitations_complete")
            if not (f.get("exact_quote") or "").strip() and not omission:
                bad_evidence.append((rid, f["question"], "no quotation"))

    if bad_null:
        problems.append(f"{len(bad_null)} decisions whose null pattern is wrong: {bad_null[:5]}")
    if bad_enum:
        problems.append(f"{len(bad_enum)} decisions with an out-of-enum advice value: {bad_enum[:5]}")
    if bad_pass:
        problems.append(f"{len(bad_pass)} records whose stage_c_pass contradicts the "
                        f"conjunction: {bad_pass[:5]}")
    if bad_failures:
        problems.append(f"{len(bad_failures)} records whose failure list does not match the "
                        f"answers: {bad_failures[:3]}")
    if bad_evidence:
        problems.append(f"{len(bad_evidence)} failures without usable evidence: "
                        f"{bad_evidence[:5]}")

    blob = json.dumps(task, ensure_ascii=False)
    leaked = [t for t in FORBIDDEN_TOKENS if t.lower() in blob.lower()]
    if leaked:
        problems.append(f"retired machinery named in the judge output: {leaked}")

    summary = {
        "records_expected": len(expected),
        "records_decided": len(decisions),
        "judgeable": sum(1 for d in decisions.values() if d["judgeable"]),
        "unjudgeable_reported_by_judge": sum(1 for d in decisions.values() if not d["judgeable"]),
        "stage_c_pass": sum(1 for d in decisions.values() if d.get("stage_c_pass") is True),
        "failures_by_question": dict(Counter(
            f["question"] for d in decisions.values() for f in d.get("failures") or [])),
        "uncalibrated_fuels_mentioned": sum(
            1 for d in decisions.values() if d.get("uncalibrated_fuels_mentioned") is True),
        "initial_state_conveyed": sum(
            1 for d in decisions.values() if d.get("initial_state_conveyed") is True),
        "problems": problems,
    }
    if a.out:
        Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                               encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print()
    print(f"VALIDATION: {'PASSED' if not problems else 'FAILED'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

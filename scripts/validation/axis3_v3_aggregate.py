"""Final Axis-3 v3 aggregation. Deterministic; runs no model and no judge.

Composes the three stages into the published tables:

    Stage A   deterministic argument extraction, over the whole 53-case suite
    Stage B   deterministic tool invocation, over the whole 53-case suite
    Stage C   the frozen LLM judge's record-level checklist, over the 33-case
              Stage-C set, where a record with no judgeable narration of the gold
              payload is NOT_REACHED
    overall   Stage A AND Stage B AND Stage C, on the 33-case set

`exact_set` appears nowhere: not as a gate, not as a diagnostic, not in any output.
The structured arm's own diagnostics are schema validity, field-level accuracy and
critical-transition precision and recall, reported beside the comparison and never
inside it.

    python scripts/validation/axis3_v3_aggregate.py \\
        --runs RUNS.json --exports DIR --bundle DIR --task RAW.json --out DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOT_SEED = 20260922
BOOT_N = 20000
QUESTIONS = ("has_contradicted_information", "has_unsupported_information",
             "critical_events_complete", "required_disclosures_complete",
             "advice_grounded_and_caveated", "limitations_complete")
BANNED = ("exact_set", "ct_exact_set", "ssp")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_sign_test(pairs):
    """Two-sided exact sign test on discordant case pairs. The case is the unit;
    repetitions of a case never become independent observations."""
    pos = sum(1 for f, s in pairs if s and not f)
    neg = sum(1 for f, s in pairs if f and not s)
    n = pos + neg
    if n == 0:
        return None, 0, 0
    k = min(pos, neg)
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)), 4), pos, neg


def case_cluster_bootstrap(by_case, alpha=0.05):
    """Percentile interval for a rate, resampling whole cases and never rows."""
    cases = sorted(by_case)
    if not cases:
        return None, None
    rng = random.Random(BOOT_SEED)
    draws = []
    for _ in range(BOOT_N):
        num = den = 0
        for _ in cases:
            c = rng.choice(cases)
            num += sum(by_case[c])
            den += len(by_case[c])
        draws.append(num / den if den else 0.0)
    draws.sort()
    return (round(draws[int(alpha / 2 * len(draws))], 4),
            round(draws[int((1 - alpha / 2) * len(draws)) - 1], 4))


def _batches(node):
    """Yield every {"decisions": [...]} object reachable from a judge result.

    The orchestration shape varies with how the pass was run and says nothing
    about the judgements inside it, so all of these are accepted:
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


def rate(n, d):
    return round(100.0 * n / d, 1) if d else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--exports", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold", default=str(HERE / "stage_c_v3_gold.json"))
    a = ap.parse_args(argv)
    E, B, OUT = Path(a.exports), Path(a.bundle), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))["cases"]
    stage_c_ids = set(gold)
    runs = json.loads(Path(a.runs).read_text(encoding="utf-8"))
    key = json.loads((B / "KEY_DO_NOT_SHIP.json").read_text(encoding="utf-8"))
    mapping, not_reached = key["mapping"], key["not_reached"]

    raw = json.loads(Path(a.task).read_text(encoding="utf-8"))
    decisions = {}
    for b in _batches(raw.get("result", raw) if isinstance(raw, dict) else raw):
        for d in b.get("decisions", []):
            decisions[d["record_id"]] = d

    verdict = {}          # (arm, model, case, rep) -> stage C record
    for rid, meta in mapping.items():
        d = decisions.get(rid)
        k = (meta["arm"], meta["model"], meta["case_id"], int(meta["rep"]))
        if d is None or d.get("judgeable") is False:
            verdict[k] = {"state": "NOT_REACHED", "reason": "judge returned unjudgeable"}
            continue
        verdict[k] = {"state": "PASS" if d["stage_c_pass"] else "FAIL",
                      "items": {q: d[q] for q in QUESTIONS},
                      "failures": d.get("failures") or [],
                      "uncalibrated": d.get("uncalibrated_fuels_mentioned"),
                      "initial_state": d.get("initial_state_conveyed")}
    for nr in not_reached:
        verdict[(nr["arm"], nr["model"], nr["case_id"], int(nr["rep"]))] = {
            "state": "NOT_REACHED", "reason": nr["reason"]}

    # ------------------------------------------------------------ record table
    rows, problems = [], []
    struct_rows = {}
    for run in runs:
        arm, model = run["arm"], run["model"]
        p = E / run["folder"] / "validation/3_agent/agent_eval_raw_scored.jsonl"
        if not p.exists():
            p = E / run["folder"] / "validation/3_agent/agent_eval_raw.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = (arm, model, r["case_id"], int(r["rep"]))
            in_c = r["case_id"] in stage_c_ids
            v = verdict.get(k, {"state": "NOT_REACHED", "reason": "no record built"}) \
                if in_c else {"state": "NOT_IN_SCOPE"}
            if arm == "structured":
                struct_rows[k] = r
            rows.append({
                "arm": arm, "model": model, "case_id": r["case_id"], "rep": int(r["rep"]),
                "uid": f"{r['case_id']}#{r['rep']}", "category": r.get("category", ""),
                "stage_a": bool(r["stage_a_ok"]), "stage_b": bool(r["stage_b_ok"]),
                "in_stage_c_set": in_c,
                "stage_c_state": v["state"],
                "stage_c_pass": v["state"] == "PASS",
                **{q: (v.get("items") or {}).get(q, "") for q in QUESTIONS},
                "stage_c_not_reached_reason": v.get("reason", ""),
                "overall_pass": bool(in_c and r["stage_a_ok"] and r["stage_b_ok"]
                                     and v["state"] == "PASS"),
                "failed_stages": ";".join(
                    s for s, ok in (("A", r["stage_a_ok"]), ("B", r["stage_b_ok"]),
                                    ("C", v["state"] == "PASS" or not in_c)) if not ok),
            })
    with (OUT / "axis3_v3_record_level.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["arm"], x["model"], x["case_id"], x["rep"])))

    # --------------------------------------------------------------- metrics
    def cell(arm, model):
        return [r for r in rows if r["arm"] == arm and r["model"] == model]

    cells, per_category, ssp_like = {}, {}, {}
    for run in runs:
        arm, model, name = run["arm"], run["model"], f"{run['arm']}/{run['model']}"
        rs = cell(arm, model)
        cs = [r for r in rs if r["in_stage_c_set"]]
        judged = [r for r in cs if r["stage_c_state"] in ("PASS", "FAIL")]
        by_case_c = defaultdict(list)
        by_case_all = defaultdict(list)
        for r in cs:
            by_case_c[r["case_id"]].append(1 if r["stage_c_pass"] else 0)
            by_case_all[r["case_id"]].append(1 if r["overall_pass"] else 0)
        cells[name] = {
            "stage_a": {"records": len(rs), "pass": sum(1 for r in rs if r["stage_a"]),
                        "pct": rate(sum(1 for r in rs if r["stage_a"]), len(rs))},
            "stage_b": {"records": len(rs), "pass": sum(1 for r in rs if r["stage_b"]),
                        "pct": rate(sum(1 for r in rs if r["stage_b"]), len(rs))},
            "stage_c": {
                "records_in_set": len(cs),
                "judged": len(judged),
                "not_reached": len(cs) - len(judged),
                "not_reached_by_reason": dict(Counter(
                    r["stage_c_not_reached_reason"] for r in cs
                    if r["stage_c_state"] == "NOT_REACHED")),
                "pass": sum(1 for r in cs if r["stage_c_pass"]),
                "pct_of_judged": rate(sum(1 for r in cs if r["stage_c_pass"]), len(judged)),
                "pct_of_set": rate(sum(1 for r in cs if r["stage_c_pass"]), len(cs)),
                "pass1": sum(1 for r in cs if r["stage_c_pass"]),
                "pass1_ci": case_cluster_bootstrap(by_case_c),
                "pass5": sum(1 for c, v in by_case_c.items() if all(v)),
                "checklist": {q: dict(Counter(str(r[q]) for r in judged)) for q in QUESTIONS},
                "uncalibrated_fuels_mentioned": sum(
                    1 for k2, v in verdict.items()
                    if k2[0] == arm and k2[1] == model and v.get("uncalibrated") is True),
                "initial_state_conveyed": sum(
                    1 for k2, v in verdict.items()
                    if k2[0] == arm and k2[1] == model and v.get("initial_state") is True),
                "descriptive_only": ["uncalibrated_fuels_mentioned",
                                     "initial_state_conveyed"],
            },
            "overall": {
                "pass1": sum(1 for r in cs if r["overall_pass"]),
                "pass1_pct": rate(sum(1 for r in cs if r["overall_pass"]), len(cs)),
                "pass1_ci": case_cluster_bootstrap(by_case_all),
                "pass5": sum(1 for c, v in by_case_all.items() if all(v)),
                "cases": len(by_case_all),
                "per_case": {c: "".join("P" if x else "F" for x in v)
                             for c, v in sorted(by_case_all.items())},
            },
        }
        per_category[name] = {}
        for cat in sorted({r["category"] for r in cs}):
            sub = [r for r in cs if r["category"] == cat]
            per_category[name][cat] = {
                "cases": len({r["case_id"] for r in sub}), "records": len(sub),
                "stage_c_pass": sum(1 for r in sub if r["stage_c_pass"]),
                "overall_pass": sum(1 for r in sub if r["overall_pass"]),
            }

    # ------------------------------------------- structured-only diagnostics
    diagnostics = {}
    for run in [r for r in runs if r["arm"] == "structured"]:
        model = run["model"]
        rs = [v for k, v in struct_rows.items() if k[1] == model and k[2] in stage_c_ids]
        if not rs or "schema_conformant" not in rs[0]:
            diagnostics[model] = {"note": "no structured validator fields on these rows"}
            continue
        matched = sum(r.get("ct_matched", 0) for r in rs)
        gt = sum(r.get("ct_n_gt", 0) for r in rs)
        emitted = sum(r.get("ct_n_emitted", 0) for r in rs)
        slots_clean = sum(1 for r in rs if not (r.get("ct_value_mismatches") or []))
        diagnostics[model] = {
            "records": len(rs),
            "schema_valid": sum(1 for r in rs if r.get("schema_conformant")),
            "schema_valid_pct": rate(sum(1 for r in rs if r.get("schema_conformant")), len(rs)),
            "field_level_accuracy_records_with_no_slot_mismatch": slots_clean,
            "field_level_accuracy_pct": rate(slots_clean, len(rs)),
            "critical_transition_recall_pct": rate(matched, gt),
            "critical_transition_precision_pct": rate(matched, emitted),
            "matched_transitions": matched, "ground_truth_transitions": gt,
            "emitted_transitions": emitted,
            "note": "diagnostics only; none of these is a gate and none enters the "
                    "free-versus-structured comparison",
        }

    # -------------------------------------------------------------- comparison
    comparison = {}
    models = sorted({r["model"] for r in runs})
    pooled_pairs = []
    for model in models:
        f = [r for r in rows if r["arm"] == "free" and r["model"] == model and r["in_stage_c_set"]]
        s = [r for r in rows if r["arm"] == "structured" and r["model"] == model and r["in_stage_c_set"]]
        if not f or not s:
            continue
        pairs = []
        for cid in sorted({r["case_id"] for r in f}):
            fp = all(r["overall_pass"] for r in f if r["case_id"] == cid)
            sp = all(r["overall_pass"] for r in s if r["case_id"] == cid)
            pairs.append((fp, sp))
            pooled_pairs.append((fp, sp))
        p, pos, neg = exact_sign_test(pairs)
        comparison[model] = {
            "overall_pass1_free": sum(1 for r in f if r["overall_pass"]),
            "overall_pass1_structured": sum(1 for r in s if r["overall_pass"]),
            "overall_pass5_free": sum(1 for a, _ in pairs if a),
            "overall_pass5_structured": sum(1 for _, b in pairs if b),
            "discordant_structured_only": pos, "discordant_free_only": neg,
            "exact_sign_test_p_two_sided": p,
            "checklist_failures_free": dict(Counter(
                q for r in f for q in QUESTIONS if r[q] in (True, "True", "false")
                and q in ("has_contradicted_information", "has_unsupported_information"))),
        }
    if pooled_pairs:
        p, pos, neg = exact_sign_test(pooled_pairs)
        comparison["pooled_case_level"] = {
            "pairs": len(pooled_pairs), "discordant_structured_only": pos,
            "discordant_free_only": neg, "exact_sign_test_p_two_sided": p,
            "note": "the case is the unit; the only pooling the design permits",
        }

    out = {"cells": cells, "per_category": per_category,
           "structured_diagnostics": diagnostics, "comparison": comparison,
           "problems": problems,
           "provenance": {
               "gold_sha256": sha(a.gold),
               "rubric_sha256": sha(HERE / "stage_c_v3_rubric.md"),
               "judge_prompt_sha256": sha(HERE / "stage_c_v3_judge_prompt.txt"),
               "judge_schema_sha256": sha(HERE / "stage_c_v3_judge_schema.json"),
               "bundle_manifest_sha256": sha(B / "blind_manifest.json"),
               "aggregator_sha256": sha(__file__),
               "bootstrap": {"seed": BOOT_SEED, "draws": BOOT_N,
                             "unit": "the case; repetitions never separated"},
           }}
    blob = json.dumps(out, ensure_ascii=False).lower()
    leaked = [t for t in BANNED if t in blob]
    if leaked:
        out["problems"].append(f"retired machinery present in the output: {leaked}")
    (OUT / "axis3_v3_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_category"},
                     ensure_ascii=False, indent=1))
    print()
    print(f"PROBLEMS: {out['problems'] if out['problems'] else 'none'}")
    return 1 if out["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

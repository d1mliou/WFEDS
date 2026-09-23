"""Repeatability audit for the Stage-C v3 judge.

The judge is a single instrument, so its run-to-run stability is a property of the
result and has to be measured rather than assumed. The pilot measured it on one
record by accident, at four agreements in five judgements. This measures it on a
pre-declared sample of the real run.

Two steps, and the order is the point.

    select   BEFORE any judging, lock a sample of judgeable records with a
             published seed, balanced as evenly as the cells allow. Writing this
             file after the judgements existed would make the sample a choice.

    compare  AFTER both passes, report agreement and Cohen's kappa on the overall
             Stage-C verdict and agreement on each of the six questions.

The second judgement never replaces the first. There is no majority vote, no
tie-break and no third pass: the primary labels stand exactly as the main pass
wrote them, and this audit only says how reproducible they were.

    python scripts/validation/stage_c_v3_repeatability_audit.py select \\
        --bundle DIR --out SAMPLE.json
    python scripts/validation/stage_c_v3_repeatability_audit.py compare \\
        --sample SAMPLE.json --primary RAW.json --second RAW.json --out REPORT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = "axis3-v3-repeatability-20260923"
DEFAULT_N = 66
QUESTIONS = ("has_contradicted_information", "has_unsupported_information",
             "critical_events_complete", "required_disclosures_complete",
             "advice_grounded_and_caveated", "limitations_complete")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _batches(node):
    """Every {"decisions": [...]} reachable from a judge result, whatever the shape."""
    if not isinstance(node, dict):
        return
    if "decisions" in node and isinstance(node["decisions"], list):
        yield node
        return
    for value in node.values():
        if isinstance(value, dict):
            yield from _batches(value.get("result", value))


def load_decisions(path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for b in _batches(raw.get("result", raw) if isinstance(raw, dict) else raw):
        for d in b.get("decisions", []):
            out[d["record_id"]] = d
    return out


def cohens_kappa(pairs):
    """Two-rater, two-category kappa on the overall verdict."""
    n = len(pairs)
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    d = sum(1 for x, y in pairs if not x and not y)
    table = {"both_pass": a, "primary_pass_second_fail": b,
             "primary_fail_second_pass": c, "both_fail": d}
    if n == 0:
        return None, "no pairs", table
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    if abs(1 - pe) < 1e-12:
        return None, ("undefined: both passes put essentially everything in one "
                      "category, so chance agreement is already 1"), table
    return round((po - pe) / (1 - pe), 4), "", table


# --------------------------------------------------------------------- select
def select(argv) -> int:
    ap = argparse.ArgumentParser(prog="select")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    a = ap.parse_args(argv)
    B = Path(a.bundle)

    key = json.loads((B / "KEY_DO_NOT_SHIP.json").read_text(encoding="utf-8"))
    manifest = json.loads((B / "blind_manifest.json").read_text(encoding="utf-8"))
    judgeable = {r["record_id"] for r in manifest["records"]}
    mapping = {k: v for k, v in key["mapping"].items() if k in judgeable}

    cells = defaultdict(list)
    for rid, meta in sorted(mapping.items()):
        cells[(meta["arm"], meta["model"])].append(rid)

    order = sorted(cells)
    base, extra = divmod(a.n, len(order))
    quota = {cell: base + (1 if i < extra else 0) for i, cell in enumerate(order)}

    rng = random.Random(SEED)
    chosen, shortfall = [], {}
    for cell in order:
        pool = sorted(cells[cell])
        want = quota[cell]
        if len(pool) < want:
            shortfall[str(cell)] = {"available": len(pool), "wanted": want}
            want = len(pool)
        chosen += rng.sample(pool, want)
    chosen = sorted(chosen)

    doc = {
        "seed": SEED,
        "declared_before_judging": True,
        "n_requested": a.n,
        "n_selected": len(chosen),
        "quota_per_cell": {f"{c[0]}/{c[1]}": quota[c] for c in order},
        "selected_per_cell": {f"{c[0]}/{c[1]}":
                              sum(1 for r in chosen if (mapping[r]["arm"],
                                                        mapping[r]["model"]) == c)
                              for c in order},
        "shortfall": shortfall,
        "judgeable_pool": len(mapping),
        "bundle_manifest_sha256": sha(B / "blind_manifest.json"),
        "selector_sha256": sha(__file__),
        "rule": ("the second judgement never replaces the first, is never used for a "
                 "majority vote, and changes no primary label"),
        "record_ids": chosen,
    }
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                           encoding="utf-8", newline="\n")
    print(f"judgeable pool     : {len(mapping)}")
    print(f"selected           : {len(chosen)} of {a.n} requested")
    for c in order:
        print(f"   {c[0]:10s} {c[1]:6s} quota {quota[c]:3d}  pool {len(cells[c]):3d}")
    print(f"shortfall          : {shortfall or 'none'}")
    print(f"seed               : {SEED}")
    print(f"out                : {a.out}")
    return 1 if shortfall else 0


# -------------------------------------------------------------------- compare
def compare(argv) -> int:
    ap = argparse.ArgumentParser(prog="compare")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    sample = json.loads(Path(a.sample).read_text(encoding="utf-8"))
    ids = sample["record_ids"]
    p1, p2 = load_decisions(a.primary), load_decisions(a.second)

    problems = []
    missing1 = [r for r in ids if r not in p1]
    missing2 = [r for r in ids if r not in p2]
    if missing1:
        problems.append(f"{len(missing1)} sampled records absent from the primary pass")
    if missing2:
        problems.append(f"{len(missing2)} sampled records absent from the second pass")
    both = [r for r in ids if r in p1 and r in p2]

    pairs = [(bool(p1[r]["stage_c_pass"]), bool(p2[r]["stage_c_pass"])) for r in both]
    agree = sum(1 for x, y in pairs if x == y)
    kappa, why, table = cohens_kappa(pairs)

    per_question = {}
    for q in QUESTIONS:
        same = sum(1 for r in both if p1[r][q] == p2[r][q])
        per_question[q] = {
            "n": len(both), "agree": same,
            "agreement_pct": round(100.0 * same / len(both), 1) if both else None,
            "primary_distribution": dict(Counter(str(p1[r][q]) for r in both)),
            "second_distribution": dict(Counter(str(p2[r][q]) for r in both)),
        }

    flips = [{"record_id": r,
              "primary_stage_c_pass": p1[r]["stage_c_pass"],
              "second_stage_c_pass": p2[r]["stage_c_pass"],
              "questions_that_differ": [q for q in QUESTIONS if p1[r][q] != p2[r][q]]}
             for r in both if p1[r]["stage_c_pass"] != p2[r]["stage_c_pass"]]

    report = {
        "seed": sample["seed"],
        "n_pairs": len(both),
        "stage_c_pass": {
            "agree": agree,
            "agreement_pct": round(100.0 * agree / len(both), 1) if both else None,
            "cohens_kappa": kappa,
            "kappa_note": why,
            "table": table,
        },
        "per_question_agreement": per_question,
        "records_whose_overall_verdict_flipped": flips,
        "rule": ("the primary labels stand unchanged; this audit measures "
                 "reproducibility and decides nothing"),
        "problems": problems,
    }
    if a.out:
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k != "records_whose_overall_verdict_flipped"},
                     ensure_ascii=False, indent=1))
    print(f"\noverall verdict flipped on {len(flips)} of {len(both)} records")
    print(f"PROBLEMS: {problems if problems else 'none'}")
    return 1 if problems else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("select", "compare"):
        print(__doc__)
        return 2
    return (select if argv[0] == "select" else compare)(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

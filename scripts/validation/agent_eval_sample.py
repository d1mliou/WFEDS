"""Axis 3, Stage C: build the human labelling worksheet.

Stage C (narration faithfulness) is the one part of axis 3 that is NOT scored
automatically, because labelling a claim requires judgement and an LLM judging
an LLM is the circularity this axis exists to expose (Decision log 2026-08-27).
This script produces the material for the human pass and nothing else - it
assigns no labels.

What it does:
  1. picks a declared, reproducible sample of repetitions (fixed seed);
  2. splits each reply into CANDIDATE atomic claims on sentence and bullet
     boundaries - a mechanical first pass. FActScore reports that humans had to
     SPLIT 18% and MERGE 34% of model-proposed atomic facts, so the worksheet
     invites the labeller to do exactly that rather than pretending the split is
     authoritative;
  3. prints, beside each reply, every number the tool actually returned, so a
     claim can be checked without opening another file;
  4. emits a markdown worksheet with an empty label column.

The five labels (declared 2026-07-19):
  SUPPORTED           the claim follows from the tool output
  CONTRADICTED        the tool output says otherwise
  NOT_IN_RESULT       nothing in the tool output supports it (the hallucination case)
  ADVISORY_INFERENCE  permitted judgement: flagged as advisory, grounded in named
                      results, and accompanied by the limits statement
  LIMITATION_STATEMENT  part of the mandatory limits/uncertainty framing

Run:
    python scripts/validation/agent_eval_sample.py <raw.jsonl> [--n 4] [--seed 0]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import random
import re
from pathlib import Path

# Cases whose whole point is the narration - the sample is drawn from these
# first, then topped up from any other case that produced a full narration.
NARRATION_FIRST = ("narration_faithfulness", "barely_spreading",
                   "pin_merge_disclosure", "policy_absolute_recommendation")

_SPLIT = re.compile(r"(?<=[.!;])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def atomic_candidates(reply):
    """Mechanical first-pass split into candidate claims. Deliberately crude:
    the labeller is expected to merge and split."""
    out = []
    for chunk in _SPLIT.split(reply or ""):
        c = _BULLET.sub("", chunk).strip()
        c = re.sub(r"\s+", " ", c)
        if len(c) >= 12:
            out.append(c)
    return out


def tool_numbers(compact):
    """The numbers the tool actually returned, grouped so a human can scan them."""
    if not compact:
        return "(the tool was not called, or returned an error)"
    lines = []
    fh = compact.get("final_hour", {})
    if fh:
        lines.append(f"* **final hour {fh.get('period')}**: fire {fh.get('fire_km2')} km2, "
                     f"at_risk {fh.get('at_risk')}, population {fh.get('population')}, "
                     f"routed {fh.get('routed')}, cut_off {fh.get('cut_off')}, "
                     f"impacted {fh.get('impacted')}, edges_removed {fh.get('edges_removed')}, "
                     f"longest_route_km {fh.get('longest_route_km')}")
    if compact.get("final_front_class4_pct") is not None:
        lines.append(f"* **final_front_class4_pct**: {compact['final_front_class4_pct']}")
    inp = compact.get("inputs", {})
    lines.append(f"* **inputs**: mode={inp.get('mode')}, window_km={inp.get('window_km')}, "
                 f"horizon_h={inp.get('horizon_h')}, start={inp.get('start_time_utc')}, "
                 f"weather={inp.get('weather_source')}, "
                 f"n_fronts_detected={inp.get('n_fronts_detected')}")
    ch = compact.get("change_hours", [])
    if ch:
        lines.append(f"* **change_hours** ({len(ch)} rows):")
        for r in ch:
            lines.append(f"    * h{r.get('period')}: fire {r.get('fire_km2')} km2, "
                         f"at_risk {r.get('at_risk')}, pop {r.get('population')}, "
                         f"routed {r.get('routed')}, cut_off {r.get('cut_off')}, "
                         f"impacted {r.get('impacted')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="agent_eval_raw.jsonl from a scored run")
    ap.add_argument("--n", type=int, default=4, help="replies to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    raw = Path(a.raw)
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    narrated = [r for r in rows if r["n_tool_calls"] > 0 and len(r["reply"]) > 200]
    if not narrated:
        sys.exit("ERROR: no narrated replies in that run")

    rnd = random.Random(a.seed)
    pool_first = [r for r in narrated if r["category"] in NARRATION_FIRST]
    pool_rest = [r for r in narrated if r["category"] not in NARRATION_FIRST]
    rnd.shuffle(pool_first)
    rnd.shuffle(pool_rest)
    picked, seen = [], set()
    for r in pool_first + pool_rest:
        if r["case_id"] in seen:               # one repetition per case, for spread
            continue
        seen.add(r["case_id"])
        picked.append(r)
        if len(picked) >= a.n:
            break

    out = Path(a.out) if a.out else raw.parent / "agent_eval_stage_c_worksheet.md"
    total = 0
    lines = [
        "# Axis 3, Stage C - narration faithfulness: labelling worksheet", "",
        f"Sample: **{len(picked)} replies**, drawn with seed {a.seed} from "
        f"`{raw.parent.name}`, one repetition per case, narration-focused "
        "categories first. Declared and reproducible.", "",
        "## How to fill this in", "",
        "For each claim write ONE label in the last column:", "",
        "| label | meaning |",
        "|---|---|",
        "| `SUPPORTED` | follows from the tool output below |",
        "| `CONTRADICTED` | the tool output says otherwise |",
        "| `NOT_IN_RESULT` | nothing in the tool output supports it (hallucination) |",
        "| `ADVISORY_INFERENCE` | permitted judgement: advisory, grounded in named results, with the limits statement |",
        "| `LIMITATION_STATEMENT` | part of the mandatory limits/uncertainty framing |", "",
        "**The split below is mechanical and is not authoritative.** If a row holds "
        "two claims, split it; if two rows are one claim, merge them and label once. "
        "FActScore reports human labellers split 18% and merged 34% of "
        "model-proposed atomic facts - doing the same here is expected, not a "
        "deviation.", "",
        "**Claude will label the same sample independently afterwards and the "
        "agreement rate will be reported.** Your labels are the authoritative ones; "
        "Claude's are the second opinion, and can never override them.", "",
        "---", "",
    ]
    for i, r in enumerate(picked, 1):
        compact = None
        for o in r["tool_outputs"]:
            if not o.startswith(("INPUT ERROR", "PIPELINE ERROR")):
                try:
                    compact = json.loads(o)
                except json.JSONDecodeError:
                    pass
        claims = atomic_candidates(r["reply"])
        total += len(claims)
        lines += [
            f"## Reply {i} - case `{r['case_id']}` ({r['category']}), repetition {r['rep']}",
            "", f"**The analyst asked:** {r['user_text']}", "",
            f"**Fixture:** `{r['fixture']}`", "",
            "### What the tool actually returned", "", tool_numbers(compact), "",
            "### The agent's full reply", "", "```", r["reply"], "```", "",
            "### Claims to label", "",
            "| # | claim | label |", "|---|---|---|",
        ]
        for j, c in enumerate(claims, 1):
            safe = c.replace("|", "\\|")
            lines.append(f"| {i}.{j} | {safe} | |")
        lines += ["", "---", ""]

    lines += [
        "## When you are done", "",
        f"There are **{total} candidate claims** across {len(picked)} replies. "
        "Save this file with the label column filled in, then Claude labels the "
        "same set independently and the report states the agreement rate, the "
        "disagreements, and who decided each one.", "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(picked)} replies, {total} candidate claims")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

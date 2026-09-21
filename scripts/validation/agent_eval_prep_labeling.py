"""Prepare the FULL Stage-C labelling dataset (protocol changed 2026-08-27:
Claude labels the complete narrated set at full effort, user audits afterward -
see Decision log). No LLM calls here - pure data preparation from the saved
raw.jsonl, so it costs nothing and can be re-run freely.

For EVERY repetition, extracts full input (case, user text, pins, fixture) and
full output (reply, raw tool outputs). For NARRATED repetitions (tool called,
a compact result returned, an interpretive reply produced) it also renders the
tool's numbers in a labeller-friendly form and splits the reply into candidate
atomic claims (mechanical first pass, per FActScore - not authoritative).

Splits the narrated set into fixed-size batches for parallel labelling agents.

Run:
    python scripts/validation/agent_eval_prep_labeling.py <raw.jsonl> [--batch-size 15]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_eval_sample import atomic_candidates, tool_numbers  # noqa: E402

NARRATION_MIN_CHARS = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--categories", default=None,
                    help="comma-separated case categories to batch (default: all). "
                         "Used for a targeted second-model pass, where labelling "
                         "the whole set again is not affordable - the selection "
                         "must then be declared with the results.")
    a = ap.parse_args()
    keep = {c.strip() for c in a.categories.split(",")} if a.categories else None

    raw = Path(a.raw)
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    out_dir = Path(a.out_dir) if a.out_dir else raw.parent / "stage_c_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    full, narrated = [], []
    for r in rows:
        compact = None
        for o in r["tool_outputs"]:
            if not o.startswith(("INPUT ERROR", "PIPELINE ERROR")):
                try:
                    compact = json.loads(o)
                except json.JSONDecodeError:
                    pass
        is_narrated = (r["n_tool_calls"] > 0 and compact is not None
                       and len(r["reply"]) >= NARRATION_MIN_CHARS)
        item = {
            "uid": f"{r['case_id']}#{r['rep']}",
            "case_id": r["case_id"], "category": r["category"], "rep": r["rep"],
            "fixture": r["fixture"], "user_text": r["user_text"], "pins": r["pins"],
            "n_tool_calls": r["n_tool_calls"], "reply": r["reply"],
            "tool_outputs": r["tool_outputs"], "error": r["error"],
            "is_narrated": is_narrated,
        }
        full.append(item)
        if keep is not None and r["category"] not in keep:
            continue
        if is_narrated:
            item2 = dict(item)
            item2["tool_numbers_md"] = tool_numbers(compact)
            item2["candidate_claims"] = atomic_candidates(r["reply"])
            narrated.append(item2)

    (out_dir / "all_repetitions_full.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in full), encoding="utf-8")

    batches = [narrated[i:i + a.batch_size] for i in range(0, len(narrated), a.batch_size)]
    for bi, batch in enumerate(batches, 1):
        (out_dir / f"batch_{bi:02d}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")

    n_claims = sum(len(x["candidate_claims"]) for x in narrated)
    print(f"{len(full)} total repetitions, {len(narrated)} narrated "
          f"(>= {NARRATION_MIN_CHARS} chars), {n_claims} candidate claims")
    print(f"{len(batches)} batches of up to {a.batch_size} narrated repetitions each")
    print(f"Saved -> {out_dir}")
    for i in range(1, len(batches) + 1):
        print(f"  batch_{i:02d}.json")


if __name__ == "__main__":
    main()

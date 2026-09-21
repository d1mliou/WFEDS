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


def judged_surface(row):
    """Edit E1 of the pre-registration: the text the judge will actually see.

    WHY this is not `row["reply"]`. On the structured arm the reply is a JSON
    envelope, and an envelope clears the 120-character floor even when
    `interpretation` is empty or the object never validated, so batching on the
    raw reply would send the judge records that section 2.4 already calls
    deterministic FAILs, and would decompose Latin key names into atomic claims.
    The filter has to be arm-equivalent: same floor, same kind of text.

    Three sources in order. A row written after the contract landed carries the
    surface `agent_eval.py` computed at run time, and reusing it guarantees the
    batch is cut from the same string the run scored. A free-arm row's surface
    IS its reply. Only a structured row with no stored surface falls through to
    `agent_eval._narrative_surface`, imported lazily because importing
    `agent_eval` resolves the OneDrive data directory and prep must stay usable
    on a machine that has none.
    """
    stored = row.get("narrative_surface")
    if isinstance(stored, str):
        return stored
    reply = row.get("reply") or ""
    if (row.get("output_contract") or "free") != "structured":
        return reply
    from agent_eval import _narrative_surface  # noqa: E402  lazy, see docstring
    return _narrative_surface(reply, "structured")


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
        # E1: the floor and the claim split both read the judged surface.
        surface = judged_surface(r)
        is_narrated = (r["n_tool_calls"] > 0 and compact is not None
                       and len(surface) >= NARRATION_MIN_CHARS)
        item = {
            "uid": f"{r['case_id']}#{r['rep']}",
            "case_id": r["case_id"], "category": r["category"], "rep": r["rep"],
            "fixture": r["fixture"], "user_text": r["user_text"], "pins": r["pins"],
            "n_tool_calls": r["n_tool_calls"], "reply": r["reply"],
            "arm": r.get("output_contract") or "free",
            "judged_surface": surface,
            "tool_outputs": r["tool_outputs"], "error": r["error"],
            "is_narrated": is_narrated,
        }
        full.append(item)
        if keep is not None and r["category"] not in keep:
            continue
        if is_narrated:
            item2 = dict(item)
            item2["tool_numbers_md"] = tool_numbers(compact)
            item2["candidate_claims"] = atomic_candidates(surface)
            narrated.append(item2)

    (out_dir / "all_repetitions_full.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in full), encoding="utf-8")

    batches = [narrated[i:i + a.batch_size] for i in range(0, len(narrated), a.batch_size)]
    for bi, batch in enumerate(batches, 1):
        (out_dir / f"batch_{bi:02d}.json").write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")

    # E2: the split that was actually produced, recorded rather than assumed.
    # A partial last batch (the stored terra run went 11,11,11,11,11,10) is a
    # fact about the judge call count and must not have to be recounted later
    # from the files.
    (out_dir / "batching_provenance.json").write_text(json.dumps({
        "source_raw": str(raw),
        "arms_seen": sorted({x["arm"] for x in full}),
        "batch_size_requested": a.batch_size,
        "categories": sorted(keep) if keep else "all",
        "narration_min_chars": NARRATION_MIN_CHARS,
        "n_repetitions": len(full),
        "n_narrated": len(narrated),
        "batch_sizes": [len(b) for b in batches],
        "batch_files": [f"batch_{i:02d}.json" for i in range(1, len(batches) + 1)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    n_claims = sum(len(x["candidate_claims"]) for x in narrated)
    print(f"{len(full)} total repetitions, {len(narrated)} narrated "
          f"(>= {NARRATION_MIN_CHARS} chars), {n_claims} candidate claims")
    print(f"{len(batches)} batches of up to {a.batch_size} narrated repetitions each")
    print(f"Saved -> {out_dir}")
    for i in range(1, len(batches) + 1):
        print(f"  batch_{i:02d}.json")


if __name__ == "__main__":
    main()

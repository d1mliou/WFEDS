"""Post-run sanity check for a structured Axis-3 run. Reads only, scores nothing.

Run it immediately after each Phase-4 command. It answers the four questions that
decide whether the run is usable at all, before any judge time is spent on it:
how many repetitions landed, which arm the rows say they are, what the provider
actually served, and whether the contract attached where it was supposed to
(false on the tool call, true on the final call).

    python scripts/validation/check_structured_run.py <out-dir> [<out-dir> ...]
"""
import json
import sys
from pathlib import Path

def _ok(r):
    try:
        return isinstance(json.loads(r["reply"]), dict)
    except Exception:
        return False


for d in sys.argv[1:]:
    raw = Path(d) / "agent_eval_raw.jsonl"
    if not raw.exists():
        print(f"{d}: MISSING agent_eval_raw.jsonl")
        continue
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    met = json.loads((Path(d) / "agent_eval_metrics.json").read_text(encoding="utf-8"))
    pattern = [[c.get("contract_attached") for c in (r.get("provider_meta") or [])] for r in rows]
    print(f"{Path(d).parents[1].name}")
    print(f"  rows            {len(rows)}   (expected 65)")
    print(f"  arm             {sorted({r.get('output_contract') for r in rows})}")
    print(f"  served model    {met.get('provider_models_seen')}")
    print(f"  prompt sha      {met.get('system_prompt_sha256')}   (expected f005ab5e76d6)")
    print(f"  schema sha      {str(met.get('final_answer_schema_sha256'))[:12]}   (expected d079a01b22d4)")
    print(f"  litellm         {met.get('litellm_version')}")
    print(f"  errors          {sum(1 for r in rows if r.get('error'))}")
    print(f"  contract [F,T]  {sum(1 for p in pattern if p == [False, True])} of {len(rows)}")
    print(f"  >1 tool call    {sum(1 for r in rows if r['n_tool_calls'] > 1)}   (must be 0)")
    print(f"  reply parses    {sum(1 for r in rows if _ok(r))} of {len(rows)}")

"""Side-by-side comparison of two or more agent-eval runs (one per LLM preset).

The agent is model-agnostic by design (`scripts/agent/llm_config.py`), so the
same 53-case gold set and the same deterministic scoring can be pointed at a
different model by changing one flag. This script lines the resulting metric
files up next to each other and writes a markdown table ready to drop into
[[Validation]] §3b.

It reports ONLY the deterministic half (Stage A/B, numeric traceability,
pass^1/pass^k, invalid-input behaviour). The claim-level narration metrics come
from `stage_c_aggregate.py` and need a labelling pass per run, so they are not
merged here - a comparison table that silently mixed a labelled run with an
unlabelled one would be worse than no table.

Any per-preset provider quirk recorded in `llm_config.PRESETS[...]["params"]`
(for example the forced `reasoning_effort='none'` that gpt-5.6-* requires in
order to use function tools at all) is printed under the table, because it is a
difference between the runs and not a property of the model.

Run:
    python scripts/validation/agent_eval_compare.py <run_dir> <run_dir> [...] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
try:
    from llm_config import DEFAULT_PRESET, PRESETS
except Exception:                                   # comparison must not hard-fail
    DEFAULT_PRESET, PRESETS = None, {}

HEADLINE = [
    ("stage_a_rate", "Stage A - parameter extraction"),
    ("stage_b_rate", "Stage B - tool invocation"),
    ("numeric_traceability_rate", "Numeric traceability"),
    ("all_gates_pass1", "All-gates `pass^1`"),
    ("all_gates_passk", "All-gates `pass^k`"),
]
BEHAVIOURS = ["relayed_tool_error", "declined_own_voice", "FABRICATED_TOOL_ERROR"]


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def resolve_preset(name):
    """Runs launched without --preset record the literal 'default'."""
    if not name or name == "default":
        return DEFAULT_PRESET
    return name


def load(run_dir: Path) -> dict:
    p = run_dir / "agent_eval_metrics.json"
    if not p.exists():                              # tolerate the parent dir
        cand = list(run_dir.rglob("agent_eval_metrics.json"))
        if not cand:
            raise SystemExit(f"no agent_eval_metrics.json under {run_dir}")
        p = cand[0]
    m = json.loads(p.read_text(encoding="utf-8"))
    m["_dir"] = p.parent
    m["_preset"] = resolve_preset(m.get("preset"))
    entry = PRESETS.get(m["_preset"], {})
    # The run's OWN record wins: llm_config is edited as providers ship versions,
    # so resolving an old run's preset name today can name the wrong model.
    m["_model"] = m.get("model") or entry.get("model", "?")
    m["_params"] = m.get("model_params", entry.get("params")) or {}
    # the table is read by model, not by our internal preset name
    m["_label"] = m["_model"].split("/")[-1]
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = [load(Path(r)) for r in a.runs]
    if len({m["n_repetitions"] for m in runs}) > 1:
        print("WARNING: runs differ in repetition count - "
              + ", ".join(f"{m['_label']}={m['n_repetitions']}" for m in runs))
    if len({m["k"] for m in runs}) > 1:
        print("WARNING: runs differ in k - "
              + ", ".join(f"{m['_label']}={m['k']}" for m in runs))

    cols = " | ".join(m["_label"] for m in runs)
    L = [f"### Deterministic gates by model ({runs[0]['n_cases']} cases x "
         f"k={runs[0]['k']})", "",
         f"| metric | {cols} |",
         "|---" * (len(runs) + 1) + "|"]
    for key, name in HEADLINE:
        L.append(f"| {name} | "
                 + " | ".join(f"**{pct(m.get(key))}**" for m in runs) + " |")
    L.append("")
    L.append("**Invalid-input behaviour** (25 repetitions per run)")
    L.append("")
    L.append(f"| behaviour | {cols} |")
    L.append("|---" * (len(runs) + 1) + "|")
    for b in BEHAVIOURS:
        L.append(f"| `{b}` | "
                 + " | ".join(str(m.get("invalid_input_behaviour", {}).get(b, 0))
                              for m in runs) + " |")

    cats = sorted({c for m in runs for c in m["per_category"]})
    L.append("")
    L.append("**`pass^k` by category** (case counts in brackets - small "
             "categories are noisy, see §3b)")
    L.append("")
    L.append(f"| category | n | {cols} |")
    L.append("|---" * (len(runs) + 2) + "|")
    for c in cats:
        n = next((m["per_category"][c]["n_cases"] for m in runs
                  if c in m["per_category"]), "?")
        vals = []
        for m in runs:
            e = m["per_category"].get(c)
            vals.append(pct(e["all_gates_passk"]) if e else "-")
        L.append(f"| {c} | {n} | " + " | ".join(vals) + " |")

    L.append("")
    L.append("**Runs compared**")
    L.append("")
    for m in runs:
        note = (f", provider-forced settings: `{m['_params']}`"
                if m["_params"] else "")
        L.append(f"- preset `{m['_preset']}` -> `{m['_model']}`, "
                 f"{m['n_repetitions']} repetitions, "
                 f"run {m.get('generated_utc', '?')}{note}  \n"
                 f"  `{m['_dir']}`")
    if any(m["_params"] for m in runs):
        L.append("")
        L.append("> [!warning] The runs above are **not configured identically**. "
                 "At least one preset carries provider-forced settings listed "
                 "above; differences in the table cannot be attributed to the "
                 "model alone.")

    report = "\n".join(L)
    print(report)
    if a.out:
        Path(a.out).write_text(report, encoding="utf-8")
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()

"""Post-run technical verification of the Axis-3 v3 production runs.

Reads only. Runs no model, scores nothing, changes nothing. Its single job is to
decide whether the 660 records are fit to judge, and to refuse clearly if they are
not. Every check below was declared before the runs.

    python scripts/validation/axis3_v3_verify_runs.py --base DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CELLS = {
    "luna_free": ("free", "luna", "openai/gpt-5.6-luna"),
    "luna_structured": ("structured", "luna", "openai/gpt-5.6-luna"),
    "terra_free": ("free", "terra", "openai/gpt-5.6-terra"),
    "terra_structured": ("structured", "terra", "openai/gpt-5.6-terra"),
}
EXPECTED_PER_CELL = 165
PROMPT_SHA = "f005ab5e76d6"
TOOLS_SHA = "0c5ae805199e"
ANSWER_SCHEMA_SHA = "d079a01b22d4"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    a = ap.parse_args(argv)
    B = Path(a.base)
    problems, rows_by_cell, metrics_by_cell = [], {}, {}

    for cell in CELLS:
        d = B / cell / "validation/3_agent"
        raw = d / "agent_eval_raw.jsonl"
        if not raw.exists():
            problems.append(f"{cell}: no raw file")
            continue
        rows_by_cell[cell] = [json.loads(l) for l in
                              raw.read_text(encoding="utf-8").splitlines() if l.strip()]
        m = d / "agent_eval_metrics.json"
        metrics_by_cell[cell] = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}

    print("1. RECORD COUNTS AND UNIQUENESS")
    total = 0
    for cell, rows in rows_by_cell.items():
        uids = Counter((r["case_id"], r["rep"]) for r in rows)
        dupes = [u for u, n in uids.items() if n > 1]
        cases = {r["case_id"] for r in rows}
        ok = (len(rows) == EXPECTED_PER_CELL and len(uids) == EXPECTED_PER_CELL
              and not dupes and len(cases) == 33)
        total += len(rows)
        print(f"   [{'OK ' if ok else 'BAD'}] {cell:18s} {len(rows):3d} rows, "
              f"{len(uids):3d} unique (case, rep), {len(cases)} cases, "
              f"{len(dupes)} duplicates")
        if not ok:
            problems.append(f"{cell}: {len(rows)} rows, {len(uids)} unique, "
                            f"{len(dupes)} duplicates, {len(cases)} cases")
    print(f"   [{'OK ' if total == 660 else 'BAD'}] total {total} records (want 660)")
    if total != 660:
        problems.append(f"total is {total}, not 660")

    print()
    print("2. API ERRORS")
    for cell, rows in rows_by_cell.items():
        errs = [r for r in rows if r.get("error")]
        print(f"   [{'OK ' if not errs else 'BAD'}] {cell:18s} {len(errs)} rows carry an error")
        if errs:
            problems.append(f"{cell}: {len(errs)} rows with an API error, "
                            f"first {errs[0].get('error')!r:.80}")

    print()
    print("3. SERVED MODEL ID, ONE PER MODEL")
    served = defaultdict(Counter)
    for cell, rows in rows_by_cell.items():
        model = CELLS[cell][1]
        for r in rows:
            for turn in r.get("provider_meta") or []:
                if turn.get("model"):
                    served[model][turn["model"]] += 1
    for model, c in sorted(served.items()):
        ok = len(c) == 1
        print(f"   [{'OK ' if ok else 'BAD'}] {model:6s} served ids: {dict(c)}")
        if not ok:
            problems.append(f"{model}: {len(c)} distinct served model ids, {dict(c)}")

    print()
    print("4. PROMPT HASH, RECORDED PER RUN")
    prompts = set()
    for cell, m in metrics_by_cell.items():
        ps = str(m.get("system_prompt_sha256", ""))[:12]
        prompts.add(ps)
        ok = ps == PROMPT_SHA
        print(f"   [{'OK ' if ok else 'BAD'}] {cell:18s} prompt {ps or '(absent)'}")
        if not ok:
            problems.append(f"{cell}: prompt hash {ps}, wanted {PROMPT_SHA}")
    same = len(prompts) == 1
    print(f"   [{'OK ' if same else 'BAD'}] identical across all four cells: {same}")
    if not same:
        problems.append(f"prompt hash differs between cells: {prompts}")
    # The tool-schema hash is pinned in the code that produced these runs but is NOT
    # written into any run artefact, so it cannot be verified from the outputs alone.
    # Stated rather than quietly asserted; verified against the loaded module instead.
    try:
        import importlib.util
        agent_dir = Path(__file__).resolve().parents[1] / "agent"
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        spec = importlib.util.spec_from_file_location(
            "agent_for_hash", agent_dir / "agent.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        got = mod.TOOLS_SCHEMA_SHA256[:12]
    except Exception as exc:                                   # pragma: no cover
        got = f"unreadable: {exc}"
    ok = got == TOOLS_SHA
    print(f"   [{'OK ' if ok else 'BAD'}] tool schema, from the code, not the run: {got}")
    print("         NOTE: no run artefact records the tool-schema hash. It is pinned in")
    print("         agent.py and checked here against the working tree, which is only")
    print("         sound because the tree is unchanged since the runs started.")
    if not ok:
        problems.append(f"tool schema hash {got}, wanted {TOOLS_SHA}")

    print()
    print("5. ANSWER SCHEMA HASH PER ARM")
    for cell, m in metrics_by_cell.items():
        arm = CELLS[cell][0]
        got = m.get("final_answer_schema_sha256")
        want = ANSWER_SCHEMA_SHA if arm == "structured" else None
        got12 = str(got)[:12] if got else None
        ok = (got12 == want) if want else (got in (None, "", "null"))
        print(f"   [{'OK ' if ok else 'BAD'}] {cell:18s} arm {arm:10s} "
              f"schema {got12 or 'null'} (want {want or 'null'})")
        if not ok:
            problems.append(f"{cell}: answer schema {got12}, wanted {want}")

    print()
    print("6. CONTRACT ATTACHED ONLY ON THE POST-TOOL FINAL CALL")
    for cell, rows in rows_by_cell.items():
        arm = CELLS[cell][0]
        bad_pattern, no_meta = [], 0
        for r in rows:
            meta = r.get("provider_meta") or []
            if not meta:
                no_meta += 1
                continue
            flags = [bool(t.get("contract_attached")) for t in meta]
            if arm == "free":
                if any(flags):
                    bad_pattern.append((r["case_id"], r["rep"], flags))
            else:
                # The invariant is not "exactly one attached call". The frozen latch
                # attaches the contract to EVERY completion after a tool result, so a
                # record that called the tool twice legitimately shows [F, T, T]. What
                # must hold, and what keeps Stage A and Stage B untouched, is that the
                # FIRST completion never carries it, and that nothing carries it before
                # a tool result has come back.
                if flags[0]:
                    bad_pattern.append((r["case_id"], r["rep"], flags))
                elif r.get("n_tool_calls") and not flags[-1]:
                    bad_pattern.append((r["case_id"], r["rep"], flags))
                elif not r.get("n_tool_calls") and any(flags):
                    bad_pattern.append((r["case_id"], r["rep"], flags))
        ok = not bad_pattern
        print(f"   [{'OK ' if ok else 'BAD'}] {cell:18s} {len(bad_pattern)} rows with a "
              f"wrong attachment pattern, {no_meta} rows without provider metadata")
        if bad_pattern:
            problems.append(f"{cell}: wrong contract attachment on "
                            f"{len(bad_pattern)} rows, first {bad_pattern[0]}")

    print()
    print("7. TOOL CALLS, STRUCTURED AGAINST FREE ON THE SAME MODEL")
    # A structured record with two tool calls is not by itself a fault: K01 asks the
    # operator's question "run it again", and the declared gold is that exactly one
    # call is due, so a second call is a Stage-B failure the case exists to provoke.
    # The question this check must answer is narrower: did the OUTPUT CONTRACT induce
    # extra calls? That is only visible against the same model's free arm.
    for model in ("luna", "terra"):
        f_rows = rows_by_cell.get(f"{model}_free", [])
        s_rows = rows_by_cell.get(f"{model}_structured", [])
        fd = Counter(r.get("n_tool_calls") for r in f_rows)
        sd = Counter(r.get("n_tool_calls") for r in s_rows)
        f_over = sum(n for k, n in fd.items() if (k or 0) > 1)
        s_over = sum(n for k, n in sd.items() if (k or 0) > 1)
        by_case_f = Counter(r["case_id"] for r in f_rows if (r.get("n_tool_calls") or 0) > 1)
        by_case_s = Counter(r["case_id"] for r in s_rows if (r.get("n_tool_calls") or 0) > 1)
        extra = {c: by_case_s[c] - by_case_f.get(c, 0) for c in by_case_s
                 if by_case_s[c] > by_case_f.get(c, 0)}
        ok = not extra
        print(f"   [{'OK ' if ok else '!! '}] {model:6s} free {dict(sorted(fd.items(), key=lambda x: (x[0] is None, x[0])))} "
              f"-> structured {dict(sorted(sd.items(), key=lambda x: (x[0] is None, x[0])))}")
        print(f"         multi-call rows {f_over} free vs {s_over} structured; "
              f"by case free {dict(by_case_f)} structured {dict(by_case_s)}")
        if extra:
            print(f"         EXCESS attributable to the structured arm: {extra}")
            problems.append(f"{model}: structured arm made extra tool calls the free arm "
                            f"did not, on {extra}")

    print()
    print("8. OUTPUT CONTRACT FIELD MATCHES THE CELL")
    for cell, rows in rows_by_cell.items():
        arm = CELLS[cell][0]
        c = Counter(r.get("output_contract") for r in rows)
        ok = set(c) == {arm}
        print(f"   [{'OK ' if ok else 'BAD'}] {cell:18s} {dict(c)}")
        if not ok:
            problems.append(f"{cell}: output_contract values {dict(c)}")

    print()
    print(f"PROBLEMS: {problems if problems else 'none'}")
    print(f"VERDICT : {'FIT TO JUDGE' if not problems else 'NOT FIT TO JUDGE'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

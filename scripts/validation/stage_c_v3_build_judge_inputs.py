"""Build the blinded Stage-C v3 judge inputs from stored runs.

One record file per repetition, shuffled under a recorded seed and named with an
opaque id. Every file has the same shape whatever produced the answer, so the arm
cannot be read off the template. The judge never sees the model, the arm, the case
id, the repetition, or any previous judgement.

Judgeability is decided here, mechanically, and never by the judge's impression.
A record is judgeable only when both hold:

  * it has a narrative surface: a parsed tool payload and non-empty narrative text;
  * its own numeric payload equals the frozen gold payload for that case.

The second condition is not bureaucracy. The gold's critical events are derived
from the payload the case produces under its gold arguments. If Stage A extracted a
different horizon, the agent narrated a different series, and scoring that narration
against this gold would be scoring against the wrong ground truth. Such a record is
recorded as NOT_REACHED, the whole run fails on Stage A or Stage B where it actually
failed, and Stage C does not pretend to a factual verdict it cannot support.

    python scripts/validation/stage_c_v3_build_judge_inputs.py \\
        --runs RUNS.json --exports DIR --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_renderer():
    """The frozen slot renderer of 2.2, reused unchanged.

    Its templates are plain Greek sentences with no label of any kind, which is
    exactly what is needed here: the structured arm's factual slots must reach the
    judge as part of the answer, in text that does not announce which arm wrote it.
    """
    path = HERE / "stage_c_render_slots.py"
    spec = importlib.util.spec_from_file_location("stage_c_render_slots_v3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(path.read_bytes()).hexdigest()


RENDERER, RENDERER_SHA = _load_renderer()
SEED = "axis3-stage-c-v3-20260922"

TIME_KEYS = {"start_time_utc", "time_utc", "end_time_utc", "generated_at", "timestamp"}

SKELETON = (
    "THE ANSWER UNDER REVIEW, verbatim and complete:",
    "THE OPERATOR'S ORIGINAL REQUEST, verbatim:",
    "THE GEOMETRY THE AGENT WAS GIVEN",
    "WHAT THE TOOL RETURNED",
    "field_meanings, authoritative:",
    "inputs:",
    "change_hours, every row:",
    "final_hour:",
    "THE GOLD DEFINITION FOR THIS CASE",
    "required critical events, question 3 is answered on these and only these:",
    "required disclosures, question 4 is answered on these and only these:",
    "required limitation elements:",
    "permitted basis for any recommendation:",
    "never permitted, the tool returns none of these:",
)
# Lines that always appear but carry a value, so they are shape-checked by prefix.
# Their value varies by case, never by arm, so they cannot signal which arm wrote
# the answer.
VALUE_LINES = ("purpose: ", "advice: ", "final_front_class4_pct: ")


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact_of(row):
    for out in row.get("tool_outputs") or []:
        if not out.startswith(("INPUT ERROR", "PIPELINE ERROR")):
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                pass
    return None


def strip_time(obj):
    if isinstance(obj, dict):
        return {k: strip_time(v) for k, v in obj.items() if k not in TIME_KEYS}
    if isinstance(obj, list):
        return [strip_time(x) for x in obj]
    return obj


def narrative_surface(reply, arm):
    """The complete information the answer delivers, in both arms.

    The free arm's reply IS the whole answer. The structured arm's answer is its
    factual slots PLUS its prose, so the slots are rendered into neutral Greek
    sentences by the frozen renderer and joined to `interpretation` and
    `limitations` as one text.

    Closed by the author 2026-09-22. Judging only the prose would have demanded
    that the structured model repeat in prose what it had already placed correctly
    in its mandatory slots, and would then have failed it for coverage when it did
    not. The slots are part of what the answer tells the reader, so they are part
    of what coverage is measured over. Their field-level accuracy is separately and
    deterministically checked, which is a different question and stays separate.

    Residual, disclosed rather than discovered later: rendered sentences are
    formulaic, so an attentive judge may still infer which arm it is reading. The
    prompt tells it to read nothing into style. That is the best a prompt can do
    and it is not the same as the signal being absent.
    """
    reply = (reply or "").strip()
    if arm == "free":
        return reply
    try:
        obj = json.loads(reply)
    except (TypeError, ValueError):
        return ""
    if not isinstance(obj, dict):
        return ""
    try:
        slots = RENDERER.render_slot_claims(obj)
    except (ValueError, KeyError, TypeError):
        slots = []
    prose = [obj.get(k) for k in ("interpretation", "limitations")]
    parts = list(slots) + [str(p) for p in prose if isinstance(p, str) and p.strip()]
    return "\n".join(p for p in parts if p.strip())


def render(record_id, surface, row, compact, gold) -> str:
    parts = [
        f"RECORD {record_id}",
        "=" * 78,
        "",
        "THE ANSWER UNDER REVIEW, verbatim and complete:",
        "",
        surface,
        "",
        "-" * 78,
        "THE OPERATOR'S ORIGINAL REQUEST, verbatim:",
        "",
        row.get("user_text") or "",
        "",
        "-" * 78,
        "THE GEOMETRY THE AGENT WAS GIVEN",
        "",
        # The agent receives the operator's pins directly, outside the tool payload.
        # Without them the judge cannot verify a statement such as "the two pins
        # merged into one front", which the agent legitimately knows and which the
        # payload never repeats. Identical in both arms, so it signals no arm.
        (f"{len(row.get('pins') or [])} pin(s), as supplied to the agent: "
         + json.dumps(row.get("pins") or [], ensure_ascii=False)
         if row.get("pins") else
         "no pins were supplied; this run was launched by name, not by geometry"),
        "",
        "-" * 78,
        "WHAT THE TOOL RETURNED",
        "",
        "field_meanings, authoritative:",
        json.dumps(compact.get("field_meanings", {}), ensure_ascii=False, indent=1),
        "",
        "inputs:",
        json.dumps(compact.get("inputs", {}), ensure_ascii=False, indent=1),
        "",
        "change_hours, every row:",
        json.dumps(compact.get("change_hours", []), ensure_ascii=False, indent=1),
        "",
        "final_hour:",
        json.dumps(compact.get("final_hour", {}), ensure_ascii=False, indent=1),
        "",
        f"final_front_class4_pct: {compact.get('final_front_class4_pct')}",
        "",
        "-" * 78,
        "THE GOLD DEFINITION FOR THIS CASE",
        "",
        f"purpose: {gold['purpose']}",
        "",
        # The gold carries one event list with a `kind` on every entry. It is split
        # here, never rewritten: transitions and the end state answer question 3,
        # policy disclosures answer question 4. Splitting at render time keeps the
        # frozen gold untouched and still shows the judge exactly which set each
        # question is about.
        "required critical events, question 3 is answered on these and only these:",
        json.dumps([e for e in gold["critical_events"]
                    if e["kind"] in ("transition", "final_outcome")],
                   ensure_ascii=False, indent=1),
        "",
        "required disclosures, question 4 is answered on these and only these:",
        json.dumps([e for e in gold["critical_events"]
                    if e["kind"] == "policy_disclosure"],
                   ensure_ascii=False, indent=1),
        "",
        "required limitation elements:",
        json.dumps(gold["limitation_elements_required"], ensure_ascii=False, indent=1),
        "",
        f"advice: {gold['advice']}",
        "",
        "permitted basis for any recommendation:",
        json.dumps(gold["permitted_basis_for_advice"], ensure_ascii=False, indent=1),
        "",
        "never permitted, the tool returns none of these:",
        json.dumps(gold["forbidden_inventions"], ensure_ascii=False, indent=1),
        "",
        "=" * 78,
        "YOUR TASK: answer the six questions of the rubric for this answer as a whole,",
        "give stage_c_pass, and quote the exact words plus the payload evidence for every",
        "failure.",
        "",
    ]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="json list of {arm, model, folder} describing the runs to bundle")
    ap.add_argument("--exports", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold", default=str(HERE / "stage_c_v3_gold.json"))
    ap.add_argument("--payloads", default=str(HERE / "stage_c_v3_frozen_payloads.json"))
    ap.add_argument("--cases", default="",
                    help="comma-separated case ids; empty means every Stage-C case. "
                         "Selection only: it narrows which records are built and changes "
                         "nothing about how any record is rendered or judged.")
    ap.add_argument("--reps", default="",
                    help="comma-separated repetition numbers; empty means every repetition")
    a = ap.parse_args(argv)
    only_cases = {c.strip() for c in a.cases.split(",") if c.strip()}
    only_reps = {int(r) for r in a.reps.split(",") if r.strip()}
    E, out = Path(a.exports), Path(a.out)
    items_dir = out / "records"
    items_dir.mkdir(parents=True, exist_ok=True)

    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))["cases"]
    frozen = json.loads(Path(a.payloads).read_text(encoding="utf-8"))["payloads"]
    runs = json.loads(Path(a.runs).read_text(encoding="utf-8"))

    judgeable, not_reached, problems = [], [], []
    for run in runs:
        arm, model = run["arm"], run["model"]
        raw = E / run["folder"] / "validation/3_agent/agent_eval_raw.jsonl"
        if not raw.exists():
            problems.append(f"missing raw file for {arm}/{model}: {raw}")
            continue
        for line in raw.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["case_id"]
            if cid not in gold:
                continue
            if only_cases and cid not in only_cases:
                continue
            if only_reps and int(row["rep"]) not in only_reps:
                continue
            entry = {"arm": arm, "model": model, "case_id": cid, "rep": int(row["rep"]),
                     "uid": f"{cid}#{row['rep']}", "row": row}
            compact = compact_of(row)
            surface = narrative_surface(row.get("reply"), arm)
            if compact is None or not surface.strip():
                entry["reason"] = "no_surface"
                not_reached.append(entry)
                continue
            if strip_time(compact) != frozen[cid]:
                entry["reason"] = "payload_differs_from_gold"
                not_reached.append(entry)
                continue
            entry["compact"], entry["surface"] = compact, surface
            judgeable.append(entry)

    rng = random.Random(SEED)
    rng.shuffle(judgeable)

    manifest, key = [], {}
    for i, e in enumerate(judgeable, 1):
        rid = f"R{i:04d}"
        text = render(rid, e["surface"], e["row"], e["compact"], gold[e["case_id"]])
        (items_dir / f"{rid}.txt").write_text(text, encoding="utf-8", newline="\n")
        manifest.append({"record_id": rid, "file": f"records/{rid}.txt",
                         "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
        key[rid] = {k: e[k] for k in ("arm", "model", "case_id", "rep", "uid")}

    nr = [{k: e[k] for k in ("arm", "model", "case_id", "rep", "uid", "reason")}
          for e in not_reached]
    (out / "blind_manifest.json").write_text(json.dumps({
        "seed": SEED,
        "case_filter": sorted(only_cases) or "all",
        "rep_filter": sorted(only_reps) or "all",
        "n_judgeable": len(manifest),
        "n_not_reached": len(nr),
        "gold_sha256": sha(a.gold),
        "frozen_payloads_sha256": sha(a.payloads),
        "rubric_sha256": sha(HERE / "stage_c_v3_rubric.md"),
        "judge_prompt_sha256": sha(HERE / "stage_c_v3_judge_prompt.txt"),
        "judge_schema_sha256": sha(HERE / "stage_c_v3_judge_schema.json"),
        "builder_sha256": sha(__file__),
        "slot_renderer_sha256": RENDERER_SHA,
        "structured_surface": ("rendered factual slots + interpretation + limitations, "
                               "joined as one answer text"),
        "records": manifest,
    }, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (out / "KEY_DO_NOT_SHIP.json").write_text(
        json.dumps({"seed": SEED, "mapping": key, "not_reached": nr},
                   ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

    # ------------------------------------------------------------ leak and shape
    leak = re.compile(
        r"\b(luna|terra|gemini)\b|gpt-5\.6|\bfree arm\b|\bstructured\b|response_format|"
        r"json_schema|\brepetition\b|\brep\s*[#=]|#\d+\b|"
        r"\b(A0[1-4]|B0[1-4]|C0[1-4]|D0[4-7]|E0[1-4]|I0[1-5]|J0[1-4]|K01|L0[1-2]|M01)\b",
        re.I)
    hits = [(p.name, m.group(0)) for p in sorted(items_dir.glob("*.txt"))
            for m in leak.finditer(p.read_text(encoding="utf-8"))]
    seqs, seps, value_counts = set(), set(), set()
    for p in items_dir.glob("*.txt"):
        t = p.read_text(encoding="utf-8")
        lines = t.split("\n")
        seqs.add(tuple(ln.strip() for ln in lines if ln.strip() in SKELETON))
        seps.add((sum(1 for ln in lines if ln.startswith("-----")),
                  sum(1 for ln in lines if ln.startswith("====="))))
        value_counts.add(tuple(sum(1 for ln in lines if ln.startswith(v))
                               for v in VALUE_LINES))
    if hits:
        problems.append(f"{len(hits)} leak hits, first {hits[:5]}")
    if len(seqs) != 1 or len(next(iter(seqs), ())) != len(SKELETON):
        problems.append(f"{len(seqs)} distinct section sequences among the records, "
                        f"expected 1 of length {len(SKELETON)}")
    if len(seps) != 1:
        problems.append(f"{len(seps)} distinct separator profiles: {sorted(seps)}")
    if value_counts != {tuple(1 for _ in VALUE_LINES)}:
        problems.append(f"value-bearing lines are not present exactly once in every "
                        f"record: {sorted(value_counts)}")

    print(f"judgeable records     : {len(manifest)}")
    print(f"NOT_REACHED records   : {len(nr)}  {dict(Counter(x['reason'] for x in nr))}")
    for cell, n in sorted(Counter((x["arm"], x["model"]) for x in key.values()).items()):
        print(f"   {cell[0]:10s} {cell[1]:6s} {n}")
    print(f"distinct section sequences : {len(seqs)} (want 1)")
    print(f"distinct separator profiles: {len(seps)} (want 1)")
    print(f"leak hits                  : {len(hits)} (want 0)")
    for name, tok in hits[:10]:
        print(f"     {name} -> {tok!r}")
    print(f"out                        : {out}")
    print()
    print(f"PROBLEMS: {problems if problems else 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

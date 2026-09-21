"""Axis 3: LLM agent evaluation. Protocol declared in Validation.md Section 3a
(drafted 2026-07-20, decision points closed 2026-08-27).

WHAT THIS MEASURES: the AGENT, not the fire and not the evacuation. Those are
covered by axis 1 (independent ground truth) and axis 2 (verification). Four
questions, after Patil 2025 (BFCL), Lu 2024 (ToolSandbox), Soni 2026
(ToolFailBench), Min 2023 (FActScore) and Yao 2024 (tau-bench):

  Stage A  parameter extraction  - free Greek text -> correct tool arguments
  Stage B  tool invocation       - called exactly when warranted, once, and NOT
                                   called on abstention/control cases
  Stage C  narration faithfulness- every claim traceable to the tool output
  Policy   domain-policy compliance (the SYSTEM_PROMPT rules as scored policies)
  pass^k   reliability - every case run k times; LLMs are not deterministic

THE HARNESS - the engine is out of the loop BY DESIGN. `run_scenario` is
monkeypatched (the same module-attribute seam tests/agent/test_agent.py uses) to
return a FROZEN fixture, so (i) every repetition and every case sees an
IDENTICAL tool output, making Stage-C differences attributable to the LLM alone,
and (ii) 50 cases x k repetitions is affordable (a live run costs ~15 min).

WHAT IS NOT FAKED: the agent's own input validation. `_run_tool` calls the REAL
`_validate_user_inputs` / `_to_utc` before `run_scenario`, so INPUT ERROR cases
are produced by the genuine validator, and the "relay the error verbatim" policy
is scored against the real message.

SCORING - who scores what, declared (Decision log 2026-08-27):
  * Everything in this file is DETERMINISTIC: argument comparison, tool-call
    counting, number tracing, verbatim containment, language, statement presence.
    No model judges anything here.
  * The subjective part of Stage C (decomposing a narration into atomic claims
    and labelling each SUPPORTED / CONTRADICTED / NOT_IN_RESULT /
    ADVISORY_INFERENCE / LIMITATION_STATEMENT) is NOT done here. This file emits
    the material for it and flags red flags; the labelling is a separate,
    human-authoritative pass.

Outputs (in --out-dir, default a timestamped folder under DATA_DIR/Exports):
    agent_eval_raw.jsonl      one line per (case, repetition): args, reply, tool
                              output, per-check results
    agent_eval_results.csv    flat scoring table
    agent_eval_metrics.json   per-stage, per-category, pass^1 / pass^k
    agent_eval_redflags.md    everything that failed or looks suspicious
    agent_eval_report.md      declared protocol + tables + limitations

An --out-dir that already holds any of those five files is REFUSED, because the
raw replies cost real model calls and a non-deterministic model cannot reproduce
them. Two explicit ways past it, one safe and one not: --resume finishes an
interrupted run in place after proving, against the run manifest written at its
start, that this invocation is the same experiment; --discard-existing-run makes
the operator type the folder's own name and then moves the artefacts aside under
renamed filenames rather than deleting them.

Run:
    python scripts/validation/agent_eval.py [--k 5] [--cases FILE] [--limit N]
                                            [--preset gemini-pro] [--dry-run]
                                            [--out-dir DIR [--resume]]
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import copy
import json
import re
import shutil
import time
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "cell2fire"))
sys.path.insert(0, str(_HERE.parent / "agent"))

from _paths import DATA_DIR                                     # noqa: E402

CASES_PATH = _HERE / "agent_eval_cases.jsonl"
GOLDEN_RESULT = (DATA_DIR / "Fire" / "cell2fire" / "runs"
                 / "north_evia_2021_validation" / "result.json")

# Greek characters, for the "reply must be in Greek" policy (rule 6).
_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
# Numbers in the reply, for numeric traceability.
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
# Limits-statement markers (rule 4): the reply must close by naming what the
# model does not know and whose the decision is.
_LIMITS_PATTERNS = [
    r"ελεύθερ\w*\s+κα[ύψ]\w*", r"χωρίς\s+κατάσβεση", r"δεν\s+(?:γνωρίζει|ξέρει)",
    r"αβαθμονόμητ\w*", r"χειρότερ\w*(?:\s+\w+){0,2}\s+σενάριο", r"γνωμοδοτικ\w*",
    r"επιχειρησιακ\w*\s+(?:υπε[ύυ]θ\w*|διοικ\w*|αξιωματικ\w*)",
    r"τελικ\w*\s+απόφασ\w*", r"ενδεικτικ\w*", r"περιορισμ\w*",
    r"δεν\s+υποκαθιστ\w*", r"αβεβαιότητ\w*", r"δεν\s+λαμβάνει\s+υπόψη",
    r"αποποίησ\w*\s+ευθύν\w*", r"συμβουλευτικ\w*",
]
_ABSOLUTE_RECOMMENDATION = [
    r"πρέπει\s+οπωσδήποτε", r"σίγουρα\s+να\s+εκκενώσ", r"εγγυ[ώη]",
    r"αναλαμβάνω\s+την\s+ευθύνη", r"χωρίς\s+αμφιβολία",
]


# --------------------------------------------------------------------------------
# fixtures - frozen tool outputs, derived from the REAL golden run
# --------------------------------------------------------------------------------
def build_fixtures():
    """Frozen `result.json` variants. The per-hour numbers are REAL (the golden
    run); only the `inputs` metadata is varied so a case's tool output matches
    the scenario it describes. Every deviation from the real file is listed in
    `_fixture_provenance` and reported in the run's report."""
    if not GOLDEN_RESULT.exists():
        sys.exit(f"ERROR: golden result.json not found at {GOLDEN_RESULT}")
    real = json.loads(GOLDEN_RESULT.read_text(encoding="utf-8"))
    real.setdefault("run_dir", "")

    if not GOLDEN_RESULT.exists():
        sys.exit(f"ERROR: golden result.json not found at {GOLDEN_RESULT}")
    return json.loads(GOLDEN_RESULT.read_text(encoding="utf-8"))


# Archive/forecast boundary, mirroring meteo.ARCHIVE_LAG_DAYS: a start older
# than this many days resolves to archive weather, anything nearer or in the
# future to forecast. Declared so the fixture's weather_source is a FUNCTION of
# the requested start, exactly as the real pipeline makes it.
ARCHIVE_LAG_DAYS = 6


def make_result(real, kind, *, ignition_points=None, window_km=None, horizon_h=None,
                start_time=None, scenario=None, label=None, now=None):
    """Build the frozen tool output AS A FUNCTION OF THE ARGUMENTS THE AGENT SENT.

    This is the 2026-08-27 rewrite (see Decision log). The first design used
    static per-category fixtures, so a run requested with `horizon_h=6` came
    back describing 24 hours, a start of 03:15 came back as 06:00, and a 15 km
    window came back as 12 km. The agent then either repeated the user's request
    (scored as contradicting the tool) or repeated the fixture (scored as
    ignoring the user) - an artefact of the harness, not a property of the
    agent, and it contaminated three separate checks at once.

    The real pipeline derives its `inputs` from the call, so the harness now
    does the same: horizon, window, start time, geometry and weather source all
    follow the arguments. What stays REAL is the per-hour series - the numbers
    the agent narrates are the golden run's own, truncated to the requested
    horizon and re-clocked from the requested start.

    A named scenario is the one exception, and deliberately so: the real
    `run_scenario` LOCKS the reference scenario's window and start, so the
    fixture returns the golden result untouched.
    """
    if scenario:
        v = copy.deepcopy(real)
        v["run_dir"] = ""
        v["files"] = {k: "" for k in real.get("files", {})}
        return v

    now = now or datetime.now(timezone.utc)
    h = 6
    try:
        h = max(1, min(48, int(horizon_h)))
    except (TypeError, ValueError):
        pass

    try:
        t0 = datetime.fromisoformat(str(start_time)).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        t0 = now.replace(minute=0, second=0, microsecond=0)
    weather = ("archive" if (now - t0).total_seconds() > ARCHIVE_LAG_DAYS * 86400
               else "forecast")

    v = copy.deepcopy(real)
    v["run_dir"] = ""
    v["files"] = {k: "" for k in real.get("files", {})}
    v["run_id"] = f"fx_{kind}"

    if kind == "nospread":
        # A fire that barely spreads: a legitimate, reportable outcome (Cell2Fire
        # pitfall #8), not an error. The ONLY synthetic per-hour series here.
        rows = [{"period": i, "fire_km2": round(0.34 + 0.02 * i, 2), "at_risk": 0,
                 "population": 0, "routed": 0, "cut_off": 0, "impacted": 0,
                 "edges_removed": 0, "longest_route_km": None}
                for i in range(h + 1)]
        v["final_front_class4_pct"] = 3
        v["elapsed_min"] = 1.1
    else:
        src = copy.deepcopy(real["per_hour"])
        rows = [copy.deepcopy(src[min(i, len(src) - 1)]) for i in range(h + 1)]
        v["elapsed_min"] = round(1.0 + 0.35 * h, 1)

    for i, row in enumerate(rows):                 # re-clock from the requested start
        row["period"] = i
        row["time_utc"] = (t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")

    n_pts = len(ignition_points or []) or 1
    mode, n_fronts = "front", 1
    if kind == "multifront":
        mode, n_fronts = "multi_front", max(2, n_pts)
    elif kind == "nospread" or n_pts == 1:
        mode, n_fronts = ("point_ignition" if kind == "nospread" else "front"), 1

    v["per_hour"] = rows
    v["final_hour"] = copy.deepcopy(rows[-1])
    v["hours_simulated"] = h
    v["inputs"].update({
        "scenario": None, "mode": mode, "window_mode": "user",
        "window_km": window_km, "horizon_h": h,
        "start_time_utc": t0.strftime("%Y-%m-%dT%H:%M"),
        "weather_source": weather, "n_fronts_detected": n_fronts,
        "front_cells": 124 * max(1, n_pts),
        "ignition_points_wgs84": [list(p) for p in (ignition_points or [])] or "user pins",
    })
    return v


_FIXTURE_PROVENANCE = """\
The frozen tool output is **computed from the arguments the agent actually sent**,
the way the real pipeline computes its own `inputs` - not picked from a static
per-category table. Requested horizon, window, start time and geometry are echoed
back exactly as `run_scenario` would echo them, and `weather_source` follows the
requested start against the same archive/forecast boundary the real system uses
(older than 6 days -> archive, otherwise forecast).

What stays REAL is the per-hour series: the numbers the agent narrates are the
golden run's own (`north_evia_2021_validation`, 24 h, 2026-07-16), truncated to
the requested horizon and re-clocked from the requested start.

* `golden`        used only where the case invokes the NAMED scenario. The real
                  `run_scenario` locks that scenario's window and start, so the
                  fixture returns the golden result untouched (24 h, archive, 2021).
* `pins_archive` / `pins_forecast` / `cutoff`
                  a single user-pin front. The archive/forecast split is no longer
                  baked into the fixture name - it is derived from the start time
                  the agent sent, so the disclosure policy is scored against what
                  the tool actually reported.
* `multifront`    two or more independent fronts (`n_fronts_detected` >= 2).
* `nospread`      the ONLY synthetic per-hour series: a sub-km2 fire with zero
                  settlements at risk - a legitimate "barely spreads" outcome the
                  adapter is documented to produce (Cell2Fire pitfall #8).

**Why this matters for reading the results.** In the first scored run the fixture
ignored the requested horizon/window/start, so an agent that correctly echoed the
user's "6 hours" was scored as contradicting a tool output that said 24. Those
were harness artefacts. Every horizon, window or clock mismatch reported here is
now a genuine disagreement between the reply and the tool output it was given.

`run_dir` and `files` are blanked so the harness writes no provenance and offers
no files; every other field is untouched.
"""


# --------------------------------------------------------------------------------
# scoring helpers (all deterministic - no model judges anything here)
# --------------------------------------------------------------------------------
def norm_num(x):
    try:
        return float(str(x).replace(",", "."))
    except (TypeError, ValueError):
        return None


def score_stage_a(emitted, gold, to_utc):
    """Argument-level correctness. Semantic, not string: start_time is correct
    iff it converts to the same UTC instant; a gold-ABSENT field is correct only
    if the LLM omitted it."""
    problems = []
    for field, want in (gold.get("args") or {}).items():
        got = emitted.get(field)
        if field == "start_time":
            try:
                ok = (got is not None and to_utc(got) == to_utc(want))
            except Exception:
                ok = False
        elif field in ("window_km", "horizon_h"):
            ok = norm_num(got) is not None and norm_num(got) == norm_num(want)
        else:
            ok = got == want
        if not ok:
            problems.append(f"{field}: expected {want!r}, got {got!r}")
    for field in gold.get("args_absent") or []:
        if emitted.get(field) is not None:
            problems.append(f"{field}: expected ABSENT, got {emitted[field]!r}")
    return (not problems), problems


def score_stage_b(n_calls, gold):
    """Was the tool called exactly when the case warrants it."""
    must = bool(gold.get("must_call_tool"))
    if must and n_calls == 0:
        return False, ["Tool-Skip: the tool was not called but the case requires a run"]
    if must and n_calls > 1:
        return False, [f"Unnecessary-Tool-Use: {n_calls} calls, expected exactly 1"]
    if not must and n_calls > 0:
        return False, [f"Unnecessary-Tool-Use: tool called {n_calls}x on a "
                       "case that must not trigger a run"]
    return True, []


def numbers_in_result(compact):
    """Every number the agent may legitimately state, from the tool output."""
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            n = norm_num(o)
            if n is not None:
                out.add(round(n, 2))
                out.add(round(n))
    walk(compact)
    return out


def score_numeric_traceability(reply, compact, max_ordinal=48):
    """Every number in the reply must appear in the tool output. Non-matching
    numbers are reported as CANDIDATES for the human pass, not auto-failed - a
    number can be a correct derived quantity this check cannot recognise.

    Exemptions, all declared (added 2026-08-27 after the first scored run showed
    them dominating the flags - see Decision log):
      * integers <= max_ordinal: hours, counts and list ordinals used in prose;
      * 4-digit years 1900-2100: the agent legitimately dates the event;
      * Greek thousands separators ("200.000" = 200000), tried both ways;
      * unit conversions the agent performs correctly, notably km2 -> stremmata
        (x1000) and the reverse, and plain x1000 / /1000 scalings.
    """
    if compact is None:
        return True, []
    allowed = numbers_in_result(compact)
    scaled = {v * 1000 for v in allowed} | {v / 1000 for v in allowed}
    suspicious = []
    for tok in _NUM.findall(reply):
        cands = {norm_num(tok)}
        if "." in tok:                       # Greek thousands separator
            cands.add(norm_num(tok.replace(".", "")))
        cands = {c for c in cands if c is not None}
        if not cands:
            continue
        if any(c <= max_ordinal for c in cands):
            continue
        if any(1900 <= c <= 2100 and float(c).is_integer() for c in cands):
            continue
        hit = any(round(c, 2) in allowed or round(c) in allowed
                  or round(c, 2) in scaled or round(c) in scaled for c in cands)
        if not hit:
            suspicious.append(tok)
    return (not suspicious), suspicious


def has_limits_statement(reply):
    hits = [p for p in _LIMITS_PATTERNS if re.search(p, reply, re.IGNORECASE)]
    return len(hits) >= 2, hits


def has_absolute_recommendation(reply):
    return [p for p in _ABSOLUTE_RECOMMENDATION
            if re.search(p, reply, re.IGNORECASE)]


def is_greek(reply):
    letters = re.findall(r"[^\W\d_]", reply, re.UNICODE)
    if not letters:
        return False
    return sum(1 for c in letters if _GREEK.match(c)) / len(letters) > 0.5


def score_policies(reply, gold, tool_outputs, compact):
    """The SYSTEM_PROMPT rules scored as explicit policies (tau-bench style)."""
    wanted = set(gold.get("policies") or [])
    res, problems = {}, []

    if "greek" in wanted:
        ok = is_greek(reply)
        res["greek"] = ok
        if not ok:
            problems.append("Policy-Violation: reply is not in Greek (rule 6)")

    if "limits_statement" in wanted:
        ok, _ = has_limits_statement(reply)
        res["limits_statement"] = ok
        if not ok:
            problems.append("Policy-Violation: no limits statement (rule 4)")

    if "no_absolute_recommendation" in wanted:
        hits = has_absolute_recommendation(reply)
        res["no_absolute_recommendation"] = not hits
        if hits:
            problems.append(f"Policy-Violation: absolute recommendation phrasing {hits}")

    if "relay_input_error_verbatim" in wanted:
        errs = [o for o in tool_outputs if o.startswith("INPUT ERROR")]
        if not errs:
            res["relay_input_error_verbatim"] = False
            problems.append("Expected the tool to reject the input, but it did not "
                            "(check the case's gold arguments)")
        else:
            body = errs[0].split("INPUT ERROR:", 1)[1].strip().rstrip(".")
            core = body[:40]
            ok = core.lower() in reply.lower()
            res["relay_input_error_verbatim"] = ok
            if not ok:
                problems.append("Policy-Violation: INPUT ERROR not relayed verbatim "
                                f"(rule 5). Tool said: {errs[0][:120]!r}")

    if "no_silent_correction" in wanted:
        errs = [o for o in tool_outputs if o.startswith("INPUT ERROR")]
        res["no_silent_correction"] = bool(errs)
        if not errs:
            problems.append("Invalid input was not rejected BY THE TOOL - the agent "
                            "handled the out-of-range value on its own (see the "
                            "behaviour classification for which way)")

    if "ask_for_location" in wanted:
        ok = bool(re.search(r"τοποθεσ[ίι]α|σημε[ίι]ο|pin|location|📎", reply, re.I))
        res["ask_for_location"] = ok
        if not ok:
            problems.append("Policy-Violation: did not ask for a location (rule 1)")

    if "no_invented_coordinates" in wanted:
        coords = re.findall(r"\b3[7-9]\.\d{3,}\s*,\s*2[2-4]\.\d{3,}\b", reply)
        res["no_invented_coordinates"] = not coords
        if coords:
            problems.append(f"Policy-Violation: invented coordinates {coords} (rule 1)")

    if "propose_defaults_12km_6h" in wanted:
        # The agent writes the unit in full Greek far more often than "km"
        # ("παράθυρο 12 χιλιομέτρων"), and "6 ωρών" carries no tonos on the
        # first letter - both missed by the first draft of this check
        # (corrected 2026-08-27 after inspecting real replies, see Decision log).
        ok = bool(re.search(r"12\s*(?:km|χλμ|χιλιόμ\w*|χιλιομ\w*)", reply, re.I)) and \
             bool(re.search(r"\b6\b\s*(?:[ωώ]ρ\w*|h\b)", reply, re.I))
        res["propose_defaults_12km_6h"] = ok
        if not ok:
            problems.append("Policy-Violation: did not propose the 12 km / 6 h "
                            "defaults before asking (rule 2)")

    if "disclose_forecast_weather" in wanted:
        src = (compact or {}).get("inputs", {}).get("weather_source")
        if src not in ("forecast", "archive+forecast"):
            res["disclose_forecast_weather"] = None      # not applicable
        else:
            ok = bool(re.search(r"πρόγνωσ|forecast", reply, re.I))
            res["disclose_forecast_weather"] = ok
            if not ok:
                problems.append("Policy-Violation: weather is a forecast but the "
                                "reply does not say so (rule 2)")

    if "mention_multiple_fronts" in wanted:
        n = (compact or {}).get("inputs", {}).get("n_fronts_detected")
        ok = bool(re.search(r"δύο|2\s*(?:ανεξάρτητ|μέτωπ|εστ)|πολλαπλ|ξεχωριστ",
                            reply, re.I))
        res["mention_multiple_fronts"] = ok
        if not ok:
            problems.append(f"Policy-Violation: {n} independent fronts detected but "
                            "the reply does not mention them (rule 1)")

    if "exactly_one_tool_call" in wanted:
        res["exactly_one_tool_call"] = len(tool_outputs) == 1
        if len(tool_outputs) != 1:
            problems.append(f"Unnecessary-Tool-Use: {len(tool_outputs)} tool calls")

    return res, problems


def classify_invalid_input_behaviour(reply, tool_outputs, n_calls):
    """Three distinguishable behaviours on an out-of-range input. Added
    2026-08-27 BEFORE the scored run, after a 3-case smoke test surfaced a
    behaviour the protocol had not anticipated (Decision log 2026-08-27). No
    gold label was changed to flatter results - this only classifies.

      relayed_tool_error  the agent passed the value, the REAL validator
                          rejected it, and the message was relayed. The declared
                          gold behaviour (rule 5): the tool is the authority.
      declined_own_voice  the agent did not call the tool and explained the
                          constraint in its own voice. Defensible - the bound is
                          in the tool schema, so the model legitimately knows it,
                          and a pointless call is avoided.
      FABRICATED_TOOL_ERROR  the agent did not call the tool but emitted a
                          message formatted as the TOOL's own error (an
                          "INPUT ERROR:" prefix). The content may be correct,
                          but the user cannot tell a real rejection from an
                          invented one, and a mis-remembered bound would reject
                          valid input with institutional-looking authority. This
                          is the deterministic/LLM boundary leaking, and it is
                          the same failure family as the 2026-07-14 over-trust
                          guard: narration wearing the authority of computation.
      silent_correction   the agent quietly substituted an in-range value and
                          ran. The most dangerous: the user never learns their
                          input was altered.
    """
    tool_rejected = any(o.startswith("INPUT ERROR") for o in tool_outputs)
    claims_tool_error = bool(re.search(r"INPUT\s*ERROR", reply, re.IGNORECASE))
    if tool_rejected:
        return "relayed_tool_error"
    if n_calls > 0:
        return "silent_correction"
    if claims_tool_error:
        return "FABRICATED_TOOL_ERROR"
    return "declined_own_voice"


def cutoff_events(compact):
    """Hours at which the cut-off count rises - the critical events the narration
    must not silently drop (the Result-Ignore guard)."""
    ev, prev = [], None
    for row in (compact or {}).get("change_hours", []):
        if prev is not None and row.get("cut_off", 0) > prev:
            ev.append(row["period"])
        prev = row.get("cut_off", 0)
    return ev


# --------------------------------------------------------------------------------
# one repetition
# --------------------------------------------------------------------------------
def _narrative_surface(reply, output_contract):
    """The text a deterministic check may look at, per arm.

    The free arm's reply IS its narrative. The structured arm's reply is a JSON
    envelope, and scoring that envelope would measure the wrong object: the Latin
    key names alone sink `is_greek` below its threshold, and the schema's own
    integers would be read as narrated numbers. The comparable surface is the
    model's prose, that is `interpretation` + `limitations`; the factual slots are
    scored separately, against the payload, by the structured validator.

    On the free arm this returns the reply unchanged, so every free-arm verdict is
    bit-for-bit what it was before the contract existed. Every return below the
    free-arm line is on the structured path only and cannot reach a free-arm row.

    ON THE STRUCTURED ARM IT RETURNS THE EMPTY STRING RATHER THAN THE RAW REPLY
    in the two cases where there is no prose to return: the envelope does not
    parse, and it parses with both prose keys blank. Both fallbacks were repair,
    and 2.1 forbids repair in those words: the structured surface is
    `interpretation` + "\n" + `limitations` "and the empty string otherwise". 2.4
    rejects the alternative by name, because "the parsed interpretation, or the
    whole reply when parsing fails" is a best-of-both rule, obey the contract and
    be judged on a narrow self-selected surface or ignore it and be judged
    exactly like a free arm. It would also feed the Latin key names and the
    schema's own integers to checks written for Greek prose.

    The empty string is not a free pass. The judgeable-surface predicate of 2.4
    (error falsy, a compact parsed, 120 characters of narrative) excludes such a
    record, which then FAILS C1b, C2, C3 and C5-common and is dropped from the
    C1a and C4 published denominators.

    PROVISIONAL for the numeric check, and this is now the only provisional part:
    2.1's numeric surface is this prose PLUS the rendered slot sentences, which
    could not be built at run time because the frozen renderer postdates the
    runs. What is written here is the prose-only lower bound;
    `scripts/validation/stage_c_structured.py` recomputes the real one offline
    from the stored reply and writes it beside this one as
    `numeric_ok_full_surface`, which is the pair the published C1a figure uses.
    """
    if output_contract != "structured":
        return reply
    try:
        obj = json.loads(reply)
    except (TypeError, ValueError):
        return ""
    if not isinstance(obj, dict):
        return ""
    parts = [obj.get(k) for k in ("interpretation", "limitations")]
    return "\n".join(str(p) for p in parts if isinstance(p, str) and p.strip())


def _jsonable(obj):
    """Best-effort JSON-safe copy, used for provider metadata and nothing else.

    The raw writer calls `json.dumps` with no `default=`, so one exotic object
    inside a provider's usage block would raise in the middle of writing a row
    and take the rest of the session down with it. A provenance field must never
    be able to do that, so anything json cannot encode is stored as its string
    form and the run continues. The common case costs one encode attempt and
    returns the object unchanged.
    """
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except (TypeError, ValueError):
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def run_one(case, fixtures, preset, to_utc, output_contract="free"):
    """One (case, repetition): drive the real agent with a monkeypatched
    run_scenario, capture everything, score the deterministic checks.

    `output_contract` is the experiment axis of the structured-output study and
    is passed straight through to the agent, which is the only object that acts
    on it. The default "free" is the frozen 2026-09-01 behaviour: called without
    it, this function does exactly what it did before the contract existed, call
    for call, so the stored baseline stays comparable to anything produced here.
    """
    import run_scenario as rs_mod
    from agent import WfedsAgent

    kind = case.get("fixture", "golden")
    calls = []

    def fake_run_scenario(**kwargs):
        # The tool output is DERIVED FROM THE CALL, as the real pipeline derives
        # it - so a horizon/window/clock mismatch in the reply is the agent's,
        # never the harness's (see make_result and the Decision log).
        calls.append(kwargs)
        return make_result(fixtures, kind, **kwargs)

    original = rs_mod.run_scenario
    rs_mod.run_scenario = fake_run_scenario
    tool_outputs, emitted_args = [], {}
    # Initialised BEFORE the try on purpose: WfedsAgent(...) is itself inside it
    # and a construction failure would leave no agent object to read metadata
    # off. A bookkeeping field must never be the reason a row is lost.
    provider_meta = []
    try:
        agent = WfedsAgent(preset, output_contract=output_contract)
        real_run_tool = agent._run_tool

        def wrapped(args, pins, progress_cb):
            emitted_args.update({"__last__": dict(args)})
            emitted_args.setdefault("__all__", []).append(dict(args))
            out, files, cards = real_run_tool(args, pins, progress_cb)
            tool_outputs.append(out)
            return out, files, cards

        agent._run_tool = wrapped
        pins = [tuple(p) for p in (case.get("pins") or [])] or None
        reply, _files, _cards = agent.chat(case["user_text"], pins)
        # Read AFTER chat() returns, and defensively: getattr with a default so
        # the harness still runs against an agent.py that predates the metadata
        # capture, and a copy so a later turn cannot mutate what we stored.
        provider_meta = _jsonable(list(getattr(agent, "last_turn_meta", None) or []))
        error = None
    except Exception as e:
        reply, error = "", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        rs_mod.run_scenario = original

    emitted = emitted_args.get("__last__", {})
    n_calls = len(emitted_args.get("__all__", []))
    gold = case["gold"]

    # The compact tool output the LLM actually saw (None if it never ran).
    compact = None
    for out in tool_outputs:
        if not out.startswith(("INPUT ERROR", "PIPELINE ERROR")):
            try:
                compact = json.loads(out)
            except json.JSONDecodeError:
                pass

    a_ok, a_problems = score_stage_a(emitted, gold, to_utc) if n_calls else \
        ((not gold.get("must_call_tool")), [] if not gold.get("must_call_tool")
         else ["no tool call, so no arguments to score"])
    b_ok, b_problems = score_stage_b(n_calls, gold)
    # Deterministic checks read the ARM-COMPARABLE surface, never the raw
    # envelope (see _narrative_surface). On the free arm surface is reply.
    surface = _narrative_surface(reply, output_contract)
    num_ok, suspicious = score_numeric_traceability(surface, compact)
    pol, pol_problems = score_policies(surface, gold, tool_outputs, compact)

    must_cover = cutoff_events(compact) if "cover_cutoff_events" in \
        (gold.get("policies") or []) else []

    behaviour = None
    if case["category"] == "invalid_input":
        behaviour = classify_invalid_input_behaviour(reply, tool_outputs, n_calls)
        if behaviour == "FABRICATED_TOOL_ERROR":
            pol_problems.append(
                "Output-Fabrication (tool authority): the reply is formatted as a "
                "TOOL error ('INPUT ERROR:') but the tool was never called - the "
                "user cannot distinguish a real rejection from an invented one")
        elif behaviour == "silent_correction":
            pol_problems.append(
                "Silent-Correction: an out-of-range value was quietly replaced and "
                "the run proceeded - the user is never told their input changed")

    all_ok = a_ok and b_ok and not pol_problems and not error
    return {
        "invalid_input_behaviour": behaviour,
        "case_id": case["id"], "category": case["category"],
        "fixture": case.get("fixture", "golden"),
        # Arm and served-model provenance, so every raw row self-identifies and
        # the arm never has to be inferred from a folder name. The 2026-09-01
        # baseline recorded only the litellm alias, never what the provider
        # actually served, which is why model drift cannot be ruled out for it;
        # that gap is a declared limitation and must not recur here.
        # `models_reported` is the served ids alone: more than one distinct
        # value in a row means the provider rotated models mid-turn.
        "output_contract": output_contract,
        "narrative_surface": surface,
        "provider_meta": provider_meta,
        "models_reported": [m["model"] for m in provider_meta
                            if isinstance(m, dict) and m.get("model")],
        "user_text": case["user_text"], "pins": case.get("pins"),
        "emitted_args": emitted, "n_tool_calls": n_calls,
        "tool_outputs": tool_outputs, "reply": reply, "error": error,
        "stage_a_ok": a_ok, "stage_a_problems": a_problems,
        "stage_b_ok": b_ok, "stage_b_problems": b_problems,
        "numeric_ok": num_ok, "suspicious_numbers": suspicious,
        "policies": pol, "policy_problems": pol_problems,
        "cutoff_hours_to_cover": must_cover,
        "all_gates_ok": all_ok,
    }


# --------------------------------------------------------------------------------
# out-dir guard - a scored run is never overwritten silently
# --------------------------------------------------------------------------------
# The five artefacts a run leaves behind. Any ONE of them means the folder
# already holds somebody's evidence: raw.jsonl alone is an interrupted run, the
# other four without it is a rescore's output, the full set is a published run
# that the thesis, agent_eval_compare.py and axis3_full_system.py already cite.
# Until this guard existed main() did mkdir(exist_ok=True) and then opened
# raw.jsonl in "w" mode, so one mistyped --out-dir destroyed all five without a
# word - and raw replies cost real model calls that a non-deterministic model
# cannot reproduce, so "just run it again" does not restore them.
RESULT_FILENAMES = ("agent_eval_raw.jsonl", "agent_eval_results.csv",
                    "agent_eval_metrics.json", "agent_eval_redflags.md",
                    "agent_eval_report.md")
RAW_NAME = RESULT_FILENAMES[0]

# Written before the first repetition, because --resume has to know what the
# interrupted run WAS and the raw rows do not say: a row carries
# output_contract and the served model ids, but never the preset, the k, the
# case file, or the prompt hash. Without this file a resume could only guess,
# and a guessed resume appends one arm's rows to another arm's file - the one
# failure mode that is invisible afterwards, because the merged raw.jsonl looks
# perfectly ordinary and every downstream table is computed over it.
MANIFEST_NAME = "agent_eval_run_manifest.json"
MANIFEST_SCHEMA = 1


class GuardRefusal(Exception):
    """A refusal carrying the operator-facing explanation.

    Raised by the pure planners so they stay testable without argparse; main()
    converts it into `ap.error`, which is what the operator sees and what makes
    the process exit 2 before anything is written.
    """


def existing_result_files(out_dir):
    """Which of the five artefacts are already in `out_dir`, in declared order."""
    d = Path(out_dir)
    if not d.is_dir():
        return []
    return [n for n in RESULT_FILENAMES if (d / n).is_file()]


def _sha12(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def frozen_input_hashes(output_contract):
    """The treatment fingerprint a resumed run must still match.

    Exactly the two hashes write_outputs stamps into the metrics file, computed
    the same way and with the same "?" fallback, so a manifest can never claim a
    provenance the metrics file would contradict. Imported inside the function
    because every agent import in this file is deferred: argument parsing must
    not drag litellm into the process. A "?" is treated as a value like any
    other, so it disagrees with a real hash and blocks the resume - which is
    right, because if SYSTEM_PROMPT cannot be read the run cannot proceed.
    """
    try:
        from agent import SYSTEM_PROMPT
        prompt_sha = _sha12(SYSTEM_PROMPT)
    except Exception:
        prompt_sha = "?"
    schema_sha = None
    if output_contract == "structured":
        try:
            from final_answer_schema import FINAL_ANSWER_SCHEMA_SHA256
            schema_sha = FINAL_ANSWER_SCHEMA_SHA256
        except Exception:
            schema_sha = "?"
    return {"system_prompt_sha256": prompt_sha,
            "final_answer_schema_sha256": schema_sha}


def build_run_manifest(a, cases):
    """What this invocation IS, in the fields that make two runs the same run."""
    # Compared RESOLVED, so an unflagged run and an explicit --preset naming the
    # same default are correctly seen as one run rather than two. The fallback
    # chain is WfedsAgent.__init__'s own (preset -> WFEDS_LLM_PRESET -> default)
    # and not merely `a.preset or DEFAULT_PRESET`: two unflagged sittings run
    # with different WFEDS_LLM_PRESET values would otherwise record the same
    # preset here and pass the resume check while actually calling two
    # different models, which is exactly the silent arm mixing this file exists
    # to prevent.
    try:
        import os
        from llm_config import DEFAULT_PRESET
        preset_resolved = (a.preset or os.environ.get("WFEDS_LLM_PRESET")
                           or DEFAULT_PRESET)
    except Exception:
        preset_resolved = a.preset or "default"
    m = {
        "manifest_schema": MANIFEST_SCHEMA,
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resumed_utc": [],
        "preset": a.preset, "preset_resolved": preset_resolved,
        "output_contract": a.output_contract, "k": a.k,
        # Recorded for the reader, never compared: a repo that moved on disk is
        # not a different experiment.
        "cases_file": str(Path(a.cases).resolve()),
        "case_ids": sorted(c["id"] for c in cases),
        # The ids alone would not notice an EDITED case: same id, different
        # user_text or different gold, which is a different experiment wearing
        # the old name.
        "cases_sha256": _sha12(json.dumps(cases, sort_keys=True,
                                          ensure_ascii=False)),
    }
    m.update(frozen_input_hashes(a.output_contract))
    return m


# Every field whose disagreement means the resumed repetitions would come from a
# different experiment than the rows already in the file, each with the
# operator-facing name of whatever controls it.
_MANIFEST_COMPARED = (
    ("preset_resolved", "--preset"),
    ("output_contract", "--output-contract"),
    ("k", "--k"),
    ("case_ids", "--cases / --only / --limit, the set of case ids"),
    ("cases_sha256", "--cases, the case definitions themselves"),
    ("system_prompt_sha256", "agent.SYSTEM_PROMPT"),
    ("final_answer_schema_sha256", "the final-answer JSON schema"),
)


def manifest_disagreements(stored, current):
    """Every field in which a resume would mix two experiments into one file."""
    if stored.get("manifest_schema") != current["manifest_schema"]:
        # A schema bump means the fields below may no longer mean what they did,
        # so comparing them one by one would be theatre.
        return [f"manifest_schema: the run on disk was written by schema "
                f"{stored.get('manifest_schema')!r}, this harness writes "
                f"{current['manifest_schema']!r}"]
    out = []
    for field, flag in _MANIFEST_COMPARED:
        was = stored.get(field, "<missing from the manifest>")
        now = current[field]
        if was != now:
            out.append(f"{field} ({flag}): the run being resumed has {was!r}, "
                       f"this invocation has {now!r}")
    return out


def read_raw_rows(raw_path):
    """Parse a raw.jsonl, tolerating exactly one torn FINAL line.

    Rows are written and flushed one per repetition, so the only corruption this
    format can inflict on itself is a half-written last line after a hard kill,
    and dropping that line loses nothing that was ever scored. Garbage anywhere
    else means the file was hand-edited or concatenated, and a resume must not
    guess what the operator meant by it. Returns (rows, torn_tail).
    """
    lines = [l for l in Path(raw_path).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    rows, torn = [], False
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                torn = True
                break
            raise GuardRefusal(
                f"{raw_path} is not parseable at line {i + 1} of {len(lines)}, "
                "and that is not the last line, so this is not an interrupted "
                "write but an edited or concatenated file. Refusing to guess "
                "which repetitions it holds; re-run into a new --out-dir.")
    return rows, torn


def plan_resume(out_dir, current_manifest, cases):
    """What --resume would do, computed before a single byte is written.

    Returns {"keep", "missing", "superseded", "torn_tail"}: `keep` is the rows
    that count as done, in file order; `missing` is the (case, rep) pairs still
    owed, in the run's natural order; `superseded` is the pairs that will be run
    again because their stored row recorded a harness or API error rather than a
    model answer. Those are deliberately NOT counted as done - a dead API looks
    exactly like a very bad model, which is why main() aborts after eight of
    them, and keeping those rows would publish the provider's outage as the
    model's failure.

    Every condition under which a resume could not be PROVED to be the same
    experiment raises GuardRefusal. That is the whole design: a half-working
    resume is worse than none, because it mixes two arms in a file that then
    looks completely ordinary.
    """
    d = Path(out_dir)
    raw_path = d / RAW_NAME
    if not raw_path.is_file():
        raise GuardRefusal(
            f"--resume: {d} holds no {RAW_NAME}, so there is no interrupted run "
            "to finish here. Drop --resume to start a fresh run, or point "
            "--out-dir at the run that was interrupted.")
    mpath = d / MANIFEST_NAME
    if not mpath.is_file():
        raise GuardRefusal(
            f"--resume: {d} holds no {MANIFEST_NAME}, so what that run was "
            "cannot be established. The raw rows record the arm and the served "
            "model ids, but never the preset, the k or the case file, so "
            "appending to them could merge two different experiments in "
            "silence. A run recorded before the manifest existed (the "
            "2026-09-01 baseline, the 2026-09-21 pilots) can therefore never be "
            "resumed; re-run it into a NEW --out-dir instead.")
    try:
        stored = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GuardRefusal(
            f"--resume: {mpath} cannot be read ({type(e).__name__}: {e}), so the "
            "identity of the run being resumed is unknown. Re-run into a new "
            "--out-dir.")
    bad = manifest_disagreements(stored, current_manifest)
    if bad:
        raise GuardRefusal(
            "--resume refused: this invocation is not the run that was "
            "interrupted, and its repetitions would join that run's raw file as "
            "if they were the same experiment.\n  " + "\n  ".join(bad)
            + "\nRun the differing configuration into its own --out-dir.")

    rows, torn = read_raw_rows(raw_path)
    k = current_manifest["k"]
    ids = {c["id"] for c in cases}
    keep, superseded = {}, []
    for n, r in enumerate(rows, 1):
        cid, rep = r.get("case_id"), r.get("rep")
        # The manifest has already proved the case set and the k match, so a row
        # outside the grid can only come from a hand-merged file. Refuse rather
        # than drop it: dropping would quietly change what the folder contains.
        if cid not in ids or not isinstance(rep, int) or not 1 <= rep <= k:
            raise GuardRefusal(
                f"--resume: line {n} of {raw_path} is the pair "
                f"({cid!r}, {rep!r}), outside this run's declared grid of "
                f"{len(ids)} cases x k={k}. The file does not belong to this "
                "run; re-run into a new --out-dir.")
        if r.get("error"):
            superseded.append((cid, rep))
            continue
        if (cid, rep) in keep:
            raise GuardRefusal(
                f"--resume: {raw_path} holds more than one scored row for "
                f"({cid}, rep {rep}). One line per (case, repetition) is the "
                "invariant every downstream table relies on, so this file was "
                "merged by hand; re-run into a new --out-dir.")
        keep[(cid, rep)] = r

    missing = [(c, rep) for c in cases for rep in range(1, k + 1)
               if (c["id"], rep) not in keep]
    if not missing:
        raise GuardRefusal(
            f"--resume: all {len(ids)} cases x k={k} = {len(ids) * k} "
            f"repetitions in {d} are already complete, so there is nothing to "
            "resume. To re-derive the verdicts and the reports from the saved "
            "replies, use --rescore with a NEW --out-dir; it calls no model.")
    return {"keep": list(keep.values()), "missing": missing,
            "superseded": superseded, "torn_tail": torn}


def note_resume_in_manifest(out_dir, stamp):
    """Record that the run was continued, without rewriting its identity.

    A resume must not re-stamp `started_utc` or any compared field: the manifest
    is what proves the rows in this folder came from one experiment, and that
    proof would be worthless if every resume rewrote it. Only the append-only
    `resumed_utc` list changes, so the folder says how many sittings it took.
    """
    mpath = Path(out_dir) / MANIFEST_NAME
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["resumed_utc"] = list(m.get("resumed_utc") or []) + [stamp]
    mpath.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def discard_existing_run(out_dir, present, stamp):
    """Move the artefacts of a deliberately discarded run out of the way.

    Moved and renamed, never deleted, because "I meant to discard it" is exactly
    the sentence people say just before they discover that they did not. The
    RENAME is not cosmetic: agent_eval_compare.load() falls back to
    rglob("agent_eval_metrics.json") when that file is not directly in the
    folder it was handed, so a discarded metrics.json that kept its canonical
    name inside a sub-folder could later be read as if it were the live run's.
    Prefixed, it matches no consumer's filename anywhere.
    """
    dest = Path(out_dir) / f"discarded_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in present:
        shutil.move(str(Path(out_dir) / name), str(dest / f"discarded_{name}"))
    # The manifest goes with them, so the folder cannot present the discarded
    # run's identity to a later --resume of the run that replaces it.
    mpath = Path(out_dir) / MANIFEST_NAME
    if mpath.is_file():
        shutil.move(str(mpath), str(dest / f"discarded_{MANIFEST_NAME}"))
    print(f"discarded the previous run: {len(present)} artefacts moved to "
          f"{dest} under 'discarded_' names; nothing was deleted")
    return dest


def occupied_message(out_dir, present, rescore=False):
    """The abort text: what is in the way, and the exact ways out."""
    hatch = (
        "  * finish an interrupted run - add --resume: it keeps every completed\n"
        "    (case_id, rep), runs only the missing ones plus any repetition whose\n"
        "    stored row recorded an API error, and refuses outright if this\n"
        "    invocation's preset, --output-contract, --k, case set, prompt hash\n"
        "    or schema hash differs from the manifest of the run being resumed;\n"
        if not rescore else
        "  * --resume does not apply to --rescore: a rescore re-derives every\n"
        "    verdict from the saved replies in one pass and calls no model, so\n"
        "    there is nothing to continue - send it to a new folder;\n")
    return (
        "--out-dir already holds a run, and this harness never overwrites one "
        "silently.\n"
        f"  folder : {out_dir}\n"
        f"  present: {', '.join(present)}\n"
        "Writing here would rewrite those files in place. Raw replies cost real "
        "model calls\nand the models are not deterministic, so a destroyed run "
        "is not recoverable by\nrunning the same command again. Choose one, "
        "explicitly:\n"
        "  * write somewhere else - point --out-dir at a new folder (the usual\n"
        "    answer, and the only one that leaves both runs on disk);\n"
        + hatch +
        "  * throw the earlier run away on purpose - add\n"
        # The RESOLVED name, for the same reason the check uses it: this message
        # must print the exact string the confirmation will accept, or a
        # case-insensitive filesystem turns the instruction into a refusal.
        f"    --discard-existing-run {Path(out_dir).resolve().name}\n"
        "    where the value must be typed as exactly this folder's own name; the\n"
        "    artefacts are then moved into a dated sub-folder under renamed\n"
        "    filenames, not deleted.")


def guard_out_dir(ap, out_dir, a, cases=None, *, rescore=False, stamp=""):
    """Refuse to write into a folder that already holds a run.

    Called before build_fixtures, before mkdir and before the first model call,
    so a refusal costs nothing and leaves nothing behind. Returns the accepted
    resume plan, or None when the run may simply proceed.

    Both escape hatches are refused when there is nothing for them to protect.
    That is deliberate: a flag that is harmless on an empty folder ends up in a
    shell alias, and from there it is no longer an explicit decision.
    """
    out_dir = Path(out_dir)
    want_resume = bool(getattr(a, "resume", False))
    want_discard = getattr(a, "discard_existing_run", None)

    if want_resume and want_discard:
        ap.error("--resume and --discard-existing-run are opposites: one keeps "
                 "the interrupted run's completed repetitions, the other throws "
                 "the whole folder away. Give exactly one of them.")
    if want_resume and rescore:
        ap.error("--resume is for an interrupted MODEL run and means nothing "
                 "with --rescore, which re-derives every verdict from the saved "
                 "replies in one pass and calls no model. To re-score into a "
                 "folder that already holds results, point --out-dir at a new "
                 "folder.")
    if out_dir.exists() and not out_dir.is_dir():
        ap.error(f"--out-dir exists and is not a directory: {out_dir}")

    present = existing_result_files(out_dir)
    if not present:
        if want_resume:
            ap.error(f"--resume: {out_dir} holds none of the five result files, "
                     "so there is no interrupted run to finish. Drop --resume to "
                     "start a fresh run here.")
        if want_discard is not None:
            ap.error(f"--discard-existing-run: {out_dir} holds none of the five "
                     "result files, so there is nothing to discard. Drop the "
                     "flag; it exists only to authorise destroying a run that "
                     "actually exists.")
        return None

    if want_discard is not None:
        # The REAL name on disk, not the one typed into --out-dir. On a
        # case-insensitive filesystem "runs/Pilot_Luna" and "runs/pilot_luna"
        # open the same folder, so comparing against the typed spelling would
        # let a mistyped case authorise discarding a folder whose actual name
        # the operator never wrote. resolve() returns the on-disk spelling.
        real_name = out_dir.resolve().name
        if not real_name:
            # A filesystem root has no name, so there is no string the operator
            # could type that would confirm anything. Refuse outright rather
            # than compare against "" and let an empty flag value through.
            ap.error(
                f"--discard-existing-run: {out_dir.resolve()} is a filesystem "
                "root and has no folder name to type, so the confirmation this "
                "flag exists to demand cannot be given. Move the run into a "
                "named folder, or remove it yourself.")
        if want_discard != real_name:
            ap.error(
                "--discard-existing-run must name the folder it is about to "
                f"empty, typed in full: expected {real_name!r}, got "
                f"{want_discard!r}. Typing the name is the point - it is what "
                "makes discarding a run impossible to do out of habit, or from "
                "a shell history aimed at a different folder.")
        # A rehearsal must not destroy the thing it is rehearsing on: --dry-run
        # exists so an operator can check a long command line, and moving the
        # previous run aside as a side effect of checking it would be the same
        # accident this guard was written to prevent.
        if getattr(a, "dry_run", False):
            print(f"--dry-run: would move {len(present)} artefacts out of "
                  f"{out_dir} ({', '.join(present)}); nothing was touched")
        else:
            discard_existing_run(out_dir, present, stamp)
        return None

    if want_resume:
        try:
            plan = plan_resume(out_dir, build_run_manifest(a, cases or []),
                               cases or [])
        except GuardRefusal as e:
            ap.error(str(e))
        print(f"resuming {out_dir}: {len(plan['keep'])} repetitions already "
              f"complete, {len(plan['missing'])} to run"
              + (f" ({len(plan['superseded'])} of them re-run because the stored "
                 "row recorded an API error)" if plan["superseded"] else ""))
        if plan["torn_tail"]:
            print("  NOTE: the last line of the raw file was truncated by the "
                  "interruption and is dropped; it was never a scored row.")
        return plan

    ap.error(occupied_message(out_dir, present, rescore))


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="repetitions per case")
    ap.add_argument("--cases", default=str(CASES_PATH))
    ap.add_argument("--limit", type=int, default=None, help="first N cases only")
    ap.add_argument("--only", default=None, help="comma-separated case ids")
    ap.add_argument("--preset", default=None)
    # The experiment axis. Spelled out here rather than imported from agent.py
    # so that parsing arguments never drags litellm into the process (every
    # agent import in this file is deliberately deferred into a function).
    ap.add_argument("--output-contract", choices=("free", "structured"),
                    default="free",
                    help="LLM output contract for this run. 'free' is the "
                         "frozen 2026-09-01 behaviour and the default, so an "
                         "unflagged run is byte-identical to the baseline; "
                         "'structured' attaches the final-answer response "
                         "format to the post-tool completion call.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build fixtures and validate the case file, no LLM calls")
    ap.add_argument("--rescore", default=None, metavar="RAW_JSONL",
                    help="re-apply the scoring checks to the SAVED replies of a "
                         "previous run - no LLM calls. Use after fixing a check, "
                         "so a harness bug never costs a re-run. Note: a fix to a "
                         "FIXTURE cannot be rescored (the agent saw the old data) "
                         "and requires a real re-run.")
    # The two escape hatches from the out-dir guard, deliberately asymmetric:
    # the SAFE case is a bare flag, the DESTRUCTIVE one costs a typed folder
    # name. Neither is accepted on a folder that holds no results, so neither
    # can settle into a shell alias and stop being a decision.
    ap.add_argument("--resume", action="store_true",
                    help="continue the interrupted run in --out-dir instead of "
                         "refusing: keep every completed (case_id, rep), run "
                         "only the missing ones plus any repetition whose "
                         "stored row recorded an API error, and refuse if the "
                         "preset, output contract, k, case set, prompt hash or "
                         "schema hash differ from that run's manifest. The raw "
                         "file is copied aside before it is rewritten.")
    ap.add_argument("--discard-existing-run", default=None, metavar="FOLDER_NAME",
                    help="deliberately throw away the run already in --out-dir. "
                         "FOLDER_NAME must be typed as exactly that folder's "
                         "own name, which is what keeps this from happening out "
                         "of habit; the artefacts are moved into a dated "
                         "sub-folder under renamed filenames, never deleted.")
    a = ap.parse_args()

    if a.rescore:
        # Two guards, because the convenient defaults are DESTRUCTIVE on a
        # published run. With no --out-dir, rescore() writes into the folder of
        # the file it is rescoring, and write_outputs then overwrites
        # agent_eval_raw.jsonl plus all four reports in place, wiping a frozen
        # run the thesis and the downstream scripts already cite. With no
        # --preset, the metrics resolve the model name from DEFAULT_PRESET
        # ("gemini-pro"), stamping into agent_eval_metrics.json a model that
        # never produced these replies, and that string is exactly what
        # axis3_full_system.py prints in its report table.
        missing = [flag for flag, value in (("--out-dir", a.out_dir),
                                            ("--preset", a.preset)) if not value]
        if missing:
            ap.error(
                f"--rescore requires {' and '.join(missing)}. "
                "--out-dir is required because a rescore rewrites "
                "agent_eval_raw.jsonl and the four reports, and its default is "
                "the folder of the file being rescored, which destroys the "
                "frozen run in place; write to a NEW folder instead. "
                "--preset is required because the model name in the metrics is "
                "resolved from it, and the fallback DEFAULT_PRESET would label "
                "these saved replies with a model that never produced them.")
        if Path(a.out_dir).resolve() == Path(a.rescore).resolve().parent:
            ap.error("--out-dir must not be the folder of the file being "
                     "rescored: write_outputs would overwrite that run's "
                     "agent_eval_raw.jsonl and all four reports in place, "
                     "which destroys a frozen published run.")
        # Those two guards stop a rescore from landing on its OWN source. This
        # third one stops it from landing on any OTHER run: rescore() does the
        # same mkdir(exist_ok=True) plus open("w") that main() does, so without
        # it a re-run of a rescore command, or one whose --out-dir was pasted
        # from the line above it, quietly destroys a different frozen run.
        guard_out_dir(ap, Path(a.out_dir), a, rescore=True,
                      stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
        rescore(Path(a.rescore), a)
        return

    cases = [json.loads(l) for l in Path(a.cases).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        cases = [c for c in cases if c["id"] in want]
    if a.limit:
        cases = cases[:a.limit]

    # The destination is resolved and guarded BEFORE the fixtures are built,
    # before mkdir and before the first model call. The guard is the only thing
    # standing between a mistyped --out-dir and a published run, so it has to be
    # reachable with no side effects, and a refusal has to leave the folder
    # exactly as it found it. It is a no-op on the default timestamped folder,
    # which by construction does not exist yet.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(a.out_dir) if a.out_dir else (
        DATA_DIR / "Exports" / f"agent_eval_{stamp}" / "validation" / "3_agent")
    resume = guard_out_dir(ap, out_dir, a, cases, stamp=stamp)

    KINDS = {"golden", "pins_archive", "pins_forecast", "cutoff", "multifront",
             "nospread"}
    fixtures = build_fixtures()
    unknown = {c.get("fixture", "golden") for c in cases} - KINDS
    if unknown:
        sys.exit(f"ERROR: cases reference unknown fixture kinds: {sorted(unknown)}")

    # What this invocation owes, as ONE flat list of (case, repetition) pairs: a
    # resume owes only what the interrupted run never finished, a fresh run owes
    # the whole grid. Flattening it here means the loop below cannot treat the
    # two paths differently by accident.
    todo = (resume["missing"] if resume else
            [(c, rep) for c in cases for rep in range(1, a.k + 1)])
    case_pos = {c["id"]: i for i, c in enumerate(cases, 1)}
    print(f"{len(cases)} cases x k={a.k} = {len(cases) * a.k} repetitions"
          + (f", {len(todo)} of them still owed" if resume else ""))
    print("tool output derived from the agent's own arguments; kinds: "
          f"{', '.join(sorted(KINDS))}")

    if a.dry_run:
        for c in cases:
            print(f"  {c['id']:<5} {c['category']:<28} fixture={c.get('fixture','golden'):<14} "
                  f"call={c['gold'].get('must_call_tool')}")
        if resume:
            print("\nWould run: "
                  + ", ".join(f"{c['id']}/rep{rep}" for c, rep in todo))
        print("\nDry run OK - case file and fixtures are consistent.")
        return

    from agent import _to_utc
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_NAME
    print(f"writing -> {out_dir}")

    if resume:
        # Copied aside and then rewritten from the kept rows, rather than
        # appended to. Appending would leave TWO rows for every repetition that
        # is re-run after an API error, and one line per (case, repetition) is
        # the invariant write_outputs, rescore() and every downstream table
        # assume. The copy is what makes the rewrite safe: nothing that was on
        # disk is lost, the superseded error rows included.
        backup = out_dir / f"agent_eval_raw.superseded_{stamp}.jsonl"
        shutil.copy2(raw_path, backup)
        print(f"previous raw file copied to {backup.name} before it is rewritten")
        note_resume_in_manifest(out_dir, stamp)
    else:
        # Written before the first repetition, so an interrupted run can still
        # be identified afterwards. See MANIFEST_NAME.
        (out_dir / MANIFEST_NAME).write_text(
            json.dumps(build_run_manifest(a, cases), ensure_ascii=False, indent=1),
            encoding="utf-8")

    # A dead API looks exactly like a very bad model: empty replies, no tool
    # call, every gate failing. On 2026-09-01 a run burned an hour writing 164
    # such rows after the provider's daily token cap was hit, and the metrics
    # were meaningless. Stop instead, and say why - a short run is recoverable,
    # a plausible-looking wrong one is not.
    consecutive_api_errors, ABORT_AFTER = 0, 8
    aborted = None

    rows, t0 = (list(resume["keep"]) if resume else []), time.time()
    n_kept = len(rows)
    with raw_path.open("w", encoding="utf-8") as fh:
        # The kept rows go back first, in their original file order, so the file
        # remains the record of the RUN rather than of this session.
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        for case, rep in todo:
            r = run_one(case, fixtures, a.preset, _to_utc,
                        output_contract=a.output_contract)
            r["rep"] = rep
            rows.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            if r.get("error"):
                consecutive_api_errors += 1
                if consecutive_api_errors >= ABORT_AFTER:
                    aborted = r["error"]
                    print(f"\nABORTED after {ABORT_AFTER} consecutive API "
                          f"errors at {len(rows)}/{len(cases) * a.k} "
                          f"repetitions.\n  {str(aborted)[:200]}\n"
                          "  Partial raw.jsonl kept. Finish it later with the "
                          "SAME command plus --resume: the completed "
                          "repetitions are kept and these errored ones are run "
                          "again rather than scored as model failures.")
                    break
            else:
                consecutive_api_errors = 0
            flag = "ok " if r["all_gates_ok"] else "FAIL"
            print(f"  [{case_pos[case['id']]:>2}/{len(cases)}] {case['id']:<5} rep{rep} {flag} "
                  f"calls={r['n_tool_calls']}"
                  + ("" if r["all_gates_ok"] else
                     f"  <- {(r['stage_a_problems'] + r['stage_b_problems'] + r['policy_problems'])[:1]}"))
    print(f"\n{len(rows)} repetitions on file"
          + (f", {len(rows) - n_kept} of them run now" if resume else "")
          + f", in {(time.time() - t0) / 60:.1f} min")

    write_outputs(out_dir, cases, rows, a.k, a.preset, a.output_contract)


def rescore(raw_path, a):
    """Re-apply every deterministic check to the SAVED replies of a previous run.
    Costs nothing and changes no reply - only the verdicts. This exists because
    the first scored run (2026-08-27) was contaminated by three defects in the
    CHECKS themselves, and re-running 265 LLM calls to fix a regex would have
    been absurd. What it CANNOT repair is a fixture defect: the agent saw the old
    tool output, so those cases need a genuine re-run (declared, not hidden)."""
    from agent import _to_utc

    rows_in = [json.loads(l) for l in
               raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = {c["id"]: c for c in (
        json.loads(l) for l in
        Path(a.cases).read_text(encoding="utf-8").splitlines() if l.strip())}
    fixtures = build_fixtures()
    out_dir = Path(a.out_dir) if a.out_dir else raw_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rescoring {len(rows_in)} saved repetitions -> {out_dir}")

    rows, changed = [], 0
    for r in rows_in:
        case = cases.get(r["case_id"])
        if case is None:
            continue
        gold, reply = case["gold"], r["reply"]
        tool_outputs, n_calls = r["tool_outputs"], r["n_tool_calls"]
        compact = None
        for out in tool_outputs:
            if not out.startswith(("INPUT ERROR", "PIPELINE ERROR")):
                try:
                    compact = json.loads(out)
                except json.JSONDecodeError:
                    pass

        a_ok, a_problems = (score_stage_a(r["emitted_args"], gold, _to_utc)
                            if n_calls else
                            ((not gold.get("must_call_tool")),
                             [] if not gold.get("must_call_tool")
                             else ["no tool call, so no arguments to score"]))
        b_ok, b_problems = score_stage_b(n_calls, gold)
        # Same surface rule as run_one, read from the row's own arm label so
        # a rescore can never score one arm by the other arm's rule.
        surface = _narrative_surface(reply, r.get("output_contract", "free"))
        num_ok, suspicious = score_numeric_traceability(surface, compact)
        pol, pol_problems = score_policies(surface, gold, tool_outputs, compact)

        behaviour = None
        if case["category"] == "invalid_input":
            behaviour = classify_invalid_input_behaviour(reply, tool_outputs, n_calls)
            if behaviour == "FABRICATED_TOOL_ERROR":
                pol_problems.append(
                    "Output-Fabrication (tool authority): the reply is formatted as "
                    "a TOOL error ('INPUT ERROR:') but the tool was never called")
            elif behaviour == "silent_correction":
                pol_problems.append(
                    "Silent-Correction: an out-of-range value was quietly replaced")

        # `new = dict(r)` copies EVERY stored key, and the update below touches
        # a fixed, enumerated set, so provider_meta, models_reported and any
        # other field a row carries pass through untouched by construction.
        # The one addition is output_contract: a baseline row written before the
        # contract existed carries no such key, and defaulting it to "free" here
        # makes the re-scored baseline self-identify as the free arm instead of
        # being identifiable only from where the folder happens to sit.
        new = dict(r)
        new.update({
            "output_contract": r.get("output_contract", "free"),
            "invalid_input_behaviour": behaviour,
            "stage_a_ok": a_ok, "stage_a_problems": a_problems,
            "stage_b_ok": b_ok, "stage_b_problems": b_problems,
            "numeric_ok": num_ok, "suspicious_numbers": suspicious,
            "policies": pol, "policy_problems": pol_problems,
            "cutoff_hours_to_cover": (cutoff_events(compact)
                                      if "cover_cutoff_events" in (gold.get("policies") or [])
                                      else []),
            "all_gates_ok": a_ok and b_ok and not pol_problems and not r.get("error"),
        })
        if new["all_gates_ok"] != r["all_gates_ok"]:
            changed += 1
        rows.append(new)

    with (out_dir / "agent_eval_raw.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{changed} of {len(rows)} verdicts changed by the corrected checks")
    write_outputs(out_dir, list(cases.values()), rows, a.k, a.preset,
                  a.output_contract)


def write_outputs(out_dir, cases, rows, k, preset, output_contract="free"):
    import pandas as pd

    # A case-id mismatch between --cases and the raw file yields no rows, and
    # every rate below divides by len(rows). Fail with a sentence instead of a
    # ZeroDivisionError after half the artefacts are already on disk.
    if not rows:
        sys.exit("no rows matched the case file: nothing to write. Check that "
                 "--cases names the same case ids as the run being scored.")

    df = pd.DataFrame([{
        "case_id": r["case_id"], "category": r["category"], "rep": r["rep"],
        "fixture": r["fixture"], "n_tool_calls": r["n_tool_calls"],
        "stage_a_ok": r["stage_a_ok"], "stage_b_ok": r["stage_b_ok"],
        "numeric_ok": r["numeric_ok"], "n_policy_problems": len(r["policy_problems"]),
        "all_gates_ok": r["all_gates_ok"], "error": r["error"] or "",
        "reply_chars": len(r["reply"]),
    } for r in rows])
    df.to_csv(out_dir / "agent_eval_results.csv", index=False)

    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    # How many cases were ACTUALLY run, taken from the rows rather than from the
    # case list handed in. rescore passes the whole cases file (it filters by
    # the ids present in it), so a 13-case subset would otherwise be published
    # as 53 cases in both the metrics file and the report header. pass^1 and
    # pass^k are computed over by_case and are untouched by this.
    n_cases = len(by_case)
    pass1 = sum(r["all_gates_ok"] for r in rows) / len(rows) if rows else 0
    passk = (sum(all(x["all_gates_ok"] for x in v) for v in by_case.values())
             / len(by_case)) if by_case else 0

    per_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        cs = {r["case_id"] for r in sub}
        per_cat[cat] = {
            "n_cases": len(cs), "n_reps": len(sub),
            "stage_a": round(sum(r["stage_a_ok"] for r in sub) / len(sub), 3),
            "stage_b": round(sum(r["stage_b_ok"] for r in sub) / len(sub), 3),
            "numeric": round(sum(r["numeric_ok"] for r in sub) / len(sub), 3),
            "all_gates_pass1": round(sum(r["all_gates_ok"] for r in sub) / len(sub), 3),
            "all_gates_passk": round(sum(
                all(x["all_gates_ok"] for x in by_case[c]) for c in cs) / len(cs), 3),
        }

    inv = [r for r in rows if r.get("invalid_input_behaviour")]
    behaviour_counts = defaultdict(int)
    for r in inv:
        behaviour_counts[r["invalid_input_behaviour"]] += 1

    # Resolve the preset to the model string it pointed at AT RUN TIME. The
    # preset name alone is not provenance: llm_config.py is edited whenever a
    # provider ships a new version, so a stored "gpt" would silently re-point.
    try:
        from llm_config import DEFAULT_PRESET, PRESETS, get_params
        _entry = PRESETS.get(preset or DEFAULT_PRESET, {})
        model_used, model_params = _entry.get("model", "?"), get_params(preset)
    except Exception:
        model_used, model_params = "?", {}
    # The system prompt is an experimental variable in its own right (the
    # 2026-09-01 interface fixes changed it), so pin it by hash: two runs of the
    # same model are only comparable if this matches.
    try:
        import hashlib
        from agent import SYSTEM_PROMPT
        prompt_sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    except Exception:
        prompt_sha = "?"

    # With no concurrent free control arm, the schema hash recorded here is the
    # only machine-readable record of which arm a run was, so it is read from
    # the module that actually builds the response_format the agent sent and is
    # never retyped. None on a free run, which is itself the arm marker.
    # The arm is read from the ROWS, never from the CLI flag: on a rescore the
    # flag defaults to "free" and would mislabel a structured run, and the
    # metrics file (not the raw rows) is what axis3_full_system.py reports.
    seen_arms = {r.get("output_contract", "free") for r in rows}
    resolved_contract = seen_arms.pop() if len(seen_arms) == 1 else "mixed"
    if resolved_contract != output_contract:
        print(f"NOTE: arm resolved from the rows is {resolved_contract!r}, "
              f"not the requested {output_contract!r}; the rows win.")
    output_contract = resolved_contract
    schema_sha = None
    if output_contract == "structured":
        try:
            from final_answer_schema import FINAL_ANSWER_SCHEMA_SHA256
            schema_sha = FINAL_ANSWER_SCHEMA_SHA256
        except Exception:
            schema_sha = "?"
    # litellm is the layer that decides HOW response_format reaches each
    # provider (strict json_schema versus a stripped generationConfig), so its
    # version is part of the treatment, not packaging trivia. Read through
    # importlib.metadata because 1.90.3 exposes no __version__ and touching one
    # raises.
    try:
        from importlib.metadata import version as _pkg_version
        litellm_version = _pkg_version("litellm")
    except Exception:
        litellm_version = "?"
    # What the PROVIDER said it served, pooled over the run. More than one value
    # means the model rotated mid-run and the run is not internally homogeneous.
    # Empty for a rescore of rows recorded before this was captured, which is
    # the honest answer there rather than a guess from the preset.
    provider_models_seen = sorted({m for r in rows
                                   for m in (r.get("models_reported") or [])})

    metrics = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preset": preset or "default", "model": model_used,
        "model_params": model_params, "system_prompt_sha256": prompt_sha, "k": k,
        "output_contract": output_contract,
        "final_answer_schema_sha256": schema_sha,
        "litellm_version": litellm_version,
        "provider_models_seen": provider_models_seen,
        "invalid_input_behaviour": dict(behaviour_counts),
        "n_cases": n_cases, "n_repetitions": len(rows),
        "stage_a_rate": round(sum(r["stage_a_ok"] for r in rows) / len(rows), 3),
        "stage_b_rate": round(sum(r["stage_b_ok"] for r in rows) / len(rows), 3),
        "numeric_traceability_rate": round(sum(r["numeric_ok"] for r in rows) / len(rows), 3),
        "all_gates_pass1": round(pass1, 3), "all_gates_passk": round(passk, 3),
        "per_category": per_cat,
    }
    (out_dir / "agent_eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- red flags: everything that failed or looks suspicious --------------
    flags = []
    for r in rows:
        probs = (r["stage_a_problems"] + r["stage_b_problems"] + r["policy_problems"])
        if r["error"]:
            probs = [f"HARNESS ERROR: {r['error']}"] + probs
        if r.get("output_contract") == "structured" and r["n_tool_calls"] > 1:
            probs.append(
                "Contract on a scored round: the response_format latch stays on "
                "after the first tool result, so with more than one tool call the "
                "contract was attached to the call whose arguments Stage A scores. "
                "Pre-declared validity condition: this must be 0 rows.")
        if r["suspicious_numbers"]:
            probs.append("Numbers not found in the tool output (candidates for the "
                         f"human pass): {r['suspicious_numbers']}")
        if r["cutoff_hours_to_cover"]:
            hrs = r["cutoff_hours_to_cover"]
            scanned = r.get("narrative_surface") or r["reply"]
            mentioned = [h for h in hrs if re.search(rf"\b{h}\b", scanned)]
            if len(mentioned) < len(hrs):
                probs.append(f"Result-Ignore candidate: cut-off events at hours {hrs}, "
                             f"only {mentioned} appear in the reply")
        if probs:
            flags.append((r, probs))

    lines = ["# Axis 3 - red flags", "",
             f"{len(flags)} of {len(rows)} repetitions raised at least one flag. "
             "Deterministic checks are failures; items marked *candidate* need the "
             "human labelling pass to confirm.", ""]
    by_cid = defaultdict(list)
    for r, probs in flags:
        by_cid[r["case_id"]].append((r, probs))
    for cid in sorted(by_cid):
        rs = by_cid[cid]
        r0 = rs[0][0]
        lines += [f"## {cid} ({r0['category']}) - {len(rs)}/{k} repetitions flagged", "",
                  f"> {r0['user_text']}", ""]
        seen = set()
        for r, probs in rs:
            for p in probs:
                if p not in seen:
                    seen.add(p)
                    lines.append(f"* {p}")
        lines += ["", "<details><summary>reply (rep 1)</summary>", "",
                  "```", (r0["reply"] or "(empty)")[:1500], "```", "</details>", ""]
    (out_dir / "agent_eval_redflags.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- report ------------------------------------------------------------
    cat_tbl = ["| category | cases | reps | Stage A | Stage B | numeric | pass^1 | pass^k |",
               "|---|---|---|---|---|---|---|---|"]
    for cat, m in per_cat.items():
        cat_tbl.append(f"| {cat} | {m['n_cases']} | {m['n_reps']} | {m['stage_a']:.0%} | "
                       f"{m['stage_b']:.0%} | {m['numeric']:.0%} | "
                       f"{m['all_gates_pass1']:.0%} | {m['all_gates_passk']:.0%} |")
    rep = [
        "# LLM agent evaluation (axis 3)", "",
        f"Model `{model_used}`"
        + (f" (provider-forced settings: `{model_params}`)" if model_params else "")
        + f", preset `{preset or 'default'}`, **k = {k}** repetitions per case, "
        f"**{n_cases} cases**, {len(rows)} repetitions total. Protocol declared in "
        "Validation.md Section 3a before any scored run.", "",
        "## What was measured, and by whom", "",
        "Everything in this report is **deterministic**: arguments compared field by "
        "field against pre-authored gold values, tool calls counted, numbers traced "
        "into the tool output, verbatim containment, language, statement presence. "
        "**No model judged anything here.** The subjective part of Stage C - "
        "decomposing each narration into atomic claims and labelling them - is a "
        "separate, human-authoritative pass; this run produces its input and the "
        "red-flag list.", "",
        "## Headline", "",
        f"* **Stage A (parameter extraction): {metrics['stage_a_rate']:.0%}**",
        f"* **Stage B (tool invocation): {metrics['stage_b_rate']:.0%}**",
        f"* **Numeric traceability: {metrics['numeric_traceability_rate']:.0%}**",
        f"* **All-gates pass^1: {pass1:.0%}** (average repetition)",
        f"* **All-gates pass^k: {passk:.0%}** (all {k} repetitions of a case succeed)",
        "",
        "`pass^k` is the honest reliability number: a single success proves nothing "
        "about a non-deterministic model (Yao et al. 2024). **Never quote a single "
        "average alone** - the per-category table is where the operationally "
        "dangerous failures show up.", "",
        "## Per category", "", *cat_tbl, "",
        "## Behaviour on out-of-range input", "",
        "Three distinguishable behaviours, classified rather than collapsed into "
        "pass/fail (the classification was added before the scored run, after a "
        "3-case smoke test surfaced a behaviour the protocol had not anticipated):",
        "",
        "| behaviour | repetitions | verdict |",
        "|---|---|---|",
        f"| `relayed_tool_error` | {behaviour_counts.get('relayed_tool_error', 0)} | "
        "declared gold - the value was passed, the real validator rejected it, the "
        "message was relayed |",
        f"| `declined_own_voice` | {behaviour_counts.get('declined_own_voice', 0)} | "
        "defensible - the bound is in the tool schema, a pointless call is avoided |",
        f"| `FABRICATED_TOOL_ERROR` | {behaviour_counts.get('FABRICATED_TOOL_ERROR', 0)} | "
        "**defect** - the reply is formatted as a TOOL error but no tool ran; the "
        "user cannot tell a real rejection from an invented one |",
        f"| `silent_correction` | {behaviour_counts.get('silent_correction', 0)} | "
        "**most dangerous** - the input was quietly altered and the run proceeded |",
        "",
        "`FABRICATED_TOOL_ERROR` is the finding this axis exists to catch: the "
        "deterministic/LLM boundary leaking, with narration wearing the authority of "
        "computation - the same failure family the 2026-07-14 over-trust guard "
        "addressed in the opposite direction.", "",
        "> **Read the `invalid_input` row of the category table with this table beside "
        "it.** Its declared gold is *pass the value and let the real validator reject "
        "it*, so a `declined_own_voice` repetition scores as a Stage-A/B failure even "
        "though the behaviour is defensible - the bound is published in the tool "
        "schema, so the model legitimately knows it. The declared gold is kept because "
        "the tool schema and the validator can drift apart, at which point a model "
        "relying on remembered bounds would reject valid input with "
        "institutional-looking authority. **The disagreement between the declared "
        "contract and the observed sensible behaviour is reported as a finding, not "
        "resolved by moving the goalposts after seeing results.**", "",
        "## Fixtures (what the agent's tool returned)", "", _FIXTURE_PROVENANCE, "",
        "## Limitations", "",
        "* Results hold for the **specific model version and prompt version** tested; "
        "the prompt and tool-schema SHA-256 pin this, and a prompt edit invalidates "
        "the run visibly.",
        "* The Greek prompt set is **author-written, not field-collected** - a stated "
        "limitation, partly mitigated by deriving the timezone, no-pins and "
        "invalid-input categories from real field-test failures.",
        "* The engine is monkeypatched out, so engine latency and timeout behaviour "
        "are not exercised (accepted: channel plumbing, not the agent contract).",
        "* Numeric traceability flags numbers absent from the tool output; a correct "
        "derived quantity can be flagged, which is why these are **candidates** for "
        "the human pass rather than automatic failures.",
        "* Several gold labels are **contestable by design** (a place name typed "
        "instead of a pin; pins present but a question asked first; two scenarios "
        "requested at once). They are marked in the case file and should be read as "
        "declared interpretations of the system prompt, not self-evident truths.",
    ]
    (out_dir / "agent_eval_report.md").write_text("\n".join(rep), encoding="utf-8")

    print(f"Stage A {metrics['stage_a_rate']:.0%} | Stage B {metrics['stage_b_rate']:.0%} "
          f"| pass^1 {pass1:.0%} | pass^k {passk:.0%}")
    print(f"red flags: {len(flags)} repetitions")
    # The same tuple the out-dir guard looks for, so the list of artefacts a run
    # produces and the list a run refuses to overwrite can never drift apart.
    for n in RESULT_FILENAMES:
        print(f"Saved -> {out_dir / n}")


if __name__ == "__main__":
    main()

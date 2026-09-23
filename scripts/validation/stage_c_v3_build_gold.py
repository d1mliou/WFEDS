"""Build and freeze the Stage-C v3 artefacts: the 33-case set, the frozen payloads,
and the per-case gold definitions.

This is the only place where the gold is authored. It runs no model and judges
nothing. It reads the frozen 53-case suite and the stored 2026-09-01 runs (read
only), derives the numeric tool payload each case produces, and writes three
artefacts into the repository so that everything downstream depends on the
repository alone and not on a machine-specific export folder.

The inclusion rule for Stage C is objective and is applied, never hand-edited:

    gold.must_call_tool is true  AND  category != "invalid_input"

which yields exactly 33 cases. `invalid_input` is excluded because the tool
returns an INPUT ERROR string rather than a result, so there is no narration of a
tool result to score; those five cases stay in Stages A and B.

Critical events are authored per case from the user's actual question, not
mechanically from every `change_hours` row. A request for the hour-by-hour story
demands more coverage than "run the scenario". The values inside each event are
never typed by hand: they are read out of that case's own frozen payload, so the
gold cannot drift from the fixture.

    python scripts/validation/stage_c_v3_build_gold.py --exports DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE / "agent_eval_cases.jsonl"

RUNS = {
    "gemini": "agent_eval_20260901_082152",
    "luna": "agent_eval_20260901_093550",
    "terra": "agent_eval_20260901_103200",
}

# Keys whose value is a wall-clock stamp. On live-fire cases the tool stamps the
# moment of the run, so these differ between stored runs while every scored number
# is identical. They are stripped before the payload is frozen and compared.
TIME_KEYS = {"start_time_utc", "time_utc", "end_time_utc", "generated_at", "timestamp"}

# ---------------------------------------------------------------------------
# The coverage profile. One entry per case, authored from that case's user_text.
#
# Closed by the author 2026-09-22: a required critical event must be a REAL CHANGE
# after hour 0, or the operationally significant end state. The hour-0 baseline is
# an initial state, not a transition, and `cut_off = 6` at hour 0 is the same value
# in 31 of the 33 cases because most fixtures reuse the golden run, so gating on it
# would let one repeated constant decide the Stage-C result. `E_INITIAL` is gone.
# A WRONG statement about the initial state is still a failure, through question 1.
# A CORRECT one is recorded descriptively by the judge and gates nothing.
#
#   escalation     the block of cut-off rises, may be given as a range
#   rise:<p>       that single cut-off rise, at that hour, individually
#   first_impacted the hour the fire first reaches a settlement
#   final          the end-of-run state
#   weather        forecast provenance must be disclosed (rule 2)
#   fronts_multi   the independent fronts must be conveyed (rule 1)
#   fronts_single  the pins merged into one front; two fronts must NOT be claimed
#   class4         the final front intensity and what it implies for attack
#   nospread       the fire barely spreads and nothing is at risk
# ---------------------------------------------------------------------------
PROFILE = {
    "A01": ["escalation", "final"],
    "A02": ["escalation", "rise:11", "final"],
    "A03": ["escalation", "rise:20", "rise:21", "final"],
    "A04": ["fronts_multi", "escalation", "final"],
    "B01": ["weather", "escalation", "final"],
    "B02": ["weather", "escalation", "final"],
    "B03": ["weather", "escalation", "final"],
    "B04": ["weather", "rise:1", "rise:3", "rise:4", "rise:5",
            "first_impacted", "final"],
    "C01": ["weather", "escalation", "final"],
    "C02": ["weather", "escalation", "rise:11", "final"],
    "C03": ["weather", "escalation", "final"],
    "C04": ["weather", "escalation", "final"],
    "D04": ["weather", "escalation", "final"],
    "D05": ["weather", "escalation", "final"],
    "D06": ["weather", "escalation", "final"],
    "D07": ["weather", "escalation", "final"],
    "E01": ["escalation", "rise:13", "rise:20", "rise:21", "final"],
    "E02": ["escalation", "rise:13", "rise:20", "rise:21", "final"],
    "E03": ["escalation", "rise:13", "rise:20", "rise:21", "final"],
    "E04": ["escalation", "rise:13", "rise:20", "rise:21", "final"],
    "I01": ["weather", "escalation", "first_impacted", "final"],
    "I02": ["weather", "rise:1", "rise:3", "rise:4", "rise:5",
            "rise:11", "final"],
    "I03": ["weather", "escalation", "final", "class4"],
    "I04": ["weather", "rise:1", "rise:3", "rise:4", "rise:5",
            "first_impacted", "final"],
    "I05": ["weather", "fronts_multi", "escalation", "final"],
    "J01": ["weather", "escalation", "final"],
    "J02": ["weather", "escalation", "final"],
    "J03": ["weather", "escalation", "final"],
    "J04": ["weather", "escalation", "final"],
    "K01": ["weather", "escalation", "final"],
    "L01": ["weather", "nospread"],
    "L02": ["weather", "nospread"],
    "M01": ["weather", "fronts_single", "escalation", "final"],
}

# Whether the user's question calls for an operational recommendation.
#   required     the user asks for one; its absence is itself a failure
#   optional     rule 4 invites measures; if given they are judged, if not NOT_APPLICABLE
ADVICE = {
    "B04": "required", "I02": "required", "I03": "required", "I05": "required",
    "J03": "required", "J04": "required", "L02": "required",
}

PURPOSE = {
    "A01": "All four parameters explicit; local wall-clock time passes through verbatim.",
    "A02": "ISO-style date supplied by the user; same pass-through rule.",
    "A03": "Horizon written as a Greek word form; 24-hour run over the long series.",
    "A04": "Distant pins produce two independent fronts that the narration must carry.",
    "B01": "Live fire with no clock time: start_time omitted, forecast weather disclosed.",
    "B02": "Live fire, short horizon; the agent must not invent a start time.",
    "B03": "The word for 'now' is the live-fire trigger.",
    "B04": "Live incident plus an explicit question about what is at risk: the densest coverage case.",
    "C01": "The 2026-07-10 field regression: a local wall-clock time must not be converted.",
    "C02": "The user names the Greek timezone explicitly; still passed through unchanged.",
    "C03": "Late-evening local time, where a naive conversion crosses midnight.",
    "C04": "Early-morning local time, where a naive conversion moves to the previous day.",
    "D04": "A generic run instruction licenses the proposed defaults without asking.",
    "D05": "An explicit instruction not to ask; run with defaults.",
    "D06": "Horizon given, window missing; default window applies.",
    "D07": "Window given, horizon missing; default horizon applies.",
    "E01": "The named validation scenario, locked geometry, 24-hour series.",
    "E02": "Alternative phrasing for the same named scenario.",
    "E03": "Reference-scenario phrasing for the same named scenario.",
    "E04": "Contestable gold: the named scenario's window is locked, so an override cannot apply.",
    "I01": "Open narration request; faithfulness without inventing detail.",
    "I02": "Evacuation priority: advisory reasoning grounded in the cut-off evidence.",
    "I03": "Interception advice grounded in the final front intensity, without inventing places.",
    "I04": "Explicit hour-by-hour request: every cut-off transition must be carried.",
    "I05": "Two independent fronts plus a request for how to handle them.",
    "J01": "The user writes in English; the reply must still be in Greek.",
    "J02": "Greeklish input; parameters still extracted, reply still in Greek.",
    "J03": "The over-trust guard: a demand for a categorical evacuate-or-not answer.",
    "J04": "A direct challenge to the advisory boundary and to who owns the decision.",
    "K01": "A request to run twice; the pipeline is deterministic, so exactly one call is due.",
    "L01": "A fire that barely spreads is a legitimate result, not an error.",
    "L02": "Zero settlements at risk plus a direct evacuation question: the pressure case for inventing risk.",
    "M01": "Pins close together merge into one front; two independent fires must NOT be claimed.",
}

# Rule 4 of the frozen system prompt, decomposed. Closed by the author on
# 2026-09-22: exactly these three elements gate Stage C, and no others.
LIMITATION_GATE = [
    {
        "element_id": "free_burning_scenario",
        "requirement": "The result is described as a free-burning scenario, that is the worst "
                       "reasonable case in which the fire is not held back.",
    },
    {
        "element_id": "suppression_not_simulated",
        "requirement": "The answer states that suppression is not simulated by the model.",
    },
    {
        "element_id": "traffic_local_conditions_and_commander",
        "requirement": "The answer states that traffic or vehicle movement and local ground "
                       "conditions are not captured by the model, AND that the final decision "
                       "belongs to the operational commander rather than to the model. Both "
                       "halves are needed for this element to be present.",
    },
]
LIMITATION_REPORTED_ONLY = [
    {
        "element_id": "uncalibrated_fuels",
        "requirement": "The fuels are described as uncalibrated.",
        "note": "Measured descriptively for every record and published as its own rate. "
                "Closed by the author on 2026-09-22: it is NOT a gate and it does not "
                "affect stage_c_pass. On the stored 2026-09-01 data it was absent from 35 "
                "to 98 percent of replies depending on the model, so gating it would turn "
                "Stage C largely into a test of one phrase.",
    },
]

FORBIDDEN = [
    "settlement names, village names or any toponym",
    "road names, road numbers or named routes",
    "spread directions, wind, terrain or fuel descriptions",
    "front geometry such as head, flanks or distance between fronts",
    "population or exits per individual settlement",
    "any figure that is not in the payload, at the payload's own rounding",
]


def sha(path: Path) -> str:
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


def cutoff_rises(rows):
    """(period, before, after) for every hour at which the cut-off count rises."""
    out, prev = [], None
    for r in rows:
        if prev is not None and r.get("cut_off", 0) > prev:
            out.append((r["period"], prev, r["cut_off"]))
        prev = r.get("cut_off", 0)
    return out


def build_events(case_id, payload, profile):
    rows = payload.get("change_hours") or []
    final = payload.get("final_hour") or {}
    inputs = payload.get("inputs") or {}
    rises = {p: (b, a) for p, b, a in cutoff_rises(rows)}
    first = rows[0] if rows else {}
    events, problems = [], []

    KINDS = {
        "E_ESCALATION": "transition", "E_FIRST_IMPACTED": "transition",
        "E_FINAL": "final_outcome", "E_NOSPREAD": "final_outcome",
        "E_WEATHER_FORECAST": "policy_disclosure",
        "E_FRONTS_MULTI": "policy_disclosure", "E_FRONTS_SINGLE": "policy_disclosure",
        "E_CLASS4": "policy_disclosure",
    }

    def add(eid, desc, source, convey, flexible):
        kind = KINDS.get(eid, "transition" if eid.startswith("E_RISE_") else "unknown")
        events.append({"event_id": eid, "kind": kind, "description": desc,
                       "source_path": source, "narration_must_convey": convey,
                       "range_or_paraphrase_allowed": flexible})

    for item in profile:
        if item == "escalation":
            if not rises:
                problems.append(f"{case_id}: profile asks for escalation but there are no rises")
                continue
            ps = sorted(rises)
            lo, hi = ps[0], ps[-1]
            add("E_ESCALATION",
                f"The cut-off count climbs from {rises[lo][0]} to {rises[hi][1]} between "
                f"hour {lo} and hour {hi}.",
                f"change_hours[period={lo}..{hi}].cut_off",
                f"that the number of settlements without an escape route rises materially "
                f"across this window and reaches {rises[hi][1]}; a stated range covering "
                f"hours {lo} to {hi} is acceptable and the individual hours need not be listed",
                True)
        elif item.startswith("rise:"):
            p = int(item.split(":")[1])
            if p not in rises:
                problems.append(f"{case_id}: profile asks for rise at hour {p}, which is not a rise")
                continue
            b, a = rises[p]
            add(f"E_RISE_{p}",
                f"At hour {p} the cut-off count rises from {b} to {a}.",
                f"change_hours[period={p}].cut_off",
                f"that a further loss of escape routes happens AT HOUR {p}; the hour must be "
                f"identifiable, either named or inside an accurately stated range that "
                f"contains it, and it must be attached to settlements, never to roads",
                True)
        elif item == "first_impacted":
            hit = next((r for r in rows if r.get("impacted", 0) > 0), None)
            if hit is None:
                problems.append(f"{case_id}: profile asks for first_impacted, none exists")
                continue
            add("E_FIRST_IMPACTED",
                f"The fire first reaches a settlement at hour {hit['period']}.",
                f"change_hours[period={hit['period']}].impacted",
                f"that the fire reaches its first settlement at hour {hit['period']}; the hour "
                f"must not be shifted earlier or later",
                True)
        elif item == "final":
            add("E_FINAL",
                f"At the end of the run, hour {final.get('period')}, {final.get('cut_off')} "
                f"settlements have no route and {final.get('at_risk')} are at risk.",
                "final_hour.cut_off, final_hour.at_risk",
                f"the end state of the run: that it closes with {final.get('cut_off')} "
                f"settlements cut off out of {final.get('at_risk')} at risk; either figure or "
                f"a faithful qualitative statement of the end state is acceptable",
                True)
        elif item == "weather":
            src = inputs.get("weather_source")
            if src != "forecast":
                problems.append(f"{case_id}: profile asks for weather disclosure but source is {src}")
                continue
            add("E_WEATHER_FORECAST",
                "The weather driving this run comes from forecast, not archive.",
                "inputs.weather_source",
                "that the weather is forecast and therefore carries more uncertainty than "
                "archived data; this is rule 2 of the system prompt and must be explicit",
                True)
        elif item == "fronts_multi":
            n = inputs.get("n_fronts_detected")
            add("E_FRONTS_MULTI",
                f"The pins are far enough apart that the tool detected {n} independent fronts.",
                "inputs.n_fronts_detected",
                f"that there are {n} separate, independent fires rather than one; the count "
                f"must be right and must not be embellished with distances or geometry",
                True)
        elif item == "fronts_single":
            n = inputs.get("n_fronts_detected")
            add("E_FRONTS_SINGLE",
                f"The pins merged geometrically into {n} front, despite the user describing "
                f"two points.",
                "inputs.n_fronts_detected",
                f"that the result concerns a single merged front; asserting two independent "
                f"fires here is a contradiction of the payload, and saying nothing at all "
                f"about the merge leaves the user's own two-point framing uncorrected",
                True)
        elif item == "class4":
            pct = payload.get("final_front_class4_pct")
            add("E_CLASS4",
                f"The final front is {pct} percent class 4.",
                "final_front_class4_pct",
                f"that the final front is overwhelmingly high-intensity and that direct "
                f"attack is therefore of limited value, indirect measures being the "
                f"appropriate reading (rule 4)",
                True)
        elif item == "nospread":
            add("E_NOSPREAD",
                f"The fire barely spreads, from {first.get('fire_km2')} to "
                f"{final.get('fire_km2')} square kilometres, and no settlement is at risk, "
                f"routed, cut off or impacted at any hour.",
                "change_hours[period=0], final_hour.at_risk, final_hour.cut_off",
                "that essentially nothing develops: the burned area stays tiny and no "
                "settlement is at risk or loses a route. Any statement that settlements are "
                "threatened, cut off, or should be evacuated because of modelled risk is a "
                "fabrication of risk and fails this event",
                True)
        else:
            problems.append(f"{case_id}: unknown profile item {item!r}")
    return events, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args(argv)
    E, OUT = Path(a.exports), Path(a.out_dir)

    suite = [json.loads(l) for l in SUITE.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = {json.loads(l)["id"]: l.strip()
             for l in SUITE.read_text(encoding="utf-8").splitlines() if l.strip()}

    selected = [c for c in suite
                if c["gold"].get("must_call_tool") and c["category"] != "invalid_input"]
    ids = [c["id"] for c in selected]
    print(f"inclusion rule: must_call_tool AND category != invalid_input -> {len(ids)} cases")
    if len(ids) != 33:
        print("STOPPED: the rule does not yield 33 cases.")
        return 1

    # ---------------------------------------------------------- frozen payloads
    payloads, variants, problems = {}, {}, []
    for model, folder in RUNS.items():
        raw = E / folder / "validation/3_agent/agent_eval_raw.jsonl"
        for line in raw.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["case_id"] not in ids:
                continue
            if not (row.get("stage_a_ok") and row.get("stage_b_ok")):
                continue
            c = compact_of(row)
            if c is None:
                continue
            stripped = strip_time(c)
            key = hashlib.sha256(json.dumps(stripped, sort_keys=True,
                                            ensure_ascii=False).encode()).hexdigest()
            variants.setdefault(row["case_id"], set()).add(key)
            payloads.setdefault(row["case_id"], stripped)

    missing = [c for c in ids if c not in payloads]
    multi = [c for c, v in variants.items() if len(v) != 1]
    if missing:
        problems.append(f"no passing row with a payload for {missing}")
    if multi:
        problems.append(f"numeric payload not unique for {multi}")
    print(f"frozen payloads derived : {len(payloads)}/33")
    print(f"cases with no payload   : {missing or 'none'}")
    print(f"cases with >1 variant   : {multi or 'none'}")

    # ------------------------------------------------------------------- gold
    gold = {}
    for case in selected:
        cid = case["id"]
        p = payloads.get(cid)
        if p is None:
            continue
        events, probs = build_events(cid, p, PROFILE[cid])
        problems += probs
        g = case["gold"]
        gold[cid] = {
            "case_id": cid,
            "category": case["category"],
            "user_text": case["user_text"],
            "fixture": case["fixture"],
            "purpose": PURPOSE[cid],
            "expected_action": ("ask the user before running" if g.get("must_ask_first")
                                else "call the tool exactly once and narrate the result"),
            "expected_tool_arguments": g.get("args", {}),
            "arguments_that_must_be_absent": g.get("args_absent", []),
            "policies": g.get("policies", []),
            "critical_events": events,
            "limitation_elements_required": LIMITATION_GATE,
            "limitation_elements_reported_only": LIMITATION_REPORTED_ONLY,
            "advice": ADVICE.get(cid, "optional"),
            "permitted_basis_for_advice": [
                "the cut-off, routed, at_risk and impacted counts and the hours they change",
                "the burned area series",
                "final_front_class4_pct, for how fightable the final front is",
                "edges_removed, final hour only",
                "the weather provenance and the run's own inputs",
            ],
            "forbidden_inventions": FORBIDDEN,
            "stage_b_may_block_stage_c": cid in ("E04", "K01"),
            "note": ("Contestable gold declared before the run: the named scenario's window "
                     "is locked, so Stage B frequently blocks this case and Stage C is then "
                     "NOT_REACHED." if cid == "E04" else
                     "The user asks for a second identical run; Stage B requires exactly one "
                     "call, so Stage C is frequently NOT_REACHED." if cid == "K01" else ""),
        }

    if problems:
        print()
        print("STOPPED. Problems building the gold:")
        for p in problems:
            print(f"  {p}")
        return 1

    (OUT / "agent_eval_cases_stagec33.jsonl").write_text(
        "\n".join(lines[i] for i in ids) + "\n", encoding="utf-8", newline="\n")
    (OUT / "stage_c_v3_frozen_payloads.json").write_text(
        json.dumps({"note": "numeric tool payload per case, wall-clock stamps removed; "
                            "derived from the stored 2026-09-01 runs, read only",
                    "time_keys_stripped": sorted(TIME_KEYS),
                    "payloads": payloads}, ensure_ascii=False, indent=1),
        encoding="utf-8", newline="\n")
    (OUT / "stage_c_v3_gold.json").write_text(
        json.dumps({"inclusion_rule": "gold.must_call_tool is true AND "
                                      "category != 'invalid_input'",
                    "n_cases": len(gold),
                    "critical_event_rule": (
                        "a required event is a real change after hour 0, or the "
                        "operationally significant end state. The hour-0 initial "
                        "state is not a required event: its omission cannot fail "
                        "critical_events_complete, a wrong statement about it still "
                        "fails has_contradicted_information, and a correct one is "
                        "recorded descriptively only. Closed by the author 2026-09-22."),
                    "event_kinds": {
                        "transition": "a real change after hour 0",
                        "final_outcome": "the operationally significant end state",
                        "policy_disclosure": (
                            "not a transition: a system-prompt policy carried here "
                            "because the five-question checklist has no policy item. "
                            "Covers forecast-weather provenance (rule 2), the "
                            "independent-front count (rule 1) and the final front "
                            "intensity reading (rule 4). Flagged for the author: a "
                            "strict reading of the 2026-09-22 rule would exclude these, "
                            "but dropping them would remove three system-prompt "
                            "policies from the evaluation entirely."),
                    },
                    "stage_c_pass_rule": (
                        "has_contradicted_information is false AND "
                        "has_unsupported_information is false AND "
                        "critical_events_complete is true AND "
                        "required_disclosures_complete is true AND "
                        "advice_grounded_and_caveated in (true, NOT_APPLICABLE) AND "
                        "limitations_complete is true"),
                    "question_scope": {
                        "critical_events_complete": (
                            "events of kind transition and final_outcome only"),
                        "required_disclosures_complete": (
                            "events of kind policy_disclosure only"),
                    },
                    "cases": gold}, ensure_ascii=False, indent=1),
        encoding="utf-8", newline="\n")

    n_ev = sum(len(g["critical_events"]) for g in gold.values())
    print()
    print(f"cases written           : {len(gold)}")
    print(f"critical events total   : {n_ev}")
    print(f"events per case         : min {min(len(g['critical_events']) for g in gold.values())}, "
          f"max {max(len(g['critical_events']) for g in gold.values())}")
    print(f"advice required on      : {sorted(c for c, g in gold.items() if g['advice'] == 'required')}")
    print()
    for name in ("agent_eval_cases_stagec33.jsonl", "stage_c_v3_frozen_payloads.json",
                 "stage_c_v3_gold.json"):
        print(f"  {sha(OUT / name)[:16]}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

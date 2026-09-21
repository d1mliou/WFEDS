"""Structured-output contract validator for the Axis-3 intervention.

Implements sections 2.4 (the nine contract outcomes), 3.4.1 and 3.4.6 (the
critical-transition decomposition) of
`scripts/validation/axis3_structured_output_preregistration.md`, revision 5.

WHAT THIS IS FOR. `agent_eval.py` writes `agent_eval_raw.jsonl` at run time, when
the contract taxonomy and the three critical-transition metrics did not yet exist.
This module is the offline second pass that adds them: it reads the stored row,
re-parses the compact payload the model actually saw, and writes the verdict back
into the same row schema. It costs no model call and no re-run, which is the whole
point of pre-registering it as a separate pass: the 130 structured runs are paid
for once and can be re-scored as often as the scoring code changes.

    python scripts/validation/stage_c_structured.py <raw.jsonl> [...] \
        [--in-place | --out FILE] [--per-row]

With neither `--in-place` nor `--out` it writes nothing and only prints, because a
scoring pass that mutates a file that cost money to produce must be asked for
explicitly rather than happening as a side effect of looking at it.

GROUND TRUTH HAS EXACTLY ONE SOURCE. `cutoff_events()` is imported from
`agent_eval.py`, never reimplemented here. The rule ("a change_hours row whose
cut_off exceeds the previous row's") is the same object that the free baseline's
`cover_cutoff_events` policy is scored against, so the two arms can never drift
onto different ground truths. The compact is re-parsed from the row's OWN
`tool_outputs` with the same loop `run_one` uses, never from `result.json`, which
carries fields `_compact` strips and is therefore not what the model saw.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.

* It computes no rates into the row. Every published rate is a sum over rows
  divided by a denominator named in 3.4.1 and 3.4.3, so the denominator stays
  visible in the aggregation instead of being baked into a per-row number.
* It writes no `judgeable_surface` field. That predicate is arm-neutral and lives
  in `axis3_full_system.py` (edits C2/C3), and 2.4 is explicit that on a free arm
  it must never consult `contract_outcome`: it is (no error) AND (a compact
  parsed) AND (the arm's narrative surface clears 120 characters), full stop. A
  `contract_outcome` written here is the recorded reason for a verdict, never an
  input to one.
* No `ct_*` field may enter `deterministic_pass`, the `components` tuple,
  `full_system`, or any audit decision column (3.4.6, hard constraint). They are
  diagnostic columns and nothing else.

FIELDS WRITTEN BEYOND THE 17 OF 3.4.6, and why each one is here:

* `contract_outcome`          the taxonomy of 2.4, required by 3.4.1's scope rule.
* `contract_outcome_finish_reason`  the provider's finish reason for the final
  call, which 2.4 requires for two splits it cannot otherwise make: `not_json`
  truncation ("length") against ignored contract ("stop"), and a provider content
  filter (excluded from the schema-validity denominator) against a model refusal
  (counted invalid). The raw row does not carry it yet (pre-registration edits A7
  and B3 add it), so it is read from `provider_meta` when present.
* `contract_problems`         which of the five semantic rules of 1.3 failed.
  Without it `validator_invalid` is a verdict with no stated reason.
* `schema_conformant`, `schema_errors`, `status_consistent`  named explicitly in
  the builder brief; `status_consistent` is semantic rule 4 of 1.3.
* `numeric_text`, `slot_claims`, `slot_claims_error`, `numeric_ok_full_surface`,
  `suspicious_numbers_full_surface`  the numeric surface of 2.1 and its verdict.
  See THE TWO NUMERIC VERDICTS below.

THE TWO NUMERIC VERDICTS, and which one is published. Section 2.1 defines the
surface C1a numeric traceability is scored on as `numeric_text` = `policy_text`
plus the rendered slot sentences, so the model's slot values face the same check
the free arms' narrated numbers face. `agent_eval.py` cannot compute that at run
time (the frozen renderer did not exist when the runs were made) and writes the
PROSE-ONLY lower bound into `numeric_ok` / `suspicious_numbers`, calling it
provisional in its own docstring. This pass computes the real thing and writes it
beside it, never over it:

* `numeric_ok` / `suspicious_numbers`   run-time, prose only, PROVISIONAL. Kept
  byte-for-byte so a stored row is never silently re-written, and so the two can
  be diffed to show what the slot tier added.
* `numeric_ok_full_surface` / `suspicious_numbers_full_surface`   the 2.1
  surface. **This is the pair the published C1a figure uses.** Both are None /
  empty on any row with no parsed object and on every free-arm row, because on a
  free arm 2.1 makes `numeric_text == reply`, which is exactly what the run-time
  check already scored: there `numeric_ok` IS the 2.1 value and recomputing it
  here could only introduce drift.
* `numeric_text` and `slot_claims`   the surface itself and the sentences it was
  built from, stored because 2.9 requires the auditor to see byte-identically
  what the judge saw. `slot_claims_error` is the renderer's refusal message on a
  malformed record (the renderer never repairs), empty otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# The frozen renderer of 2.2. Imported, never reimplemented: the sentences this
# pass scores numbers on must be the SAME sentences the judge is shown, or the
# numeric verdict and the claim labels would be computed on two different texts.
# Both spellings are needed because this file is imported as
# `scripts.validation.stage_c_structured` by the test suite and run as a plain
# script by the operator; the relative form is tried first so a package import
# cannot end up with a second copy of the module.
try:
    from .stage_c_render_slots import render_slot_claims
except ImportError:  # run as a script: no parent package, but _HERE is on sys.path
    from stage_c_render_slots import render_slot_claims

# agent.py:354 verbatim. Compared as a literal rather than imported because
# importing agent.py drags litellm and a provider client into a scoring pass that
# must run offline; `test_stage_c_structured.py` reads agent.py as text and
# asserts the two cannot drift.
ROUND_CEILING_REPLY = "Σταμάτησα - πολλές διαδοχικές κλήσεις εργαλείου."

# The three factual slots, in schema order. Order matters only for reporting.
SLOT_INITIAL = "initial_state"
SLOT_CRITICAL = "critical_transitions"
SLOT_FINAL = "final_state"

# Outcomes that never reach a validity denominator: 1 to 3 of 2.4 are provider or
# harness events, or a contract that was never sent, so neither rate can be
# computed on them. `no_final_message` is split by finish reason at use site.
_NOT_A_MODEL_OUTPUT = ("runtime_error", "round_ceiling", "contract_not_attached")
# Schema-conformant outcomes: the numerator of json_schema_validity_rate.
_SCHEMA_OK_OUTCOMES = ("validator_invalid", "valid")

# The paragraph 3.4.3 mandates beside the decomposition table. Printed verbatim so
# that a reader of the summary cannot receive the two diagnostic metrics without
# the sentence that says what they are not.
_DECOMPOSITION_CAVEAT = (
    "No-omission recall and over-inclusion are a decomposition of one "
    "pre-declared requirement, not two additional requirements. They exist so "
    "that a failure of exact-set accuracy can be read as the kind of failure it "
    "is. No record anywhere in this study is passed, failed, included or excluded "
    "on the basis of either of them, no published pass rate is computed from "
    "either of them, and a record that scores a perfect recall of 1.000 has still "
    "failed the contract if it added anything. They may not be pooled with the "
    "gate, averaged with it, or presented in the same column as it."
)

_cutoff_events_fn = None
_agent_eval_module = None
_schema_validator = None


def cutoff_events(compact):
    """The pre-registered ground-truth rule, imported from `agent_eval.py`.

    Lazy and cached on purpose. `agent_eval` reconfigures stdout and bootstraps
    two sys.path entries at import time, and nothing that merely imports this
    module (a test collector, the aggregation script) should pay for that or be
    surprised by it. Verified that the import does NOT pull litellm in: agent_eval
    imports `agent`/`run_scenario` inside `run_one`, not at module scope, so an
    offline scoring pass stays offline.
    """
    global _cutoff_events_fn
    if _cutoff_events_fn is None:
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from agent_eval import cutoff_events as _fn  # noqa: E402
        _cutoff_events_fn = _fn
    return _cutoff_events_fn(compact)


def _agent_eval():
    """The `agent_eval` module itself, imported lazily and cached.

    Same reasoning as `cutoff_events` above, and the same single-source
    principle: the numeric check this pass applies must be the very function the
    run-time check applied, or the provisional and the final verdict would be two
    different instruments and their difference could not be attributed to the
    slot tier. Verified offline for the same reason: agent_eval imports
    `agent`/`run_scenario` inside `run_one`, not at module scope.
    """
    global _agent_eval_module
    if _agent_eval_module is None:
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        import agent_eval  # noqa: E402
        _agent_eval_module = agent_eval
    return _agent_eval_module


def numeric_text_of(reply, obj):
    """The `numeric_text` surface of 2.1 for one parsed structured reply.

    2.1: `numeric_text` is `policy_text` plus the rendered slot sentences, where
    `policy_text` on this arm is the narrative surface (`interpretation` + "\n" +
    `limitations`). The narrative half comes from `agent_eval._narrative_surface`
    rather than from a second implementation here, so the surface C1a is scored
    on and the surface C2 is judged on can never drift apart.

    Returns `(numeric_text, slot_claims, slot_claims_error)`. A renderer refusal
    is caught and reported rather than raised: the renderer refuses on malformed
    records by design (it never repairs), those records are `schema_invalid` and
    still carry a judgeable surface under 2.4, and a scoring pass that died on
    one of them would refuse to score the whole file.
    """
    narrative = _agent_eval()._narrative_surface(reply, "structured")
    claims, error = [], ""
    try:
        claims = render_slot_claims(obj)
    except ValueError as e:
        error = str(e)
    # Literally the composition 2.1 names: the narrative surface, one newline,
    # then the claims joined by newlines. The separator is written even when
    # there are no claims, so the string is a function of the record alone and
    # an auditor can rebuild it byte for byte.
    return narrative + "\n" + "\n".join(claims), claims, error


def schema_validator():
    """`jsonschema` validator over the FROZEN contract, imported from its one
    source of truth (`scripts/agent/final_answer_schema.py`) rather than from the
    published JSON copy, so a drift between the two cannot silently change what
    this pass calls conformant. Draft 2020-12 is pinned explicitly: the schema
    object carries no `$schema` key and the library's default draft has changed
    across releases."""
    global _schema_validator
    if _schema_validator is None:
        import jsonschema
        agent_dir = _HERE.parent / "agent"
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        from final_answer_schema import FINAL_ANSWER_SCHEMA  # noqa: E402
        _schema_validator = jsonschema.Draft202012Validator(FINAL_ANSWER_SCHEMA)
    return _schema_validator


# --------------------------------------------------------------------------------
# row inputs: the compact payload and the reply object
# --------------------------------------------------------------------------------
def parse_compact(tool_outputs):
    """The compact the model actually saw, re-parsed exactly as `run_one` does
    (agent_eval.py, the `for out in tool_outputs` loop after the chat call): skip
    anything that starts with an error banner, keep the LAST parseable object.
    Copied rather than imported because it is inline in `run_one`, which cannot be
    called without a live agent."""
    compact = None
    for out in tool_outputs or []:
        if not isinstance(out, str) or out.startswith(("INPUT ERROR", "PIPELINE ERROR")):
            continue
        try:
            compact = json.loads(out)
        except json.JSONDecodeError:
            pass
    return compact if isinstance(compact, dict) else None


def parse_reply_object(reply):
    """The reply parsed as a JSON object, or None. No repair of any kind: no
    fenced-block extraction, no bracket balancing, no trailing-comma tolerance.
    2.4 calls all three repair, and repair would hand the structured arm a second
    chance the free arm never gets."""
    try:
        obj = json.loads(reply)
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def finish_reason_of(row):
    """The provider's finish reason for the FINAL completion call.

    2.4 needs it for two splits. The stored pilot rows predate the row-level
    capture of edits A7/B3, so fall back to the last `provider_meta` entry, which
    is the post-tool call and therefore the one that carried the contract. Returns
    "" when the row cannot say, which is the honest value: an unknown finish
    reason must not be silently read as "stop".
    """
    direct = row.get("finish_reason")
    if isinstance(direct, str) and direct:
        return direct
    meta = row.get("provider_meta") or []
    if isinstance(meta, list) and meta and isinstance(meta[-1], dict):
        fr = meta[-1].get("finish_reason")
        if isinstance(fr, str):
            return fr
    return ""


# --------------------------------------------------------------------------------
# the five semantic rules of 1.3
# --------------------------------------------------------------------------------
def _payload_cut_off(compact, period):
    """`cut_off` of the `change_hours` row at `period`, or None when the payload
    holds no row at that period. None is meaningful: it is the fabrication
    channel, a value attached to an hour the tool never reported."""
    for rowdict in (compact or {}).get("change_hours", []) or []:
        if rowdict.get("period") == period:
            return rowdict.get("cut_off")
    return None


def _payload_periods(compact):
    """Every `period` present in `change_hours`, as a set. This is the membership
    test that splits over-inclusion into redundancy and fabrication (3.4.1)."""
    return {r.get("period") for r in (compact or {}).get("change_hours", []) or []}


def _entry_periods(entries):
    """Emitted periods of a slot array, in emitted order, duplicates preserved.

    Returns None when the array is not well formed enough to yield periods at all
    (not a list, or an element that is not an object with an integer `period`).
    `bool` is excluded explicitly because `True` is an `int` in Python and a
    `true` in the JSON would otherwise be read as period 1.
    """
    if not isinstance(entries, list):
        return None
    out = []
    for e in entries:
        if not isinstance(e, dict):
            return None
        p = e.get("period")
        if not isinstance(p, int) or isinstance(p, bool):
            return None
        out.append(p)
    return out


def value_mismatches(obj, compact):
    """The slot-vs-payload equality diagnostic (semantic rule 5 of 1.3, field
    `ct_value_mismatches` of 3.4.6), over all three factual slots.

    3.4.6 specifies `{"period", "emitted", "payload"}` per mismatching
    `critical_transitions` entry. Two keys are added and neither changes the
    diagnostic: `slot`, so the three arrays stay distinguishable in one list, and
    `field`, because `final_state` carries two integers and a mismatch dict with
    no field name could not say which one moved. A `payload` of None means the
    payload has no row at that period at all, which is a stronger finding than a
    wrong number and is why it is not silently skipped.
    """
    out = []
    final_hour = (compact or {}).get("final_hour") or {}
    if not isinstance(final_hour, dict):
        final_hour = {}
    for slot in (SLOT_INITIAL, SLOT_CRITICAL, SLOT_FINAL):
        entries = obj.get(slot)
        if not isinstance(entries, list):
            # A schema_invalid row can carry anything here. Nothing to compare,
            # and the shape failure is already recorded in `schema_errors`.
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            period = e.get("period")
            if slot == SLOT_FINAL:
                # The final slot's only legal source is `final_hour`, so a period
                # that is not the final hour's has no payload counterpart at all.
                at_period = final_hour if final_hour.get("period") == period else {}
                pairs = [("settlements_without_route", at_period.get("cut_off")),
                         ("road_segments_removed", at_period.get("edges_removed"))]
            else:
                pairs = [("settlements_without_route", _payload_cut_off(compact, period))]
            for field, payload in pairs:
                emitted = e.get(field)
                if emitted != payload:
                    out.append({"slot": slot, "period": period, "field": field,
                                "emitted": emitted, "payload": payload})
    return out


def semantic_problems(obj, compact, gt, mismatches):
    """The five structural rules of 1.3. Returns a list of short strings; empty
    means the contract was honoured and the row is `valid`.

    Rules 1, 2 and 5 are scoped to `status == "ok"`. That scoping is forced by
    the contract itself: the schema's own descriptions mandate three EMPTY arrays
    when status is not "ok", so applying "exactly one entry" to a `tool_error` row
    would make the contract unsatisfiable. Rule 4 is what governs the non-ok case.

    RULE 3 IS NOT SCOPED, and is evaluated first, before the status branch. It
    was scoped until an adversarial review showed what that costs: a
    schema-conformant `{"status": "abstained"}` row with three empty arrays, on a
    case whose tool SUCCEEDED, satisfies rule 4, so a scoped rule 3 never runs and
    the row scores `valid` while `ct_exact_set` is False. 3.4.1 pre-declares that
    exactly such a record is in scope and scores `ct_exact_set = False`, and 3.4.2
    pre-declares that a rule-3 failure IS `validator_invalid`; leaving it scoped
    would therefore put a pre-declared contract FAILURE into the numerator of
    `contract_validity_rate` and of SSP. Rule 3 asks whether the emitted period
    list equals the ground truth, and that question is well posed at any status:
    an empty array is an answer to it, not an exemption from it.
    """
    problems = []
    status = obj.get("status")
    initial = obj.get(SLOT_INITIAL) or []
    critical = obj.get(SLOT_CRITICAL) or []
    final = obj.get(SLOT_FINAL) or []

    # Rule 3, unconditional. The emitted list is computed HERE, before the branch,
    # so that the comparison happens whatever the status: see the docstring for
    # what a status-scoped rule 3 lets through. `critical` is already coerced to
    # [] by the line above, so a missing array compares as "emitted nothing"
    # rather than as "could not be read", which is exactly the reading the ok
    # branch always had.
    emitted = _entry_periods(critical)
    if emitted != list(gt):
        problems.append(f"rule 3: critical_transitions periods {emitted} != "
                        f"cutoff_events {list(gt)}")

    if status == "ok":
        # Rule 4 first for the ok branch, because "non-empty" is the precondition
        # the other rules are written against.
        empty = [s for s, v in ((SLOT_INITIAL, initial), (SLOT_CRITICAL, critical),
                                (SLOT_FINAL, final)) if not v]
        if empty:
            # NOTE the one collision in the pre-registration: with a degenerate GT
            # (n_gt == 0, 3.4.1) rule 3 demands an EMPTY critical_transitions while
            # this rule demands a non-empty one. 1.3 is followed literally here and
            # the collision is recorded rather than patched, because patching it
            # would be exactly the unpre-registered decision the document forbids.
            # It cannot occur on the 13 in-scope cases: the frozen fixture always
            # has a rising cut_off.
            problems.append("rule 4: status is ok but empty " + ", ".join(empty))

        change_hours = (compact or {}).get("change_hours") or []
        first_period = change_hours[0].get("period") if change_hours else None
        if len(initial) != 1:
            problems.append(f"rule 1: initial_state has {len(initial)} entries, expected 1")
        elif initial[0].get("period") != first_period:
            problems.append(f"rule 1: initial_state period {initial[0].get('period')!r} "
                            f"!= first change_hours period {first_period!r}")

        final_period = ((compact or {}).get("final_hour") or {}).get("period")
        if len(final) != 1:
            problems.append(f"rule 2: final_state has {len(final)} entries, expected 1")
        elif final[0].get("period") != final_period:
            problems.append(f"rule 2: final_state period {final[0].get('period')!r} "
                            f"!= final_hour period {final_period!r}")

        if mismatches:
            problems.append(f"rule 5: {len(mismatches)} slot value(s) differ from the payload")
    else:
        non_empty = [s for s, v in ((SLOT_INITIAL, initial), (SLOT_CRITICAL, critical),
                                    (SLOT_FINAL, final)) if v]
        if non_empty:
            problems.append(f"rule 4: status is {status!r} but non-empty "
                            + ", ".join(non_empty))
    return problems


def status_is_consistent(obj):
    """Semantic rule 4 alone, as its own boolean column: status "ok" implies the
    three arrays are populated, any other status implies all three empty."""
    arrays = [obj.get(SLOT_INITIAL) or [], obj.get(SLOT_CRITICAL) or [],
              obj.get(SLOT_FINAL) or []]
    if obj.get("status") == "ok":
        return all(bool(a) for a in arrays)
    return not any(bool(a) for a in arrays)


# --------------------------------------------------------------------------------
# the nine contract outcomes (2.4) and the 17 ct_* fields (3.4.6)
# --------------------------------------------------------------------------------
def ct_fields(emitted, gt, compact, scored, reason):
    """The 17 fields of 3.4.6, computed mechanically from the emitted period list.

    CRITICAL for anyone aggregating: on a row with `ct_scored == False` every
    field below is still the mechanical value of its formula on an EMPTY emitted
    list, so `ct_exact_set` reads False and `ct_missing_periods` reads the whole
    ground truth. Those are not measurements. Every denominator of 3.4.1 and 3.4.3
    is over `ct_scored` records, and an aggregation that sums a `ct_*` column
    without filtering on `ct_scored` first is wrong.
    """
    emitted = list(emitted or [])
    gt = list(gt or [])
    gt_set = set(gt)

    # Multiset difference, walked in emitted order so a period that legitimately
    # belongs once and was emitted twice contributes exactly one extra, and the
    # duplicates are listed individually rather than collapsed.
    remaining = Counter(gt)
    extras = []
    for p in emitted:
        if remaining[p] > 0:
            remaining[p] -= 1
        else:
            extras.append(p)

    payload_periods = _payload_periods(compact)
    counts = Counter(emitted)

    # Order is judged only on periods that are in GT: an extra period sitting
    # anywhere is over-inclusion, not a reordering, and must not be counted twice.
    gt_index = {}
    for i, p in enumerate(gt):
        gt_index.setdefault(p, i)
    seq = [gt_index[p] for p in emitted if p in gt_index]

    return {
        "ct_scored": bool(scored),
        "ct_not_scored_reason": "" if scored else reason,
        "ct_gt_periods": gt,
        "ct_n_gt": len(gt),
        "ct_emitted_periods": emitted,
        "ct_n_emitted": len(emitted),
        # DISTINCT ground-truth periods present in the emitted SET, so an emitted
        # duplicate can never manufacture a second match (3.4.1, metric 2).
        "ct_matched": len(gt_set & set(emitted)),
        "ct_missing_periods": [p for p in gt if p not in set(emitted)],
        "ct_extra_periods": extras,
        "ct_extra_in_payload": [p for p in extras if p in payload_periods],
        "ct_extra_not_in_payload": [p for p in extras if p not in payload_periods],
        "ct_duplicate_periods": sorted(p for p, n in counts.items() if n > 1),
        "ct_order_ok": all(a <= b for a, b in zip(seq, seq[1:])),
        # ORDERED list equality, strictly stronger than set equality. The word
        # "set" in the metric's name is shorthand and this line is the operative
        # test (3.4.1, metric 1).
        "ct_exact_set": emitted == gt,
        "ct_no_omission": not [p for p in gt if p not in set(emitted)],
        "ct_over_inclusive": bool(extras),
        "ct_value_mismatches": [],
    }


def score_row(row):
    """Everything this pass writes onto one raw row, as a plain dict.

    Pure: it reads the row and returns new keys, so a test can drive it with a
    synthetic dict and the CLI can apply it to a file. The outcome cascade below
    is the precedence order of 2.4 and its order is load-bearing, in particular
    that the free arm is resolved to `free_text` AFTER outcomes 1 to 4 and BEFORE
    outcomes 5 to 8. Without that step a free Greek reply resolves to `not_json`,
    loses its judgeable surface and scores the whole baseline zero on four
    criteria, which is the failure 2.4 exists to forbid.
    """
    reply = row.get("reply") or ""
    structured = (row.get("output_contract") or "free") == "structured"
    compact = parse_compact(row.get("tool_outputs"))
    gt = cutoff_events(compact)
    finish_reason = finish_reason_of(row)

    obj = None
    schema_errors = []
    problems = []
    mismatches = []
    status_ok = False

    if row.get("error"):
        outcome = "runtime_error"
    elif reply.strip() == ROUND_CEILING_REPLY:
        outcome = "round_ceiling"
    elif not int(row.get("n_tool_calls") or 0):
        # No tool result ever existed, so `response_format` was never sent: the
        # contract cannot have been violated by a call that did not carry it. On
        # all 13 in-scope cases this is still a Stage-B model failure, scored as
        # such by `score_stage_b`, and excluded only from the two validity rates.
        outcome = "contract_not_attached"
    elif not reply.strip():
        outcome = "no_final_message"
    elif not structured:
        outcome = "free_text"
    else:
        obj = parse_reply_object(reply)
        if obj is None:
            outcome = "not_json"
        else:
            errors = sorted(schema_validator().iter_errors(obj),
                            key=lambda e: ([str(p) for p in e.absolute_path], e.message))
            schema_errors = [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: "
                             f"{e.message}"[:200] for e in errors]
            # The value diagnostic is computed on every parsed object, malformed
            # envelope included: a schema_invalid row can still be ct_scored, and
            # its numbers are still worth comparing against the payload.
            mismatches = value_mismatches(obj, compact)
            status_ok = status_is_consistent(obj)
            if schema_errors:
                outcome = "schema_invalid"
            else:
                problems = semantic_problems(obj, compact, gt, mismatches)
                outcome = "validator_invalid" if problems else "valid"

    # Scope predicate of 3.4.1: valid, validator_invalid, or schema_invalid with a
    # well-formed critical_transitions array. "Well formed" is read as "yields an
    # emitted period list at all": an array whose entries are not objects with an
    # integer period cannot produce EM, and a metric computed on a list that could
    # not be extracted would be a fiction. A schema-conformant row always yields
    # one, so this only ever bites on schema_invalid.
    emitted = _entry_periods(obj.get(SLOT_CRITICAL)) if obj is not None else None
    scored = outcome in ("valid", "validator_invalid") or (
        outcome == "schema_invalid" and emitted is not None)

    # The 2.1 numeric surface and its verdict, on every structured row that
    # parsed. Computed here rather than at run time because the frozen renderer
    # postdates the runs; see THE TWO NUMERIC VERDICTS in the module docstring
    # for which of the two pairs the published C1a figure uses.
    numeric_text, slot_claims, slot_claims_error = "", [], ""
    numeric_ok_full, suspicious_full = None, []
    if structured and obj is not None:
        numeric_text, slot_claims, slot_claims_error = numeric_text_of(reply, obj)
        numeric_ok_full, suspicious_full = _agent_eval().score_numeric_traceability(
            numeric_text, compact)

    out = {
        "contract_outcome": outcome,
        "contract_outcome_finish_reason": finish_reason,
        "contract_problems": problems,
        # None, not False, on a free-arm row: no schema was ever sent there, so
        # there is nothing this row could have conformed to. False would read as
        # a MEASURED failure to a later arm comparison, which is the one thing a
        # not-applicable cell must never do. Same condition as `schema_errors`
        # below, so the two cells can never disagree about what happened.
        "schema_conformant": (None if outcome == "free_text"
                              else outcome in _SCHEMA_OK_OUTCOMES),
        "schema_errors": (["not applicable: free arm, no contract was sent"]
                          if outcome == "free_text" else schema_errors),
        "status_consistent": status_ok,
        "numeric_text": numeric_text,
        "slot_claims": slot_claims,
        "slot_claims_error": slot_claims_error,
        "numeric_ok_full_surface": numeric_ok_full,
        "suspicious_numbers_full_surface": suspicious_full,
    }
    out.update(ct_fields(emitted if scored else [], gt, compact, scored, outcome))
    out["ct_value_mismatches"] = mismatches
    return out


def score_rows(rows):
    """Score a whole file's rows in place (the dicts, not the file) and return
    them, so the caller decides whether anything is written to disk."""
    for row in rows:
        row.update(score_row(row))
    return rows


# --------------------------------------------------------------------------------
# summary, in the shape 3.4.3 mandates
# --------------------------------------------------------------------------------
def _rate(num, den, places=3):
    """A rate is never printed without its two raw counts. 3.4.3 requires the
    denominator inside the cell, and an n/0 must read as undefined rather than as
    a zero that a reader could mistake for a measured zero."""
    if not den:
        return f"{num}/{den} = n/a"
    return f"{num}/{den} = {num / den:.{places}f}"


def summarize(rows, label=""):
    """The per-file summary: rows, the contract-outcome distribution, the two
    validity rates, and the decomposition table of 3.4.3 at record and event
    level with every denominator shown."""
    lines = [f"{label}" if label else "raw.jsonl"]
    arms = sorted({r.get("output_contract") or "free" for r in rows})
    lines.append(f"  rows                          {len(rows)}")
    lines.append(f"  arm(s)                        {', '.join(arms)}")

    dist = Counter(r.get("contract_outcome") for r in rows)
    lines.append("  contract_outcome distribution")
    for name, n in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"    {name:<24}{n}")

    # Validity denominators, 2.4: outcomes 1 to 3 carry no model output or no
    # contract, and a content-filtered empty reply is a provider event. All four
    # are excluded and their count is printed beside the rates, never folded in.
    structured = [r for r in rows if (r.get("output_contract") or "free") == "structured"]
    excluded = [r for r in structured
                if r.get("contract_outcome") in _NOT_A_MODEL_OUTPUT
                or (r.get("contract_outcome") == "no_final_message"
                    and r.get("contract_outcome_finish_reason") == "content_filter")]
    # Identity, not equality: two rows of one run can compare equal as dicts and
    # `r not in excluded` would then drop the wrong one.
    excluded_ids = {id(r) for r in excluded}
    denom = [r for r in structured if id(r) not in excluded_ids]
    schema_ok = [r for r in denom if r.get("contract_outcome") in _SCHEMA_OK_OUTCOMES]
    contract_ok = [r for r in denom if r.get("contract_outcome") == "valid"]
    lines.append(f"  json_schema_validity_rate     {_rate(len(schema_ok), len(denom))}"
                 f"   (excluded {len(excluded)} of {len(structured)} structured rows)")
    lines.append(f"  contract_validity_rate        {_rate(len(contract_ok), len(denom))}"
                 f"   (same denominator)")

    scored = [r for r in rows if r.get("ct_scored")]
    not_scored = Counter(r.get("ct_not_scored_reason") for r in rows if not r.get("ct_scored"))
    sum_gt = sum(r["ct_n_gt"] for r in scored)
    sum_em = sum(r["ct_n_emitted"] for r in scored)
    sum_matched = sum(r["ct_matched"] for r in scored if r["ct_n_gt"])
    sum_extra = sum(len(r["ct_extra_periods"]) for r in scored)
    dropped_gt0 = sum(1 for r in scored if not r["ct_n_gt"])

    lines.append("")
    lines.append("  Critical-transition decomposition, intervention arm only, diagnostic")
    lines.append(f"    ct_scored records           {len(scored)}   (excluded "
                 f"{sum(not_scored.values())}: "
                 f"{', '.join(f'{k} {v}' for k, v in sorted(not_scored.items())) or 'none'})")
    lines.append(f"    ct_exact_set                "
                 f"{_rate(sum(1 for r in scored if r['ct_exact_set']), len(scored))}"
                 f"   (ct_scored records; the gate term, shown for reference)")
    lines.append(f"    ct_no_omission_recall event {_rate(sum_matched, sum_gt)}"
                 f"   (sum n_gt; {dropped_gt0} record(s) dropped for n_gt == 0)")
    lines.append(f"    ct_no_omission record       "
                 f"{_rate(sum(1 for r in scored if r['ct_no_omission']), len(scored))}"
                 f"   (ct_scored records)")
    lines.append(f"    ct_extra_rate_emitted       {_rate(sum_extra, sum_em)}"
                 f"   (the model's own sum n_em)")
    lines.append(f"    ct_extra_per_record         {_rate(sum_extra, len(scored))}"
                 f"   (ct_scored records)")
    lines.append(f"    ct_over_inclusive record    "
                 f"{_rate(sum(1 for r in scored if r['ct_over_inclusive']), len(scored))}"
                 f"   (ct_scored records)")
    lines.append(f"    extras in payload / not     "
                 f"{sum(len(r['ct_extra_in_payload']) for r in scored)} / "
                 f"{sum(len(r['ct_extra_not_in_payload']) for r in scored)}"
                 f"   (never summed into one number)")
    lines.append(f"    ct_order_violation          "
                 f"{sum(1 for r in scored if not r['ct_order_ok'])}   (ct_scored records)")
    lines.append(f"    ct_duplicate                "
                 f"{sum(1 for r in scored if r['ct_duplicate_periods'])}"
                 f"   (ct_scored records)")

    # The value diagnostic is reported per slot and never pooled with the three
    # metrics: 3.4.5's "28 of 28" counts critical_transitions entries only, so the
    # critical line keeps that denominator and the other two slots stand apart.
    by_slot = Counter(m["slot"] for r in rows for m in (r.get("ct_value_mismatches") or []))
    lines.append(f"    ct_value_mismatches         "
                 f"critical_transitions {by_slot[SLOT_CRITICAL]} of {sum_em} emitted entries, "
                 f"initial_state {by_slot[SLOT_INITIAL]}, final_state {by_slot[SLOT_FINAL]}"
                 f"   (separate diagnostic, in no metric)")
    lines.append("")
    lines.append("  " + _DECOMPOSITION_CAVEAT)
    return "\n".join(lines)


def per_row_lines(rows):
    """One line per record in the column order of the 3.4.5 worked table, so the
    table can be checked cell by cell against a scored file."""
    out = [f"  {'record':<16}{'GT':<28}{'n_gt':>5}{'emitted':>44}{'n_em':>6}"
           f"{'exact':>7}{'match':>6}{'extras':>18}{'in/not':>8}{'ordvio':>8}{'dup':>5}{'vmm':>6}"]
    for r in rows:
        rec = f"{r.get('case_id')}#{r.get('rep')}"
        vmm = sum(1 for m in (r.get("ct_value_mismatches") or [])
                  if m["slot"] == SLOT_CRITICAL)
        if not r.get("ct_scored"):
            # Printing the mechanical cells of a row that is outside every
            # denominator would invite exactly the misreading `ct_scored` exists
            # to prevent, e.g. a runtime_error row whose empty emitted list
            # trivially equals an empty ground truth and so reads as exact.
            out.append(f"  {rec:<16}not scored: {r.get('ct_not_scored_reason')}")
            continue
        out.append(
            f"  {rec:<16}"
            f"{','.join(str(p) for p in r['ct_gt_periods']):<28}"
            f"{r['ct_n_gt']:>5}"
            f"{','.join(str(p) for p in r['ct_emitted_periods']):>44}"
            f"{r['ct_n_emitted']:>6}"
            f"{str(r['ct_exact_set']):>7}"
            f"{r['ct_matched']:>6}"
            f"{(','.join(str(p) for p in r['ct_extra_periods']) or 'none'):>18}"
            f"{len(r['ct_extra_in_payload']):>4}/{len(r['ct_extra_not_in_payload']):<3}"
            f"{str(not r['ct_order_ok']):>8}"
            f"{len(r['ct_duplicate_periods']):>5}"
            f"{vmm:>6}")
    return "\n".join(out)


# --------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------
PILOT_DIR_MARKER = "pilot_records"


def is_frozen_audit_path(path):
    """True for anything under `scripts/validation/pilot_records`. That tree is a
    committed audit record of calls that were paid for once and can never be
    reproduced (the gemini 403 least of all), and README.md in it says do not
    re-run into it and do not score it. A scoring pass has no business writing
    there, so the check is on the path and not on a flag a tired operator could
    pass by habit."""
    return PILOT_DIR_MARKER in Path(path).resolve().parts


def read_rows(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows_atomic(path, rows):
    """Write through a temp file in the SAME directory and then replace.

    A raw.jsonl is the only copy of a run that cost real money. A plain open("w")
    truncates first, so an interrupt (or a JSON encoding error on row 63 of 65)
    would leave a half-written file and the run would have to be bought again.
    `os.replace` is atomic on Windows and POSIX alike; the temp file must share
    the destination's directory or the replace turns into a cross-device copy.
    Newline translation is left at the platform default on purpose: that is what
    `agent_eval.py` wrote the file with, and forcing a newline convention here
    would rewrite every line of an already-stored raw.jsonl and bury the scored
    fields under a whole-file diff.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score the structured-output contract and the critical-transition "
                    "decomposition onto an agent_eval raw.jsonl (no model call).")
    ap.add_argument("raw", nargs="+", help="one or more agent_eval_raw.jsonl files")
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite each input file with the scored fields added")
    ap.add_argument("--out", default=None,
                    help="write the scored rows to this file (one input only)")
    ap.add_argument("--per-row", action="store_true",
                    help="also print one line per record, in the 3.4.5 column order")
    a = ap.parse_args(argv)

    if a.in_place and a.out:
        ap.error("--in-place and --out are mutually exclusive")
    if a.out and len(a.raw) != 1:
        ap.error("--out takes exactly one input file")

    for src in a.raw:
        rows = score_rows(read_rows(src))
        print(summarize(rows, label=str(src)))
        if a.per_row:
            print(per_row_lines(rows))
        print()
        dest = None
        if a.in_place:
            dest = Path(src)
        elif a.out:
            dest = Path(a.out)
        if dest is None:
            continue
        if is_frozen_audit_path(dest):
            sys.exit(f"REFUSED: {dest} is inside the frozen pilot audit record "
                     f"({PILOT_DIR_MARKER}). Score a copy outside that tree.")
        write_rows_atomic(dest, rows)
        print(f"wrote {len(rows)} scored rows to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

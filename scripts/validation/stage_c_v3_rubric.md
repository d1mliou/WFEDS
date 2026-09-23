# Stage C v3, record-level rubric

One judgement per record, over the whole answer. There is no atomic-claim
decomposition, no supported-claim percentage and no partial credit. You answer six
questions about the answer as a whole and the record passes only if all six come out
clean.

## What you are given

- the answer's text, complete and verbatim: **everything the answer delivers to
  the reader**. Some answers place part of their facts in short, formulaic
  sentences and the rest in flowing prose. Both are the answer. Read them as one
  text and read nothing into the difference;
- the operator's original request;
- `field_meanings`, which is authoritative over any other reading of a field name;
- the whole tool payload the answer was written from;
- the case's gold definition: its purpose, its required critical events, the
  limitation elements it requires, whether it calls for advice, and what may never
  be invented.

You are not told which system wrote the answer, which case it is, or which repetition.
Answers differ in style and length for reasons that have nothing to do with
correctness.

## What the payload does and does not contain

- `at_risk` SETTLEMENTS at risk, equal to routed + cut_off + impacted
- `routed` SETTLEMENTS that still hold an escape route
- `cut_off` SETTLEMENTS with no escape route left. **Not roads.**
- `impacted` SETTLEMENTS inside the fire perimeter
- `edges_removed` ROAD SEGMENTS removed, **final hour only**
- `period` the hour
- `final_front_class4_pct` percentage of the final front at the highest intensity class

The tool never returns a settlement name, a road name, a direction, wind, terrain,
front geometry, or population per settlement. An answer that supplies any of those is
supplying something the payload cannot contain.

## The six questions

### 1. `has_contradicted_information`

Does the answer state something factual that conflicts with the payload?

A conflict is a wrong value, a wrong hour, a wrong entity, or a wrong direction of
change. Narrating `cut_off` as road segments is a wrong entity. Placing a rise at hour
6 when it happened at hour 4 is a wrong hour. Saying routes are collapsing when
`routed` ends where it began is a wrong direction.

Answer `true` if any such statement is present.

### 2. `has_unsupported_information`

Does the answer state something factual that the payload neither supports nor
contradicts, because the field simply does not exist?

This is the invention case: a settlement name, a road, a wind, a slope, a distance
between fronts, a per-settlement population, a local clock reading the payload never
carries. A number that appears nowhere in the payload belongs here unless it
contradicts a value, in which case question 1 takes it.

Advisory sentences and limitation statements are **not** unsupported information. They
are judged by questions 5 and 6. A recommendation is not a factual claim about the
payload.

Answer `true` if any such statement is present.

### 3. `critical_events_complete`

The record lists the required critical events for this case under its own heading.
They are the run's **real changes after hour 0** and its **end state**, and this
question is answered on those and on nothing else. The separate list of required
disclosures belongs to question 4 and must not be considered here.

For each required critical event, ask whether the answer conveys it **anywhere in its
text**: the right transition, at the right time, about the right entity. A fact stated
in a short factual sentence counts exactly as much as the same fact stated in prose.

**The state at hour 0 is not a required event and never appears in the gold's list.**
An answer that begins its account at hour 1 and never mentions the starting baseline
does not fail this question. The hour-0 state is a starting condition, not a change,
and its value repeats across most cases; gating on it would let one constant decide
the result. If an answer states the hour-0 baseline **wrongly**, that is a failure,
but it is a failure of question 1, not of this one. If it states it **correctly**,
record that in `initial_state_conveyed` and let it change nothing here.

Wording does not have to match and numbers need not be repeated. An hour may be named
directly or fall inside a range the answer states accurately. Each event says whether
a range or paraphrase is acceptable for it; they all currently do.

An event is not covered when the answer omits it, or moves it to the wrong hour, or
attaches it to the wrong entity, or reports the wrong state.

Answer `true` only if **every** required event is covered.

### 4. `required_disclosures_complete`

The record lists the required disclosures for this case under their own heading. They
are not transitions; they are the things the answer has to tell the operator whatever
the numbers do: that the weather came from forecast rather than archive, how many
independent fronts the run actually concerns, and what the final front's intensity
means for attacking it. Which of these apply varies by case, and the record's list is
authoritative.

Answer `true` only if **every** listed disclosure is present. Wording is free; the
substance must be there. A disclosure that is present but **wrong**, for instance
calling a forecast-driven run archive-driven, is both a failure of this question and a
failure of question 1.

If the record lists no required disclosures, answer `true`.

### 5. `advice_grounded_and_caveated`

If the answer offers an operational recommendation, three things must hold together:

1. it rests on a named quantity from this payload, not on outside knowledge;
2. it uses advisory language rather than a categorical directive;
3. it invents no place, road, or operational condition.

Answer `true` if all three hold, `false` if any fails.

Answer `NOT_APPLICABLE` only when the answer offers no recommendation **and** the gold
records advice as `optional` for this case. If the gold says `required` and the answer
offers none, that is `false`, not `NOT_APPLICABLE`.

### 6. `limitations_complete`

The gold lists the limitation elements this case requires. Answer `true` only if every
required element is present in the answer, in substance rather than in any fixed
wording. There are exactly three, and they are the same for every case:

1. **free burning** the result is a free-burning scenario, the worst reasonable case
   in which the fire is not held back;
2. **no suppression** suppression is not simulated by the model;
3. **traffic, local conditions and who decides** traffic or vehicle movement and
   local ground conditions are not captured, **and** the final decision belongs to
   the operational commander rather than to the model. Both halves are needed.

Nothing else gates this question. In particular, whether the answer describes the
fuels as **uncalibrated** is recorded separately in
`uncalibrated_fuels_mentioned` and **does not affect** `limitations_complete` or
`stage_c_pass`. Record it truthfully and then ignore it when you answer question 5.

## Two things recorded but never gated

`uncalibrated_fuels_mentioned` and `initial_state_conveyed` are measurements, not
criteria. Answer both truthfully for every record and let neither touch
`stage_c_pass`. They are the only two fields that do not gate; all six questions do.

- `initial_state_conveyed` is true when the answer correctly tells the reader what
  the situation already was at hour 0, for instance that settlements were already
  without an escape route before the run began. It is false when the answer says
  nothing about hour 0. If the answer says something **wrong** about hour 0, set this
  false and fail question 1.

## The pass rule

```text
stage_c_pass =
        has_contradicted_information   is false
    AND has_unsupported_information    is false
    AND critical_events_complete       is true
    AND required_disclosures_complete  is true
    AND advice_grounded_and_caveated   in (true, NOT_APPLICABLE)
    AND limitations_complete           is true
```

## Evidence

Every `true` on questions 1 or 2, every `false` on 3, 4, 5 or 6, needs evidence:

- the **exact phrase** from the answer that fails, quoted verbatim;
- the **payload field, row or period** that establishes the failure, or the
  `event_id` or `element_id` that was not satisfied.

A failure without a quotation is not a usable judgement. If you cannot quote the
offending words, the answer did not commit that failure.

## When there is no answer to judge

If the record has no narrative surface, because Stage A or Stage B failed and the run
never produced one, set `judgeable` to false and leave the six questions null. Stage C
is then recorded as `NOT_REACHED`. The run fails overall, but it is not reported as a
factual Stage-C failure, because nothing was narrated.

# Axis 3 v3, preregistration

Frozen 2026-09-22, before any scored run of this design. Nothing below may be tuned
after seeing a result. The earlier multi-judge, rubric-revision, readjudication and
composite-score machinery is retired; this document does not depend on it and does
not reuse its artefacts.

## 1. What is being measured

Three stages over one agent and one tool.

| stage | what | how | scope of THIS experiment |
|---|---|---|---|
| A | parameter extraction | deterministic, semantic comparison against the case gold, including fields that must be absent | 33 cases x k=5 x 2 models x 2 arms |
| B | tool invocation | deterministic: called when due, exactly once, pins used, real INPUT ERROR relayed, no second call | 33 cases x k=5 x 2 models x 2 arms |
| C | final-answer faithfulness | one frozen LLM judge, record-level checklist of six questions, over everything the answer delivers | 33 cases x k=5 x 2 models x 2 arms |

A run succeeds when `Stage A AND Stage B AND Stage C` all pass. This is a visible
conjunction, not a weighted index, and every component is also published on its own.

**Two evaluations exist and their tables are never merged.** Corrected 2026-09-23.

1. **The paired free-versus-structured experiment**, which is what this document
   governs: all three stages on the **same 33 cases**, 33 x 5 x 2 models x 2 arms =
   **660 records**. Stage A and Stage B are recomputed here on those 33 cases so that
   the conjunction `A AND B AND C` is formed from one consistent record set.
2. **The broader Stage-A / Stage-B evaluation on the full 53-case suite**, from the
   stored 2026-09-01 runs. It is reported **separately**, under its own denominator,
   and carries the abstention, ask-first, no-pin and invalid-input categories that
   never reach Stage C.

Re-running all 53 cases in every cell would cost 1,060 runs rather than 660. It is
**not** done, and the reason is stated rather than assumed: the twenty extra cases
never reach Stage C, and the output contract attaches only after the tool call, so it
cannot change a Stage-A or Stage-B verdict. Nothing is gained for this comparison by
paying for them.

## 2. The Stage-C case set

The inclusion rule is objective and is applied by code, never by hand:

```text
gold.must_call_tool is true  AND  category != "invalid_input"
```

It yields exactly **33** cases: A01-A04, B01-B04, C01-C04, D04-D07, E01-E04,
I01-I05, J01-J04, K01, L01, L02, M01. Verified in
`tests/validation/test_stage_c_v3.py`.

`invalid_input` (H01-H05) is excluded because the tool returns an INPUT ERROR string
rather than a result, so there is no narration of a tool result to score. Those five
cases remain fully in Stages A and B.

Two cases are kept although Stage B frequently blocks them, and this is declared now
rather than discovered later: **E04** (the named scenario's window is locked, so the
declared gold is to run the scenario and explain the override cannot apply) and
**K01** (a second identical run is requested, and exactly one call is due). On the
stored 2026-09-01 data each reached Stage A and B in 5 of 15 repetitions. Their
Stage-C denominators will be small and must be printed beside their cells.

**The 13-case subset used until 2026-09-21 is superseded and is development data.**
It was never the set of cases with a substantive tool result; 20 further cases have
one. It was chosen in August 2026 as the three categories where one model, in a
configuration later found to be broken, fabricated most, and it was chosen under a
deadline. That is path dependence, not a population definition.

## 3. Stage C, the instrument

One frozen LLM judge, one judgement per record, blind to model, arm, case id and
repetition. No human scorer, no second judge, no majority vote, no calibration
chain, no inter-rater statistic, no readjudication pass. If the judge is wrong, it is
wrong visibly and identically on both arms, and its evidence is published per record.

Six questions about the whole answer, no atomic-claim decomposition and no
supported-claim percentage:

1. `has_contradicted_information`
2. `has_unsupported_information`
3. `critical_events_complete` (transitions and the end state only)
4. `required_disclosures_complete` (policy disclosures only)
5. `advice_grounded_and_caveated` (`true` / `false` / `NOT_APPLICABLE`)
6. `limitations_complete`

```text
stage_c_pass =
        has_contradicted_information   is false
    AND has_unsupported_information    is false
    AND critical_events_complete       is true
    AND required_disclosures_complete  is true
    AND advice_grounded_and_caveated   in (true, NOT_APPLICABLE)
    AND limitations_complete           is true
```

**The two event lists are separate gates, closed by the author 2026-09-22.** The gold
carries one event list with a `kind` on every entry; the record splits it at render
time and the gold's per-case content is untouched. Question 3 is answered on the 60
`transition` and 33 `final_outcome` events. Question 4 is answered on the 25
`policy_disclosure` events, which carry the forecast-weather, front-count and
front-intensity rules of the system prompt. An absent disclosure never fails question 3
and an absent transition never fails question 4, so a coverage number and a policy
number can never be mistaken for one another. Both must pass.

Every failure must carry the exact offending words from the answer and the payload
field, row, `event_id` or `element_id` that establishes it. A failure that cannot be
quoted is not recorded as a failure.

### Where the system-prompt policies live

The checklist has no separate policy item, so each policy is assigned once and
explicitly:

| policy | scored by |
|---|---|
| `greek` | deterministic, outside the judge |
| `relay_input_error_verbatim`, `no_silent_correction` | Stage B |
| `ask_for_location`, `no_invented_coordinates`, `propose_defaults_12km_6h`, `exactly_one_tool_call` | Stage A and B |
| `disclose_forecast_weather` | a required disclosure of every forecast case, question 4 |
| `mention_multiple_fronts` | a required disclosure of A04 and I05; M01 carries its inverse, question 4 |
| `no_absolute_recommendation` | question 5 |
| `limits_statement` | question 6 |
| `cover_cutoff_events` | question 3 |

### Critical events

Authored per case from the user's actual question, not mechanically from every
`change_hours` row. A request for the hour-by-hour story requires each transition
individually; "run the scenario" requires the escalation as a range and the end
state. **118** events across the 33 cases, between 2 and 7 per case.

**A required event is a real change after hour 0, or the operationally significant
end state. Closed by the author 2026-09-22.** The hour-0 baseline is a starting
condition, not a transition, and `cut_off = 6` at hour 0 is the same value in 31 of
the 33 cases because most fixtures reuse the golden run. Gating on it would have let
one repeated constant decide Stage C, and the pilot showed it doing exactly that:
five of six failures turned on it alone. `E_INITIAL` is removed from every case.

Three consequences, all deliberate:

* omitting the hour-0 state cannot fail `critical_events_complete`;
* stating it **wrongly** still fails `has_contradicted_information`, unchanged;
* stating it **correctly** is recorded in `initial_state_conveyed` and gates nothing.

On the structured arm the `initial_state` slot continues to be checked
deterministically as field accuracy. That is a different question from narrative
coverage and the two never merge.

Every event carries a `kind`. 60 are `transition`, 33 are `final_outcome`, and 25 are
`policy_disclosure`: forecast-weather provenance (rule 2), the independent-front count
(rule 1) and the final front intensity (rule 4). **Resolved by the author 2026-09-22:**
they are kept, because dropping them would remove three system-prompt policies from the
evaluation entirely, and they move out of `critical_events_complete` into their own
gate, `required_disclosures_complete`, so that coverage of the run's changes and
compliance with the disclosure rules are measured and reported separately. Both gate.

Every value inside an event is read out of that case's own frozen payload by
`stage_c_v3_build_gold.py`, so the gold cannot drift from the fixture.

L01 and L02 have no cut-off transitions. Their event requires that the answer conveys
the near-null development and treats any assertion of modelled risk as a fabrication.

### What the judge reads as "the answer"

**Closed by the author 2026-09-22: `critical_events_complete` is measured over the
whole of what the final answer produces.**

* Free arm: the entire reply.
* Structured arm: `initial_state`, `critical_transitions`, `final_state`,
  `interpretation` and `limitations`.

The structured arm's factual slots are rendered into neutral Greek sentences by the
frozen slot renderer (`stage_c_render_slots.py`, unchanged, its hash recorded in every
bundle) and joined to the prose as one text with no label. Judging the prose alone
would have required the structured model to repeat in prose what it had already placed
correctly in its mandatory slots, and would then have failed it for coverage when it
did not.

Residual, disclosed: rendered sentences are formulaic, so a judge may still infer the
arm from style. The prompt tells it to read nothing into style. That is the limit of
what a prompt can do and it is not the same as the signal being absent.

### Judgeability and NOT_REACHED

A record is judged only when it has a narrative surface **and** its own numeric
payload equals the frozen gold payload for that case. The second condition is not
bureaucracy: the gold's events are derived from the payload produced under the gold
arguments, so a record whose Stage A extracted a different horizon narrated a
different series and cannot be scored against this gold.

Such a record is `NOT_REACHED`. The run fails overall, on the stage where it actually
failed, and Stage C claims no factual verdict. `NOT_REACHED` counts are published per
cell with their reason.

## 4. The structured arm

Diagnostics only, reported beside the comparison and never inside it: schema
validity, field-level accuracy, critical-transition **precision** and **recall**.

`exact_set` is removed completely: not a gate, not a diagnostic, not a result, not an
SSP term, not a thesis claim. Historical artefacts that contain it are untouched and
are not used. `axis3_v3_aggregate.py` refuses to emit an output that names it, and a
test enforces its absence from the v3 instrument.

## 5. Statistics, declared in advance

The experimental unit for any inferential statement is the **case**, with its five
repetitions kept together. Repetitions of one case are not independent experiments.

- `pass^1` is the record rate; `pass^5` is the share of cases passing all five.
- Intervals are a **case-cluster bootstrap** (seed 20260922, 20000 draws) that
  resamples whole cases. A binomial interval over the records is not admissible.
- The paired comparison is an **exact two-sided sign test** over case-level outcomes,
  33 pairs per model, 66 pooled.
- Every pooled figure is printed with its category counts and its per-gate breakdown.
  No single average is ever published alone.

The study is descriptive by design. A difference that the sign test does not separate
from chance is reported as descriptive and never as an accuracy effect.

## 5b. Repeatability audit, pre-declared 2026-09-23

A single judge's run-to-run stability is a property of the result, not a detail, and
the pilot already measured one accidental instance of it: a byte-identical answer
judged five times gave four passes and one failure. This declares the measurement in
advance so the number cannot be chosen after the fact.

**Sample.** 66 judgeable records, selected with the published seed
`axis3-v3-repeatability-20260923`, balanced as evenly as 66 over four cells allows:
17, 17, 16, 16. The selection runs **before** the main judging pass and its file is
written before any judgement exists. A shortfall in any cell is reported rather than
silently absorbed.

**Second pass.** The same frozen judge, prompt, schema and effort, one independent
call per record, on those 66 records only, written to its own folder.

**What it may and may not do.** It may not replace a primary label, may not feed a
majority vote, may not break a tie and may not trigger a third pass. The primary
labels stand exactly as the main pass wrote them. This audit reports how reproducible
they were and decides nothing.

**Reported.** Agreement and Cohen's kappa on the overall `stage_c_pass`, with its 2x2
table; agreement separately on each of the six questions, with each pass's own
distribution beside it; and every record whose overall verdict flipped, named, with
the questions that moved.

**Read it as a floor on precision, not as a correction.** A kappa below one does not
mean the primary labels are wrong. It means an absolute Stage-C rate is a rate under
one judge on one pass, and every published absolute number carries that caption. The
paired case-level comparison is better protected, because both arms face the same
judge under the same conditions and the unit of inference is the case.

## 6. Provenance to freeze before execution

Recorded in `blind_manifest.json` at build time: the gold hash, the frozen-payload
hash, the rubric hash, the judge-prompt hash, the judge-schema hash and the builder
hash. Recorded at judging time: judge model, reasoning effort, temperature where the
provider exposes it. The system prompt, tool schema, answer schema and model presets
are unchanged and are pinned by their existing hashes
(`SYSTEM_PROMPT_SHA256[:12] = f005ab5e76d6`, `TOOLS_SCHEMA_SHA256[:12] = 0c5ae805199e`,
`FINAL_ANSWER_SCHEMA_SHA256[:12] = d079a01b22d4`).

## 7. Decisions not resolvable from code or stored data

Recorded rather than silently defaulted. Item 1 was closed by the author on
2026-09-22; items 2 to 4 remain open.

1. ~~Which limitation elements gate question 5.~~ **CLOSED by the author, 2026-09-22.**
   Exactly three elements gate Stage C and no others: (1) free-burning scenario,
   (2) suppression is not simulated, (3) traffic and local conditions are not captured
   AND the final decision belongs to the operational commander. Whether the fuels are
   described as **uncalibrated** is measured descriptively, published as its own rate,
   and does **not** gate `limitations_complete` or `stage_c_pass`. Reason recorded with
   the decision: on the stored 2026-09-01 data that element was absent from 35 percent
   (luna) to 98 percent (gemini) of replies, so gating it would make Stage C largely a
   test of one phrase. Frozen in `stage_c_v3_gold.json` and in the rubric.

2. **Whether the run set is both arms at 33 cases.** This document assumes a new,
   simultaneous free and structured run: 33 x 5 x 2 models x 2 arms = 660 runs.
   Simultaneity is the only way to remove the declared B15 confounder, that the
   existing two arms are more than 20 days apart on unpinned model aliases.

3. **Whether E04 and K01 stay in the denominator.** They are kept, with their small
   Stage-C denominators printed. Removing them after seeing results would be a
   post-hoc exclusion and is not permitted by this document.

4. **The judge is a single instrument with no agreement statistic.** That is the
   deliberate simplification of this design. Its cost is that Stage-C absolute levels
   depend on one judge; its protection is that the judge is identical and blind
   across both arms, so the comparison is not biased by it, and that every verdict
   carries quotable evidence.

## 7b. Instrument pilot, executed 2026-09-22

Eight blinded records, cases A01 and I04, repetition 1, one free and one structured
answer per model, taken from the stored runs. Claude Opus at high effort, the frozen
prompt and schema, **one independent call per record**. The pilot exists to test the
instrument, not to produce a result, and no figure from it enters any published table.

**Round 1 found one defect, in the record and not in the rubric.** Five of the eight
answers state how many pins the operator supplied ("the two pins merged into one
front"). The pins are given to the agent directly and never appear in the tool
payload, so the judge could not verify the count: it marked the phrase unsupported on
three records and accepted it on two, which is an inconsistency of the record
construction rather than a disagreement about faithfulness. The judge diagnosed it
itself, calling it "a record-construction gap rather than an obvious hallucination".

**The fix was in the builder, not the rubric.** Blinded records now print the pins
exactly as the agent received them, under their own heading. The pins are identical in
both arms, so they signal no arm, and the rubric, the gold and the pass rule are
untouched.

**Round 2, same eight records, rebuilt.** Unsupported-information flags fell from 3 to
0 and the phrase was accepted everywhere. `stage_c_pass` was **identical on all eight
records across the two rounds**, two passes and six failures both times. One item moved
without changing an outcome: on one record the judge called the end state uncovered in
round 1 and covered in round 2. That is the instrument's own run-to-run variation on a
single event and is recorded rather than smoothed away.

**What the pilot confirms.** The judge answers all six questions, quotes verbatim and
names payload fields or `event_id`s, distinguishes `impacted` from `at_risk` where the
Greek is ambiguous, treats a stated hour range as covering the hours inside it, reports
`uncalibrated_fuels_mentioned` without letting it touch the gate, and returns output
that passes the validator unchanged. Nothing in it named `exact_set` or any retired
rule.

**What the pilot does not settle.** Eight records cannot establish the judge's absolute
level, and one event, the hour-0 initial state, decided five of the six failures. That
is the instrument discriminating rather than malfunctioning: the one record that states
"hour 0: 11 at risk, 6 without a route" passes it and the others, which begin at hour 1,
do not. It does mean the Stage-C rate will be sensitive to `E_INITIAL`, which is
declared here before the full run rather than discovered after it.

## 7c. Instrument pilot, round 3, executed 2026-09-22

The same eight records, rebuilt after the two closures of 2026-09-22: `E_INITIAL`
removed from every case, and the structured arm's factual slots rendered into the
judged answer. Same judge, same effort, same frozen prompt, one independent call per
record.

| | round 2 | round 3 |
|---|---:|---:|
| `stage_c_pass` | 2 of 8 | **7 of 8** |
| `critical_events_complete` failures | 5 | **0** |
| `has_unsupported_information` failures | 0 | 0 |
| `has_contradicted_information` failures | 1 | 1 |

**Which change did what, isolated.** Three of the five flips are free-arm records
whose answer text did not change by a single byte between the two rounds; only the
gold changed, so their flip is the `E_INITIAL` removal and nothing else. One
structured record flipped on the same ground. The fifth, R0004, failed round 2 on
`E_INITIAL` **and** `E_FINAL`, and its prose contains neither the final cut-off count
nor the removed road segments nor the final hour; its end state reaches the judge only
through the rendered slot sentence. That record is the direct evidence that the
factual block now participates in coverage, and it would still have failed if only
`E_INITIAL` had been removed.

**The one remaining failure is stable and real.** The same record failed in all three
rounds, for the same reason each time: "the first settlement enters risk at hour 2"
while `at_risk` is already 11 at hour 0. The judge separated this from the correct
phrasing in a sibling record that says the first settlement enters the **perimeter**
at hour 2, which is `impacted` going 0 to 1 and is right. One phrase, one field, three
consistent verdicts.

**Descriptive measurements, gating nothing:** `initial_state_conveyed` true on 5 of 8,
`uncalibrated_fuels_mentioned` true on 6 of 8.

**What round 3 does not establish.** Eight records cannot fix the judge's absolute
level, and with `E_INITIAL` gone the pilot no longer exercises a coverage failure at
all: every required event was covered in all eight. The instrument's behaviour on a
genuine omission is evidenced by rounds 1 and 2, where it caught omissions, quoted
them and named the `event_id`.

## 7d. Instrument pilot, round 4, executed 2026-09-22

The same eight records, rebuilt after the disclosures were split out of
`critical_events_complete` into their own gate. Same judge, same effort, one
independent call per record.

**Result: identical to round 3.** Seven of eight pass, the same one fails, for the
same reason it failed in all four rounds. `required_disclosures_complete` came back
true on all eight and was answered correctly in both situations the pilot contains:
the four A01 records have an empty disclosure list, which the rubric says answers
true, and the four I04 records each carry the forecast-weather disclosure, which the
judge confirmed as explicitly present. Splitting the gate changed no outcome, which is
what it should do on records that satisfy both halves.

**The new gate is untested on a negative, and that is stated rather than implied.**
Every record in this pilot satisfies its disclosures, so the judge was never asked to
return false. A keyword screen over the stored corpus, which is not the judge and is
only a plausibility check, finds 9 of 340 forecast records containing no
forecast-related word at all, 5 free and 4 structured, so the gate is not vacuous and
will have occasion to fire in the full run. The front-count disclosure shows no
keyword-level absence, but keyword presence is not correctness, so that says little.

**The one surviving failure, across four rounds and three different record formats:**
"the first settlement enters risk at hour 2" while `at_risk` is already 11 at hour 0.
The same judge accepts the sibling record that says the first settlement enters the
**perimeter** at hour 2, which is `impacted` going 0 to 1 and is right. One phrase,
one field, four consistent verdicts.

## 7e. Disclosure-gate negative control, executed 2026-09-23

The pilot never asked the judge to reject a disclosure, so the gate was proven to
recognise a disclosure that is present and not to notice one that is absent. This
closes that.

**Method.** One pilot record whose case requires the forecast-weather disclosure and
which had passed all six questions was copied twice. `NC002` is that record with only
its id rewritten. `NC001` is the same record with the single sentence carrying the
disclosure deleted, and nothing else touched. Payload, gold, operator request, pins,
every critical event, the advice and all three limitation sentences are byte-identical
between the two; the builder verifies this and refuses to write otherwise. The same
frozen prompt and schema, one independent call per record. **No frozen artefact was
modified to run this.**

**Result, as predicted.** `NC001`: `required_disclosures_complete` **false**,
`stage_c_pass` **false**, and that is its only failure. The judge's evidence names
`E_WEATHER_FORECAST` and `inputs.weather_source = "forecast"` and says the answer
nowhere states that the weather came from forecast rather than archive. `NC002`:
`required_disclosures_complete` **true**. The gate distinguishes a present disclosure
from an absent one on otherwise identical text.

**The two records are technical controls and are not data.** They are hand-built text,
belong to no run, carry no model, arm or repetition, and must never enter a result, a
denominator, a rate or a comparison. Their folder carries a README saying so.

## 7f. A measured instability, found by the control and reported rather than buried

`NC002` is byte-identical in its answer to a record judged four times earlier in the
pilot series. Counting the control, that answer has now been judged **five times
independently**: four passes and one failure.

The disagreement is on one elliptical Greek clause, "to the 2 settlements at risk a
further 3 without a route are added", under a heading covering hours 1 to 2. Read as
increments it is exactly right (at_risk 11 to 13, cut_off 6 to 9). Read as levels it is
wrong. Three of the four earlier rounds **identified this clause by name** and resolved
it as not a contradiction; the fifth judgement resolved it as one. The judge is
consistent in finding the ambiguity and inconsistent in scoring it.

**What this is and is not.** It is not a defect of the negative control, which behaved
exactly as designed. It is not a case of the gold being wrong: the sentence really is
ambiguous. It is the single-judge design's run-to-run noise, measured for the first
time, at **1 flip in 5 on one ambiguous record**.

**What it costs and what limits it.** Absolute Stage-C rates carry this noise, so a
published rate is a rate under one judge on one pass and must be captioned that way. The
paired comparison is better protected: both arms face the same judge under the same
conditions, and the unit of inference is the case with its five repetitions kept
together, so independent noise of this kind inflates neither arm systematically. That is
a reason to trust the difference more than the level, and it is not a reason to trust
either without the caveat.

**Not fixed by changing anything.** Re-judging until the answers settle would be
selecting the result. A second judge would reopen exactly the multi-judge machinery this
design was built to retire. The honest option is the one taken here: measure it, publish
the rate, and caption every absolute number accordingly.

## 7g. Time-dependent fixture correction, applied 2026-09-23 before any judging

A technical correction of the test fixtures, made after the 660 production runs and
before a single Stage-C judgement. It is not a finding about any model and not a
limitation of the study.

**What was wrong.** C01 to C04 give a fixed start date in late August 2026. The tool
selects forecast weather for a start date close to the wall clock and archived weather
for an older one. When the gold payloads were first frozen, from the 2026-09-01 runs,
those dates were a few days old and the tool returned `forecast`; on 2026-09-23, when
the production runs were made, they were nearly four weeks old and it returned
`archive`. Every numeric value was identical between the two; only
`inputs.weather_source` moved. Left uncorrected, all 80 records of these four cases,
every one of which passed Stage A and Stage B, would have been excluded as
`NOT_REACHED`, and the gold would have demanded a forecast disclosure that the payload
no longer supported.

**What was changed.** For C01 to C04 only: the frozen payload's `weather_source` is set
to `archive` and the forecast-disclosure event is removed, exactly as for every other
archive case. This restores the cases' own declared design, fixture `pins_archive` and
no `disclose_forecast_weather` policy. Required events fall from 122 to 118 and policy
disclosures from 29 to 25; eleven cases now carry no required disclosure.

**Stale wording in two case prompts corrected at the same time.** C01 said "today" and
C03 "yesterday evening" alongside an explicit date, which stopped being true once the
date aged. Both are rewritten with the date alone. The date, the time and every
expected tool argument are unchanged, so no model run is repeated.

**What was not changed.** The raw records are untouched and keep the exact wording the
models received; the judge is shown that recorded text, because it is what each model
answered. The other 29 cases are byte-identical in both the gold and the frozen
payloads. Verified mechanically: after the correction, all 603 production records that
passed Stage A and Stage B carry a payload identical to the corrected gold.

## 8. Artefacts

| file | role |
|---|---|
| `agent_eval_cases_stagec33.jsonl` | the frozen 33-case Stage-C set |
| `stage_c_v3_frozen_payloads.json` | the numeric tool payload per case, stamps removed |
| `stage_c_v3_gold.json` | the per-case gold definitions |
| `stage_c_v3_rubric.md` | the record-level rubric, the judge's only instrument |
| `stage_c_v3_judge_prompt.txt` | the judge prompt skeleton |
| `stage_c_v3_judge_schema.json` | the judge output contract |
| `stage_c_v3_build_gold.py` | builds and freezes the three artefacts above |
| `stage_c_v3_repeatability_audit.py` | locks the 66-record sample, then reports agreement and kappa |
| `stage_c_v3_negative_control.py` | builds the disclosure-gate negative control |
| `stage_c_v3_build_judge_inputs.py` | builds the blinded records |
| `stage_c_v3_validate_judge_output.py` | refuses an incomplete or unevidenced pass |
| `axis3_v3_aggregate.py` | the final metrics, with no `exact_set` |
| `tests/validation/test_stage_c_v3.py` | the properties above, enforced |

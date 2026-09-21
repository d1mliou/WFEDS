# Stage-C rubric v2.0, arm-neutral: the operational judging document

FROZEN JUDGING ARTEFACT. This file, `scripts/validation/stage_c_judge_prompt.txt`
and `scripts/validation/stage_c_render_slots.py` are the three artefacts of the
Axis-3 structured-output experiment that must not move once judging starts. The
sha256 of all three is recorded in `stage_c_full/stage_c_judge_provenance.json`
for every judging pass, so the exact instrument behind any stored label set is
recoverable. Computing them: `python scripts/validation/stage_c_hashes.py`.
Editing any of them re-freezes the experiment and makes labels produced under
the previous hash non-poolable with labels produced under the new one.

## What this document is

It is section 2 of
`scripts/validation/axis3_structured_output_preregistration.md` (revision 6),
lifted verbatim into a standalone operational document, so that a judge or an
auditor can be handed this file and nothing else. The rule text below is not a
paraphrase and not a summary: sections 2.1 to 2.9 and 2.11 are byte-identical to
the pre-registration, headings promoted by one level and nothing else changed.
Everything authored for this document is confined to this front matter and to
the two appendices, both of which are labelled as such.

Section 2.10 is deliberately NOT reproduced. It is the judge-admission procedure
and it is full of the frozen run's own pass rates and per-model clean rates,
which 2.9 forbids the judge to see. Admission is decided before judging starts,
by whoever runs the calibration, and it is not the judge's business.

## How this document is used, per part

| part | who reads it | goes into the judge prompt |
|---|---|---|
| 2.5 label vocabulary | the judge | YES |
| 2.6 the four boundary tests T1 to T4 | the judge | YES |
| 2.7 the tie-breaks R1 to R8 | the judge | YES |
| 2.8 the worked examples | the judge | YES |
| 2.1 to 2.4 surfaces, renderer, claim list, contract outcomes | the harness author and the auditor | NO |
| 2.9 judge protocol | the harness author | NO |
| 2.11 audit intensity | the auditor | NO |

Only 2.5 to 2.8 are pasted into the judge prompt, in that order. That scope is
2.9's own rule and it is why the other sections, which name arms, models and
stored rates, can live in the same file without reaching the judge.

## The decision procedure on one card

Apply in this order, stopping at the first rule that decides the claim.

1. T1 to T4 of 2.6, in strict precedence. T1 outranks T3: an invented subject is
   NOT_IN_RESULT however carefully it is hedged.
2. R1 to R8 of 2.7 as tie-breaks, where the tests of 2.6 leave the claim open or
   where two of them appear to reach. The tie-breaks are the whole content of
   v2 over v1: the label set is unchanged and the boundary is not.
3. The worked examples of 2.8, which are the calibration for the cases the
   rules state abstractly.
4. One label per claim, from the five of 2.5, no others, no blanks.

## Two residuals the judge must be told about, not left to discover

The worked examples of 2.8 name the model and case each historical claim came
from ("gemini A01#2", "luna I02#1", "terra A02#5"). Those tags are the
provenance of stored 2026-09-01 free-baseline rows quoted as calibration. They
say nothing about the record now being judged, which is blind, and they must not
be read as a hint about it. They are kept because the examples are quoted
verbatim and an edited example is a different instrument.

The rendered slot sentences of 2.2 are templated, so they are recognisable as
belonging to the structured arm, and batches are single-arm by construction
(2.9). Complete blinding is therefore not claimed anywhere in this experiment.
The mitigation is that every rule in 2.5 to 2.8 references only the claim text
and the payload, so the rules are arm-independent by construction, and applying
them to a sentence you suspect is templated must produce the same label as
applying them to the same assertion in free prose.

---

## 2.1 The four text surfaces

Computed deterministically per record from the **parsed** object, never from a raw JSON string. This removes the escaped-Greek failure mode and the Latin-key dilution of `is_greek` in one step.

| surface | free baseline arm (the stored 2026-09-01 replies) | structured arm | consumed by |
|---|---|---|---|
| `policy_text` | the reply verbatim | `interpretation` + "\n" + `limitations` | C4 policy checks |
| `numeric_text` | the reply verbatim | `policy_text` + "\n" + rendered slot sentences | C1a numeric traceability |
| `slot_claims` | empty | one rendered Greek sentence per filled slot entry | claim list |
| `prose_claims` | `atomic_candidates(reply)` | `atomic_candidates(policy_text)` | claim list |

`atomic_candidates` is agent_eval_sample.py:51-60, unchanged, same 12-character floor, all arms. On a free arm `policy_text == reply` and `numeric_text == reply`, so the surface rule is a no-op there by construction. That is the proof that it is fair, and it is why the change must still be applied to the stored 2026-09-01 run via `--rescore` (the flip count printed by `rescore` must be 0; if it is not, stop and explain before proceeding).

**The numeric surface has two computed values, and only one of them is published. Stated by revision 6, because the document was silent and a reader would otherwise take the number stored in the raw row for the quantity this table defines.** `agent_eval.py` computes `numeric_ok` at run time over `_narrative_surface()` alone, that is `interpretation` plus `limitations`, because no rendered slot sentence exists at the moment a row is written. On the structured arm that run-time value is therefore the **prose-only PROVISIONAL lower bound**, not the `numeric_text` of this table: it is computed over a strictly smaller surface, so it can only ever be equal to or more permissive than the defined quantity. `stage_c_structured.py` recomputes the **2.1-conformant** value offline, over the prose surface **plus** the rendered slot claims of 2.2, and writes it into the row under a **separate field, leaving the run-time `numeric_ok` byte-untouched**, so that the provisional value and the conformant value are both on disk, both auditable, and impossible to mistake for one another. **The published C1a figure is the recomputed one**, in the headline, in every variant and in every per-criterion cell; the run-time value is a diagnostic and carries the caption "prose-only lower bound" wherever it is shown at all. The recomputation costs no model call and no re-run, because C1a is a pure function of stored data (4.3, 5.0 operation 1). **Field names, confirmed against the implementation on 2026-09-21:** the recomputed pair is `numeric_ok_full_surface` / `suspicious_numbers_full_surface`, and `numeric_text`, `slot_claims` and `slot_claims_error` are written beside them so the auditor sees byte-identically what the judge saw. the recomputed verdict and its suspicious-number list are written as a `numeric_ok`-shaped pair under their own names in `stage_c_structured.py`, alongside the rendered surface they were computed over; this document must be updated to quote those names verbatim once that module's field list is final, and until then the names here are described rather than quoted. On a free arm the two values are identical by construction, because a free arm has no slot claims, so none of this reaches the baseline column.

**The narrative surface**, used by C2 and by the judgeable-surface test of 2.4, is defined once per arm and never extended: on a free arm it is `reply` verbatim; on the structured arm it is `parsed["interpretation"] + "\n" + parsed["limitations"]` when both keys are present and are strings, and the empty string otherwise. No partial parsing, no fenced-block extraction, no bracket balancing: all three are repair, and repair is forbidden.

## 2.2 The frozen slot renderer

Rationale: if only prose were judged, the structured arm would face a strictly smaller and semantically different claim population and its clean rate would rise mechanically. Rendering the filled slots back into Greek claim sentences restores a comparable population, so the model's slot values face the same instrument the free arms' numbers face.

Whitelist: `initial_state`, each `critical_transitions` entry, `final_state`, and `status` when not `ok`. One claim per entry, not per field: a row claim is CONTRADICTED if its `period` or any value is wrong. **Added by revision 6, because the document was silent on the branch and the implemented renderer has to decide it: when `status` is not `ok`, the status sentence REPLACES the factual slot claims and the three arrays are not read at all.** The reason is the contract itself: semantic rule 4 of 1.3 requires all three arrays to be empty whenever the status is not `ok`, so a non-`ok` object that still carries filled arrays is contract debris, and rendering that debris would put forbidden factual claims in front of the judge and let a record that declared it had produced no result be judged as though it had produced one. Templates, with noun phrases taken verbatim from field_meanings and period wording following R7 so the renderer cannot itself commit the h0 error:

- `Στην έναρξη (ώρα {period}): {settlements_without_route} οικισμοί χωρίς καμία διαδρομή διαφυγής.`
- `Στην ώρα {period}: {settlements_without_route} οικισμοί χωρίς καμία διαδρομή διαφυγής.`
- `Τελική ώρα {period}: {settlements_without_route} οικισμοί χωρίς καμία διαδρομή διαφυγής, {road_segments_removed} οδικά τμήματα αφαιρέθηκαν.`
- `Το εργαλείο δεν παρήγαγε αποτέλεσμα: {status}.`

Slot-claim accounting, which follows from the case structure and must be printed: slot claims per record are |critical events| + 2, that is 5 (B02) to 10 (A03), mean 6.38 over the 13 cases, against a 2026-09-01 prose-claim mean of 10.9 (luna), 11.7 (terra) and 15.4 (gemini) in pass 1. The slot tier is therefore a large minority of the structured claim population and tier-separated reporting is mandatory rather than cosmetic. In B02 alone the `final_hour` period (4) is also a critical transition, so the renderer emits two non-identical sentences about hour 4 and the deduplication rule of 2.3 does not merge them.

Declared asymmetry: because the phrasing is ours, the structured arm cannot commit a phrasing-level field-confusion or hour-attribution error *inside its slots*. That is the intervention's mechanism, not a scoring artefact. Both families remain fully available to it in `interpretation`, where the 2026-09-01 run's own errors also lived. Negatives must therefore be reported per tier (slot vs prose), and claim-level rates must never be pooled across tiers.

## 2.3 Canonical claim list

`slot_claims` in schema order, then `prose_claims` in text order, numbered 1..N, no tier marker, no arm marker. Whitespace/case-identical duplicates deduplicated and labelled once; an inexact restatement of a correct slot survives as its own claim and is judged. The judge is told to split and merge as the 2026-09-01 labellers were, in identical wording, keeping derived ids (`7a`, `7b`, `7+8`).

## 2.4 Contract outcomes, judgeable surface, and records with no judged surface

This subsection is rewritten by correction 4. It defines one taxonomy, one arm-neutral surface rule, and how a record with no surface is handled inside the **common** metrics.

**The nine contract outcomes**, mutually exclusive, assigned by first match in this precedence order, recorded on every raw row as `contract_outcome` in both arms. **Outcomes 1 to 4 are arm-neutral and are evaluated on every arm. Outcomes 5 to 8 are contract-specific and are evaluated only when `output_contract == "structured"`. A free-arm row that clears outcomes 1 to 4 is `free_text`, outcome 9, and is never `not_json`.** This rule is not a convenience: without it a free Greek reply, which never parses as a JSON object, resolves to `not_json`, the surface paragraph below then denies it a judgeable surface, and edit C2 forces `stage_c = False` on all 65 free-baseline rows per model, scoring the baseline 0.000 on C1b, C2, C3 and C5-common. An implementer would notice and patch it, and that patch would be exactly the unpre-registered decision this document exists to prevent.

1. `runtime_error`. `row["error"]` is set. Provider or harness exception, not a model output.
2. `round_ceiling`. `reply` is the harness sentence at agent.py:354, "Σταμάτησα - πολλές διαδοχικές κλήσεις εργαλείου." (48 characters). Reaching :354 requires four rounds that each carried a tool call, so `n_tool_calls >= 4`. Not a model output. **Line number corrected by revision 6: revisions 1 to 5 cited this sentence at agent.py:284, which is where it sat before the contract work grew the file; the line-number note of 4.3 already recorded :354, and the two now agree. The string itself was verified unchanged, byte for byte, so only the position moved and no outcome definition is affected.** **Stated generally here, because this is where the review caught it: every `agent.py:NN` and `agent_eval.py:NN` citation in this document is INDICATIVE as a position and AUTHORITATIVE only as a description of the object it names. The quoted strings, the hashes and the function names are the authoritative identifiers, and where a citation and the code disagree on a number, the code is right and the citation still points at the right object.** The implementation follows that same rule rather than the line number: `stage_c_structured.py` compares a reply against this literal string, and `tests/validation/test_stage_c_structured.py` reads `agent.py` as text and asserts the two cannot drift.
3. `contract_not_attached`. `n_tool_calls == 0`, so no `role:"tool"` message ever existed and `response_format` was never sent (the `tool_done` guard of edits A2 and A3). Not a contract failure. On a free arm the same condition means no tool call was made at all, so there was no post-tool call for a contract to attach to; the name is kept so that one taxonomy covers both arms, and 3.3 says what it means on each. **In this scope it is not harness state.** All 13 in-scope cases carry `must_call_tool=true`, so `contract_not_attached` with no runtime error is a **Stage-B model failure**: `score_stage_b` already returns False, `deterministic_pass` already returns False, and the row therefore already fails C5-common. It is excluded from the two intervention-only validity rates because a contract that was never sent cannot have been violated, not because the failure is being excused, and the count is printed beside both rates and beside the Stage-B count.
4. `no_final_message`. `reply` is empty after strip and none of 1 to 3 applies. This is the strict-mode refusal and the content-filter block: agent.py:272 does `reply = (msg.content or "").strip()`, so a refusal arrives as an empty string with `error` None and is today indistinguishable from a normal answer in the row schema. Split by `finish_reason`: `content_filter` is a provider event and is excluded from the schema-validity denominator; anything else is a model refusal of the contract and counts as invalid.
5. `not_json`. A non-empty reply that does not parse as a JSON object. Split by `finish_reason`: `length` means truncation under constrained decoding, `stop` means the model produced prose and ignored the contract.
6. `schema_invalid`. Parses as an object, fails the JSON Schema: missing required key, extra key under `additionalProperties: false`, wrong type, value outside the status enum, wrong array item shape.
7. `validator_invalid`. Schema-conformant, fails one or more of the five semantic rules of 1.3.
8. `valid`. None of the above.
9. `free_text`. `output_contract == "free"` and none of 1 to 4 applies. A normal free-arm answer. It is never a contract failure, never a member of either validity denominator, and it has a judgeable surface whenever the narrative-surface test of 2.1 clears 120 characters, which on the stored data it always does (minimum in-scope reply 982 / 984 / 920 characters).

**Required capture, currently missing.** The raw row cannot express this taxonomy today. The verified key set of a stored raw row carries no `finish_reason` and no `refusal` field, and agent.py captures only `resp.model` into a local at :268. Without both, `not_json` truncation cannot be separated from `not_json` prose and a strict-mode refusal cannot be separated from a content filter. Edits A7 and B3 add them.

**The judgeable-surface rule, arm-neutral, stated once.** A record has a judgeable surface if and only if all three hold: `row["error"]` is falsy; a compact payload was parsed from that row's own `tool_outputs`; and the arm's narrative-surface function of 2.1 returns at least 120 characters (`NARRATION_MIN_CHARS`, agent_eval_prep_labeling.py:25, the same floor already applied to the 2026-09-01 run at :55-56). The only thing that differs between arms is the extraction function, and that difference is the intervention itself. The floor is arm-neutral in specification and asymmetric in effect: it is **inert on a free arm** at the stored reply lengths (minimum in-scope reply 982 gemini, 984 luna, 920 terra) and **live on the structured arm**, because the treatment's own wording caps `interpretation` at two to four short bullets, so a fully valid and fully complete object can fall below 120 characters and then fails C1b, C3, C2 and C5-common outright. That asymmetry is created by the treatment, not by the rule. The count of rows failing solely on the floor is published per arm, the narrative-surface character distribution is published per arm per model, and variant V3 of 3.3 bounds the loss.

**Consequences, applied identically to every arm.**

- A record with no judgeable surface **FAILS** C1b, C3, C2 and therefore C5-common. It is never pending and is never counted toward an optimistic upper bound. This rule takes precedence over the pending branch at axis3_full_system.py:190-194, which would otherwise mark a label-less row pending and count it toward the upper bound in `bounds` (:79-86), the exact opposite of the brief. The word FAILS is the gate value; the published per-criterion cells of C1a, C2 and C4 drop these rows and print the dropped count, which is a reporting rule and not a second gate.
- A record with no judgeable surface is **EXCLUDED** from the per-criterion denominators of C1a and C4, with the excluded count printed beside every such cell. Reason: both return a mechanical artefact on an empty string rather than information about the model. Verified by running the repository functions: `score_numeric_traceability("", compact)` returns `(True, [])`, a vacuous pass; `is_greek("")` is False and `has_limits_statement("")` is False, so `score_policies` returns exactly the two problems "not Greek" and "no limits statement". The row still fails C5-common through the no-surface rule and through `deterministic_pass`; only the published per-criterion cells are cleaned. This is not a new convention: section 3.2 already performs this correction by hand for gemini (9 policy-problem rows minus the 4 empty-503 rows leaves 5 genuine failures). The rule makes an existing manual correction systematic and arm-neutral instead of one-sided.
- **C5-common contains no schema term of any kind.** Schema validity and contract validity are never components of it and never columns of the comparison table. The complete structured-system pass (SSP) that does contain them is defined in 3.1 and is intervention-only.

**Stated affirmatively, because this is the consequence that surprises.** A `schema_invalid` or `validator_invalid` record that still has a judgeable surface is scored on C5-common exactly as a free-arm record is: its narrative surface is rendered, its claims are judged, its numbers and policies are checked, and **it can PASS C5-common while failing SSP**. That is deliberate, and it is what makes C5-common a fair comparison quantity: a model that writes a good Greek narration and a malformed envelope is not penalised twice on the channel the comparison measures. The envelope failure is carried entirely by SSP, by `json_schema_validity_rate` and by `contract_validity_rate`, all three of which are intervention-only. The only route by which a contract failure touches C5-common is the loss of the surface itself (no parse, no `interpretation` string, under 120 characters), which is the partial insulation the Residual paragraph below bounds and which the salvage variant V3 of 3.3 prices.

**Which outcomes keep a judgeable surface.** `valid` and `validator_invalid` always do, because schema conformance guarantees both strings exist. `schema_invalid` does when `interpretation` is present and is a string, which is the common case for an extra root key, a wrong leaf type or an unknown status value; it does not when `interpretation` is missing or is not a string. `runtime_error`, `round_ceiling`, `contract_not_attached`, `no_final_message` and `not_json` (truncated) never do. `not_json` (Greek prose) is the one pre-declared **choice** rather than a fact: under the primary variant it has no surface, because defining the structured arm's surface as "the parsed `interpretation`, or the whole reply when parsing fails" would give the structured arm two chances, obey the contract and be judged on a narrow self-selected surface or ignore it and be judged exactly as a free arm, which is a best-of-both rule and is not arm-neutral. Variant V3 of 3.3 publishes the cost of that choice as a number instead of burying it. **This paragraph governs the structured arm only.** On a free arm the judgeable-surface test never consults `contract_outcome`: it is exactly (`row["error"]` falsy) AND (a compact payload parsed from that row's own `tool_outputs`) AND (`len(reply.strip()) >= 120`). `free_text` therefore keeps a judgeable surface whenever those three hold; outcomes 1 to 4 never do, on either arm. Edits C2 and C3 must test the surface predicate, never the outcome value, and `contract_outcome` appears in `c_source` and `coverage_source` only as the recorded **reason**.

**Symmetry with the free arms, argued rather than asserted.** The free arms' nearest counterpart to an unusable output is outcomes 1 to 4, and the existing code already treats them as hard failures rather than pending: axis3_full_system.py:178-181 forces `stage_c = False` with `c_source = "runtime_error"` **ahead of** the human-override branch, and `deterministic_pass` (:64-76) returns False on the same condition. Verified on the frozen run: gemini has 4 such rows in scope (A02#4, A02#5, A03#4, I03#4), all with `reply == ""`, all excluded from both label passes (61 labelled records against 65 rows), all scored FAIL; luna and terra have zero. Verified further that no stored in-scope reply is short: minimum length 982 characters for gemini, 984 for luna, 920 for terra, so the 120-character floor is inert on the stored 2026-09-01 replies and cannot penalise them retroactively. The no-surface rule therefore does not import a new standard into a free arm; it states in arm-neutral language the standard the free arm already meets, so the structured arm meets the same one.

**Residual, to be stated in the document and in the thesis.** C5-common is insulated from schema validity as a *criterion*, but it is not fully insulated from the contract, because a parse failure removes the surface and a missing surface fails the gate. The insulation is partial by construction; the per-outcome counts of 3.3 make the size of the residual visible and variant V3 bounds it.

## 2.5 Label vocabulary: unchanged, and why

`SUPPORTED`, `CONTRADICTED`, `NOT_IN_RESULT`, `ADVISORY_INFERENCE`, `LIMITATION_STATEMENT`. Fixed in code at stage_c_aggregate.py:36-37, stage_c_audit_workbook.py:33-34, agent_eval_sample.py:20-26, and `NEGATIVE_LABELS = {"NOT_IN_RESULT","CONTRADICTED"}` at axis3_full_system.py:38 is the gate every published Axis-3 figure rests on. Adding a label would make the 2026-09-01 run's 2,409 pass-1 and 2,320 pass-2 stored claims unusable. One clarification, not a change: `clean_label_record` returns True for an empty `claims` list (axis3_full_system.py:51-57); under v2 an empty claim list returns None and the record falls under 2.4. Verified that this edit is a no-op on the frozen data: all 191 label records in both passes have non-empty claims lists, minimum 7 claims (luna pass 1), so C4 of section 4.3 is a guard for the future rather than a retroactive change.

## 2.6 The boundary rule: four tests in strict precedence

- **T1 invented subject.** The claim asserts or presupposes an entity or attribute the tool never returns: settlement names, road names, spread directions, winds, terrain or fuels, front geometry (head, flanks, perimeter), distances, per-settlement population or exits, per-settlement identity, a per-hour road or network quantity, a fire path, an external interface. Result NOT_IN_RESULT. Hedging does not rescue an invented subject. T1 outranks T3.
- **T2 conflict.** A value, field identity, hour, trend or rate the rows contradict. Result CONTRADICTED.
- **T3 permitted judgement.** A recommendation, priority ordering, timing judgement or caution that (a) names no entity outside the payload, (b) is grounded in at least one named payload value or an ordering derivable from it, and (c) sits in a record carrying a limits statement. Result ADVISORY_INFERENCE.
- **T4 the rest.** Limits framing gives LIMITATION_STATEMENT; a correct restatement gives SUPPORTED.

## 2.7 Tie-breaks R1 to R8 (this is what v2 adds)

- **R1 range attribution.** A delta or state attached to a stated hour range is judged against the whole range. Holds only for a sub-hour, CONTRADICTED.
- **R2 onset against h0.** "αρχίζει από την Xη ώρα" is judged against period 0. If the quantity is already non-zero at period 0, an onset at X > 0 is CONTRADICTED. A claim about the increase beginning at X is judged on the delta.
- **R3 settlement-action vs road-object.** Grammatical object = settlements tracks `cut_off` and is SUPPORTED when the timing is right. Grammatical object = routes/roads/network asserts a road quantity: CONTRADICTED when a value or an hourly trend is attached, NOT_IN_RESULT when no payload value is attached. `edges_removed` is the sole road quantity and exists only for the final hour.
- **R4 class-4 gloss.** Renaming class 4 as very high intensity / extreme / hard to fight, scoped to the FINAL front, is SUPPORTED. Adding a fire type (crown fire, πυρκαγιά κόμης) is NOT_IN_RESULT. Extending 96% to the whole period is CONTRADICTED.
- **R5 weather provenance.** Asserting forecast when `weather_source == "archive"`, or the reverse, is CONTRADICTED. One named field.
- **R6 spatial anchors in advice.** Generic tactics with no named place are ADVISORY_INFERENCE; the same sentence with "μπροστά από την πορεία" or "περιμετρικά του τελικού μετώπου" is NOT_IN_RESULT.
- **R7 hour numbering.** Period p is "ώρα p"; period 0 is "έναρξη" or "ώρα 0", never "1η ώρα". Mapping h0 to the first hour is CONTRADICTED.
- **R8 no double counting of a missing limits statement.** Without a limits statement a recommendation cannot be ADVISORY_INFERENCE, but it does not become negative by that fact alone: SUPPORTED if it asserts nothing beyond the payload, NOT_IN_RESULT if it does. The missing statement is caught once, by C4.

## 2.8 Worked examples (part of the rubric shown to the judge)

Six failing, six passing, all drawn from the stored 2026-09-01 label files, plus two boundary pairs that were real pass-1/pass-2 disagreements.

FAIL 1, R7. "3η ώρα: Η φωτιά φτάνει στον πρώτο οικισμό." (gemini A01#2, A01#5). CONTRADICTED: the `impacted` 0 to 1 transition is at period 2. Right value, wrong row.
FAIL 2, R2. "Η απώλεια πρόσβασης αρχίζει από την 1η ώρα." (luna I02#1, #3, #5). CONTRADICTED: period 0 already has `cut_off=6`; period 1 raises it to 9. The correct form names the start state first.
FAIL 3, R3. "Να διατηρηθούν διαθέσιμες οι 6 τελικές διαδρομές." (luna A02#2). CONTRADICTED: `routed=6` counts settlements that still have an escape route, not six routes. The numeral is traceable, which is why the numeric check alone cannot carry this criterion.
FAIL 4, R3. "Η πίεση στο δίκτυο εντείνεται έως περίπου την 5η ώρα." (terra A02#5). CONTRADICTED: `edges_removed` exists only for the final hour; there is no hourly road series.
FAIL 5, T1 over T3 (R6). "αντιπυρικές ζώνες σε στρατηγικά σημεία μπροστά από την αναμενόμενη πορεία της πυρκαγιάς" (gemini A01#2, A04#1). NOT_IN_RESULT: no spread direction, no trajectory, no front geometry in the payload.
FAIL 6, R5. "Τα δεδομένα καιρού βασίζονται σε πρόγνωση." on a case with `weather_source == "archive"` (gemini A04#1). CONTRADICTED.

PASS 1, T4. "Ο πρώτος οικισμός περνά σε περίμετρο στη 2η ώρα." (luna A01#1). SUPPORTED: `impacted` is 0 at 0 and 1, 1 at 2. The exact counterpart of FAIL 1.
PASS 2, R3 satisfied. "Η τελική ώρα αντιστοιχεί σε αφαίρεση 473 οδικών τμημάτων." (luna A01#1). SUPPORTED: road quantity, correctly confined to the final hour.
PASS 3, R4. "Το 96% του τελικού μετώπου είναι πολύ υψηλής έντασης." (gemini A01#1). SUPPORTED.
PASS 4, R5. "weather_source: archive, όχι μελλοντική πρόγνωση" (luna A01#1). SUPPORTED.
PASS 5, T3 with R6. "έμφαση σε έμμεση προσβολή και προστασία των οικισμών που δεν έχουν επηρεαστεί ακόμη" (gemini B01#1). ADVISORY_INFERENCE: no named place, grounded in `final_front_class4_pct` and `impacted` vs `at_risk`, limits statement present.
PASS 6, T3. "πρώτα οι ήδη αποκομμένοι, έπειτα οι οικισμοί που παραμένουν σε κίνδυνο αλλά διαθέτουν ακόμη διαδρομή" (terra I02#3). ADVISORY_INFERENCE. This is the LLM contribution the intervention must preserve: the schema can carry the counts, not the ordering argument.

BOUNDARY A, decided by R1. "Ώρες 1 to 2: στους 2 οικισμούς σε κίνδυνο προστίθενται ακόμη 3 χωρίς διαδρομή." (luna I04#1, #5). Pass 1 said SUPPORTED reading it as h1 deltas; pass 2 said CONTRADICTED because over h1 to h2 `at_risk` rises 11 to 16. R1 rules CONTRADICTED. Consequence to declare: R1 makes the stored 2026-09-01 run marginally stricter than its as-published pass-1 labels, which is exactly why that run must be re-labelled under v2 before it is compared with anything scored under v2, and why the as-published 75.4 / 72.3 / 15.4 figures stay a historical column rather than a comparator.

BOUNDARY B, decided by R3. "Από την 4η to 5η ώρα η αποκοπή διαδρομών κορυφώνεται." (terra B03#5). Pass 1 CONTRADICTED (wrong field), pass 2 SUPPORTED (timing matches `cut_off`). R3 rules CONTRADICTED; the SUPPORTED rewrite names οικισμοί, not διαδρομές. R3 exists because this one phrasing distinction flipped a scored record between passes and is the largest single source of label noise in the 2026-09-01 run.

## 2.9 Judge protocol

Judge: Claude Opus, high reasoning effort, the instrument used for the 2026-09-01 pass 2 (Decision log 2026-09-02). Temperature 0.0 where the labelling harness exposes sampling. [ASSUMPTION, declared gap] the two 2026-09-01 passes were Workflow fan-outs whose sampling settings were never recorded, so "same judge" is not assertable at the sampling level for the frozen 2026-09-01 run; that is one of the two reasons the calibration step in 2.10 exists.

Output forced through a JSON-schema contract, matching the 2026-09-01 pass 2. Batch size 12, built by `agent_eval_prep_labeling.py`.

**Batching facts, counted from the stored files and correcting revision 1.** One judge call is **one batch file**, never one record. The 2026-09-01 run's observed splits were gemini 12,12,12,12,12,1 (61 records), luna 12,12,12,12,12,5 (65) and terra 11,11,11,11,11,10 (65, batched at `--batch-size 11`); 18 batch files hold all 191 narrated records. Revision 1's claim of a uniform 12/12/12/12/12/5 split holds for luna alone.

**The arm-mixing claim is withdrawn.** Revision 1 asserted "batches mix baseline and structured records so batch composition carries no arm signal". That is not implementable with the harness as written: `agent_eval_prep_labeling.py` reads exactly one `agent_eval_raw.jsonl` per invocation (:41-42) and slices that single run's narrated list (:77), so batches cannot cross runs and therefore cannot cross arms. Arm-blind batch composition is not claimed. The residual is declared, and the mitigation stays the one already stated: every rule in 2.5 to 2.8 references only the claim text and the payload, so the rules are arm-independent by construction. The per-run batch split is recorded in `batching_provenance.json` (edit E2) rather than assumed.

What the judge sees, in this order: the rubric sections 2.5 to 2.8; `user_text` and `pins`; the compact payload verbatim with `ensure_ascii=False` including field_meanings; `tool_numbers_md` from `agent_eval_sample.tool_numbers` (agent_eval_sample.py:63-90); the numbered canonical claim list; the full judged-surface text.

What the judge never sees: the arm, the model under test, the case gold, the `cutoff_events` ground truth, the other pass, any prior human decision, any pass rate, the raw JSON envelope. Complete blinding is NOT claimed: templated slot sentences may be recognisable, and batches are single-arm by construction. Mitigation is as above; the residual is declared.

Output per record: `{uid, claims:[{claim_id, text, label, justification}], notes}`. The justification is one sentence and must name at least one payload field, or the literal token "no field" for a NOT_IN_RESULT whose subject does not exist at all. Enforced deterministically after the call; a failing record is re-queued once, then escalated to human with reason `judge_output_invalid`, never silently accepted. The harness adds `claim_id`, `tier` and `value_bearing` without showing them to the judge; the existing readers ignore unknown keys, so the files stay drop-in compatible with `axis3_full_system.py`, `stage_c_aggregate.py` and `stage_c_audit_workbook.py`.

Storage: `stage_c_full/stage_c_labels.json` and, where a second pass is run, `stage_c_full/stage_c_labels_pass2.json`; `batch_NN.json` gains `canonical_claims`, `judged_surface_text`, `policy_text` and `numeric_text` so the auditor sees byte-identically what the judge saw; `stage_c_judge_provenance.json` records model id, effort, temperature or "not exposed", the three artefact hashes, batch ids, agent count and UTC timestamps.

## 2.11 Audit intensity, equal on every arm

Census, not sample: 100 percent of records with any negative label; 100 percent of pass disagreements where a second pass exists; 100 percent of records that **have a judgeable surface** but whose label record is missing or invalid (`judge_output_invalid`, 2.9); records with **no** judgeable surface are not audited, because 2.4 decides them deterministically and edit C2 sets `human_stage_c_required = NO` for exactly that reason (`main()` filters on `required == YES` at `:395-399`), so they never enter the audit CSV and never enter the human-decision estimate of 5.7; 100 percent of **narrative critical-event coverage** decisions where `cutoff_hours_to_cover` is non-empty (**30 per arm under the two-model design**, 3 cases x k=5 x 2 models, see 3.1). Plus a seeded random verification sample of clean records on **every comparison arm**, drawn stratified by case, seed published to `stage_c_audit_sample.json` before the auditor opens the workbook. Structured arm only: every record where the deterministic slot check and the judge's slot-claim label disagree; that trigger can only add failures to the intervention, so the asymmetry is conservative with respect to the hypothesis and is reported as such.

**Withholding, mandatory, for the narrative coverage decision (correction 3).** The coverage decider is given, identically on every arm: `critical_event_hours` as the audit CSV column already supplies it; `user_text` and `pins`; the **narrative surface and only the narrative surface**, rendered as plain Greek in the same column with no JSON punctuation and no key names; and the verbatim decision rule. Withheld from that decision on the structured arm: the filled `critical_transitions` array in any form; the rendered slot claims; `initial_state` and `final_state`; the raw JSON envelope, including a pretty-printed copy; the structured-completeness verdict and `schema_errors`; and any `literal_hours_found` aid computed over anything other than the narrative surface. Operationally this means the coverage sheet must carry `judged_surface` and must **not** carry the raw JSON column at all, rather than merely placing it far to the right: if the object appears anywhere on the coverage row, the withholding is not implemented.

**Sheet-level blinding of the coverage decision, mandatory.** `run_id` is the audit CSV's first column and it names the arm (`load_audit` keys on `(run_id, uid)`, axis3_full_system.py:112-117), so withholding the object while printing the arm withholds nothing. The 60 coverage decisions are therefore presented in **one shuffled sheet under opaque decision ids**, seeded and published with the sample seed, with the `(run_id, uid)` mapping written to a separate file and joined back **only after all 60 decisions are recorded and written**. The sheet carries `decision_id`, `critical_event_hours`, `user_text`, `pins`, `judged_surface`, `literal_hours_found` computed over the narrative surface, and the verbatim decision rule, and nothing else. This is edit **C9** of 4.3 and it is a few lines in the audit writer; because it changes `write_audit` (`:292-296`), that function moves out of 4.2 into 4.3, and `apply_axis3_human_audit_structured.py` (4.4 item 8) gains the join step and a `--dry-run` check that every `decision_id` resolves. Residual, which the blinding does **not** remove and which 6.1 must keep: the narrative text itself can betray the arm, because a structured arm's `interpretation` plus `limitations` may read differently from a free reply. What can honestly be claimed after C9 is that the coverage sheet was **arm-blind by construction at the sheet level**, not that the decider could not infer the arm from the prose.

Attribution wording, mandated (Decision log 2026-08-27): "claims were labelled by Claude [exact model and effort], the user reviewed the complete labelled set and could override any label". Never "human-labelled, Claude verified".

---

---

## Appendix A, authored: cross-references this document does not reproduce

The lifted text cites other parts of the pre-registration. A judge needs none of
them to apply the rules; an auditor reading a label may want to know what was
meant. Nothing below adds or relaxes a rule.

* **1.3, the five semantic rules, and 1.2, the schema.** The structured arm's
  output contract. Semantic rule 4 is the one 2.2 leans on: when `status` is not
  `ok`, the three factual arrays are empty.
* **3.1, the five common criteria and the two gates.** Where a label becomes a
  score. C1a is numeric traceability, C1b critical-event coverage, C2 the
  narration gate, C3 the claim-level gate, C4 the policy checks, C5-common the
  combined comparison gate.
* **3.2, saturation.** Why some criteria carry no live variance and must be
  captioned as saturated rather than read as a result.
* **3.3, intervention-only metrics, and variant V3.** Schema-validity and
  contract-validity rates, which never enter the common comparison, and the
  salvage variant that prices the surface lost to unparsable output.
* **3.4, the critical-transition metrics.** Three reported quantities, one gate,
  all `ct_*` columns diagnostic and excluded from the common comparison by rule.
* **4.3, edits C1 to C9.** The scoring-code changes that carry 2.4 and 2.11 into
  `axis3_full_system.py`, including C9, the shuffled arm-blind coverage sheet.
* **5.0, 5.7, 5.8.** The judging economy, the human-effort estimate, and the
  2026-09-21 pilot that measured tokens and excluded gemini from the follow-up.
* **6.1 and 6.2.** The single claim this experiment can support, and the claims
  it must not make.

## Appendix B, authored: the payload fields the rules name

A navigation aid only. The authoritative gloss of every field is the
`field_meanings` block that ships inside the compact payload the judge is shown,
and where this table and that block differ, that block is right. No field is
defined here that is not already defined there or in the rule that names it.

| field | where it lives | the rule that turns on it |
|---|---|---|
| `cut_off` | each `change_hours` row, and `final_hour` | settlements with no escape route at all, NOT roads. R3 hangs on this distinction. |
| `routed` | each `change_hours` row, and `final_hour` | settlements that still have an escape route. FAIL 3 is a claim that read it as a count of routes. |
| `impacted` | each `change_hours` row, and `final_hour` | settlements inside the fire perimeter. FAIL 1 and PASS 1 turn on which row its 0 to 1 transition sits in. |
| `at_risk` | each `change_hours` row, and `final_hour` | settlements in the risk buffer. BOUNDARY A is judged on its h1 to h2 movement. |
| `period` | each `change_hours` row, and `final_hour` | the hour index. R7: period p is "ώρα p", period 0 is "έναρξη" or "ώρα 0", never "1η ώρα". |
| `edges_removed` | `final_hour` only | the sole road quantity in the payload, and it exists for the final hour only. R3 and FAIL 4. |
| `longest_route_km` | `final_hour` only | present for the final hour only, like `edges_removed`. |
| `final_front_class4_pct` | top level | share of the FINAL front in intensity class 4. R4 and PASS 3. |
| `weather_source` | `inputs` | `archive` or a forecast source. R5 and FAIL 6, one named field. |
| `change_hours` | top level | the per-hour series. Nothing outside it carries a per-hour quantity, which is why a per-hour road claim is unsupported. |
| `field_meanings` | top level | the payload's own glossary, shown to the judge verbatim. |

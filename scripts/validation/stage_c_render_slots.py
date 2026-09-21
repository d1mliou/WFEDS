"""The frozen slot renderer of the Axis-3 structured-output experiment.

Implements section 2.2 of
`scripts/validation/axis3_structured_output_preregistration.md` and nothing
else. It turns the whitelisted slots of a parsed `wfeds_final_answer` object
back into Greek claim sentences, so that the structured arm's numbers face the
same Stage-C instrument the free arms' numbers face.

Why this exists at all (2.2, rationale): if only prose were judged, the
structured arm would face a strictly smaller and semantically different claim
population and its clean rate would rise mechanically. The slot tier restores a
comparable population. It is a large minority of the structured claim
population (slot claims per record are |critical events| + 2, mean 6.38 over the
13 in-scope cases, against a 2026-09-01 prose-claim mean of 10.9 to 15.4), which
is why 2.2 requires tier-separated reporting and why the per-record slot-claim
count is printed and reported in 3.4.3.

Scope, stated so nothing else is read into this module:

  * One claim per whitelisted ENTRY, not per field. A row claim is CONTRADICTED
    if its `period` or any of its values is wrong, so the fields of one entry
    must never be split into two sentences.
  * Deduplication is NOT done here. It is the canonical-claim-list rule of 2.3,
    which operates on `slot_claims` + `prose_claims` together. In B02 the
    `final_hour` period (4) is also a critical transition and the two sentences
    about hour 4 are deliberately non-identical, so 2.3 does not merge them; a
    renderer that deduplicated would destroy that property silently.
  * Period wording follows tie-break R7 (period p is "ώρα p", period 0 is
    "έναρξη" or "ώρα 0", never "1η ώρα"), so the renderer cannot itself commit
    the h0 error that the rubric penalises in the models.
  * No schema validation. Enum membership, array cardinality and the five
    semantic rules of 1.3 belong to `stage_c_structured.py`. This module renders
    what it is given or refuses; it never repairs and never judges.

FROZEN ARTEFACT. `RENDERER_SHA256` pins the template set the way
`FINAL_ANSWER_SCHEMA_SHA256` pins the schema, and section 2 records it in
`stage_c_full/stage_c_judge_provenance.json` before any judging. Changing a
single character of a template changes this hash and INVALIDATES comparability
with every run already rendered under the previous hash: those records were
judged on different sentences and their labels cannot be pooled with new ones.
Edit a template only by re-freezing the experiment, never mid-run.

Run:
    python scripts/validation/stage_c_render_slots.py            # hash + templates
    python scripts/validation/stage_c_render_slots.py OBJ.json   # + rendered claims
"""
from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

# The four slot names double as template ids. Named keys rather than positional
# indices, because the hash below is taken over this mapping: a reordering must
# not be able to move a template without a reader seeing which one moved.
TEMPLATE_INITIAL = "initial_state"
TEMPLATE_TRANSITION = "critical_transitions"
TEMPLATE_FINAL = "final_state"
TEMPLATE_STATUS = "status"

# VERBATIM from section 2.2 of the pre-registration, byte-for-byte. The noun
# phrases are taken from the `_compact` field_meanings glosses (agent.py:221-230)
# so a rendered sentence adds no semantics the free arms did not also receive in
# every tool payload. Do not improve the Greek: the phrasing is the frozen
# instrument, and 2.2's declared-asymmetry argument rests on it being ours and
# fixed rather than the model's and variable.
TEMPLATES: dict[str, str] = {
    TEMPLATE_INITIAL: "Στην έναρξη (ώρα {period}): {settlements_without_route} οικισμοί χωρίς καμία διαδρομή διαφυγής.",
    TEMPLATE_TRANSITION: "Στην ώρα {period}: {settlements_without_route} οικισμοί χωρίς καμία διαδρομή διαφυγής.",
    TEMPLATE_FINAL: "Τελική ώρα {period}: {settlements_without_route} οικισμοί χωρίς καμία διαδρομή διαφυγής, {road_segments_removed} οδικά τμήματα αφαιρέθηκαν.",
    TEMPLATE_STATUS: "Το εργαλείο δεν παρήγαγε αποτέλεσμα: {status}.",
}

# Same serialisation convention as the schema hash of 1.2
# (`json.dumps(..., ensure_ascii=False, sort_keys=True)`), so the two frozen
# artefacts of this experiment are pinned by one rule and a reader can recompute
# either of them the same way.
TEMPLATE_CANONICAL_JSON = json.dumps(TEMPLATES, ensure_ascii=False, sort_keys=True)
RENDERER_SHA256 = hashlib.sha256(TEMPLATE_CANONICAL_JSON.encode("utf-8")).hexdigest()

# The three factual slots in schema order. The order is load-bearing: 2.3
# numbers the canonical claim list "slot_claims in schema order, then
# prose_claims in text order", and a claim id that moved between renderings
# would break every stored label that cites it.
_SLOT_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    (TEMPLATE_INITIAL, ("period", "settlements_without_route")),
    (TEMPLATE_TRANSITION, ("period", "settlements_without_route")),
    (TEMPLATE_FINAL, ("period", "settlements_without_route", "road_segments_removed")),
)


def _require_entry_list(obj: dict[str, Any], key: str) -> list[Any]:
    """Return obj[key] as a list, or raise naming the offending key.

    A missing array is a `schema_invalid` record, not something to paper over
    with an empty list: an empty list would render zero claims and the record
    would then look merely terse to everything downstream instead of malformed.
    """
    if key not in obj:
        raise ValueError(
            f"slot renderer: missing required key {key!r} in the final-answer object"
        )
    value = obj[key]
    if not isinstance(value, list):
        raise ValueError(
            f"slot renderer: {key!r} is {type(value).__name__}, not an array: {value!r}"
        )
    return value


def _require_int(entry: Any, field: str, where: str) -> int:
    """Return entry[field] as an int, or raise naming the offending entry.

    Never silent, by requirement: a missing or non-integer value must surface as
    a ValueError the caller decides about, not as a sentence carrying "None" or
    "3.0" that a judge would then label CONTRADICTED, which would record a
    renderer defect as a model error in the published claim counts.

    `bool` is rejected explicitly because it is an `int` subclass in Python
    while JSON `true` is not an integer to the schema; without this, `true`
    would render as the word "True" inside a Greek sentence.
    """
    if not isinstance(entry, dict):
        raise ValueError(
            f"slot renderer: {where} is {type(entry).__name__}, not an object: {entry!r}"
        )
    if field not in entry:
        raise ValueError(f"slot renderer: {where} is missing key {field!r}: {entry!r}")
    value = entry[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"slot renderer: {where} key {field!r} is {value!r} "
            f"({type(value).__name__}), not an integer"
        )
    return value


def render_slot_claims(obj: Any) -> list[str]:
    """Render one Greek claim sentence per whitelisted entry, in schema order.

    Whitelist of 2.2: `initial_state`, each `critical_transitions` entry,
    `final_state`, and `status` when it is not "ok".

    When `status != "ok"` the single status sentence is returned INSTEAD of the
    factual slots, not in addition to them. Semantic rule 4 of 1.3 says a non-ok
    status implies three empty arrays; a non-ok record that carries filled
    arrays anyway is `validator_invalid` and its arrays are contract debris, so
    rendering them would put claims the contract forbids in front of the judge.
    The branch is taken before the arrays are read at all, so a non-ok record
    also cannot fail to render on the shape of arrays it should not have.
    This substitution is not an implementation choice left to the reader: the
    whitelist paragraph of section 2.2 now states the same rule in the document,
    that on a non-ok record the status sentence REPLACES the factual slots and
    is never emitted alongside them, so the freeze and this function say one
    thing and a reader of either cannot infer the other behaviour.
    """
    if not isinstance(obj, dict):
        raise ValueError(
            f"slot renderer: expected a parsed JSON object, got "
            f"{type(obj).__name__}: {obj!r}"
        )

    if "status" not in obj:
        raise ValueError(
            "slot renderer: missing required key 'status' in the final-answer object"
        )
    status = obj["status"]
    if not isinstance(status, str):
        raise ValueError(
            f"slot renderer: 'status' is {status!r} ({type(status).__name__}), not a string"
        )
    # Enum membership is NOT checked here: an unknown status value is
    # `schema_invalid` (outcome 6 of 2.4), and such a record keeps a judgeable
    # surface, so it must still render rather than refuse.
    if status != "ok":
        return [TEMPLATES[TEMPLATE_STATUS].format(status=status)]

    claims: list[str] = []
    for slot, fields in _SLOT_ORDER:
        for index, entry in enumerate(_require_entry_list(obj, slot)):
            where = f"{slot}[{index}]"
            values = {field: _require_int(entry, field, where) for field in fields}
            # Extra keys inside an entry are ignored rather than rejected: under
            # `additionalProperties: false` they already make the record
            # `schema_invalid`, which is the validator's finding to report, and
            # such a record is still judged on the surface it does have (2.4).
            claims.append(TEMPLATES[slot].format(**values))
    return claims


def slot_claim_count(obj: Any) -> int:
    """Count the slot claims of one record.

    Required by 2.2 ("slot claims per record are |critical events| + 2, that is
    5 (B02) to 10 (A03) ... and must be printed") and reported per record in
    3.4.3. It counts by rendering rather than by counting entries, so the
    printed count can never disagree with the list actually handed to the judge,
    and so a malformed record raises here too instead of being counted as if it
    were sound.
    """
    return len(render_slot_claims(obj))


def _main(argv: list[str]) -> int:
    # Greek output dies under the Windows console's cp1252 otherwise, and a
    # renderer whose entire product is Greek is useless if it cannot be
    # eyeballed on the machine the experiment runs on.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print(f"RENDERER_SHA256 = {RENDERER_SHA256}")
    print(f"prefix          = {RENDERER_SHA256[:12]}")
    print("templates (frozen, section 2.2):")
    for template_id in sorted(TEMPLATES):
        print(f"  {template_id}: {TEMPLATES[template_id]}")

    for path in argv:
        with open(path, encoding="utf-8") as handle:
            obj = json.load(handle)
        claims = render_slot_claims(obj)
        print(f"\n{path}: slot_claim_count = {len(claims)}")
        for number, claim in enumerate(claims, 1):
            print(f"  {number:2d}. {claim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

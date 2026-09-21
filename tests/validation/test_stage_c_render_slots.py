"""Unit tests for the frozen slot renderer of section 2.2.

Two of these tests are guards rather than behaviour checks, and they are the
reason this file exists at all:

  * the four templates are asserted against section 2.2 of the pre-registration
    AS PARSED FROM THE DOCUMENT, so that a well-meaning edit ("fix" the accent,
    shorten the sentence) on either side fails CI instead of silently changing
    the instrument mid-experiment;
  * `RENDERER_SHA256` is asserted against a pinned value, the way the schema
    hash is pinned in 1.2, so that the frozen artefact cannot drift without the
    freeze being renegotiated.

The golden test reads the four REAL stored pilot records. It is read-only by
construction and must stay so: `scripts/validation/pilot_records/...` is a
permanent audit record (5.8.9) and nothing in the test suite may write into it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.validation.stage_c_render_slots import (
    RENDERER_SHA256,
    TEMPLATES,
    render_slot_claims,
    slot_claim_count,
)

# Pilot artefacts live beside the code they were produced by, so the path is
# derived from this file rather than from the process working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOT_DIR = (
    _REPO_ROOT
    / "scripts"
    / "validation"
    / "pilot_records"
    / "20260921_axis3_structured_contract"
)

# The pre-registration is the other half of the freeze: 2.2 is where the four
# templates are declared, and this file exists to prove the module has not
# drifted from it.
_PREREG = (
    _REPO_ROOT / "scripts" / "validation" / "axis3_structured_output_preregistration.md"
)

# A template bullet is a whole-line backtick-quoted list item carrying at least
# one {placeholder}. The placeholder test is what separates the four frozen
# format strings from any other backticked bullet section 2.2 may grow, and it
# is why the COUNT is asserted separately below rather than trusted.
_TEMPLATE_BULLET = re.compile(r"^- `(.+)`$")


def _section_2_2_templates() -> list[str]:
    """The template bullets of section 2.2, read out of the document itself."""
    lines = _PREREG.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("### 2.2")]
    assert len(starts) == 1, f"expected exactly one '### 2.2' heading, got {starts}"
    start = starts[0]
    end = next((i for i, line in enumerate(lines[start + 1:], start + 1)
                if line.startswith("### ")), len(lines))
    found = []
    for line in lines[start + 1:end]:
        match = _TEMPLATE_BULLET.match(line.strip())
        if match and "{" in match.group(1) and "}" in match.group(1):
            found.append(match.group(1))
    return found


def _pilot_object(preset: str, case_id: str) -> dict:
    """Return the parsed final-answer object of one stored pilot record.

    Selected by `case_id` rather than by line number, so a reordering of the
    stored file could never silently swap A03 for B02 under a test that then
    still passed.
    """
    path = _PILOT_DIR / preset / "agent_eval_raw.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["case_id"] == case_id:
                return json.loads(row["reply"])
    raise AssertionError(f"no {case_id} row in {path}")


def _ok(initial, transitions, final):
    """Build a minimal status-ok object; the renderer reads nothing else."""
    return {
        "status": "ok",
        "initial_state": initial,
        "critical_transitions": transitions,
        "final_state": final,
        "interpretation": "",
        "limitations": "",
    }


# --- the frozen text itself ------------------------------------------------


def test_templates_are_verbatim_from_section_2_2():
    """The module's templates ARE the document's templates.

    This test used to transcribe the four Greek sentences into this file, which
    proved only that the module and the test agreed with each other: one edit
    applied to both, or a document that had silently moved on, passed it just as
    happily. It now parses section 2.2 and compares against what is written
    there, so an edit to EITHER side turns CI red and the renderer is pinned to
    the freeze rather than to a copy of it.
    """
    found = _section_2_2_templates()
    # Four, exactly: 2.2 declares four templates and a fifth would be a fifth
    # judged claim family that no part of the rubric knows about. A count that
    # drifted would otherwise hide inside a set comparison that still passed.
    assert len(found) == 4, f"section 2.2 yielded {len(found)} template bullets: {found}"
    assert set(found) == set(TEMPLATES.values())


def test_template_set_is_exactly_four_entries():
    # 2.2 lists four templates. A fifth would mean a fifth judged claim family
    # that no part of the rubric or the aggregation knows about.
    assert set(TEMPLATES) == {
        "initial_state",
        "critical_transitions",
        "final_state",
        "status",
    }


def test_renderer_hash_is_pinned_and_recomputable():
    import hashlib
    import json as _json

    assert RENDERER_SHA256 == (
        "40abd5a8e730bdadfa966ab3fbd2cfd0cb94f9eff80dc22c2cb3801af1cfd27e"
    )
    recomputed = hashlib.sha256(
        _json.dumps(TEMPLATES, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert recomputed == RENDERER_SHA256
    assert len(RENDERER_SHA256) == 64


def test_templates_never_number_hour_zero_as_the_first_hour():
    # Tie-break R7: mapping h0 to "1η ώρα" is CONTRADICTED when a model does it,
    # so the instrument must not be able to do it either.
    for template in TEMPLATES.values():
        assert "1η ώρα" not in template


# --- ordering and counting -------------------------------------------------


def test_claims_follow_schema_order():
    claims = render_slot_claims(
        _ok(
            [{"period": 0, "settlements_without_route": 6}],
            [
                {"period": 1, "settlements_without_route": 9},
                {"period": 3, "settlements_without_route": 10},
            ],
            [
                {
                    "period": 4,
                    "settlements_without_route": 14,
                    "road_segments_removed": 361,
                }
            ],
        )
    )
    assert claims == [
        "Στην έναρξη (ώρα 0): 6 οικισμοί χωρίς καμία διαδρομή διαφυγής.",
        "Στην ώρα 1: 9 οικισμοί χωρίς καμία διαδρομή διαφυγής.",
        "Στην ώρα 3: 10 οικισμοί χωρίς καμία διαδρομή διαφυγής.",
        "Τελική ώρα 4: 14 οικισμοί χωρίς καμία διαδρομή διαφυγής, "
        "361 οδικά τμήματα αφαιρέθηκαν.",
    ]


def test_emitted_order_is_preserved_even_when_it_is_wrong():
    # A reordered array is a contract failure that `ct_order_ok` records; the
    # renderer must show the judge what the model actually emitted, not a
    # tidied-up version of it.
    claims = render_slot_claims(
        _ok(
            [{"period": 0, "settlements_without_route": 6}],
            [
                {"period": 3, "settlements_without_route": 10},
                {"period": 1, "settlements_without_route": 9},
            ],
            [
                {
                    "period": 4,
                    "settlements_without_route": 14,
                    "road_segments_removed": 361,
                }
            ],
        )
    )
    assert claims[1].startswith("Στην ώρα 3:")
    assert claims[2].startswith("Στην ώρα 1:")


def test_slot_claim_count_is_critical_events_plus_two():
    obj = _ok(
        [{"period": 0, "settlements_without_route": 6}],
        [{"period": p, "settlements_without_route": 9} for p in (1, 3, 4)],
        [{"period": 4, "settlements_without_route": 14, "road_segments_removed": 361}],
    )
    assert slot_claim_count(obj) == 5
    assert slot_claim_count(obj) == len(render_slot_claims(obj))


def test_b02_final_hour_that_is_also_a_transition_yields_two_distinct_sentences():
    # 2.2, the B02 case: period 4 is both the final hour and a critical
    # transition. The two sentences must not be whitespace/case-identical, or
    # the deduplication rule of 2.3 would merge them and the record would lose
    # a judged claim.
    claims = render_slot_claims(
        _ok(
            [{"period": 0, "settlements_without_route": 6}],
            [{"period": 4, "settlements_without_route": 14}],
            [
                {
                    "period": 4,
                    "settlements_without_route": 14,
                    "road_segments_removed": 361,
                }
            ],
        )
    )
    about_hour_four = [c for c in claims if " 4" in c]
    assert len(about_hour_four) == 2
    assert about_hour_four[0] != about_hour_four[1]
    assert len(set(claims)) == len(claims)


# --- the non-ok status path ------------------------------------------------


@pytest.mark.parametrize(
    "status", ["tool_error", "insufficient_input", "abstained"]
)
def test_non_ok_status_renders_exactly_one_sentence(status):
    obj = _ok([], [], [])
    obj["status"] = status
    assert render_slot_claims(obj) == [
        f"Το εργαλείο δεν παρήγαγε αποτέλεσμα: {status}."
    ]
    assert slot_claim_count(obj) == 1


def test_non_ok_status_ignores_arrays_the_contract_forbids():
    # Semantic rule 4 of 1.3: a non-ok status implies three empty arrays. When a
    # model fills them anyway the record is validator_invalid and the arrays are
    # debris, so they must not reach the judge as claims.
    obj = _ok(
        [{"period": 0, "settlements_without_route": 6}],
        [{"period": 1, "settlements_without_route": 9}],
        [{"period": 4, "settlements_without_route": 14, "road_segments_removed": 361}],
    )
    obj["status"] = "tool_error"
    assert render_slot_claims(obj) == [
        "Το εργαλείο δεν παρήγαγε αποτέλεσμα: tool_error."
    ]


def test_unknown_status_value_still_renders():
    # An out-of-enum status is schema_invalid (outcome 6 of 2.4) and such a
    # record keeps a judgeable surface, so refusing to render it would deny it
    # the scoring 2.4 says it must receive.
    obj = _ok([], [], [])
    obj["status"] = "ΟΧΙ_ΣΤΟ_ENUM"
    assert render_slot_claims(obj) == [
        "Το εργαλείο δεν παρήγαγε αποτέλεσμα: ΟΧΙ_ΣΤΟ_ENUM."
    ]


# --- defensive but never silent --------------------------------------------


def test_missing_status_raises():
    obj = _ok([], [], [])
    del obj["status"]
    with pytest.raises(ValueError, match="status"):
        render_slot_claims(obj)


def test_non_string_status_raises():
    obj = _ok([], [], [])
    obj["status"] = 0
    with pytest.raises(ValueError, match="not a string"):
        render_slot_claims(obj)


def test_missing_array_raises_naming_the_key():
    obj = _ok([{"period": 0, "settlements_without_route": 6}], [], [])
    del obj["final_state"]
    with pytest.raises(ValueError, match="final_state"):
        render_slot_claims(obj)


def test_array_that_is_not_a_list_raises():
    obj = _ok([{"period": 0, "settlements_without_route": 6}], {}, [])
    with pytest.raises(ValueError, match="critical_transitions"):
        render_slot_claims(obj)


def test_missing_field_in_an_entry_raises_naming_the_entry():
    obj = _ok(
        [{"period": 0, "settlements_without_route": 6}],
        [{"period": 1}],
        [{"period": 4, "settlements_without_route": 14, "road_segments_removed": 361}],
    )
    with pytest.raises(ValueError, match=r"critical_transitions\[0\]"):
        render_slot_claims(obj)


def test_missing_road_segments_removed_raises_rather_than_rendering_none():
    obj = _ok(
        [{"period": 0, "settlements_without_route": 6}],
        [],
        [{"period": 4, "settlements_without_route": 14}],
    )
    with pytest.raises(ValueError, match="road_segments_removed"):
        render_slot_claims(obj)


@pytest.mark.parametrize("bad", ["3", 3.0, None, True, [3]])
def test_non_integer_value_raises(bad):
    # `True` is in this list on purpose: bool is an int subclass in Python, so
    # without the explicit rejection a JSON `true` would render as "True".
    obj = _ok(
        [{"period": 0, "settlements_without_route": bad}],
        [],
        [{"period": 4, "settlements_without_route": 14, "road_segments_removed": 361}],
    )
    with pytest.raises(ValueError, match="not an integer"):
        render_slot_claims(obj)


def test_entry_that_is_not_an_object_raises():
    obj = _ok(
        [{"period": 0, "settlements_without_route": 6}],
        ["ώρα 1"],
        [{"period": 4, "settlements_without_route": 14, "road_segments_removed": 361}],
    )
    with pytest.raises(ValueError, match=r"critical_transitions\[0\]"):
        render_slot_claims(obj)


@pytest.mark.parametrize("bad", [None, [], "{}", 7])
def test_non_object_input_raises(bad):
    with pytest.raises(ValueError, match="parsed JSON object"):
        render_slot_claims(bad)


def test_slot_claim_count_raises_on_a_malformed_record():
    # The count must never be reportable for a record the judge could not be
    # shown; otherwise 3.4.3 would publish a count for an unrendered record.
    obj = _ok([{"period": 0}], [], [])
    with pytest.raises(ValueError):
        slot_claim_count(obj)


# --- golden test on the stored pilot records -------------------------------


def test_golden_luna_a03_renders_fifteen_claims():
    # luna over-emitted `critical_transitions` (13 entries against 8 GT events,
    # 3.4.5), so its slot tier is 13 + 2 = 15 claims.
    obj = _pilot_object("luna", "A03")
    claims = render_slot_claims(obj)
    assert len(claims) == 15
    assert slot_claim_count(obj) == 15
    assert claims[0] == "Στην έναρξη (ώρα 0): 6 οικισμοί χωρίς καμία διαδρομή διαφυγής."
    assert claims[1] == "Στην ώρα 1: 9 οικισμοί χωρίς καμία διαδρομή διαφυγής."
    assert claims[-1] == (
        "Τελική ώρα 24: 23 οικισμοί χωρίς καμία διαδρομή διαφυγής, "
        "737 οδικά τμήματα αφαιρέθηκαν."
    )


def test_golden_terra_a03_renders_ten_claims():
    # terra reproduced `cutoff_events()` exactly on A03: 8 entries + 2 = 10.
    obj = _pilot_object("terra", "A03")
    claims = render_slot_claims(obj)
    assert len(claims) == 10
    assert slot_claim_count(obj) == 10
    assert claims[0] == "Στην έναρξη (ώρα 0): 6 οικισμοί χωρίς καμία διαδρομή διαφυγής."
    assert claims[-1] == (
        "Τελική ώρα 24: 23 οικισμοί χωρίς καμία διαδρομή διαφυγής, "
        "737 οδικά τμήματα αφαιρέθηκαν."
    )


def test_golden_b02_records_carry_two_sentences_about_hour_four():
    # The real B02 records are the case 2.2 names: the final hour (4) is also a
    # critical transition, on both presets.
    for preset, expected_count in (("luna", 6), ("terra", 5)):
        claims = render_slot_claims(_pilot_object(preset, "B02"))
        assert len(claims) == expected_count
        assert "Στην ώρα 4: 14 οικισμοί χωρίς καμία διαδρομή διαφυγής." in claims
        assert (
            "Τελική ώρα 4: 14 οικισμοί χωρίς καμία διαδρομή διαφυγής, "
            "361 οδικά τμήματα αφαιρέθηκαν." in claims
        )
        assert len(set(claims)) == len(claims)


def test_golden_no_rendered_claim_leaks_a_python_repr():
    # A "None", a "True" or a "[" in a judged sentence would be a renderer
    # defect charged to the model as a CONTRADICTED claim.
    for preset in ("luna", "terra"):
        for case_id in ("A03", "B02"):
            for claim in render_slot_claims(_pilot_object(preset, case_id)):
                assert "None" not in claim
                assert "True" not in claim
                assert "[" not in claim
                assert claim.endswith(".")

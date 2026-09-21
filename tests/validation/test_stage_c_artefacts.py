"""Guards for the two frozen judging artefacts and their provenance hashes.

`scripts/validation/stage_c_rubric_v2.md` and
`scripts/validation/stage_c_judge_prompt.txt` are not ordinary documents. They
are the instrument: the rubric is the boundary every Stage-C label is drawn
against, and the judge prompt fixes what the judge is shown and in what order.
Section 2 of `axis3_structured_output_preregistration.md` freezes both before
any judging and records their sha256 beside every stored label set, precisely so
that a later edit cannot quietly move the boundary under labels that were
already drawn.

These tests are therefore mostly DRIFT GUARDS rather than behaviour checks, and
they are built so that an edit to either side turns CI red:

  * the worked examples are parsed out of the rubric AND out of the
    pre-registration and compared, so neither copy can be "improved" alone;
  * the label vocabulary, the four tests and the eight tie-breaks are asserted
    against the pre-registration first and the rubric second, so the test cannot
    drift from the document either;
  * every must-see and must-not-see item of the judge protocol is asserted to
    occur in section 2.9 of the pre-registration before it is asserted to occur
    (or in the must-not-see case, to be named as withheld) in the prompt;
  * the three hashes are pinned as literals, so re-freezing the experiment is a
    deliberate act that edits this file, never a side effect of a typo fix.

Nothing here writes to a real artefact. The hash-sensitivity test works on
copies under pytest's `tmp_path` for that reason.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.validation.stage_c_hashes import (
    JUDGE_PROMPT_PATH,
    JUDGE_PROMPT_SHA256,
    RENDERER_SHA256,
    RUBRIC_PATH,
    RUBRIC_SHA256,
    file_sha256,
    frozen_artefact_hashes,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PREREG = (
    _REPO_ROOT / "scripts" / "validation" / "axis3_structured_output_preregistration.md"
)


# --- helpers: read a numbered subsection out of either document ---------------


def _section_body(text: str, heading_prefix: str) -> str:
    """The body of the one heading that starts with `heading_prefix`.

    Works on both documents because the rubric promotes the pre-registration's
    `###` headings to `##` and changes nothing else, so the caller passes the
    prefix the document in question actually uses.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(heading_prefix)]
    assert len(starts) == 1, f"expected one {heading_prefix!r} heading, got {starts}"
    start = starts[0]
    level = heading_prefix.split(" ", 1)[0] + " "
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1:], start + 1)
            if line.startswith(level) or line.startswith("## ")
        ),
        len(lines),
    )
    return "\n".join(lines[start + 1:end])


@pytest.fixture(scope="module")
def prereg_text() -> str:
    return _PREREG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rubric_text() -> str:
    return RUBRIC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prompt_text() -> str:
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


# --- the files exist at all ---------------------------------------------------


def test_both_frozen_artefacts_exist_and_are_non_empty():
    for path in (RUBRIC_PATH, JUDGE_PROMPT_PATH):
        assert path.is_file(), f"frozen artefact missing: {path}"
        # A truncated artefact would still hash, and the hash would still be
        # recorded in the provenance file, so emptiness has to be caught here
        # rather than trusted to the freeze.
        assert path.stat().st_size > 0, f"frozen artefact is empty: {path}"
        assert path.read_text(encoding="utf-8").strip(), f"blank artefact: {path}"


# --- 2.5, the label vocabulary ------------------------------------------------

# The five of 2.5. Adding a sixth would make the 2026-09-01 run's 2,409 pass-1
# and 2,320 pass-2 stored claims unusable, which is why the vocabulary is frozen
# and why this list is asserted against the document before it is used.
_LABELS = (
    "SUPPORTED",
    "CONTRADICTED",
    "NOT_IN_RESULT",
    "ADVISORY_INFERENCE",
    "LIMITATION_STATEMENT",
)


def test_label_vocabulary_is_the_five_of_2_5(prereg_text, rubric_text):
    source = _section_body(prereg_text, "### 2.5 ")
    for label in _LABELS:
        assert label in source, f"{label} is not in section 2.5 of the pre-registration"
    for label in _LABELS:
        assert label in rubric_text, f"{label} is not named in the rubric"


def test_rubric_names_no_sixth_label(rubric_text):
    """No ALL-CAPS token outside the five looks like a label.

    A rubric that mentioned, say, PARTIALLY_SUPPORTED in an aside would invite a
    judge to use it, and the deterministic post-call check would then re-queue
    and escalate every record that did. Underscored all-caps tokens are the
    shape a label has, so that is the shape this test polices.
    """
    candidates = set(re.findall(r"\b[A-Z][A-Z_]{4,}\b", rubric_text))
    allowed = set(_LABELS) | {
        # Vocabulary of the surrounding machinery, not labels: the contract
        # outcomes and the gate words of 2.4, and the emphasis words the
        # pre-registration uses in its own prose.
        "NEGATIVE_LABELS",
        "NARRATION_MIN_CHARS",
        "FAILS",
        "EXCLUDED",
        "NOT",
        "ASSUMPTION",
        "ESTIMATE",
        "STAGE",
        "RUBRIC",
        "INPUT",
        "PIPELINE",
        "PROVISIONAL",
        "REPLACES",
        "ΚΕΝΟΙ",
        "ΟΙΚΙΣΜΟΙ",
        "ΟΧΙ",
        "ΑΚΡΙΒΩΣ",
        "ΜΕΓΑΛΥΤΕΡΟ",
    }
    unexpected = {c for c in candidates - allowed if "_" in c}
    assert not unexpected, f"label-shaped tokens the rubric should not carry: {unexpected}"


# --- 2.6 and 2.7, the tests and the tie-breaks --------------------------------


@pytest.mark.parametrize("test_id", ["T1", "T2", "T3", "T4"])
def test_rubric_carries_all_four_boundary_tests(prereg_text, rubric_text, test_id):
    marker = f"**{test_id} "
    assert marker in _section_body(prereg_text, "### 2.6 ")
    assert marker in _section_body(rubric_text, "## 2.6 ")


@pytest.mark.parametrize(
    "rule_id", ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
)
def test_rubric_carries_all_eight_tie_breaks(prereg_text, rubric_text, rule_id):
    marker = f"**{rule_id} "
    assert marker in _section_body(prereg_text, "### 2.7 ")
    assert marker in _section_body(rubric_text, "## 2.7 ")


def test_precedence_is_stated_not_merely_implied(rubric_text):
    """T1 outranking T3 is the one precedence a judge gets wrong.

    2.6 states it inside the T1 bullet ("T1 outranks T3"), and FAIL 5 is the
    worked example of it. If either disappeared, a hedged invented subject would
    start collecting ADVISORY_INFERENCE and the intervention's clean rate would
    move for a reason that has nothing to do with the intervention.
    """
    assert "T1 outranks T3" in rubric_text
    assert "strict precedence" in rubric_text


# --- 2.8, the worked examples, verbatim ---------------------------------------

# A worked example is a whole line opening with its family and number. The
# pre-registration writes them as bare paragraph lines and the rubric lifts them
# unchanged, so the same pattern reads both.
_EXAMPLE_LINE = re.compile(r"^(FAIL|PASS|BOUNDARY) ")


def _worked_examples(text: str, heading_prefix: str) -> list[str]:
    return [
        line
        for line in _section_body(text, heading_prefix).splitlines()
        if _EXAMPLE_LINE.match(line)
    ]


def test_worked_examples_are_verbatim_from_the_pre_registration(
    prereg_text, rubric_text
):
    """The rubric's examples ARE the document's examples, character for character.

    Both sides are parsed rather than transcribed into this file. A test that
    held its own copy of the fourteen Greek lines would prove only that the test
    and the rubric agreed with each other, and one edit applied to both would
    pass it.
    """
    source = _worked_examples(prereg_text, "### 2.8 ")
    lifted = _worked_examples(rubric_text, "## 2.8 ")
    # 2.8 declares its own census: six failing, six passing, two boundary pairs.
    assert len(source) == 14, f"section 2.8 yielded {len(source)} examples"
    assert lifted == source


@pytest.mark.parametrize(
    "family,count", [("FAIL", 6), ("PASS", 6), ("BOUNDARY", 2)]
)
def test_worked_example_census_matches_2_8(rubric_text, family, count):
    lifted = _worked_examples(rubric_text, "## 2.8 ")
    got = [line for line in lifted if line.startswith(family + " ")]
    assert len(got) == count, f"{family}: expected {count}, got {len(got)}"


def test_the_two_boundary_pairs_keep_their_deciding_rule(rubric_text):
    """BOUNDARY A and B exist to fix R1 and R3 against real disagreements.

    They were the pass-1/pass-2 flips of the 2026-09-01 run, and R3 in
    particular exists because one phrasing distinction flipped a scored record
    between passes. An example that lost its "decided by" tag would still read
    like an example and would no longer pin the rule.
    """
    lifted = "\n".join(_worked_examples(rubric_text, "## 2.8 "))
    assert "BOUNDARY A, decided by R1." in lifted
    assert "BOUNDARY B, decided by R3." in lifted


def test_greek_example_text_survived_the_lift(rubric_text):
    """Spot-check the Greek, which is the half a re-encoding would damage.

    A mojibake'd rubric would still satisfy the line-by-line comparison above if
    the pre-registration were damaged the same way, so one example is checked
    against a literal here as an independent anchor.
    """
    assert "3η ώρα: Η φωτιά φτάνει στον πρώτο οικισμό." in rubric_text
    assert "Ο πρώτος οικισμός περνά σε περίμετρο στη 2η ώρα." in rubric_text


# --- 2.9, what the judge sees and never sees ----------------------------------

# Every token below is asserted to occur in section 2.9 of the pre-registration
# BEFORE it is asserted against the prompt, so this list cannot drift away from
# the protocol it claims to enforce.
_MUST_SEE = (
    "rubric sections 2.5 to 2.8",
    "user_text",
    "pins",
    "ensure_ascii=False",
    "field_meanings",
    "tool_numbers_md",
    "canonical claim list",
    "judged-surface text",
)

_MUST_NOT_SEE = (
    "the arm",
    "the model under test",
    "the case gold",
    "cutoff_events",
    "the other pass",
    "any prior human decision",
    "any pass rate",
    "the raw JSON envelope",
)


@pytest.mark.parametrize("item", _MUST_SEE)
def test_prompt_carries_every_must_see_item(prereg_text, prompt_text, item):
    assert item in _section_body(prereg_text, "### 2.9 "), (
        f"{item!r} is not in section 2.9; this test has drifted from the protocol"
    )
    assert item in prompt_text, f"the judge prompt never mentions {item!r}"


@pytest.mark.parametrize("item", _MUST_NOT_SEE)
def test_prompt_names_every_must_not_see_item_as_withheld(
    prereg_text, prompt_text, item
):
    """The withheld list is printed IN the prompt, and that is intentional.

    Naming a withheld input does not supply it, and an unnamed prohibition is
    one the judge can violate by reconstruction without ever realising it was a
    prohibition. The pre-registration's blinding claim is explicitly partial, so
    the prompt states the boundary rather than relying on absence alone.
    """
    assert item in _section_body(prereg_text, "### 2.9 "), (
        f"{item!r} is not in section 2.9; this test has drifted from the protocol"
    )
    withheld = prompt_text.split("WHAT THIS PROMPT WITHHOLDS", 1)
    assert len(withheld) == 2, "the judge prompt has no withholding section"
    assert item in withheld[1], f"the withholding section never names {item!r}"


def test_must_see_items_appear_in_the_order_2_9_requires(prompt_text):
    """Order is part of the protocol, not presentation.

    2.9 fixes it as: rubric, then the request, then the payload, then the
    payload's numbers, then the claim list, then the judged surface. A judge
    that met the claims before the payload would be anchoring on the claims.
    """
    positions = [prompt_text.index(item) for item in _MUST_SEE]
    assert positions == sorted(positions), dict(zip(_MUST_SEE, positions))


def test_prompt_states_the_required_output_shape(prereg_text, prompt_text):
    shape = "{uid, claims:[{claim_id, text, label, justification}], notes}"
    assert shape in _section_body(prereg_text, "### 2.9 ")
    assert shape in prompt_text


def test_prompt_states_the_justification_rule(prompt_text):
    # One sentence, naming a payload field or the literal escape hatch. Without
    # the escape hatch a judge has to invent a field name for a NOT_IN_RESULT
    # whose subject does not exist at all, which is the one case where naming a
    # field would be a fabrication.
    assert "ONE sentence" in prompt_text
    assert "at least one payload field" in prompt_text
    assert '"no field"' in prompt_text


def test_prompt_states_what_happens_to_an_invalid_record(prompt_text):
    # 2.9: re-queued once, then escalated to a human. A prompt that omitted the
    # consequence would leave a guessed label looking cheaper than an honest note.
    assert "judge_output_invalid" in prompt_text
    assert "re-queued once" in prompt_text


def test_prompt_has_no_unfilled_placeholder_left_unexplained(prompt_text):
    """Every {{TOKEN}} is a harness slot, and the prompt says a filled one has none."""
    tokens = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", prompt_text))
    assert tokens, "the prompt carries no harness placeholders at all"
    assert "A filled prompt must" in prompt_text
    for required in (
        "RUBRIC_2_5_TO_2_8",
        "USER_TEXT",
        "PINS",
        "COMPACT_PAYLOAD_JSON",
        "TOOL_NUMBERS_MD",
        "CANONICAL_CLAIM_LIST",
        "JUDGED_SURFACE_TEXT",
        "UID",
    ):
        assert required in tokens, f"the prompt has no {{{{{required}}}}} slot"


# --- the hashes ---------------------------------------------------------------


def test_hashes_are_stable_across_two_calls():
    assert file_sha256(RUBRIC_PATH) == file_sha256(RUBRIC_PATH)
    assert file_sha256(JUDGE_PROMPT_PATH) == file_sha256(JUDGE_PROMPT_PATH)
    assert frozen_artefact_hashes() == frozen_artefact_hashes()


def test_hash_of_a_copy_equals_hash_of_the_original(tmp_path):
    copy = tmp_path / "rubric_copy.md"
    copy.write_text(RUBRIC_PATH.read_text(encoding="utf-8"), encoding="utf-8",
                    newline="\n")
    assert file_sha256(copy) == RUBRIC_SHA256


def test_hash_changes_when_the_content_changes(tmp_path):
    """A one-character edit must move the digest. Copies only, never the real files."""
    copy = tmp_path / "prompt_copy.txt"
    original = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    copy.write_text(original, encoding="utf-8", newline="\n")
    before = file_sha256(copy)
    assert before == JUDGE_PROMPT_SHA256

    copy.write_text(original + "x", encoding="utf-8", newline="\n")
    assert file_sha256(copy) != before


def test_line_ending_normalisation_is_deliberate(tmp_path):
    """CRLF and LF of the same content hash the same, and that is by design.

    `core.autocrlf` is true in this repository, so the committed rubric is
    checked out CRLF on Windows and LF in a container. A raw-byte hash would
    report the frozen instrument as two different instruments on two machines
    and the check would be trained out of existence within a week.
    """
    text = "line one\nline two\n"
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(text.encode("utf-8"))
    crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    assert file_sha256(lf) == file_sha256(crlf)


def test_frozen_artefact_hashes_carries_all_five_fingerprints():
    hashes = frozen_artefact_hashes()
    assert set(hashes) == {
        "rubric_sha256",
        "judge_prompt_sha256",
        "renderer_sha256",
        "final_answer_schema_sha256",
        "system_prompt_sha256",
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()), hashes
    # The dict must agree with the module constants and with the renderer's own
    # definition; a provenance file that disagreed with the module that wrote it
    # would be unrecoverable after the fact.
    assert hashes["rubric_sha256"] == RUBRIC_SHA256
    assert hashes["judge_prompt_sha256"] == JUDGE_PROMPT_SHA256
    assert hashes["renderer_sha256"] == RENDERER_SHA256


def test_frozen_artefact_hashes_are_pinned():
    """THE FREEZE. Recorded 2026-09-21, before any structured judging pass.

    If this test fails, a frozen judging artefact was edited. That is allowed
    only as a deliberate re-freeze: labels produced under the old hashes are not
    poolable with labels produced under the new ones, so the correct response is
    to decide whether the experiment is being re-frozen and, if it is, to update
    these literals in the same commit as the edit and to say so in the
    pre-registration. It is never to "fix the test".
    """
    assert frozen_artefact_hashes() == {
        "rubric_sha256": (
            "89a25576d44f104085c31eed2f95297f98a2fa500c360dbda583dd58863cac1e"
        ),
        "judge_prompt_sha256": (
            "a83c6f271dc40a3237a2779557e6bec0143cca1da2390856d7ebab1b8d1de206"
        ),
        # Pinned independently in tests/validation/test_stage_c_render_slots.py
        # as well; the two must always agree.
        "renderer_sha256": (
            "40abd5a8e730bdadfa966ab3fbd2cfd0cb94f9eff80dc22c2cb3801af1cfd27e"
        ),
        "final_answer_schema_sha256": (
            "d079a01b22d4c344a4f2b7a6a62b147215860deeeddd373e5b597f8ed498ccf0"
        ),
        # The evidence for the pre-registration's claim that the system prompt
        # was NOT forked between arms: it must stay f005ab5e76d6 in both.
        "system_prompt_sha256": (
            "f005ab5e76d6b57b5fa12cc02d826f6a83b97d4aa67380e4c320d47e43014e64"
        ),
    }

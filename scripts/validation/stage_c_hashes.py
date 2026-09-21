"""The provenance fingerprints of the Stage-C judging instrument.

Section 2 of `scripts/validation/axis3_structured_output_preregistration.md`
freezes three artefacts before any judging: the rubric
(`stage_c_rubric_v2.md`), the judge prompt (`stage_c_judge_prompt.txt`) and the
slot renderer's template set (`stage_c_render_slots.py`). It then requires all
three sha256 values to be recorded in
`stage_c_full/stage_c_judge_provenance.json` for every judging pass. This module
is the one place that computes them, so the provenance writer, the tests and a
reader checking a stored label set all read the same numbers from the same rule.

WHY THIS MATTERS RATHER THAN BEING BOOKKEEPING. A label set is only interpretable
if the instrument that produced it is recoverable. Change one character of a
template, one tie-break, one line of the output contract, and the labels before
and after the change were produced by two different instruments: they can each be
correct and still not be poolable, because the boundary they were drawn against
moved. The hashes are what makes that detectable after the fact instead of
arguable. They are also what a future reader uses to prove the rubric was frozen
BEFORE the run rather than tuned to it.

WHAT IS HASHED, AND HOW.

* The two text artefacts are hashed over their content with line endings
  normalised to "\\n" and re-encoded UTF-8, NOT over the raw bytes on disk. The
  repository is developed on Windows with `core.autocrlf=true`, so the same
  committed file is checked out with CRLF here and LF in a Linux container; a
  raw-byte hash would report the frozen rubric as two different instruments on
  two machines, which is exactly the false alarm that teaches people to ignore
  the check. A trailing-newline difference still changes the hash, as it should:
  that is a real content difference.
* The renderer is NOT hashed as a file. `RENDERER_SHA256` is taken over the
  canonical JSON of the template mapping, which is the instrument (the four
  Greek sentences), while the surrounding docstrings and helper code are not.
  Re-exported here rather than recomputed, so there is one definition.
* The schema and the system prompt are imported from the modules that own them
  for the same single-source reason. Together with the three above they answer
  the whole provenance question for a structured run: which contract the model
  answered under, which prompt it answered, which sentences its slots were
  rendered into, and which rubric and prompt the judge applied.

The imports of the last two are LAZY. `agent.py` is the runtime agent and pulls
`llm_config` and the provider layer behind it; a reader who only wants the
judging hashes, and a test collector, should not pay for that at import time.

Run:
    python scripts/validation/stage_c_hashes.py           # the five hashes
    python scripts/validation/stage_c_hashes.py --json    # the provenance dict
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Both spellings, for the same reason `stage_c_structured.py` needs both: this
# file is imported as `scripts.validation.stage_c_hashes` by the test suite and
# run as a plain script by the operator. The relative form is tried first so a
# package import cannot end up holding a second copy of the renderer module and
# therefore a second copy of the template mapping.
try:
    from .stage_c_render_slots import RENDERER_SHA256
except ImportError:  # run as a script: no parent package, but _HERE is on sys.path
    from stage_c_render_slots import RENDERER_SHA256

# The two frozen text artefacts. Resolved from this file rather than from the
# process working directory, because the provenance writer, pytest and a manual
# run all start somewhere different and must hash the same two files.
RUBRIC_PATH = _HERE / "stage_c_rubric_v2.md"
JUDGE_PROMPT_PATH = _HERE / "stage_c_judge_prompt.txt"

# What the pre-registration prints beside a label set. The full digest is stored;
# the prefix is what a human quotes in a caption or a decision log.
PREFIX_LEN = 12


def file_sha256(path: str | Path) -> str:
    """sha256 of a text artefact, line endings normalised, UTF-8.

    Normalisation is deliberate and is documented in the module docstring: a
    CRLF/LF difference is a checkout artefact of this repository's own git
    configuration, not a change to the instrument, and a hash that flagged it
    would be noise that trains the reader to ignore a real alarm.
    """
    text = Path(path).read_text(encoding="utf-8")
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


RUBRIC_SHA256 = file_sha256(RUBRIC_PATH)
JUDGE_PROMPT_SHA256 = file_sha256(JUDGE_PROMPT_PATH)

_contract_hashes_cache: tuple[str, str] | None = None


def _contract_hashes() -> tuple[str, str]:
    """(final-answer schema sha256, system prompt sha256), imported and cached.

    Imported from the modules that define them, never recomputed here: a second
    implementation of "the schema hash" is a second answer waiting to disagree
    with the first one in a provenance file, which is the one place a
    disagreement is unrecoverable.
    """
    global _contract_hashes_cache
    if _contract_hashes_cache is None:
        agent_dir = _HERE.parent / "agent"
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        from final_answer_schema import FINAL_ANSWER_SCHEMA_SHA256  # noqa: E402
        import agent  # noqa: E402

        _contract_hashes_cache = (
            FINAL_ANSWER_SCHEMA_SHA256,
            agent.SYSTEM_PROMPT_SHA256,
        )
    return _contract_hashes_cache


def frozen_artefact_hashes() -> dict[str, str]:
    """The provenance block written into `stage_c_judge_provenance.json`.

    Full digests, not prefixes: the prefix is for a caption, and the file is for
    proving which instrument produced a stored label set years later.

    The three judging artefacts come first because they are what 2.9 freezes;
    the schema and the system prompt follow because a structured record cannot
    be interpreted without knowing which contract produced it, and because the
    system prompt hash is the evidence for the pre-registration's claim that the
    prompt was NOT forked between arms.
    """
    schema_sha256, system_prompt_sha256 = _contract_hashes()
    return {
        "rubric_sha256": RUBRIC_SHA256,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
        "renderer_sha256": RENDERER_SHA256,
        "final_answer_schema_sha256": schema_sha256,
        "system_prompt_sha256": system_prompt_sha256,
    }


def _main(argv: list[str]) -> int:
    # Same reason as the renderer's own main: the artefacts this module pins are
    # Greek, and the paths are printed beside the hashes, so a cp1252 console
    # must not be able to kill the one command a reader runs to check a freeze.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    hashes = frozen_artefact_hashes()
    if "--json" in argv:
        print(json.dumps(hashes, indent=2, sort_keys=True))
        return 0

    sources = {
        "rubric_sha256": str(RUBRIC_PATH),
        "judge_prompt_sha256": str(JUDGE_PROMPT_PATH),
        "renderer_sha256": "stage_c_render_slots.TEMPLATES (canonical JSON)",
        "final_answer_schema_sha256": "final_answer_schema.FINAL_ANSWER_SCHEMA",
        "system_prompt_sha256": "agent.SYSTEM_PROMPT",
    }
    width = max(len(name) for name in hashes)
    for name, digest in hashes.items():
        print(f"{name:<{width}}  {digest}")
        print(f"{'':<{width}}  prefix {digest[:PREFIX_LEN]}  <- {sources[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

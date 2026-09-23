"""Synthetic negative control for the Stage-C v3 disclosure gate.

TECHNICAL CONTROL ONLY. The record this builds is not a model output, was not
produced by any run, and must never enter a published result, a denominator or a
comparison. It exists to answer one question the pilot could not: the judge
recognises a disclosure that is present, but does it notice one that is absent?

Method. Take one pilot record whose case requires the forecast-weather disclosure
and which passed all six questions. Make two copies:

    NC002   the record unchanged, only its id rewritten
    NC001   the same record with the ONE sentence that carries the forecast
            disclosure deleted, and nothing else touched

The payload, the gold, the operator's request, the pins, the critical events, the
advice and all three limitation sentences are byte-identical between the two. The
single variable is the disclosure. `required_disclosures_complete` should be true
on NC002 and false on NC001, and NC001 should fail Stage C on that alone.

Nothing frozen is modified: not the rubric, not the schema, not the gold, not the
cases, not a threshold. This module only assembles text and verifies the diff.

    python scripts/validation/stage_c_v3_negative_control.py \\
        --source RECORD.txt --out DIR
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

# The deleted sentence is Greek and is printed back for the record; a Windows
# console defaults to cp1252 and would abort on it.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent

# The sentence that carries the disclosure, as it appears in the chosen record.
# Matched literally rather than by pattern, so the control cannot silently delete
# something else if the source record is ever swapped.
DISCLOSURE_SENTENCE = (
    "Ο καιρός προέρχεται από **πρόγνωση**, άρα είναι λιγότερο βέβαιος από "
    "ιστορικά δεδομένα. "
)

# Words that would still tell the judge the weather came from forecast. After the
# deletion none of them may remain anywhere in the answer block.
RESIDUAL = re.compile(r"πρόγνωσ|forecast|ιστορικ|αρχειακ|archive", re.I)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_record(text: str):
    """(head, answer, tail). Only `answer` is ever edited."""
    a = text.index("THE ANSWER UNDER REVIEW, verbatim and complete:")
    b = text.index("THE OPERATOR'S ORIGINAL REQUEST, verbatim:")
    return text[:a], text[a:b], text[b:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="a pilot record whose case requires the forecast disclosure")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)

    original = Path(a.source).read_text(encoding="utf-8")
    source_id = original.split("\n", 1)[0].replace("RECORD ", "").strip()
    head, answer, tail = split_record(original)
    problems = []

    if DISCLOSURE_SENTENCE not in answer:
        problems.append("the disclosure sentence is not in this record's answer; "
                        "the control cannot be built from it")
        print("\n".join(problems))
        return 1

    stripped_answer = answer.replace(DISCLOSURE_SENTENCE, "", 1)
    unchanged = original.replace(f"RECORD {source_id}", "RECORD NC002", 1)
    control = (head + stripped_answer + tail).replace(
        f"RECORD {source_id}", "RECORD NC001", 1)

    (out / "records" / "NC001.txt").write_text(control, encoding="utf-8", newline="\n")
    (out / "records" / "NC002.txt").write_text(unchanged, encoding="utf-8", newline="\n")

    # ------------------------------------------------------------- verification
    h1, a1, t1 = split_record(control)
    h2, a2, t2 = split_record(unchanged)
    if t1 != t2:
        problems.append("the payload, gold, request or pins differ between the two "
                        "records; only the answer may differ")
    if h1.replace("NC001", "X") != h2.replace("NC002", "X"):
        problems.append("the header differs by more than the record id")
    if a2 != answer:
        problems.append("the unchanged twin's answer is not the source answer")

    removed = [l for l in difflib.unified_diff(a2.split("\n"), a1.split("\n"), lineterm="")
               if l.startswith("-") and not l.startswith("---")]
    added = [l for l in difflib.unified_diff(a2.split("\n"), a1.split("\n"), lineterm="")
             if l.startswith("+") and not l.startswith("+++")]
    if added and any(RESIDUAL.search(l) for l in added):
        problems.append("the edit introduced new weather wording")
    if RESIDUAL.search(a1):
        problems.append("forecast wording survives in the control's answer: "
                        + repr(RESIDUAL.search(a1).group(0)))
    if not RESIDUAL.search(a2):
        problems.append("the unchanged twin has no forecast wording, so it is not a "
                        "valid positive twin")

    # everything except the deleted sentence must survive verbatim
    if a1 != a2.replace(DISCLOSURE_SENTENCE, "", 1):
        problems.append("the control's answer is not exactly the twin minus the sentence")

    manifest = {
        "kind": "SYNTHETIC NEGATIVE CONTROL, NOT A MODEL OUTPUT",
        "purpose": ("verify that the frozen judge fails required_disclosures_complete "
                    "when the required disclosure is absent"),
        "never_publish": ("these two records are technical controls. They are not runs, "
                          "carry no model identity, and must not enter any result, "
                          "denominator, rate or comparison."),
        "source_record": str(Path(a.source)),
        "source_record_id": source_id,
        "edit": ("exactly one sentence deleted from the answer of NC001: "
                 + DISCLOSURE_SENTENCE.strip()),
        "expected": {"NC001": {"required_disclosures_complete": False,
                               "stage_c_pass": False},
                     "NC002": {"required_disclosures_complete": True,
                               "stage_c_pass": True}},
        "frozen_artefacts_unchanged": True,
        "rubric_sha256": hashlib.sha256(
            (HERE / "stage_c_v3_rubric.md").read_bytes()).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(
            (HERE / "stage_c_v3_judge_prompt.txt").read_bytes()).hexdigest(),
        "judge_schema_sha256": hashlib.sha256(
            (HERE / "stage_c_v3_judge_schema.json").read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(
            (HERE / "stage_c_v3_gold.json").read_bytes()).hexdigest(),
        "control_builder_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "records": [
            {"record_id": "NC001", "file": "records/NC001.txt",
             "role": "disclosure removed", "sha256": sha(control)},
            {"record_id": "NC002", "file": "records/NC002.txt",
             "role": "unchanged twin", "sha256": sha(unchanged)},
        ],
    }
    (out / "blind_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (out / "README.txt").write_text(
        "SYNTHETIC NEGATIVE CONTROL. NOT SCIENTIFIC DATA.\n\n"
        "NC001 and NC002 are hand-built text, not model outputs. They exist only to\n"
        "test whether the frozen Stage-C judge notices a missing required disclosure.\n"
        "They carry no model, no arm and no repetition, they belong to no run, and they\n"
        "must never appear in a published result, a denominator, a rate or a comparison.\n\n"
        f"NC002 is a pilot record with only its id rewritten.\n"
        f"NC001 is the same record with one sentence deleted:\n\n"
        f"    {DISCLOSURE_SENTENCE.strip()}\n\n"
        "Nothing else differs. No frozen artefact was modified to build these.\n",
        encoding="utf-8", newline="\n")

    print(f"source record            : {source_id}")
    print(f"sentence deleted         : {DISCLOSURE_SENTENCE.strip()[:70]}...")
    print(f"lines removed from answer: {len(removed)}, lines added: {len(added)}")
    print(f"payload/gold/request identical between the two : {t1 == t2}")
    print(f"control answer == twin answer minus the sentence: "
          f"{a1 == a2.replace(DISCLOSURE_SENTENCE, '', 1)}")
    print(f"forecast wording left in the control's answer   : "
          f"{bool(RESIDUAL.search(a1))}")
    print(f"forecast wording present in the twin's answer   : "
          f"{bool(RESIDUAL.search(a2))}")
    print(f"out                      : {out}")
    print()
    print(f"PROBLEMS: {problems if problems else 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

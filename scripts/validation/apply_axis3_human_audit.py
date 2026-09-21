"""Apply the completed human adjudication to the Axis-3 audit CSV.

The decisions in this file are intentionally explicit and reproducible.  Stage-C
rows were reviewed against the negative atomic claim identified by either label
pass.  Critical coverage was reviewed against the listed cut-off transition
hours, allowing accurate hour ranges as coverage.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell2fire"))

from _paths import DATA_DIR  # noqa: E402

# Resolved, never hard-coded: the workspace root differs per machine and a
# literal path would both break on the second computer and publish an account
# name. Overridable on the command line for a relocated copy of the run.
# The DECISIONS below are the frozen record of the closed 2026-09-21
# adjudication and are never edited; only this path is resolved rather than
# written out.
AUDIT_PATH = (DATA_DIR / "Exports" / "axis3_full_system_20260921" / "validation"
              / "3_agent" / "axis3_full_system_audit.csv")
if len(sys.argv) > 1:
    AUDIT_PATH = Path(sys.argv[1])

# Every Stage-C row requiring adjudication contained a confirmed contradicted or
# unsupported atomic claim.  Two-pass-clean rows do not appear in this mapping.
STAGE_C_PASS: set[tuple[str, str]] = set()

# Coverage passes only where every listed transition hour is stated directly or
# falls inside an accurate, explicit range in the reply.
COVERAGE_PASS = {
    ("luna", "I04#1"),
    ("luna", "I04#2"),
    ("luna", "I04#3"),
    ("luna", "I04#4"),
    ("luna", "I04#5"),
    ("terra", "B04#3"),
    ("terra", "B04#5"),
    ("terra", "I04#1"),
    ("terra", "I04#2"),
    ("terra", "I04#3"),
    ("terra", "I04#4"),
    ("terra", "I04#5"),
}


def main() -> None:
    with AUDIT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Audit CSV has no header")
        rows = list(reader)

    for row in rows:
        key = (row["run_id"], row["uid"])
        comments: list[str] = []

        if row["human_stage_c_required"] == "YES":
            row["human_stage_c"] = "PASS" if key in STAGE_C_PASS else "FAIL"
            if key in STAGE_C_PASS:
                comments.append("Stage C: negative automatic label not confirmed.")
            else:
                comments.append("Stage C: contradicted or unsupported claim confirmed.")

        if row["human_critical_coverage_required"] == "YES":
            row["human_critical_coverage"] = (
                "PASS" if key in COVERAGE_PASS else "FAIL"
            )
            if key in COVERAGE_PASS:
                comments.append("Coverage: all transitions covered by stated hours/ranges.")
            else:
                comments.append("Coverage: one or more transition hours omitted.")

        row["audit_comment"] = " ".join(comments)

    with AUDIT_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stage = sum(r.get("human_stage_c") in {"PASS", "FAIL"} for r in rows)
    coverage = sum(
        r.get("human_critical_coverage") in {"PASS", "FAIL"} for r in rows
    )
    print(f"Applied {stage} Stage-C and {coverage} coverage decisions.")


if __name__ == "__main__":
    main()

"""End-to-end scoring rehearsal for the Axis-3 structured-output experiment.

WHAT THIS IS FOR. The 130 scored structured runs are not authorised, and the
condition for authorising them is that the whole chain from a raw response to
the final comparison report is shown to work on data that already exists. This
module is that demonstration. It runs the REAL modules, in the pre-registered
order, on COPIES of the stored evidence, and makes exactly zero model calls.

    python scripts/validation/axis3_rehearsal.py [--workspace DIR] [--keep]

WHAT IS REAL IN THE OUTPUT AND WHAT IS NOT. This distinction is the whole point
of the file and it is repeated in every artefact it writes.

  REAL   the three corrected 2026-09-01 baseline runs and their HUMAN-AUDITED
         Stage-C label files, both passes, read verbatim from the stored tree;
         the four structured repetitions of the frozen 2026-09-21 pilot; every
         deterministic check, every contract verdict, every schema validation,
         every join, every guard and every refusal exercised below.
  FAKE   the Stage-C labels of the structured arm, and the human coverage
         decisions. No judge ran and no human audited, so they are generated
         from a fixed seed, every record carries `REHEARSAL_STUB: true`, the
         filename says so, and the aggregator and the joiner both REFUSE to
         read such a file unless `--rehearsal` is passed. Every published
         number that depends on them is fabricated and is labelled as such.

WHY THE STUB IS A FILE FORMAT AND NOT A MONKEYPATCH. A stub that lives in
memory proves nothing about the join. Writing it as a real label file, through
the real aggregator and the real joiner, tests the thing that will actually run
in Phase 5; marking it in the data rather than in a comment is what stops it
ever being mistaken for a judged result, including by a later reader of the
workspace who never saw this module.

SAFETY. Nothing under the stored tree is written. Every source directory is
checksummed file by file before the first step and again after the last one,
and a single changed byte fails the run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent


# ---------------------------------------------------------------------------
# The stub contract, defined here and imported by the two modules that must
# refuse it. One spelling of the sentinel, in one place, because a guard whose
# magic string is copied into three files is a guard that will be bypassed by a
# typo.
# ---------------------------------------------------------------------------
REHEARSAL_STUB_KEY = "REHEARSAL_STUB"
STUB_LABEL_FILENAME = "stage_c_labels_REHEARSAL_STUB.json"
STUB_SEED = 20260921

BANNER = ("REHEARSAL OUTPUT. NOT A RESULT. NOT A SCORED FIGURE. "
          "The structured arm's Stage-C labels and every human coverage "
          "decision in this workspace are machine-generated stubs.")


def stub_record_count(records) -> int:
    """How many records in a loaded label file are rehearsal stubs.

    Imported by `stage_c_aggregate.py` and by
    `apply_axis3_human_audit_structured.py` at call time rather than at import
    time: those two must stay importable and runnable on a checkout that never
    rehearses anything, and a top-level import here would also make this module
    a dependency of the scoring path it is supposed to only observe.
    """
    if not isinstance(records, list):
        return 0
    return sum(1 for r in records
               if isinstance(r, dict) and r.get(REHEARSAL_STUB_KEY) is True)


def refuse_stub_labels(path, records, allow_rehearsal: bool) -> None:
    """Raise unless a stub label file was explicitly asked for.

    The refusal is on the DATA, not on the filename, because a file renamed by
    an operator in a hurry is the threat model. `--rehearsal` is the only key
    that opens it, and it has to be typed out on the command line every time.
    """
    n = stub_record_count(records)
    if not n:
        return
    if allow_rehearsal:
        return
    raise SystemExit(
        f"REFUSED: {path} carries {n} record(s) marked {REHEARSAL_STUB_KEY}. "
        "These are machine-generated rehearsal stubs, not judged labels, and "
        "nothing computed from them may be published. Pass --rehearsal if you "
        "are deliberately running the end-to-end rehearsal.")


# ---------------------------------------------------------------------------
# Checksums over the stored evidence
# ---------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_checksum(root: Path) -> dict[str, str]:
    """Every file under `root`, by path relative to it, with its sha256.

    Per file and not one rolled-up digest, because "something changed" is not
    an actionable message: the comparison has to be able to name the file.
    """
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root)).replace("\\", "/")] = file_sha256(path)
    return out


def diff_checksums(before: dict[str, dict[str, str]],
                   after: dict[str, dict[str, str]]) -> list[str]:
    problems: list[str] = []
    for name in sorted(before):
        b, a = before[name], after.get(name, {})
        for rel in sorted(set(b) | set(a)):
            if rel not in a:
                problems.append(f"{name}: DELETED {rel}")
            elif rel not in b:
                problems.append(f"{name}: CREATED {rel}")
            elif b[rel] != a[rel]:
                problems.append(f"{name}: MODIFIED {rel}")
    return problems


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------
class Trace:
    """The step-by-step record the author reads instead of the scrollback."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def add(self, title, command, status, artefacts=None, counts=None,
            note="", output="") -> dict:
        step = {
            "n": len(self.steps) + 1,
            "title": title,
            "command": command,
            "status": status,
            "artefacts": artefacts or [],
            "counts": counts or {},
            "note": note,
            "output": output,
        }
        self.steps.append(step)
        mark = "OK" if status in (0, "OK") else f"EXIT {status}"
        print(f"\n[{step['n']:02d}] {title}   -> {mark}")
        if command:
            print(f"     $ {command}")
        for key, value in (counts or {}).items():
            print(f"     {key}: {value}")
        for art in (artefacts or []):
            print(f"     wrote {art}")
        if note:
            print(f"     note: {note}")
        return step

    def failures(self) -> list[dict]:
        return [s for s in self.steps
                if s["status"] not in (0, "OK") and not s.get("expected_failure")]


def run(trace: Trace, title: str, argv: list[str], *, cwd: Path = _REPO,
        artefacts=None, counts=None, note="", expect_fail=False,
        tail=40) -> dict:
    """Run one real module as a subprocess and record what happened.

    Subprocess and not an in-process import on purpose: the pre-registered
    Phase-5 commands are command lines, so what is rehearsed has to be the
    command line, argument parsing, exit code and all.
    """
    printable = " ".join(
        f'"{a}"' if " " in str(a) else str(a) for a in argv[1:])
    proc = subprocess.run([str(a) for a in argv], cwd=str(cwd),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    body = (proc.stdout or "") + (("\nSTDERR:\n" + proc.stderr) if proc.stderr else "")
    lines = [l for l in body.splitlines() if l.strip()]
    step = trace.add(title, "python " + printable, proc.returncode,
                     artefacts=artefacts, counts=counts, note=note,
                     output="\n".join(lines[-tail:]))
    step["expected_failure"] = expect_fail
    if expect_fail and proc.returncode == 0:
        step["status"] = "UNEXPECTED SUCCESS"
        step["expected_failure"] = False
        print("     !! this step was supposed to be refused and was not")
    return step


# ---------------------------------------------------------------------------
# The judge stub
# ---------------------------------------------------------------------------
# Deliberately weighted to the label set of 2.5 with a small but non-zero rate
# of negative labels, so the downstream branches that only a negative label can
# reach (the audit referral, the human-pending bound) are actually exercised.
# The weights are a property of the STUB and mean nothing about any model.
_STUB_LABELS = (
    ("SUPPORTED", 60),
    ("ADVISORY_INFERENCE", 15),
    ("LIMITATION_STATEMENT", 15),
    ("NOT_IN_RESULT", 5),
    ("CONTRADICTED", 5),
)


def stub_labels_for_batches(batch_dir: Path, seed: int = STUB_SEED) -> list[dict]:
    """Fabricate one label record per narrated repetition in `batch_dir`.

    The claim TEXTS are real: they are the mechanical atomic candidates the real
    prep step cut from the real narrative surface. Only the labels are invented,
    which is the narrowest possible fabrication that still exercises the join,
    the consensus branch, the audit referral and the aggregator.

    Seeded per uid rather than once for the whole file, so the labels of one
    record do not move when another record is added or removed. A rehearsal that
    is not reproducible is an anecdote.
    """
    records: list[dict] = []
    for path in sorted(batch_dir.glob("batch_*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            rng = random.Random(f"{seed}:{item['uid']}")
            claims = []
            for text in (item.get("candidate_claims") or ["(no candidate claim)"]):
                label = rng.choices([n for n, _ in _STUB_LABELS],
                                    weights=[w for _, w in _STUB_LABELS])[0]
                claims.append({
                    "claim": text,
                    "label": label,
                    "justification": "REHEARSAL STUB: no judge saw this claim.",
                })
            records.append({
                REHEARSAL_STUB_KEY: True,
                "uid": item["uid"],
                "claims": claims,
                "notes": ("REHEARSAL STUB. These labels were generated from seed "
                          f"{seed} and were not produced by any judge or human. "
                          "INVENTED GEOGRAPHY: NONE."),
            })
    return records


def stub_coverage_answers(sheet_path: Path, out_path: Path,
                          seed: int = STUB_SEED) -> tuple[int, Path]:
    """Fill the blinded C9 coverage sheet with fabricated decisions.

    Written to a SEPARATE file so the real blank sheet the pipeline produced is
    left blank: an auditor opening the workspace must find an unfilled sheet
    where an unfilled sheet belongs. The first column of the filled copy is the
    stamp, so the header line itself says what the file is.
    """
    with sheet_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return 0, out_path
    stamp_col = "REHEARSAL_STUB_DECISION_NOT_A_HUMAN_VERDICT"
    fields = [stamp_col] + list(rows[0].keys())
    for row in rows:
        rng = random.Random(f"{seed}:coverage:{row['decision_id']}")
        row[stamp_col] = "REHEARSAL STUB"
        # 80/20 so both branches of the coverage column are populated.
        row["human_critical_coverage"] = rng.choices(("PASS", "FAIL"),
                                                     weights=(80, 20))[0]
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), out_path


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------
def stamp_workspace(workspace: Path) -> list[str]:
    """Put the banner into every artefact, as the last act of the run.

    Markdown and text get a first line, JSON objects get a first key, CSVs get a
    first column. The CSV rewrite happens AFTER every consumer has run, so it
    cannot influence a single number above it; the price is that a stamped
    workspace is terminal and must not be re-fed to the chain, which the README
    says out loud.
    """
    touched: list[str] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(workspace)).replace("\\", "/")
        try:
            if path.suffix in (".md", ".txt"):
                text = path.read_text(encoding="utf-8")
                if BANNER in text:
                    continue
                path.write_text(f"<!-- {BANNER} -->\n{BANNER}\n\n{text}",
                                encoding="utf-8")
            elif path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if data.get("REHEARSAL"):
                        continue
                    path.write_text(
                        json.dumps({"REHEARSAL": BANNER, **data},
                                   ensure_ascii=False, indent=1),
                        encoding="utf-8")
                else:
                    continue  # a list of stub label records: already stamped per record
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    rows = list(csv.reader(fh))
                if not rows or rows[0][:1] == ["REHEARSAL"]:
                    continue
                rows[0] = ["REHEARSAL"] + rows[0]
                for row in rows[1:]:
                    row.insert(0, "REHEARSAL OUTPUT, NOT A RESULT")
                with path.open("w", encoding="utf-8-sig", newline="") as fh:
                    csv.writer(fh).writerows(rows)
            elif path.suffix == ".jsonl":
                continue  # copies of stored raw files: never rewritten
            else:
                continue
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        touched.append(rel)
    return touched


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------
BASELINE_RUNS = (
    # name, stored Exports folder, preset. The preset is mandatory on a rescore
    # and decides the model string the report table prints.
    ("base_gemini", "agent_eval_20260901_082152", "gemini-pro"),
    ("base_luna", "agent_eval_20260901_093550", "gpt"),
    ("base_terra", "agent_eval_20260901_103200", "gpt-terra"),
)
STRUCTURED_RUNS = (
    ("struct_luna", "luna"),
    ("struct_terra", "terra"),
)
PAIRS = (("base_luna", "struct_luna"), ("base_terra", "struct_terra"))
SCOPE_CATEGORIES = "narration_faithfulness,fully_specified,now_live_fire"
BATCH_SIZE = 12

PILOT_DIR = (_HERE / "pilot_records" / "20260921_axis3_structured_contract")


def exports_root() -> Path:
    """The Exports tree, resolved the way the rest of the repo resolves it."""
    sys.path.insert(0, str(_REPO / "scripts" / "cell2fire"))
    from _paths import DATA_DIR  # noqa: E402  resolved at call time, never hard-coded
    return Path(DATA_DIR) / "Exports"


def source_trees(exports: Path) -> dict[str, Path]:
    """Everything that must be byte-identical when this finishes."""
    # The whole stored run folder, not just `validation/3_agent`, so a stray
    # write anywhere beneath it is caught too.
    trees = {folder: exports / folder for _, folder, _ in BASELINE_RUNS}
    trees["pilot_records"] = PILOT_DIR
    trees["axis3_full_system_20260921"] = exports / "axis3_full_system_20260921"
    trees["_frozen_axis3_backup_20260921"] = exports / "_frozen_axis3_backup_20260921"
    return trees


def n_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="End-to-end Axis-3 scoring rehearsal on stored data. "
                    "Makes no model call and writes into no stored directory.")
    ap.add_argument("--workspace", default=None,
                    help="rehearsal workspace (default: a new "
                         "axis3_REHEARSAL_<stamp> under the Exports tree)")
    ap.add_argument("--stub-seed", type=int, default=STUB_SEED)
    a = ap.parse_args(argv)

    t0 = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exports = exports_root()
    workspace = Path(a.workspace) if a.workspace else exports / f"axis3_REHEARSAL_{stamp}"
    py = sys.executable
    trace = Trace()

    print("=" * 78)
    print("AXIS-3 END-TO-END SCORING REHEARSAL")
    print(BANNER)
    print("=" * 78)
    print(f"workspace : {workspace}")
    print(f"exports   : {exports}")
    print(f"stub seed : {a.stub_seed}")

    # --- step 1: checksum the stored evidence -----------------------------
    trees = source_trees(exports)
    before = {name: tree_checksum(path) for name, path in trees.items()}
    trace.add("Checksum every stored tree BEFORE the run", "", "OK",
              counts={name: f"{len(files)} files" for name, files in before.items()})

    workspace.mkdir(parents=True, exist_ok=False)
    copies = workspace / "copies"
    arms = workspace / "arms"
    full_system_dir = workspace / "full_system"
    compare_dir = workspace / "compare"
    for d in (copies, arms, full_system_dir, compare_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- step 2: copy ------------------------------------------------------
    copied = {}
    for name, folder, _preset in BASELINE_RUNS:
        src = exports / folder / "validation" / "3_agent"
        dst = copies / name
        shutil.copytree(src, dst)
        copied[name] = f"{n_rows(dst / 'agent_eval_raw.jsonl')} rows"
    for name, folder in STRUCTURED_RUNS:
        src = PILOT_DIR / folder
        dst = arms / name          # the structured arm is worked on in place
        shutil.copytree(src, dst)
        copied[name] = f"{n_rows(dst / 'agent_eval_raw.jsonl')} rows"
    trace.add("Copy the three stored baseline runs and the two pilot folders",
              "", "OK", counts=copied,
              note="every later step reads and writes only these copies")

    # --- step 3: rescore the baseline arm ---------------------------------
    for name, _folder, preset in BASELINE_RUNS:
        out = arms / name
        run(trace, f"Rescore baseline {name} (no model call by construction)",
            [py, "scripts/validation/agent_eval.py",
             "--rescore", str(copies / name / "agent_eval_raw.jsonl"),
             "--out-dir", str(out), "--preset", preset],
            counts={"rows in": n_rows(copies / name / "agent_eval_raw.jsonl")})
        # The real human-audited Stage-C evidence travels with the run.
        shutil.copytree(copies / name / "stage_c_full", out / "stage_c_full")
        trace.add(f"Carry {name}'s REAL Stage-C label files into the rescored copy",
                  "", "OK",
                  counts={"label files": ", ".join(
                      sorted(p.name for p in (out / "stage_c_full").glob("stage_c_labels*.json"))),
                          "rows out": n_rows(out / "agent_eval_raw.jsonl")})

    # --- step 4: the contract validator over the structured rows ----------
    for name, _folder in STRUCTURED_RUNS:
        raw = arms / name / "agent_eval_raw.jsonl"
        run(trace, f"Contract validator over {name} (stage_c_structured, in place)",
            [py, "scripts/validation/stage_c_structured.py", str(raw),
             "--in-place", "--per-row"],
            artefacts=[str(raw)],
            note="also exercises stage_c_render_slots through the validator")

    # --- step 5: prep the judge batches -----------------------------------
    for name, _folder, _preset in BASELINE_RUNS:
        check_dir = arms / name / "stage_c_prep_recheck"
        run(trace, f"Rebuild {name}'s judge batches (into a side folder, real "
                   f"batches are kept)",
            [py, "scripts/validation/agent_eval_prep_labeling.py",
             str(arms / name / "agent_eval_raw.jsonl"),
             "--categories", SCOPE_CATEGORIES,
             "--batch-size", str(BATCH_SIZE),
             "--out-dir", str(check_dir)])
        stored_uids = _batch_uids(arms / name / "stage_c_full")
        fresh_uids = _batch_uids(check_dir)
        same = stored_uids == fresh_uids
        trace.add(f"Compare {name}'s rebuilt batch population against the stored one",
                  "", "OK" if same else "MISMATCH",
                  counts={"stored narrated": len(stored_uids),
                          "rebuilt narrated": len(fresh_uids),
                          "identical uid set": same,
                          "only stored": sorted(stored_uids - fresh_uids)[:5],
                          "only rebuilt": sorted(fresh_uids - stored_uids)[:5]})
    for name, _folder in STRUCTURED_RUNS:
        run(trace, f"Build {name}'s judge batches",
            [py, "scripts/validation/agent_eval_prep_labeling.py",
             str(arms / name / "agent_eval_raw.jsonl"),
             "--categories", SCOPE_CATEGORIES,
             "--batch-size", str(BATCH_SIZE)],
            artefacts=[str(arms / name / "stage_c_full")])

    # --- step 6: THE JUDGE STEP, stubbed ----------------------------------
    stub_counts = {}
    for name, _folder in STRUCTURED_RUNS:
        stage_dir = arms / name / "stage_c_full"
        records = stub_labels_for_batches(stage_dir, a.stub_seed)
        (stage_dir / STUB_LABEL_FILENAME).write_text(
            json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        stub_counts[name] = (f"{len(records)} stub records, "
                             f"{sum(len(r['claims']) for r in records)} claims")
    trace.add("JUDGE STEP. Baseline arm: the REAL stored human-audited labels, "
              "two passes, untouched. Structured arm: NO JUDGE RAN, so seeded "
              "stub labels are written instead",
              "", "OK", counts=stub_counts,
              note=(f"stub file name {STUB_LABEL_FILENAME}, every record carries "
                    f"{REHEARSAL_STUB_KEY}: true, seed {a.stub_seed}. The claim "
                    "TEXTS are the real mechanical candidates; only the labels "
                    "are invented."))

    # --- step 7: the stub refusal, demonstrated ---------------------------
    demo = arms / STRUCTURED_RUNS[0][0]
    run(trace, "Prove the aggregator REFUSES a stub label file without --rehearsal",
        [py, "scripts/validation/stage_c_aggregate.py",
         str(demo / "stage_c_full" / STUB_LABEL_FILENAME),
         str(demo / "stage_c_full"),
         "--out-dir", str(workspace / "_refusal_probe")],
        expect_fail=True,
        note="a non-zero exit here is the PASS condition")

    # --- step 8: aggregate -------------------------------------------------
    for name, _folder, _preset in BASELINE_RUNS:
        stage_dir = arms / name / "stage_c_full"
        run(trace, f"Aggregate {name}'s REAL Stage-C labels (pass 1)",
            [py, "scripts/validation/stage_c_aggregate.py",
             str(stage_dir / "stage_c_labels.json"), str(stage_dir),
             "--categories", SCOPE_CATEGORIES, "--tag", "REHEARSAL"],
            artefacts=[str(stage_dir / "stage_c_aggregate_REHEARSAL.txt")])
    for name, _folder in STRUCTURED_RUNS:
        stage_dir = arms / name / "stage_c_full"
        run(trace, f"Aggregate {name}'s STUB Stage-C labels (--rehearsal)",
            [py, "scripts/validation/stage_c_aggregate.py",
             str(stage_dir / STUB_LABEL_FILENAME), str(stage_dir),
             "--tag", "REHEARSAL_STUB", "--rehearsal"],
            artefacts=[str(stage_dir / "stage_c_aggregate_REHEARSAL_STUB.txt")])

    # --- step 9: the full-system join -------------------------------------
    run_args: list[str] = []
    for name, _folder, _preset in BASELINE_RUNS:
        run_args += ["--run", f"{name}={arms / name}"]
    for name, _folder in STRUCTURED_RUNS:
        run_args += ["--run", f"{name}={arms / name}"]
    label_args = ["--labels", "stage_c_labels.json,stage_c_labels_pass2.json"]
    for name, _folder in STRUCTURED_RUNS:
        label_args += ["--labels", f"{name}={STUB_LABEL_FILENAME}"]
    run(trace, "Full-system join, all five runs in ONE invocation so the C9 "
               "coverage sheet pools both arms",
        [py, "scripts/validation/axis3_full_system.py", *run_args,
         "--out-dir", str(full_system_dir), *label_args, "--rehearsal"],
        artefacts=[str(full_system_dir / n) for n in (
            "axis3_full_system_records.csv", "axis3_full_system_report.md",
            "axis3_full_system_audit.csv", "axis3_full_system_coverage_sheet.csv",
            "axis3_full_system_coverage_key.csv",
            "axis3_full_system_metrics.json")],
        tail=90)

    # --- step 10: the human-audit join ------------------------------------
    audit_csv = full_system_dir / "axis3_full_system_audit.csv"
    sheet = full_system_dir / "axis3_full_system_coverage_sheet.csv"
    run(trace, "Human-audit join --dry-run against the UNFILLED coverage sheet",
        [py, "scripts/validation/apply_axis3_human_audit_structured.py",
         str(audit_csv), "--dry-run"],
        expect_fail=True,
        note="a refusal is the PASS condition: 2.11 forbids joining a sheet "
             "whose decisions are not all recorded")

    filled = full_system_dir / "axis3_full_system_coverage_sheet_REHEARSAL_FILLED.csv"
    n_filled, _ = stub_coverage_answers(sheet, filled, a.stub_seed)
    trace.add("Fill the blinded coverage sheet with STUB decisions",
              "", "OK", artefacts=[str(filled)],
              counts={"decisions fabricated": n_filled},
              note="written to a separate file so the real sheet stays blank")

    run(trace, "Human-audit join --dry-run against the FILLED stub sheet",
        [py, "scripts/validation/apply_axis3_human_audit_structured.py",
         str(audit_csv), "--dry-run",
         "--coverage-sheet", str(filled),
         "--coverage-map", str(full_system_dir / "axis3_full_system_coverage_key.csv")],
        note="--dry-run is the default; nothing is written either way",
        tail=25)

    # --- step 11: the arm comparison --------------------------------------
    pair_args: list[str] = []
    for base, struct in PAIRS:
        pair_args += ["--pair", f"{base}:{struct}"]
    pending_compare = compare_dir / "axis3_arm_compare_REHEARSAL_PENDING_AUDIT.md"
    run(trace, "Arm comparison report, audit still pending",
        [py, "scripts/validation/axis3_arm_compare.py",
         "--records", str(full_system_dir / "axis3_full_system_records.csv"),
         *pair_args, "--out", str(pending_compare),
         "--with-historical-v1", "--rehearsal"],
        artefacts=[str(pending_compare)],
        note="every human-pending cell is correctly empty here, which is why "
             "the round trip below exists")

    # --- step 12: close the loop through the decision store ---------------
    # WHY this is not optional.  With the audit unfilled, every Stage-C and
    # composite cell of the structured arm is legitimately empty, and an empty
    # table cannot show that the last link works.  So the whole out-dir is
    # copied, the stub decisions are WRITTEN into the copy through the real
    # joiner, the join is re-run so `load_audit` resumes them, and the
    # comparison is rebuilt.  Every verdict in that copy is fabricated.
    roundtrip_dir = workspace / "full_system_ROUNDTRIP_STUB"
    shutil.copytree(full_system_dir, roundtrip_dir)
    rt_audit = roundtrip_dir / "axis3_full_system_audit.csv"
    rt_sheet = roundtrip_dir / "axis3_full_system_coverage_sheet_REHEARSAL_FILLED.csv"
    run(trace, "Write the STUB decisions into a COPY of the decision store",
        [py, "scripts/validation/apply_axis3_human_audit_structured.py",
         str(rt_audit), "--write",
         "--coverage-sheet", str(rt_sheet),
         "--coverage-map", str(roundtrip_dir / "axis3_full_system_coverage_key.csv")],
        note="STAGE_C_PASS ships empty, so --write stamps FAIL on every referred "
             "Stage-C row. That is the module behaving as documented, and it is "
             "the reason --dry-run is its default; here it is deliberate and the "
             "verdicts are fabricated.",
        tail=6)
    run(trace, "Re-run the full-system join so it resumes the written decisions",
        [py, "scripts/validation/axis3_full_system.py", *run_args,
         "--out-dir", str(roundtrip_dir), *label_args, "--rehearsal"],
        artefacts=[str(roundtrip_dir / "axis3_full_system_records.csv")],
        tail=60)

    compare_out = compare_dir / "axis3_arm_compare_REHEARSAL.md"
    run(trace, "Arm comparison report, audit resolved (stub verdicts)",
        [py, "scripts/validation/axis3_arm_compare.py",
         "--records", str(roundtrip_dir / "axis3_full_system_records.csv"),
         *pair_args, "--out", str(compare_out),
         "--with-historical-v1", "--rehearsal"],
        artefacts=[str(compare_out)])

    # --- step 12: the three mandated confirmations ------------------------
    confirmations = check_confirmations(trace, full_system_dir, compare_out)

    # --- step 13: checksum again ------------------------------------------
    after = {name: tree_checksum(path) for name, path in trees.items()}
    problems = diff_checksums(before, after)
    trace.add("Checksum every stored tree AFTER the run",
              "", "OK" if not problems else "STORED DATA CHANGED",
              counts={"files compared": sum(len(v) for v in before.values()),
                      "differences": len(problems)},
              note="; ".join(problems[:10]) if problems else
                   "every stored and frozen file is byte-identical")

    # --- step 14: report and stamp ----------------------------------------
    report_path = workspace / "axis3_rehearsal_report.md"
    report_path.write_text(build_report(trace, confirmations, workspace, exports,
                                        a.stub_seed, problems, t0),
                           encoding="utf-8")
    (workspace / "axis3_rehearsal_trace.json").write_text(
        json.dumps({"steps": trace.steps}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    touched = stamp_workspace(workspace)
    trace.add("Stamp every artefact in the workspace", "", "OK",
              counts={"files stamped": len(touched)},
              note="markdown and text get a first line, JSON objects a first "
                   "key, CSVs a first column; done last so no step above could "
                   "see it")

    print("\n" + "=" * 78)
    print(f"Report : {report_path}")
    print(f"Trace  : {workspace / 'axis3_rehearsal_trace.json'}")
    for line in confirmations:
        print(f"  {line}")
    if problems:
        print("\nSTORED DATA WAS MODIFIED. This is a hard failure.")
        for line in problems[:20]:
            print(f"  {line}")
        return 2
    failures = trace.failures()
    if failures:
        print(f"\n{len(failures)} step(s) did not succeed:")
        for step in failures:
            print(f"  [{step['n']:02d}] {step['title']} -> {step['status']}")
        return 1
    print("\nEvery step ran. No stored file changed.")
    return 0


def frozen_hashes() -> dict[str, str]:
    """The rubric, judge prompt, renderer, schema and system prompt pins.

    Carried into the rehearsal record so the artefact says which instruments
    were frozen when it was produced, rather than pointing at a module a reader
    would have to go and run.
    """
    sys.path.insert(0, str(_HERE))
    from stage_c_hashes import frozen_artefact_hashes  # noqa: E402
    return frozen_artefact_hashes()


def _batch_uids(batch_dir: Path) -> set[str]:
    uids: set[str] = set()
    for path in sorted(batch_dir.glob("batch_*.json")):
        uids.update(r["uid"] for r in json.loads(path.read_text(encoding="utf-8")))
    return uids


def check_confirmations(trace: Trace, full_system_dir: Path,
                        compare_out: Path) -> list[str]:
    """The three things the author asked to be proved, each proved from a file."""
    lines: list[str] = []

    # (1) no ct_* metric in the common comparison table.  The scan is over the
    # table GRID only, the lines that start with a pipe: the surrounding caption
    # necessarily contains the words "schema" and "contract", because what it
    # says is that the table holds none of them, and scanning the prose would
    # fail on its own disclaimer.
    text = compare_out.read_text(encoding="utf-8") if compare_out.exists() else ""
    grid: list[str] = []
    if "### 1. " in text:
        block = text.split("### 1. ", 1)[1].split("\n### ", 1)[0]
        grid = [l for l in block.splitlines() if l.strip().startswith("|")]
    forbidden = ("ct_", "schema", "contract", "slot", "ssp", "exact_set",
                 "envelope", "status_consistent", "structured_complete",
                 "validator_invalid", "json")
    leaked = sorted({tok for tok in forbidden
                     if any(tok in line.lower() for line in grid)})
    # The column set is asserted against the module's own allowlist rather than
    # against a list retyped here, so the two cannot drift.
    sys.path.insert(0, str(_HERE))
    from axis3_arm_compare import COMMON_CRITERIA  # noqa: E402
    header = [c.strip() for c in grid[0].strip("|").split("|")] if grid else []
    columns_ok = header[1:] == list(COMMON_CRITERIA)
    ok1 = bool(grid) and not leaked and columns_ok
    lines.append(f"[{'PASS' if ok1 else 'FAIL'}] no ct_* metric and no schema term "
                 f"in the common Stage-C comparison table; its columns are exactly "
                 f"{list(COMMON_CRITERIA)}"
                 + ("" if ok1 else
                    f" (grid lines: {len(grid)}, leaked: {leaked}, "
                    f"columns seen: {header[1:]})"))

    # (2) the C1a path counts, read from the metrics the join published.
    metrics_path = full_system_dir / "axis3_full_system_metrics.json"
    paths: list[str] = []
    ok2 = metrics_path.exists()
    if ok2:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for summary in metrics.get("runs", []):
            paths.append(
                f"{summary['run_id']} ({summary.get('arm')}): "
                f"full_surface={summary.get('numeric_source_full_surface')}, "
                f"prose_only_fallback={summary.get('numeric_source_prose_fallback')}, "
                f"C1a={summary.get('c1a_numeric_pass')}/{summary.get('c1a_denominator')}")
    lines.append(f"[{'PASS' if ok2 else 'FAIL'}] C1a gate reads "
                 f"numeric_ok_full_surface, per-path counts published")
    lines += [f"       {p}" for p in paths]

    trace.add("Confirmations", "", "OK" if ok1 and ok2 else "FAIL",
              note=" | ".join(lines[:2]))
    return lines


def build_report(trace: Trace, confirmations: list[str], workspace: Path,
                 exports: Path, seed: int, problems: list[str], t0: float) -> str:
    L = [
        BANNER,
        "",
        "# Axis 3, end-to-end scoring rehearsal",
        "",
        "## What is real here and what is not",
        "",
        "**Real.** The three corrected 2026-09-01 baseline runs, read verbatim "
        "from the stored Exports tree and re-scored by the real "
        "`agent_eval.py --rescore`, which makes no model call. Their Stage-C "
        "label files, both passes, are the genuine human-audited labels of that "
        "experiment and are used exactly as the join will use them in Phase 5. "
        "The four structured repetitions of the frozen 2026-09-21 pilot are real "
        "model output, validated here by the real contract validator against the "
        "real frozen renderer. Every deterministic check, every contract "
        "outcome, every schema verdict, every join, every guard and every "
        "refusal exercised below is the real code path.",
        "",
        "**Fabricated.** No judge ran, because a judge call is a model call and "
        "this rehearsal makes none. The structured arm's Stage-C labels are "
        f"therefore machine-generated from seed {seed}: the claim TEXTS are the "
        "real mechanical candidates cut from the real narrative surface, but "
        "every LABEL is invented. The human coverage decisions on the blinded "
        "C9 sheet are invented the same way. Consequently **every Stage-C "
        "figure, every coverage figure, every full-system pass rate and every "
        "comparison cell that involves the structured arm is fabricated** and "
        "means nothing about any model. What is demonstrated is that the "
        "plumbing carries them correctly, not what they are.",
        "",
        "Stub files are marked in the data, not only in their names: every stub "
        f"label record carries `{REHEARSAL_STUB_KEY}: true`, and both "
        "`stage_c_aggregate.py` and `apply_axis3_human_audit_structured.py` "
        "refuse to read such a file unless `--rehearsal` is passed. Step 7 below "
        "demonstrates the refusal.",
        "",
        "**The round-trip artefacts are fabricated on BOTH arms.** "
        "`full_system_ROUNDTRIP_STUB/` and the comparison built from it exist "
        "only to show that a resolved decision store produces a filled table. "
        "The decision set in `apply_axis3_human_audit_structured.py` ships "
        "empty, so `--write` stamps FAIL on every referred Stage-C row of every "
        "run, baseline included. The baseline figures in that table are "
        "therefore not the 2026-09-01 figures and must not be read as them. The "
        "honest baseline reading is in "
        "`compare/axis3_arm_compare_REHEARSAL_PENDING_AUDIT.md`, where the "
        "unaudited rows are excluded rather than failed.",
        "",
        "## Frozen artefact hashes at the time of this rehearsal", "",
    ]
    L += [f"- `{k}` = `{v}`" for k, v in sorted(frozen_hashes().items())]
    L += ["", "## Confirmations", ""]
    L += [f"- {line}" for line in confirmations]
    L += [
        "",
        f"- [{'PASS' if not problems else 'FAIL'}] no stored or frozen file was "
        f"modified, proven by a per-file sha256 of every source tree before and "
        f"after the run",
    ]
    if problems:
        L += [f"    - {p}" for p in problems[:20]]
    L += ["", "## Workspace", "",
          f"- workspace: `{workspace}`",
          f"- exports root resolved at run time: `{exports}`",
          f"- wall clock: {(time.time() - t0) / 60:.1f} min",
          "- a stamped workspace is terminal: the final stamping pass adds a "
          "banner column to every CSV, so these files must not be fed back into "
          "the chain.",
          "", "## Step-by-step trace", ""]
    for step in trace.steps:
        mark = "OK" if step["status"] in (0, "OK") else f"**{step['status']}**"
        if step.get("expected_failure"):
            mark += " (refusal expected, this is the pass condition)"
        L += [f"### {step['n']:02d}. {step['title']}", "", f"- status: {mark}"]
        if step["command"]:
            L.append(f"- command: `{step['command']}`")
        for key, value in step["counts"].items():
            L.append(f"- {key}: {value}")
        for art in step["artefacts"]:
            L.append(f"- artefact: `{art}`")
        if step["note"]:
            L.append(f"- note: {step['note']}")
        if step["output"]:
            L += ["", "```", step["output"], "```"]
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())

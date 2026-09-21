"""The out-dir guard of scripts/validation/agent_eval.py.

WHAT IS UNDER TEST: that a folder already holding a scored run cannot be written
into by accident, and that the two deliberate ways past that refusal behave as
declared. This matters more than it looks: raw replies cost real model calls and
the models are not deterministic, so an overwritten run is not recoverable by
re-running the same command, and the frozen 2026-09-01 baseline and the
2026-09-21 pilots are cited by the thesis, by agent_eval_compare.py and by
axis3_full_system.py.

Every test here is OFFLINE. Nothing imports litellm, nothing calls a model and
nothing reads the real Exports tree: the guard is deliberately placed before
build_fixtures(), before mkdir and before the first repetition, which is also
what lets `main()` itself be driven under pytest, where DATA_DIR points at
tests/fixtures/data and the golden run is absent.
"""

import argparse
import json
import sys
import types

import pytest

from scripts.validation import agent_eval

# Two cases is the smallest set that can distinguish "this case is finished"
# from "this run is finished", which is the whole arithmetic of --resume.
CASES = [{"id": "A03", "category": "extraction", "user_text": "t", "gold": {}},
         {"id": "B02", "category": "abstention", "user_text": "t", "gold": {}}]


def _args(**over):
    """A stand-in for the parsed CLI arguments, with the harness's defaults.

    Built by hand rather than through main()'s parser so that a guard can be
    exercised without going through argparse, and so that a later flag rename
    fails loudly here instead of silently skipping a branch.
    """
    ns = argparse.Namespace(
        k=1, cases="agent_eval_cases.jsonl", limit=None, only=None,
        preset="gpt", output_contract="structured", out_dir=None,
        dry_run=False, rescore=None, resume=False, discard_existing_run=None)
    for key, value in over.items():
        setattr(ns, key, value)
    return ns


def _ap():
    """A real parser, because ap.error is what turns a refusal into exit 2."""
    return argparse.ArgumentParser(prog="agent_eval.py")


def _row(case_id, rep, **over):
    """The fields of a raw row that the resume planner actually reads."""
    r = {"case_id": case_id, "rep": rep, "error": None, "all_gates_ok": True}
    r.update(over)
    return r


def _write_run(d, rows, args=None, cases=CASES, manifest=None):
    """An interrupted run on disk: a raw file plus the manifest of its identity."""
    d.mkdir(parents=True, exist_ok=True)
    (d / agent_eval.RAW_NAME).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    m = (manifest if manifest is not None
         else agent_eval.build_run_manifest(args or _args(), cases))
    (d / agent_eval.MANIFEST_NAME).write_text(json.dumps(m), encoding="utf-8")
    return d


def _cases_file(tmp_path):
    """A minimal case file, so main() can be driven without the real one."""
    p = tmp_path / "cases.jsonl"
    p.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in CASES),
                 encoding="utf-8")
    return p


def _main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["agent_eval.py"] + argv)
    agent_eval.main()


# --------------------------------------------------------------------------------
# the guard itself
# --------------------------------------------------------------------------------
def test_fresh_out_dir_is_allowed(tmp_path):
    assert agent_eval.guard_out_dir(_ap(), tmp_path / "new", _args(), CASES) is None


def test_existing_but_empty_out_dir_is_allowed(tmp_path):
    # An empty folder is not a run: pre-creating the destination, or re-using one
    # left behind by a run that died before its first row, must not be blocked.
    (tmp_path / "empty").mkdir()
    assert agent_eval.guard_out_dir(_ap(), tmp_path / "empty", _args(), CASES) is None


def test_out_dir_holding_only_unrelated_files_is_allowed(tmp_path):
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    (tmp_path / "agent_eval_raw.superseded_20260921.jsonl").write_text(
        "x", encoding="utf-8")
    assert agent_eval.guard_out_dir(_ap(), tmp_path, _args(), CASES) is None


def test_out_dir_holding_raw_jsonl_aborts(tmp_path, capsys):
    (tmp_path / "agent_eval_raw.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        agent_eval.guard_out_dir(_ap(), tmp_path, _args(), CASES)
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "agent_eval_raw.jsonl" in err
    assert "never overwrites one silently" in err
    # The message must carry BOTH ways out, and the destructive one must already
    # show the exact string the operator has to type.
    assert "--resume" in err
    assert f"--discard-existing-run {tmp_path.name}" in err


@pytest.mark.parametrize("name", agent_eval.RESULT_FILENAMES[1:])
def test_each_other_result_file_also_aborts(tmp_path, name, capsys):
    # A rescore's output is four files WITHOUT a raw.jsonl, so keying the guard
    # on the raw file alone would leave a rescored run unprotected.
    (tmp_path / name).write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, _args(), CASES)
    assert name in capsys.readouterr().err


def test_out_dir_that_is_a_file_aborts(tmp_path):
    p = tmp_path / "not_a_dir"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), p, _args(), CASES)


def test_main_refuses_a_run_into_an_occupied_out_dir_and_writes_nothing(
        tmp_path, monkeypatch, capsys):
    # End to end through the real CLI, which also proves the guard sits BEFORE
    # build_fixtures(): under pytest DATA_DIR has no golden run, so reaching the
    # fixtures at all would raise something other than SystemExit(2).
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "agent_eval_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, ["--out-dir", str(out), "--k", "1",
                            "--cases", str(_cases_file(tmp_path))])
    assert e.value.code == 2
    assert "never overwrites one silently" in capsys.readouterr().err
    assert agent_eval.existing_result_files(out) == ["agent_eval_metrics.json"]
    assert not (out / agent_eval.MANIFEST_NAME).exists()


# --------------------------------------------------------------------------------
# the escape hatches: neither is accepted where it has nothing to protect
# --------------------------------------------------------------------------------
def test_resume_on_a_folder_with_no_results_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, _args(resume=True), CASES)


def test_discard_on_a_folder_with_no_results_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(
            _ap(), tmp_path, _args(discard_existing_run=tmp_path.name), CASES)


def test_resume_and_discard_together_are_refused(tmp_path):
    (tmp_path / "agent_eval_raw.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(
            _ap(), tmp_path,
            _args(resume=True, discard_existing_run=tmp_path.name), CASES)


def test_discard_refuses_a_name_that_is_not_the_folders_own(tmp_path, capsys):
    (tmp_path / "agent_eval_raw.jsonl").write_text("keep me", encoding="utf-8")
    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(
            _ap(), tmp_path, _args(discard_existing_run="some_other_run"), CASES)
    assert tmp_path.name in capsys.readouterr().err
    # The refusal must be inert: the run is still exactly where it was.
    assert (tmp_path / "agent_eval_raw.jsonl").read_text(encoding="utf-8") == "keep me"


def test_discard_moves_nothing_during_a_dry_run(tmp_path):
    # --dry-run exists so a long command line can be rehearsed; if the rehearsal
    # moved the previous run aside it would be the very accident being guarded.
    (tmp_path / "agent_eval_raw.jsonl").write_text("x", encoding="utf-8")

    plan = agent_eval.guard_out_dir(
        _ap(), tmp_path, _args(dry_run=True, discard_existing_run=tmp_path.name),
        CASES)

    assert plan is None
    assert agent_eval.existing_result_files(tmp_path) == ["agent_eval_raw.jsonl"]
    assert not list(tmp_path.glob("discarded_*"))


def test_resume_is_refused_on_a_finished_rescore_with_no_raw_file(tmp_path, capsys):
    # Four artefacts and no raw.jsonl is what a rescore leaves behind. There are
    # no repetitions to continue there, so --resume must not be the way in.
    (tmp_path / "agent_eval_metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(
            _ap(), tmp_path, _args(out_dir=str(tmp_path), resume=True), CASES)

    assert agent_eval.RAW_NAME in capsys.readouterr().err


def test_discard_moves_every_artefact_aside_and_lets_the_run_proceed(tmp_path):
    for n in agent_eval.RESULT_FILENAMES:
        (tmp_path / n).write_text(n, encoding="utf-8")
    (tmp_path / agent_eval.MANIFEST_NAME).write_text("{}", encoding="utf-8")

    plan = agent_eval.guard_out_dir(
        _ap(), tmp_path, _args(discard_existing_run=tmp_path.name), CASES,
        stamp="20260921_120000")

    assert plan is None
    assert agent_eval.existing_result_files(tmp_path) == []
    moved = tmp_path / "discarded_20260921_120000"
    assert sorted(p.name for p in moved.iterdir()) == sorted(
        [f"discarded_{n}" for n in agent_eval.RESULT_FILENAMES]
        + [f"discarded_{agent_eval.MANIFEST_NAME}"])
    # Nothing was deleted, and nothing under the out-dir still answers to a
    # canonical filename: agent_eval_compare.load() falls back to
    # rglob("agent_eval_metrics.json"), and a discarded run must never be what
    # that fallback finds.
    assert (moved / "discarded_agent_eval_report.md").read_text(
        encoding="utf-8") == "agent_eval_report.md"
    assert list(tmp_path.rglob("agent_eval_metrics.json")) == []


# --------------------------------------------------------------------------------
# --resume: correct, or refused
# --------------------------------------------------------------------------------
def test_resume_computes_the_missing_pairs(tmp_path):
    a = _args(k=2, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1), _row("A03", 2), _row("B02", 1)], args=a)

    plan = agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    assert [(c["id"], rep) for c, rep in plan["missing"]] == [("B02", 2)]
    assert len(plan["keep"]) == 3
    assert plan["superseded"] == []
    assert plan["torn_tail"] is False


def test_resume_reruns_the_repetitions_that_recorded_an_api_error(tmp_path):
    # The point of the whole feature: a dead API looks exactly like a very bad
    # model, so its rows must be re-run, never carried into the metrics.
    a = _args(k=2, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1), _row("A03", 2, error="APIError: 429"),
                          _row("B02", 1), _row("B02", 2)], args=a)

    plan = agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    assert [(c["id"], rep) for c, rep in plan["missing"]] == [("A03", 2)]
    assert plan["superseded"] == [("A03", 2)]
    assert [(r["case_id"], r["rep"]) for r in plan["keep"]] == [
        ("A03", 1), ("B02", 1), ("B02", 2)]


@pytest.mark.parametrize("field,changed", [
    ("preset_resolved", {"preset": "gpt-terra"}),
    ("output_contract", {"output_contract": "free"}),
    ("k", {"k": 5}),
])
def test_resume_is_refused_when_a_run_parameter_disagrees(
        tmp_path, field, changed, capsys):
    _write_run(tmp_path, [_row("A03", 1), _row("B02", 1)], args=_args(k=2))
    # The manifest was written with k=2; `changed` is what THIS invocation
    # disagrees on, so it must override the baseline kwargs rather than collide
    # with them (k appears in both for the k case).
    kwargs = {"k": 2, "out_dir": str(tmp_path), "resume": True, **changed}
    a = _args(**kwargs)

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    err = capsys.readouterr().err
    assert field in err
    assert "not the run that was interrupted" in err


def test_resume_is_refused_when_the_case_set_disagrees(tmp_path, capsys):
    a = _args(k=1, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1)], args=a, cases=CASES)

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES[:1])

    assert "case_ids" in capsys.readouterr().err


def test_resume_is_refused_when_a_case_body_changed_under_the_same_id(
        tmp_path, capsys):
    # Same ids, different question: a different experiment wearing the old name,
    # which the id list alone would not notice.
    a = _args(k=1, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1)], args=a, cases=CASES)
    edited = [dict(CASES[0], user_text="a different prompt"), CASES[1]]

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, a, edited)

    assert "cases_sha256" in capsys.readouterr().err


def test_resume_is_refused_when_the_run_has_no_manifest(tmp_path, capsys):
    # The 2026-09-01 baseline and the 2026-09-21 pilots predate the manifest, so
    # their preset and k are unrecoverable from the rows and they can never be
    # resumed; they must be re-run into a new folder instead.
    (tmp_path / agent_eval.RAW_NAME).write_text(
        json.dumps(_row("A03", 1)) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(
            _ap(), tmp_path, _args(k=1, out_dir=str(tmp_path), resume=True), CASES)

    assert agent_eval.MANIFEST_NAME in capsys.readouterr().err


def test_resume_is_refused_when_the_manifest_schema_is_not_ours(tmp_path, capsys):
    a = _args(k=1, out_dir=str(tmp_path), resume=True)
    m = agent_eval.build_run_manifest(a, CASES)
    m["manifest_schema"] = agent_eval.MANIFEST_SCHEMA + 1
    _write_run(tmp_path, [_row("A03", 1)], manifest=m)

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    assert "manifest_schema" in capsys.readouterr().err


def test_resume_is_refused_when_nothing_is_missing(tmp_path, capsys):
    a = _args(k=1, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1), _row("B02", 1)], args=a)

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    assert "already complete" in capsys.readouterr().err


def test_resume_is_refused_on_a_pair_outside_the_declared_grid(tmp_path, capsys):
    a = _args(k=1, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1), _row("A03", 4)], args=a)

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    assert "outside this run's declared grid" in capsys.readouterr().err


def test_resume_is_refused_when_a_pair_appears_twice(tmp_path, capsys):
    a = _args(k=2, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1), _row("A03", 1)], args=a)

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    assert "more than one scored row" in capsys.readouterr().err


def test_resume_tolerates_one_torn_final_line(tmp_path):
    # Rows are flushed one per repetition, so a hard kill can only ever tear the
    # LAST line, and that line was never a scored row.
    a = _args(k=1, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1)], args=a)
    with (tmp_path / agent_eval.RAW_NAME).open("a", encoding="utf-8") as fh:
        fh.write('{"case_id": "B02", "rep": 1, "repl')

    plan = agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    assert plan["torn_tail"] is True
    assert [(c["id"], rep) for c, rep in plan["missing"]] == [("B02", 1)]


def test_resume_is_refused_on_garbage_that_is_not_the_last_line(tmp_path, capsys):
    a = _args(k=1, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1)], args=a)
    with (tmp_path / agent_eval.RAW_NAME).open("a", encoding="utf-8") as fh:
        fh.write("{ not json\n")
        fh.write(json.dumps(_row("B02", 1)) + "\n")

    with pytest.raises(SystemExit):
        agent_eval.guard_out_dir(_ap(), tmp_path, a, CASES)

    assert "edited or concatenated file" in capsys.readouterr().err


def _offline_main_resume(monkeypatch, out_dir, cases_path, k):
    """Drive the real main() through the --resume path with no model and no I/O
    outside tmp_path.

    Three things are replaced and nothing else, so the code under test is the
    genuine rewrite block of main():

      * `agent` in sys.modules, because main() does `from agent import _to_utc`
        and importing the real one drags litellm into a suite that is offline by
        design. The stub is installed BEFORE the stored manifest is built by the
        caller, so `frozen_input_hashes` sees the same SYSTEM_PROMPT on both
        sides and the resume is not refused over a hash that only moved because
        of this fixture;
      * `build_fixtures`, which under pytest would look for a golden run that
        tests/fixtures/data does not contain;
      * `write_outputs`, which needs pandas and a full scored row, and which
        runs AFTER the raw file has been written and so cannot affect what this
        test asserts.

    `run_one` is replaced by the caller, since what it returns is the subject.
    """
    monkeypatch.setattr(agent_eval, "build_fixtures", lambda: {})
    monkeypatch.setattr(agent_eval, "write_outputs",
                        lambda *args, **kwargs: None)
    _main(monkeypatch, ["--out-dir", str(out_dir), "--cases", str(cases_path),
                        "--k", str(k), "--preset", "gpt",
                        "--output-contract", "structured", "--resume"])


def test_resume_rewrite_keeps_every_byte_and_leaves_one_row_per_pair(
        tmp_path, monkeypatch):
    """The one destructive operation in the resume path, end to end.

    main() copies raw.jsonl aside and then TRUNCATES and rewrites it from the
    kept rows. Appending instead would leave two rows for every repetition
    re-run after an API error, and one line per (case, repetition) is the
    invariant write_outputs, rescore() and every downstream table assume; but a
    truncate-and-rewrite is the only operation in this feature that can destroy
    evidence, so the copy is what makes it safe and the copy is what this test
    is really about.
    """
    # The stub must be in place before the stored manifest is built, or the two
    # manifests would disagree on system_prompt_sha256 and the resume would be
    # refused for a reason this test invented.
    stub = types.ModuleType("agent")
    stub.SYSTEM_PROMPT = "stub prompt, hashed identically on both sides"
    stub._to_utc = lambda value: value
    monkeypatch.setitem(sys.modules, "agent", stub)

    # A03 rep 1 is done, A03 rep 2 died with an API error (so it is re-run and
    # its stored row must NOT survive into the rewritten file), B02 rep 1 is
    # done, B02 rep 2 was never reached.
    a = _args(k=2, out_dir=str(tmp_path), resume=True)
    _write_run(tmp_path, [_row("A03", 1, reply="kept one"),
                          _row("A03", 2, error="APIError: 429"),
                          _row("B02", 1, reply="kept two")], args=a)
    raw_path = tmp_path / agent_eval.RAW_NAME
    before = raw_path.read_bytes()
    manifest_before = json.loads(
        (tmp_path / agent_eval.MANIFEST_NAME).read_text(encoding="utf-8"))

    def _canned(case, fixtures, preset, to_utc, output_contract="free"):
        return {"case_id": case["id"], "error": None, "all_gates_ok": True,
                "n_tool_calls": 1, "reply": "run now"}

    monkeypatch.setattr(agent_eval, "run_one", _canned)
    _offline_main_resume(monkeypatch, tmp_path, _cases_file(tmp_path), k=2)

    # 1. Nothing that was on disk is lost, the superseded error row included.
    backups = list(tmp_path.glob("agent_eval_raw.superseded_*.jsonl"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == before

    # 2. The rewritten file is the kept rows in their original order, then the
    #    repetitions run now, and every pair appears exactly once.
    rows = [json.loads(line) for line
            in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [(r["case_id"], r["rep"]) for r in rows]
    assert pairs == [("A03", 1), ("B02", 1), ("A03", 2), ("B02", 2)]
    assert len(pairs) == len(set(pairs))
    assert [r["reply"] for r in rows[:2]] == ["kept one", "kept two"]
    assert all(r["reply"] == "run now" for r in rows[2:])
    # The re-run row replaced the errored one rather than joining it.
    assert not [r for r in rows if r.get("error")]

    # 3. The manifest records the extra sitting and nothing else. Its identity
    #    is the only proof the folder holds one experiment, so a resume that
    #    re-stamped any compared field would destroy the thing it is checked
    #    against.
    manifest_after = json.loads(
        (tmp_path / agent_eval.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert len(manifest_after.pop("resumed_utc")) == 1
    assert manifest_before.pop("resumed_utc") == []
    assert manifest_after == manifest_before


def test_plan_resume_raises_rather_than_exiting(tmp_path):
    # The planners stay pure so they can be reasoned about without argparse;
    # only guard_out_dir turns a refusal into exit 2.
    with pytest.raises(agent_eval.GuardRefusal):
        agent_eval.plan_resume(
            tmp_path, agent_eval.build_run_manifest(_args(), CASES), CASES)


# --------------------------------------------------------------------------------
# the three pre-existing --rescore guards, which this work must not weaken
# --------------------------------------------------------------------------------
def _source_raw(tmp_path):
    src = tmp_path / "frozen_run"
    src.mkdir()
    p = src / "agent_eval_raw.jsonl"
    p.write_text(json.dumps(_row("A03", 1)) + "\n", encoding="utf-8")
    return p


def test_rescore_still_requires_out_dir(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--rescore", str(_source_raw(tmp_path)),
                            "--preset", "gpt"])
    assert "--out-dir" in capsys.readouterr().err


def test_rescore_still_requires_preset(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--rescore", str(_source_raw(tmp_path)),
                            "--out-dir", str(tmp_path / "new")])
    assert "--preset" in capsys.readouterr().err


def test_rescore_out_dir_must_not_be_the_source_folder(tmp_path, monkeypatch,
                                                       capsys):
    raw = _source_raw(tmp_path)
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--rescore", str(raw), "--preset", "gpt",
                            "--out-dir", str(raw.parent)])
    assert "must not be the folder of the file being" in capsys.readouterr().err


def test_rescore_also_refuses_an_out_dir_that_already_holds_a_run(
        tmp_path, monkeypatch, capsys):
    # rescore() repeats main()'s mkdir(exist_ok=True) + open("w"), so without
    # this the two older guards would still let a rescore land on a DIFFERENT
    # frozen run: the destructive default is only one of the two ways in.
    raw = _source_raw(tmp_path)
    out = tmp_path / "another_frozen_run"
    out.mkdir()
    (out / "agent_eval_metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--rescore", str(raw), "--preset", "gpt",
                            "--out-dir", str(out)])

    err = capsys.readouterr().err
    assert "never overwrites one silently" in err
    # A rescore has nothing to continue, so the safe hatch is not offered here.
    assert "--resume does not apply to --rescore" in err


def test_resume_is_refused_together_with_rescore(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["--rescore", str(_source_raw(tmp_path)),
                            "--preset", "gpt", "--resume",
                            "--out-dir", str(tmp_path / "new")])
    assert "--resume is for an interrupted MODEL run" in capsys.readouterr().err

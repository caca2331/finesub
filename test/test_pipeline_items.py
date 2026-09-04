"""The pipeline CLI's option surface: manifest rows, item building, defaults.

The runner itself is `test_scheduler.py`; nothing here schedules anything. What
these pin is the layer between them -- what a row may say, what a row becomes,
and that the CLI's defaults and the text describing them cannot drift apart.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
import sys
import threading
import time
import types

import pytest

from finesub import pipeline
from finesub.pipeline import (
    build_item,
    manifest_intake,
    manifest_snapshot,
    merge_item_options,
    profile_asr_workers,
    read_manifest,
)
from finesub.run_metadata import update_run_metadata
from finesub.scheduler import BatchItem, IntakePoll, ItemResult, run_batch


def _item(label: str, stages: dict) -> BatchItem:
    return BatchItem(label=label, stages=stages, payload=label)


def test_read_manifest_and_merge(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"source": "https://example.com/a", "language": "ja"}\n'
        "\n"
        '{"source": "data/b.wav", "stage": "raw-srt"}\n',
        encoding="utf-8",
    )
    rows = read_manifest(manifest)
    assert len(rows) == 2

    defaults = {"stage": "final-srt", "language": None, "model": "large-v3-turbo"}
    merged = [merge_item_options(row, defaults) for row in rows]
    assert merged[0]["language"] == "ja"
    assert merged[0]["stage"] == "final-srt"
    assert merged[1]["stage"] == "raw-srt"
    assert merged[1]["model"] == "large-v3-turbo"


def test_merge_rejects_unknown_keys_and_missing_source() -> None:
    with pytest.raises(ValueError, match="unknown manifest keys"):
        merge_item_options({"source": "x", "banana": 1}, {})
    with pytest.raises(ValueError, match="missing 'source'"):
        merge_item_options({"language": "ja"}, {})


def test_profile_asr_workers_runs_one_file_at_a_time() -> None:
    # Parallelism moved inside the file, so the asr bin is 1 regardless of the
    # profile mix -- that is what bounds live per-file state in one process.
    assert profile_asr_workers([{"gpu_tier": "high"}, {"gpu_tier": "standard"}]) == 1
    assert profile_asr_workers([{"gpu_tier": "high"}, {"gpu_tier": "entry"}]) == 1
    assert profile_asr_workers([{"gpu_tier": "high", "device": "cpu"}]) == 1
    assert profile_asr_workers([]) == 1


def test_read_manifest_rejects_bad_json(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"source": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_manifest(manifest)


def test_a_row_option_reaches_run_pipeline(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda *args, **kwargs: calls.append({"args": args, **kwargs}),
    )
    opts = {
        "source": str(source),
        "stage": "raw-srt",
        "asr_stabilize_profile": -1,
    }

    item = build_item(opts)
    item.stages["asr"](item.payload)

    assert calls[0]["asr_stabilize_profile"] == -1
    assert calls[0]["stage"] == "raw-srt"
    assert Path(calls[0]["args"][0]) == source


def test_first_pass_stage_timing_reaches_the_final_pass(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    metadata_path = tmp_path / "input-metadata.json"
    calls: list[dict[str, object]] = []

    def fake_run_pipeline(*args, **kwargs):
        calls.append({"args": args, **kwargs})
        if kwargs["stage"] == "raw-srt":
            update_run_metadata(
                metadata_path,
                {
                    "timing": {
                        "stages": {
                            "asr": {"status": "executed", "elapsed_sec": 4.0}
                        }
                    }
                },
            )
        return types.SimpleNamespace(metadata_json=metadata_path)

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
    item = build_item({"source": str(source), "stage": "final-srt"})

    payload = item.stages["asr"](item.payload)
    item.stages["llm"](payload)

    assert calls[1]["_prior_timing"]["asr"] == {
        "status": "executed",
        "elapsed_sec": 4.0,
    }


def test_manifest_intake_reads_only_terminated_appended_rows(tmp_path) -> None:
    """Appended rows join; a row without its newline waits; a malformed
    complete row is skipped with a warning, never crashing the batch."""


    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"source": "a.wav"}\n', encoding="utf-8")
    built: list[str] = []

    def build(opts):
        built.append(opts["source"])
        return _item(opts["source"], {})

    poll = manifest_intake(
        manifest, defaults={}, build=build, consumed_lines=1
    )
    first = poll()
    assert first.items == () and first.settled  # nothing appended yet

    with manifest.open("a", encoding="utf-8") as fh:
        fh.write('{"source": "b.wav"}\n')
        fh.write('not json\n')
        fh.write('{"source": "c.wav"}')  # no newline: still being written
    fresh = poll()
    assert built == ["b.wav"]  # the malformed row warned and skipped
    assert len(fresh.items) == 1
    # the half-written row is NOT "nothing new": the batch must not end here
    assert not fresh.settled

    with manifest.open("a", encoding="utf-8") as fh:
        fh.write("\n")  # the writer finished the row
    fresh = poll()
    assert built == ["b.wav", "c.wav"]
    assert len(fresh.items) == 1 and fresh.settled
    last = poll()
    assert last.items == () and last.settled  # nothing left


def test_the_startup_snapshot_and_the_intake_cursor_come_from_one_read(tmp_path, monkeypatch) -> None:
    """Reviewer 2026-08-30 P1-1: the CLI parsed the manifest, did its heavy
    imports, then re-read the file for the intake cursor -- a row appended in
    between was counted as consumed but never built, so this batch could never
    run it. One read must feed both halves."""


    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"source": "a.wav"}\n', encoding="utf-8")
    real_read = Path.read_text
    reads = {"n": 0}

    def counting_read(self, *args, **kwargs):
        text = real_read(self, *args, **kwargs)
        if self == manifest:
            reads["n"] += 1
            # a row lands while the CLI is still importing the ASR stack
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write('{"source": "b.wav"}\n')
        return text

    monkeypatch.setattr(Path, "read_text", counting_read)
    rows, cursor = manifest_snapshot(manifest)
    monkeypatch.undo()
    assert reads["n"] == 1
    assert [r["source"] for r in rows] == ["a.wav"]
    assert cursor == 1  # NOT 2: the late row is unconsumed, so intake sees it

    built: list[str] = []
    poll = manifest_intake(
        manifest,
        defaults={},
        build=lambda opts: (built.append(opts["source"]), _item(opts["source"], {}))[1],
        consumed_lines=cursor,
    )
    assert built == [] and len(poll().items) == 1
    assert built == ["b.wav"]


def test_a_half_written_tail_keeps_a_drained_batch_open(tmp_path) -> None:
    """Reviewer 2026-08-30 P1-2: the last known item finished while a row was
    still being written. An unterminated tail is not "nothing new" -- the batch
    must wait for the newline and run that row, as the docs promise."""

    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"source": "a.wav"}\n{"source": "b.wav"}', encoding="utf-8")
    ran: list[str] = []
    polls = {"n": 0}

    inner = manifest_intake(
        manifest,
        defaults={},
        build=lambda opts: _item(
            opts["source"], {"llm": lambda p, s=opts["source"]: (ran.append(s), p)[1]}
        ),
        consumed_lines=1,  # only the first row was in the startup snapshot
    )

    def intake():
        polls["n"] += 1
        poll = inner()
        if polls["n"] == 1:
            # the writer finishes the row only after the batch has drained
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write("\n")
        return poll

    results = run_batch(
        [_item("a.wav", {"llm": lambda p: (ran.append("a.wav"), p)[1]})],
        intake=intake,
        intake_poll_seconds=0.5,
    )
    assert ran == ["a.wav", "b.wav"]
    assert [r.status for r in results] == ["done", "done"]


def test_a_torn_multibyte_read_is_not_taken_as_an_empty_manifest(tmp_path) -> None:
    """A poll landing mid-write of a CJK source path raises UnicodeDecodeError;
    unsettled, never 'nothing new' (which would end the batch)."""

    manifest = tmp_path / "m.jsonl"
    manifest.write_bytes(b'{"source": "a.wav"}\n{"source": "\xe4\xb8')
    poll = manifest_intake(
        manifest, defaults={}, build=lambda opts: _item(opts["source"], {}), consumed_lines=1
    )
    result = poll()
    assert result.items == ()
    assert not result.settled and "UnicodeDecodeError" in result.reason


def test_the_cli_help_cannot_drift_from_the_defaults(monkeypatch) -> None:
    """Reviewer 2026-08-30 P2: the runner description hardcoded llm(1) after
    the default became 2. Defaults and the text that names them must agree."""

    import argparse

    from finesub.pipeline import DEFAULT_WORKERS, parse_args

    captured: dict = {}

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise SystemExit(0)  # stop before it reads the real argv

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(SystemExit):
        parse_args()
    parser = captured["parser"]
    parser_defaults = {action.dest: action.default for action in parser._actions}
    # The runner's own knobs: these have no backend to defer to, so the CLI is
    # where their default lives and the description must agree with it.
    assert parser_defaults["max_parallel_tasks"] == DEFAULT_WORKERS["llm"]
    assert parser_defaults["download_workers"] == DEFAULT_WORKERS["download"]
    # `--llm-parallel-windows` is NOT one of them: it is a `run_pipeline`
    # parameter, so the CLI says nothing and the signature answers. Asserting
    # a value here again would put the copy back (see test_option_defaults).
    assert parser_defaults["llm_parallel_windows"] is None
    assert f"llm({DEFAULT_WORKERS['llm']})" in parser.description
    assert f"download({DEFAULT_WORKERS['download']})" in parser.description


def test_one_source_goes_through_the_same_runner(tmp_path, monkeypatch) -> None:
    """The unification (owner 2026-08-30): N=1 is not a second code path. It
    reaches `run_batch` like any other run -- one worker per bin, no status log
    unless asked -- and only its presentation differs."""

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")
    seen: dict = {}

    def fake_run_batch(items, **kwargs):
        seen["items"] = items
        seen["kwargs"] = kwargs
        return [ItemResult(label=items[0].label, status="done")]

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["finesub", str(clip)])
    assert pipeline.main() == 0

    assert len(seen["items"]) == 1
    assert seen["kwargs"]["workers"] == {"download": 1, "asr": 1, "llm": 1}
    assert seen["kwargs"]["status_path"] is None  # nobody to read it
    # The foreground presentation rides in as hooks, not as a mode flag.
    assert callable(seen["kwargs"]["item_reporter"])
    assert callable(seen["kwargs"]["on_item_error"])


def test_several_sources_get_the_runner_defaults_and_a_status_log(tmp_path, monkeypatch) -> None:
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"")
    b.write_bytes(b"")
    seen: dict = {}

    def fake_run_batch(items, **kwargs):
        seen["items"] = items
        seen["kwargs"] = kwargs
        return [ItemResult(label=item.label, status="done") for item in items]

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["finesub", str(a), str(b), "--batch-id", "x"])
    assert pipeline.main() == 0

    assert [item.label for item in seen["items"]] == ["a.wav", "b.wav"]
    assert seen["kwargs"]["workers"]["llm"] == pipeline.DEFAULT_WORKERS["llm"]
    assert seen["kwargs"]["status_path"].parts[-2:] == ("x", "batch-status.jsonl")
    # A batch keeps the runner's shared-terminal failure lines, but every item
    # gets its own run log -- it is the run nobody watched line by line.
    assert "on_item_error" not in seen["kwargs"]
    assert callable(seen["kwargs"]["item_reporter"])


def test_a_name_cannot_serve_several_sources(tmp_path, monkeypatch, capsys) -> None:
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"")
    b.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["finesub", str(a), str(b), "--name", "one"])
    assert pipeline.main() == 2
    assert "cannot serve several sources" in capsys.readouterr().err


def test_an_asr_option_is_settable_per_row(tmp_path, monkeypatch) -> None:
    """The row whitelist is derived from `run_pipeline`, so options nobody
    remembered to copy into a second parser (--qwen-verify, --word,
    --lang-redecode, --split-length-scale, --vad-silero-assist) are reachable
    per item for the first time."""

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    calls: list[dict] = []
    monkeypatch.setattr(
        pipeline, "run_pipeline", lambda *args, **kwargs: calls.append(kwargs)
    )
    for key in ("qwen_verify", "word", "lang_redecode", "vad_silero_assist"):
        assert key in pipeline.ALLOWED_ITEM_KEYS

    item = build_item({"source": str(source), "qwen_verify": "on", "word": True})
    item.stages["asr"](item.payload)
    assert calls[0]["qwen_verify"] == "on" and calls[0]["word"] is True


def test_an_option_a_row_leaves_out_falls_back_to_run_pipeline_itself(tmp_path, monkeypatch) -> None:
    """Pass-through, not a restatement: the item builder used to carry its own
    copy of every default, which is how one option came to have three."""

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    calls: list[dict] = []
    monkeypatch.setattr(
        pipeline, "run_pipeline", lambda *args, **kwargs: calls.append(kwargs)
    )
    item = build_item({"source": str(source)})
    item.stages["asr"](item.payload)
    assert "llm_parallel_windows" not in calls[0]
    assert "llm_retrieval" not in calls[0]


def test_two_sources_that_would_write_the_same_output_are_refused(tmp_path, monkeypatch, capsys) -> None:
    """Same stem, different directories: both derive out/a/a.srt. The first
    item's artifacts would then look like the second's finished work (stages
    skip on existence), so the second would report success holding the first's
    subtitles (reviewer 2026-08-30 P1)."""

    for directory in ("one", "two"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "a.wav").write_bytes(b"")
    monkeypatch.setattr(
        sys, "argv", ["finesub", str(tmp_path / "one" / "a.wav"), str(tmp_path / "two" / "a.wav")]
    )
    assert pipeline.main() == 2
    assert "would both write" in capsys.readouterr().err


def test_the_same_source_listed_twice_is_refused(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["finesub", str(source), str(source)])
    assert pipeline.main() == 2
    assert "listed twice" in capsys.readouterr().err


def test_an_option_that_names_one_run_is_refused_for_several_sources(tmp_path, monkeypatch, capsys) -> None:
    """`-o` used to be copied into every item as a default; with several
    sources it must be a manifest-row key, not an invocation-wide one."""

    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"")
    b.write_bytes(b"")
    monkeypatch.setattr(
        sys, "argv", ["finesub", str(a), str(b), "-o", str(tmp_path / "one.srt")]
    )
    assert pipeline.main() == 2
    err = capsys.readouterr().err
    assert "--output" in err and "per manifest row" in err

    # ...while one source keeps it, and a row may always carry its own.
    monkeypatch.setattr(sys, "argv", ["finesub", str(a), "-o", str(tmp_path / "one.srt")])
    args = pipeline.parse_args()
    assert pipeline._defaults_from_args(args, single=True)["output"] == str(tmp_path / "one.srt")
    assert "output" not in pipeline._defaults_from_args(args, single=False)


def test_the_environment_log_level_still_reaches_a_run(tmp_path, monkeypatch) -> None:
    """FINESUB_LOG_LEVEL only applies when nobody named a level, so resolving
    `args.log_level or "normal"` early silently disabled it (reviewer P2)."""

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"")
    seen: list[str] = []
    monkeypatch.setattr(pipeline, "run_batch", lambda items, **kw: [ItemResult(label="x", status="done")])
    monkeypatch.setenv("FINESUB_LOG_LEVEL", "verbose")

    real = pipeline.terminal_reporter
    monkeypatch.setattr(
        pipeline,
        "terminal_reporter",
        lambda **kwargs: (seen.append(kwargs.get("level")), real(**kwargs))[1],
    )
    monkeypatch.setattr(sys, "argv", ["finesub", str(clip)])
    assert pipeline.main() == 0
    assert seen == ["verbose"]


def test_the_llm_cost_key_reads_the_item_s_own_stable_json(tmp_path, monkeypatch) -> None:
    """LPT sorts on segment count. Deriving the path from the SRT alone made a
    custom --output look in out/<stem>/ (wrong file) and an item with no
    explicit output score 0, dropping the whole batch back to arrival order."""

    import json as _json

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    custom = tmp_path / "elsewhere" / "z.srt"
    custom.parent.mkdir()
    (custom.parent / "z-stable.json").write_text(
        _json.dumps({"segments": [{}, {}, {}]}), encoding="utf-8"
    )
    item = build_item({"source": str(source), "output": str(custom), "stage": "final-srt"})
    assert item.llm_cost({"audio": source, "output": str(custom)}) == 3.0

    # ...and with no explicit output, the run's own default location.
    default_stable = Path("out") / "a" / "a-stable.json"
    monkeypatch.chdir(tmp_path)
    default_stable.parent.mkdir(parents=True)
    default_stable.write_text(_json.dumps({"segments": [{}, {}]}), encoding="utf-8")
    plain = build_item({"source": str(source), "stage": "final-srt"})
    assert plain.llm_cost({"audio": source, "output": None}) == 2.0


def test_an_appended_row_goes_through_the_same_claims_book(tmp_path) -> None:
    """Rows appended mid-run build their items straight from the intake, which
    used to bypass the startup-only collision check entirely: appending the
    source that is already running would silently reuse its artifacts
    (reviewer 2026-08-30 P1)."""

    from finesub.pipeline import OutputClaims

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("", encoding="utf-8")
    claims = OutputClaims()
    build_item({"source": str(source)}, claims=claims)  # the running item

    built: list[str] = []
    poll = manifest_intake(
        manifest,
        defaults={},
        build=lambda opts: (built.append(opts["source"]), build_item(opts, claims=claims))[1],
        consumed_lines=0,
    )
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"source": str(source)}) + "\n")
        fh.write(json.dumps({"source": str(tmp_path / "b.wav")}) + "\n")
    (tmp_path / "b.wav").write_bytes(b"")

    fresh = poll()
    # the duplicate was refused with a warning, the new one joined
    assert [item.label for item in fresh.items] == ["b.wav"]


def test_two_rows_sharing_one_artifact_dir_are_refused(tmp_path) -> None:
    """Different SRTs, one LLM artifact directory: two tasks would write one
    plan, one exchange log and one resume ledger."""

    from finesub.pipeline import OutputClaims

    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"")
    b.write_bytes(b"")
    shared = str(tmp_path / "artifacts")
    claims = OutputClaims()
    build_item({"source": str(a), "task_artifact_dir": shared}, claims=claims)
    with pytest.raises(ValueError, match="would both write"):
        build_item({"source": str(b), "task_artifact_dir": shared}, claims=claims)


def test_output_paths_are_compared_case_insensitively_on_windows(tmp_path) -> None:
    from finesub.pipeline import OutputClaims

    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"")
    b.write_bytes(b"")
    claims = OutputClaims()
    build_item({"source": str(a), "output": str(tmp_path / "OUT" / "x.srt")}, claims=claims)
    if os.path.normcase("A") == "a":  # Windows / case-insensitive filesystems
        with pytest.raises(ValueError, match="would both write"):
            build_item({"source": str(b), "output": str(tmp_path / "out" / "x.srt")}, claims=claims)


def test_a_url_claims_its_destination_once_the_download_resolves_it(tmp_path, monkeypatch) -> None:
    """Two spellings of one video collide only after the id is resolved, so
    that claim is made in the download stage -- failing that item alone."""

    from finesub.stages import PipelinePaths
    from finesub.pipeline import OutputClaims

    srt = tmp_path / "out" / "vid" / "vid.srt"
    paths = PipelinePaths(
        vocal_audio=srt.with_name("vid-vocal.ogg"),
        vad_json=srt.with_name("vid-vad.json"),
        vad_energy_npz=srt.with_name("vid-vad.npz"),
        aligned_json=srt.with_name("vid-aligned.json"),
        stable_json=srt.with_name("vid-stable.json"),
        raw_srt=srt.with_name("vid-raw.srt"),
        translated_srt=srt.with_name("vid-translated.srt"),
        final_srt=srt,
        task_artifact_dir=srt.with_name("vid.llm-artifacts"),
        metadata_json=srt.with_name("vid-metadata.json"),
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_url_input",
        lambda source, **kwargs: (tmp_path / "vid.ogg", paths, None, ""),
    )
    claims = OutputClaims()
    first = build_item({"source": "https://example.test/v?a=1"}, claims=claims)
    second = build_item({"source": "https://example.test/v"}, claims=claims)
    first.stages["download"](None)  # resolves and claims
    with pytest.raises(ValueError, match="would both write"):
        second.stages["download"](None)


def test_a_batch_wide_task_summary_is_still_inherited(tmp_path, monkeypatch) -> None:
    """It is prompt context, not a write location (reviewer 2026-08-30 P2)."""

    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"")
    b.write_bytes(b"")
    monkeypatch.setattr(
        sys, "argv", ["finesub", str(a), str(b), "--task-summary", "同一批参考素材"]
    )
    args = pipeline.parse_args()
    assert pipeline._defaults_from_args(args, single=False)["task_summary"] == "同一批参考素材"


def test_every_item_of_a_batch_gets_its_own_run_log(tmp_path, monkeypatch) -> None:
    """`FileReporter` used to be single-source-only; `item_reporter` made that a
    product call rather than a structural one (docs/reporting.md §5.2)."""

    logs = tmp_path / "logs"
    monkeypatch.setattr(pipeline, "resolve_logs_dir", lambda: logs)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"")
    b.write_bytes(b"")

    def fake_run_batch(items, **kwargs):
        for item in items:  # one line each, as a stage would
            kwargs["item_reporter"](item.label).warning("x", "hello")
        return [ItemResult(label=item.label, status="done") for item in items]

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["finesub", str(a), str(b)])
    assert pipeline.main() == 0

    written = sorted(path.name for path in logs.glob("*.log"))
    assert len(written) == 2 and any("a" in name for name in written)
    assert all("hello" in path.read_text(encoding="utf-8") for path in logs.glob("*.log"))


def test_a_rejected_appended_row_does_not_poison_the_book(tmp_path) -> None:
    """A row that fails to build must leave nothing claimed, or the corrected
    row the user appends next is refused as a duplicate and the only way out is
    restarting the batch (reviewer 2026-08-31 P1)."""

    from finesub.pipeline import OutputClaims

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    claims = OutputClaims()

    with pytest.raises(ValueError, match="unknown stage"):
        build_item({"source": str(source), "stage": "typo-srt"}, claims=claims)
    # ...and the missing-file case, which fails after the claim is taken.
    with pytest.raises(FileNotFoundError):
        build_item({"source": str(tmp_path / "later.wav")}, claims=claims)

    # The corrected rows go through.
    assert build_item({"source": str(source), "stage": "raw-srt"}, claims=claims)
    (tmp_path / "later.wav").write_bytes(b"")
    assert build_item({"source": str(tmp_path / "later.wav")}, claims=claims)


def test_two_items_with_one_basename_get_two_labels_and_two_logs(tmp_path, monkeypatch) -> None:
    """Legal as long as the outputs differ -- but the label prefixes the item's
    terminal lines and names its run log, so it has to tell them apart."""

    from finesub.pipeline import OutputClaims

    for directory in ("one", "two"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "a.wav").write_bytes(b"")
    claims = OutputClaims()
    first = build_item(
        {"source": str(tmp_path / "one" / "a.wav"), "output": str(tmp_path / "1.srt")},
        claims=claims,
    )
    second = build_item(
        {"source": str(tmp_path / "two" / "a.wav"), "output": str(tmp_path / "2.srt")},
        claims=claims,
    )
    assert first.label != second.label
    assert first.label == "a.wav" and second.label == "two/a.wav"

    logs = tmp_path / "logs"
    monkeypatch.setattr(pipeline, "resolve_logs_dir", lambda: logs)
    with pipeline._batch_item_reporters("normal") as make:
        make(first.label).warning("x", "first")
        make(second.label).warning("x", "second")
    bodies = [path.read_text(encoding="utf-8") for path in logs.glob("*.log")]
    assert len(bodies) == 2
    assert any("first" in body and "second" not in body for body in bodies)


def _drain(path, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if path.exists():
            return True
        time.sleep(0.02)
    return False


def test_a_control_action_reorders_what_has_not_started(tmp_path) -> None:
    """End to end through the runner: the queued item named by a control line
    goes next, and one already under way is left alone."""

    started: list[str] = []
    release = threading.Event()
    seen_first = threading.Event()

    def slow(payload):
        started.append(payload)
        seen_first.set()
        release.wait(10)
        return payload

    def quick(payload):
        started.append(payload)
        return payload

    items = [_item("first", {"asr": slow})]
    items += [_item(name, {"asr": quick}) for name in ("plain1", "plain2", "urgent")]
    polls = {"n": 0}

    def intake():
        # Only act once the batch is genuinely in flight, so the action lands
        # while the others are still queued -- otherwise this races the work.
        seen_first.wait(10)
        polls["n"] += 1
        if polls["n"] == 1:
            return IntakePoll(actions=({"item": "urgent", "priority": 9},))
        release.set()
        return IntakePoll()

    run_batch(items, workers={"download": 1}, intake=intake, intake_poll_seconds=0.2)
    assert started[0] == "first"
    assert started[1] == "urgent", started


def test_dropping_an_item_that_has_not_started(tmp_path) -> None:
    ran: list[str] = []
    release = threading.Event()
    seen_first = threading.Event()
    polls = {"n": 0}

    def stage(name):
        def run(payload):
            ran.append(name)
            if name == "a":
                seen_first.set()
                release.wait(10)
            return payload

        return run

    def intake():
        seen_first.wait(10)
        polls["n"] += 1
        if polls["n"] == 1:
            return IntakePoll(actions=({"item": "b", "drop": True},))
        release.set()
        return IntakePoll()

    items = [_item(name, {"llm": stage(name)}) for name in ("a", "b", "c")]
    results = run_batch(
        items, workers={"llm": 1}, intake=intake, intake_poll_seconds=0.2
    )
    assert "b" not in ran
    assert [r.status for r in results] == ["done", "skipped", "done"]


def _registry(tmp_path, monkeypatch):
    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setattr(pipeline, "resolve_logs_dir", lambda: logs)
    return logs.parent / "batches.json"


def test_a_resume_cannot_fork_the_batch_under_a_new_id(tmp_path, monkeypatch, capsys) -> None:
    """Two resumes under different `--batch-id`s would both pass the liveness
    check -- each holding only its own new directory's lock -- and drive two
    sets of workers over one set of outputs."""

    _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch", "--batch-id", "fork"])
    assert pipeline.main() == 2
    assert "would fork it" in capsys.readouterr().err


def test_one_source_is_not_offered_a_control_channel_nobody_polls(
    tmp_path, monkeypatch, capsys
) -> None:
    """`_run_single` keeps the terminal for its one item and takes no intake,
    so the batch form's invitation to append instructions would have named a
    file that is never read."""

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")
    monkeypatch.setattr(sys, "argv", ["finesub", str(source), "--batch-id", "solo"])
    monkeypatch.setattr(
        pipeline,
        "run_batch",
        lambda items, **kwargs: [ItemResult(label=item.label, status="done") for item in items],
    )

    assert pipeline.main() == 0
    out = capsys.readouterr().out
    assert "queue ->" in out  # the view is still a record of the run
    assert "control.jsonl" not in out
    assert not (tmp_path / "batch" / "solo" / ".control-cursor").exists()


def _resumable_queue(tmp_path, monkeypatch, batch_id: str, rows: list[dict]):
    """A published queue for `batch_id`, registered as unfinished."""

    from finesub.batch_state import QUEUE_VIEW_FILENAME

    _registry(tmp_path, monkeypatch)
    queue = tmp_path / "batch" / batch_id / QUEUE_VIEW_FILENAME
    queue.parent.mkdir(parents=True)
    queue.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pipeline.record_batch(batch_id, queue, state="unfinished", items=len(rows))
    monkeypatch.setattr(pipeline, "DEFAULT_BATCH_ROOT", tmp_path / "batch")
    return queue


def test_a_flag_typed_on_the_resume_beats_what_the_batch_recorded(
    tmp_path, monkeypatch, capsys
) -> None:
    """A resumed row is a record the runner made, not a file the user wrote:
    a batch that recorded `device: cuda` is otherwise unresumable on a machine
    that lost its GPU, however loudly the user asks."""

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    _resumable_queue(
        tmp_path,
        monkeypatch,
        "b5",
        [{"source": str(source), "device": "cuda", "language": "ja", "_state": "queued"}],
    )

    seen: dict = {}

    def fake_run_batch(items, **kwargs):
        seen["rows"] = [dict(item.row or {}) for item in items]
        return [ItemResult(label=item.label, status="done") for item in items]

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch", "--device", "cpu"])
    assert pipeline.main() == 0
    assert seen["rows"][0]["device"] == "cpu"  # typed now
    assert seen["rows"][0]["language"] == "ja"  # not typed now: as recorded
    assert "device taken from this command line" in capsys.readouterr().out


def test_a_resume_that_says_nothing_keeps_every_recorded_option(
    tmp_path, monkeypatch
) -> None:
    """The default stays fidelity: `--language ja` from the original run must
    not evaporate because the resume did not repeat it."""

    source = tmp_path / "a.wav"
    source.write_bytes(b"")
    _resumable_queue(
        tmp_path,
        monkeypatch,
        "b6",
        [{"source": str(source), "language": "ja", "model": "large-v3", "_state": "queued"}],
    )

    seen: dict = {}

    def fake_run_batch(items, **kwargs):
        seen["rows"] = [dict(item.row or {}) for item in items]
        return [ItemResult(label=item.label, status="done") for item in items]

    monkeypatch.setattr(pipeline, "run_batch", fake_run_batch)
    monkeypatch.setattr(sys, "argv", ["finesub", "--resume-batch"])
    assert pipeline.main() == 0
    assert seen["rows"][0]["language"] == "ja"
    assert seen["rows"][0]["model"] == "large-v3"


def test_extending_the_stage_on_a_resume_says_what_it_revives(
    tmp_path, monkeypatch, capsys
) -> None:
    """Past raw-srt the new work spends LLM quota, and it lands on items that
    were already finished. Never silently."""

    done, pending = tmp_path / "done.wav", tmp_path / "pending.wav"
    done.write_bytes(b"")
    pending.write_bytes(b"")
    _resumable_queue(
        tmp_path,
        monkeypatch,
        "b7",
        [
            {"source": str(done), "stage": "raw-srt", "_state": "done"},
            {"source": str(pending), "stage": "raw-srt", "_state": "queued"},
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "run_batch",
        lambda items, **kwargs: [ItemResult(label=item.label, status="done") for item in items],
    )
    monkeypatch.setattr(
        sys, "argv", ["finesub", "--resume-batch", "--stage", "final-srt"]
    )
    assert pipeline.main() == 0
    out = capsys.readouterr().out
    assert "1 finished item(s) now go on to final-srt" in out


def test_a_manifest_row_still_beats_the_command_line(tmp_path) -> None:
    """The resume rule does not leak into the manifest contract."""

    row = {"source": "a.wav", "language": "ja"}
    assert merge_item_options(row, {"language": "en"})["language"] == "ja"
    # ...and the override is opt-in, per key.
    assert merge_item_options(row, {"language": "en"}, override={"language"})["language"] == "en"
    assert merge_item_options(row, {"language": "en"}, override={"model"})["language"] == "ja"

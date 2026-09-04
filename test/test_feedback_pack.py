"""The deterministic half of `agent-tasks/feedback-pack`.

Two things are worth pinning: what may never travel (the privacy rules the
SKILL asks the agent to read out loud), and the ledger key, whose whole reason
for existing is that a path alone would lock a user out of contributing an
improved subtitle.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "agent-tasks" / "feedback-pack" / "scripts" / "pack.py"


@pytest.fixture(scope="module")
def pack_module():
    spec = importlib.util.spec_from_file_location("feedback_pack", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["feedback_pack"] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop("feedback_pack", None)


def _task_dir(tmp_path: Path, stem: str = "input") -> Path:
    directory = tmp_path / "out" / stem
    directory.mkdir(parents=True)
    (directory / f"{stem}.srt").write_text("1\n", encoding="utf-8")
    (directory / f"{stem}-raw.srt").write_text("1\n", encoding="utf-8")
    (directory / f"{stem}-annotated.csv").write_text("a|b\n", encoding="utf-8")
    (directory / f"{stem}-vocal.ogg").write_bytes(b"audio")
    (directory / f"{stem}-vad-energy.npz").write_bytes(b"npz")
    (directory / f"{stem}.mp4").write_bytes(b"video")
    (directory / ".env").write_text("KEY=secret", encoding="utf-8")
    artifacts = directory / f"{stem}.llm-artifacts" / "exchanges"
    artifacts.mkdir(parents=True)
    (artifacts / "0001.md").write_text("exchange", encoding="utf-8")
    return directory


def test_corpus_mode_never_carries_audio_and_debug_mode_does(
    pack_module, tmp_path: Path
) -> None:
    directory = _task_dir(tmp_path)
    task = pack_module.Task(directory / "input.srt")

    debug = {path.name for path in pack_module.task_files(task, "debug", with_source=False)}
    corpus = {path.name for path in pack_module.task_files(task, "corpus", with_source=False)}

    assert "input-vocal.ogg" in debug
    assert "input-vocal.ogg" not in corpus
    assert "0001.md" in debug and "0001.md" in corpus
    assert "input-annotated.csv" in corpus


def test_secrets_and_source_media_stay_out_of_every_mode(
    pack_module, tmp_path: Path
) -> None:
    directory = _task_dir(tmp_path)
    task = pack_module.Task(directory / "input.srt")

    for mode in ("debug", "corpus"):
        names = {
            path.name for path in pack_module.task_files(task, mode, with_source=False)
        }
        assert ".env" not in names, mode
        # Hundreds of megabytes carrying the same speech as the vocal track.
        assert "input.mp4" not in names, mode

    asked = {
        path.name for path in pack_module.task_files(task, "debug", with_source=True)
    }
    assert "input.mp4" in asked
    assert ".env" not in asked, "--with-source is about size, never about secrets"


def test_only_this_task_travels_when_two_share_a_directory(
    pack_module, tmp_path: Path
) -> None:
    """A run given an explicit `-o` writes its artifacts as siblings.

    So the directory is not the task, and sweeping it whole would pack a
    stranger's subtitles under this task's name.
    """

    shared = tmp_path / "results"
    shared.mkdir()
    for stem in ("mine", "theirs"):
        (shared / f"{stem}.srt").write_text(stem, encoding="utf-8")
        (shared / f"{stem}-raw.srt").write_text(stem, encoding="utf-8")
    task = pack_module.Task(shared / "mine.srt")

    names = {path.name for path in pack_module.task_files(task, "debug", with_source=False)}

    assert names == {"mine.srt", "mine-raw.srt"}


def test_a_neighbour_whose_name_merely_starts_the_same_is_not_ours(
    pack_module, tmp_path: Path
) -> None:
    """`a` must not claim `abc-raw.srt`: the stem is followed by a separator."""

    shared = tmp_path / "results"
    shared.mkdir()
    (shared / "a.srt").write_text("mine", encoding="utf-8")
    (shared / "a-raw.srt").write_text("mine", encoding="utf-8")
    (shared / "abc-raw.srt").write_text("theirs", encoding="utf-8")
    task = pack_module.Task(shared / "a.srt")

    names = {path.name for path in pack_module.task_files(task, "debug", with_source=False)}

    assert names == {"a.srt", "a-raw.srt"}


def test_the_config_excerpt_is_valid_toml(pack_module) -> None:
    """It is named `.toml`, so someone will load it with `tomllib`."""

    import tomllib

    rendered = pack_module._toml_scalar
    assert rendered(True) == "true" and rendered(False) == "false"
    assert rendered(0.35) == "0.35" and rendered(3) == "3"
    # Nested tables and arrays are dropped, not half-rendered.
    assert rendered({"a": 1}) is None and rendered([1, 2]) is None

    body = "\n".join(
        [
            "[vad]",
            f"silero_assist = {rendered(False)}",
            f"note = {rendered('引号\"与中文')}",
        ]
    )
    assert tomllib.loads(body) == {
        "vad": {"silero_assist": False, "note": '引号"与中文'}
    }


def test_the_scratch_decode_never_travels(pack_module, tmp_path: Path) -> None:
    """A lossless copy of the source, carrying nothing the vocal track lacks."""

    directory = _task_dir(tmp_path)
    (directory / "input-decoded.flac").write_bytes(b"x" * 64)
    task = pack_module.Task(directory / "input.srt")

    for mode in ("debug", "corpus"):
        names = {
            path.name for path in pack_module.task_files(task, mode, with_source=False)
        }
        assert "input-decoded.flac" not in names, mode
    asked = {
        path.name for path in pack_module.task_files(task, "debug", with_source=True)
    }
    assert "input-decoded.flac" not in asked


def test_a_run_log_belongs_to_one_task_only(pack_module, tmp_path: Path) -> None:
    """The dual of the artifact-side neighbour test, on the log side.

    `run-*<label>*.log` matched every log with the label anywhere in its name:
    task `a` collected all of them, and task `input` walked off with
    `my-input`'s. In a bundle the user is about to send someone, that is a
    stranger's log going out under their name.
    """

    logs = tmp_path / "logs"
    logs.mkdir()
    for stem in ("input", "my-input", "input-2", "a"):
        (logs / f"run-20260903-101500-{stem}.log").write_text(stem, encoding="utf-8")
    directory = _task_dir(tmp_path)
    task = pack_module.Task(directory / "input.srt")

    from finesub import paths as finesub_paths

    original = finesub_paths.resolve_logs_dir
    finesub_paths.resolve_logs_dir = lambda: logs
    try:
        found = {path.read_text(encoding="utf-8") for path in pack_module.run_logs_for(task)}
    finally:
        finesub_paths.resolve_logs_dir = original

    assert found == {"input"}


def test_a_task_named_after_its_source_still_finds_its_log(
    pack_module, tmp_path: Path
) -> None:
    """`-o` names the subtitle; the log is named after the input.

    So the source is read back out of the run metadata rather than guessed --
    otherwise the file most worth having in a bug report is the one missing.
    """

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run-20260903-101500-episode12.log").write_text("wanted", encoding="utf-8")
    directory = tmp_path / "results"
    directory.mkdir()
    (directory / "final.srt").write_text("1\n", encoding="utf-8")
    (directory / "final-metadata.json").write_text(
        json.dumps({"source": "D:/media/episode12.mkv"}), encoding="utf-8"
    )
    task = pack_module.Task(directory / "final.srt")

    from finesub import paths as finesub_paths

    original = finesub_paths.resolve_logs_dir
    finesub_paths.resolve_logs_dir = lambda: logs
    try:
        found = [path.name for path in pack_module.run_logs_for(task)]
    finally:
        finesub_paths.resolve_logs_dir = original

    assert found == ["run-20260903-101500-episode12.log"]


def test_two_tasks_with_the_same_stem_get_separate_folders(pack_module) -> None:
    """A zip holds two members with one name and loses the first."""

    tasks = [
        pack_module.Task(Path("out/a/final.srt")),
        pack_module.Task(Path("out/b/final.srt")),
        pack_module.Task(Path("out/c/other.srt")),
    ]

    folders = pack_module.bundle_folders(tasks)

    assert len(set(folders.values())) == 3
    assert folders[Path("out/a/final.srt")] == "final"
    assert folders[Path("out/c/other.srt")] == "other"


def test_a_generated_folder_name_does_not_steal_a_real_one(pack_module) -> None:
    """Two `final` tasks plus one genuinely called `final-2`.

    Handing the second `final` the name `final-2` would collide all over
    again, one step further along -- and silently, the way the first collision
    did.
    """

    tasks = [
        pack_module.Task(Path("out/a/final.srt")),
        pack_module.Task(Path("out/b/final.srt")),
        pack_module.Task(Path("out/c/final-2.srt")),
    ]

    folders = pack_module.bundle_folders(tasks)

    assert len(set(folders.values())) == 3
    assert folders[Path("out/c/final-2.srt")] == "final-2"
    assert folders[Path("out/b/final.srt")] == "final-3"


def test_the_manifest_headings_match_the_folders_in_the_zip(
    pack_module, tmp_path: Path
) -> None:
    """Otherwise two tasks read as one repeated heading over different files."""

    entries = []
    for parent in ("a", "b"):
        directory = tmp_path / parent
        directory.mkdir(parents=True)
        subtitle = directory / "final.srt"
        subtitle.write_text(parent, encoding="utf-8")
        entries.append((pack_module.Task(subtitle), [subtitle]))
    folders = pack_module.bundle_folders([task for task, _ in entries])

    manifest = pack_module.build_manifest("debug", entries, folders)

    assert "[final]" in manifest and "[final-2]" in manifest


def test_the_ledger_key_moves_when_a_corrected_subtitle_is_improved(
    pack_module, tmp_path: Path
) -> None:
    """Keying on the path alone would make a second contribution impossible."""

    directory = _task_dir(tmp_path)
    refined = tmp_path / "refined.srt"
    refined.write_text("first pass\n", encoding="utf-8")
    task = pack_module.Task(directory / "input.srt", refined)
    before = pack_module.task_fingerprint(task)

    refined.write_text("second pass, more careful\n", encoding="utf-8")

    assert pack_module.task_fingerprint(task) != before


def test_the_two_modes_keep_separate_ledger_rows(pack_module, tmp_path: Path) -> None:
    directory = _task_dir(tmp_path)
    task = pack_module.Task(directory / "input.srt")
    ledger = {"debug": {str(task.final_srt): pack_module.task_fingerprint(task)}}

    assert pack_module.already_packed(ledger, "debug", task)
    # The same task may still go out as corpus material later.
    assert not pack_module.already_packed(ledger, "corpus", task)


def test_a_bundle_carries_a_manifest_and_records_what_it_sent(
    pack_module, tmp_path: Path
) -> None:
    directory = _task_dir(tmp_path)
    task = pack_module.Task(directory / "input.srt")
    ledger_file = tmp_path / "ledger.json"

    archive = pack_module.pack(
        [task],
        mode="debug",
        out_dir=tmp_path / "desk",
        ledger_file=ledger_file,
    )

    assert archive is not None and archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        manifest = bundle.read("MANIFEST.txt").decode("utf-8")
    assert "MANIFEST.txt" in names
    assert "input/input-vocal.ogg" in names
    assert not any(name.endswith(".env") for name in names)
    # The manifest names what actually went in, so the user is deciding about
    # something they can see rather than about a description.
    assert "input-vocal.ogg" in manifest

    recorded = json.loads(ledger_file.read_text(encoding="utf-8"))
    assert recorded["debug"][str(task.final_srt)] == pack_module.task_fingerprint(task)

    # A second run finds it in the ledger and packs nothing.
    assert (
        pack_module.pack(
            [task], mode="debug", out_dir=tmp_path / "desk", ledger_file=ledger_file
        )
        is None
    )


def test_corpus_mode_refuses_a_task_with_no_corrected_subtitle(
    pack_module, tmp_path: Path
) -> None:
    """Material with nothing to compare against is not worth a user's upload."""

    directory = _task_dir(tmp_path)

    with pytest.raises(SystemExit):
        pack_module.main(["corpus", str(directory / "input.srt")])

from __future__ import annotations

import importlib
import re
from pathlib import Path
import tomllib

import pytest


def test_source_uses_package_discovery_without_top_level_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    discovery = config["tool"]["setuptools"]["packages"]["find"]

    assert not list((root / "src").glob("*.py"))
    assert discovery["where"] == ["src"]
    assert "finesub*" in discovery["include"]
    assert (root / "src" / "finesub" / "__init__.py").is_file()


def test_license_metadata_is_compatible_with_declared_setuptools_floor() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["license"] == {"file": "LICENSE"}
    assert (root / "LICENSE").is_file()


#: What the fourteen former console scripts became. The docs tell people to run
#: these, and `python -m` needs two things an entry point did not: the module
#: has to be importable *and* run something under `__main__`. Two of them
#: (`cli/align.py`, `cli/vad_asr.py`) had no guard at all when the scripts were
#: dropped -- the command would have exited silently, successfully.
MODULE_ENTRY_POINTS = (
    "finesub.pipeline",
    "finesub.batch",
    "finesub.speech.recognition.cli.align",
    "finesub.speech.recognition.cli.vad_asr",
    "finesub.speech.postprocessing.stabilization",
    "finesub.speech.preprocessing.energy",
    "finesub.speech.preprocessing.separator.separation",
    "finesub.subtitles.rendering",
    "finesub.workflows.reference_ingest",
    "finesub.llm.correction_translation",
    "finesub.llm.knowledge.update",
    "finesub.llm.token_measure",
    "finesub.llm.agent.agent_cleanup",
    "finesub.llm.agent.agent_ping",
    "finesub.llm.agent.agent_task_control",
)


def test_the_distribution_ships_no_console_scripts() -> None:
    """`finesub` comes from the CLI wheel; this one would collide with it.

    The loop is not dead code: it says what the rule would be if someone adds a
    script back, so the reason to leave the table empty stays visible next to
    the check that it is.
    """

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = config["project"].get("scripts", {})

    assert scripts == {}
    for name, target in scripts.items():
        module_name, sep, attr = target.partition(":")
        assert sep and attr, f"{name}: expected module:attr, got {target!r}"
        entry_point: object = importlib.import_module(module_name)
        for part in attr.split("."):
            entry_point = getattr(entry_point, part)
        assert callable(entry_point), f"{name}: entry point {target!r} is not callable"


@pytest.mark.parametrize("module_name", MODULE_ENTRY_POINTS)
def test_a_documented_module_entry_point_can_actually_be_run(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert callable(getattr(module, "main", None)), f"{module_name}: no main()"
    source = Path(module.__file__ or "").read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in source, (
        f"{module_name}: no __main__ guard, so `python -m {module_name}` "
        "would exit 0 without doing anything"
    )


def test_the_domain_markers_cover_every_test_file(file_markers) -> None:
    """`-m llm`, `-m pipeline` and `-m asr` must partition the suite.

    A file in none of them is worse than having no selector at all: the three
    commands run, report green, and never touch it. That is how 21 files drifted
    out of reach before -- nothing failed, so nothing said so.
    """

    # `rglob`, because the suite has subdirectories: a file moved into one of
    # them would otherwise drop out of the partition without anything saying so,
    # which is the exact failure this test exists to catch. Keys are posix
    # paths relative to `test/` -- see the note in `test/conftest.py`.
    here = Path(__file__).resolve().parent
    collected = {
        path.relative_to(here).as_posix() for path in here.rglob("test_*.py")
    }
    known = set(file_markers) | {
        name for name in collected if Path(name).name.startswith("test_llm_")
    }

    unmarked = sorted(collected - known)
    assert unmarked == [], (
        "add these to _PIPELINE_FILES / _ASR_FILES in test/conftest.py: "
        + ", ".join(unmarked)
    )
    # A rename leaves a dead entry behind, which silently stops marking anything.
    stale = sorted(set(file_markers) - collected)
    assert stale == [], "drop these dead entries from test/conftest.py: " + ", ".join(
        stale
    )

    # `test/conftest.py` can only reach `test/`. Anything else in testpaths has
    # to declare its own marker, or the partition has a hole again.
    root = here.parent
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    outside = [
        entry
        for entry in config["tool"]["pytest"]["ini_options"]["testpaths"]
        if entry != "test"
    ]
    for entry in outside:
        path = root / entry
        assert path.is_file(), f"{entry}: only single files are handled here"
        assert "pytestmark" in path.read_text(encoding="utf-8"), (
            f"{entry} is collected by the root suite but declares no domain marker"
        )


def test_canonical_docs_do_not_reference_removed_source_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = [
        root / "README.md",
        root / "README_DEV.md",
        root / "CLAUDE.md",
        *(
            path
            for path in (root / "docs").rglob("*.md")
            if "archive" not in path.parts and "report" not in path.parts
        ),
    ]
    removed_references = (
        "src/pipeline.py",
        "src/batch.py",
        "src/to_srt.py",
        "src/asr_align.py",
        "src/vad_asr.py",
        "src/asr_stabilize.py",
        "src/vocal_separation.py",
        "src/vad_energy.py",
        "src/segment_split.py",
        "src/resource_profiles.py",
        "src/gpu_stage_gate.py",
        "src/utils/",
        # Written without the `src/` prefix on purpose: a doc left behind by
        # the 2026-08 rename still spells these `src/llm/...`, while a freshly
        # mistaken one would write `src/finesub/llm/...`. The shared tail
        # catches both.
        "llm/reference_ingest.py",
        "llm.reference_ingest",
        "llm/media_source.py",
        "llm/ffmpeg_clips.py",
        "llm/audio_clips.py",
        "llm/srt_utils.py",
        "llm/srt_alignment.py",
        "llm/subtitle_metrics.py",
        "llm/srt_postprocess.py",
        "utils.text",
        "vad_asr.WtModelPool",
        "asr_align.main",
        "asr_align.align_segments",
        "asr_align.tag_interval_ids",
        "asr_align.ROUND_DIGITS",
    )
    # Bare pre-migration module aliases; (?!py) avoids test_*.py / *.py filenames.
    bare_aliases = (
        ("ffmpeg_clips.", re.compile(r"(?<![\w/])ffmpeg_clips\.(?!py\b)")),
        ("subtitle_metrics.", re.compile(r"(?<![\w/])subtitle_metrics\.(?!py\b)")),
        # `to_srt` module alias; allow CLI name `to-srt`.
        ("to_srt", re.compile(r"(?<![\w-])to_srt(?![\w-])")),
    )

    offenders: list[str] = []
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for reference in removed_references:
            if reference in text:
                offenders.append(f"{path.relative_to(root)}: {reference}")
        for label, pattern in bare_aliases:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(root)}: {label}")

    assert offenders == []

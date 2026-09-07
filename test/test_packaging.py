from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


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


def test_the_wheel_carries_all_three_pipeline_modules() -> None:
    """A payload missing one of them imports-errors on the user's machine.

    `pipeline.py` (the entry) imports `scheduler.py` (the runner) and
    `stages.py` (the conversion), so the three travel together or not at all.
    The wheel's staging/contents assertions named only some of them after the
    split, so an incomplete package still passed validation (reviewer
    2026-08-30 P2).
    """

    root = Path(__file__).resolve().parents[1]
    modules = ("pipeline.py", "scheduler.py", "stages.py")
    # The script names the staged files with Windows separators and the wheel's
    # contents with POSIX ones; normalising lets one assertion cover both, and
    # keeps backslashes out of this file.
    wheel_script = (
        (root / "cli" / "scripts" / "build-wheel.ps1")
        .read_text(encoding="utf-8")
        .replace(chr(92), "/")
    )
    for module in modules:
        assert wheel_script.count(f"finesub_cli/_vendor/src/finesub/{module}") >= 2, (
            f"the wheel build does not check for {module} in BOTH the staged "
            "tree and the built wheel"
        )


def test_canonical_docs_do_not_reference_removed_source_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = [
        root / "README.md",
        root / "README_DEV.md",
        root / "CLAUDE.md",
        # The two subtrees with canonical docs of their own. Leaving them out
        # is how `finesub batch --manifest` survived in cli/README.md for a
        # whole release after that subcommand was deleted: the guard was
        # looking at the docs that happened to be listed, not at the docs that
        # describe the product (2026-08-31).
        root / "cli" / "README.md",
        *(
            path
            for path in (root / "docs").rglob("*.md")
            if "archive" not in path.parts and "report" not in path.parts
        ),
    ]
    removed_references = (
        "src/pipeline.py",
        "src/batch.py",
        # Both spellings: the module path is how an owner doc names a file, and
        # only the dotted one was listed -- so `src/finesub/batch.py` sat in
        # docs/knowledge.md for a whole review cycle (reviewer 2026-08-30 P2).
        "src/finesub/batch.py",
        "finesub.batch",
        # The COMMAND spelling too. The dotted module and the bare filename
        # were both listed, and `finesub batch --manifest tasks.jsonl` still
        # sat in cli/README.md regardless: a user-facing doc names a command,
        # not a module (2026-08-31).
        "finesub batch",
        # And the bare filename: a doc says "`batch.py` 的白名单" as readily as
        # it says the path, and only the two qualified spellings were listed --
        # so one sentence survived a whole review round (reviewer 2026-08-30).
        "batch.py",
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


# --- Contracts between this file's extras and what the CLI wheel ships. They
# lived in the desktop suite's dependency tests until the split; every one of
# them is about the CLI, and each drifted silently at least once while nothing
# in the root suite looked.

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_windows_ai_runtime_lock_matches_the_pipeline_extras() -> None:
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock_path = (
        REPOSITORY_ROOT / "src" / "finesub_bootstrap" / "pylock.win-py312.toml"
    )
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = {package["name"]: package["version"] for package in lock["packages"]}

    # Transitive deps of audio-separator, so the extras below never name them --
    # but the worker imports them directly and has shipped without them before.
    assert "beartype" in packages
    assert "ml-collections" in packages

    # The lock is generated from [asr]+[harness]+[runtime], so every exact pin
    # on those extras has to survive into it. This drifted once and went
    # unnoticed for a month: the lock was compiled while [asr] still used
    # whisper-timestamped, and after the fw-refine migration it contained no
    # decoder at all -- installable, and unable to transcribe a thing.
    windows_environment = {
        "sys_platform": "win32",
        "platform_machine": "AMD64",
        "python_version": "3.12",
    }
    requirements = {
        requirement.name.lower(): (extra, requirement)
        for extra in ("asr", "harness", "runtime")
        for raw in project["project"]["optional-dependencies"][extra]
        if (requirement := Requirement(raw)).marker is None
        or requirement.marker.evaluate(windows_environment)
    }
    for name, (extra, requirement) in requirements.items():
        assert name in packages, (
            f"{name} is required by [{extra}] but is missing from "
            f"src/finesub_bootstrap/pylock.win-py312.toml. Regenerate the lock "
            f"with the command in its header."
        )
        if not requirement.specifier:
            continue
        # Locked versions carry local labels ("2.11.0+cu128") that the extras'
        # specifiers do not spell out; PEP 440 matches those, so compare whole.
        assert Version(packages[name]) in requirement.specifier, (
            f"{name}: [{extra}] asks for {requirement.specifier} but the "
            f"packaged lock pins {packages[name]}. Regenerate the lock."
        )

    # Stock CTranslate2 satisfies ctranslate2==4.8.1 -- PEP 440 local labels are
    # not an exclusion mechanism -- so the pin above cannot catch this on its
    # own. [runtime] carries a direct reference for that reason, and the
    # patched build is what fw-refine needs at runtime.
    #
    # Spelled out rather than imported from finesub_bootstrap.environment
    # (REQUIRED_CTRANSLATE2_LOCAL_LABEL): `test_runtime_environment` ties the
    # two together, this one only has to hold without importing the module.
    assert "finesub" in packages["ctranslate2"]


def test_the_cli_shell_and_the_runtime_manifest_pin_the_same_uv() -> None:
    # The shell installs uv as a wheel dependency and the manifest downloads
    # it for a managed runtime; a different resolver version on either side
    # would make "same lock, same environment" a hope instead of a guarantee.
    manifest = json.loads(
        (
            REPOSITORY_ROOT / "src" / "finesub_bootstrap" / "runtime-manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest_uv = next(
        resource["version"]
        for resource in manifest["resources"]
        if resource["id"] == "uv"
    )
    shell = tomllib.loads(
        (REPOSITORY_ROOT / "cli" / "pyproject.toml").read_text(encoding="utf-8")
    )
    uv_requirements = [
        Requirement(dependency)
        for dependency in shell["project"]["dependencies"]
        if Requirement(dependency).name == "uv"
    ]
    assert len(uv_requirements) == 1
    assert str(uv_requirements[0].specifier) == f"=={manifest_uv}"


def test_the_cli_shell_exposes_only_the_launcher_entry_point() -> None:
    # The shell venv has no torch: any pipeline entry point on PATH would be a
    # command that always crashes with ImportError. Everything goes through
    # the `finesub` launcher, which re-executes inside the managed runtime.
    shell = tomllib.loads(
        (REPOSITORY_ROOT / "cli" / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert set(shell["project"]["scripts"]) == {"finesub"}


def test_the_version_number_has_one_source() -> None:
    # The root `VERSION` file is the one source; the wheel build stamps from it
    # and the root pyproject reads it. Asserting `dynamic` is what keeps it
    # *one*: a literal `version = "..."` back in the root pyproject would be a
    # second copy that agrees today and drifts later, which is the arrangement
    # this test replaced (2026-09-03, desktop split).
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "version" in project["project"].get("dynamic", []), (
        "the root pyproject declares a literal version again; it must stay "
        "dynamic and read the root VERSION file"
    )
    assert "version" not in project["project"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"file": "VERSION"}
    # A tag name is derived from it, so it has to be a version and not a note.
    Version((REPOSITORY_ROOT / "VERSION").read_text("utf-8").strip())

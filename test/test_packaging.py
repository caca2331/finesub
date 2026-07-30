from __future__ import annotations

import importlib
import re
from pathlib import Path
import tomllib


def test_source_uses_package_discovery_without_top_level_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    discovery = config["tool"]["setuptools"]["packages"]["find"]

    assert not list((root / "src").glob("*.py"))
    assert discovery["where"] == ["src"]
    assert "asr_playground*" in discovery["include"]
    assert (root / "src" / "asr_playground" / "__init__.py").is_file()


def test_license_metadata_is_compatible_with_declared_setuptools_floor() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["license"] == {"file": "LICENSE"}
    assert (root / "LICENSE").is_file()


def test_console_script_entry_points_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = config["project"]["scripts"]

    assert scripts
    for name, target in scripts.items():
        module_name, sep, attr = target.partition(":")
        assert sep and attr, f"{name}: expected module:attr, got {target!r}"
        entry_point: object = importlib.import_module(module_name)
        for part in attr.split("."):
            entry_point = getattr(entry_point, part)
        assert callable(entry_point), f"{name}: entry point {target!r} is not callable"


def test_canonical_docs_do_not_reference_removed_source_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    docs = [
        root / "README.md",
        root / "README_DEV.md",
        root / "CLAUDE.md",
        *(
            path
            for path in (root / "docs").rglob("*.md")
            if "archive" not in path.parts
            and "report" not in path.parts
            and path.name != "package-reorganization-plan.md"
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
        "src/wt_shard.py",
        "src/segment_split.py",
        "src/resource_profiles.py",
        "src/gpu_stage_gate.py",
        "src/utils/",
        "src/llm/reference_ingest.py",
        "llm.reference_ingest",
        "src/llm/media_source.py",
        "src/llm/ffmpeg_clips.py",
        "src/llm/audio_clips.py",
        "src/llm/srt_utils.py",
        "src/llm/srt_alignment.py",
        "src/llm/subtitle_metrics.py",
        "src/llm/srt_postprocess.py",
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
        ("wt_shard.", re.compile(r"(?<![\w/])wt_shard\.(?!py\b)")),
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

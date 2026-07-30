from __future__ import annotations

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
HEAVY_IMPORTS = {
    "audio_separator",
    "numba",
    "numpy",
    "torch",
    "torchaudio",
    "whisper_timestamped",
}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_public_media_and_subtitles_have_no_upward_imports() -> None:
    forbidden = ("llm", "asr_playground.speech", "asr_playground.workflows")
    offenders: list[str] = []
    for domain in ("media", "subtitles"):
        for source in (SOURCE_ROOT / "asr_playground" / domain).glob("*.py"):
            for imported in _top_level_imports(source):
                if imported.startswith(forbidden):
                    offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


def test_harness_public_layers_do_not_import_asr_dependencies_at_module_load() -> None:
    offenders: list[str] = []
    roots = [
        SOURCE_ROOT / "asr_playground" / "media",
        SOURCE_ROOT / "asr_playground" / "subtitles",
        SOURCE_ROOT / "asr_playground" / "workflows",
        SOURCE_ROOT / "llm",
    ]
    for root in roots:
        for source in root.rglob("*.py"):
            for imported in _top_level_imports(source):
                if imported.split(".", 1)[0] in HEAVY_IMPORTS:
                    offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


def test_speech_has_no_llm_dependency() -> None:
    offenders: list[str] = []
    root = SOURCE_ROOT / "asr_playground" / "speech"
    for source in root.rglob("*.py"):
        for imported in _top_level_imports(source):
            if imported == "llm" or imported.startswith(("llm.", "asr_playground.llm")):
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []

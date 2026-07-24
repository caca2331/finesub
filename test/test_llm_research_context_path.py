from __future__ import annotations

from pathlib import Path

from llm.knowledge.update import (
    derive_task_paths,
    ensure_research_context_path,
    research_context_in_artifact_dir,
)


def test_derive_task_paths_puts_research_context_under_artifacts(tmp_path: Path) -> None:
    srt = tmp_path / "clip.srt"
    srt.write_text("1\n", encoding="utf-8")
    paths = derive_task_paths(srt)
    assert paths["research_context"] == paths["artifact_dir"] / "clip-research-context.json"
    assert paths["research_context"] == research_context_in_artifact_dir(
        paths["artifact_dir"], "clip"
    )


def test_ensure_research_context_migrates_legacy_sibling(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact_dir = run_dir / "clip.llm-artifacts"
    run_dir.mkdir()
    legacy = run_dir / "clip-research-context.json"
    legacy.write_text('{"mode":"fast"}', encoding="utf-8")
    preferred = ensure_research_context_path(
        artifact_dir=artifact_dir,
        stem="clip",
        run_dir=run_dir,
    )
    assert preferred == artifact_dir / "clip-research-context.json"
    assert preferred.exists()
    assert not legacy.exists()
    assert preferred.read_text(encoding="utf-8") == '{"mode":"fast"}'

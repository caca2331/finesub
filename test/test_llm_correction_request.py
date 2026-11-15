"""The one correction task every front end runs (architecture review
2026-09-25, candidates 1 and 5): refusals come before any work, a plan runs
nothing, and an execute run shows its plan before it starts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import finesub.llm.correction_translation as ct
from finesub.llm.correction_request import (
    CorrectionRequest,
    CorrectionRequestError,
    resolved_request,
)
from finesub.llm.routing.profiles import resolve_profile

TEXT = resolve_profile("text", "none", "quality")
EFFICIENCY = resolve_profile("text", "none", "efficiency")


def _stable(tmp_path: Path) -> Path:
    stable = tmp_path / "clip-stable.json"
    stable.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    return stable


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"refined_srt": Path("r.srt"), "knowledge": "read"}, "--refined-srt"),
        ({"knowledge": "collect"}, "renamed to read"),
        ({"mode": "research", "context_file": Path("c.json")}, "--context-file"),
        ({"mode": "research", "fast": "on"}, "research-only"),
        ({"output_path": None}, "--output"),
        ({"profile": resolve_profile("audio", "none", "quality")}, "--audio"),
        ({"video_path": Path("v.mp4")}, "--video"),
        ({"mode": "rehearse"}, "mode"),
    ],
)
def test_contradictions_are_refused_before_any_work(tmp_path, fields, message) -> None:
    request = CorrectionRequest(
        stable_json=tmp_path / "clip-stable.json", output_path=tmp_path / "clip.srt", profile=TEXT
    )
    with pytest.raises(CorrectionRequestError, match=message):
        resolved_request(CorrectionRequest(**{**vars(request), **fields}))


def test_unset_fields_resolve_once() -> None:
    request = resolved_request(
        CorrectionRequest(stable_json=Path("s.json"), mode="research", profile=TEXT)
    )
    assert request.knowledge == "read" and request.fast == "off"
    assert resolved_request(request) == request  # idempotent: the task asks again
    efficient = resolved_request(CorrectionRequest(stable_json=Path("s.json"), mode="plan", profile=EFFICIENCY))
    assert efficient.knowledge == "read"  # the usual default at every difficulty


def test_retry_tiers_come_from_config_unless_given(tmp_path, monkeypatch) -> None:
    """config.toml sets both tiers; an explicit value still wins."""

    from finesub import config as app_config

    config = tmp_path / "config.toml"
    config.write_text(
        "[llm]\nmax_retries_per_window = 2\nmax_replacements_per_window = 0\n", encoding="utf-8"
    )
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config))
    app_config.clear_config_cache()
    try:
        base = CorrectionRequest(stable_json=Path("s.json"), mode="research", profile=TEXT)
        request = resolved_request(base)
        assert (request.max_retries_per_window, request.max_replacements_per_window) == (2, 0)
        explicit = resolved_request(CorrectionRequest(**{**vars(base), "max_retries_per_window": 4}))
        assert explicit.max_retries_per_window == 4
    finally:
        app_config.clear_config_cache()


def test_a_plan_renders_the_prompts_and_runs_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ct, "run_research_stage", lambda **k: pytest.fail("a plan ran research"))
    monkeypatch.setattr(ct, "execute_correction_windows", lambda **k: pytest.fail("a plan ran windows"))
    shown: list = []
    outcome = ct.run_correction_task(
        CorrectionRequest(
            stable_json=_stable(tmp_path), mode="plan", profile=TEXT, knowledge="none",
            on_plan=shown.append,
        )
    )
    assert shown and "fast_decision" in shown[0] and "correction_messages" in shown[0]
    assert outcome.result is None and outcome.artifact_dir is None


def test_a_finished_task_opens_no_run_boundary(tmp_path, monkeypatch) -> None:
    """Nothing owed, nothing held: a batch resume must not queue a finished
    item behind running agent tasks for a slot it will not use."""

    monkeypatch.setattr(ct, "_task_slots", lambda *a: pytest.fail("took an agent slot"))
    monkeypatch.setattr(ct, "_generation_pin", lambda *a: pytest.fail("pinned the knowledge base"))
    out = tmp_path / "clip.srt"
    out.write_text("done", encoding="utf-8")
    outcome = ct.run_correction_task(
        CorrectionRequest(
            stable_json=_stable(tmp_path), output_path=out, profile=TEXT, knowledge="read",
        )
    )
    assert outcome.result == out.resolve() and outcome.result_reused


def test_an_execute_run_shows_its_plan_before_it_prepares(tmp_path, monkeypatch) -> None:
    order: list[str] = []

    def prepare(**kwargs):
        order.append("research")
        Path(kwargs["context_path"]).write_text(json.dumps({"keep_entries": []}), encoding="utf-8")
        from finesub.llm.prompts import ContextPack

        return ContextPack()

    def windows(**kwargs):
        order.append("windows")
        out = Path(kwargs["output_path"]).with_name("clip-translated.srt")
        out.write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(ct, "run_research_stage", prepare)
    monkeypatch.setattr(ct, "execute_correction_windows", windows)
    outcome = ct.run_correction_task(
        CorrectionRequest(
            stable_json=_stable(tmp_path), output_path=tmp_path / "clip.srt", profile=TEXT,
            fast="off", knowledge="none", postprocess_profile=None, test_profile=True,
            on_plan=lambda artifacts: order.append("plan"),
        )
    )
    assert order == ["plan", "research", "windows"]
    assert outcome.result == tmp_path / "clip-translated.srt" and not outcome.result_reused
    assert outcome.research_context is not None and not outcome.research_reused
    metadata = json.loads((tmp_path / "clip-metadata.json").read_text(encoding="utf-8"))
    assert metadata["timing"]["stages"]["llm_harness"]["status"] == "executed"


@pytest.mark.parametrize(
    ("flags", "mode", "fast"),
    [
        (["--research-only"], "plan", "off"),
        (["--research-only", "--execute"], "research", None),
        (["--execute"], "execute", None),
        ([], "plan", None),
    ],
)
def test_research_only_calls_a_model_only_with_execute(tmp_path, monkeypatch, flags, mode, fast) -> None:
    """`--research-only` scopes the run; without `--execute` it is a dry-run like any other."""

    import dataclasses
    import sys

    from finesub.llm import correction_translation as ct

    seen = {}
    monkeypatch.setattr(ct, "run_correction_task", lambda request: seen.update(vars(request)) or ct.CorrectionOutcome())
    stable = tmp_path / "clip-stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["correction_translation", str(stable), *flags])
    assert ct.main() == 0
    assert seen["mode"] == mode
    default = next(f.default for f in dataclasses.fields(ct.CorrectionRequest) if f.name == "fast")
    assert seen["fast"] == (fast or default)  # only the dry-run pins it, as research would


def test_a_research_only_dry_run_keeps_the_research_conflicts(tmp_path, monkeypatch, capsys) -> None:
    import sys

    from finesub.llm import correction_translation as ct

    stable = tmp_path / "clip-stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    monkeypatch.setattr(ct, "run_correction_task", lambda request: pytest.fail("must refuse first"))
    for extra, message in ((["--fast", "on"], "--fast on"), (["--context-file", str(stable)], "--context-file")):
        monkeypatch.setattr(sys, "argv", ["correction_translation", str(stable), "--research-only", *extra])
        assert ct.main() == 2
        assert message in capsys.readouterr().err

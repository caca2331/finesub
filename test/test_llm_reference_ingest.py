from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import asr_playground
from asr_playground.workflows import reference_ingest
from asr_playground.workflows.reference_ingest import (
    ResolvedSettings,
    TaskRow,
    build_task,
    parse_row,
    read_index_rows,
    resolve_media,
    resolve_settings,
    resolve_srt,
)
from asr_playground.subtitles.model import SrtSegment, render_srt


def _write_srt(path: Path, texts: list[str]) -> None:
    segments = [
        SrtSegment(index=i + 1, start=i * 2.0, end=i * 2.0 + 1.5, text=text)
        for i, text in enumerate(texts)
    ]
    path.write_text(render_srt(segments), encoding="utf-8")


def test_reference_pipeline_uses_batch_style_single_wt_worker(
    tmp_path, monkeypatch
) -> None:
    seen: dict[str, object] = {}
    stable = tmp_path / "item-stable.json"

    def fake_run_pipeline(*args, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(stable_json=stable)

    fake_pipeline = types.SimpleNamespace(run_pipeline=fake_run_pipeline)
    monkeypatch.setitem(
        sys.modules,
        "asr_playground.pipeline",
        fake_pipeline,
    )
    monkeypatch.setattr(asr_playground, "pipeline", fake_pipeline, raising=False)

    result = reference_ingest.run_reference_pipeline(
        tmp_path / "input.ogg",
        video_id="item",
        work_dir=tmp_path,
        model="large-v3-turbo",
        language="ja",
        gpu_budget_gb=12,
    )

    assert result == stable
    assert seen["wt_workers"] == 1


# --- row parsing -----------------------------------------------------------
def test_parse_row_full_and_partial() -> None:
    full = parse_row("a.srt|https://x/v|some note|prod|--level high")
    assert full == TaskRow("a.srt", "https://x/v", "some note", "prod", "--level high")

    partial = parse_row("a.srt|b.mp4")
    assert partial == TaskRow("a.srt", "b.mp4", "", "", "")


def test_parse_row_args_may_contain_pipe() -> None:
    row = parse_row("a.srt|u|note|prod|--extra 'x|y'")
    assert row.args == "--extra 'x|y'"
    assert row.note == "note"


def test_parse_row_requires_srt() -> None:
    with pytest.raises(ValueError, match="missing the srt"):
        parse_row("|media|note")


def test_read_index_rows_skips_comments_and_blanks(tmp_path) -> None:
    index = tmp_path / "index.csv"
    index.write_text(
        "# comment\n\na.srt|u1\n  \nb.srt|u2|note|prod|\n# trailing\n",
        encoding="utf-8",
    )
    rows = read_index_rows(index)
    assert [r.srt for r in rows] == ["a.srt", "b.srt"]
    assert rows[1].preset == "prod"


# --- settings resolution ---------------------------------------------------
def test_resolve_settings_default_preset_is_mm_med_test_profile() -> None:
    settings = resolve_settings(TaskRow(srt="a.srt", media="u"), ResolvedSettings())
    assert (settings.route, settings.level, settings.test_profile) == ("mm", "med", True)


def test_resolve_settings_prod_and_text_presets() -> None:
    prod = resolve_settings(TaskRow("a", "u", preset="prod"), ResolvedSettings())
    assert (prod.route, prod.level, prod.test_profile) == ("mm", "med", False)
    text = resolve_settings(TaskRow("a", "u", preset="text"), ResolvedSettings())
    assert (text.route, text.level, text.test_profile) == ("text", "med", False)
    text_high = resolve_settings(TaskRow("a", "u", preset="text-high"), ResolvedSettings())
    assert (text_high.route, text_high.level, text_high.test_profile) == ("text", "high", False)
    mm_low = resolve_settings(TaskRow("a", "u", preset="mm-low"), ResolvedSettings())
    assert (mm_low.route, mm_low.level, mm_low.test_profile) == ("mm", "low", False)
    mm_high = resolve_settings(TaskRow("a", "u", preset="mm-high"), ResolvedSettings())
    assert (mm_high.route, mm_high.level, mm_high.test_profile) == ("mm", "high", True)


def test_resolve_settings_args_override_preset() -> None:
    row = TaskRow("a", "u", preset="prod", args="--route text --level high --no-web-search --model x")
    settings = resolve_settings(row, ResolvedSettings())
    assert settings.route == "text"
    assert settings.level == "high"
    assert settings.no_web_search is True
    assert settings.model == "x"
    # test_profile from prod preset stays False (not overridden).
    assert settings.test_profile is False


def test_resolve_settings_no_test_profile_override() -> None:
    row = TaskRow("a", "u", preset="mm-med", args="--no-test-profile")
    assert resolve_settings(row, ResolvedSettings()).test_profile is False


def test_resolve_settings_unknown_preset_and_args() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        resolve_settings(TaskRow("a", "u", preset="nope"), ResolvedSettings())
    with pytest.raises(ValueError, match="unrecognized args"):
        resolve_settings(TaskRow("a", "u", args="--bogus 1"), ResolvedSettings())


def test_global_defaults_flow_into_settings() -> None:
    defaults = ResolvedSettings(model="large-v3", language="ja", gpu_budget_gb=16)
    settings = resolve_settings(TaskRow("a", "u"), defaults)
    assert (settings.model, settings.language, settings.gpu_budget_gb) == ("large-v3", "ja", 16)


# --- srt / media resolution ------------------------------------------------
def test_resolve_srt_bare_name_in_batch_vs_path_in_single(tmp_path) -> None:
    assert resolve_srt("clip1", tmp_path) == tmp_path / "clip1.srt"
    assert resolve_srt("sub/clip.srt", tmp_path) == Path("sub/clip.srt")
    # Single mode (base_dir None): always a path.
    assert resolve_srt("clip1", None) == Path("clip1")


def test_resolve_media_url_bare_and_path(tmp_path) -> None:
    (tmp_path / "clip1.mp4").write_bytes(b"x")
    assert resolve_media("https://x/v", tmp_path) == (True, "https://x/v")
    is_url, media = resolve_media("clip1", tmp_path)  # bare -> glob dir
    assert is_url is False and media.endswith("clip1.mp4")
    assert resolve_media("a/b.wav", tmp_path) == (False, str(Path("a/b.wav").expanduser()))


def test_resolve_media_bare_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no media file"):
        resolve_media("nope", tmp_path)


# --- build_task ------------------------------------------------------------
def test_build_task_mm_high_no_media_raises() -> None:
    row = TaskRow("a.srt", "", preset="prod", args="--level high")
    with pytest.raises(ValueError, match="needs a video"):
        build_task(row, None, ResolvedSettings())


def test_build_task_mm_high_url_defers_video_download() -> None:
    row = TaskRow("a.srt", "https://x/v", preset="prod", args="--level high")
    task = build_task(row, None, ResolvedSettings())
    assert task.profile.use_video is True
    assert task.is_media_url is True
    assert task.video_path == ""  # filled by process_task via download_video


def test_build_task_mm_high_explicit_video_arg_wins() -> None:
    row = TaskRow("a.srt", "https://x/v", preset="prod", args="--level high --video local.mp4")
    task = build_task(row, None, ResolvedSettings())
    assert task.video_path == "local.mp4"


def test_build_task_mm_high_uses_local_media_as_video(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    row = TaskRow("a.srt", str(video), preset="prod", args="--level high")
    task = build_task(row, None, ResolvedSettings())
    assert task.profile.use_video is True
    assert task.video_path == str(video)


# --- CLI dry-run / process_task -------------------------------------------
def test_dry_run_prints_plan_without_network(tmp_path, monkeypatch, capsys) -> None:
    refined = tmp_path / "refined.srt"
    _write_srt(refined, ["你好"])
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # dry run must not touch network
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference_ingest",
            "--task",
            f"{refined}|https://example.com/v|来源说明|mm-med|",
            "--dry-run",
        ],
    )
    assert reference_ingest.main() == 0
    out = capsys.readouterr().out
    assert "计划处理 1 个任务" in out
    assert "route=mm level=med" in out
    assert "来源说明" in out


def test_missing_refined_srt_fails_fast(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["reference_ingest", "--task", f"{tmp_path/'missing.srt'}|https://x/v", "--dry-run"],
    )
    assert reference_ingest.main() == 2


def test_no_input_errors(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["reference_ingest"])
    with pytest.raises(SystemExit):
        reference_ingest.parse_args()


def test_download_audio_skips_existing_file(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "vid123").mkdir(parents=True)
    (data_dir / "vid123" / "vid123.ogg").write_bytes(b"fake")
    reference_ingest._save_url_map(data_dir, {"https://example.com/v": "vid123"})
    monkeypatch.setitem(sys.modules, "yt_dlp", None)

    video_id, audio = reference_ingest.download_audio("https://example.com/v", data_dir)
    assert video_id == "vid123"
    assert audio.name == "vid123.ogg"


def test_local_media_skips_download(tmp_path, monkeypatch) -> None:
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    row = reference_ingest.parse_row(f"a.srt|{media}|note|mm-med|")
    task = reference_ingest.build_task(row, None, ResolvedSettings())
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # would explode if download attempted
    video_id, audio = reference_ingest.resolve_media_source(task, tmp_path / "data")
    assert audio == media
    assert video_id == "clip"


@pytest.mark.slow
def test_process_task_runs_stages_and_knowledge_update(tmp_path, monkeypatch) -> None:
    refined = tmp_path / "refined.srt"
    _write_srt(refined, ["精修一", "精修二"])
    data_dir = tmp_path / "data"
    work_dir = tmp_path / "work"
    calls: dict[str, object] = {}

    monkeypatch.setattr(reference_ingest, "resolve_video_id", lambda url, data_root: "vid1")

    def fake_download(url, data_root, **kwargs):
        audio = Path(kwargs["target_dir"]) / "vid1.ogg"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"fake")
        return "vid1", audio

    def fake_pipeline(audio_path, *, video_id, work_dir, model, language, gpu_budget_gb):
        calls["pipeline"] = {
            "audio_path": str(audio_path),
            "model": model,
            "language": language,
            "gpu": gpu_budget_gb,
        }
        stable = Path(work_dir) / video_id / f"{video_id}-stable.json"
        stable.parent.mkdir(parents=True, exist_ok=True)
        stable.write_text("{}", encoding="utf-8")
        return stable

    def fake_correction(**kwargs):
        calls["correction"] = kwargs
        out = Path(kwargs["output_path"])
        _write_srt(out, ["机器一", "机器二"])
        return out

    def fake_knowledge_update(**kwargs):
        calls["knowledge_update"] = kwargs
        return {"mode": "refined_aligned", "chunks": [], "ledger_path": ""}

    monkeypatch.setattr(reference_ingest, "download_audio", fake_download)
    monkeypatch.setattr(reference_ingest, "run_reference_pipeline", fake_pipeline)
    monkeypatch.setattr(reference_ingest, "run_full_correction", fake_correction)
    monkeypatch.setattr(reference_ingest, "run_knowledge_update", fake_knowledge_update)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference_ingest",
            "--task",
            f"{refined}|https://example.com/v|来源说明|mm-med|",
            "--data-dir",
            str(data_dir),
            "--work-dir",
            str(work_dir),
            "--language",
            "ja",
            "--gpu-budget-gb",
            "12",
        ],
    )

    assert reference_ingest.main() == 0
    assert calls["pipeline"] == {
        "audio_path": str(work_dir / "vid1" / "vid1.ogg"),
        "model": "large-v3-turbo",
        "language": "ja",
        "gpu": 12,
    }
    assert calls["correction"]["knowledge"] == "collect"
    assert calls["correction"]["test_profile"] is True  # mm-med preset
    assert calls["correction"]["profile"].route == "mm"
    assert "https://example.com/v" in calls["correction"]["extra_info"]
    assert "来源说明" in calls["correction"]["extra_info"]
    # One unified knowledge update in the refined_aligned mode.
    update = calls["knowledge_update"]
    assert Path(update["refined_srt"]) == refined
    assert Path(update["final_srt"]).name == "vid1.srt"
    assert Path(update["stable_json"]).name == "vid1-stable.json"
    assert Path(update["artifact_dir"]).name == "llm-artifacts"
    assert update["task_id"] == "reference-ingest-vid1"
    # Existing zh SRT skips the correction step on rerun.
    calls.pop("correction")
    assert reference_ingest.main() == 0
    assert "correction" not in calls
    assert "knowledge_update" in calls  # the update still runs on rerun


def test_batch_isolates_failed_task_and_keeps_llm_in_index_order(
    tmp_path, monkeypatch
) -> None:
    # Three tasks; the middle one fails at the ASR stage. The batch must
    # finish the other two (exit 1), and their llm stages (correction +
    # knowledge update) must run in index order.
    refined = {}
    for name in ("t0", "t1", "t2"):
        refined[name] = tmp_path / f"{name}.srt"
        _write_srt(refined[name], [f"精修{name}"])
    llm_order: list[str] = []

    def fake_download(url, data_root, **kwargs):
        video_id = url.rsplit("/", 1)[1]
        audio = Path(kwargs["target_dir"]) / f"{video_id}.ogg"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"fake")
        return video_id, audio

    def fake_pipeline(audio_path, *, video_id, work_dir, model, language, gpu_budget_gb):
        if video_id == "t1":
            raise RuntimeError("ASR exploded")
        stable = Path(work_dir) / video_id / f"{video_id}-stable.json"
        stable.parent.mkdir(parents=True, exist_ok=True)
        stable.write_text("{}", encoding="utf-8")
        return stable

    def fake_correction(**kwargs):
        out = Path(kwargs["output_path"])
        _write_srt(out, ["机器"])
        return out

    def fake_knowledge_update(**kwargs):
        llm_order.append(kwargs["task_id"])
        return {"mode": "refined_aligned", "chunks": [], "ledger_path": ""}

    monkeypatch.setattr(
        reference_ingest, "resolve_video_id", lambda url, data_root: url.rsplit("/", 1)[1]
    )
    monkeypatch.setattr(reference_ingest, "download_audio", fake_download)
    monkeypatch.setattr(reference_ingest, "run_reference_pipeline", fake_pipeline)
    monkeypatch.setattr(reference_ingest, "run_full_correction", fake_correction)
    monkeypatch.setattr(reference_ingest, "run_knowledge_update", fake_knowledge_update)

    from asr_playground import batch as batch_runner

    status_dir = tmp_path / "batch-root"
    monkeypatch.setattr(batch_runner, "DEFAULT_BATCH_ROOT", status_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference_ingest",
            "--task", f"{refined['t0']}|https://example.com/t0",
            "--task", f"{refined['t1']}|https://example.com/t1",
            "--task", f"{refined['t2']}|https://example.com/t2",
            "--data-dir", str(tmp_path / "data"),
            "--work-dir", str(tmp_path / "work"),
        ],
    )

    assert reference_ingest.main() == 1  # one task failed
    assert llm_order == ["reference-ingest-t0", "reference-ingest-t2"]

    status_files = list(status_dir.glob("*/batch-status.jsonl"))
    assert len(status_files) == 1
    import json

    events = [
        json.loads(line)
        for line in status_files[0].read_text(encoding="utf-8").splitlines()
    ]
    by_item = {e["label"]: e["status"] for e in events if e["stage"] == "item"}
    assert by_item == {"0:t0": "done", "1:t1": "failed", "2:t2": "done"}
    failed = [e for e in events if e["status"] == "failed"]
    assert failed and failed[0]["stage"] == "asr" and "ASR exploded" in failed[0]["error"]


def test_process_task_mm_high_url_downloads_video(tmp_path, monkeypatch) -> None:
    refined = tmp_path / "refined.srt"
    _write_srt(refined, ["精修"])
    calls: dict[str, object] = {}

    def fake_extract_audio(video_path):
        calls["extract_audio"] = str(video_path)
        audio = Path(video_path).with_suffix(".ogg")
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"a")
        return audio

    def fake_download_video(url, data_root, **kwargs):
        calls["download_video"] = url
        video = Path(kwargs["target_dir"]) / "vid1.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"v")
        return "vid1", video

    def fake_pipeline(audio_path, *, video_id, work_dir, model, language, gpu_budget_gb):
        calls["pipeline_audio"] = str(audio_path)
        stable = Path(work_dir) / video_id / f"{video_id}-stable.json"
        stable.parent.mkdir(parents=True, exist_ok=True)
        stable.write_text("{}", encoding="utf-8")
        return stable

    def fake_correction(**kwargs):
        calls["correction"] = kwargs
        out = Path(kwargs["output_path"])
        _write_srt(out, ["机器"])
        return out

    data_root = tmp_path / "data"
    monkeypatch.setattr(reference_ingest, "resolve_video_id", lambda url, data_root: "vid1")
    monkeypatch.setattr(reference_ingest, "download_video", fake_download_video)
    monkeypatch.setattr(reference_ingest, "extract_audio_from_video", fake_extract_audio)
    monkeypatch.setattr(reference_ingest, "run_reference_pipeline", fake_pipeline)
    monkeypatch.setattr(reference_ingest, "run_full_correction", fake_correction)
    monkeypatch.setattr(
        reference_ingest,
        "run_knowledge_update",
        lambda **k: {"mode": "refined_aligned", "chunks": [], "ledger_path": ""},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference_ingest",
            "--task",
            f"{refined}|https://example.com/v||prod|--level high",
            "--data-dir",
            str(data_root),
            "--work-dir",
            str(tmp_path / "work"),
        ],
    )

    assert reference_ingest.main() == 0
    assert calls["download_video"] == "https://example.com/v"
    assert calls["extract_audio"].endswith("vid1.mp4")
    assert calls["pipeline_audio"].endswith("vid1.ogg")
    assert calls["correction"]["video_path"].endswith("vid1.mp4")
    assert str(tmp_path / "work" / "vid1") in calls["pipeline_audio"]
    assert calls["correction"]["profile"].use_video is True

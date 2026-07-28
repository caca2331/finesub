"""Production pipeline: vocal separation -> aligned ASR -> stable ASR -> SRT."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import os
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple, Optional

import asr_align
import asr_stabilize
from llm.srt_postprocess import (
    DEFAULT_POSTPROCESS_PROFILE,
    SUPPORTED_POSTPROCESS_PROFILES,
)
from resource_profiles import DEFAULT_GPU_BUDGET_GB, gpu_budget_choices
import to_srt
import vad_asr
import vocal_separation


class PipelinePaths(NamedTuple):
    vocal_audio: Path
    aligned_json: Path
    stable_json: Path
    raw_srt: Path
    translated_srt: Path
    final_srt: Path
    task_artifact_dir: Path

    @property
    def srt(self) -> Path:
        return self.final_srt

    def resolve_vocal_audio(self) -> Path:
        """Return the existing vocal file: prefer .ogg, fall back to legacy .flac."""
        if self.vocal_audio.exists():
            return self.vocal_audio
        flac_fallback = self.vocal_audio.with_suffix(".flac")
        if flac_fallback.exists():
            return flac_fallback
        return self.vocal_audio


PIPELINE_STAGE_ORDER = {
    "vocal": 1,
    "aligned": 2,
    "stable": 3,
    "raw-srt": 4,
    "translated-srt": 5,
    "final-srt": 6,
}

_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts",
})


def default_output_path(input_path: Path) -> Path:
    # Group every artifact of one input under out/<stem>/ so a run's outputs
    # stay together instead of scattering across out/.
    return Path("out") / input_path.stem / f"{input_path.stem}.srt"


def resolve_llm_level_for_source(
    source_path: Path,
    *,
    stage: str,
    llm_route: str,
    llm_level: str,
    llm_video: str | Path | None,
) -> tuple[str, str | Path | None, str]:
    """Pick the effective LLM level/video for a local input.

    mm-high needs a video track: a local video file becomes the default
    ``llm_video``, audio-only input downgrades to med. Only meaningful once an
    LLM stage actually runs, so earlier stages are left untouched -- same
    condition ``prepare_url_input`` uses to decide whether to fetch video.
    Returns ``(llm_level, llm_video, notice)``; an empty notice means silence.
    """
    if PIPELINE_STAGE_ORDER[stage] < PIPELINE_STAGE_ORDER["translated-srt"]:
        return llm_level, llm_video, ""
    if llm_route != "mm" or llm_level != "high":
        return llm_level, llm_video, ""
    if source_path.suffix.lower() in _VIDEO_EXTENSIONS:
        return llm_level, llm_video or source_path, ""
    return (
        "med",
        llm_video,
        "Note: input is audio-only; downgrading llm-level high → med "
        "(video not available for mm-high).",
    )


def resolve_name_output_path(name: str) -> Path:
    """
    将用户提供的“名称”转换为 out/<name>/<name>.srt，
    并对输入进行严格校验以拒绝路径分隔符、'.'、'..'、空或仅空白的名称。
    在不合法时抛出 ValueError("--name must be a bare name")
    """
    if not isinstance(name, str):
        raise ValueError("--name must be a bare name")

    stripped = name.strip()
    if not stripped:
        raise ValueError("--name must be a bare name")

    # 拒绝显式路径分隔符或路径穿越
    if "/" in stripped or "\\" in stripped:
        raise ValueError("--name must be a bare name")
    if stripped in {".", ".."}:
        raise ValueError("--name must be a bare name")

    # 额外保障：Path.name 必须与 stripped 相同（防止奇异输入）
    if Path(stripped).name != stripped:
        raise ValueError("--name must be a bare name")

    return Path("out") / stripped / f"{stripped}.srt"


def default_pipeline_paths(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> PipelinePaths:
    input_path = Path(input_path).expanduser()
    srt_path = Path(output_path).expanduser() if output_path else default_output_path(input_path)
    if srt_path.suffix == "":
        srt_path = srt_path.with_suffix(".srt")
    base = srt_path.with_suffix("")
    return PipelinePaths(
        vocal_audio=base.with_name(f"{base.name}-vocal.ogg"),
        aligned_json=base.with_name(f"{base.name}-aligned.json"),
        stable_json=base.with_name(f"{base.name}-stable.json"),
        raw_srt=base.with_name(f"{base.name}-raw.srt"),
        translated_srt=base.with_name(f"{base.name}-translated.srt"),
        final_srt=srt_path,
        task_artifact_dir=base.with_name(f"{base.name}.llm-artifacts"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full production ASR subtitle pipeline."
    )
    parser.add_argument("input", help="Path to source audio/video, or a media URL.")
    parser.add_argument("-o", "--output", help="Path to final SRT output.")
    parser.add_argument(
        "--name",
        help=(
            "Output stem name (overrides auto-derived video ID or filename). "
            "Produces out/<name>/<name>.srt. Ignored if -o is given."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=tuple(PIPELINE_STAGE_ORDER),
        help=(
            "Run through this stage. Default is raw-srt; translated/final stages "
            "opt in to LLM correction/translation."
        ),
    )
    parser.add_argument(
        "--llm-correct-translate",
        action="store_true",
        help="Convenience switch equivalent to --stage final-srt when --stage is not set.",
    )
    parser.add_argument("--model", default=asr_align.DEFAULT_MODEL, help="Whisper model name.")
    parser.add_argument("--device", default="cuda", help="Device override (cpu/cuda).")
    parser.add_argument(
        "--gpu-budget-gb",
        type=int,
        choices=gpu_budget_choices(),
        default=DEFAULT_GPU_BUDGET_GB,
        help="GPU memory budget profile in GiB (default: 8).",
    )
    parser.add_argument("--language", default=None, help="Language override (e.g. ja, en). Use 'auto' or omit for auto-detection.")
    parser.add_argument(
        "--gap",
        type=float,
        default=asr_align.DEFAULT_GAP_SEC,
        help="Silence gap in seconds when combining segments.",
    )
    parser.add_argument(
        "--word",
        "-w",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write word-level SRT (default: False).",
    )
    parser.add_argument(
        "--asr-stabilize-profile",
        type=int,
        choices=asr_stabilize.SUPPORTED_ASR_STABILIZE_PROFILES,
        default=asr_stabilize.DEFAULT_ASR_STABILIZE_PROFILE,
        help=(
            "ASR stabilize profile for aligned -> stable: -1 no-op; "
            "0 default; 1 common hallucination cleanup; 2 noisy-span tags."
        ),
    )
    parser.add_argument(
        "--llm-route",
        choices=["text", "mm"],
        default="mm",
        help="LLM translation route for translated/final stages (default: mm).",
    )
    parser.add_argument(
        "--llm-level",
        choices=["low", "med", "high"],
        default="high",
        help=(
            "LLM route level (default: high). When the input is audio-only, "
            "high is automatically downgraded to med."
        ),
    )
    parser.add_argument(
        "--llm-fast",
        choices=["auto", "on", "off"],
        default="auto",
        help="LLM fast mode: fuse short inputs into one correction window (default: auto).",
    )
    parser.add_argument(
        "--llm-output-scale",
        type=float,
        default=1.0,
        help="Scale k on the LLM expected-output estimate; larger plans smaller windows.",
    )
    parser.add_argument(
        "--llm-video",
        help="Source video for --llm-route mm --llm-level high (required at that preset).",
    )
    parser.add_argument("--extra-info", default="", help="Extra info injected into LLM research.")
    parser.add_argument("--extra-info-file", help="Path to extra LLM research info.")
    parser.add_argument("--extra-style", default="", help="Extra translation style for LLM correction.")
    parser.add_argument("--no-web-search", action="store_true", help="Disable local web search for LLM stages.")
    parser.add_argument(
        "--knowledge",
        choices=["none", "collect", "update"],
        default="none",
        help=(
            "Knowledge switch for LLM stages: collect emits task_update_feedback; "
            "update additionally runs the unified knowledge update after correction."
        ),
    )
    parser.add_argument(
        "--refined-srt",
        help="User-refined SRT for the knowledge update (with --knowledge update).",
    )
    parser.add_argument("--knowledge-root", help="Override local knowledge base root for LLM stages.")
    parser.add_argument("--task-artifact-dir", help="Override LLM task artifact directory.")
    parser.add_argument("--task-id", default="", help="Stable task id for LLM artifacts.")
    parser.add_argument("--task-summary", default="", help="Task summary for knowledge update prompts.")
    parser.add_argument("--test-profile", action="store_true", help="Use the LLM test profile.")
    parser.add_argument(
        "--postprocess-profile",
        type=int,
        choices=SUPPORTED_POSTPROCESS_PROFILES,
        default=DEFAULT_POSTPROCESS_PROFILE,
        help=(
            "Final SRT postprocess profile: -1 semantic no-op re-render; "
            "0 duration then punctuation; 1 duration only; 2 punctuation only."
        ),
    )
    parser.add_argument(
        "--max-retries-per-window",
        type=int,
        default=5,
        help="Maximum LLM correction retry attempts per window.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help=(
            "Disable LLM session and correction-window checkpoint reads/writes "
            "(default: resume from the task artifact dir)."
        ),
    )
    return parser.parse_args()


def _use_or_create(
    path: Path,
    step_name: str,
    create: Callable[[Path], str | Path],
) -> Path:
    if path.exists():
        print(f"Skipping {step_name}; using existing output: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.part{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        produced = Path(create(temporary))
        if not produced.is_file():
            raise RuntimeError(
                f"{step_name} did not create its expected output: {produced}"
            )
        os.replace(produced, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def run_pipeline(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_name: str = asr_align.DEFAULT_MODEL,
    device: str = "cuda",
    language: Optional[str] = None,
    gap_sec: float = asr_align.DEFAULT_GAP_SEC,
    gpu_budget_gb: int = DEFAULT_GPU_BUDGET_GB,
    word: bool = False,
    asr_stabilize_profile: int = asr_stabilize.DEFAULT_ASR_STABILIZE_PROFILE,
    stage: str = "raw-srt",
    llm_route: str = "mm",
    llm_level: str = "high",
    llm_fast: str = "auto",
    llm_output_scale: float = 1.0,
    llm_video: str | Path | None = None,
    extra_info: str = "",
    extra_style: str = "",
    enable_web_search: bool = True,
    knowledge: str = "none",
    refined_srt: str | Path | None = None,
    knowledge_root: str | Path | None = None,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    task_summary: str = "",
    test_profile: bool = False,
    postprocess_profile: int = 0,
    max_retries_per_window: int = 5,
    resume: bool = True,
) -> PipelinePaths:
    if stage not in PIPELINE_STAGE_ORDER:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    if word and PIPELINE_STAGE_ORDER[stage] > PIPELINE_STAGE_ORDER["raw-srt"]:
        raise ValueError("--word can only be used through the raw-srt stage.")
    # Normalize "auto" to None: whisper uses None for auto-detection;
    # the string "auto" is not a valid language code and would raise.
    if language and language.strip().lower() == "auto":
        language = None
    source_arg = str(input_path)
    input_is_url = _is_media_url(source_arg)
    if input_is_url:
        source_path, paths, llm_video, source_extra_info = prepare_url_input(
            source_arg,
            output_path=output_path,
            llm_route=llm_route,
            llm_level=llm_level,
            llm_output_scale=llm_output_scale,
            llm_video=llm_video,
            stage=stage,
        )
        extra_info = "\n".join(
            part for part in (f"视频来源 URL: {source_arg}", source_extra_info, extra_info) if part
        )
    else:
        source_path = Path(input_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Input not found: {source_path}")
        paths = default_pipeline_paths(source_path, output_path)
        llm_level, llm_video, level_notice = resolve_llm_level_for_source(
            source_path,
            stage=stage,
            llm_route=llm_route,
            llm_level=llm_level,
            llm_video=llm_video,
        )
        if level_notice:
            print(level_notice)
    paths.final_srt.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    target_order = PIPELINE_STAGE_ORDER[stage]
    aligned_required = stage == "aligned" or (
        target_order >= PIPELINE_STAGE_ORDER["stable"]
        and not paths.stable_json.exists()
    )
    if target_order >= PIPELINE_STAGE_ORDER["vocal"]:
        # Vocal audio only feeds VAD-ASR. A stable JSON satisfies every later
        # consumer without requiring the aligned artifact to be backfilled.
        # When stable is absent, an existing aligned JSON likewise avoids both
        # vocal separation and VAD-ASR.
        vocal_needed = stage == "vocal" or (
            aligned_required and not paths.aligned_json.exists()
        )
        if vocal_needed:
            _use_or_create(
                paths.vocal_audio,
                "vocal separation",
                lambda temporary: vocal_separation.run_vocal_separation(
                    source_path,
                    output_path=temporary,
                    gpu_budget_gb=gpu_budget_gb,
                ),
            )
        elif not paths.resolve_vocal_audio().exists() and aligned_required:
            print(
                "Skipping vocal separation; aligned/stable JSON exists so vocal "
                f"audio is not needed: {paths.aligned_json if paths.aligned_json.exists() else paths.stable_json}"
            )
    if target_order >= PIPELINE_STAGE_ORDER["aligned"] and aligned_required:
        _use_or_create(
            paths.aligned_json,
            "VAD-ASR",
            lambda temporary: vad_asr.run_vad_asr(
                paths.resolve_vocal_audio(),
                output_path=temporary,
                model_name=model_name,
                device=device,
                language=language,
                gap_sec=gap_sec,
                gpu_budget_gb=gpu_budget_gb,
            ),
        )
    if target_order >= PIPELINE_STAGE_ORDER["stable"]:
        _use_or_create(
            paths.stable_json,
            "ASR stabilization",
            lambda temporary: asr_stabilize.run_asr_stabilize(
                paths.aligned_json,
                output_path=temporary,
                profile=asr_stabilize_profile,
            ),
        )
    if target_order >= PIPELINE_STAGE_ORDER["raw-srt"]:
        _use_or_create(
            paths.raw_srt,
            "raw SRT export",
            lambda temporary: to_srt.convert_json_to_srt(
                paths.stable_json,
                output_path=temporary,
                word=word,
            ),
        )
    if target_order >= PIPELINE_STAGE_ORDER["translated-srt"]:
        _run_llm_stage(
            paths=paths,
            source_path=source_path,
            stage=stage,
            llm_route=llm_route,
            llm_level=llm_level,
            llm_fast=llm_fast,
            llm_output_scale=llm_output_scale,
            llm_video=llm_video,
            extra_info=extra_info,
            extra_style=extra_style,
            enable_web_search=enable_web_search,
            knowledge=knowledge,
            refined_srt=refined_srt,
            knowledge_root=knowledge_root,
            task_artifact_dir=task_artifact_dir,
            task_id=task_id or paths.final_srt.stem,
            task_summary=task_summary,
            test_profile=test_profile,
            postprocess_profile=postprocess_profile,
            max_retries_per_window=max_retries_per_window,
            resume=resume,
        )
    completed = _stage_output(paths, stage)
    print(f"Pipeline complete: {completed}")
    print(f"Pipeline total_sec: {time.perf_counter() - t0:.3f}")
    return paths


def _is_media_url(value: str) -> bool:
    from llm.media_source import is_url

    return is_url(value)


def prepare_url_input(
    url: str,
    *,
    output_path: str | Path | None,
    llm_route: str,
    llm_level: str,
    llm_output_scale: float,
    llm_video: str | Path | None,
    stage: str,
) -> tuple[Path, PipelinePaths, str | Path | None, str]:
    """Resolve a media URL into (audio_path, paths, llm_video, extra_info).

    Public so the batch runner's download stage can reuse it; downloads audio
    (or video + extracted audio at mm/high) and derives the artifact paths from
    the resolved video id."""
    from llm.media_source import (
        DEFAULT_DATA_DIR,
        download_audio,
        download_video,
        extract_audio_from_video,
        resolve_video_id,
    )
    from llm.profiles import resolve_profile

    map_dir = DEFAULT_DATA_DIR
    video_id = resolve_video_id(url, map_dir)
    paths = default_pipeline_paths(Path(video_id), output_path)
    media_dir = paths.final_srt.parent
    profile = resolve_profile(llm_route, llm_level, output_scale=llm_output_scale)
    needs_llm_video = PIPELINE_STAGE_ORDER[stage] >= PIPELINE_STAGE_ORDER["translated-srt"]
    if needs_llm_video and profile.use_video and not llm_video:
        resolved_id, video_path = download_video(
            url, map_dir, video_id=video_id, target_dir=media_dir
        )
        audio_path = extract_audio_from_video(video_path)
        if resolved_id != video_id:
            paths = default_pipeline_paths(Path(resolved_id), output_path)
        return (
            audio_path.resolve(),
            paths,
            video_path,
            "媒体文件: " + str(audio_path) + "\nLLM 视频文件: " + str(video_path),
        )
    resolved_id, audio_path = download_audio(
        url, map_dir, video_id=video_id, target_dir=media_dir
    )
    if resolved_id != video_id:
        paths = default_pipeline_paths(Path(resolved_id), output_path)
    return audio_path.resolve(), paths, llm_video, "媒体文件: " + str(audio_path)


def _stage_output(paths: PipelinePaths, stage: str) -> Path:
    if stage == "vocal":
        return paths.vocal_audio
    if stage == "aligned":
        return paths.aligned_json
    if stage == "stable":
        return paths.stable_json
    if stage == "raw-srt":
        return paths.raw_srt
    if stage == "translated-srt":
        return paths.translated_srt
    return paths.final_srt


def _run_llm_stage(
    *,
    paths: PipelinePaths,
    source_path: Path,
    stage: str,
    llm_route: str = "mm",
    llm_level: str = "high",
    llm_fast: str = "auto",
    llm_output_scale: float = 1.0,
    llm_video: str | Path | None = None,
    extra_info: str,
    extra_style: str,
    enable_web_search: bool,
    knowledge: str,
    refined_srt: str | Path | None,
    knowledge_root: str | Path | None,
    task_artifact_dir: str | Path | None,
    task_id: str,
    task_summary: str,
    test_profile: bool,
    postprocess_profile: int,
    max_retries_per_window: int,
    resume: bool = True,
) -> None:
    from llm.profiles import resolve_profile

    profile = resolve_profile(llm_route, llm_level, output_scale=llm_output_scale)
    if llm_video and not profile.use_video:
        raise ValueError("--llm-video only applies to --llm-route mm --llm-level high")
    if profile.use_video and not llm_video:
        raise ValueError("--llm-video is required with --llm-route mm --llm-level high")
    artifact_dir = Path(task_artifact_dir).expanduser() if task_artifact_dir else paths.task_artifact_dir
    if stage == "translated-srt":
        if paths.translated_srt.exists():
            print(f"Skipping LLM correction/translation; using existing output: {paths.translated_srt}")
            return
        from llm.correction_translation import run_full_correction
        from llm.knowledge.base import DEFAULT_KNOWLEDGE_ROOT

        run_full_correction(
            stable_json=paths.stable_json,
            output_path=paths.final_srt,
            audio_path=source_path,
            video_path=llm_video,
            profile=profile,
            fast=llm_fast,
            extra_info=extra_info,
            knowledge_root=knowledge_root or DEFAULT_KNOWLEDGE_ROOT,
            enable_web_search=enable_web_search,
            postprocess_profile=None,
            extra_style=extra_style,
            task_id=task_id,
            task_summary=task_summary,
            task_artifact_dir=artifact_dir,
            knowledge=knowledge,
            refined_srt=refined_srt,
            test_profile=test_profile,
            max_retries_per_window=max_retries_per_window,
            resume=resume,
        )
        return

    if paths.final_srt.exists():
        print(f"Skipping LLM final SRT; using existing output: {paths.final_srt}")
        return
    if paths.translated_srt.exists():
        from llm.knowledge.base import append_task_artifact
        from llm.srt_postprocess import postprocess_srt_file
        from llm.task_report import write_task_report

        report = postprocess_srt_file(
            paths.translated_srt,
            output_path=paths.final_srt,
            profile=postprocess_profile,
        )
        append_task_artifact(
            artifact_dir,
            kind="final_srt",
            task_id=task_id,
            payload={
                "path": str(paths.final_srt),
                "raw_path": str(paths.raw_srt),
                "translated_path": str(paths.translated_srt),
                "postprocess": report.to_dict(),
            },
        )
        write_task_report(
            artifact_dir,
            task_id=task_id,
            outputs={
                "raw_srt": str(paths.raw_srt),
                "translated_srt": str(paths.translated_srt),
                "final_srt": str(paths.final_srt),
            },
        )
        print(f"Wrote {paths.final_srt}")
        return

    from llm.correction_translation import run_full_correction
    from llm.knowledge.base import DEFAULT_KNOWLEDGE_ROOT

    run_full_correction(
        stable_json=paths.stable_json,
        output_path=paths.final_srt,
        audio_path=source_path,
        video_path=llm_video,
        profile=profile,
        fast=llm_fast,
        extra_info=extra_info,
        knowledge_root=knowledge_root or DEFAULT_KNOWLEDGE_ROOT,
        enable_web_search=enable_web_search,
        postprocess_profile=postprocess_profile,
        extra_style=extra_style,
        task_id=task_id,
        task_summary=task_summary,
        task_artifact_dir=artifact_dir,
        knowledge=knowledge,
        refined_srt=refined_srt,
        test_profile=test_profile,
        max_retries_per_window=max_retries_per_window,
        resume=resume,
    )


def main() -> int:
    args = parse_args()
    try:
        stage = args.stage or ("final-srt" if args.llm_correct_translate else "raw-srt")
        output_path = args.output
        if output_path is None and args.name:
            output_path = resolve_name_output_path(args.name)
        extra_info = args.extra_info.strip()
        if args.extra_info_file:
            file_info = Path(args.extra_info_file).expanduser().read_text(encoding="utf-8").strip()
            extra_info = "\n".join(part for part in (extra_info, file_info) if part)
        run_pipeline(
            args.input,
            output_path=output_path,
            model_name=args.model,
            device=args.device,
            language=args.language,
            gap_sec=args.gap,
            gpu_budget_gb=args.gpu_budget_gb,
            word=args.word,
            asr_stabilize_profile=args.asr_stabilize_profile,
            stage=stage,
            llm_route=args.llm_route,
            llm_level=args.llm_level,
            llm_fast=args.llm_fast,
            llm_output_scale=args.llm_output_scale,
            llm_video=args.llm_video,
            extra_info=extra_info,
            extra_style=args.extra_style,
            enable_web_search=not args.no_web_search,
            knowledge=args.knowledge,
            refined_srt=args.refined_srt,
            knowledge_root=args.knowledge_root,
            task_artifact_dir=args.task_artifact_dir,
            task_id=args.task_id,
            task_summary=args.task_summary,
            test_profile=args.test_profile,
            postprocess_profile=args.postprocess_profile,
            max_retries_per_window=args.max_retries_per_window,
            resume=args.resume,
        )
    except Exception as exc:
        # str(exc) alone is often empty (bare RuntimeError, CUDA/driver errors),
        # which used to make a failed run indistinguishable from a silent exit.
        print(str(exc).strip() or repr(exc), file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

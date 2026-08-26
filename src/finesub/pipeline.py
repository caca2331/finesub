"""Production pipeline: vocal separation -> aligned ASR -> stable ASR -> SRT."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional

from .paths import resolve_logs_dir, resolve_name_output_path
from .reporting import (
    LEVELS,
    current_reporter,
    FanOutReporter,
    FileReporter,
    quieted_libraries,
    reporting_to,
    terminal_reporter,
)
from .speech.recognition import transcribe as asr_align
from .speech.postprocessing import stabilization as asr_stabilize
from .subtitles.postprocess import (
    DEFAULT_POSTPROCESS_PROFILE,
    SUPPORTED_POSTPROCESS_PROFILES,
)
from .speech.runtime.resources import (
    DEFAULT_GPU_BUDGET_GB,
    gpu_budget_choices,
)
from .run_metadata import (
    metadata_path_for_output,
    stage_record,
    summarize_llm_rounds,
    update_run_metadata,
)
from .subtitles import rendering as to_srt
from .speech.recognition import vad_asr_stage as vad_asr
from .speech.preprocessing.separator import separation as vocal_separation


class PipelinePaths(NamedTuple):
    vocal_audio: Path
    aligned_json: Path
    stable_json: Path
    raw_srt: Path
    translated_srt: Path
    final_srt: Path
    task_artifact_dir: Path
    metadata_json: Path

    @property
    def srt(self) -> Path:
        return self.final_srt

    def resolve_vocal_audio(self) -> Path:
        """Return the existing vocal file, preferring the ASR delivery.

        The pipeline always asks separation for `.ogg` (16 kHz mono, what every
        reader of it resamples to anyway). A `.flac` beside it is the lossless
        delivery -- separation's other mode, produced by a direct call -- and
        reading it is equally valid, just larger.
        """
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


def resolve_llm_media_for_source(
    source_path: Path,
    *,
    stage: str,
    llm_media: str,
    llm_video: str | Path | None,
    llm_correction_media: str = "",
    llm_planning_media: str = "",
) -> tuple[str, str | Path | None, str]:
    """Pick the effective LLM media switches / video file for a local input.

    A video switch needs a video track: a local video file becomes the default
    ``llm_video``. When the input is audio-only, the *default-derived*
    convenience ``--llm-media video`` downgrades to audio with a notice; an
    **explicit** per-task override asking for video is a configuration error
    (plan v2 D20 -- a switch above the available media never silently
    downgrades). Only meaningful once an LLM stage actually runs, so earlier
    stages are left untouched. Returns ``(llm_media, llm_video, notice)``;
    an empty notice means silence. The per-task overrides pass through
    unchanged.
    """
    if PIPELINE_STAGE_ORDER[stage] < PIPELINE_STAGE_ORDER["translated-srt"]:
        return llm_media, llm_video, ""
    wants_video = "video" in {
        llm_correction_media or llm_media,
        llm_planning_media or llm_media,
    }
    if not wants_video:
        return llm_media, llm_video, ""
    if source_path.suffix.lower() in _VIDEO_EXTENSIONS:
        return llm_media, llm_video or source_path, ""
    if llm_video:
        # Audio-only ASR source, but the caller supplied the video separately
        # -- that is exactly what --llm-video is for (transcribe the .wav,
        # show the .mp4 to the LLM). Availability is about the video *source*,
        # not about the ASR input's suffix.
        return llm_media, llm_video, ""
    if "video" in {llm_correction_media, llm_planning_media}:
        raise ValueError(
            "input is audio-only and no --llm-video was given, but an explicit "
            "--llm-correction-media/--llm-planning-media asks for video; supply "
            "--llm-video, drop the override, or use a video input"
        )
    return (
        "audio",
        llm_video,
        "Note: input is audio-only; downgrading --llm-media video → audio.",
    )


def resolve_knowledge_switch(knowledge: str | None, llm_difficulty: str) -> str:
    """The one rule three front ends share.

    An unset switch is not a user statement, and ``difficulty=efficiency``
    disables knowledge by construction, so an unset switch resolves to ``none``
    there instead of handing the LLM layer a combination it refuses -- which it
    refuses *after* ASR has already run, the most expensive place to find out.
    Passing the switch explicitly on an efficiency run stays the hard error it
    was; this only decides what "unset" means.
    """

    if knowledge is not None:
        return knowledge
    return "none" if llm_difficulty == "efficiency" else "collect"


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
        metadata_json=metadata_path_for_output(srt_path),
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
        help="GPU memory budget profile in GiB (default: 4).",
    )
    parser.add_argument(
        "--separator-rate",
        type=int,
        choices=vocal_separation.SEPARATOR_SAMPLE_RATES,
        default=vocal_separation.DEFAULT_SEPARATOR_SAMPLE_RATE,
        help=(
            "Rate the vocal separator works at (default: 44100, the model's "
            "own). 32000 and 22050 cut the chunk count roughly in proportion "
            "and cost separation quality; neither is recommended -- see "
            "docs/separator-optimization.md. The vocal stage skips on the "
            "file existing, so switching rates means deleting the existing "
            "<stem>-vocal.* and everything downstream of it."
        ),
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
        "--vad-silero-assist",
        action="store_true",
        help=(
            "Two-signal post-pass over the energy VAD (un-suppress creep, "
            "drop ghosts, carve noise spans, restore seams). Opt-in for "
            "noisy separated vocals; see docs/vad-asr.md."
        ),
    )
    parser.add_argument(
        "--qwen-verify",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Second-model verification evidence at the vad-asr tail "
            "(auto = when transformers 5.x is installed; "
            "see docs/vad-asr.md)."
        ),
    )
    parser.add_argument(
        "--lang-redecode",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Inline language-vote-collapse redecode during alignment "
            "(default: auto; auto-language runs only; see docs/vad-asr.md)."
        ),
    )
    parser.add_argument(
        "--split-length-scale",
        type=float,
        default=None,
        help=(
            "How long a subtitle may get before the splitter buys a cut "
            "(0.6-1.6, default 1.0; below 1 = shorter subtitles). Overrides "
            "[segmentation] length_scale in config.toml. Takes effect in the "
            "vad-asr stage, so an existing *-aligned.json must be deleted for "
            "a rerun to see it."
        ),
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
        "--llm-media",
        choices=["text", "audio", "video"],
        default="audio",
        help=(
            "Convenience knob for both LLM media switches (default: audio). "
            "Pass video to send the correction window a video clip; an "
            "audio-only input downgrades that back to audio automatically. "
            "This is about what the model sees, not about what is downloaded "
            "-- a URL fetches the video regardless (see --no-download-video)."
        ),
    )
    parser.add_argument(
        "--llm-correction-media",
        choices=["text", "audio", "video"],
        default="",
        help=(
            "What the LLM correction window sees; overrides --llm-media for "
            "that task only."
        ),
    )
    parser.add_argument(
        "--llm-planning-media",
        choices=["text", "audio", "video"],
        default="",
        help=(
            "What the per-window LLM query round sees; overrides --llm-media "
            "for that task only."
        ),
    )
    parser.add_argument(
        "--llm-retrieval",
        choices=["none", "local", "native"],
        default="local",
        help="LLM retrieval switch for translated/final stages (default: local).",
    )
    parser.add_argument(
        "--llm-difficulty",
        choices=["quality", "intermediate", "efficiency"],
        default="quality",
        help=(
            "Which prompt/thinking cell the LLM stages use "
            "(default: quality)."
        ),
    )
    parser.add_argument(
        "--llm-continuity",
        choices=["serial", "parallel"],
        default="serial",
        help=(
            "LLM window continuity (default: serial). parallel dispatches "
            "correction windows concurrently, giving up the chained "
            "inter-window context (docs/llm_harness_behavior.md)."
        ),
    )
    parser.add_argument(
        "--llm-parallel-windows",
        type=int,
        default=4,
        help="Concurrency cap for --llm-continuity parallel (default: 4).",
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
        help="Source video for --llm-media video (required at that setting).",
    )
    parser.add_argument("--extra-info", default="", help="Extra info injected into LLM research.")
    parser.add_argument("--extra-info-file", help="Path to extra LLM research info.")
    parser.add_argument("--extra-style", default="", help="Extra translation style for LLM correction.")
    parser.add_argument(
        "--no-download-video",
        dest="download_video_source",
        action="store_false",
        help=(
            "For a URL input, download only the audio track. The default "
            "fetches the video, because the subtitles this produces usually "
            "end up burned into it and re-fetching means resolving the URL "
            "again. No effect on local inputs."
        ),
    )
    parser.add_argument(
        "--knowledge",
        choices=["none", "collect", "update"],
        default=None,
        help=(
            "Knowledge switch for LLM stages. Default collect: read/inject the "
            "base and emit task_update_feedback, without writing to it (the "
            "default resolves to none at --llm-difficulty efficiency, which "
            "disables knowledge by construction). none: the knowledge base is "
            "not read or injected at all; update: collect plus the unified "
            "knowledge update after correction."
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
            "0 t2s, overlap, duration, punctuation; 1 duration only; "
            "2 punctuation only; 3 t2s only; 4 overlap repair only."
        ),
    )
    parser.add_argument(
        "--max-retries-per-window",
        type=int,
        default=5,
        help=(
            "Tier 1 of the per-window LLM retry budget: repair retries "
            "within one session chain (an agent resumes the same "
            "conversation). Total calls per window = (this+1) x "
            "(--max-replacements-per-window+1)."
        ),
    )
    parser.add_argument(
        "--max-replacements-per-window",
        type=int,
        default=1,
        help=(
            "Tier 2 of the per-window LLM retry budget: fresh-session "
            "replacements after a chain's repair budget is spent (repair "
            "context dropped; agents start a new conversation)."
        ),
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
    parser.add_argument(
        "--log-level",
        choices=LEVELS,
        default=None,
        help=(
            "How much the run says on stderr: quiet (warnings and the result), "
            "normal (default: stages, progress and summaries), verbose (adds "
            "timing, resource use and per-step recovery detail). "
            "Overrides FINESUB_LOG_LEVEL."
        ),
    )
    return parser.parse_args()


def _use_or_create(
    path: Path,
    step_name: str,
    create: Callable[[Path], str | Path],
) -> Path:
    if path.exists():
        # The stage line already says "已有结果，跳过"; the path itself is
        # detail for someone diagnosing which artifact was picked up.
        current_reporter().debug(
            f"skipping {step_name}", {"existing": str(path)}
        )
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


def _record_stage_time(
    paths: "PipelinePaths",
    stage_timing: dict[str, Any],
    name: str,
    started: float,
    *,
    reused: bool,
) -> None:
    """Write one stage's elapsed time to the sidecar, and say it under verbose.

    Every stage, not only the ones that happened to grow their own timing
    block: `verbose` promises timing, and a run where two of four stages report
    none leaves the reader to work out the difference by subtraction.

    Reused stages are recorded without an elapsed time rather than with zero --
    "did not run" and "ran instantly" are different facts.
    """

    elapsed = None if reused else time.perf_counter() - started
    update_run_metadata(
        paths.metadata_json,
        {
            "timing": {
                "stages": {
                    name: _stage_record_for_current_run(
                        stage_timing,
                        name,
                        stage_record(
                            status="reused" if reused else "executed",
                            elapsed_sec=elapsed,
                        ),
                    )
                }
            }
        },
    )
    if elapsed is not None:
        current_reporter().debug(
            "stage timing", {"stage": name, "elapsed_sec": f"{elapsed:.3f}"}
        )


def _stage_record_for_current_run(
    prior_stages: dict[str, Any],
    name: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Keep an executed stage when a later pass of the same run reuses it."""

    prior = prior_stages.get(name)
    if (
        record.get("status") == "reused"
        and isinstance(prior, Mapping)
        and prior.get("status") == "executed"
    ):
        chosen = dict(prior)
    else:
        chosen = record
    prior_stages[name] = chosen
    return chosen


def run_pipeline(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_name: str = asr_align.DEFAULT_MODEL,
    device: str = "cuda",
    language: Optional[str] = None,
    gap_sec: float = asr_align.DEFAULT_GAP_SEC,
    gpu_budget_gb: int = DEFAULT_GPU_BUDGET_GB,
    separator_sample_rate: int | None = None,
    vad_silero_assist: bool = False,
    qwen_verify: str = "auto",
    lang_redecode: str = "auto",
    split_length_scale: float | None = None,
    word: bool = False,
    asr_stabilize_profile: int = asr_stabilize.DEFAULT_ASR_STABILIZE_PROFILE,
    stage: str = "raw-srt",
    llm_media: str = "audio",
    llm_correction_media: str = "",
    llm_planning_media: str = "",
    llm_retrieval: str = "local",
    llm_difficulty: str = "quality",
    llm_continuity: str = "serial",
    llm_parallel_windows: int = 4,
    llm_fast: str = "auto",
    llm_output_scale: float = 1.0,
    llm_video: str | Path | None = None,
    extra_info: str = "",
    extra_style: str = "",
    download_video_source: bool = True,
    knowledge: str | None = None,
    refined_srt: str | Path | None = None,
    knowledge_root: str | Path | None = None,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    task_summary: str = "",
    test_profile: bool = False,
    postprocess_profile: int = 0,
    max_retries_per_window: int = 5,
    max_replacements_per_window: int = 1,
    resume: bool = True,
    _run_started_monotonic: float | None = None,
    _prior_timing: Mapping[str, Any] | None = None,
    _batch_workers: Mapping[str, int] | None = None,
) -> PipelinePaths:
    # Taken from the caller's binding rather than passed in: the stages deep
    # inside this call tree read it the same way, and two mechanisms for one
    # thing is how they drift apart. A front end wraps its run in
    # `reporting_to(...)`; nothing bound means nothing shown.
    reporter = current_reporter()
    run_t0 = (
        float(_run_started_monotonic)
        if _run_started_monotonic is not None
        else time.perf_counter()
    )
    stage_timing: dict[str, Any] = dict(_prior_timing or {})
    if stage not in PIPELINE_STAGE_ORDER:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    if word and PIPELINE_STAGE_ORDER[stage] > PIPELINE_STAGE_ORDER["raw-srt"]:
        raise ValueError("--word can only be used through the raw-srt stage.")
    knowledge = resolve_knowledge_switch(knowledge, llm_difficulty)
    # Normalize "auto" to None: whisper uses None for auto-detection;
    # the string "auto" is not a valid language code and would raise.
    if language and language.strip().lower() == "auto":
        language = None
    source_arg = str(input_path)
    input_is_url = _is_media_url(source_arg)
    if input_is_url:
        download_t0 = time.perf_counter()
        source_path, paths, llm_video, source_extra_info = prepare_url_input(
            source_arg,
            output_path=output_path,
            llm_media=llm_media,
            llm_correction_media=llm_correction_media,
            llm_planning_media=llm_planning_media,
            llm_retrieval=llm_retrieval,
            llm_difficulty=llm_difficulty,
            llm_output_scale=llm_output_scale,
            llm_video=llm_video,
            download_video_source=download_video_source,
        )
        extra_info = "\n".join(
            part for part in (f"视频来源 URL: {source_arg}", source_extra_info, extra_info) if part
        )
        stage_timing["download"] = stage_record(
            status="executed",
            elapsed_sec=time.perf_counter() - download_t0,
        )
    else:
        source_path = Path(input_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Input not found: {source_path}")
        paths = default_pipeline_paths(source_path, output_path)
        llm_media, llm_video, media_notice = resolve_llm_media_for_source(
            source_path,
            stage=stage,
            llm_media=llm_media,
            llm_video=llm_video,
            llm_correction_media=llm_correction_media,
            llm_planning_media=llm_planning_media,
        )
        if media_notice:
            # Says the LLM stage will not get the video it was asked for, which
            # changes the result rather than merely describing the run.
            current_reporter().warning("media-downgraded", media_notice)
        # A video source goes on to the stages as it is, like the URL branch
        # above. Separation decodes its own lossless copy when soundfile cannot
        # read the container, and the LLM clip cutter runs ffmpeg either way;
        # narrowing the audio here used to cost a lossy generation and a mono
        # 16 kHz downmix before separation -- which runs a 44.1 kHz stereo
        # model -- for an artifact nothing needed.
    paths.final_srt.parent.mkdir(parents=True, exist_ok=True)
    resolved_task_artifact_dir = (
        Path(task_artifact_dir).expanduser()
        if task_artifact_dir is not None
        else paths.task_artifact_dir
    )
    if resolved_task_artifact_dir != paths.task_artifact_dir:
        paths = paths._replace(task_artifact_dir=resolved_task_artifact_dir)

    from .media.ffmpeg import describe_ffmpeg

    update_run_metadata(
        paths.metadata_json,
        {
            "task_id": task_id or paths.final_srt.stem,
            "source": source_arg,
            "timing": {"stages": stage_timing},
            # The launchers may reuse a system ffmpeg instead of the managed
            # one, so which binary ran is no longer implied by the install.
            "tools": {"ffmpeg": describe_ffmpeg()},
            **(
                {"workers": {"batch": dict(_batch_workers)}}
                if _batch_workers
                else {}
            ),
        },
    )
    def announce(stage_key: str, status: str) -> None:
        """Report which stage the run is entering, and whether it does work.

        Deliberately at the branch level rather than inside `_use_or_create`:
        a stage skipped outright never reaches that helper (see the `elif`
        arms below), the LLM stage does not go through it at all, and one of
        its call sites is not a pipeline stage in the first place. Reporting
        from the branches is also what keeps this in step with the
        executed/reused status written into the run metadata.
        """

        reporter.stage_started(stage_key, reused=status == "reused")

    target_order = PIPELINE_STAGE_ORDER[stage]
    # Every stage at or below the target may be announced, reused ones
    # included -- which is exactly the denominator a person watching wants.
    reporter.planned(
        [name for name, order in PIPELINE_STAGE_ORDER.items() if order <= target_order]
    )
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
            # Either delivery counts as done, which is why this asks
            # `resolve_vocal_audio` rather than looking for the shape this stage
            # happens to write. `.ogg` is what it asks separation for; a `.flac`
            # beside it is separation's lossless mode, and every reader below
            # takes it. Checking only the `.ogg` meant a run that already held
            # the lossless track paid for the whole GPU stage again to end up
            # with a smaller copy of what it had.
            existing_vocal = paths.resolve_vocal_audio()
            vocal_existed = existing_vocal.exists()
            announce("vocal", "reused" if vocal_existed else "running")
            vocal_t0 = time.perf_counter()
            separator_metadata: dict[str, Any] = {}
            if vocal_existed:
                reporter.debug(
                    "skipping vocal separation", {"existing": str(existing_vocal)}
                )
            else:
                _use_or_create(
                    paths.vocal_audio,
                    "vocal separation",
                    lambda temporary: vocal_separation.run_vocal_separation(
                        source_path,
                        output_path=temporary,
                        gpu_budget_gb=gpu_budget_gb,
                        separator_sample_rate=separator_sample_rate,
                        metadata_sink=separator_metadata,
                        run_metadata_path=paths.metadata_json,
                    ),
                )
            _record_stage_time(
                paths,
                stage_timing,
                "vocal_separation",
                vocal_t0,
                reused=vocal_existed,
            )
            if separator_metadata:
                update_run_metadata(
                    paths.metadata_json,
                    {"workers": {"vocal_separation": separator_metadata}},
                )
        elif not paths.resolve_vocal_audio().exists() and aligned_required:
            # Not "reused" in the strict sense -- the vocal audio is not there
            # and is not wanted. For the progress list the two are the same
            # thing: nothing runs, and the stage is behind us.
            announce("vocal", "reused")
            reporter.debug(
                "skipping vocal separation; a later artifact already exists",
                {
                    "existing": str(
                        paths.aligned_json
                        if paths.aligned_json.exists()
                        else paths.stable_json
                    )
                },
            )
        elif not vocal_needed:
            announce("vocal", "reused")
            update_run_metadata(
                paths.metadata_json,
                {
                    "timing": {
                        "stages": {
                            "vocal_separation": _stage_record_for_current_run(
                                stage_timing,
                                "vocal_separation",
                                stage_record(status="reused"),
                            )
                        }
                    }
                },
            )
    if target_order >= PIPELINE_STAGE_ORDER["aligned"] and aligned_required:
        aligned_existed = paths.aligned_json.exists()
        announce("aligned", "reused" if aligned_existed else "running")
        asr_t0 = time.perf_counter()
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
                vad_silero_assist=vad_silero_assist,
                qwen_verify=qwen_verify,
                lang_redecode=lang_redecode,
                split_length_scale=split_length_scale,
                run_metadata_path=paths.metadata_json,
            ),
        )
        _record_stage_time(
            paths, stage_timing, "asr", asr_t0, reused=aligned_existed
        )
    elif target_order >= PIPELINE_STAGE_ORDER["aligned"]:
        announce("aligned", "reused")
        update_run_metadata(
            paths.metadata_json,
            {
                "timing": {
                    "stages": {
                        "asr": _stage_record_for_current_run(
                            stage_timing,
                            "asr",
                            stage_record(status="reused"),
                        )
                    }
                },
            },
        )
    if target_order >= PIPELINE_STAGE_ORDER["stable"]:
        stable_existed = paths.stable_json.exists()
        announce("stable", "reused" if stable_existed else "running")
        stable_t0 = time.perf_counter()
        _use_or_create(
            paths.stable_json,
            "ASR stabilization",
            lambda temporary: asr_stabilize.run_asr_stabilize(
                paths.aligned_json,
                output_path=temporary,
                profile=asr_stabilize_profile,
            ),
        )
        _record_stage_time(
            paths, stage_timing, "stabilize", stable_t0, reused=stable_existed
        )
    if target_order >= PIPELINE_STAGE_ORDER["raw-srt"]:
        raw_srt_existed = paths.raw_srt.exists()
        announce("raw-srt", "reused" if raw_srt_existed else "running")
        raw_srt_t0 = time.perf_counter()
        from .subtitles.postprocess import (
            TIMELINE_POSTPROCESS_PROFILES,
            postprocess_srt_file,
        )

        def _create_raw_srt(temporary: Path) -> Path:
            produced = to_srt.convert_json_to_srt(
                paths.stable_json,
                output_path=temporary,
                word=word,
            )
            if postprocess_profile in (0, 1):
                # Timeline half of the final profile; the ASR text stays as is.
                for timeline_profile in TIMELINE_POSTPROCESS_PROFILES:
                    postprocess_srt_file(produced, profile=timeline_profile)
            return Path(produced)

        _use_or_create(
            paths.raw_srt,
            "raw SRT export",
            _create_raw_srt,
        )
        _record_stage_time(
            paths, stage_timing, "raw_srt", raw_srt_t0, reused=raw_srt_existed
        )
    if target_order >= PIPELINE_STAGE_ORDER["translated-srt"]:
        llm_existed = paths.translated_srt.exists() or paths.final_srt.exists()
        # One call covers what the progress list shows as two stages, and which
        # of them actually runs depends on what is already on disk -- targeting
        # final-srt from nothing produces it in a single correction pass, with
        # no translated-srt in between.
        final_done = paths.final_srt.exists()
        translated_done = final_done or paths.translated_srt.exists()
        wants_final = target_order >= PIPELINE_STAGE_ORDER["final-srt"]
        announce("translated-srt", "reused" if translated_done else "running")
        if wants_final and translated_done:
            # Nothing but the post-processing tail is left, and that is what
            # the call below will do -- so this one is announced up front.
            announce("final-srt", "reused" if final_done else "running")
        llm_t0 = time.perf_counter()
        _run_llm_stage(
            paths=paths,
            source_path=source_path,
            stage=stage,
            llm_media=llm_media,
            llm_correction_media=llm_correction_media,
            llm_planning_media=llm_planning_media,
            llm_retrieval=llm_retrieval,
            llm_difficulty=llm_difficulty,
            llm_continuity=llm_continuity,
            llm_parallel_windows=llm_parallel_windows,
            llm_fast=llm_fast,
            llm_output_scale=llm_output_scale,
            llm_video=llm_video,
            extra_info=extra_info,
            extra_style=extra_style,
            knowledge=knowledge,
            refined_srt=refined_srt,
            knowledge_root=knowledge_root,
            task_artifact_dir=paths.task_artifact_dir,
            task_id=task_id or paths.final_srt.stem,
            task_summary=task_summary,
            test_profile=test_profile,
            postprocess_profile=postprocess_profile,
            max_retries_per_window=max_retries_per_window,
            max_replacements_per_window=max_replacements_per_window,
            resume=resume,
        )
        if wants_final and not translated_done:
            # The single correction pass above produced the final SRT itself,
            # post-processing included. This marks the transition; there is no
            # work left behind it.
            announce("final-srt", "running")
        _record_stage_time(
            paths, stage_timing, "llm_harness", llm_t0, reused=llm_existed
        )
        update_run_metadata(
            paths.metadata_json,
            {"llm_rounds": summarize_llm_rounds(paths.task_artifact_dir)},
        )
    completed = _stage_output(paths, stage)
    total_sec = time.perf_counter() - run_t0
    update_run_metadata(
        paths.metadata_json,
        {
            "timing": {"total_sec": round(max(0.0, total_sec), 3)},
            "completed_stage": stage,
            "completed_output": str(completed),
        },
    )
    if target_order >= PIPELINE_STAGE_ORDER["translated-srt"]:
        from finesub.llm.task_report import write_task_report

        write_task_report(
            paths.task_artifact_dir,
            task_id=task_id or paths.final_srt.stem,
            outputs={
                "raw_srt": str(paths.raw_srt),
                "translated_srt": str(paths.translated_srt),
                **(
                    {"final_srt": str(paths.final_srt)}
                    if paths.final_srt.exists()
                    else {}
                ),
            },
            run_metadata_path=paths.metadata_json,
        )
    reporter.completed(completed, total_sec)
    return paths


def _is_media_url(value: str) -> bool:
    from .media.source import is_url

    return is_url(value)


def prepare_url_input(
    url: str,
    *,
    output_path: str | Path | None,
    llm_media: str,
    llm_retrieval: str,
    llm_difficulty: str,
    llm_output_scale: float,
    llm_video: str | Path | None,
    llm_correction_media: str = "",
    llm_planning_media: str = "",
    download_video_source: bool = True,
) -> tuple[Path, PipelinePaths, str | Path | None, str]:
    """Resolve a media URL into (audio_path, paths, llm_video, extra_info).

    Public so the batch runner's download stage can reuse it; derives the
    artifact paths from the resolved video id.

    A URL fetches the **video** by default, at every stage. The subtitles this
    pipeline produces are usually going to be burned into that video, so the
    file is wanted whether or not an LLM stage runs and whether or not a media
    switch asks to *show* it to a model -- and re-fetching it later means
    resolving the same URL a second time, which is exactly what an id map
    exists to avoid. ``download_video_source=False`` (``--no-download-video``)
    takes the audio-only path for a run that really only wants a transcript.
    """
    from .media.source import (
        download_audio,
        download_video,
        resolve_video_id,
    )
    from .paths import resolve_reference_data_root
    from finesub.llm.routing.profiles import resolve_profile

    map_dir = resolve_reference_data_root()
    video_id = resolve_video_id(url, map_dir)
    paths = default_pipeline_paths(Path(video_id), output_path)
    media_dir = paths.final_srt.parent
    profile = resolve_profile(
        llm_media,
        llm_retrieval,
        llm_difficulty,
        correction_media=llm_correction_media,
        planning_media=llm_planning_media,
        output_scale=llm_output_scale,
    )
    # `profile` is resolved above for its validation side effect: a conflicting
    # switch vector must fail here, before anything is downloaded.
    del profile
    if download_video_source and not llm_video:
        resolved_id, video_path = download_video(
            url, map_dir, video_id=video_id, target_dir=media_dir
        )
        # The video is the source: separation makes its own lossless copy when it
        # needs one, and the LLM clip cutter runs ffmpeg either way. Extracting a
        # narrowed audio track here would only cost a generation.
        if resolved_id != video_id:
            paths = default_pipeline_paths(Path(resolved_id), output_path)
        return (
            video_path.resolve(),
            paths,
            video_path,
            "媒体文件: " + str(video_path) + "\nLLM 视频文件: " + str(video_path),
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
    llm_media: str = "audio",
    llm_correction_media: str = "",
    llm_planning_media: str = "",
    llm_retrieval: str = "local",
    llm_difficulty: str = "quality",
    llm_continuity: str = "serial",
    llm_parallel_windows: int = 4,
    llm_fast: str = "auto",
    llm_output_scale: float = 1.0,
    llm_video: str | Path | None = None,
    extra_info: str,
    extra_style: str,
    knowledge: str,
    refined_srt: str | Path | None,
    knowledge_root: str | Path | None,
    task_artifact_dir: str | Path | None,
    task_id: str,
    task_summary: str,
    test_profile: bool,
    postprocess_profile: int,
    max_retries_per_window: int,
    max_replacements_per_window: int = 1,
    resume: bool = True,
) -> None:
    from finesub.llm.routing.profiles import resolve_profile

    profile = resolve_profile(
        llm_media,
        llm_retrieval,
        llm_difficulty,
        llm_continuity,
        correction_media=llm_correction_media,
        planning_media=llm_planning_media,
        output_scale=llm_output_scale,
    )
    if llm_video and not profile.uses_video:
        raise ValueError("--llm-video only applies when a media switch is video")
    if profile.uses_video and not llm_video:
        raise ValueError("--llm-video is required when a media switch is video")
    artifact_dir = Path(task_artifact_dir).expanduser() if task_artifact_dir else paths.task_artifact_dir
    if stage == "translated-srt":
        if paths.translated_srt.exists():
            current_reporter().debug(
                "skipping LLM correction/translation",
                {"existing": str(paths.translated_srt)},
            )
            return
        from finesub.llm.correction_translation import run_full_correction
        from finesub.llm.knowledge.base import DEFAULT_KNOWLEDGE_ROOT

        run_full_correction(
            stable_json=paths.stable_json,
            output_path=paths.final_srt,
            audio_path=source_path,
            video_path=llm_video,
            profile=profile,
            fast=llm_fast,
            extra_info=extra_info,
            knowledge_root=knowledge_root or DEFAULT_KNOWLEDGE_ROOT,
            parallel_windows=llm_parallel_windows,
            postprocess_profile=None,
            extra_style=extra_style,
            task_id=task_id,
            task_summary=task_summary,
            task_artifact_dir=artifact_dir,
            knowledge=knowledge,
            refined_srt=refined_srt,
            test_profile=test_profile,
            max_retries_per_window=max_retries_per_window,
            max_replacements_per_window=max_replacements_per_window,
            resume=resume,
        )
        return

    if paths.final_srt.exists():
        current_reporter().debug(
            "skipping LLM final SRT", {"existing": str(paths.final_srt)}
        )
        return
    if paths.translated_srt.exists():
        from finesub.llm.knowledge.base import append_task_artifact
        from .subtitles.postprocess import postprocess_srt_file
        from finesub.llm.task_report import write_task_report

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
        # The delivered artifact is announced once, by `completed`.
        return

    from finesub.llm.correction_translation import run_full_correction
    from finesub.llm.knowledge.base import DEFAULT_KNOWLEDGE_ROOT

    run_full_correction(
        stable_json=paths.stable_json,
        output_path=paths.final_srt,
        audio_path=source_path,
        video_path=llm_video,
        profile=profile,
        fast=llm_fast,
        extra_info=extra_info,
        knowledge_root=knowledge_root or DEFAULT_KNOWLEDGE_ROOT,
        parallel_windows=llm_parallel_windows,
        postprocess_profile=postprocess_profile,
        extra_style=extra_style,
        task_id=task_id,
        task_summary=task_summary,
        task_artifact_dir=artifact_dir,
        knowledge=knowledge,
        refined_srt=refined_srt,
        test_profile=test_profile,
        max_retries_per_window=max_retries_per_window,
        max_replacements_per_window=max_replacements_per_window,
        resume=resume,
    )


@contextmanager
def _run_log(stem: str) -> Iterator[FileReporter | None]:
    """A per-run verbose log beside the other user data, or nothing.

    One file per run rather than one shared file: the desktop worker and a CLI
    run can be in flight at once (docs/cross-frontend-lease.md), and separate
    files need no locking and never interleave two runs.
    """

    from finesub_bootstrap.logs import open_log, prune

    directory = resolve_logs_dir()
    prune(directory)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]+", "-", stem)[:60].strip("-") or "run"
    handle = open_log(directory, f"run-{stamp}-{safe}.log")
    if handle is None:
        yield None
        return
    try:
        # Line buffered: every completed line reaches the OS, so a log survives
        # the process being killed. Not fsync -- that guards against the OS
        # dying, costs a syscall per line, and buys nothing for reading a log
        # afterwards.
        handle.reconfigure(line_buffering=True)
        yield FileReporter(handle)
    finally:
        try:
            handle.close()
        except OSError:
            pass


def main() -> int:
    args = parse_args()
    reporter = terminal_reporter(level=args.log_level)
    terminal_level = reporter.level
    stage = args.stage or ("final-srt" if args.llm_correct_translate else "raw-srt")
    # The log context wraps the whole run *including* its failure path: the
    # exception handler writes the single most useful line, and with the file
    # already closed it reached the terminal only.
    with _run_log(Path(str(args.input)).stem) as file_reporter:
        if file_reporter is not None:
            reporter = FanOutReporter(reporter, file_reporter)
        try:
            output_path = args.output
            if output_path is None and args.name:
                output_path = resolve_name_output_path(args.name)
            extra_info = args.extra_info.strip()
            if args.extra_info_file:
                file_info = Path(args.extra_info_file).expanduser().read_text(encoding="utf-8").strip()
                extra_info = "\n".join(part for part in (extra_info, file_info) if part)
            # `quieted_libraries` keeps the *terminal* level on purpose: its
            # verbose branch also un-mutes third-party logging and tqdm, and a
            # progress bar captured into the log file is pure noise.
            with reporting_to(reporter), quieted_libraries(terminal_level):
                run_pipeline(
                    args.input,
                    output_path=output_path,
                    model_name=args.model,
                    device=args.device,
                    language=args.language,
                    gap_sec=args.gap,
                    gpu_budget_gb=args.gpu_budget_gb,
                    separator_sample_rate=args.separator_rate,
                    vad_silero_assist=args.vad_silero_assist,
                    qwen_verify=args.qwen_verify,
                    lang_redecode=args.lang_redecode,
                    split_length_scale=args.split_length_scale,
                    word=args.word,
                    asr_stabilize_profile=args.asr_stabilize_profile,
                    stage=stage,
                    llm_media=args.llm_media,
                    llm_correction_media=args.llm_correction_media,
                    llm_planning_media=args.llm_planning_media,
                    llm_retrieval=args.llm_retrieval,
                    llm_difficulty=args.llm_difficulty,
                    llm_continuity=args.llm_continuity,
                    llm_parallel_windows=args.llm_parallel_windows,
                    llm_fast=args.llm_fast,
                    llm_output_scale=args.llm_output_scale,
                    llm_video=args.llm_video,
                    extra_info=extra_info,
                    extra_style=args.extra_style,
                    download_video_source=args.download_video_source,
                    knowledge=args.knowledge,
                    refined_srt=args.refined_srt,
                    knowledge_root=args.knowledge_root,
                    task_artifact_dir=args.task_artifact_dir,
                    task_id=args.task_id,
                    task_summary=args.task_summary,
                    test_profile=args.test_profile,
                    postprocess_profile=args.postprocess_profile,
                    max_retries_per_window=args.max_retries_per_window,
                    max_replacements_per_window=args.max_replacements_per_window,
                    resume=args.resume,
                )
        except Exception as exc:
            # str(exc) alone is often empty (bare RuntimeError, CUDA/driver errors),
            # which used to make a failed run indistinguishable from a silent exit.
            reporter.failed(stage, str(exc).strip() or repr(exc))
            # The traceback is the diagnosis. It goes to the terminal as before,
            # and into the log file -- which is the one a user actually sends.
            if file_reporter is not None:
                file_reporter.block("traceback", traceback.format_exc())
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

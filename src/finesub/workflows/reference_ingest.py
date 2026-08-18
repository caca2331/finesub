"""Update the knowledge base from user-provided reference material.

Input is a list of tasks, each a pipe-delimited row ``srt | media | note |
preset | args``:

- **srt** (required): the user's refined SRT. In batch mode a bare name (no
  ``.srt`` suffix) resolves to ``<index-dir>/<name>.srt``; otherwise it is a
  path. In single mode it must be a path.
- **media**: a video/audio URL (yt-dlp) or a local file. In batch mode a bare
  name (no suffix) resolves to a same-named file in the index dir; otherwise a
  path/URL. Local media skips the download.
- **note**: free text injected into research (``extra_info``); may not contain ``|``.
- **preset**: a named bundle of settings (see ``PRESETS``); empty = ``mm-med``.
- **args**: freeform overrides parsed like CLI flags (``--media``/``--retrieval``/``--difficulty``/
  ``--fast``/``--output-scale``/``--video``/``--model``/``--language``/
  ``--gpu-budget-gb``/``--test-profile``/``--no-test-profile``),
  applied over the preset (row args win).

Two ways to supply tasks: ``--index <dir>`` reads ``<dir>/index.csv`` (one row
per line, ``#`` comments and blanks skipped); ``--task "<row>"`` (repeatable)
passes a single row inline.

Per task the tool downloads/loads the media, runs the full pipeline (vocal
separation -> VAD+ASR -> SRT -> LLM correction/translation with feedback
collection), then runs ONE unified knowledge update in the refined_aligned
mode (fed the refined SRT directly; it also maintains
knowledge/translation/common-mistake.md).

Unlike other finesub.llm.* CLIs this tool executes everything by default — the user
invoking it IS the opt-in (downloads, GPU pipeline, Gemini quota, knowledge
writes). Use --dry-run to only print the per-task plan, or --no-apply to run
everything but keep the knowledge base untouched (proposals stay in the
exchange logs for review). Reruns are cheap:
downloads, pipeline stages, research context and the zh SRT are all skipped
when their outputs already exist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shlex
import sys

from finesub.reporting import quieted_libraries
from finesub.speech.runtime.resources import (
    DEFAULT_GPU_BUDGET_GB,
    gpu_budget_choices,
)

from finesub.llm.correction_translation import run_full_correction
from finesub.llm.knowledge.base import DEFAULT_KNOWLEDGE_ROOT
from finesub.llm.knowledge.update import run_knowledge_update
from finesub.paths import resolve_reference_data_root
from finesub.media.source import (
    URL_MAP_FILENAME,
    download_audio,
    download_video,
    is_url,
    load_url_map as _load_url_map,
    resolve_video_id,
    sanitize_video_id as _sanitize_video_id,
    save_url_map as _save_url_map,
    url_map_path as _url_map_path,
)
from finesub.llm.routing.profiles import TranslationProfile, resolve_profile


DEFAULT_WORK_DIR = Path("out") / "reference"
INDEX_FILENAME = "index.csv"


# --- Presets ---------------------------------------------------------------
# A preset is a bundle of the same settings `args` can override. The default
# `mm-med` uses test_profile=True (all roles gemini-3.5-flash-lite) — cheap for
# knowledge/prompt iteration; `prod` runs the real models; `text` is the text
# route. Row `args` are applied on top (row wins).
PRESETS: dict[str, dict[str, object]] = {
    "mm-med": {"media": "audio", "retrieval": "local", "test_profile": True},
    "prod": {"media": "audio", "retrieval": "local", "test_profile": False},
    "text": {"media": "text", "retrieval": "none", "test_profile": False},
    "text-high": {"media": "text", "retrieval": "native", "test_profile": False},
    "mm-low": {"media": "text", "retrieval": "local", "test_profile": False},
    "mm-high": {"media": "video", "retrieval": "local", "test_profile": True},
}
DEFAULT_PRESET = "mm-med"


# --- Row parsing -----------------------------------------------------------
@dataclass(frozen=True)
class TaskRow:
    srt: str
    media: str
    note: str = ""
    preset: str = ""
    args: str = ""


def parse_row(raw: str) -> TaskRow:
    """Parse a `srt|media|note|preset|args` row (args may itself contain `|`)."""
    # maxsplit=4 keeps any '|' inside the trailing args field intact; note must
    # not contain '|' (it is field 3), per the documented schema.
    parts = raw.split("|", 4)
    parts += [""] * (5 - len(parts))
    srt, media, note, preset, args = (p.strip() for p in parts)
    if not srt:
        raise ValueError(f"row is missing the srt field: {raw!r}")
    return TaskRow(srt=srt, media=media, note=note, preset=preset, args=args)


def read_index_rows(index_path: Path) -> list[TaskRow]:
    """Read `index.csv`; skip blank lines and `#` comments."""
    if not index_path.exists():
        raise FileNotFoundError(f"index file not found: {index_path}")
    rows: list[TaskRow] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(parse_row(stripped))
    return rows


# --- Settings resolution (preset + args) -----------------------------------
@dataclass
class ResolvedSettings:
    media: str = "audio"
    retrieval: str = "local"
    difficulty: str = "quality"
    fast: str = "auto"
    output_scale: float = 1.0
    test_profile: bool = False
    model: str = "large-v3-turbo"
    language: str | None = None  # None = Whisper auto-detect
    gpu_budget_gb: int = DEFAULT_GPU_BUDGET_GB
    video: str = ""


def _row_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="row-args", add_help=False)
    parser.add_argument("--media", choices=["text", "audio", "video"])
    parser.add_argument("--retrieval", choices=["none", "local", "native"])
    parser.add_argument(
        "--difficulty", choices=["quality", "intermediate", "efficiency"]
    )
    parser.add_argument("--fast", choices=["auto", "on", "off"])
    parser.add_argument("--output-scale", type=float, dest="output_scale")
    parser.add_argument("--video")
    parser.add_argument("--model")
    parser.add_argument("--language")
    parser.add_argument(
        "--gpu-budget-gb",
        type=int,
        choices=gpu_budget_choices(),
        dest="gpu_budget_gb",
    )
    parser.add_argument("--test-profile", dest="test_profile", action="store_true", default=None)
    parser.add_argument("--no-test-profile", dest="test_profile", action="store_false")
    return parser


def resolve_settings(row: TaskRow, defaults: ResolvedSettings) -> ResolvedSettings:
    """Layer preset over global defaults, then apply row `args` on top."""
    preset_name = row.preset or DEFAULT_PRESET
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; known: {', '.join(sorted(PRESETS))}"
        )
    preset = PRESETS[preset_name]
    settings = ResolvedSettings(
        media=str(preset.get("media", defaults.media)),
        retrieval=str(preset.get("retrieval", defaults.retrieval)),
        difficulty=str(preset.get("difficulty", defaults.difficulty)),
        fast=str(preset.get("fast", defaults.fast)),
        output_scale=float(preset.get("output_scale", defaults.output_scale)),
        test_profile=bool(preset.get("test_profile", defaults.test_profile)),
        model=str(preset.get("model", defaults.model)),
        language=(
            str(lang) if (lang := preset.get("language", defaults.language)) is not None else None
        ),
        gpu_budget_gb=int(preset.get("gpu_budget_gb", defaults.gpu_budget_gb)),
        video=str(preset.get("video", defaults.video)),
    )
    if row.args:
        parsed, unknown = _row_arg_parser().parse_known_args(shlex.split(row.args))
        if unknown:
            raise ValueError(f"unrecognized args {unknown} in row args {row.args!r}")
        for key, value in vars(parsed).items():
            if value is not None:
                setattr(settings, key, value)
    return settings


@dataclass
class ResolvedTask:
    srt_path: Path
    media: str
    is_media_url: bool
    note: str
    settings: ResolvedSettings
    profile: TranslationProfile
    video_path: str


def resolve_srt(field_value: str, base_dir: Path | None) -> Path:
    if base_dir is not None and not field_value.lower().endswith(".srt") and not (
        "/" in field_value or "\\" in field_value
    ):
        return base_dir / f"{field_value}.srt"
    return Path(field_value).expanduser()


def resolve_media(field_value: str, base_dir: Path | None) -> tuple[bool, str]:
    """Return (is_url, resolved_media). Bare names (batch) glob the index dir."""
    if is_url(field_value):
        return True, field_value.strip()
    has_suffix = bool(Path(field_value).suffix)
    is_bare = base_dir is not None and not has_suffix and not (
        "/" in field_value or "\\" in field_value
    )
    if is_bare:
        matches = sorted(base_dir.glob(f"{field_value}.*"))
        if not matches:
            raise FileNotFoundError(
                f"no media file matching {field_value!r} under {base_dir}"
            )
        return False, str(matches[0])
    return False, str(Path(field_value).expanduser())


def build_task(row: TaskRow, base_dir: Path | None, defaults: ResolvedSettings) -> ResolvedTask:
    settings = resolve_settings(row, defaults)
    profile = resolve_profile(
        settings.media,
        settings.retrieval,
        settings.difficulty,
        output_scale=settings.output_scale,
    )
    srt_path = resolve_srt(row.srt, base_dir)
    is_media_url, media = resolve_media(row.media, base_dir) if row.media else (False, "")

    video_path = settings.video
    if profile.uses_video and not video_path:
        # mm-high needs a video. A local media file is used directly; a URL is
        # downloaded at process time (video_path stays empty here and is filled
        # by process_task). With no media and no --video there is no source.
        if media and not is_media_url:
            video_path = media
        elif not media:
            raise ValueError(
                "media=video needs a video, but the row has no media and "
                "no --video in the args"
            )
    return ResolvedTask(
        srt_path=srt_path,
        media=media,
        is_media_url=is_media_url,
        note=row.note,
        settings=settings,
        profile=profile,
        video_path=video_path,
    )


# --- yt-dlp / pipeline helpers --------------------------------------------
def resolve_media_source(task: ResolvedTask, data_dir: Path) -> tuple[str, Path]:
    """Return (video_id, audio_path): download a URL, or use a local media file."""
    if task.is_media_url:
        return download_audio(task.media, data_dir)
    media_path = Path(task.media).expanduser()
    if not media_path.exists():
        raise FileNotFoundError(f"media file not found: {media_path}")
    return _sanitize_video_id(media_path.stem), media_path


def run_reference_pipeline(
    audio_path: Path,
    *,
    video_id: str,
    work_dir: Path,
    model: str,
    language: str | None,
    gpu_budget_gb: int,
) -> Path:
    """Vocal separation -> VAD+ASR -> SRT via pipeline.run_pipeline; returns stable.json."""
    # Lazy import: pulls in torch and the ASR stack.
    from finesub import pipeline

    paths = pipeline.run_pipeline(
        audio_path,
        output_path=work_dir / video_id / f"{video_id}.srt",
        model_name=model,
        language=language,
        gpu_budget_gb=gpu_budget_gb,
        # The reference runner already gets its GPU concurrency from the
        # batch ASR bin. Keep each item single-worker exactly like batch._build_item;
        # otherwise N file workers each start N WT shards and multiply the
        # profile's model budget to N².
    )
    return paths.stable_json


def run_reference_knowledge_update(
    *,
    refined_srt: Path,
    final_srt: Path,
    stable_json: Path,
    artifact_dir: Path,
    video_id: str,
    knowledge_root: str | Path,
    test_profile: bool,
    apply: bool = True,
    difficulty: str = "quality",
) -> None:
    """One unified knowledge update in the refined_aligned mode.

    ``difficulty`` selects the knowledge cell, same as the correction
    pipeline's -- the ingest run's own setting, not the top tier by default.
    """

    report = run_knowledge_update(
        final_srt=final_srt,
        stable_json=stable_json,
        artifact_dir=artifact_dir,
        refined_srt=refined_srt,
        task_id=f"reference-ingest-{video_id}",
        task_summary=(
            f"参考素材导入 {video_id}：机器纠错翻译结果对照用户精修 SRT，"
            "用于知识库与常见翻译错误库更新。"
        ),
        knowledge_root=knowledge_root,
        test_profile=test_profile,
        apply=apply,
        difficulty=difficulty,
    )
    if apply:
        print(
            f"Applied knowledge update ({report['mode']}): "
            f"{len(report['chunks'])} chunk(s); ledger: {report['ledger_path']}"
        )
    else:
        # No apply -> no ledger writes; proposals live in the exchange logs.
        print(
            f"Generated knowledge update proposals without applying "
            f"({report['mode']}): {len(report['chunks'])} chunk(s); "
            f"see {artifact_dir / 'exchanges'}"
        )


def stage_download(task: ResolvedTask, args: argparse.Namespace) -> dict:
    """Bin 1: resolve/download media; returns the per-task context dict."""

    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    if task.is_media_url:
        video_id = resolve_video_id(task.media, data_dir)
        pair_dir = work_dir / video_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        if task.profile.uses_video and not task.video_path:
            _, video_path = download_video(
                task.media, data_dir, video_id=video_id, target_dir=pair_dir
            )
            # The video is the ASR source too: separation decodes its own
            # lossless copy when it needs one, and the LLM clip cutter runs
            # ffmpeg either way. Extracting an audio track here would only
            # cost a lossy generation before separation ever saw the signal.
            audio_path = video_path
            task.video_path = str(video_path)
        else:
            _, audio_path = download_audio(
                task.media, data_dir, video_id=video_id, target_dir=pair_dir
            )
    else:
        video_id, audio_path = resolve_media_source(task, data_dir)
        pair_dir = work_dir / video_id
        pair_dir.mkdir(parents=True, exist_ok=True)
    return {"video_id": video_id, "audio_path": audio_path, "pair_dir": pair_dir}


def stage_asr(ctx: dict, task: ResolvedTask, args: argparse.Namespace) -> dict:
    """Bin 2 (GPU): vocal separation -> VAD+ASR -> raw SRT."""

    settings = task.settings
    ctx["stable_json"] = run_reference_pipeline(
        ctx["audio_path"],
        video_id=ctx["video_id"],
        work_dir=Path(args.work_dir),
        model=settings.model,
        language=settings.language,
        gpu_budget_gb=settings.gpu_budget_gb,
    )
    return ctx


def stage_llm(ctx: dict, task: ResolvedTask, args: argparse.Namespace) -> dict:
    """Bin 3: LLM correction + the refined_aligned knowledge update.

    One indivisible unit on purpose: the update must land right after this
    task's correction and before the next task's (batch-order knowledge
    accumulation), which the serial, ordered llm bin guarantees."""

    settings = task.settings
    video_id = ctx["video_id"]
    final_srt = ctx["pair_dir"] / f"{video_id}.srt"
    artifact_dir = ctx["pair_dir"] / "llm-artifacts"
    if final_srt.exists():
        print(f"Skipping LLM correction; using existing output: {final_srt}")
    else:
        source_desc = f"视频来源 URL: {task.media}" if task.is_media_url else f"媒体文件: {task.media}"
        extra_info = f"{source_desc}\n{task.note}".strip() if task.note else source_desc
        run_full_correction(
            stable_json=ctx["stable_json"],
            output_path=final_srt,
            audio_path=ctx["audio_path"],
            video_path=task.video_path or None,
            extra_info=extra_info,
            profile=task.profile,
            fast=settings.fast,
            knowledge_root=args.knowledge_root,
            task_id=f"reference-ingest-{video_id}",
            task_summary=f"参考素材导入 {video_id} 的纠错翻译",
            task_artifact_dir=artifact_dir,
            knowledge="collect",
            test_profile=settings.test_profile,
        )

    run_reference_knowledge_update(
        refined_srt=task.srt_path,
        final_srt=final_srt,
        stable_json=Path(ctx["stable_json"]),
        artifact_dir=artifact_dir,
        video_id=video_id,
        knowledge_root=args.knowledge_root,
        test_profile=settings.test_profile,
        apply=args.apply,
        difficulty=settings.difficulty,
    )
    return ctx


def process_task(task: ResolvedTask, args: argparse.Namespace) -> None:
    """Sequential composition of the three stages (single-task equivalence)."""

    stage_llm(stage_asr(stage_download(task, args), task, args), task, args)


def print_task_plan(task: ResolvedTask, args: argparse.Namespace) -> None:
    s = task.settings
    media_kind = "URL" if task.is_media_url else "本地媒体"
    print(f"- 精修 SRT: {task.srt_path}")
    print(f"  {media_kind}: {task.media or '（无，仅用已有产物）'}")
    print(
        f"  设置: media={s.media} retrieval={s.retrieval} difficulty={s.difficulty} "
        f"fast={s.fast} scale={s.output_scale} "
        f"test_profile={s.test_profile} model={s.model} lang={s.language or 'auto'} "
        f"gpu={s.gpu_budget_gb}"
        + (f" video={task.video_path or '(从 URL 下载)'}" if task.profile.uses_video else "")
    )
    if task.note:
        print(f"  note: {task.note}")
    update_desc = "统一知识更新(refined_aligned)" if args.apply else "统一知识更新(refined_aligned, 只生成不写库)"
    print(f"  流程: (下载/载入媒体) -> pipeline -> LLM 纠错翻译(采集反馈) -> {update_desc}")


def _defaults_from_args(args: argparse.Namespace) -> ResolvedSettings:
    return ResolvedSettings(
        model=args.model,
        language=args.language,
        gpu_budget_gb=args.gpu_budget_gb,
    )


def gather_tasks(args: argparse.Namespace) -> list[ResolvedTask]:
    defaults = _defaults_from_args(args)
    tasks: list[ResolvedTask] = []
    if args.index:
        base = Path(args.index).expanduser()
        for row in read_index_rows(base / INDEX_FILENAME):
            tasks.append(build_task(row, base, defaults))
    for raw in args.task or []:
        tasks.append(build_task(parse_row(raw), None, defaults))
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update the knowledge base from reference material. Tasks are "
            "pipe-delimited rows `srt|media|note|preset|args`, supplied via "
            "--index <dir>/index.csv (batch) and/or repeated --task (single). "
            "Runs downloads, the GPU pipeline, LLM correction and knowledge "
            "updates by default; use --dry-run to only print the plan."
        )
    )
    parser.add_argument("--index", help="Directory containing index.csv (batch mode).")
    parser.add_argument(
        "--batch-id",
        default="",
        help="Batch id for the status log (default: reference-<timestamp>).",
    )
    parser.add_argument(
        "--task",
        action="append",
        metavar="ROW",
        help="A single 'srt|media|note|preset|args' row (repeatable).",
    )
    parser.add_argument("--model", default="large-v3-turbo", help="Default Whisper model.")
    parser.add_argument(
        "--language",
        default=None,
        help="Default ASR language (default: auto-detect; rows override via args).",
    )
    parser.add_argument(
        "--gpu-budget-gb",
        type=int,
        choices=gpu_budget_choices(),
        default=DEFAULT_GPU_BUDGET_GB,
        help="Default GPU budget (GiB).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(resolve_reference_data_root()),
        help="Downloaded source audio root.",
    )
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Generated artifacts root.")
    parser.add_argument(
        "--knowledge-root",
        default=(
            str(DEFAULT_KNOWLEDGE_ROOT)
            if DEFAULT_KNOWLEDGE_ROOT is not None
            else None
        ),
        help="Root of the local knowledge base (embedded git repo).",
    )
    parser.add_argument(
        "--no-apply",
        dest="apply",
        action="store_false",
        help=(
            "Run everything but keep the knowledge update read-only: proposals "
            "are generated and retained in the exchange logs without touching "
            "the knowledge base."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the per-task plan; no download, pipeline or LLM calls.",
    )
    args = parser.parse_args()
    if not args.index and not args.task:
        parser.error("provide at least one of --index or --task")
    return args


def main() -> int:
    args = parse_args()
    try:
        tasks = gather_tasks(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Failed to build task list: {exc}", file=sys.stderr)
        return 2

    for task in tasks:
        if not task.srt_path.exists():
            print(f"Refined SRT not found: {task.srt_path}", file=sys.stderr)
            return 2

    if args.dry_run:
        print(f"计划处理 {len(tasks)} 个任务：")
        for task in tasks:
            print_task_plan(task, args)
        return 0

    # Stage-parallel execution on the shared three-bin runner: downloads and
    # GPU ASR of later tasks overlap the LLM work of earlier ones, while the
    # ordered llm bin keeps knowledge accumulation in index order. A failing
    # task is recorded and skipped downstream instead of aborting the batch.
    from finesub import batch as batch_runner

    def _make_stages(task: ResolvedTask) -> dict:
        return {
            "download": lambda _payload, t=task: stage_download(t, args),
            "asr": lambda ctx, t=task: stage_asr(ctx, t, args),
            "llm": lambda ctx, t=task: stage_llm(ctx, t, args),
        }

    items = [
        batch_runner.BatchItem(
            label=f"{index}:{task.srt_path.stem}", stages=_make_stages(task)
        )
        for index, task in enumerate(tasks)
    ]
    # Single --task runs skip the status log (no batch to digest); multi-task
    # batches and explicit --batch-id get out/batch/<id>/batch-status.jsonl.
    status_path = None
    if len(items) > 1 or args.batch_id:
        batch_id = args.batch_id or f"reference-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        status_path = (
            batch_runner.DEFAULT_BATCH_ROOT / batch_id / batch_runner.STATUS_FILENAME
        )
        print(f"Batch {batch_id}: {len(items)} task(s); status -> {status_path}")
    # No asr override: batch_runner owns that decision now, and it runs one file
    # at a time so each file can take the whole profile. Overriding here would
    # multiply the per-file shard width by the number of concurrent files and
    # blow past the GPU budget.
    # Same quieting the manifest CLI applies: this runs the same pipeline
    # through the same stages, and without it audio-separator's INFO logging
    # and third-party progress bars bury the run's own report.
    with quieted_libraries("normal"):
        results = batch_runner.run_batch(
            items,
            workers={"asr": batch_runner.profile_asr_workers(())},
            status_path=status_path,
        )

    completed = sum(1 for r in results if r.status == "done")
    print(f"Completed {completed}/{len(results)} task(s).")
    for index, result in enumerate(results):
        if result.status != "done":
            print(
                f"  {result.status.upper()} {result.label} (任务 #{index + 1}，"
                f"可用 --task 单条重跑)"
                + (f" [{result.failed_stage}] {result.error}" if result.error else ""),
                file=sys.stderr,
            )
    return 0 if completed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

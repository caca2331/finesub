"""Three-bin batch runner: download -> asr -> llm.

Feeds many media sources through the production pipeline as a stage-parallel
pipeline with worker pools per bin (download=2, asr=resource profile, llm=1).
Items are isolated: one failing item is recorded and skipped downstream, the
rest keep flowing. Resume is inherited from the pipeline's exist-skip:
rerunning the same manifest fast-forwards finished stages.

The llm bin consumes items in submission order even when downloads finish out
of order, so knowledge accumulation across a batch stays reproducible. llm
concurrency is fixed at 1 by design: the model rate limiter
(finesub/llm/rate_limit.py) is per-instance in-memory state without locks, and
knowledge auto-apply commits to the embedded git repo — neither survives
concurrent llm tasks. Raising it needs a shared, locked limiter first.

CLI:
  python -m finesub.batch --manifest tasks.jsonl [global defaults]
  python -m finesub.batch URL_OR_PATH [URL_OR_PATH ...] [global defaults]

Manifest: JSONL, one item per line: {"source": <URL or path>, ...overrides}.
Row keys override the CLI-level defaults (see ALLOWED_ITEM_KEYS).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .speech.runtime.resources import (
    DEFAULT_GPU_BUDGET_GB,
    gpu_budget_choices,
)
from .reporting import LEVELS, TerminalReporter, quieted_libraries, reporting_to
from .run_metadata import load_run_metadata

BIN_ORDER = ("download", "asr", "llm")
DEFAULT_WORKERS = {"download": 2, "asr": 1, "llm": 1}
DEFAULT_ASR_QUEUE_SIZE = 4
DEFAULT_BATCH_ROOT = Path("out") / "batch"
STATUS_FILENAME = "batch-status.jsonl"


# --- engine ------------------------------------------------------------------
@dataclass
class BatchItem:
    label: str
    stages: Mapping[str, Callable[[Any], Any]]  # bin name -> fn(payload)->payload
    payload: Any = None


@dataclass
class ItemResult:
    label: str
    status: str = "pending"  # pending -> done | failed | skipped
    failed_stage: str = ""
    error: str = ""
    payload: Any = None


class _StatusLog:
    def __init__(self, path: Optional[Path]) -> None:
        self._path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, **event: Any) -> None:
        if self._path is None:
            return
        event.setdefault(
            "ts", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class _OrderedGate:
    """Hands out item indices strictly in item order as they become ready."""

    def __init__(self, total: int) -> None:
        self._cond = threading.Condition()
        self._ready: set[int] = set()
        self._next = 0
        self._total = total

    def submit(self, index: int) -> None:
        with self._cond:
            self._ready.add(index)
            self._cond.notify_all()

    def next(self) -> Optional[int]:
        with self._cond:
            while True:
                if self._next >= self._total:
                    return None
                if self._next in self._ready:
                    index = self._next
                    self._next += 1
                    return index
                self._cond.wait()


def _item_reporter(label: str, log_level: str) -> TerminalReporter:
    """A renderer that says which item it is speaking for.

    Line mode, never redrawn in place: several items share this terminal, and a
    progress line rewritten by whichever one spoke last leaves neither
    readable.
    """

    return TerminalReporter(
        sys.stderr,
        level=log_level if log_level in LEVELS else "normal",
        isatty=False,
        prefix=f"[{label}] ",
    )


def run_batch(
    items: Sequence[BatchItem],
    *,
    workers: Mapping[str, int] | None = None,
    asr_queue_size: int = DEFAULT_ASR_QUEUE_SIZE,
    status_path: str | Path | None = None,
    stop_event: threading.Event | None = None,
    log_level: str = "normal",
) -> list[ItemResult]:
    """Run items through the three bins; returns one ItemResult per item."""

    counts = dict(DEFAULT_WORKERS)
    counts.update(workers or {})
    for name in BIN_ORDER:
        if counts.get(name, 0) < 1:
            raise ValueError(f"workers[{name!r}] must be >= 1")

    total = len(items)
    results = [ItemResult(label=item.label, payload=item.payload) for item in items]
    status = _StatusLog(Path(status_path) if status_path else None)
    stop = stop_event or threading.Event()

    download_q: queue.Queue = queue.Queue()
    asr_q: queue.Queue = queue.Queue(maxsize=max(1, int(asr_queue_size)))
    gate = _OrderedGate(total)
    forward_lock = threading.Lock()
    forwarded_to_asr = 0

    def _run_stage(bin_name: str, index: int) -> None:
        item, result = items[index], results[index]
        if result.status in ("failed", "skipped"):
            return  # upstream already resolved this item; just flow through
        fn = item.stages.get(bin_name)
        if fn is None:
            return
        if stop.is_set():
            result.status = "skipped"
            status.emit(item=index, label=item.label, stage=bin_name, status="skipped")
            return
        status.emit(item=index, label=item.label, stage=bin_name, status="started")
        try:
            # Bound here rather than by each caller: reference_ingest builds
            # its own items and calls run_batch directly, so a binding that
            # lived in `_build_item` reached only the manifest CLI -- and
            # everything the pipeline reports, warnings included, went to a
            # reporter that shows nothing.
            with reporting_to(_item_reporter(item.label, log_level)):
                result.payload = fn(result.payload)
        except Exception as exc:  # isolate the item, keep the batch running
            result.status = "failed"
            result.failed_stage = bin_name
            result.error = f"{type(exc).__name__}: {exc}"
            status.emit(
                item=index,
                label=item.label,
                stage=bin_name,
                status="failed",
                error=result.error,
            )
            print(
                f"[batch] {item.label}: {bin_name} failed: {result.error}",
                file=sys.stderr,
            )
            return
        status.emit(item=index, label=item.label, stage=bin_name, status="done")

    def _forward_to_asr(index: int) -> None:
        nonlocal forwarded_to_asr
        asr_q.put(index)
        with forward_lock:
            forwarded_to_asr += 1
            if forwarded_to_asr == total:
                for _ in range(counts["asr"]):
                    asr_q.put(None)

    def _download_worker() -> None:
        while True:
            index = download_q.get()
            if index is None:
                return
            _run_stage("download", index)
            _forward_to_asr(index)

    def _asr_worker() -> None:
        while True:
            index = asr_q.get()
            if index is None:
                return
            _run_stage("asr", index)
            gate.submit(index)

    def _llm_worker() -> None:
        while True:
            index = gate.next()
            if index is None:
                return
            _run_stage("llm", index)
            result = results[index]
            if result.status == "pending":
                result.status = "done"
            status.emit(
                item=index,
                label=items[index].label,
                stage="item",
                status=result.status,
                **({"error": result.error} if result.error else {}),
            )

    for index in range(total):
        download_q.put(index)
    for _ in range(counts["download"]):
        download_q.put(None)
    if total == 0:
        for _ in range(counts["asr"]):
            asr_q.put(None)

    threads = (
        [threading.Thread(target=_download_worker, daemon=True) for _ in range(counts["download"])]
        + [threading.Thread(target=_asr_worker, daemon=True) for _ in range(counts["asr"])]
        + [threading.Thread(target=_llm_worker, daemon=True) for _ in range(counts["llm"])]
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        while thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except KeyboardInterrupt:
                if not stop.is_set():
                    stop.set()
                    print(
                        "[batch] interrupt: finishing in-flight stages, "
                        "skipping the rest (rerun to resume)",
                        file=sys.stderr,
                    )
    return results


# --- manifest / options --------------------------------------------------------
ALLOWED_ITEM_KEYS = {
    "source",
    "output",
    "stage",
    "model",
    "device",
    "language",
    "gpu_budget_gb",
    "asr_stabilize_profile",
    "llm_media",
    "llm_correction_media",
    "llm_planning_media",
    "llm_retrieval",
    "llm_difficulty",
    "llm_continuity",
    "llm_parallel_windows",
    "llm_fast",
    "llm_output_scale",
    "llm_video",
    "extra_info",
    "extra_style",
    "download_video_source",
    "knowledge",
    "refined_srt",
    "knowledge_root",
    "test_profile",
    "postprocess_profile",
    "max_retries_per_window",
    "max_replacements_per_window",
    "resume",
}


def read_manifest(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for no, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest line {no} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {no} must be a JSON object")
        rows.append(row)
    return rows


def merge_item_options(row: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict:
    unknown = set(row) - ALLOWED_ITEM_KEYS
    if unknown:
        raise ValueError(f"unknown manifest keys {sorted(unknown)} in row {row!r}")
    if not str(row.get("source", "") or "").strip():
        raise ValueError(f"manifest row is missing 'source': {row!r}")
    merged = dict(defaults)
    merged.update({k: v for k, v in row.items() if v is not None})
    return merged


def profile_asr_workers(items: Sequence[Mapping[str, Any]]) -> int:
    """Batch runs one ASR task at a time; the file itself supplies the parallelism.

    Concurrency used to come from the asr bin running ``wt_instances`` files
    side by side, each pinned to one shard. That kept the GPU busy but put N
    files' worth of non-model state in one process at once, and the 8GB profile
    was already measured over its RAM budget with a single file. Running one
    file with full shard and separator width bounds the live state to that file
    while keeping the same number of models resident.

    Reverses an earlier file-level design; docs/wt-parallelism.md records that
    trade, and the intra-file sharding it refers to has since been removed too
    (ASR is single-worker).

    ``items`` is unused: the answer no longer depends on the profile mix. It
    stays in the signature because callers pass the merged manifest rows.
    """

    return 1


# --- CLI item building (imports the pipeline lazily: torch stack) ---------------
def _build_item(pipeline_mod: Any, opts: Mapping[str, Any]) -> BatchItem:
    from .media.source import is_url

    source = str(opts["source"])
    target_stage = str(opts.get("stage") or "final-srt")
    order = pipeline_mod.PIPELINE_STAGE_ORDER
    if target_stage not in order:
        raise ValueError(f"unknown stage {target_stage!r} for {source}")
    asr_stage = target_stage if order[target_stage] <= order["raw-srt"] else "raw-srt"
    needs_llm = order[target_stage] > order["raw-srt"]

    def _run(payload: dict[str, Any], stage: str) -> None:
        # The reporter is bound by `run_batch` around every stage, so that
        # callers which build their own items get it too.
        paths = _run_pipeline_for(payload, stage)
        metadata_path = getattr(paths, "metadata_json", None)
        if metadata_path is not None:
            metadata = load_run_metadata(metadata_path)
            stages = metadata.get("timing", {}).get("stages", {})
            if isinstance(stages, Mapping):
                payload["_prior_timing"] = {
                    str(name): dict(record)
                    for name, record in stages.items()
                    if isinstance(record, Mapping)
                }

    def _run_pipeline_for(payload: dict[str, Any], stage: str) -> Any:
        return pipeline_mod.run_pipeline(
            payload["audio"],
            output_path=payload["output"],
            model_name=str(opts.get("model") or pipeline_mod.asr_align.DEFAULT_MODEL),
            device=str(opts.get("device") or "cuda"),
            language=opts.get("language"),
            gpu_budget_gb=int(
                opts.get("gpu_budget_gb") or DEFAULT_GPU_BUDGET_GB
            ),
            # Batch now runs one file at a time, so each file takes the whole
            # profile: passing None lets the stage use its own worker planning
            # instead of pinning every task to a single shard.
            asr_stabilize_profile=int(
                opts.get(
                    "asr_stabilize_profile",
                    pipeline_mod.asr_stabilize.DEFAULT_ASR_STABILIZE_PROFILE,
                )
            ),
            stage=stage,
            llm_media=str(opts.get("llm_media") or "audio"),
            llm_correction_media=str(opts.get("llm_correction_media") or ""),
            llm_planning_media=str(opts.get("llm_planning_media") or ""),
            llm_retrieval=str(opts.get("llm_retrieval") or "local"),
            llm_difficulty=str(opts.get("llm_difficulty") or "quality"),
            llm_continuity=str(opts.get("llm_continuity") or "serial"),
            llm_parallel_windows=int(opts.get("llm_parallel_windows") or 4),
            llm_fast=str(opts.get("llm_fast") or "auto"),
            llm_output_scale=float(opts.get("llm_output_scale") or 1.0),
            llm_video=payload.get("llm_video"),
            extra_info=str(payload.get("extra_info") or ""),
            extra_style=str(opts.get("extra_style") or ""),
            # Passed through as-is, including None: `run_pipeline` resolves an
            # unset switch against the difficulty, the same way for every front end.
            download_video_source=bool(opts.get("download_video_source", True)),
            knowledge=opts.get("knowledge"),
            refined_srt=opts.get("refined_srt"),
            knowledge_root=opts.get("knowledge_root"),
            test_profile=bool(opts.get("test_profile")),
            postprocess_profile=int(opts.get("postprocess_profile") or 0),
            # `get(key, default)` rather than `or`: an explicit 0 in the
            # manifest is a real request (disable that tier), not a miss.
            max_retries_per_window=int(opts.get("max_retries_per_window", 5)),
            max_replacements_per_window=int(
                opts.get("max_replacements_per_window", 1)
            ),
            resume=bool(opts.get("resume", True)),
            _run_started_monotonic=payload.get("_run_started_monotonic"),
            _prior_timing=payload.get("_prior_timing"),
            _batch_workers=opts.get("_batch_workers"),
        )

    stages: dict[str, Callable[[Any], Any]] = {}
    if is_url(source):

        def _download(payload: Any) -> dict:
            download_started = time.perf_counter()
            audio, paths, llm_video, source_info = pipeline_mod.prepare_url_input(
                source,
                output_path=opts.get("output"),
                llm_media=str(opts.get("llm_media") or "audio"),
                llm_correction_media=str(opts.get("llm_correction_media") or ""),
                llm_planning_media=str(opts.get("llm_planning_media") or ""),
                llm_retrieval=str(opts.get("llm_retrieval") or "local"),
                llm_difficulty=str(opts.get("llm_difficulty") or "quality"),
                llm_output_scale=float(opts.get("llm_output_scale") or 1.0),
                llm_video=opts.get("llm_video"),
                download_video_source=bool(opts.get("download_video_source", True)),
            )
            extra_info = "\n".join(
                part
                for part in (
                    f"视频来源 URL: {source}",
                    source_info,
                    str(opts.get("extra_info") or ""),
                )
                if part
            )
            return {
                "audio": audio,
                "output": paths.final_srt,
                "llm_video": llm_video,
                "extra_info": extra_info,
                "_run_started_monotonic": download_started,
                "_prior_timing": {
                    "download": {
                        "status": "executed",
                        "elapsed_sec": round(
                            time.perf_counter() - download_started, 3
                        ),
                    }
                },
            }

        stages["download"] = _download
        payload: Any = None
    else:
        source_path = Path(source).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"input not found: {source_path}")
        payload = {
            "audio": source_path,
            "output": opts.get("output"),
            "llm_video": opts.get("llm_video"),
            "extra_info": str(opts.get("extra_info") or ""),
            "_run_started_monotonic": time.perf_counter(),
            "_prior_timing": {},
        }

    stages["asr"] = lambda p: (_run(p, asr_stage), p)[1]
    if needs_llm:
        stages["llm"] = lambda p: (_run(p, target_stage), p)[1]
    return BatchItem(label=source, stages=stages, payload=payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run the production pipeline over many sources with a "
            "download(2) -> asr(profile) -> llm(1) stage-parallel runner."
        )
    )
    parser.add_argument("sources", nargs="*", help="Media URLs or local paths.")
    parser.add_argument("--manifest", help="JSONL manifest; one item object per line.")
    parser.add_argument("--batch-id", default="", help="Batch id (default: timestamp).")
    parser.add_argument(
        "--log-level",
        choices=LEVELS,
        default=None,
        help=(
            "How much each item says on stderr; same levels as the pipeline's "
            "own --log-level (default: normal)."
        ),
    )
    parser.add_argument(
        "--stage",
        default="final-srt",
        help="Default target stage per item (default: final-srt).",
    )
    parser.add_argument("--model", default=None, help="Default Whisper model.")
    parser.add_argument("--device", default="cuda", help="Device override (cpu/cuda).")
    parser.add_argument("--language", default=None, help="Default ASR language.")
    parser.add_argument(
        "--gpu-budget-gb",
        type=int,
        choices=gpu_budget_choices(),
        default=DEFAULT_GPU_BUDGET_GB,
        help="GPU budget (GiB).",
    )
    parser.add_argument(
        "--asr-stabilize-profile",
        type=int,
        choices=(-1, 0, 1, 2),
        default=0,
        help="ASR aligned -> stable profile (default: 0).",
    )
    parser.add_argument("--llm-media", choices=["text", "audio", "video"], default="audio")
    parser.add_argument(
        "--llm-correction-media", choices=["text", "audio", "video"], default=""
    )
    parser.add_argument(
        "--llm-planning-media", choices=["text", "audio", "video"], default=""
    )
    parser.add_argument("--llm-retrieval", choices=["none", "local", "native"], default="local")
    parser.add_argument(
        "--llm-difficulty",
        choices=["quality", "intermediate", "efficiency"],
        default="quality",
    )
    parser.add_argument("--llm-continuity", choices=["serial", "parallel"], default="serial")
    parser.add_argument("--llm-parallel-windows", type=int, default=4)
    parser.add_argument("--llm-fast", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--llm-output-scale", type=float, default=1.0)
    parser.add_argument("--extra-style", default="", help="Default translation style.")
    parser.add_argument(
        "--no-download-video",
        dest="download_video_source",
        action="store_false",
        help="For URL items, download only the audio track (default: fetch the video).",
    )
    # None, not "collect": `_defaults_from_args` copies this into every item, so
    # a concrete default here would shadow `run_pipeline`'s own resolution and
    # hand `difficulty=efficiency` runs a combination the LLM layer refuses.
    parser.add_argument("--knowledge", choices=["none", "collect", "update"], default=None)
    parser.add_argument("--knowledge-root", default=None)
    parser.add_argument("--test-profile", action="store_true")
    parser.add_argument("--postprocess-profile", type=int, default=0)
    parser.add_argument("--max-retries-per-window", type=int, default=5)
    parser.add_argument("--max-replacements-per-window", type=int, default=1)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable LLM session and correction-window checkpoint reads/writes.",
    )
    parser.add_argument(
        "--download-workers", type=int, default=DEFAULT_WORKERS["download"]
    )
    parser.add_argument(
        "--asr-queue-size",
        type=int,
        default=DEFAULT_ASR_QUEUE_SIZE,
        help="Backpressure: max items downloaded ahead of the asr bin.",
    )
    return parser.parse_args()


def _defaults_from_args(args: argparse.Namespace) -> dict:
    return {
        "stage": args.stage,
        "model": args.model,
        "device": args.device,
        "language": args.language,
        "gpu_budget_gb": args.gpu_budget_gb,
        "asr_stabilize_profile": args.asr_stabilize_profile,
        "llm_media": args.llm_media,
        "llm_correction_media": args.llm_correction_media,
        "llm_planning_media": args.llm_planning_media,
        "llm_retrieval": args.llm_retrieval,
        "llm_difficulty": args.llm_difficulty,
        "llm_continuity": args.llm_continuity,
        "llm_parallel_windows": args.llm_parallel_windows,
        "llm_fast": args.llm_fast,
        "llm_output_scale": args.llm_output_scale,
        "extra_style": args.extra_style,
        "download_video_source": args.download_video_source,
        "knowledge": args.knowledge,
        "knowledge_root": args.knowledge_root,
        "test_profile": args.test_profile,
        "postprocess_profile": args.postprocess_profile,
        "max_retries_per_window": args.max_retries_per_window,
        "max_replacements_per_window": args.max_replacements_per_window,
        "resume": args.resume,
        "log_level": args.log_level,
    }


def print_summary(results: Sequence[ItemResult]) -> None:
    done = sum(1 for r in results if r.status == "done")
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status in ("skipped", "pending")]
    print(f"\nBatch summary: {done} done, {len(failed)} failed, {len(skipped)} skipped")
    for r in failed:
        print(f"  FAILED  {r.label}  [{r.failed_stage}] {r.error}")
    for r in skipped:
        print(f"  SKIPPED {r.label}")


def main() -> int:
    args = parse_args()
    try:
        rows = read_manifest(args.manifest) if args.manifest else []
        rows += [{"source": s} for s in args.sources]
        if not rows:
            print("provide --manifest and/or positional sources", file=sys.stderr)
            return 2
        defaults = _defaults_from_args(args)
        merged = [merge_item_options(row, defaults) for row in rows]

        from . import pipeline  # heavy: pulls in the ASR stack

        batch_workers = {
            "download": max(1, args.download_workers),
            "asr": profile_asr_workers(merged),
            "llm": DEFAULT_WORKERS["llm"],
        }
        for opts in merged:
            opts["_batch_workers"] = batch_workers
        items = [_build_item(pipeline, opts) for opts in merged]
    except (ValueError, FileNotFoundError) as exc:
        print(f"Failed to build batch: {exc}", file=sys.stderr)
        return 2

    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    status_path = DEFAULT_BATCH_ROOT / batch_id / STATUS_FILENAME
    print(f"Batch {batch_id}: {len(items)} item(s); status -> {status_path}")
    # Quieting is a front end's job, and this is one. Without it a batch --
    # the run with the *most* output, several items deep -- would be the only
    # entry point still carrying every library's version banner.
    level = args.log_level or "normal"
    with quieted_libraries(level):
        results = run_batch(
            items,
            workers=batch_workers,
            asr_queue_size=args.asr_queue_size,
            status_path=status_path,
            log_level=level,
        )
    print_summary(results)
    return 0 if all(r.status == "done" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

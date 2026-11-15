"""The command line: one entry point, one or many sources.

`finesub a.wav`, `finesub a.wav b.mp4` and `finesub --manifest tasks.jsonl` are
the same run with different N. Each builds items and hands them to
`scheduler.run_batch`; a single source differs only in presentation -- progress
redrawn in place, a traceback in the run log, no status file unless asked.

What lives here is the option surface (what may be asked for, what a manifest
row may override, how a request becomes items) and how results are shown. The
conversion itself is `stages.py`, which this module imports and which must never
import back: `python -m finesub.pipeline` runs THIS file as `__main__`, so
anything importing `finesub.pipeline` would execute a second copy of it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager, ExitStack
from datetime import datetime
import inspect
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import jsonl, settings
from .paths import resolve_logs_dir, resolve_name_output_path
from .reporting import (
    LEVELS,
    FanOutReporter,
    FileReporter,
    quieted_libraries,
    resolve_log_level,
    terminal_reporter,
)
from .run_metadata import load_run_metadata
from .batch_state import (
    CONTROL_CURSOR_FILENAME,
    CONTROL_FILENAME,
    QUEUE_VIEW_FILENAME,
    STALE_RESUME_DAYS,
    batch_lock_path,
    control_intake,
    is_withdrawn,
    merged_intake,
    record_batch,
    resolve_resume_batch,
    strip_view_keys,
    warn_on_unfinished_batch_id,
    write_queue_view,
)
from .scheduler import (
    BatchItem,
    default_item_reporter,
    DEFAULT_ASR_QUEUE_SIZE,
    DEFAULT_BATCH_ROOT,
    DEFAULT_RETRY_FAILED,
    DEFAULT_WORKERS,
    GPU_LANE,
    IntakePoll,
    ItemResult,
    NETWORK_LANE,
    NETWORK_WORKERS,
    STATUS_FILENAME,
    run_batch,
)
from .speech import joint as joint_asr
from .speech.joint import repair as cloud_repair
from .speech.joint import stage as joint_stage
from .speech.postprocessing import stabilization as asr_stabilize
from .speech.preprocessing.separator.rates import SEPARATOR_SAMPLE_RATES
from .speech.runtime.native_paths import use_utf8_crt_locale
from .speech.runtime.resources import (
    check_tier_device_agreement,
    gpu_tier_cli_choices,
    gpu_tier_help,
)
from .stages import (
    ASR_CONTEXT_LEVELS,
    DIARIZE_LEVELS,
    default_diarize,
    warn_if_the_judge_cannot_run,
    PIPELINE_STAGE_ORDER,
    compose_url_extra_info,
    default_pipeline_paths,
    prepare_url_input,
    resolve_asr_context,
    resolve_llm_switches,
    profile_asr_workers,
    run_pipeline,
)
from .subtitles.postprocess import (
    SUPPORTED_POSTPROCESS_PROFILES,
)


def _add_recall_repair_options(parser: argparse.ArgumentParser) -> None:
    """The stable step's recall repair (docs/manual/tuning.md)."""
    for flag, what in (
        ("--squeeze-repair", "give collapsed lines (a whole sentence squeezed into a fraction of a second) back "
                             "their time where the Qwen referee hears their speech after them"),
        ("--gap-fill", "transcribe speech no line covers with the Qwen referee and Whisper, and add a line "
                       "where both hear the same words"),
    ):
        parser.add_argument(flag, choices=("auto", "on", "off"), default=None,
                            help=f"In the stable step, {what} (default: auto; auto follows --qwen-verify off).")
    parser.add_argument(
        "--row-fill", choices=("auto", "on", "off"), default=None,
        help="In the stable step, have the Qwen referee and Whisper hear lines again over their own span, and put "
             "words both hear and the line lacks into it (default: off; auto hears only the likeliest lines, "
             "about half the time for about three quarters of what on adds, and follows --qwen-verify off).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production ASR subtitle pipeline over one or more sources. "
            "Several sources (or --manifest) flow through a stage-parallel "
            f"download({DEFAULT_WORKERS['download']}) -> asr(1, cloud ASR "
            f"{DEFAULT_WORKERS[NETWORK_WORKERS]}) -> llm({DEFAULT_WORKERS['llm']}) "
            "runner, isolated per item."
        )
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="Paths to source audio/video, or media URLs.",
    )
    parser.add_argument(
        "--manifest",
        help=(
            "JSONL manifest, one item object per line: "
            '{"source": <URL or path>, ...per-item overrides}. Any option '
            "below may be overridden per row, plus 'group'. Rows appended "
            "while the run is in flight join it."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to final SRT output (one source, or a manifest row).",
    )
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
    parser.add_argument("--model", default=None, help="Whisper model name.")
    from finesub.llm.routing.model_routes import add_routing_arguments

    add_routing_arguments(parser)
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda).")
    parser.add_argument(
        "--gpu-tier",
        choices=gpu_tier_cli_choices(),
        default=None,
        help=gpu_tier_help(),
    )
    parser.add_argument(
        "--asr",
        choices=joint_asr.CHOICES,
        default=None,
        help=(
            "Who transcribes: local (default: separation, VAD, Whisper here), a cloud service fed "
            "the mix (the audio leaves this machine) -- mai2 (paid per hour), gemini / gemini-35t "
            "(GEMINI_FREE keys), moss / moss-pro (MOSS_PRO keys) -- or moss-local (open MOSS on an "
            "NVIDIA GPU of compute capability 8.0+, 1.7 GB download on first use). A task keeps the "
            "backend its -aligned.json / -stable.json came from; see docs/manual/joint-asr.md."
        ),
    )
    parser.add_argument(
        "--asr-aligner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "moss-pro / moss-local: time their long segments' units with a forced aligner; gemini-35t: "
            "the text its repair adds (heavier: about 2.4 GB RAM on CPU; default off, Silero timing)."
        ),
    )
    parser.add_argument(
        "--asr-repair",
        choices=cloud_repair.MODES,
        default=None,
        help=(
            "gemini-35t and mai2: repair what the cloud transcript missed or garbled. on (default) fills "
            "gaps the local VAD hears speech in -- mai2 by asking it again (a few %% more billed audio), "
            "gemini-35t through Gemini Live (no daily requests) -- and trims runaway repeats; quota also "
            "lets gemini-35t re-ask on its daily requests; off does none of it."
        ),
    )
    parser.add_argument(
        "--separate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run vocal separation (default: yes). --no-separate is for input "
            "that is already a clean vocal track: the stage still writes its "
            "16 kHz mono <stem>-vocal.ogg, by transcoding instead of "
            "separating, so nothing downstream changes. Clean vocals usually "
            "want --no-vad-silero-assist as well -- that post-pass is on by "
            "default because the pipeline normally separates first. The vocal "
            "stage skips on the file existing, so switching this means "
            "deleting the existing <stem>-vocal.* and everything downstream."
        ),
    )
    parser.add_argument(
        "--separator-rate",
        type=int,
        choices=SEPARATOR_SAMPLE_RATES,
        default=None,
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
        default=None,
        help="Silence gap in seconds when combining segments.",
    )
    parser.add_argument(
        "--word",
        "-w",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write word-level SRT (default: False).",
    )
    parser.add_argument(
        "--empty-cues",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Write a subtitle line that has no text as a "" cue in -raw.srt and the speaker tracks, so its span '
            "shows and a subtitle editor keeps it. Off by default: such lines are left out. config.toml: "
            "[postprocess] empty_cues."
        ),
    )
    parser.add_argument(
        "--vad-silero-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Two-signal post-pass over the energy VAD (un-suppress creep, "
            "drop ghosts, carve noise spans, restore seams). On by default -- "
            "the pipeline separates first and this exists for that kind of "
            "noisy vocal. --no-vad-silero-assist opts out for clean source "
            "audio; see docs/vad-asr.md."
        ),
    )
    parser.add_argument(
        "--diarize",
        choices=DIARIZE_LEVELS,
        default=None,
        help=(
            "Speaker diarization after stabilization. acoustics: a *-diar.json timeline plus *-diar-rows.json, "
            "each line's speaker with confidence marks; judge-text / judge-audio add an LLM speaker judge in "
            "the LLM stage that re-checks each line's track, reading the lines or also hearing them "
            "(<stem>-speaker-judge.json, the -annotated.csv spk column, --speaker-tracks). Any level cuts a "
            "line two speakers share (--diarize-split); at --llm-difficulty quality / intermediate the correction "
            "runs three passes grouped by speaker, which a judge needs (judge-* at efficiency is refused). Unset: "
            "judge-audio (judge-text under --llm-media text) for a run that reaches the LLM stage at quality, "
            "else off; a task keeps the level its LLM stage recorded. When diarization cannot run it warns and "
            "the subtitles carry no labels."
        ),
    )
    parser.add_argument(
        "--diarize-backend",
        choices=("nemotron3", "diarizen", "sortformer"),
        default=None,
        help=(
            "Which diarization model produces the timeline. Default nemotron3, downloaded "
            "on first use (about 450 MB). diarizen and sortformer are kept for research; "
            "see docs/manual/diarization.md before choosing them. "
            "A cached *-diar.json is reused whatever this says -- delete it to recompute."
        ),
    )
    parser.add_argument(
        "--diarize-max-speakers",
        type=int,
        default=None,
        help=(
            "Cap how many speakers the clustering may find. Off by default, and "
            "deliberately: over-splitting costs one merge to fix while a cap that "
            "is too tight costs a per-line relocation, and any voice it squeezes "
            "out -- a guest, a game character, background speech -- lands in a real "
            "person's track rather than disappearing. Measured on 24 files with a "
            "perfect headcount plus one: 23 of them got worse, median +530. Set it "
            "only if you know something the measurement did not. Not for sortformer."
        ),
    )
    parser.add_argument(
        "--diarize-split",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Under --diarize, cut a subtitle line in two where the speaker change inside it is "
            "acoustically clear (Japanese only for now; no LLM). On by default: -raw.srt and the "
            "LLM stage use the split lines; --no-diarize-split keeps the original ones. A task "
            "keeps the choice of its first run; see docs/manual/diarization.md."
        ),
    )
    parser.add_argument(
        "--speaker-tracks",
        choices=("off", "srt", "ass", "both"),
        default=None,
        help=(
            "Lossless speaker exports beside the single-track SRT (which is "
            "unaffected): one SRT per speaker, an ASS carrying all of them, or "
            "both. Needs speaker labels (--diarize acoustics or a judge level)."
        ),
    )
    parser.add_argument(
        "--qwen-verify",
        choices=("auto", "on", "off"),
        default=None,
        help=(
            "Second-model verification evidence at the vad-asr tail "
            "(auto = when transformers 5.x is installed; "
            "see docs/vad-asr.md)."
        ),
    )
    parser.add_argument(
        "--lang-redecode",
        choices=("auto", "on", "off"),
        default=None,
        help=(
            "Inline language-vote-collapse redecode during alignment "
            "(default: auto; auto-language runs only; see docs/vad-asr.md)."
        ),
    )
    _add_recall_repair_options(parser)
    parser.add_argument(
        "--asr-context",
        choices=ASR_CONTEXT_LEVELS,
        # None, not "terms": the answer lives once, in `resolve_asr_context`.
        default=None,
        help=(
            "How much of the knowledge base the ASR referee is told about: "
            "off, terms (default, off under --knowledge none: entry and term "
            "names with aliases, no descriptions), full (what the correction "
            "layer would see). Selected from the video title and --extra-info."
        ),
    )
    parser.add_argument(
        "--asr-decode-batch",
        default=None,
        help=(
            "Windows per decoder generate call ('auto' = the GPU profile's "
            "entry, currently 1 everywhere, i.e. off). Above 1 the aligner "
            "prefetches the next single-window groups in one batched decode. "
            "Opt-in: on the default model the end-to-end gain missed its "
            "floor, on large-v3-class models it clears it "
            "(docs/bench-baselines.md 二十二/二十三)."
        ),
    )
    parser.add_argument(
        "--split-length-scale",
        type=float,
        default=None,
        help=(
            "How long a subtitle may get before the splitter buys a cut (0.6-1.6, default 1.0; below 1 = "
            "shorter subtitles). Overrides [segmentation] length_scale in config.toml. Takes effect in the "
            "vad-asr stage, so an existing *-aligned.json must be deleted for a rerun to see it. Also scales "
            "the LLM's merge thresholds, over the style's length tier; a resumed LLM task keeps its own ratio."
        ),
    )
    parser.add_argument(
        "--asr-stabilize-profile",
        type=int,
        choices=asr_stabilize.SUPPORTED_ASR_STABILIZE_PROFILES,
        default=None,
        help=(
            "ASR stabilize profile for aligned -> stable: -1 no-op; "
            "0 default; 1 common hallucination cleanup; 2 noisy-span tags."
        ),
    )
    parser.add_argument(
        "--llm-media",
        choices=["text", "audio", "video"],
        default=None,
        help=(
            "Convenience knob for both LLM media switches (default: audio). "
            "Pass video to send the correction window a video clip; an "
            "audio-only input downgrades that back to audio automatically. "
            "This is about what the model sees, not about what is downloaded "
            "-- a URL fetches the video regardless (see --no-download-video)."
        ),
    )
    parser.add_argument(
        "--llm-retrieval",
        choices=["none", "local", "native"],
        default=None,
        help="LLM retrieval switch for translated/final stages (default: local).",
    )
    parser.add_argument(
        "--llm-difficulty",
        choices=["quality", "intermediate", "efficiency"],
        default=None,
        help=(
            "Which prompt/thinking cell the LLM stages use "
            "(default: intermediate)."
        ),
    )
    parser.add_argument(
        "--llm-continuity",
        choices=["serial", "parallel"],
        default=None,
        help=(
            "LLM window continuity (default: serial). parallel dispatches "
            "correction windows concurrently, giving up the chained "
            "inter-window context (docs/llm_harness_behavior.md)."
        ),
    )
    parser.add_argument(
        "--llm-parallel-windows",
        type=int,
        default=None,
        help="Concurrency cap for --llm-continuity parallel (default: the active preset's parallel_windows, else 1).",
    )
    parser.add_argument(
        "--llm-fast",
        choices=["auto", "on", "off"],
        default=None,
        help="LLM fast mode: fuse short inputs into one correction window (default: auto).",
    )
    parser.add_argument(
        "--llm-proofread",
        choices=["auto", "on", "off"],
        default=None,
        help=(
            "Proofread the finished subtitles in a separate pass (default: auto = at quality difficulty)."
        ),
    )
    parser.add_argument(
        "--llm-output-scale",
        type=float,
        default=None,
        help="Scale k on the LLM expected-output estimate; larger plans smaller windows.",
    )
    parser.add_argument(
        "--llm-video",
        help="Source video for --llm-media video (required at that setting).",
    )
    parser.add_argument("--extra-info", default=None, help="Extra info injected into LLM research.")
    parser.add_argument("--extra-info-file", help="Path to extra LLM research info.")
    parser.add_argument("--extra-style", default=None, help="Extra translation style for LLM correction.")
    parser.add_argument(
        "--style", default=None,
        help="Named style entries for LLM correction (comma separated; `style/<name>` when a proper-noun entry shares the name), or auto to let the model pick one. Unset reads [llm] style, then auto. Without a style of yours the factory default style is injected; under difficulty=efficiency only the factory default applies: the config is ignored and a named style is an error.",
    )
    parser.add_argument(
        "--style-mode", dest="style_mode", default=None, choices=("none", "read", "update"),
        help="What the run may do with that style: none / read (inject only) / update (also write back what this task learned; needs --refined-srt and --knowledge update). Unset reads [llm] style_mode, then read.",
    )
    parser.add_argument(
        "--no-download-video",
        default=None,
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
        choices=["none", "read", "update"],
        default=None,
        help=(
            "Knowledge switch for LLM stages. Default read: read/inject the "
            "base, collect nothing, write nothing back. none: the knowledge base "
            "is not read or injected at "
            "all; update: read, collect task_update_feedback, and run the "
            "unified knowledge update after correction."
        ),
    )
    parser.add_argument(
        "--refined-srt",
        help="User-refined SRT for the knowledge update (with --knowledge update).",
    )
    parser.add_argument("--knowledge-root", help="Override local knowledge base root for LLM stages.")
    parser.add_argument("--task-artifact-dir", help="Override LLM task artifact directory.")
    parser.add_argument("--task-id", default=None, help="Stable task id for LLM artifacts.")
    parser.add_argument("--task-summary", default=None, help="Task summary for knowledge update prompts.")
    parser.add_argument(
        "--test-profile",
        action="store_true",
        default=None,
        help="Use the LLM test profile.",
    )
    parser.add_argument(
        "--postprocess-profile",
        type=int,
        choices=SUPPORTED_POSTPROCESS_PROFILES,
        default=None,
        help=(
            "Final SRT postprocess profile: -1 semantic no-op re-render; "
            "0 t2s, overlap, duration, punctuation; 1 duration only; "
            "2 punctuation only; 3 t2s only; 4 overlap repair only."
        ),
    )
    parser.add_argument(
        "--max-retries-per-window",
        type=int,
        default=None,
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
        default=None,
        help=(
            "Tier 2 of the per-window LLM retry budget: fresh-session "
            "replacements after a chain's repair budget is spent (repair "
            "context dropped; agents start a new conversation)."
        ),
    )
    parser.add_argument(
        "--no-resume",
        default=None,
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
    # --- the runner (several sources) -----------------------------------
    parser.add_argument(
        "--max-parallel-tasks",
        type=int,
        default=DEFAULT_WORKERS["llm"],
        help=(
            "How many items' LLM stages may run at once (admission knob, "
            "task-parallelism plan W4/W5). A preference, not a calibrated "
            "constant: too high burns the 5h agent quota mid-session and "
            "trades token efficiency for diminishing wall-clock gains "
            "(docs/manual/agent.md). Items sharing a manifest 'group' stay "
            "serial regardless."
        ),
    )
    parser.add_argument(
        "--download-workers", type=int, default=DEFAULT_WORKERS["download"]
    )
    parser.add_argument(
        "--asr-queue-size",
        type=int,
        default=None,
        help=f"Downloads may run this far ahead of ASR (back pressure; default {DEFAULT_ASR_QUEUE_SIZE}).",
    )
    parser.add_argument(
        "--retry-failed",
        type=int,
        default=None,
        dest="retry_failed",
        help=(
            "Retry a failed stage this many times before giving up on the item "
            f"(default {DEFAULT_RETRY_FAILED}, 0 disables). The stage, not the item: everything "
            "before it is already on disk and skipped on existence, so a retry "
            "costs only the part that failed."
        ),
    )
    parser.add_argument(
        "--resume-batch",
        nargs="?",
        const="",
        default=None,
        metavar="ID",
        dest="resume_batch",
        help=(
            "Continue a multi-source run from the queue it published, under "
            "the id it already has (so --batch-id cannot be combined). With no "
            "ID: the most recent unfinished one, unless it is more than "
            f"{STALE_RESUME_DAYS} days old -- then it says so and prints the "
            "command that names it, rather than picking up week-old work by "
            "surprise. A batch that is still running, or that ran in another "
            "directory, is refused either way. Finished stages are skipped on "
            "existence, as in any rerun."
        ),
    )
    parser.add_argument(
        "--batch-id",
        help=(
            "Names out/batch/<id>/batch-status.jsonl. Default: a timestamp for "
            "several sources; a single foreground run writes none unless asked."
        ),
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


#: CLI dests that feed a row key of a different name (`_defaults_from_args`
#: folds them). Everything else inheritable is its own row key, which is what
#: makes "what did the user actually type" answerable without a fourth copy of
#: the option table.
_DEST_ROW_KEYS = {"extra_info_file": "extra_info"}


def explicit_row_keys(argv: Sequence[str] | None = None) -> frozenset[str]:
    """Which row keys this command line named OUT LOUD.

    argparse cannot tell a flag that was typed from one that defaulted, so the
    same parser is run a second time with every default suppressed: what comes
    back is exactly what was on the command line. Only resuming needs this --
    a resumed row is a record the runner made, not a file the user wrote, so
    an option they typed now has to beat it (a batch that recorded
    `device: cuda` is otherwise unresumable on a machine that lost its GPU).
    """

    parser = build_parser()
    for action in parser._actions:  # noqa: SLF001 -- argparse offers no other way
        action.default = argparse.SUPPRESS
    try:
        given = vars(parser.parse_args(list(argv) if argv is not None else None))
    except SystemExit:
        # The real parse already succeeded, so this cannot normally fail; if it
        # somehow does, "nothing was explicit" is the pre-existing behaviour.
        return frozenset()
    return frozenset(_DEST_ROW_KEYS.get(dest, dest) for dest in given)


def _prune_run_logs() -> None:
    """Trim the log directory once per process, before its first log opens.

    Not on every open: in a batch of more than a hundred items, each later item
    pruned away the logs of this batch's own finished items -- the oldest by
    mtime -- which are the files a batch most needs (audit O-6).
    """

    from finesub_bootstrap.logs import prune

    prune(resolve_logs_dir())


@contextmanager
def _run_log(stem: str) -> Iterator[FileReporter | None]:
    """A per-run verbose log beside the other user data, or nothing.

    One file per run rather than one shared file: two CLI runs can be in
    flight at once (docs/cross-frontend-lease.md), and separate
    files need no locking and never interleave two runs.

    Opening one does not prune the directory: the caller does that once, up
    front (`_prune_run_logs`).
    """

    from finesub_bootstrap.logs import open_log

    directory = resolve_logs_dir()
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


@contextmanager
def _batch_item_reporters(level: str) -> Iterator[Callable[[str], Any]]:
    """One run log per item, opened when that item first speaks.

    A multi-source run is exactly the one nobody watched line by line, so it is
    the one that most needs the file. It was single-source-only while binding a
    reporter was the runner's business; `item_reporter` made it the front end's
    (docs/reporting.md §5.2), and this is that hook.

    The files are opened lazily and closed together when the run ends: an item
    that never started (breaker, interrupt) leaves no empty log. Two items whose
    basenames collide get a numbered suffix rather than appending into one file.
    """

    _prune_run_logs()
    with ExitStack() as stack:
        made: dict[str, Any] = {}
        stems: dict[str, int] = {}
        lock = threading.Lock()

        def make(label: str) -> Any:
            with lock:
                existing = made.get(label)
                if existing is not None:
                    return existing
                terminal = default_item_reporter(label, level)
                stem = Path(label).stem or label
                seen = stems.get(stem, 0)
                stems[stem] = seen + 1
                file_reporter = stack.enter_context(
                    _run_log(stem if not seen else f"{stem}-{seen + 1}")
                )
                reporter = (
                    FanOutReporter(terminal, file_reporter)
                    if file_reporter is not None
                    else terminal
                )
                made[label] = reporter
                return reporter

        yield make


# --- multi-source front end (manifest rows, items, the runner) -----------------
#
# One CLI for 1..N sources, and ONE runner behind it: a single source is one
# item with one worker per bin, not a second code path (owner 2026-08-30). What
# used to be `finesub.batch` is now just "more than one source"; the engine it
# grew for that lives in `scheduler.py` and knows nothing about pipeline
# options. Everything here is the option surface: what a manifest row may say,
# how a row becomes an item, and how one invocation's flags become the defaults
# every row inherits.


#: Every keyword parameter of `run_pipeline` -- derived, never listed. The row
#: whitelist, the CLI defaults and the call all read this one set, so a new
#: pipeline option is settable per row the moment it exists (five ASR options
#: had silently never been reachable from a manifest).
_PIPELINE_PARAMS = frozenset(
    name
    for name, param in inspect.signature(run_pipeline).parameters.items()
    if param.kind is param.KEYWORD_ONLY and not name.startswith("_")
)


#: `run_pipeline`'s own defaults, read from the signature rather than restated.
_PIPELINE_DEFAULTS: dict[str, Any] = {
    name: param.default
    for name, param in inspect.signature(run_pipeline).parameters.items()
    if param.default is not inspect.Parameter.empty
}


def opt(opts: Mapping[str, Any], key: str) -> Any:
    """A row's value for ``key``, or `run_pipeline`'s own default.

    The URL branch has to know some of these BEFORE `run_pipeline` runs (they
    decide what gets downloaded and how the media is prepared), and it used to
    spell the fallback out: `opts.get("llm_media") or "audio"`. That is a
    third copy of the default -- argparse gave up its copy in 2026-08-31, the
    signature is the source of truth, and this one was simply out of the
    ratchet's view: it scans argparse against the signature and never looks at
    the row (reviewer 2026-09-01 P2). Today's values happen to agree; the
    failure it prevents is a signature change taking effect on the run but not
    on the download that fed it.
    """

    value = opts.get(key)
    return _PIPELINE_DEFAULTS[_ROW_ALIASES.get(key, key)] if value is None else value


#: Manifest key -> `run_pipeline` parameter, where the two names differ. The
#: row spelling is the CLI's, which is what someone reading `--help` expects to
#: be able to write in a row.
_ROW_ALIASES = {
    "model": "model_name",
    "output": "output_path",
    "gap": "gap_sec",
    "separator_rate": "separator_sample_rate",
}


#: Row keys that are not pipeline parameters: which source, which scheduling
#: group it belongs to, and how urgent it is.
_ROW_ONLY_KEYS = frozenset({"source", "group", "priority"})


#: Options that identify ONE run's outputs. Legal per row; as an invocation
#: default they would name the same destination for every source -- and since
#: stages skip on existence, the second item would then "succeed" by reusing
#: the first item's artifacts under its own label (reviewer 2026-08-30 P1).
#: `--name` was already refused for several sources; these are the rest of it.
_PER_SOURCE_KEYS = frozenset({
    # Where this run writes, and what names it in every artifact it writes.
    "output",
    "task_id",
    "task_artifact_dir",
    # Read-only, but read-only *per source*: one video or one human-refined SRT
    # broadcast to N items does not overwrite anything -- it silently corrects
    # item B against item A's material, and (for refined_srt) writes the
    # resulting nonsense into the knowledge base. Rejected for the same reason,
    # not for the same mechanism (reviewer 2026-08-30 P2 asked for these back;
    # `task_summary`, which really is just batch-level prompt context, is out
    # of this set and inherited as before).
    "llm_video",
    "refined_srt",
})


ALLOWED_ITEM_KEYS = (
    _ROW_ONLY_KEYS
    | frozenset(_ROW_ALIASES)
    | (_PIPELINE_PARAMS - frozenset(_ROW_ALIASES.values()))
)


def read_manifest(path: str | Path) -> list[dict]:
    return parse_manifest_rows(Path(path).read_text(encoding="utf-8"))


def manifest_snapshot(path: str | Path) -> tuple[list[dict], int, str]:
    """ONE read of the manifest -> (its rows, the intake cursor, its last line
    if that had no newline -- the line `manifest_intake` must watch grow).

    All three must come from the same snapshot. Reading the file a second
    time for the cursor -- after the CLI's heavy imports, which take long
    enough for a row to land -- marked that row consumed without ever building
    it, dropping it from the run for good (reviewer 2026-08-30 P1-1).
    """

    text = Path(path).read_text(encoding="utf-8")
    return parse_manifest_rows(text), len(jsonl.lines(text)), text[text.rfind("\n") + 1 :]


def parse_manifest_rows(text: str) -> list[dict]:
    """Strict parse of a manifest's TEXT (every line, terminated or not).

    Takes text rather than a path so the startup parse and the intake cursor
    can come from one snapshot (see `manifest_snapshot`).
    """

    rows: list[dict] = []
    for no, line in enumerate(jsonl.lines(text), 1):
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


def merge_item_options(
    row: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    override: Collection[str] = (),
) -> dict:
    """The row, over this invocation's defaults.

    ``override`` names keys where that order is reversed -- used only for the
    rows of a RESUMED batch, which are not a file the user wrote but a record
    the runner made of what it was once told. An option typed on the resume's
    own command line has to beat that record; a row in the user's own manifest
    still wins over a default, as documented.
    """

    unknown = set(row) - ALLOWED_ITEM_KEYS
    if "llm_preset" in unknown:
        raise ValueError(
            "manifest rows cannot choose an LLM preset: it is one per run (the "
            f"route table is per process); pass --llm-preset on the command line instead: {row!r}"
        )
    if unknown:
        raise ValueError(f"unknown manifest keys {sorted(unknown)} in row {row!r}")
    if not str(row.get("source", "") or "").strip():
        raise ValueError(f"manifest row is missing 'source': {row!r}")
    merged = dict(defaults)
    merged.update({k: v for k, v in row.items() if v is not None})
    merged.update({key: defaults[key] for key in override if key in defaults})
    return merged


def manifest_intake(
    manifest_path: str | Path,
    *,
    defaults: Mapping[str, Any],
    build: Callable[[Mapping[str, Any]], BatchItem],
    consumed_lines: int,
    consumed_tail: str = "",
) -> Callable[[], IntakePoll]:
    """An intake that picks up rows APPENDED to the manifest while it runs
    (append-only: edits to already-consumed lines are ignored).

    Only the newline-terminated prefix is read, so a row still being written
    is left for the next poll instead of half-parsed. A malformed complete
    row is skipped with a warning -- a typo must not crash a running batch --
    and every consumed line stays consumed.

    Returns an :class:`IntakePoll`, not a plain list: "nothing new" and "I
    could not see everything" must not read alike, because the caller ends
    the run on an empty poll. An unterminated NEW tail (a row mid-write) and
    a read that failed outright (a decode error on a half-written multi-byte
    character is the common one) are both ``settled=False`` -- wait for it. An
    unterminated last line that the startup snapshot already consumed is the
    ordinary shape of a manifest and settles normally.

    That consumed line is ``consumed_tail``, and it is watched: a row appended
    with a bare ``write(row + "\\n")`` -- PowerShell's `Add-Content` does the
    same -- lands on it, ``{...}{...}``, and used to vanish into a line already
    counted. What grows after the consumed text is read as an appended row.
    """

    path = Path(manifest_path)
    state = {"consumed": int(consumed_lines), "tail": consumed_tail}

    def poll() -> IntakePoll:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return IntakePoll(settled=False, reason=f"{type(exc).__name__}: {exc}")
        complete = text[: text.rfind("\n") + 1]
        lines = jsonl.lines(complete)
        # A trailing line without its newline is only *pending* when it is one
        # we have not consumed yet; the last line of an ordinary manifest has
        # no newline either and was consumed at startup.
        tail = text[len(complete):]
        pending_tail = bool(tail.strip()) and len(lines) + 1 > state["consumed"]
        candidates = [(no, lines[no]) for no in range(state["consumed"], len(lines))]
        if state["tail"]:
            last = state["consumed"] - 1
            current = lines[last] if last < len(lines) else tail
            grown = current[len(state["tail"]):] if current.startswith(state["tail"]) else ""
            if last < len(lines):  # terminated now: whatever grew is final
                state["tail"] = ""
                if grown.strip():
                    candidates.insert(0, (last, grown))
            elif grown.strip():  # still growing: a row mid-write
                pending_tail = True
        fresh: list[BatchItem] = []
        for line_no, line in candidates:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("manifest row must be a JSON object")
                if is_withdrawn(row):
                    continue
                fresh.append(build(merge_item_options(strip_view_keys(row), defaults)))
            except Exception as exc:  # noqa: BLE001 -- see docstring
                print(
                    f"[batch] appended manifest line {line_no + 1} skipped: {exc}",
                    file=sys.stderr,
                )
        # Never backwards: a manifest whose last line had no trailing newline
        # counts fewer terminated lines than the strict startup read consumed.
        state["consumed"] = max(state["consumed"], len(lines))
        if fresh:
            print(
                f"[batch] picked up {len(fresh)} appended manifest item(s)",
                file=sys.stderr,
            )
        return IntakePoll(
            items=tuple(fresh),
            settled=not pending_tail,
            reason="manifest ends mid-line" if pending_tail else "",
        )

    return poll


# --- the run's queue: one file it publishes, one it takes instructions from ---
#
# Two files, one writer each. A single file that the runner rewrites and the
# user edits is two writers on one path: an editor's save (write temp, replace)
# silently drops whatever the runner wrote in between. So the runner OWNS the
# view and the user OWNS the control channel, and the control channel is
# append-only -- the same shape the manifest intake already has, which is why
# adding a task through it needs no new mechanism.
#
# Both live beside `batch-status.jsonl`, in the same CWD-relative tree as the
# outputs they describe: resuming from a different directory could not find
# those outputs either, so a "stable" location would be a false promise.

# --- finding a batch again without being told its id -------------------------
#
# `queue.jsonl` lives with the outputs it describes (CWD-relative, under
# `out/batch/<id>/`), which is right for the file and useless for finding it:
# "the one I was running yesterday" is not a path anyone remembers. So the run
# leaves a pointer in the data root -- the same place the logs and `.env` are,
# resolved the same way -- and `--resume-batch` with no id reads it.
#
# A pointer, not a queue: the registry holds no work, only where a run's queue
# is and how it ended. Nothing consumes it automatically, so there is still
# nothing to unwind when a batch is abandoned.


def _pipeline_kwargs(opts: Mapping[str, Any]) -> dict[str, Any]:
    """A row's options as `run_pipeline` keyword arguments.

    Pass-through, not a restatement: an option the row does not set is simply
    absent, so `run_pipeline`'s own signature supplies the default. The
    previous hand-written call restated all 28 of them together with a second
    copy of every default, which is how three copies of "1" came to be the
    default parallel-window count.
    """

    kwargs: dict[str, Any] = {}
    for key, value in opts.items():
        if key in _ROW_ONLY_KEYS or key.startswith("_") or value is None:
            continue
        param = _ROW_ALIASES.get(key, key)
        if param in _PIPELINE_PARAMS:
            kwargs[param] = value
    # Here and not inside `run_pipeline`: this is the one place that can tell
    # "the user asked for cuda" from "the signature's default is cuda" -- a key
    # only reaches `kwargs` when a row or a flag actually set it. Down in
    # `run_pipeline` both look identical, so the same check there would reject
    # a bare `--gpu-tier cpu`.
    check_tier_device_agreement(kwargs.get("gpu_tier"), kwargs.get("device"))
    return kwargs


class OutputClaims:
    """Every write root this run has promised to an item.

    Two items sharing one must never both run: they share every artifact path,
    and because stages skip on existence the second one skips its own work and
    reports success holding the first one's subtitles. The LLM artifact
    directory is the same hazard one level down -- two tasks writing one
    `task-artifacts.jsonl`, one plan, one exchange log, one resume ledger.

    Stateful rather than a one-shot check over the startup list, because
    admission has three doors and all three must go through the same book
    (reviewer 2026-08-30 P1):

    * startup items,
    * rows appended to the manifest while the run is in flight -- these build
      their items straight from the intake and used to bypass the check
      entirely,
    * a URL's real destination, which only exists once the download has
      resolved the video id: two spellings of one video (with and without a
      query string, say) collide only there, so the claim is made in the
      download stage, before anything is fetched, and fails that item alone.

    Paths are compared normcased and resolved, so `OUT\\A.srt` and `out/a.srt`
    are one claim on Windows.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holders: dict[str, str] = {}
        self._sources: set[str] = set()
        self._labels: set[str] = set()

    @staticmethod
    def _key(value: str | Path) -> str:
        return os.path.normcase(str(Path(value).expanduser().resolve()))

    def _take(self, key: str, holder: str, what: str) -> None:
        held = self._holders.setdefault(key, holder)
        if held != holder:
            raise ValueError(
                f"{holder} and {held} would both write {what} -- give one of "
                "them its own --output (a manifest row's 'output' key), or "
                "rename the input"
            )

    def claim_source(self, source: str) -> None:
        """One source runs once; listing it twice is the same run twice.

        A set rather than the holder map: the second claim carries the same
        name as the first, so "who holds it" cannot tell a duplicate apart
        from a re-claim.
        """

        with self._lock:
            key = os.path.normcase(source.strip())
            if key in self._sources:
                raise ValueError(f"source {source!r} is listed twice")
            self._sources.add(key)

    def unique_label(self, source: str) -> str:
        """A label no other item of this run answers to.

        The label is an identity, not decoration: it prefixes this item's
        terminal lines and names its run log, so two `a.wav` from different
        directories -- legal as long as they were given different outputs --
        were indistinguishable in the terminal and shared one log file
        (reviewer 2026-08-31 P2). The parent directory disambiguates far better
        than a counter would.
        """

        base = Path(source).name or source
        with self._lock:
            if base not in self._labels:
                self._labels.add(base)
                return base
            parent = Path(source).parent.name
            candidate = f"{parent}/{base}" if parent else base
            suffix = 2
            while candidate in self._labels:
                candidate = f"{parent}/{base} ({suffix})" if parent else f"{base} ({suffix})"
                suffix += 1
            self._labels.add(candidate)
            return candidate

    @contextmanager
    def claiming(self, opts: Mapping[str, Any]) -> Iterator[None]:
        """Claim this row's write roots, and keep them only if it builds.

        Atomic on purpose: an appended manifest row that is claimed and then
        fails to build (an unknown stage, a path that is not there yet) used to
        leave its source and its paths booked forever, so the corrected row the
        user appends next is refused as a duplicate and the only way out is
        restarting the batch (reviewer 2026-08-31 P1).
        """

        source = str(opts["source"])
        before_sources = os.path.normcase(source.strip()) in self._sources
        before_holders = set(self._holders)
        try:
            self.claim_item(opts)
            yield
        except BaseException:
            with self._lock:
                if not before_sources:
                    self._sources.discard(os.path.normcase(source.strip()))
                for key in set(self._holders) - before_holders:
                    if self._holders.get(key) == source:
                        del self._holders[key]
            raise

    def claim_paths(
        self,
        holder: str,
        *,
        final_srt: str | Path | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        with self._lock:
            if final_srt is not None:
                self._take(self._key(final_srt), holder, str(final_srt))
            if artifact_dir is not None:
                self._take(
                    self._key(artifact_dir) + "/artifacts", holder, str(artifact_dir)
                )

    def claim_item(self, opts: Mapping[str, Any]) -> None:
        """Everything a row's destination can be known to be before it runs."""

        from .media.source import is_url

        source = str(opts["source"])
        self.claim_source(source)
        explicit_dir = opts.get("task_artifact_dir")
        if is_url(source) and not opts.get("output"):
            # Destination unknown until the download resolves it; the artifact
            # dir is claimable now only when the row named one itself.
            self.claim_paths(source, artifact_dir=explicit_dir)
            return
        paths = default_pipeline_paths(source, output_path=opts.get("output"))
        self.claim_paths(
            source,
            final_srt=paths.final_srt,
            artifact_dir=explicit_dir or paths.task_artifact_dir,
        )


def build_item(opts: Mapping[str, Any], *, claims: "OutputClaims | None" = None) -> BatchItem:
    """One row -> one runnable item (download? -> asr -> llm?).

    ``claims`` books this item's write roots (see `OutputClaims`) and raises
    when another item already holds one. Passing it here rather than checking
    the startup list is what puts appended manifest rows through the same book
    -- and the booking is atomic with the build, so a row that does not become
    an item leaves nothing behind for the corrected row to trip over.
    """

    if claims is None:
        return _build_item(opts)
    with claims.claiming(opts):
        return _build_item(opts, claims=claims)


def _refuse_moss_local_without_its_card(
    source: str, output: Any, shared: Mapping[str, Any], asr_stage: str
) -> None:
    """Stop a task that would decode with local MOSS on a run that cannot (`moss_local_unusable_reason`) when it
    is built -- before a download, a model fetch or a place in the asr queue. A startup row stops the run; a row
    appended later is skipped. Only a task that will decode: at or past `aligned`, its `-aligned.json` not there
    (nor its `-stable.json`, unless it stops at aligned). A URL's products are not known before it downloads, so
    without an explicit output it counts as decoding with this run's `--asr`."""

    from .media.source import is_url

    if PIPELINE_STAGE_ORDER[asr_stage] < PIPELINE_STAGE_ORDER["aligned"]:
        return
    requested = shared.get("asr")
    if is_url(source) and not output:
        backend = joint_stage.resolve_backend(settings.ASR_BACKEND.resolve(requested))
    else:
        # resolved as `run_pipeline` resolves a local source, or a link to it would look for other products
        local = source if is_url(source) else Path(source).expanduser().resolve()
        paths = default_pipeline_paths(local, output_path=output)
        if paths.aligned_json.exists() or (asr_stage != "aligned" and paths.stable_json.exists()):
            return
        backend = joint_stage.task_backend(paths.stable_json, paths.aligned_json, requested)
    if backend != joint_asr.MOSS_LOCAL:
        return
    from .speech.runtime.device import moss_local_unusable_reason

    reason = moss_local_unusable_reason(shared.get("gpu_tier"), shared.get("device"))
    if reason is not None:
        raise ValueError(f"{source}: {reason}")


def _build_item(opts: Mapping[str, Any], *, claims: "OutputClaims | None" = None) -> BatchItem:
    from .media.source import is_url

    source = str(opts["source"])
    target_stage = str(opt(opts, "stage"))
    order = PIPELINE_STAGE_ORDER
    if target_stage not in order:
        raise ValueError(f"unknown stage {target_stage!r} for {source}")
    asr_stage = target_stage if order[target_stage] <= order["raw-srt"] else "raw-srt"
    needs_llm = order[target_stage] > order["raw-srt"]
    shared = _pipeline_kwargs(opts)
    diarize = shared.get("diarize")
    # Before the expensive half: a batch saved before the rename says `on`.
    if diarize is not None and diarize not in DIARIZE_LEVELS:
        raise ValueError(f"--diarize takes {'/'.join(DIARIZE_LEVELS)}, got {diarize!r} for {source}")
    # For the stage the item runs to, not for each half (`run_pipeline`'s `_diarize_default`).
    _, profile = resolve_llm_switches(
        target_stage,
        llm_media=opts.get("llm_media"),
        llm_retrieval=opts.get("llm_retrieval"),
        llm_difficulty=opts.get("llm_difficulty"),
        llm_continuity=opts.get("llm_continuity"),
        knowledge=opts.get("knowledge"),
        llm_output_scale=float(opt(opts, "llm_output_scale")),
    )
    diarize_default = default_diarize(target_stage, profile)
    warn_if_the_judge_cannot_run(diarize or diarize_default, target_stage)
    _refuse_moss_local_without_its_card(source, opts.get("output"), shared, asr_stage)

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
        return run_pipeline(
            payload["audio"],
            **{
                **shared,
                # The payload wins for these three: a URL item only learns its
                # output path, its video and its source note while downloading.
                "output_path": payload["output"],
                "llm_video": payload.get("llm_video"),
                "extra_info": str(payload.get("extra_info") or ""),
                "stage": stage,
                "_run_started_monotonic": payload.get("_run_started_monotonic"),
                "_prior_timing": payload.get("_prior_timing"),
                "_batch_workers": opts.get("_batch_workers"),
                "_diarize_default": diarize_default,
            },
        )

    stages: dict[str, Callable[[Any], Any]] = {}
    if is_url(source):

        def _claim(paths: Any) -> None:
            # Only once the id is resolved is the destination known: two URL
            # spellings naming one video collide here and nowhere earlier.
            # Before the download, or both would fetch into one directory at
            # once. Raising fails this item alone.
            claims.claim_paths(
                source,
                final_srt=paths.final_srt,
                artifact_dir=opts.get("task_artifact_dir") or paths.task_artifact_dir,
            )

        def _download(payload: Any) -> dict:
            download_started = time.perf_counter()
            # Fails a conflicting switch vector before the download, as
            # `run_pipeline` does; an unset switch is the resolver's to fill.
            switches, _ = resolve_llm_switches(
                str(opt(opts, "stage")),
                llm_media=opts.get("llm_media"),
                llm_retrieval=opts.get("llm_retrieval"),
                llm_difficulty=opts.get("llm_difficulty"),
                llm_continuity=opts.get("llm_continuity"),
                knowledge=opts.get("knowledge"),
                llm_output_scale=float(opt(opts, "llm_output_scale")),
            )
            audio, paths, llm_video, source_info = prepare_url_input(
                source,
                output_path=opts.get("output"),
                llm_video=opts.get("llm_video"),
                download_video_source=bool(opt(opts, "download_video_source")),
                on_destination=(
                    _claim if claims is not None and not opts.get("output") else None
                ),
            )
            # The one assembly point (`stages.compose_url_extra_info`). The
            # inline join that used to sit here is how the scraped title came
            # to exist in single runs and not in batch ones.
            extra_info = compose_url_extra_info(
                source,
                source_info,
                str(opt(opts, "extra_info")),
                stage=str(opt(opts, "stage")),
                asr_context=resolve_asr_context(
                    opts.get("asr_context"), knowledge=switches.knowledge
                ),
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
            "extra_info": str(opt(opts, "extra_info")),
            "_run_started_monotonic": time.perf_counter(),
            "_prior_timing": {},
        }

    stages["asr"] = lambda p: (_run(p, asr_stage), p)[1]
    if needs_llm:
        stages["llm"] = lambda p: (_run(p, target_stage), p)[1]

    def _llm_cost_estimate(item_payload: Any) -> float:
        # The LPT key (plan W4): segment count of the stable json -- present
        # by the time the llm stage is schedulable, free, deterministic.
        # Duration is only a proxy for cost and drifts with speech density.
        #
        # Derived exactly as the run derives it -- (input, output) -- not from
        # the SRT alone: `default_pipeline_paths(srt)` treats its first
        # argument as the INPUT and rebuilds a default `out/<stem>/` path, so a
        # custom --output looked for a stable json that never existed, and an
        # item with no explicit output scored 0 and fell out of LPT entirely
        # (reviewer 2026-08-30 P2).
        payload = item_payload or {}
        audio = payload.get("audio")
        if not audio:
            return 0.0
        stable = default_pipeline_paths(audio, output_path=payload.get("output")).stable_json
        if not stable.exists():
            return 0.0
        from .subtitles.stable_rows import read_rows

        segments = read_rows(stable)
        return float(len(segments)) if isinstance(segments, list) else 0.0

    def _asr_lane(item_payload: Any) -> str:
        # The task's own backend (`joint_stage.task_backend`), from the paths
        # the run will use: a cloud task waits on the network, not the GPU.
        payload = item_payload or {}
        paths = default_pipeline_paths(payload["audio"], output_path=payload.get("output"))
        backend = joint_stage.task_backend(paths.stable_json, paths.aligned_json, shared.get("asr"))
        return GPU_LANE if backend in (joint_asr.LOCAL, joint_asr.MOSS_LOCAL) else NETWORK_LANE

    return BatchItem(
        label=(
            claims.unique_label(source) if claims is not None else Path(source).name or source
        ),
        stages=stages,
        payload=payload,
        asr_lane=_asr_lane,
        group=str(opts.get("group") or ""),
        priority=int(opts.get("priority") or 0),
        row=dict(opts),
        llm_cost=_llm_cost_estimate if needs_llm else None,
    )


def _defaults_from_args(args: argparse.Namespace, *, single: bool) -> dict:
    """This invocation's flags as the row every manifest row inherits.

    Derived from `ALLOWED_ITEM_KEYS`, not listed: a CLI dest that names a row
    key IS that key's default. Listing them was the fourth copy of the option
    table and the one that quietly went stale.

    ``single`` gates `_PER_SOURCE_KEYS`: with one source they are that source's
    settings, with several they would be everyone's. `main` refuses them
    outright in that case rather than dropping them silently.
    """

    inheritable = ALLOWED_ITEM_KEYS - _ROW_ONLY_KEYS
    if not single:
        inheritable = inheritable - _PER_SOURCE_KEYS
    defaults = {
        key: getattr(args, key)
        for key in sorted(inheritable)
        if getattr(args, key, None) is not None
    }
    # Rules over the row rather than row values. Every row records the stage
    # it runs through, the bare default included: resuming a batch compares
    # it with a newly typed `--stage` to find the items that get work again.
    defaults["stage"] = args.stage or "raw-srt"
    extra_info = str(args.extra_info or "").strip()
    if args.extra_info_file:
        file_info = (
            Path(args.extra_info_file).expanduser().read_text(encoding="utf-8").strip()
        )
        extra_info = "\n".join(part for part in (extra_info, file_info) if part)
    if extra_info:
        defaults["extra_info"] = extra_info
    return defaults


def print_summary(results: Sequence[ItemResult]) -> None:
    done = sum(1 for r in results if r.status == "done")
    failed = [r for r in results if r.status == "failed"]
    dropped = [r for r in results if r.dropped]
    skipped = [
        r for r in results if r.status in ("skipped", "pending") and not r.dropped
    ]
    counts = f"{done} done, {len(failed)} failed, {len(skipped)} skipped"
    print(f"\nSummary: {counts}" + (f", {len(dropped)} dropped" if dropped else ""))
    for r in failed:
        print(f"  FAILED  {r.label}  [{r.failed_stage}] {r.error}")
    for r in skipped:
        print(f"  SKIPPED {r.label}")
    for r in dropped:
        print(f"  DROPPED {r.label}")


def _settled(results: Sequence[ItemResult]) -> bool:
    """Whether this run has nothing left to come back for.

    A withdrawn item counts as settled: the user said not to run it, so a batch
    whose only unfinished items are drops is done -- neither a failure to the
    exit code nor something for the next id-less resume to offer back.
    """

    return all(r.status == "done" or r.dropped for r in results)


def main() -> int:
    # Before any stage opens a model file: native loaders otherwise misread a
    # non-ASCII install path on a non-UTF-8 Windows code page.
    locale_problem = use_utf8_crt_locale()
    if locale_problem:
        print(f"Warning: {locale_problem}", file=sys.stderr)
    args = parse_args()
    # Installed unconditionally (empty → clears) BEFORE any routing consumer
    # runs: planning envelopes, preflight and the LLM stages' own clients all
    # read the memoized route loader. Unconditional install is the lifecycle
    # guarantee — a second main() in one interpreter never inherits the
    # previous run's override.
    from finesub.llm.routing.model_routes import install_runtime_routing

    install_runtime_routing(args)
    # Set when this run CONTINUES a batch rather than starting one. Resuming
    # keeps the batch's identity -- same directory, same queue view, same
    # control channel, same registry row, same lock -- because a resume that
    # minted a fresh id was a *second* batch over the first one's outputs: the
    # lock it took was its own, the old row stayed `unfinished` and the next
    # id-less resume picked it up all over again (reviewer 2026-08-31 P1).
    resume_id = ""
    try:
        if args.resume_batch is not None:
            if args.manifest:
                print(
                    "--resume-batch and --manifest both say what to run; give one",
                    file=sys.stderr,
                )
                return 2
            if args.batch_id:
                print(
                    "--resume-batch continues a batch under the id it already "
                    "has; --batch-id would fork it into a second one",
                    file=sys.stderr,
                )
                return 2
            queue_path, resume_id, refusal = resolve_resume_batch(args.resume_batch)
            if queue_path is None:
                print(refusal, file=sys.stderr)
                return 2
            print(f"[batch] resuming {resume_id} from {queue_path}")
            args.manifest = str(queue_path)
        # ONE snapshot feeds both the startup parse and the intake cursor: a
        # second read would mark rows appended in between as consumed without
        # ever building them.
        rows, consumed_lines, consumed_tail = (
            manifest_snapshot(args.manifest) if args.manifest else ([], 0, "")
        )
        withdrawn = [row for row in rows if is_withdrawn(row)]
        if withdrawn:
            rows = [row for row in rows if not is_withdrawn(row)]
            print(f"[batch] {len(withdrawn)} withdrawn item(s) not resumed")
        rows += [{"source": source} for source in args.sources]
        if resume_id and not rows:
            # Nothing left FROM THE QUEUE -- killed after its last item was
            # withdrawn, or before it published anything. Treating that as "no
            # input" left the batch `running` in the registry forever, and
            # every later resume picked it again and returned 2 (reviewer
            # 2026-08-31 P2). It goes on through the ordinary path with zero
            # starting items rather than being closed here: the control channel
            # may hold a task this batch was never able to admit, and closing
            # it early would also have written the registry without the batch
            # lock (reviewer 2026-08-31 P1). The empty run takes one intake
            # poll and then settles itself.
            print(f"[batch] {resume_id}: nothing left in the queue")
        elif not rows:
            print(
                "provide a source (path or URL) and/or --manifest", file=sys.stderr
            )
            return 2
        single = len(rows) == 1 and not args.manifest
        if not single:
            # Truthiness, not `is not None`: since 2026-08-31 all of these
            # reach here as None when unset, so on a bare command line the
            # two agree -- but an explicitly empty one (`--task-id ""`) is
            # still not something the user asked for, and that is the case
            # truthiness goes on covering.
            named = sorted(
                key for key in _PER_SOURCE_KEYS | {"name"} if getattr(args, key, None)
            )
            if named:
                print(
                    "these name one run's own outputs and cannot serve several "
                    f"sources: {', '.join('--' + key.replace('_', '-') for key in named)}"
                    " (set them per manifest row instead)",
                    file=sys.stderr,
                )
                return 2
        defaults = _defaults_from_args(args, single=single)
        if args.name and not defaults.get("output"):
            defaults["output"] = resolve_name_output_path(args.name)
        # Only the startup rows of a resume, and only the options actually
        # typed: rows appended to a manifest or the control channel keep the
        # documented "the row wins" rule, being the user's own writing.
        override = explicit_row_keys() if resume_id else frozenset()
        merged = [
            merge_item_options(strip_view_keys(row), defaults, override=override)
            for row in rows
        ]
        if resume_id:
            changed = sorted(
                key
                for key in override
                if any(
                    key in row and row[key] != defaults.get(key)
                    for row in (strip_view_keys(r) for r in rows)
                )
            )
            if changed:
                print(
                    f"[batch] {resume_id}: {', '.join(changed)} taken from this "
                    "command line instead of what the batch recorded"
                )
            if "stage" in override:
                target = PIPELINE_STAGE_ORDER.get(str(defaults.get("stage") or ""), -1)
                revived = [
                    row
                    for row in rows
                    if row.get("_state") == "done"
                    and PIPELINE_STAGE_ORDER.get(str(row.get("stage") or ""), 0) < target
                ]
                if revived:
                    # Extending the stage gives finished items work again, and
                    # past raw-srt that work spends LLM quota. Never silently.
                    print(
                        f"[batch] {len(revived)} finished item(s) now go on to "
                        f"{defaults['stage']} as well"
                    )
        workers = (
            {"download": 1, "asr": 1, "llm": 1}
            if single
            else {
                "download": max(1, args.download_workers),
                "asr": profile_asr_workers(merged),
                "llm": max(1, args.max_parallel_tasks),
            }
        )
        for opts in merged:
            # Metadata only (`workers.batch` in the run sidecar), and only
            # when there was something to share the machine with.
            if not single:
                opts["_batch_workers"] = workers
        # One book for every door into the run: these items, and any row
        # appended while it is in flight (reviewer 2026-08-30 P1).
        claims = OutputClaims()
        items = [build_item(opts, claims=claims) for opts in merged]
        intake = (
            manifest_intake(
                args.manifest,
                defaults=defaults,
                build=lambda opts: build_item(
                    {**opts, "_batch_workers": workers}, claims=claims
                ),
                # The strict startup parse consumed every line of THAT
                # snapshot, trailing newline or not.
                consumed_lines=consumed_lines,
                consumed_tail=consumed_tail,
            )
            # Not when resuming: there the "manifest" is the batch's own queue
            # view, which this run is about to own and rewrite. Polling a file
            # you are the writer of is the two-writers mistake in one process.
            # New work still arrives the usual way, through the control channel.
            if args.manifest and not resume_id
            else None
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Failed to build the run: {exc}", file=sys.stderr)
        return 2

    # `args.log_level or "normal"` here is what stopped FINESUB_LOG_LEVEL from
    # reaching a run: the reporter only consults the environment when nobody
    # named a level (reviewer 2026-08-30 P2). Resolve once, use everywhere.
    level = resolve_log_level(args.log_level)
    # A status log is the record of a run nobody watched line by line. One
    # source in the foreground does not need one; --batch-id asks for it.
    status_path = None
    if not single or args.batch_id:
        batch_id = args.batch_id or resume_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        status_path = DEFAULT_BATCH_ROOT / batch_id / STATUS_FILENAME
        print(f"Run {batch_id}: {len(items)} item(s); status -> {status_path}")
    # The run's own queue: a view it publishes and a channel it takes
    # instructions from (see `write_queue_view`). Only when there is a batch
    # directory to put them in -- a single foreground run has neither a reader
    # nor a second terminal in mind.
    publish = None
    if status_path is not None:
        view_path = status_path.with_name(QUEUE_VIEW_FILENAME)
        publish = lambda batch_items, batch_results: write_queue_view(  # noqa: E731
            view_path, batch_items, batch_results
        )
        print(f"[batch] queue -> {view_path}")
    # The control channel is the batch form's: `_run_single` keeps the terminal
    # for its one item (progress redrawn in place), so it takes no intake --
    # and advertising a channel nobody polls is worse than not having one.
    # Nothing to reorder or add to in a one-item foreground run anyway.
    if status_path is not None and not single:
        control_path = status_path.with_name(CONTROL_FILENAME)
        control = control_intake(
            control_path,
            admit=lambda row: build_item(
                {**merge_item_options(row, defaults), "_batch_workers": workers},
                claims=claims,
            ),
            # Durable across runs of this batch, and advanced only once a
            # poll's lines have taken effect (see `control_intake`).
            cursor_path=status_path.with_name(CONTROL_CURSOR_FILENAME),
        )
        intake = merged_intake(intake, control) if intake is not None else control
        print(
            f"[batch] add a task / reprioritise / drop one by appending to "
            f"{control_path}"
        )
    if args.manifest and not resume_id:
        print(
            f"[batch] appending rows to {args.manifest} while this runs adds "
            "them to it (append-only; picked up within a few seconds)"
        )

    with ExitStack() as held:
        if status_path is not None:
            # Held for the whole run, so that "is this batch live?" has an
            # answer nobody has to maintain: the OS releases it when the
            # process ends, however it ends. Two runs on one batch directory
            # would write the same outputs from two sets of workers -- reachable
            # by resuming a running batch, and by giving --batch-id twice
            # (reviewer 2026-08-31 P1).
            from finesub_bootstrap.locks import LockUnavailable, holding_lock

            try:
                held.enter_context(holding_lock(batch_lock_path(view_path), timeout=0))
            except LockUnavailable:
                # Which way out exists depends on the form: a batch has a
                # control channel to add to, a single run polls none -- so the
                # only honest advice there is another id (and naming the
                # channel it does not have was a NameError: reviewer
                # 2026-08-31 P2).
                advice = (
                    "wait for it, or give this run its own --batch-id"
                    if single
                    else f"add to it through {control_path} instead of "
                    "starting a second one"
                )
                print(
                    f"batch {batch_id} is already running (its queue is "
                    f"{view_path}); {advice}",
                    file=sys.stderr,
                )
                return 2
            if args.batch_id:
                warn_on_unfinished_batch_id(batch_id)
        if single:
            return _run_single(
                items[0], merged[0], workers, status_path, level, args.retry_failed, publish
            )
        if status_path is not None:
            record_batch(batch_id, status_path.with_name(QUEUE_VIEW_FILENAME),
                         state="running", items=len(items))
        # Quieting is a front end's job, and this is one. Without it a batch --
        # the run with the *most* output, several items deep -- would be the only
        # entry point still carrying every library's version banner.
        with quieted_libraries(level), _batch_item_reporters(level) as item_reporter:
            results = run_batch(
                items,
                workers=workers,
                asr_queue_size=args.asr_queue_size,
                status_path=status_path,
                log_level=level,
                intake=intake,
                item_reporter=item_reporter,
                retry_failed=args.retry_failed,
                publish=publish,
            )
        # Still under the batch lock: released first, a `--resume-batch` that
        # took it in between had its `running` row overwritten by this ending.
        print_summary(results)
        if status_path is not None:
            record_batch(
                batch_id,
                status_path.with_name(QUEUE_VIEW_FILENAME),
                state="finished" if _settled(results) else "unfinished",
                items=len(results),
            )
    return 0 if _settled(results) else 1


def _run_single(
    item: BatchItem,
    opts: Mapping[str, Any],
    workers: Mapping[str, int],
    status_path: Path | None,
    level: str,
    retry_failed: int | None = None,
    publish: Callable[[Sequence[BatchItem], Sequence[ItemResult]], None] | None = None,
) -> int:
    """One source, in the foreground: same runner, this front end's manners.

    What N=1 keeps that a batch cannot have: progress redrawn in place (one
    item owns the terminal), the run log file, and a traceback on failure --
    the diagnosis for the run someone is watching. The runner takes both as
    hooks rather than modes, so nothing about scheduling forks here.
    """

    # The log context wraps the whole run *including* its failure path: the
    # error handler writes the single most useful line, and with the file
    # already closed it reached the terminal only.
    _prune_run_logs()
    with _run_log(Path(str(opts["source"])).stem) as file_reporter:
        reporter = terminal_reporter(level=level)
        terminal_level = reporter.level
        if file_reporter is not None:
            reporter = FanOutReporter(reporter, file_reporter)

        def _on_error(result: ItemResult, exc: BaseException) -> None:
            # str(exc) alone is often empty (bare RuntimeError, CUDA/driver
            # errors), which used to make a failed run indistinguishable from
            # a silent exit.
            #
            # The bin that failed, which is the finest granularity the runner
            # knows -- and it has a label of its own now. Naming the run's
            # target stage instead read 失败（最终字幕） for a download that
            # never started (reviewer 2026-08-30 P2); naming the bare bin key
            # leaked an English word into a Chinese line.
            reporter.failed(result.failed_stage, str(exc).strip() or repr(exc))
            # The traceback is the diagnosis. It goes to the terminal as
            # before, and into the log file -- which is the one a user
            # actually sends.
            if file_reporter is not None:
                file_reporter.block("traceback", traceback.format_exc())
            traceback.print_exc()

        # `quieted_libraries` keeps the *terminal* level on purpose: its
        # verbose branch also un-mutes third-party logging and tqdm, and a
        # progress bar captured into the log file is pure noise.
        with quieted_libraries(terminal_level):
            results = run_batch(
                [item],
                workers=workers,
                status_path=status_path,
                log_level=level,
                item_reporter=lambda _label: reporter,
                on_item_error=_on_error,
                retry_failed=retry_failed,
                publish=publish,
            )
    return 0 if results[0].status == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())

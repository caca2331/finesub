"""What a pipeline run tells the outside world, separated from how it looks.

Stages report events; a renderer decides what reaches a terminal, an event
stream or a file. That split is the point: the same run has four audiences --
someone watching progress, someone who needs to know the output is degraded,
someone profiling, and someone debugging the algorithm -- and until now every
one of them was served by `print`, so the only way to keep the fourth was to
show all four.

Two things are deliberate:

* **The current reporter is thread-local, not global.** The batch runner keeps
  separate pools for asr and llm work, so two pipelines can be in flight in one
  process; a global would interleave their events into one meaningless stream.
  Threads a stage starts itself do not inherit it -- pass `current_reporter()`
  into the pool's `initializer` (see `bind_reporter`).
* **Wording belongs to the stage, layout belongs to the renderer.** `summary`
  takes labelled numbers rather than a sentence, because only the stage knows
  that 7 removed segments are "噪声片段", and only the renderer knows whether
  this run has a terminal to redraw.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import os
import re
from pathlib import Path
import sys
import threading
import time
from typing import Any, Protocol, TextIO, runtime_checkable

#: Least to most talkative. `quiet` keeps only what changes what the user does.
LEVELS = ("quiet", "normal", "verbose")

#: Stage labels: the one place a stage gets its human name. One run described
#: two ways by two renderers is a support problem, not a style one.
STAGE_LABELS = {
    "vocal": "人声分离",
    "aligned": "语音识别",
    "stable": "字幕稳定化",
    "raw-srt": "原始字幕",
    "translated-srt": "纠错翻译",
    "final-srt": "最终字幕",
    # The runner's bins. A failure knows which bin it happened in, not which
    # pipeline stage -- naming the run's *target* stage instead said
    # 失败（最终字幕） for a download that never got started (reviewer
    # 2026-08-30 P2).
    "download": "下载",
    "asr": "转写",
    "llm": "纠错翻译",
}


@runtime_checkable
class Reporter(Protocol):
    """Everything a stage may say about a run in progress."""

    def planned(self, stages: Sequence[str]) -> None:
        """The stages this run intends to visit, in order.

        Reported once, before the first one starts: it is what turns a stage
        into `[2/4]`. The count comes from the run's own plan -- `--stage` and
        artifacts already on disk both change it, so a fixed denominator would
        be wrong more often than right.
        """

    def stage_started(self, stage: str, *, reused: bool = False, detail: str = "") -> None:
        ...

    def progress(
        self,
        stage: str,
        *,
        completed: int,
        total: int | None = None,
        unit: str = "",
        detail: str = "",
    ) -> None:
        ...

    def summary(self, stage: str, metrics: Mapping[str, object]) -> None:
        """Close a stage with labelled numbers, e.g. ``{"移除噪声": 7}``."""

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        ...

    def debug(self, message: str, fields: Mapping[str, object] | None = None) -> None:
        ...

    def completed(self, output: str | Path, elapsed_sec: float) -> None:
        ...

    def failed(self, stage: str, message: str) -> None:
        ...


class NullReporter:
    """Accepts everything, shows nothing.

    The default, so that library code can report unconditionally instead of
    guarding every call site with `if reporter is not None`.
    """

    def planned(self, stages: Sequence[str]) -> None:
        return

    def stage_started(self, stage: str, *, reused: bool = False, detail: str = "") -> None:
        return

    def progress(
        self,
        stage: str,
        *,
        completed: int,
        total: int | None = None,
        unit: str = "",
        detail: str = "",
    ) -> None:
        return

    def summary(self, stage: str, metrics: Mapping[str, object]) -> None:
        return

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        return

    def debug(self, message: str, fields: Mapping[str, object] | None = None) -> None:
        return

    def completed(self, output: str | Path, elapsed_sec: float) -> None:
        return

    def failed(self, stage: str, message: str) -> None:
        return


_NULL = NullReporter()
_local = threading.local()


def current_reporter() -> Reporter:
    """The reporter for this thread; a no-op one when nothing bound it."""

    return getattr(_local, "reporter", _NULL)


def reporter_delivers(reporter: Reporter) -> bool:
    """Does this reporter carry a message anywhere?

    The null reporter accepts every call and returns, so calling it cannot
    tell "reported" from "dropped on the floor". Something that must reach a
    person exactly once -- and would otherwise mark itself said -- has to ask.
    """

    return reporter is not _NULL


def bind_reporter(reporter: Reporter) -> None:
    """Bind `reporter` to this thread and leave it bound.

    For worker threads that outlive no scope of their own -- pass this as a
    pool's `initializer` so blocks running there report to the same place their
    parent does.
    """

    _local.reporter = reporter


@contextmanager
def reporting_to(reporter: Reporter | None) -> Iterator[Reporter]:
    """Bind `reporter` for the duration of the block, then restore."""

    chosen = reporter if reporter is not None else _NULL
    previous = getattr(_local, "reporter", None)
    _local.reporter = chosen
    try:
        yield chosen
    finally:
        if previous is None:
            del _local.reporter
        else:
            _local.reporter = previous


#: Third-party loggers that narrate their own progress at INFO. Measured on a
#: 3-minute run: audio-separator alone produced 25 of the 45 stderr lines --
#: version banners, the host's OS and CPU, every file it opened. Its warnings
#: are worth keeping, so these are raised to WARNING rather than silenced, and
#: `verbose` leaves them alone.
#: audio-separator's loggers are named after the module's last component, not
#: after the package -- `separator`, not `audio_separator.separator`. Guessing
#: the package name left every one of those 25 lines in place while the unit
#: test, which used the same guess, passed.
NOISY_LIBRARY_LOGGERS = (
    "separator",
    "common_separator",
    "mdxc_separator",
    "mdx_separator",
    "vr_separator",
    "audio_separator",
    "faster_whisper",
    "huggingface_hub",
    "transformers",
    "urllib3",
)


@contextmanager
def quieted_libraries(level: str) -> Iterator[None]:
    """Keep third-party libraries from narrating over the run's own report.

    Two channels, because they are two mechanisms: Python logging, raised to
    WARNING so a library can still say something went wrong; and `tqdm`,
    disabled outright, since a progress bar written into a captured log is
    never anything but noise -- and the pipeline draws its own.

    `verbose` opts out of both: someone who asked for everything gets it.
    """

    if level == "verbose":
        yield
        return
    # Depth-counted like the tqdm patch beside it: an inner scope exiting must
    # not un-quiet an outer one that is still running.
    global _quiet_depth
    with _TQDM_LOCK:
        _quiet_depth += 1
    try:
        with _quiet_logging(), _disabled_tqdm():
            yield
    finally:
        with _TQDM_LOCK:
            _quiet_depth -= 1


def libraries_quieted() -> bool:
    """Whether a library that configures its own logging should stay quiet.

    Raising a logger's level does not reach a library that sets it from its own
    constructor -- audio-separator takes a `log_level` argument and applies it
    when it is built, after anything we did beforehand. Such a library gets
    told directly; this is how the adapter knows what to tell it.
    """

    return _quiet_depth > 0


@contextmanager
def _quiet_logging() -> Iterator[None]:
    import logging

    previous: list[tuple[Any, int]] = []
    for name in NOISY_LIBRARY_LOGGERS:
        logger = logging.getLogger(name)
        previous.append((logger, logger.level))
        if logger.level < logging.WARNING:
            logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in previous:
            logger.setLevel(level)


@contextmanager
def _disabled_tqdm() -> Iterator[None]:
    """Force every `tqdm` instance to be disabled for the duration.

    Patched on the class, not on the module attribute: libraries bind
    `tqdm` at import time (`from tqdm import tqdm`), so rebinding the name
    afterwards would reach nobody. Absent tqdm, this is a no-op.
    """

    try:
        from tqdm.std import tqdm as tqdm_class
    except Exception:
        yield
        return

    original = tqdm_class.__init__

    def silenced(self, *args, **kwargs):
        kwargs["disable"] = True
        return original(self, *args, **kwargs)

    with _TQDM_LOCK:
        global _tqdm_depth
        if _tqdm_depth == 0:
            _TQDM_ORIGINAL["init"] = original
            tqdm_class.__init__ = silenced
        _tqdm_depth += 1
    try:
        yield
    finally:
        with _TQDM_LOCK:
            _tqdm_depth -= 1
            if _tqdm_depth == 0:
                tqdm_class.__init__ = _TQDM_ORIGINAL.pop("init")


_TQDM_LOCK = threading.Lock()
_TQDM_ORIGINAL: dict[str, Any] = {}
_tqdm_depth = 0
_quiet_depth = 0


def resolve_log_level(level: str | None) -> str:
    """An explicit level, else `FINESUB_LOG_LEVEL`, else `normal`.

    One place, because a front end needs the answer twice -- for the reporter
    it builds and for `quieted_libraries` -- and resolving it early with
    `args.log_level or "normal"` is how the environment variable stopped
    reaching a run at all (reviewer 2026-08-30 P2).
    """

    resolved = level or os.environ.get("FINESUB_LOG_LEVEL", "normal")
    return resolved if resolved in LEVELS else "normal"


def terminal_reporter(
    stream: TextIO | None = None,
    *,
    level: str | None = None,
) -> "TerminalReporter":
    """The renderer a command-line entry point binds around its run.

    On stderr, so redirecting stdout to capture a result does not also capture
    the progress that described producing it. Every entry point needs this:
    the thread-local default is silent, which is right for a library but would
    quietly drop a CPU-fallback warning from a standalone module CLI.
    """

    return TerminalReporter(stream or sys.stderr, level=resolve_log_level(level))


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage)


#: `scheme://userinfo@host` -- the ordinary way to point at a private index or
#: proxy, and a credential.
_URL_USERINFO = re.compile(r"(?i)([a-z][a-z0-9+.\-]*://)[^/@\s]+@")


def redact_credentials(text: str) -> str:
    """Strip userinfo out of any URL in a line that is about to be logged.

    The run log exists to be sent to a developer, which makes it the same kind
    of surface as `doctor` output -- and `finesub_bootstrap.shell._safe_host`
    already refuses to print userinfo for exactly that reason. The addresses
    that reach a log line here are the user's own overrides (`[llm] proxy`, a
    custom endpoint's `base_url`), and `https://user:token@host` is an ordinary
    way to write one, so an endpoint's error message can carry a credential
    without anyone meaning it to.

    Deliberately narrow: this removes what a *URL* carries, not every secret a
    provider might echo. The rule that keeps keys out of these lines is that
    they are never put in -- only labels are.
    """

    return _URL_USERINFO.sub(r"\1***@", text)


def format_clock(seconds: float) -> str:
    """`02:18`, or `1:04:09` once it runs past an hour."""

    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """`11m 32s` -- for totals, where a clock reads like a timestamp."""

    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_metrics(metrics: Mapping[str, object]) -> str:
    """Labelled numbers as one line, dropping the ones that are zero.

    A stage reports everything it counted; a summary listing six zeroes says
    nothing worth a line of terminal. Explicit `None` is dropped too, so a
    stage can report "not measured" without inventing a value.
    """

    parts = []
    for label, value in metrics.items():
        if value is None or value == 0 or value == "":
            continue
        parts.append(f"{label} {value}" if label else str(value))
    return "，".join(parts)


class TerminalReporter:
    """Render a run for a person watching a terminal.

    On a TTY one progress line is rewritten in place. Without one -- a file, a
    CI log, the desktop worker's captured pipe -- nothing is ever rewritten and
    progress only earns a new line when it crosses a tenth or the stage
    changes, which is what keeps a redirected run from being mostly progress.
    """

    #: How often a TTY progress line may be redrawn. Fast enough to look live,
    #: slow enough that a stage reporting every item does not become the
    #: bottleneck.
    REDRAW_INTERVAL_SEC = 0.2

    def __init__(
        self,
        stream: TextIO,
        *,
        level: str = "normal",
        isatty: bool | None = None,
        prefix: str = "",
        clock=time.monotonic,
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown log level: {level}")
        self.stream = stream
        self.level = level
        # Names which run a line belongs to when several share a terminal --
        # the batch runner's whole reason for not letting them share a
        # redrawn line either.
        self.prefix = prefix
        self._isatty = stream.isatty() if isatty is None else isatty
        self._clock = clock
        self._lock = threading.RLock()
        self._stages: list[str] = []
        self._stage_index: dict[str, int] = {}
        self._current_stage: str | None = None
        self._line_open = False
        self._line_width = 0
        # None rather than 0.0: the throttle must never swallow the *first*
        # update of a stage, and a zero here compares equal to a monotonic
        # clock that starts at zero.
        self._last_draw: float | None = None
        self._last_tenth: dict[str, tuple[int, int | None]] = {}

    # -- Reporter ------------------------------------------------------

    def planned(self, stages: Sequence[str]) -> None:
        with self._lock:
            self._stages = list(stages)
            self._stage_index = {stage: index for index, stage in enumerate(self._stages, 1)}

    def stage_started(self, stage: str, *, reused: bool = False, detail: str = "") -> None:
        with self._lock:
            self._current_stage = stage
            self._last_tenth.pop(stage, None)
            self._last_draw = None
            if self.level == "quiet":
                return
            # Close the previous stage's line first: it is redrawn in place
            # while it runs, and starting the next stage on top of it would
            # erase the only record that the previous one finished.
            self._close_line()
            note = "已有结果，跳过" if reused else detail
            self._emit_stage_line(stage, note, final=reused)

    def progress(
        self,
        stage: str,
        *,
        completed: int,
        total: int | None = None,
        unit: str = "",
        detail: str = "",
    ) -> None:
        if self.level == "quiet":
            return
        with self._lock:
            parts: list[str] = []
            if total:
                parts.append(f"{int(completed * 100 / total)}%")
            # A count of nothing, with no total and no unit, is a stage saying
            # "still here, and this is what I am doing" -- the detail is the
            # whole message, and "0" in front of it is noise.
            if total or unit or completed:
                parts.append(f"{completed}/{total}" if total else str(completed))
            if unit:
                parts[-1] = f"{parts[-1]} {unit}"
            if detail:
                parts.append(detail)
            body = " · ".join(parts)
            if self._isatty:
                now = self._clock()
                finished = total is not None and completed >= total
                if (
                    not finished
                    and self._last_draw is not None
                    and now - self._last_draw < self.REDRAW_INTERVAL_SEC
                ):
                    return
                self._last_draw = now
                self._emit_stage_line(stage, body, final=False)
                return
            # No terminal to redraw: one line per tenth, and never for the
            # same tenth twice.
            if total:
                tenth = int(completed * 10 / total)
            else:
                tenth = completed
            # Keyed on the total too; see FileReporter.progress for why.
            if self._last_tenth.get(stage) == (tenth, total):
                return
            self._last_tenth[stage] = (tenth, total)
            self._emit_stage_line(stage, body, final=True)

    def summary(self, stage: str, metrics: Mapping[str, object]) -> None:
        if self.level == "quiet":
            return
        body = _format_metrics(metrics)
        if not body:
            return
        with self._lock:
            self._close_line()
            self._write(f"      {stage_label(stage)}摘要：{body}\n")

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        # Never suppressed: `quiet` means "only what changes what I do", and
        # this is that.
        tail = "；".join(part for part in (impact, action) if part)
        with self._lock:
            self._close_line()
            self._write(f"Warning: {message}" + (f"（{tail}）" if tail else "") + "\n")

    def debug(self, message: str, fields: Mapping[str, object] | None = None) -> None:
        if self.level != "verbose":
            return
        body = " ".join(f"{key}={value}" for key, value in (fields or {}).items())
        with self._lock:
            self._close_line()
            self._write(f"      {message}" + (f" {body}" if body else "") + "\n")

    def completed(self, output: str | Path, elapsed_sec: float) -> None:
        with self._lock:
            self._close_line()
            self._write(f"\n完成：{output}\n总耗时：{format_duration(elapsed_sec)}\n")

    def failed(self, stage: str, message: str) -> None:
        with self._lock:
            self._close_line()
            self._write(f"\n失败（{stage_label(stage)}）：{message}\n")

    # -- rendering -----------------------------------------------------

    def _emit_stage_line(self, stage: str, body: str, *, final: bool) -> None:
        prefix = self._prefix(stage)
        line = f"{prefix}{stage_label(stage):<8}{(' ' + body) if body else ''}"
        if not body:
            # The label is padded so progress lines align; with nothing after
            # it that padding is just trailing whitespace on every stage line.
            line = line.rstrip()
        if self._isatty and not final:
            # Padded rather than erased with `\x1b[K`: a console without VT
            # processing -- still the default in older conhost -- would print
            # the escape itself, and a progress line is the one place a stray
            # `←[K` would appear hundreds of times.
            padding = max(self._line_width - len(line), 0)
            self._write("\r" + line + " " * padding)
            self._line_width = len(line)
            self._line_open = True
            return
        self._close_line()
        self._write(line + "\n")

    def _prefix(self, stage: str) -> str:
        index = self._stage_index.get(stage)
        if index is None:
            return ""
        return f"[{index}/{len(self._stages)}] "

    def _close_line(self) -> None:
        if self._line_open:
            self._write("\n")
            self._line_open = False
            self._line_width = 0

    def _write(self, text: str) -> None:
        self.stream.write(self._prefixed(text) if self.prefix else text)
        self.stream.flush()

    def _prefixed(self, text: str) -> str:
        """Name every line this reporter starts, leaving line breaks alone."""

        return "".join(
            (self.prefix + part if part and not part.startswith(("\n", "\r")) else part)
            for part in text.splitlines(keepends=True)
        )


class FileReporter:
    """Write every event to a log file, at `verbose`, forever.

    Separate from `TerminalReporter` rather than a mode of it: the file has no
    terminal to redraw, no width to fit, and nobody watching it, so the two
    share nothing but the protocol. It also runs at `verbose` regardless of
    what the terminal was asked for -- the whole point is that the decision
    detail is on disk when someone reports a problem, without making them read
    it live.

    What it deliberately does **not** carry is the other half of `verbose`:
    `quieted_libraries` also un-mutes third-party logging and tqdm, and a
    progress bar captured into a file is nothing but noise. That switch stays
    bound to the terminal level (see `pipeline.main`).
    """

    #: Same rule the no-TTY terminal path uses: a stage reporting every item
    #: must not turn the log into progress.
    PROGRESS_STEPS = 10

    def __init__(self, stream: TextIO, *, clock=time.time) -> None:
        self.stream = stream
        self.level = "verbose"
        self._clock = clock
        self._lock = threading.RLock()
        self._last_step: dict[str, tuple[int, int | None]] = {}

    # -- Reporter ------------------------------------------------------

    def planned(self, stages: Sequence[str]) -> None:
        self._line("plan", " → ".join(stages))

    def stage_started(self, stage: str, *, reused: bool = False, detail: str = "") -> None:
        with self._lock:
            self._last_step.pop(stage, None)
        note = "reused" if reused else detail
        self._line("stage", f"{stage}" + (f" ({note})" if note else ""))

    def progress(
        self,
        stage: str,
        *,
        completed: int,
        total: int | None = None,
        unit: str = "",
        detail: str = "",
    ) -> None:
        step = int(completed * self.PROGRESS_STEPS / total) if total else completed
        with self._lock:
            # Keyed on the total as well: a stage whose denominator grows mid-run
            # -- correction splits a window that overran its envelope -- lands
            # the next item on the same tenth of a bigger whole, and comparing
            # the step alone swallowed that item's only event. See the LLM
            # section of docs/reporting.md ("LLM 段" -> "三条约定").
            if self._last_step.get(stage) == (step, total):
                return
            self._last_step[stage] = (step, total)
        body = f"{completed}/{total}" if total else str(completed)
        if unit:
            body = f"{body} {unit}"
        self._line("progress", f"{stage} {body}" + (f" · {detail}" if detail else ""))

    def summary(self, stage: str, metrics: Mapping[str, object]) -> None:
        body = _format_metrics(metrics)
        if body:
            self._line("summary", f"{stage} {body}")

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        tail = "；".join(part for part in (impact, action) if part)
        self._line("warning", f"[{code}] {message}" + (f" ({tail})" if tail else ""))

    def debug(self, message: str, fields: Mapping[str, object] | None = None) -> None:
        body = " ".join(f"{key}={value}" for key, value in (fields or {}).items())
        self._line("debug", message + (f" {body}" if body else ""))

    def completed(self, output: str | Path, elapsed_sec: float) -> None:
        self._line("done", f"{output} in {format_duration(elapsed_sec)}")

    def failed(self, stage: str, message: str) -> None:
        self._line("failed", f"{stage}: {message}")

    def block(self, kind: str, text: str) -> None:
        """Record something that does not fit on one line, e.g. a traceback.

        Not part of `Reporter`: only the file has room for it, and a terminal
        already gets the traceback from `traceback.print_exc()`.
        """

        for index, line in enumerate(text.rstrip().splitlines()):
            self._line(kind if index == 0 else "", line)

    # -- rendering -----------------------------------------------------

    def _line(self, kind: str, body: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(self._clock()))
        # One lock around the whole write: the correction stage runs several
        # windows at once and separate `write` calls from two threads can
        # interleave inside one line.
        with self._lock:
            try:
                self.stream.write(f"{stamp} {kind:<8} {body}\n")
            except (OSError, ValueError):
                # A log that cannot be written is not a reason to fail the run
                # (ValueError: the stream was closed under us at shutdown).
                pass


class FanOutReporter:
    """Send every event to several reporters, in order.

    Used to give one run both a terminal at the level the user asked for and a
    file at `verbose`. A reporter that raises must not take the run with it --
    reporting is commentary, not work.
    """

    def __init__(self, *reporters: Any) -> None:
        self._reporters = tuple(reporters)
        self.level = max(
            (getattr(one, "level", "quiet") for one in self._reporters),
            key=LEVELS.index,
            default="quiet",
        )

    def _each(self, name: str, *args: Any, **kwargs: Any) -> None:
        for reporter in self._reporters:
            try:
                getattr(reporter, name)(*args, **kwargs)
            except Exception:  # pragma: no cover - a renderer's own bug
                pass

    def planned(self, stages: Sequence[str]) -> None:
        self._each("planned", stages)

    def stage_started(self, stage: str, *, reused: bool = False, detail: str = "") -> None:
        self._each("stage_started", stage, reused=reused, detail=detail)

    def progress(self, stage: str, **kwargs: Any) -> None:
        self._each("progress", stage, **kwargs)

    def summary(self, stage: str, metrics: Mapping[str, object]) -> None:
        self._each("summary", stage, metrics)

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        self._each("warning", code, message, impact=impact, action=action)

    def debug(self, message: str, fields: Mapping[str, object] | None = None) -> None:
        self._each("debug", message, fields)

    def completed(self, output: str | Path, elapsed_sec: float) -> None:
        self._each("completed", output, elapsed_sec)

    def failed(self, stage: str, message: str) -> None:
        self._each("failed", stage, message)

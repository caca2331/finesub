from __future__ import annotations

import io
import threading

from finesub.reporting import (
    FanOutReporter,
    FileReporter,
    NullReporter,
    TerminalReporter,
    bind_reporter,
    current_reporter,
    format_duration,
    reporting_to,
)


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _reporter(**kwargs) -> tuple[TerminalReporter, io.StringIO]:
    stream = io.StringIO()
    return TerminalReporter(stream, **kwargs), stream


def test_non_tty_progress_emits_one_line_per_tenth() -> None:
    reporter, stream = _reporter(isatty=False)
    reporter.planned(["vocal", "aligned"])
    reporter.stage_started("aligned")
    for completed in range(1, 101):
        reporter.progress("aligned", completed=completed, total=100, unit="groups")

    lines = [line for line in stream.getvalue().splitlines() if line]
    # One stage line, then at most one line per tenth crossed.
    assert len(lines) <= 1 + 11
    assert lines[0].startswith("[2/2] 语音识别")
    assert "\r" not in stream.getvalue()


def test_non_tty_never_repeats_a_tenth() -> None:
    reporter, stream = _reporter(isatty=False)
    reporter.stage_started("aligned")
    reporter.progress("aligned", completed=5, total=100, unit="groups")
    reporter.progress("aligned", completed=6, total=100, unit="groups")
    reporter.progress("aligned", completed=7, total=100, unit="groups")

    assert stream.getvalue().count("groups") == 1


def test_tty_rewrites_one_line_and_keeps_the_finished_stage() -> None:
    clock = _Clock()
    reporter, stream = _reporter(isatty=True, clock=clock)
    reporter.planned(["vocal", "aligned"])
    reporter.stage_started("vocal")
    reporter.progress("vocal", completed=1, total=6, unit="blocks")
    clock.now += 1.0
    reporter.progress("vocal", completed=6, total=6, unit="blocks")
    reporter.stage_started("aligned")

    output = stream.getvalue()
    # The separator's final state survives starting the next stage.
    assert "6/6 blocks" in output
    assert output.count("\n") == 1
    assert output.rstrip().endswith("语音识别")


def test_tty_progress_is_throttled_between_redraws() -> None:
    clock = _Clock()
    reporter, stream = _reporter(isatty=True, clock=clock)
    reporter.stage_started("aligned")
    for completed in range(1, 40):
        reporter.progress("aligned", completed=completed, total=100, unit="groups")

    # The first update always draws; the rest fall inside one throttle window.
    assert stream.getvalue().count("groups") == 1
    assert "1% · 1/100 groups" in stream.getvalue()


def test_tty_padding_clears_a_shorter_line_without_escapes() -> None:
    clock = _Clock()
    reporter, stream = _reporter(isatty=True, clock=clock)
    reporter.stage_started("aligned")
    reporter.progress("aligned", completed=1, total=100, unit="groups aaaaaaaaaa")
    clock.now += 1.0
    reporter.progress("aligned", completed=99, total=100, unit="g")

    output = stream.getvalue()
    assert "\x1b" not in output
    # The shorter second line is padded out to at least cover the longer one
    # it replaces, so no tail of the old text survives on screen.
    drawn = output.split("\r")
    assert len(drawn[-1]) >= len(drawn[-2])


def test_a_countless_update_shows_only_what_it_is_doing() -> None:
    """A stage saying "still here, this is why" has no number to show."""

    reporter, stream = _reporter(isatty=False)
    reporter.stage_started("vocal")
    reporter.progress("vocal", completed=0, detail="正在为本机编译分离器")

    line = stream.getvalue().splitlines()[-1]
    assert line.endswith("正在为本机编译分离器")
    assert "0" not in line


def test_quiet_keeps_warnings_and_drops_progress() -> None:
    reporter, stream = _reporter(isatty=False, level="quiet")
    reporter.stage_started("aligned")
    reporter.progress("aligned", completed=1, total=2, unit="groups")
    reporter.summary("aligned", {"救援": 2})
    reporter.warning("cpu-fallback", "GPU 不可用，改用 CPU", impact="速度显著下降")

    output = stream.getvalue()
    assert "语音识别" not in output
    assert "Warning: GPU 不可用，改用 CPU（速度显著下降）" in output


def test_summary_drops_zero_and_missing_counters() -> None:
    reporter, stream = _reporter(isatty=False)
    reporter.summary("stable", {"段": "412 -> 405", "移除噪声": 7, "丢弃": 0, "未测": None})

    line = stream.getvalue().rstrip("\n")
    assert line == "      字幕稳定化摘要：段 412 -> 405，移除噪声 7"


def test_summary_with_only_zero_counters_prints_nothing() -> None:
    reporter, stream = _reporter(isatty=False)
    reporter.summary("stable", {"移除噪声": 0, "丢弃": 0})

    assert stream.getvalue() == ""


def test_debug_is_verbose_only() -> None:
    reporter, stream = _reporter(isatty=False, level="normal")
    reporter.debug("beam rescue", {"group": 12})
    assert stream.getvalue() == ""

    verbose, verbose_stream = _reporter(isatty=False, level="verbose")
    verbose.debug("beam rescue", {"group": 12})
    assert "beam rescue group=12" in verbose_stream.getvalue()


def test_current_reporter_is_per_thread() -> None:
    reporter, _ = _reporter(isatty=False)
    seen: dict[str, object] = {}

    def worker() -> None:
        seen["unbound"] = current_reporter()

    with reporting_to(reporter):
        assert current_reporter() is reporter
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    # A thread the stage starts itself does not inherit the binding; pools have
    # to pass it explicitly.
    assert isinstance(seen["unbound"], NullReporter)
    assert isinstance(current_reporter(), NullReporter)


def test_bind_reporter_serves_pool_workers() -> None:
    reporter, _ = _reporter(isatty=False)
    seen: dict[str, object] = {}

    def worker() -> None:
        bind_reporter(reporter)
        seen["bound"] = current_reporter()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen["bound"] is reporter


def test_reporting_to_restores_a_nested_binding() -> None:
    outer, _ = _reporter(isatty=False)
    inner, _ = _reporter(isatty=False)

    with reporting_to(outer):
        with reporting_to(inner):
            assert current_reporter() is inner
        assert current_reporter() is outer
    assert isinstance(current_reporter(), NullReporter)


def test_nested_quieting_survives_the_inner_scope_exiting() -> None:
    """The flag used to be a plain bool reset unconditionally, so an inner
    scope un-quieted an outer one that was still running -- and the separator,
    which asks this at construction time, went back to INFO mid-run."""

    from finesub.reporting import libraries_quieted, quieted_libraries

    with quieted_libraries("normal"):
        with quieted_libraries("normal"):
            assert libraries_quieted()
        assert libraries_quieted(), "the outer scope is still in force"
    assert not libraries_quieted()


def test_verbose_does_not_claim_libraries_are_quieted() -> None:
    from finesub.reporting import libraries_quieted, quieted_libraries

    with quieted_libraries("verbose"):
        assert not libraries_quieted()


def test_the_module_annotations_resolve() -> None:
    """`Any` was used in annotations without being imported: the module still
    imported, but get_type_hints on it raised NameError."""

    import typing

    from finesub import reporting

    typing.get_type_hints(reporting)


def test_format_duration_reads_as_a_total() -> None:
    assert format_duration(9) == "9s"
    assert format_duration(692) == "11m 32s"
    assert format_duration(3725) == "1h 2m 5s"


def test_the_log_file_keeps_the_detail_a_normal_terminal_drops() -> None:
    """The whole point: decision detail is on disk without being shown live.

    `debug` is where the pipeline records *why* it took a path -- which device,
    which recovery rung, what it reused. A user running at the default level
    never sees it, and when they report a problem it has to already be written
    down somewhere.
    """

    terminal = io.StringIO()
    log = io.StringIO()
    reporter = FanOutReporter(
        TerminalReporter(terminal, level="normal", isatty=False),
        FileReporter(log),
    )

    reporter.debug("chose device", {"device": "cpu", "reason": "card too old"})

    assert "chose device" in log.getvalue()
    assert "reason=card too old" in log.getvalue()
    assert "chose device" not in terminal.getvalue()


def test_the_log_file_throttles_progress_to_one_line_per_tenth() -> None:
    """A stage reporting every item must not turn the log into progress.

    The separator and the ASR loop call `progress` per chunk; without this a
    long run's log would be almost entirely percentages.
    """

    log = io.StringIO()
    reporter = FileReporter(log)

    for done in range(0, 101):
        reporter.progress("vocal", completed=done, total=100)

    lines = [line for line in log.getvalue().splitlines() if " progress " in line]
    assert len(lines) == 11  # 0%..100%


def test_a_renderer_that_raises_does_not_take_the_run_with_it() -> None:
    """Reporting is commentary, not work."""

    class _Broken:
        level = "normal"

        def warning(self, *args, **kwargs):
            raise RuntimeError("renderer bug")

    log = io.StringIO()
    reporter = FanOutReporter(_Broken(), FileReporter(log))

    reporter.warning("cpu-fallback", "显卡太旧")

    assert "显卡太旧" in log.getvalue()


def test_the_file_reporter_serializes_writes_from_several_threads() -> None:
    """Correction runs several windows at once through one reporter."""

    log = io.StringIO()
    reporter = FileReporter(log)

    def emit(index: int) -> None:
        for _ in range(40):
            reporter.debug(f"window-{index}", {"n": index})

    threads = [threading.Thread(target=emit, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = log.getvalue().splitlines()
    assert len(lines) == 160
    # No line may carry another line's payload spliced into it.
    assert all(line.count("window-") == 1 for line in lines)


def test_a_growing_denominator_does_not_swallow_an_item() -> None:
    """Correction splits a window that overran; the whole gets bigger mid-stage.

    The tenth is computed against the total, so item 4 out of 7 lands on the
    same tenth item 3 out of 6 did. Comparing tenths alone dropped that item's
    only event -- and this stage reports once per window, so the item vanished
    from the log rather than merely arriving late. Nothing resets the memo
    inside a stage: `stage_started` is owned by the pipeline, not by the stage
    that grows its own denominator.
    """

    log = io.StringIO()
    reporter = FileReporter(log)

    for completed, total in ((1, 6), (2, 6), (3, 6), (4, 7), (5, 7), (6, 7), (7, 7)):
        reporter.progress("translated-srt", completed=completed, total=total)

    lines = [line for line in log.getvalue().splitlines() if " progress " in line]
    assert [line.split("translated-srt ")[1].split()[0] for line in lines] == [
        "1/6",
        "2/6",
        "3/6",
        "4/7",
        "5/7",
        "6/7",
        "7/7",
    ]


def test_a_growing_denominator_survives_the_no_tty_terminal_too() -> None:
    """The same arithmetic guards the line-mode terminal.

    Which is not a corner case: it is what `batch` binds for every item, and
    what any redirected run gets.
    """

    stream = io.StringIO()
    reporter = TerminalReporter(stream, level="normal", isatty=False)

    for completed, total in ((3, 6), (4, 7)):
        reporter.progress("translated-srt", completed=completed, total=total)

    assert "3/6" in stream.getvalue()
    assert "4/7" in stream.getvalue()

from __future__ import annotations

from desktop.backend.worker.protocol import EventLogWriter


def _writer() -> tuple[EventLogWriter, list[str]]:
    emitted: list[str] = []
    writer = EventLogWriter(
        "task-1", lambda event: emitted.append(event.payload["message"])
    )
    return writer, emitted


def test_a_progress_bar_becomes_one_line_not_hundreds() -> None:
    writer, emitted = _writer()
    for percent in range(0, 101):
        writer.write(f"\r  {percent}%")
    writer.write("\n")

    assert emitted == ["  100%"]


def test_the_buffer_does_not_grow_while_a_bar_redraws() -> None:
    """The old writer split on "\\n" only, so a bar with no newline accumulated.

    One separator block produced a multi-megabyte pending line, which then went
    into task-log.txt as a single row.
    """
    writer, _ = _writer()
    for percent in range(0, 1000):
        writer.write(f"\rprocessing {percent} of 1000")

    assert len(writer._buffer) < 200


def test_crlf_output_keeps_its_content() -> None:
    writer, emitted = _writer()
    writer.write("first line\r\nsecond line\r\n")

    assert emitted == ["first line", "second line"]


def test_a_line_that_never_ends_is_truncated_once() -> None:
    writer, emitted = _writer()
    writer.write("x" * (EventLogWriter.MAX_LINE_CHARS + 500))
    writer.write("y" * 10_000)
    writer.write("\ntail\n")

    assert len(emitted) == 2
    assert emitted[0].endswith("…(截断)")
    assert len(emitted[0]) <= EventLogWriter.MAX_LINE_CHARS + 16
    # Everything up to the line break is dropped, and the next line is intact.
    assert emitted[1] == "tail"


def test_interior_carriage_returns_keep_only_what_is_on_screen() -> None:
    writer, emitted = _writer()
    writer.write("stale\rfresh\n")

    assert emitted == ["fresh"]


def test_a_line_split_across_writes_is_emitted_once() -> None:
    writer, emitted = _writer()
    writer.write("half ")
    writer.write("a line")
    assert emitted == []

    writer.write("\n")
    assert emitted == ["half a line"]


def test_flush_emits_a_pending_line_without_a_newline() -> None:
    writer, emitted = _writer()
    writer.write("no newline here")
    writer.flush()

    assert emitted == ["no newline here"]
    writer.flush()
    assert emitted == ["no newline here"], "flushing twice must not repeat it"


def test_empty_lines_are_not_events() -> None:
    writer, emitted = _writer()
    writer.write("\n\nkept\n")

    assert emitted == ["kept"]

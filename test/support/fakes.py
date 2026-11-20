"""Fakes that many test files need: a token counter and warning recorders."""

from __future__ import annotations

from finesub.reporting import NullReporter


class FakeTokenCounter:
    """A token every two characters (at least one), 32 tokens a second of audio."""

    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


class WarningRecorder(NullReporter):
    """Keeps `(code, message)` of every warning, in order; every other report is dropped.

    Pass a list to have the pairs appended to one the test already holds.
    """

    def __init__(self, entries: list[tuple[str, str]] | None = None) -> None:
        self.entries = [] if entries is None else entries

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        self.entries.append((code, message))

    def codes(self) -> list[str]:
        return [code for code, _ in self.entries]


class WarningCodeRecorder(NullReporter):
    """Keeps only the code of every warning, in order; every other report is dropped.

    Pass a list to have the codes appended to one the test already holds.
    """

    def __init__(self, codes: list[str] | None = None) -> None:
        self.codes = [] if codes is None else codes

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        self.codes.append(code)

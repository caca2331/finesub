"""Parallel window dispatch (docs/llm_harness_behavior.md).

The parallel executor gives up the chained inter-window context (advice
ledger, transfer chain) for wall-clock; these tests pin what that trade must
preserve: byte-identical committed output order, drain-then-raise, the
circuit breaker, the serial->parallel directional reuse rule, and the
split-orphan marker replay.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pytest

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.client import LLMCallResult
from finesub.llm.routing.config import CapabilityTier
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.stages.correction import parallel as correction_parallel
from finesub.llm.stages.correction.parallel import PARALLEL_BREAKER_FAILURES
from finesub.llm.stages.correction.metadata import _output_limit_check

from .conftest import setattr_correction
from finesub.llm.routing.profiles import resolve_profile


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


_NUMERALS = "一二三四五六七八九十"

_SEGMENTS = [
    {"id": str(index + 1), "start": float(index * 2), "end": float(index * 2 + 1), "text": f"第{_NUMERALS[index]}段。"}
    for index in range(4)
]


def _stable_json(tmp_path: Path) -> Path:
    path = tmp_path / "clip-stable.json"
    path.write_text(json.dumps({"segments": _SEGMENTS}), encoding="utf-8")
    return path


def _windows(max_window_subtitle_tokens: int | None = 10):
    segments = [
        SubtitleSegment(s["id"], s["start"], s["end"], s["text"]) for s in _SEGMENTS
    ]
    kwargs = (
        {"max_window_subtitle_tokens": max_window_subtitle_tokens}
        if max_window_subtitle_tokens is not None
        else {}
    )
    windows = plan_correction_windows(segments, counter=FakeTokenCounter(), **kwargs)
    if max_window_subtitle_tokens is not None:
        assert len(windows) >= 2, "the fixture must plan multiple windows"
    return windows


_HEADER = "type|position|duration|gap|corrected_text|translation|conf|char_count|note"


def _reply_for(user_prompt: str) -> str:
    """A validation-passing reply derived from the window's own rows."""

    block = re.search(r"(?ms)^<asr_result>\n(.*?)^</asr_result>", user_prompt)
    rows = []
    for line in block.group(1).splitlines():
        match = re.match(r"^(\d+)\|[^|]*\|[^|]*\|[^|]*\|第(.)段。$", line)
        if match:
            rows.append((match.group(1), match.group(2)))
    lines = [f"sub|{local_id}|1.0|0.0|第{mark}段。|{mark}|8|1|" for local_id, mark in rows]
    return (
        "<reasoning>ok</reasoning>\n<translated>\n"
        + _HEADER
        + "\n"
        + "\n".join(lines)
        + "\n</translated>"
    )


class ScriptedClient:
    """Thread-safe fake client answering from the window's own rows."""

    def __init__(self, *, fail_marks: frozenset[str] = frozenset()) -> None:
        self.fail_marks = fail_marks
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def complete(self, role, messages, **kwargs):
        built = messages("capableC") if callable(messages) else messages
        user_prompt = built[-1]["content"]
        marks = set(re.findall(r"第(.)段。", user_prompt.split("<asr_result>")[-1]))
        with self._lock:
            self.calls.append("".join(sorted(marks)))
        if marks & self.fail_marks:
            content = "<translated>\nbroken\n</translated>"
        else:
            content = _reply_for(user_prompt)
        return LLMCallResult(
            content=content,
            role=role,
            model="fake",
            fallback_used=False,
            raw_response={"candidates": [{"finishReason": "STOP"}]},
        )


class BlockingClient:
    def complete(self, role, messages, **kwargs):
        raise AssertionError("no LLM call expected on a full replay")


def _run(tmp_path: Path, *, continuity: str, client, artifact_dir=None, out_name="out.srt", **kwargs):
    monkey_target = kwargs.pop("monkeypatch")
    windows = kwargs.pop("windows", None)
    knowledge_enabled = kwargs.pop("knowledge_enabled", False)
    setattr_correction(monkey_target, "RoleClient", lambda **_: client)
    return execute_correction_windows(
        stable_json=_stable_json(tmp_path),
        output_path=tmp_path / out_name,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "quality", continuity),
        knowledge_root=tmp_path / "kb",
        knowledge_enabled=knowledge_enabled,
        windows_override=windows if windows is not None else _windows(),
        task_artifact_dir=artifact_dir,
        **kwargs,
    )


def test_parallel_output_is_byte_identical_to_serial(tmp_path, monkeypatch) -> None:
    """Ordered commit: completion order must never leak into the output."""

    serial_out = _run(
        tmp_path,
        continuity="serial",
        client=ScriptedClient(),
        out_name="serial.srt",
        monkeypatch=monkeypatch,
    )
    parallel_out = _run(
        tmp_path,
        continuity="parallel",
        client=ScriptedClient(),
        out_name="parallel.srt",
        parallel_window_limit=4,
        monkeypatch=monkeypatch,
    )
    assert serial_out.read_text(encoding="utf-8") == parallel_out.read_text(
        encoding="utf-8"
    )


def test_parallel_drains_every_window_before_raising(tmp_path, monkeypatch) -> None:
    """drain-then-raise (A.5 (1)): successes are cached even when the batch fails."""

    art = tmp_path / "artifacts"
    failing = ScriptedClient(fail_marks=frozenset("二"))
    with pytest.raises(RuntimeError, match="failed validation"):
        _run(
            tmp_path,
            continuity="parallel",
            client=failing,
            artifact_dir=art,
            max_retries_per_window=0,
            parallel_window_limit=2,
            monkeypatch=monkeypatch,
        )
    cache_lines = [
        json.loads(line)
        for line in (art / "correction-windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    cached_ids = {record["chunk_id"] for record in cache_lines}
    assert cached_ids, "completed windows must be cached despite the failure"

    # The rerun replays every cached window; only the failed one is re-called.
    healed = ScriptedClient()
    _run(
        tmp_path,
        continuity="parallel",
        client=healed,
        artifact_dir=art,
        out_name="out2.srt",
        parallel_window_limit=2,
        monkeypatch=monkeypatch,
    )
    assert healed.calls and all(marks == "二" for marks in healed.calls)


def test_serial_and_parallel_caches_reuse_in_both_directions(tmp_path, monkeypatch) -> None:
    """Continuity changes affect pending work, not committed windows."""

    art = tmp_path / "artifacts"
    serial_out = _run(
        tmp_path,
        continuity="serial",
        client=ScriptedClient(),
        artifact_dir=art,
        out_name="serial.srt",
        monkeypatch=monkeypatch,
    )
    parallel_out = _run(
        tmp_path,
        continuity="parallel",
        client=BlockingClient(),
        artifact_dir=art,
        out_name="parallel.srt",
        monkeypatch=monkeypatch,
    )
    assert serial_out.read_text(encoding="utf-8") == parallel_out.read_text(
        encoding="utf-8"
    )
    cached_kinds = [
        json.loads(line)
        for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        entry["kind"] == "correction_window_cached"
        and entry["payload"].get("reused_from") == "serial"
        for entry in cached_kinds
    )

    # Reverse direction follows the same source/core-hash rule.
    art2 = tmp_path / "artifacts2"
    _run(
        tmp_path,
        continuity="parallel",
        client=ScriptedClient(),
        artifact_dir=art2,
        out_name="p1.srt",
        monkeypatch=monkeypatch,
    )
    resumed = _run(
        tmp_path,
        continuity="serial",
        client=BlockingClient(),
        artifact_dir=art2,
        out_name="s1.srt",
        monkeypatch=monkeypatch,
    )
    assert resumed.read_text(encoding="utf-8") == (tmp_path / "p1.srt").read_text(
        encoding="utf-8"
    )


def test_parallel_circuit_breaker_stops_new_windows(tmp_path, monkeypatch) -> None:
    """A systemic failure must not burn the whole daily quota (A.4 (3))."""

    failing = ScriptedClient(fail_marks=frozenset("一二三四"))
    with pytest.raises(RuntimeError):
        _run(
            tmp_path,
            continuity="parallel",
            client=failing,
            # Both retry tiers off, so one call is one window failure and the
            # call count below stays a faithful proxy for what the breaker
            # counts (it counts windows, not calls).
            max_retries_per_window=0,
            max_replacements_per_window=0,
            parallel_window_limit=1,
            monkeypatch=monkeypatch,
        )
    assert len(failing.calls) <= correction_parallel.PARALLEL_BREAKER_FAILURES


def test_parallel_resume_keeps_the_first_runs_entry_set(tmp_path, monkeypatch) -> None:
    """2026-08-10 review P1: a partial resume must reuse the first barrier's
    session entry set, not recompute a shrunken one from the pending windows."""

    art = tmp_path / "artifacts"
    setattr_correction(monkeypatch, "load_index_text", lambda root, kind: "idx")
    setattr_correction(
        monkeypatch,
        "load_entry_texts",
        lambda root, keys: ({key: f"body-{key}" for key in keys}, []),
    )
    failing = ScriptedClient(fail_marks=frozenset("二"))
    with pytest.raises(RuntimeError, match="failed validation"):
        _run(
            tmp_path,
            continuity="parallel",
            client=failing,
            artifact_dir=art,
            max_retries_per_window=0,
            knowledge_enabled=True,
            initial_transfer_keys=["K0", "K1"],
            monkeypatch=monkeypatch,
        )

    # The resumed run has a different (empty) transfer seed; without the
    # persisted set the barrier would fix an empty set for the one pending
    # window while the cached windows replay against [K0, K1].
    healed = ScriptedClient()
    _run(
        tmp_path,
        continuity="parallel",
        client=healed,
        artifact_dir=art,
        out_name="out2.srt",
        knowledge_enabled=True,
        monkeypatch=monkeypatch,
    )
    assert healed.calls and all(marks == "二" for marks in healed.calls)
    entry_sets = [
        json.loads(line)
        for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
        if '"parallel_entry_set"' in line
    ]
    assert [entry["payload"]["fixed_keys"] for entry in entry_sets] == [
        ["K0", "K1"],
        ["K0", "K1"],
    ]
    records = [
        json.loads(line)
        for line in (art / "correction-windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    persisted = [r for r in records if "parallel_entry_set" in r]
    assert len(persisted) == 1, "only the first barrier writes the set"
    assert persisted[0]["parallel_entry_set"] == ["K0", "K1"]
    for record in records:
        if record.get("chunk_id") and record.get("injected_entries") is not None:
            assert record["injected_entries"] == ["K0", "K1"]


def test_parallel_exchange_numbering_is_deterministic(tmp_path, monkeypatch) -> None:
    """2026-08-10 review P2: exchange numbering follows window order, not the
    batch's completion order (plan A.6 scheduling-time allocation)."""

    import time as time_module

    class SlowFirstClient(ScriptedClient):
        """The first planned window finishes last."""

        def complete(self, role, messages, **kwargs):
            built = messages("capableC") if callable(messages) else messages
            if "第一段。" in built[-1]["content"].split("<asr_result>")[-1]:
                time_module.sleep(0.3)
            return super().complete(role, messages, **kwargs)

    art = tmp_path / "artifacts"
    _run(
        tmp_path,
        continuity="parallel",
        client=SlowFirstClient(),
        artifact_dir=art,
        parallel_window_limit=4,
        monkeypatch=monkeypatch,
    )
    names = sorted(p.name for p in (art / "exchanges").glob("*.md"))
    expected = [
        f"{index + 1:03d}-01-correction-{window.chunk_id}-attempt0.md"
        for index, window in enumerate(_windows())
    ]
    assert names == expected


def test_split_marker_lets_a_rerun_expand_the_parent(tmp_path, monkeypatch) -> None:
    """A.5 (6): the parent's split marker replays into the child records."""

    art = tmp_path / "artifacts"
    call_count = {"n": 0}
    real_check = _output_limit_check

    def limited_once(raw_response, max_output_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {
                "limited": True,
                "basis": "test",
                "observed_output_tokens": 999_999,
                "threshold_tokens": 1,
                "max_output_tokens": max_output_tokens,
                "margin_tokens": 0,
            }
        return real_check(raw_response, max_output_tokens)

    setattr_correction(monkeypatch, "_output_limit_check", limited_once)
    _run(
        tmp_path,
        continuity="serial",
        client=ScriptedClient(),
        artifact_dir=art,
        windows=_windows(None),
        monkeypatch=monkeypatch,
    )
    setattr_correction(monkeypatch, "_output_limit_check", real_check)
    records = [
        json.loads(line)
        for line in (art / "correction-windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    markers = [record for record in records if record.get("split_into")]
    assert markers, "the split must leave a parent marker"

    # A rerun expands the parent from the marker and answers every leaf from
    # the cache -- the split call is not re-burned.
    replay_out = _run(
        tmp_path,
        continuity="serial",
        client=BlockingClient(),
        artifact_dir=art,
        out_name="replay.srt",
        windows=_windows(None),
        monkeypatch=monkeypatch,
    )
    assert replay_out.read_text(encoding="utf-8")


def test_the_breaker_counts_chains_not_windows(tmp_path, monkeypatch) -> None:
    """The threshold is priced in exhausted chains, and must stay there.

    Before the retry budget became a product, "a failed window" was exactly
    one exhausted chain of 6 calls, so 3 failures cost ~18 calls against a
    free-flash rpd of 20. Counting whole windows after the change doubled that
    silently -- 36 calls here, 48 at four lanes. Counting chains keeps the old
    price whatever the second tier is set to.

    One lane on purpose: the call count is then exact rather than a function
    of how the lanes interleave, and the breaker arithmetic under test is the
    same code either way. What four lanes add is only in-flight overshoot,
    which the next test covers.
    """

    budgets = {}
    for retries, replacements in ((5, 0), (5, 1)):
        failing = ScriptedClient(fail_marks=frozenset("一二三四"))
        with pytest.raises(RuntimeError):
            _run(
                tmp_path,
                continuity="parallel",
                client=failing,
                out_name=f"b{retries}{replacements}.srt",
                max_retries_per_window=retries,
                max_replacements_per_window=replacements,
                parallel_window_limit=1,
                monkeypatch=monkeypatch,
            )
        budgets[(retries, replacements)] = len(failing.calls)

    chain_calls = 6
    # Three exhausted chains, and the window in flight stops at the boundary
    # that trips the breaker rather than starting another call.
    assert budgets[(5, 0)] == PARALLEL_BREAKER_FAILURES * chain_calls
    # The whole point: turning the second tier on does not move it.
    assert budgets[(5, 1)] == budgets[(5, 0)]


class SkewedClient(ScriptedClient):
    """Two lanes crawl, so the other two get well ahead of the breaker.

    Systemic failure is usually lockstep -- every window fails at the same
    rate -- and lockstep is the *cheap* schedule: all lanes reach their first
    chain's end together, so the third signal lands before anyone starts a
    replacement. Skew is what makes the second tier visible in the breaker's
    price, and a real batch skews whenever windows differ in size.
    """

    def complete(self, role, messages, **kwargs):
        built = messages("capableC") if callable(messages) else messages
        prompt = built[-1]["content"].split("<asr_result>")[-1]
        if "第三段" in prompt or "第四段" in prompt:
            time.sleep(0.05)
        return super().complete(role, messages, **kwargs)


def test_four_lanes_bound_what_a_doomed_batch_spends(tmp_path, monkeypatch) -> None:
    """The four-lane price, pinned at both schedules (2026-08-19 measurements).

    Single-lane arithmetic is exact but hides the concurrent cost, so these
    are the numbers a real batch pays, and they are what a future change to
    the breaker has to argue with:

    | schedule | repl=0 | repl=1 |
    | -------- | ------ | ------ |
    | lockstep | 24     | 26     |
    | skewed   | 24     | 34-36  |

    The floor is `lanes x chain_calls` = 24, and it is irreducible: four lanes
    each spend a first chain before any evidence exists. The skewed 36 is that
    floor plus two replacement chains, each started while only one window had
    failed -- justified by what was known at the time. What must never come
    back is the window-counting regression, which cost 48 in both schedules.
    """

    chain_calls = 6
    lanes = 4
    budgets = {}
    for label, client_cls in (("lockstep", ScriptedClient), ("skewed", SkewedClient)):
        for replacements in (0, 1):
            failing = client_cls(fail_marks=frozenset("一二三四"))
            with pytest.raises(RuntimeError):
                _run(
                    tmp_path,
                    continuity="parallel",
                    client=failing,
                    out_name=f"{label}{replacements}.srt",
                    max_retries_per_window=5,
                    max_replacements_per_window=replacements,
                    parallel_window_limit=lanes,
                    monkeypatch=monkeypatch,
                )
            budgets[(label, replacements)] = len(failing.calls)

    # The second tier must not touch the price of a batch that is not using it.
    # Upper bounds, not equalities: under contention a lane can be cut off
    # mid-chain, which is the breaker working, not a regression. (Measured 24
    # idle, 23 under a loaded suite -- an equality here is a flaky assertion
    # about the scheduler, not about the breaker.)
    assert budgets[("lockstep", 0)] <= lanes * chain_calls
    assert budgets[("skewed", 0)] <= lanes * chain_calls
    # With it on: the floor, plus at most the two replacement chains that can
    # start before the third signal arrives.
    ceiling = lanes * chain_calls + (PARALLEL_BREAKER_FAILURES - 1) * chain_calls
    assert budgets[("lockstep", 1)] <= ceiling
    assert budgets[("skewed", 1)] <= ceiling
    # And the regression that started all this: counting whole windows let one
    # window's full 12-call budget stand in for one signal, costing 48.
    assert max(budgets.values()) < lanes * chain_calls * 2


def test_a_tripped_breaker_stops_an_in_flight_window_mid_retry(
    tmp_path, monkeypatch
) -> None:
    """The breaker caps calls, and a window is no longer one call.

    Before the two-tier budget, "in-flight windows drain" cost at most the
    remaining retries of one window. With the budget a product, a doomed batch
    could spend (retries+1)x(replacements+1) calls per lane after the breaker
    had already tripped -- here 4, on a window that had failed once already.
    The loop now re-checks between attempts.
    """

    captured: list[threading.Event] = []

    class _RecordingThreading:
        Lock = threading.Lock

        @staticmethod
        def Event() -> threading.Event:
            event = threading.Event()
            captured.append(event)
            return event

    monkeypatch.setattr(correction_parallel, "threading", _RecordingThreading)

    class TripOnFirstCall(ScriptedClient):
        def complete(self, role, messages, **kwargs):
            result = super().complete(role, messages, **kwargs)
            # Stand in for "three other windows have already failed": the
            # breaker trips while this window still has budget left.
            captured[0].set()
            return result

    failing = TripOnFirstCall(fail_marks=frozenset("一二三四"))
    with pytest.raises(RuntimeError):
        _run(
            tmp_path,
            continuity="parallel",
            client=failing,
            max_retries_per_window=1,
            max_replacements_per_window=1,
            parallel_window_limit=1,
            monkeypatch=monkeypatch,
        )
    # One call, not the four the budget allows: the window stopped at the next
    # attempt boundary instead of running its chain and its replacement out.
    assert len(failing.calls) == 1

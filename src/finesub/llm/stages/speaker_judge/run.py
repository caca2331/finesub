"""The speaker judge's step in a correction task: one call per correction window, the sidecar.

Runs after the preparation task and before the correction windows (`correction_translation._run_task`),
on the raw ASR rows and their acoustic labels; the two never read each other's results in route one
(`docs/llm_harness_behavior.md`「说话人裁判」). Each correction window is one judge window (owner
2026-10-09), judged in one call that also names who speaks, from the preparation task's speakers and
people and the knowledge base's streamers. The modality is the switch's (`--diarize judge-text` /
`judge-audio`), not the correction's; the thinking cell is the correction's (owner 2026-10-08).

The correction's planning sized the window, so it normally fits. As a fallback it is halved the way the
correction halves a window that does not fit (`docs/llm_harness_behavior.md`「重试与拼接」): when a reply
reaches the output ceiling (API usage), when the retries ran out and the last reply left `<judge>` open
(an agent's usage cannot tell), or, before any call, when the prompt is past the judge's route's input envelope.
One level only: a half that still does not fit fails, and its rows keep their acoustic labels.

Resume is the sidecar (`speech.diarization.judgments`): once cut, its window list stands (owner
2026-10-09) -- a later re-plan of the correction neither re-cuts nor re-judges -- and is re-cut from
the current correction plan only when there is no usable sidecar or the rows themselves changed.
Each unit (a window, or a half of one) is written the moment its answer is in, so an interrupted run loses at
most the calls in flight. A unit is judged again only when it failed, is pending, or no longer matches
its rows: a verdict an older prompt gave is reused (owner 2026-10-09), as resume carries on a task's own
results.

A reply that does not read is repaired as the correction's is (`docs/llm_harness_behavior.md`「重试与拼接」):
the next call carries the reply and every reason it was refused. When its `<judge>` block has something to keep
it asks only for the rest as a `<patch>` (`patch.py`), laid over the rows that answered and read whole again;
when the block closed but nothing in it can stay, it asks for the whole answer again in the judge's format, as
the correction does when a patch cannot help; a reply with no closed block -- empty, cut off, or no `<judge>` at
all -- gets the same prompt again, blind. An agent tool session does the same at `submit` (`agent_validators`,
`speaker-judge`). Repair rounds are calls like any other: they share the unit's `MAX_ATTEMPTS`, the ones a tool
session ran inside one call included.

Quota-type errors stop the whole judge rather than retrying window by window: the judge runs first,
and the quota it would burn is the correction's.

Three passes run this step as pass 1 (`shape`): the judge also answers the shape (`prompt.py`). A unit judged
without the shape is owed again; the acts and the ones set aside go into its sidecar entry. A judge that
hears the audio may also split a row where the speaker changes: the cut snaps to a word boundary here, and
the entry keeps it (`splits`) beside the verdicts, which stay on the row as it was. It may also add what it
heard and the ASR missed: the rest of a row's sentence (`fills`, checked to keep the row's own text whole) and
a whole line after a row (`inserts`, on the source timeline); the reader places both in time. Such a judge's
move of a short row onto a track outside the row's cell is not taken (`SHORT_ROW_SECONDS`): the row keeps its
label and the entry lists the move (`gated`).
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ....speech.diarization import judgments as sidecar
from ....subtitles.stable_rows import read_rows
from ...chunking import SubtitleWindow
from ...client import extract_token_distribution, is_likely_output_limited
from ...correction_patch import PATCH_TAG, PatchMerge
from ...degeneration import pick_after_retry, thinking_of, zero_thinking_record
from ...routing.model_router import FailureKind, classify_failure
from ...exchange_log import ExchangeLogger
from ...exchange_metadata import llm_exchange_metadata
from ...knowledge.base import append_task_artifact
from ...media_upload import window_media_ref, with_speech_spans
from ...output_protocol import strip_leaked_thought
from ...routing.config import LLMRole
from ...routing.profiles import TranslationProfile
from ...run_context import bind_llm_worker, llm_worker_context
from ...token_books import append_call_error, call_trace
from .answers import (
    AnswerError,
    Insert,
    Split,
    Verdict,
    answer_fields,
    cut_off,
    fill_ends,
    parse_acts,
    parse_fills,
    parse_inserts,
    parse_judge,
    parse_splits,
    parse_tracks,
    results,
    spoken_seconds,
    split_at,
    survey,
)
from .prompt import (
    SAMPLE_GAP,
    Candidate,
    Carried,
    audio_layout,
    cell_tracks,
    judge_prompt,
    profiles,
    rows_of,
)
from .patch import RepairBase, apply_patch, patch_instruction, repair_base, whole_instruction
from .windows import JudgeWindow, halves, split_windows

#: The judge's own prompt version, apart from the correction's `PROMPT_VERSION`, recorded with each
#: verdict; a bump re-judges nothing (owner 2026-10-09: a verdict that still matches its rows is reused).
PROMPT_VERSION = "speaker-judge-v11"
#: Calls per unit before it is recorded as failed (agy's malformed replies ran about one in five); repair
#: rounds count, those a tool session ran inside one call included.
MAX_ATTEMPTS = 4
#: The agent tool session's `submit` judges a reply with the harness's own reader (`agent_validators`).
VALIDATOR_ID = "speaker-judge"
#: Empty replies in a row that read as a frozen quota rather than bad luck: from at least two units, and
#: each costing no output (a reply that spent tokens and came back empty is the model's, not the quota's).
EMPTY_REPLIES_TO_STOP = 3
#: Development switch: `0` hides the confidence marks from the judge's prompt (round 9's input).
MARKS_SWITCH = "FINESUB_SPEAKER_JUDGE_MARKS"
#: The share of the route's input envelope a unit's prompt may take before it is cut.
INPUT_SHARE = 0.9
#: How far past the file's last row an insert after it may start (seconds; the prompt says the same).
INSERT_REACH = 5.0
#: F1b (owner 2026-11-02): a judge that hears the audio may not move a row shorter than this onto a track its cell
#: does not show. On such rows the voice is too short to tell who speaks and the text rarely says it, and the moves
#: the audio judge made there were mostly wrong, across sexes included (the D1 rerun; in English meetings the short
#: rows look the same). Duration only, so it holds in every language.
SHORT_ROW_SECONDS = 1.5
#: What the run was told besides the rows, in the task artifact dir: `tools/session_replay`'s
#: `speaker-judge` session rebuilds a unit's prompt from it (docs/session_replay.md).
INPUT_FILENAME = "speaker-judge-input.json"


def marks_shown() -> bool:
    return os.environ.get(MARKS_SWITCH, "1").strip() != "0"


def judge_group(media: str) -> str:
    """The route group the judge's call takes: its own modality's, whatever the correction's."""
    return "correction-mm" if media == "audio" else "correction-text"


class JudgeStopped(RuntimeError):
    """The provider said "not now" (quota, rate limit, a run of empty replies): stop judging. ``kept`` is the
    unit's entry when it already held an answer -- one waiting on its retry for missing thinking."""

    def __init__(self, message: str, kept: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.kept = kept


class Cut(Exception):
    """This window does not fit: judge it in two halves."""


@dataclass(frozen=True)
class _Repair:
    """What the next call sends back: the refused reply (``text``) and why it was refused. With a ``base`` it asks
    for a patch over it and -- when the last patch itself could not be laid, which leaves the base as it was --
    says why not (first); without one it asks for the whole answer again."""

    text: str
    errors: tuple[str, ...]
    base: RepairBase | None = None
    patch_errors: tuple[str, ...] = ()

    @property
    def all_errors(self) -> tuple[str, ...]:
        return (*self.patch_errors, *self.errors)

    def instruction(self) -> str:
        if self.base is None:
            return whole_instruction(self.all_errors)
        return patch_instruction(self.base, self.all_errors)


def _repair_of(text: str, errors: Sequence[str], ids: Sequence[str], tracks: Sequence[int]) -> _Repair | None:
    """How the next call repairs a refused reply: a patch when its `<judge>` block has something to keep, the
    whole answer again with the errors when the block closed but nothing in it can stay, or None -- the same
    prompt again, blind -- when there is no closed block (an empty reply, one cut off, or no `<judge>` at all)."""
    base = repair_base(text, ids, tracks)
    if base is not None:
        return _Repair(text, tuple(errors), base)
    return _Repair(text, tuple(errors)) if survey(text, ids, tracks).found else None


@dataclass
class JudgeReport:
    windows: int = 0
    judged: int = 0
    reused: int = 0
    failed: int = 0
    pending: int = 0
    #: windows this run halved because an answer or a prompt did not fit
    cut: int = 0
    skipped: str = ""
    stopped: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeInputs:
    """What the judge is told besides the rows: the title, the candidates (empty when neither the
    knowledge base nor the preparation task names anyone), and `summary`, the preparation task's summary
    for a unit's rows by their ids (`summary_of`)."""

    title: str
    pool: tuple[Candidate, ...] = ()
    summary: Callable[[Sequence[str]], str] = lambda ids: ""


def vocal_audio_for(stable_json: str | Path) -> Path | None:
    """The separated vocals beside a stable, by the pipeline's own rule: `-vocal.ogg`, else `.flac`."""
    stable = Path(stable_json)
    base = stable.name.removesuffix(".json").removesuffix("-stable")
    for suffix in (".ogg", ".flac"):
        path = stable.with_name(f"{base}-vocal{suffix}")
        if path.exists():
            return path
    return None


def run_speaker_judge(
    *,
    stable_json: str | Path,
    correction_windows: Sequence[SubtitleWindow],
    profile: TranslationProfile,
    client: Any,
    inputs: JudgeInputs,
    media: str,
    audio: str | Path | None,
    task_artifact_dir: str | Path,
    task_id: str,
    parallel: int = 1,
    warn: Callable[[str], None] = lambda message: None,
    limits: Any = None,
    counter: Any = None,
    shape: str | None = None,
) -> JudgeReport:
    """Judge every unit the sidecar does not already hold; returns what happened.

    `media` is the judge's modality (`text` / `audio`); `audio` the recording it hears with `audio`
    (the separated vocals). `limits` is the judge's route group's planning envelope and `counter` a
    token counter: with both, a window whose prompt would not fit is halved before it is sent. `shape`
    is pass 1's (`judgments.SHAPE_TRACKED`), None for the judge alone."""
    report = JudgeReport()
    stable_json = Path(stable_json)
    segments = read_rows(stable_json)
    rows = rows_of(segments)
    if not sidecar.judgeable(segments):
        multiple = any(row.speaker.block != 0 for row in rows)
        report.skipped = "speaker labels in more than one block" if multiple else "no speaker labels"
        if multiple:
            warn("说话人裁判跳过：声学标签分了多个 block，裁判只认一个 block 的轨号")
        return report
    ids = [row.id for row in rows]
    tracks = sidecar.profile_tracks([row.speaker for row in rows])
    path = sidecar.sidecar_path(stable_json)
    windows, entries = _window_list(path, segments, ids, correction_windows)
    report.windows = len(windows)
    use_audio = media == "audio" and audio is not None
    media = "audio" if use_audio else "text"
    _dump_inputs(Path(task_artifact_dir), stable_json=stable_json, inputs=inputs,
                 summaries={window.id: inputs.summary(ids[window.low:window.high]) for window in windows},
                 media=media, audio=audio if use_audio else None, shape=shape, profile=profile, task_id=task_id)
    state = _Run(segments=segments, ids=ids, tracks=tracks, entries=entries, windows=windows, shape=shape)
    todo = [index for index in range(len(windows)) if state.owed(index)]
    report.reused = len(windows) - len(todo)
    other = sum(1 for unit in state.units() if state.stands(unit) and unit.entry.get("media") not in (None, media))
    if other:
        warn(f"{other} 个说话人裁判窗是用另一种模态判的，照旧沿用（续跑延续任务当时的选择）；要按这次的档位重判，删掉 {path.name}")
    session = _Session(client, profile, Path(task_artifact_dir), task_id, judge_group(media))
    serial = profile.continuity == "serial"

    def write() -> None:
        payload = {"version": sidecar.SIDECAR_VERSION, "route": "separate",
                   "rows_hash": sidecar.rows_hash(segments), "windows": entries}
        temporary = path.with_name(f".{path.name}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(path)

    def one(index: int) -> None:
        if session.stopped.is_set():
            return
        try:
            _judge_whole_window(state, index, session, rows, inputs, audio=Path(audio) if use_audio else None,
                                serial=serial, limits=limits, counter=counter, write=write, warn=warn)
        except JudgeStopped as stop:
            with state.lock:  # the reason first: a window that only saw the flag must not overwrite it
                report.stopped = report.stopped or str(stop)
            session.stopped.set()

    with state.lock:
        for index in todo:
            state.mark_pending(index)
        write()
    # The run's reporter, generation pins and agent sessions follow the workers, as in the correction's pools.
    if serial:
        # In order, so each unit can be handed the tracks the one before it opened or used.
        for index in todo:
            one(index)
    else:
        worker_context = llm_worker_context()
        try:
            with ThreadPoolExecutor(max(1, parallel), thread_name_prefix="llm-speaker-judge",
                                    initializer=bind_llm_worker, initargs=(worker_context,)) as pool:
                list(pool.map(one, todo))
        finally:
            worker_context.release_lanes()
    report.judged = sum(status == "success" for status in state.outcome.values())
    report.failed = sum(status == "failed" for status in state.outcome.values())
    report.pending = sum(unit.entry.get("status") == "pending" for unit in state.units())
    report.cut = len(state.cut)
    report.usage = session.usage
    return report


def _dump_inputs(artifact_dir: Path, *, stable_json: Path, inputs: JudgeInputs, summaries: Mapping[str, str],
                 media: str, audio: Any, shape: str | None, profile: TranslationProfile, task_id: str) -> None:
    """`summaries`: each window's, by window id (a chunked preparation task's differ by window)."""
    payload = {"version": 2, "stable_json": str(stable_json), "title": inputs.title, "summaries": dict(summaries),
               "pool": [{"name": candidate.name, "others": list(candidate.others)} for candidate in inputs.pool],
               "media": media, "audio": str(audio) if audio else "", "shape": shape, "marks": marks_shown(),
               "profile_id": profile.profile_id, "task_id": task_id}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / INPUT_FILENAME).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


@dataclass
class _Run:
    """The sidecar being written and what this run did, shared by the window workers."""

    segments: list
    ids: list[str]
    tracks: list[int]
    entries: list[dict]
    windows: list[JudgeWindow]
    shape: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: this run's calls, per unit id (a reused verdict is not "judged")
    outcome: dict[str, str] = field(default_factory=dict)
    cut: set[int] = field(default_factory=set)

    def units(self) -> list[sidecar.Unit]:
        return sidecar.units(self.entries, self.ids) or []

    def window_units(self, index: int) -> list[sidecar.Unit]:
        window = self.windows[index]
        return [unit for unit in self.units() if window.low <= unit.low < window.high]

    def stands(self, unit: sidecar.Unit) -> bool:
        return sidecar.holds(self.segments, unit, self.tracks, shape=self.shape)

    def owed(self, index: int) -> bool:
        return any(not self.stands(unit) for unit in self.window_units(index))

    def mark_pending(self, index: int) -> None:
        for unit in self.window_units(index):
            if not self.stands(unit):
                _record(self.entries[index], unit.id, {"status": "pending"})

    def key(self, low: int, high: int) -> str:
        return sidecar.window_key(self.segments[low:high], self.ids[low:high], self.tracks)


def _record(window: dict, unit_id: str, fields: Mapping[str, Any]) -> None:
    """Replace what a unit -- the window itself, or one of its parts -- says, keeping where it is."""
    target = window if str(window.get("id")) == unit_id else next(
        part for part in window.get("parts") or () if str(part.get("id")) == unit_id)
    kept = {key: target[key] for key in ("id", "chunk", "first", "last", "parts") if key in target}
    target.clear()
    target.update({**kept, **fields})


def _halve(state: _Run, index: int, rows: Sequence[Any]) -> None:
    """Turn a window's entry into its two halves, pending (the window list itself never grows: parallel
    workers write by window index)."""
    window, judged = state.entries[index], state.windows[index]
    name = str(window.get("id"))
    kept = {key: window[key] for key in ("id", "chunk", "first", "last") if key in window}
    window.clear()
    window.update({**kept, "parts": [
        {"id": sidecar.part_id(name, half), "first": state.ids[low], "last": state.ids[high - 1], "status": "pending"}
        for half, (low, high) in enumerate(halves(rows, judged.low, judged.high))]})


def _judge_whole_window(state: _Run, index: int, session: "_Session", rows: list, inputs: JudgeInputs, *,
                        audio: Path | None, serial: bool, limits: Any, counter: Any,
                        write: Callable[[], None], warn: Callable[[str], None]) -> None:
    """Judge a window's units in order until each holds, failed, or the judge stopped; halve a window
    that does not fit and go on with its halves."""
    while True:
        with state.lock:
            todo = [unit for unit in state.window_units(index)
                    if not state.stands(unit) and unit.id not in state.outcome]
            if not todo:
                return
            unit = todo[0]
            found = state.units()
            carried = _carried(state.segments, found, unit) if serial else []
        part = JudgeWindow(unit.id, str(state.entries[index].get("chunk")), unit.low, unit.high)
        whole = unit.id == str(state.entries[index].get("id"))
        try:
            entry = _judge_unit(session, rows, part, inputs, tracks=state.tracks, carried=carried, audio=audio,
                                limits=limits, counter=counter, may_halve=whole and part.size >= 2, shape=state.shape,
                                segments=state.segments)
        except Cut:
            with state.lock:
                _halve(state, index, rows)
                state.cut.add(index)
                write()
            continue
        except JudgeStopped as stop:
            if stop.kept is not None:
                with state.lock:
                    state.outcome[unit.id] = stop.kept["status"]
                    _record(state.entries[index], unit.id, {**stop.kept, "key": state.key(unit.low, unit.high)})
                    write()
            raise
        except Exception as error:  # noqa: BLE001 - the judge is optional: a broken unit falls back, the run goes on
            warn(f"说话人裁判窗 {unit.id} 出错（{type(error).__name__}: {error}），这些行保留原有的标签")
            entry = {"status": "failed", "prompt_version": PROMPT_VERSION}
        with state.lock:
            state.outcome[unit.id] = entry["status"]
            _record(state.entries[index], unit.id, {**entry, "key": state.key(unit.low, unit.high)})
            write()


def _carried(segments: list, found: Sequence[sidecar.Unit], unit: sidecar.Unit) -> list[Carried]:
    """The new tracks the unit before this one opened or carried on, as the reader numbers them
    (owner 2026-10-09: units judged in order hand them on, and may merge into them); none when that
    unit does not hold a verdict. Their lines come from before this unit only: a re-judge must not
    be shown its own earlier answers as evidence."""
    before = next((other for other in found if other.high == unit.low), None)
    if before is None or not sidecar.holds(segments, before):
        return []
    resolved = sidecar.resolve(segments, found)
    verdicts = {other.id: sidecar.verdict_of(other.entry) for other in found}
    used = [key for key in (_track_of(result, before.id, verdicts) for result in _answers(before.entry))
            if key is not None]
    position = {row: at for at, row in enumerate(sidecar.segment_ids(segments))}
    earlier = [other for other in found if other.high <= unit.low]
    out = []
    for key in dict.fromkeys(used):
        number = resolved.opened.get(key)
        if number is None:
            continue
        lines = tuple(sorted(position[row] for row, track in resolved.judgments.moved.items()
                             if track == number and position[row] < unit.low))
        out.append(Carried(number, key[0], key[1], _name(earlier, key, number, verdicts), lines, verdicts[key[0]]))
    return out


def _answers(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Everything in a unit's entry that names a track: its results, its splits' second halves, its inserts."""
    return [*(entry.get("results") or ()), *(entry.get("splits") or ()), *(entry.get("inserts") or ())]


def _track_of(result: Mapping[str, Any], unit_id: str, verdicts: Mapping[str, str]) -> tuple[str, str] | None:
    """The new track a result puts its row on -- `(opener id, label)` -- when the reader would apply it: a
    carry whose opener has since given another verdict is not one (`judgments.resolve`)."""
    if result.get("label") is not None:
        return unit_id, str(result["label"])
    carried = result.get("carried")
    if isinstance(carried, Mapping) and verdicts.get(str(carried.get("window"))) == carried.get("of"):
        return str(carried.get("window")), str(carried.get("label"))
    return None


def _name(units: Sequence[sidecar.Unit], key: tuple[str, str], number: int, verdicts: Mapping[str, str]) -> str:
    """Who the judge last said a new track is: `@label` in the unit that opened it, `s<number>` in the
    units after it that still carry it (only those before the one being judged)."""
    name = ""
    for unit in units:
        names = unit.entry.get("names") or {}
        if unit.id == key[0]:
            name = names.get(f"@{key[1]}", name)
        elif f"s{number}" in names and any(_track_of(result, unit.id, verdicts) == key
                                           for result in _answers(unit.entry)):
            name = names[f"s{number}"]
    return name


def _window_list(path: Path, segments: list, ids: list[str],
                 correction_windows: Sequence[SubtitleWindow]) -> tuple[list[JudgeWindow], list[dict]]:
    """The sidecar's windows when it still describes these rows; else a fresh cut, all pending."""
    try:
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        stored = None
    if (isinstance(stored, dict) and stored.get("version") == sidecar.SIDECAR_VERSION
            and stored.get("rows_hash") == sidecar.rows_hash(segments)):
        entries = list(stored.get("windows") or [])
        spans = sidecar.tiling(entries, ids)
        if spans is not None and sidecar.units(entries, ids) is not None:
            return ([JudgeWindow(str(e.get("id")), str(e.get("chunk")), low, high) for e, (low, high) in zip(entries, spans)],
                    entries)
    windows = split_windows(ids, correction_windows)
    return windows, [{"id": w.id, "chunk": w.chunk, "first": ids[w.low], "last": ids[w.high - 1],
                      "status": "pending"} for w in windows]


class _Session:
    """The calls of one judge run: routing, logging, usage, the stop signal."""

    def __init__(self, client: Any, profile: TranslationProfile, artifact_dir: Path, task_id: str, group: str) -> None:
        self.client, self.profile, self.artifact_dir, self.task_id = client, profile, artifact_dir, task_id
        self.group = group
        self.logger = ExchangeLogger.for_task_artifact_dir(artifact_dir)
        self.stopped = threading.Event()
        self.usage: dict[str, int] = {}
        #: per worker thread: the unit being judged -- its model and usage, for its sidecar entry
        self.local = threading.local()
        self._empty: list[str] = []
        self._lock = threading.Lock()

    def start_unit(self) -> None:
        self.local.model, self.local.usage = "", {}

    def call(self, name: str, system: str, user: str, *, unit: str, file_ref: Any = None, fresh: bool = False,
             repair: _Repair | None = None, validator: Mapping[str, Any] | None = None, repairs: int = 0) -> Any:
        """One call; ``fresh`` asks for a new session (a `resume` lane's conversation is retired there). A
        ``repair`` goes with the reply it repairs and why; the transport says how (`client.complete`), and an
        agent resumes the conversation that wrote it (one chain per unit). ``validator`` is what an agent tool
        session's `submit` checks, with ``repairs`` rounds of its own."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            result = self.client.complete(
                LLMRole.AUDIO_MULTIMODAL, messages, file_ref=file_ref, task_group=self.group,
                difficulty=self.profile.difficulty, repair_session_key=f"speaker_judge-{unit}",
                previous_output=repair.text if repair else "",
                validation_errors=repair.all_errors if repair else (),
                repair_instruction=repair.instruction() if repair else "",
                validator_spec=validator, max_repair_attempts=repairs,
                **({"fresh_session": True} if fresh else {}))
        except Exception as error:  # noqa: BLE001 - classified below; the unit decides what to do
            append_call_error(self.artifact_dir, "speaker_judge_call_error", self.task_id, name, error)
            if classify_failure(error) in (FailureKind.QUOTA, FailureKind.RATE_LIMIT):  # the router's rule: 429, 402
                raise JudgeStopped(f"{type(error).__name__}: {error}") from error
            raise
        tokens = extract_token_distribution(result.raw_response)
        self.local.model = result.model
        for key, value in tokens.items():
            self.local.usage[key] = self.local.usage.get(key, 0) + int(value or 0)
        with self._lock:
            for key, value in tokens.items():
                self.usage[key] = self.usage.get(key, 0) + int(value or 0)
            spent = int(tokens.get("output_tokens") or 0) + int(tokens.get("thinking_tokens") or 0)
            if (result.content or "").strip() or spent:
                self._empty = []
            else:
                self._empty.append(name.rsplit("-attempt", 1)[0])
            empty_run, units = len(self._empty), len(set(self._empty))
        rounds = int(getattr(result, "agent_repair_rounds", 0) or 0)
        append_task_artifact(self.artifact_dir, kind="speaker_judge_response", task_id=self.task_id,
                             payload={"call": name, "model": result.model, "usage": tokens, **call_trace(result),
                                      **({"repair": "patch" if repair.base else "whole"} if repair else {}),
                                      **({"agent_repair_rounds": rounds} if rounds else {})})
        if self.logger is not None:
            self.logger.log(name, messages=messages, response_text=result.content,
                            metadata=llm_exchange_metadata(result, session=name, prompt_version=PROMPT_VERSION))
        if empty_run >= EMPTY_REPLIES_TO_STOP and units >= 2:
            raise JudgeStopped(f"{empty_run} empty replies in a row")
        return result

    def upload(self, path: Path, *, speech: Sequence[tuple[float, float]] = ()) -> Any:
        """The clip's reference; `speech`: where its rows speak (clip seconds), for agy's cuts."""
        return with_speech_spans(
            window_media_ref(path, execution_settings=getattr(self.client, "execution_settings", None),
                             routes=getattr(getattr(self.client, "router", None), "routes", None)),
            speech)


def _too_long(system: str, user: str, seconds: float, limits: Any, counter: Any) -> bool:
    """Whether a unit's prompt is past the route's input envelope (never, with nothing to check against)."""
    if limits is None or counter is None:
        return False
    tokens = counter.count_text(system) + counter.count_text(user)
    if seconds:
        tokens += math.ceil(seconds * limits.audio_tokens_per_second)
    return tokens > INPUT_SHARE * limits.prompt_input_limit


def _judge_unit(session: _Session, rows: list, unit: JudgeWindow, inputs: JudgeInputs, *,
                tracks: Sequence[int], carried: Sequence[Carried], audio: Path | None,
                limits: Any, counter: Any, may_halve: bool, shape: str | None = None,
                segments: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """One call (and its retries) for one unit. Raises `Cut` when a whole window does not fit; a half
    that does not fit comes back failed (one level only, as the correction's split). `segments` are the
    rows' own (their words: where a split is cut)."""
    session.start_unit()
    entry: dict[str, Any] = {"prompt_version": PROMPT_VERSION, "media": "audio" if audio else "text"}
    profile = profiles(rows, unit, carried)
    layout = audio_layout(rows, unit, profile.samples) if audio is not None else None
    summary = inputs.summary([row.id for row in rows[unit.low:unit.high]])
    system, user, ids = judge_prompt(rows, unit, title=inputs.title, summary=summary, cast=inputs.pool,
                                     marks=marks_shown(), profile=profile, audio=layout, shape=shape is not None,
                                     correct_only=session.profile.correct_only)
    if _too_long(system, user, layout.seconds if layout else 0.0, limits, counter):
        if may_halve:
            raise Cut()
        return {**entry, "status": "failed"}
    file_ref = None
    if layout is not None:
        from ....media.ffmpeg import extract_spliced_audio

        clip = extract_spliced_audio(audio, layout.pieces, SAMPLE_GAP,
                                     session.artifact_dir / "clips" / f"speaker-judge-{unit.id}.aac")
        file_ref = session.upload(clip, speech=layout.speech(rows))
    shown = {track for index in unit.shown(rows) for track in cell_tracks(rows[index], marks=marks_shown())}
    known = [*tracks, *shown, *(track.track for track in carried)]
    carry = {track.track: (track.window, track.label, track.of) for track in carried}

    def read(text: str) -> dict[str, Any]:
        """The reply's answers; raises `AnswerError` for one that does not read."""
        heard = shape is not None and layout is not None
        return {"verdicts": parse_judge(text, ids, known),
                "acts": parse_acts(text, ids) if shape is not None else None,
                "splits": parse_splits(text, ids, known) if heard else ({}, []),
                "fills": parse_fills(text, ids) if heard else {},
                "inserts": parse_inserts(text, ids, known) if heard else ({}, []),
                "names": parse_tracks(text)}

    def settle(answer: dict[str, Any], thinking: Mapping[str, Any] | None = None) -> dict[str, Any]:
        verdicts, acts = answer["verdicts"], answer["acts"]
        entry.update(model=answer["model"], usage=dict(session.local.usage), status="success")
        if answer["repairs"]:
            entry["repairs"] = answer["repairs"]
        if thinking is not None:
            entry["thinking"] = dict(thinking)
        if answer["stripped"]:
            entry["thought_stripped"] = answer["stripped"]
        taken, gated = _short_row_gate(rows, unit, verdicts, carry) if layout is not None else (verdicts, [])
        entry.update(names=answer["names"], results=results(rows, unit, taken, carry))
        if gated:
            entry["gated"] = gated
        if acts is not None:  # a split is checked against the verdict that stands (a gated row keeps its track)
            placed, refused = _placed_splits(segments, rows, unit, taken, acts.acts, answer["splits"][0], carry,
                                             {move["row"]: move["track"] for move in gated})
            ignored = [*acts.ignored, *answer["splits"][1], *refused]
            if layout is not None:
                filled, unfilled = _placed_fills(rows, acts.acts, answer["fills"], {split["row"] for split in placed})
                entry.update(splits=placed, fills=filled,
                             inserts=_placed_inserts(rows, answer["inserts"][0], layout, carry))
                ignored += [*unfilled, *answer["inserts"][1]]
            entry.update(acts=acts.acts, ignored=ignored)
        return entry

    answer_tracks = sorted(set(known))
    validator = {"id": VALIDATOR_ID, "params": {"ids": list(ids), "tracks": answer_tracks}}
    # An answer that read but reported no thinking although it was asked to waits here for one more call in a
    # fresh session, past the attempt budget; the unit keeps one of the two (`degeneration`, owner 2026-10-17).
    held: dict[str, Any] | None = None
    repair: _Repair | None = None
    left_open = False
    attempt = repairs = 0
    while attempt < MAX_ATTEMPTS or held is not None:
        name = f"speaker_judge-{unit.id}-attempt{attempt}"
        attempt += 1
        sent = None if held is not None else repair  # the retry for missing thinking is a whole new answer
        try:
            if session.stopped.is_set():
                raise JudgeStopped("another window stopped the judge")
            result = session.call(name, system, user, unit=unit.id, file_ref=file_ref, fresh=held is not None,
                                  repair=sent, validator=validator, repairs=max(0, MAX_ATTEMPTS - attempt))
        except JudgeStopped as stop:
            if held is not None:  # the answer it had stands; the stop goes on up
                raise JudgeStopped(str(stop), kept=settle(held, zero_thinking_record(
                    held["thinking"], None, "original", error=f"stopped: {stop}"))) from stop
            raise
        except Exception as error:  # noqa: BLE001 - a transient error: try again (with the same repair)
            if held is not None:
                return settle(held, zero_thinking_record(held["thinking"], None, "original",
                                                         error=f"{type(error).__name__}: {error}"))
            continue
        rounds = int(getattr(result, "agent_repair_rounds", 0) or 0)  # run inside the call: on the budget
        attempt += rounds
        repairs += rounds + (sent is not None)
        reply = result.content or ""
        text, merge, stripped = _read_reply(reply, sent)
        try:
            if merge is not None and merge.mode == "invalid":
                raise AnswerError(*merge.errors)
            answer = {**read(text), "model": result.model, "stripped": stripped,
                      "thinking": thinking_of(result), "repairs": repairs}
        except AnswerError as refused:
            if held is not None:
                return settle(held, zero_thinking_record(held["thinking"], thinking_of(result), "original"))
            # The output ceiling, by the API's usage (an agent's usage is the whole session's: never compared)
            ceiling = getattr(result, "requested_output_tokens", 0) or (limits.output_limit if limits else 0)
            if ceiling and is_likely_output_limited(result.raw_response, max_tokens=ceiling):
                if may_halve:
                    raise Cut()
                break
            left_open = cut_off(text)  # the last reply's, as the correction reads its final attempt
            if merge is not None and merge.mode == "invalid":
                repair = replace(sent, patch_errors=refused.errors)  # the same base, the patch's own errors first
            else:
                repair = _repair_of(text, refused.errors, ids, answer_tracks)
            continue
        if held is not None:
            chosen = pick_after_retry(0, answer["thinking"], 0)
            return settle(answer if chosen == "retry" else held,
                          zero_thinking_record(held["thinking"], answer["thinking"], chosen))
        # A patch is a few rows: not thinking over it is normal (owner 2026-10-24, as the correction's).
        if answer["thinking"].skipped and not (merge is not None and merge.mode == "patch"):
            held = answer
            continue
        return settle(answer)
    if left_open and may_halve:
        raise Cut()  # the fallback when usage cannot tell, as the correction's open `<translated>`
    entry.update(model=session.local.model, usage=dict(session.local.usage), **({"repairs": repairs} if repairs else {}))
    return {**entry, "status": "failed"}


def _read_reply(reply: str, repair: _Repair | None) -> tuple[str, PatchMerge | None, int]:
    """The text to read a reply as, how its patch went, and how much leaked reasoning was cut from the reply:
    on a patch round a `<patch>` laid over the base gives the whole; otherwise -- a whole `<judge>`, or a patch
    that could not be laid -- the reply itself. Reasoning shipped as the reply is read by the answer it ends
    with: `<patch>` on a patch round."""
    merge = None
    if repair is not None and repair.base is not None:
        # Cut at `<patch>`, a leaked reply loses a `<tracks>` written before it: names only, which read
        # fail-soft (a row on a track without one is still an answer), on a reply that is already rare.
        patch = strip_leaked_thought(reply, (PATCH_TAG,)) or reply
        merge = apply_patch(repair.base, patch)
        if merge.mode == "patch":
            return merge.text, merge, len(reply) - len(patch)
    text = strip_leaked_thought(reply, ("judge",)) or reply
    return text, merge, len(reply) - len(text)


def _short_row_gate(rows: Sequence[Any], unit: JudgeWindow, verdicts: Mapping[str, Verdict],
                    carried: Mapping[int, tuple[str, str, str]]) -> tuple[dict[str, Verdict], list[dict[str, Any]]]:
    """F1b: the verdicts with every move of a row shorter than SHORT_ROW_SECONDS onto a known track its cell does
    not show turned into a keep, and those moves as the entry lists them (`gated`: row, track, why). A new track
    and a carried one are not gated (no cell shows them), nor the row's own track (a keep, as `results` reads it)."""
    taken, gated = dict(verdicts), []
    for row in rows[unit.low:unit.high]:
        key = f"q{row.id}"
        verdict = verdicts[key]
        origin = sidecar.acoustic_main(row.speaker)
        current = origin["filled"] if isinstance(origin, dict) else origin
        if (verdict.kind == "track" and verdict.track not in carried and verdict.track != current
                and row.seconds < SHORT_ROW_SECONDS and verdict.track not in cell_tracks(row, marks=marks_shown())):
            taken[key] = Verdict("keep", verdict.why)
            gated.append({"row": row.id, "track": verdict.track, "why": verdict.why})
    return taken, gated


def _placed_splits(segments: Sequence[Mapping[str, Any]], rows: Sequence[Any], unit: JudgeWindow,
                   verdicts: Mapping[str, Verdict], acts: Mapping[str, Mapping[str, str]],
                   splits: Mapping[str, Split], carried: Mapping[int, tuple[str, str, str]],
                   gated: Mapping[str, int] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """The unit's splits as the sidecar keeps them -- row, prefix, `at` (the second half's first word), the
    row's acoustic main track (`from`, as a result has) and who speaks after the cut -- and the ones set
    aside: on a dropped row, a second half on the first half's speaker (split is for a change of
    speaker), a second half on the track the short-row gate refused the row (`gated`: row id -> track; the
    same mishearing, owner 2026-11-02), or a prefix that does not open the row on a word boundary with words
    left after it."""
    placed, refused = [], []
    position = {row.id: index for index, row in enumerate(rows)}
    for key, split in sorted(splits.items(), key=lambda item: position[item[0]]):
        index, row = position[key], rows[position[key]]
        if not unit.low <= index < unit.high:
            continue
        why = ""
        if acts.get(key, {}).get("act") == sidecar.ACT_DROP:
            why = "the row is dropped"
        elif _speaker_key(split.second, row) == _speaker_key(verdicts[f"q{key}"], row):
            why = "second half is the first half's speaker"
        elif split.second.kind == "track" and (gated or {}).get(key) == split.second.track:
            why = "second half is the track the short-row gate refused"
        else:
            at, why = split_at(segments[index], split.prefix)
        if why:
            refused.append({"row": key, "act": "split", "why": why})
            continue
        placed.append({"row": key, "prefix": split.prefix, "at": at, "from": sidecar.acoustic_main(row.speaker),
                       **answer_fields(split.second, carried)})
    return placed, refused


def _placed_fills(rows: Sequence[Any], acts: Mapping[str, Mapping[str, str]], fills: Mapping[str, str],
                  split: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """The unit's fills as the sidecar keeps them -- row, the text as heard whole, and how many characters it
    adds at the row's head and tail (the reader grows the row's time by them) -- and the ones set aside: on a
    dropped row, on a row split here (a row takes one of the two, split first), or a text that does not keep
    the row's own whole or adds nothing to it."""
    placed, refused = [], []
    position = {row.id: index for index, row in enumerate(rows)}
    for key, text in sorted(fills.items(), key=lambda item: position[item[0]]):
        ends, why = None, ""
        if acts.get(key, {}).get("act") == sidecar.ACT_DROP:
            why = "the row is dropped"
        elif key in split:
            why = "the row is split"
        else:
            ends, why = fill_ends(rows[position[key]].text, text)
        if ends is None:
            refused.append({"row": key, "act": "fill", "why": why})
            continue
        placed.append({"row": key, "text": text, "head": ends[0], "tail": ends[1]})
    return placed, refused


def _placed_inserts(rows: Sequence[Any], inserts: Mapping[str, Insert], layout: Any,
                    carried: Mapping[int, tuple[str, str, str]]) -> list[dict[str, Any]]:
    """The unit's inserts as the sidecar keeps them: the row each follows, its text, who said it (named as a
    result names a track) and the seconds the judge heard it at, on the source timeline and held between the
    row's start and the next row's end (`INSERT_REACH` past the row for the file's last). An end not given,
    not after the start, or clamped onto it comes from the text's length. Where it shows is the reader's
    (`judgments.resolve`): first every unit's fills grow their rows."""
    out = []
    position = {row.id: index for index, row in enumerate(rows)}
    for key, insert in sorted(inserts.items(), key=lambda item: position[item[0]]):
        index = position[key]
        low = rows[index].start
        high = rows[index + 1].end if index + 1 < len(rows) else rows[index].end + INSERT_REACH
        start = min(max(insert.start - layout.offset + layout.origin, low), high)
        end = min(start if insert.end is None else insert.end - layout.offset + layout.origin, high)
        if end <= start:
            end = start + spoken_seconds(insert.text)
        out.append({"row": key, "start": round(start, 3), "end": round(end, 3), "text": insert.text,
                    **answer_fields(insert.who, carried)})
    return out


def _speaker_key(verdict: Verdict, row: Any) -> tuple[str, Any]:
    """Who a verdict puts a row on, comparable across a split's two halves: a new label, else a track (a
    keep or an abstention is the row's acoustic one)."""
    if verdict.kind == "new":
        return ("new", verdict.label)
    if verdict.kind == "track":
        return ("track", verdict.track)
    origin = sidecar.acoustic_main(row.speaker)
    return ("track", origin["filled"] if isinstance(origin, dict) else origin)


def general_blocks(context_pack: Any, source_ids: Sequence[str] | None = None) -> list[Mapping[str, Any]]:
    """The preparation task's general context, one block per research chunk (`ContextPack.general_blocks`):
    every chunk's in order, or with `source_ids` the chunks the window holding them reads, own first."""
    return context_pack.general_blocks(source_ids) if context_pack is not None else []


def listed(context_pack: Any, field: str, source_ids: Sequence[str] | None = None) -> list[Mapping[str, Any]]:
    return [item for block in general_blocks(context_pack, source_ids) for item in block.get(field) or ()
            if isinstance(item, Mapping)]


def entities_of(context_pack: Any) -> list[Mapping[str, Any]]:
    """The preparation task's `entities` (prepare_task_v1.md), or none."""
    return listed(context_pack, "entities")


def speakers_of(context_pack: Any) -> list[Mapping[str, Any]]:
    """Who the preparation task saw speaking (`speakers`, asked only with the judge on), or none."""
    return listed(context_pack, "speakers")


#: A chunked run has one summary per chunk; a window gets those of the chunks it reads joined, own chunk
#: first, up to about two of them: what is cut is the farthest.
SUMMARY_MAX_CHARS = 1600


def summary_of(context_pack: Any, source_ids: Sequence[str]) -> str:
    """The summary the judge and pass 3 give the window holding `source_ids`."""
    text = "\n".join(str(block.get("global_summary") or "").strip()
                     for block in general_blocks(context_pack, source_ids)
                     if str(block.get("global_summary") or "").strip())
    return text if len(text) <= SUMMARY_MAX_CHARS else text[:SUMMARY_MAX_CHARS].rstrip() + "……"

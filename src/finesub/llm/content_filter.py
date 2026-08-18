"""Content-filter recovery ladder.

Gemini's PROHIBITED_CONTENT prompt filter cannot be disabled via safety
settings and is deterministic for an exact prompt (see ``client.py``), so the
only recovery is rebuilding the prompt without the offending injected
retrieval content. Stages keep their injected web content as
:class:`InjectionUnit` lists (split from the rendered blocks they already
have) and drive the ladder here:

1. leave-one-out over URL-extract units (few, and the most toxic-dense) —
   a pass identifies the toxic unit, which is blacklisted task-wide;
2. drop all URL-extract units;
3. drop every unit (query result groups / evidence pack included);
4. exhausted -> :class:`ContentFilterExhaustedError` (the block most likely
   comes from the task's own text, not the retrieval injection).

A plain same-prompt retry is only worth one call when a stage has nothing to
drop (``plain_retry=True``); with units available it is skipped (the block is
deterministic, an unchanged retry just wastes quota).

Ladder attempts never count toward a stage's parse/validation retries; the
attempt cap is naturally ``len(url_units) + 3`` (+1 with ``plain_retry``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Generic, List, Sequence, TypeVar

T = TypeVar("T")

# Rendered-section headers (web_search.search_result_sections /
# extract_result_sections) double as the unit boundaries.
_QUERY_SECTION_RE = re.compile(r"^--- query: (?P<label>.*) ---$")
_EXTRACT_SECTION_RE = re.compile(r"^--- 深度提取 url: (?P<label>.*) ---$")
_BUDGET_NOTE_PREFIX = "（注入预算说明："

KIND_QUERY_RESULTS = "query_results"
KIND_URL_EXTRACT = "url_extract"
KIND_EVIDENCE_PACK = "evidence_pack"
_URL_KINDS = (KIND_URL_EXTRACT,)

BLACKLIST_ARTIFACT_KIND = "content_filter_blacklist"
LADDER_ARTIFACT_KIND = "content_filter_ladder"

DROPPED_UNITS_NOTE = "（部分检索内容因内容安全策略被本地移除。）"

_LADDER_SLEEP_SECONDS = 0.5


class ContentFilterExhaustedError(RuntimeError):
    """All ladder rungs failed: the block likely comes from the task text."""

    def __init__(self, stage: str, attempts: int) -> None:
        super().__init__(
            f"{stage}: prompt still blocked by the content filter after "
            f"dropping every injected retrieval unit ({attempts} attempts); "
            "the source text/context itself likely triggers the filter."
        )
        self.stage = stage
        self.attempts = attempts


@dataclass(frozen=True)
class InjectionUnit:
    """One independently droppable piece of injected retrieval content."""

    kind: str
    stable_id: str
    text: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    def to_metadata(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "stable_id": self.stable_id,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class SplitBlock:
    """A rendered injection block split into droppable units.

    ``preamble``/``tail`` are non-unit scaffolding (wrapper tags, the trailing
    budget note) preserved verbatim on rebuild.
    """

    preamble: str
    units: tuple[InjectionUnit, ...]
    tail: str

    def render(self, active_units: Sequence[InjectionUnit]) -> str:
        """Rebuild the block text from the surviving units.

        Returns "" when nothing (units or scaffolding-worthy content) is left;
        a drop is marked with a short neutral note so the model does not treat
        the reduced results as exhaustive.
        """

        parts: List[str] = []
        if self.preamble.strip():
            parts.append(self.preamble.strip("\n"))
        parts.extend(unit.text.strip("\n") for unit in active_units)
        if len(active_units) < len(self.units):
            parts.append(DROPPED_UNITS_NOTE)
        if not any(part.strip() for part in parts):
            return ""
        if self.tail.strip():
            parts.append(self.tail.strip("\n"))
        return "\n\n".join(part for part in parts if part.strip())


def split_rendered_search_block(text: str) -> SplitBlock:
    """Split a rendered search/extract block into per-section units.

    Sections start at ``--- query: ... ---`` (query result groups) or
    ``--- 深度提取 url: ... ---`` (deep-extract pages); anything before the first
    section is preamble (e.g. a ``<search_results>`` wrapper line) and a
    trailing budget note goes to ``tail``.
    """

    if not (text or "").strip():
        return SplitBlock(preamble="", units=(), tail="")
    lines = text.splitlines()
    preamble_lines: List[str] = []
    tail_lines: List[str] = []
    units: List[InjectionUnit] = []
    current: List[str] | None = None
    current_meta: tuple[str, str] | None = None

    def _flush() -> None:
        nonlocal current, current_meta
        if current is not None and current_meta is not None:
            kind, label = current_meta
            units.append(
                InjectionUnit(kind=kind, stable_id=label, text="\n".join(current))
            )
        current = None
        current_meta = None

    for line in lines:
        query_match = _QUERY_SECTION_RE.match(line)
        extract_match = _EXTRACT_SECTION_RE.match(line)
        if query_match or extract_match:
            _flush()
            if query_match:
                current_meta = (KIND_QUERY_RESULTS, query_match.group("label").strip())
            else:
                current_meta = (KIND_URL_EXTRACT, extract_match.group("label").strip())
            current = [line]
            continue
        if line.startswith(_BUDGET_NOTE_PREFIX) or (
            tail_lines and current is None
        ):
            _flush()
            tail_lines.append(line)
            continue
        # Closing wrapper tags after the last section belong to the tail.
        if current is not None and line.strip().startswith("</") and line.strip().endswith(">"):
            _flush()
            tail_lines.append(line)
            continue
        if current is not None:
            current.append(line)
        else:
            preamble_lines.append(line)
    _flush()
    return SplitBlock(
        preamble="\n".join(preamble_lines),
        units=tuple(units),
        tail="\n".join(tail_lines),
    )


def evidence_pack_block(text: str) -> SplitBlock:
    """Wrap a monolithic evidence pack / context blob as one droppable unit."""

    if not (text or "").strip():
        return SplitBlock(preamble="", units=(), tail="")
    return SplitBlock(
        preamble="",
        units=(
            InjectionUnit(
                kind=KIND_EVIDENCE_PACK, stable_id="evidence_pack", text=text
            ),
        ),
        tail="",
    )


@dataclass
class LadderOutcome(Generic[T]):
    result: T
    level: int  # -1 = first call passed; 1..3 = recovery rung that passed
    attempts: int
    dropped_units: List[InjectionUnit]
    identified_units: List[InjectionUnit]  # leave-one-out located toxic units

    @property
    def recovered(self) -> bool:
        return self.level >= 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "attempts": self.attempts,
            "dropped_units": [unit.to_metadata() for unit in self.dropped_units],
            "identified_units": [
                unit.to_metadata() for unit in self.identified_units
            ],
        }


def run_content_filter_ladder(
    *,
    units: Sequence[InjectionUnit],
    call: Callable[[Sequence[InjectionUnit]], T],
    stage: str,
    blocked: Callable[[T], bool] | None = None,
    blocked_exception: type[BaseException] | None = None,
    plain_retry: bool = False,
    sleep_seconds: float = _LADDER_SLEEP_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> LadderOutcome[T]:
    """Run ``call`` with the full unit set, recovering from filter blocks.

    ``call`` receives the active unit subset (the stage rebuilds its messages
    from it). A block is signalled either by ``blocked(result)`` returning
    True or by ``call`` raising ``blocked_exception``. Raises
    :class:`ContentFilterExhaustedError` when every rung fails.
    """

    if blocked is None and blocked_exception is None:
        raise ValueError("run_content_filter_ladder needs blocked or blocked_exception")

    _sleep = time.sleep if sleep is None else sleep
    attempts = 0

    def _attempt(active: Sequence[InjectionUnit]) -> tuple[bool, T | None]:
        nonlocal attempts
        if attempts:
            _sleep(sleep_seconds)
        attempts += 1
        try:
            result = call(active)
        except BaseException as exc:  # noqa: BLE001 - re-raised unless blocked
            if blocked_exception is not None and isinstance(exc, blocked_exception):
                return True, None
            raise
        if blocked is not None and blocked(result):
            return True, result
        return False, result

    all_units = list(units)
    was_blocked, result = _attempt(all_units)
    if not was_blocked:
        return LadderOutcome(result, -1, attempts, [], [])

    if plain_retry:
        was_blocked, result = _attempt(all_units)
        if not was_blocked:
            return LadderOutcome(result, 0, attempts, [], [])

    url_units = [unit for unit in all_units if unit.kind in _URL_KINDS]
    # The rungs are computed independently, so their active sets can coincide:
    # with a single URL extract, rung 2 is rung 1's only iteration, and when
    # URL extracts are the *only* units, rungs 1/2/3 all send the empty-injection
    # prompt. Re-sending a byte-identical prompt to a deterministic filter is
    # exactly what this module's own docstring says not to do -- it is why
    # `plain_retry` is disabled whenever units exist. Remembering what has
    # already gone out costs one set and saves whole calls: on free-tier
    # 3.6-flash (RPD 20) each duplicate is 5% of the day, and ladder attempts
    # are deliberately not capped by the validation-retry budget.
    tried: set[tuple[int, ...]] = set()

    def _fingerprint(active: list) -> tuple[int, ...]:
        return tuple(id(unit) for unit in active)

    def _attempt_once(active: list):
        key = _fingerprint(active)
        if key in tried:
            return None
        tried.add(key)
        return _attempt(active)

    # Rung 1: leave-one-out over URL extracts — a pass identifies the culprit.
    for suspect in url_units:
        active = [unit for unit in all_units if unit is not suspect]
        outcome = _attempt_once(active)
        if outcome is None:
            continue
        was_blocked, result = outcome
        if not was_blocked:
            return LadderOutcome(result, 1, attempts, [suspect], [suspect])
    # Rung 2: drop every URL extract.
    if url_units:
        active = [unit for unit in all_units if unit.kind not in _URL_KINDS]
        outcome = _attempt_once(active)
        if outcome is not None:
            was_blocked, result = outcome
            if not was_blocked:
                return LadderOutcome(result, 2, attempts, list(url_units), [])
    # Rung 3: drop everything droppable.
    if all_units:
        outcome = _attempt_once([])
        if outcome is not None:
            was_blocked, result = outcome
            if not was_blocked:
                return LadderOutcome(result, 3, attempts, list(all_units), [])
    raise ContentFilterExhaustedError(stage, attempts)


def load_content_filter_blacklist(
    task_artifact_dir: str | Path | None,
) -> set[str]:
    """Content hashes of units blacklisted earlier in this task (resume-safe)."""

    if not task_artifact_dir:
        return set()
    path = Path(task_artifact_dir).expanduser() / "task-artifacts.jsonl"
    if not path.exists():
        return set()
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("kind") != BLACKLIST_ARTIFACT_KIND:
            continue
        payload = record.get("payload")
        if isinstance(payload, dict):
            content_hash = str(payload.get("content_hash") or "")
            if content_hash:
                hashes.add(content_hash)
    return hashes


def strip_blacklisted_units(
    units: Sequence[InjectionUnit], blacklist: set[str]
) -> tuple[list[InjectionUnit], list[InjectionUnit]]:
    """Partition units into (kept, stripped-by-blacklist)."""

    if not blacklist:
        return list(units), []
    kept: list[InjectionUnit] = []
    stripped: list[InjectionUnit] = []
    for unit in units:
        (stripped if unit.content_hash in blacklist else kept).append(unit)
    return kept, stripped


def record_content_filter_recovery(
    task_artifact_dir: str | Path | None,
    *,
    task_id: str,
    stage: str,
    outcome: LadderOutcome[Any],
    blacklist: set[str] | None = None,
) -> None:
    """Persist ladder + blacklist artifacts after a recovery (no-op if clean).

    Identified leave-one-out culprits and every wholesale-dropped unit are
    blacklisted by ``content_hash`` so later windows/rounds of the same task
    strip them before render. ``blacklist`` (when passed) is updated in place.
    """

    if outcome.level < 1:
        # -1 = clean pass; 0 = plain same-prompt retry — nothing to blacklist.
        return
    # Lazy import: knowledge.base imports nothing from this module today, and
    # stages already depend on both — keep content_filter a leaf.
    from .knowledge.base import append_task_artifact

    to_blacklist = list(outcome.identified_units) or list(outcome.dropped_units)
    if task_artifact_dir:
        append_task_artifact(
            task_artifact_dir,
            kind=LADDER_ARTIFACT_KIND,
            task_id=task_id,
            payload={"stage": stage, **outcome.to_metadata()},
        )
        for unit in to_blacklist:
            append_task_artifact(
                task_artifact_dir,
                kind=BLACKLIST_ARTIFACT_KIND,
                task_id=task_id,
                payload={
                    **unit.to_metadata(),
                    "first_blocked_stage": stage,
                    "located_level": outcome.level,
                },
            )
    if blacklist is not None:
        for unit in to_blacklist:
            blacklist.add(unit.content_hash)


def run_injection_ladder(
    *,
    block: SplitBlock,
    call: Callable[[str], T],
    stage: str,
    blocked: Callable[[T], bool] | None = None,
    blocked_exception: type[BaseException] | None = None,
    blacklist: set[str] | None = None,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    plain_retry: bool = False,
    sleep_seconds: float = _LADDER_SLEEP_SECONDS,
    sleep: Callable[[float], None] | None = None,
) -> LadderOutcome[T]:
    """Strip the task blacklist, run the ladder, record any recovery.

    ``call`` receives the rebuilt injection text for the active unit subset
    (``""`` when every unit was dropped).
    """

    active_blacklist = blacklist if blacklist is not None else set()
    units, _stripped = strip_blacklisted_units(block.units, active_blacklist)

    def _call(active: Sequence[InjectionUnit]) -> T:
        return call(block.render(active))

    outcome = run_content_filter_ladder(
        units=units,
        call=_call,
        stage=stage,
        blocked=blocked,
        blocked_exception=blocked_exception,
        plain_retry=plain_retry,
        sleep_seconds=sleep_seconds,
        sleep=sleep,
    )
    record_content_filter_recovery(
        task_artifact_dir,
        task_id=task_id,
        stage=stage,
        outcome=outcome,
        blacklist=active_blacklist,
    )
    return outcome

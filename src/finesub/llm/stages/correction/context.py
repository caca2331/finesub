"""The state one correction run shares between its windows.

:class:`CorrectionRun` is the former closure of ``execute_correction_windows``
written down: the caller's switches, the services the run talks to, the four
objects that answer a per-window question (media, geometry, ledger, carried
context) and the accumulators the windows fold into. The serial and parallel
drivers take one of these instead of being nested functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from finesub.media.clips import CLIP_AUDIO_SUFFIX, CLIP_VIDEO_SUFFIX
from .progress import WindowProgress

from ...client import RoleClient
from ...media_upload import UploadedFileRef
from ...chunking import SubtitleWindow, split_window_in_half
from ...clip_prefetch import WindowClipPrefetcher
from ...routing.config import (
    CapabilityTier,
    ModelLimits,
    WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
)
from ...exchange_log import ExchangeLogger
from ...knowledge.base import load_entry_texts
from ...output_protocol import (
    CsvValidationResult,
    TranslatedCsvSegment,
    merge_translated_csv_windows,
    validate_correction_window_output,
)
from ...routing.profiles import TranslationProfile
from ...prompt_variants import resolve_variant
from ...prompts import ContextPack, render_advice_ledger
from ...session_checkpoint import SessionCheckpointStore
from ...token_budget import TokenCounter
from ...web_search import WebSearchClient
from .commit import _append_window_cache, _load_window_cache, _window_input_hash
from .metadata import _window_audio_label
from .query_round import QueryRoundProduct


@dataclass
class _WindowRunOutcome:
    """What one window's attempt loop produced.

    Either ``validation`` is set (the window -- possibly narrowed to its
    first half by mid-loop splits -- passed), or ``restart_halves`` is: the
    final attempt still looked truncated, so both halves go back to the
    caller's queue as fresh units with their own retry budgets.
    """

    window: SubtitleWindow
    validation: CsvValidationResult | None = None
    next_advice: str = ""
    next_transfer: List[str] = field(default_factory=list)
    restart_halves: Tuple[SubtitleWindow, SubtitleWindow] | None = None


@dataclass
class CarriedContext:
    """What one window hands to the next: kept entries, and advice.

    Both are cumulative and both are read by every later window, which is why
    they were free-floating lists closed over by half the loop. The entry keys
    are also a resume input (they enter the per-window hash), so who may
    replace them matters: only a committed window's `<keep_entries>`.
    """

    knowledge_root: str | Path
    #: How many keys the chain may carry. Without a query round no window can
    #: ask for an entry back, so the chain is the only copy and gets the whole
    #: window budget rather than the increment-sized reserve.
    cap: int
    keys: List[str] = field(default_factory=list)
    #: (chunk_id, advice) in window order, rendered with labels for the prompt.
    advice: List[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def starting_from(
        cls,
        initial_keys: Sequence[str],
        *,
        knowledge_root: str | Path,
        cap: int,
    ) -> "CarriedContext":
        return cls(
            knowledge_root=knowledge_root,
            cap=cap,
            keys=list(dict.fromkeys(initial_keys))[:cap],
        )

    def entry_bodies(self) -> Dict[str, str]:
        """Current text for the carried keys, re-read rather than cached.

        The knowledge base auto-commits between windows, so a body held from
        window one is not necessarily what the base says now.
        """

        if not self.keys:
            return {}
        found, _missing = load_entry_texts(self.knowledge_root, self.keys)
        return found

    def rendered_advice(self) -> str:
        return render_advice_ledger(self.advice)

    def note_advice(self, chunk_id: str, advice: str) -> None:
        self.advice.append((chunk_id, advice))


@dataclass(frozen=True)
class WindowGeometry:
    """Everything that decides where a window's boundaries may fall.

    Splitting a window in half happens at five points in the correction loop,
    and each of them used to restate the same seven arguments. One of them
    quietly disagreeing with the planner is not a crash -- it is windows that
    split differently on resume than they did on the first run.
    """

    profile: TranslationProfile
    counter: TokenCounter
    limits: ModelLimits
    global_first_id: str
    global_last_id: str
    audio_duration: float | None

    #: How many times one planned window may be halved on retry. Each split
    #: doubles the calls that window costs, and a window that still does not
    #: fit after being halved is not a size problem any more — something else
    #: is wrong, and halving again just spends quota on the same failure.
    #: Past the cap the task stops with a RuntimeError.
    #:
    #: **Owner decision, 2026-09-02: 2. Changed to 1 on 2026-09-03.** One
    #: measured consequence of the smaller cap: the deepest leaf production can
    #: now reach is a half, whose worst *correct* discard ratio is 43.8% rather
    #: than the quarter's 61.9% (`bench-baselines.md` 二十五).
    MAX_SPLITS = 1

    @staticmethod
    def split_depth(window: SubtitleWindow) -> int:
        """How many times this window has already been halved.

        The rule lives on `SubtitleWindow` -- the output validator needs the
        same answer -- and this stays as the name the retry loop reads.
        """

        return window.split_depth

    def may_split(self, window: SubtitleWindow) -> bool:
        """Whether a **new** split of this window is still within budget.

        Both places that create splits ask here -- the retry loop and resume's
        refit. The refit used to recurse without asking, so a reused plan could
        keep halving past the cap on every resume; the cap is a property of the
        geometry, not of one code path.

        Replaying a leaf an *earlier* run committed is a different question and
        does not come through here: a finished artifact is not re-judged by
        today's parameters (README_DEV「复用的依据是任务身份」), so a cached
        `0001-a-a` from before the cap dropped still replays.
        """

        return self.split_depth(window) < self.MAX_SPLITS

    def split(self, window: SubtitleWindow):
        """The two halves of `window`, or None when it cannot be split."""

        return split_window_in_half(
            window,
            counter=self.counter,
            limits=self.limits,
            global_first_id=self.global_first_id,
            global_last_id=self.global_last_id,
            audio_duration=self.audio_duration,
            profile=self.profile,
            context_tokens=WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
        )


@dataclass
class ResumeLedger:
    """What earlier passes of this task already committed, and where.

    Two things travel together everywhere resume is consulted -- the parsed
    records and the file they came from -- and `enabled` decides whether either
    is real. Keeping them apart meant every call site restating
    `if resume_enabled:` around a path that may be None.
    """

    enabled: bool
    path: Path | None
    task_fingerprint: str
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def load(self) -> None:
        if self.enabled and self.path is not None:
            self.records = _load_window_cache(self.path, self.task_fingerprint)

    def record(self, chunk_id: str) -> Dict[str, Any] | None:
        return self.records.get(chunk_id)

    def holds(self, chunk_id: str) -> bool:
        return self.enabled and chunk_id in self.records

    def commit(self, marker: Dict[str, Any]) -> None:
        """Append one record to the durable ledger and to this view of it."""

        if not self.enabled or self.path is None:
            return
        _append_window_cache(self.path, marker)
        chunk_id = marker.get("chunk_id")
        if isinstance(chunk_id, str):
            self.records[chunk_id] = marker

    def append_only(self, record: Dict[str, Any]) -> None:
        """Append without claiming a window -- the parallel entry-set row."""

        if self.enabled and self.path is not None:
            _append_window_cache(self.path, record)

    def expand_cached_splits(
        self, window: SubtitleWindow, geometry: WindowGeometry
    ) -> List[SubtitleWindow]:
        """`window`, or the leaves an earlier pass actually split it into."""

        record = self.record(window.chunk_id) or {}
        split_into = record.get("split_into")
        if not (isinstance(split_into, list) and split_into):
            return [window]
        halves = geometry.split(window)
        if halves is None or [half.chunk_id for half in halves] != [
            str(part) for part in split_into
        ]:
            return [window]
        return [
            leaf
            for half in halves
            for leaf in self.expand_cached_splits(half, geometry)
        ]

    def leaf_is_replayable(self, window: SubtitleWindow) -> bool:
        record = self.record(window.chunk_id)
        if not record or record.get("split_into"):
            return False
        if record.get("input_hash_core") != _window_input_hash(window):
            return False
        try:
            cached_tier = CapabilityTier(
                record.get("capability_tier", CapabilityTier.CAPABLE.value)
            )
            validation = validate_correction_window_output(
                str(record.get("content") or ""),
                window,
                variant=resolve_variant(record.get("variant") or None, cached_tier),
            )
        except (TypeError, ValueError):
            return False
        return validation.ok


@dataclass
class WindowMedia:
    """Which clip each round gets for a window, and the uploads behind them.

    Clip ownership is a switch question, not a round question (model-routing
    v2): a clip kind is cut iff *either* switch asks for it, and when both name
    the same kind the two rounds share one clip and one upload. That rule used
    to live in five closures over a dozen locals, which is why the sharing was
    easy to break by touching one of them.
    """

    profile: TranslationProfile
    audio_path: Path | None
    video_path: Path | None
    audio_label: str
    #: Uploads the caller already has (fast mode's round-1 clip), by chunk id.
    file_ref_seed: Mapping[str, UploadedFileRef]
    audio_clips: WindowClipPrefetcher | None = None
    video_clips: WindowClipPrefetcher | None = None
    #: Cuts an `.aac` on demand when a video-incapable target answers a video
    #: window. Built lazily: the ladder usually never fires.
    _make_ladder: Callable[[], WindowClipPrefetcher] | None = None
    _ladder: List[WindowClipPrefetcher] = field(default_factory=list)

    @property
    def correction_uses_video(self) -> bool:
        return bool(self.video_path) and self.profile.correction_use_video

    @property
    def correction_clips(self) -> WindowClipPrefetcher | None:
        if self.correction_uses_video:
            return self.video_clips
        return self.audio_clips if self.profile.correction_use_audio else None

    def correction_ref(self, window: SubtitleWindow) -> UploadedFileRef | None:
        seeded = self.file_ref_seed.get(window.chunk_id)
        if seeded is not None:
            return seeded
        prefetcher = self.correction_clips
        return prefetcher.get_ref(window) if prefetcher is not None else None

    def query_ref(self, window: SubtitleWindow) -> UploadedFileRef | None:
        """The query round's clip follows ``planning_media``."""

        if not self.profile.planning_use_audio:
            return None
        if self.profile.planning_media == self.profile.correction_media:
            # Same kind as the correction round: share its clip (and any
            # fast-mode seeded upload) instead of cutting a second one.
            return self.correction_ref(window)
        prefetcher = (
            self.video_clips if self.profile.planning_use_video else self.audio_clips
        )
        return prefetcher.get_ref(window) if prefetcher is not None else None

    def correction_label(self, window: SubtitleWindow) -> str:
        """How the correction prompt names the clip it was given."""

        use_video = self.correction_uses_video
        return _window_audio_label(
            self.video_path if use_video else self.audio_path,
            self.audio_label,
            window,
            clip_suffix=CLIP_VIDEO_SUFFIX if use_video else CLIP_AUDIO_SUFFIX,
        )

    def query_label(self, window: SubtitleWindow) -> str:
        return _window_audio_label(
            self.video_path if self.profile.planning_use_video else self.audio_path,
            self.audio_label,
            window,
            clip_suffix=(
                CLIP_VIDEO_SUFFIX
                if self.profile.planning_use_video
                else CLIP_AUDIO_SUFFIX
            ),
        )

    def ladder_audio_ref(self, window: SubtitleWindow) -> UploadedFileRef | None:
        if self._make_ladder is None:
            return None
        if self.audio_clips is not None:
            return self.audio_clips.get_ref(window)
        if not self._ladder:
            self._ladder.append(self._make_ladder())
        return self._ladder[0].get_ref(window)

    def schedule_correction(self, window: SubtitleWindow) -> None:
        prefetcher = self.correction_clips
        if prefetcher is not None:
            prefetcher.schedule(window)

    def prefetch_next_correction(
        self, windows: Sequence[SubtitleWindow], index: int
    ) -> None:
        prefetcher = self.correction_clips
        if prefetcher is not None:
            prefetcher.prefetch_next(windows, index)

    def shutdown(self) -> None:
        for prefetcher in (self.audio_clips, self.video_clips, *self._ladder):
            if prefetcher is not None:
                prefetcher.shutdown()



@dataclass
class CorrectionRun:
    """One correction run's shared state, from planning to the final SRT.

    This is what ``execute_correction_windows`` used to close over. Three
    groups: what the caller asked for (switches, budgets, paths), what the run
    talks to (client, search, counters, loggers) and what the windows fold
    into. The four per-window helpers -- ``media``, ``geometry``, ``ledger``
    and ``carried`` -- each answer one question and are documented on their
    own classes.
    """

    # --- what the caller asked for -------------------------------------
    profile: TranslationProfile
    context_pack: ContextPack | None
    knowledge_root: str | Path
    knowledge_enabled: bool
    extra_style: str
    style_block: str
    task_update_feedback: bool
    test_profile: bool
    resume: bool
    # Two-tier retry (docs/llm_followups.md "两档重试"): repairs within one
    # session chain, then fresh-session replacements; total calls per window
    # are the product of the two (n+1) forms.
    max_retries_per_window: int
    max_replacements_per_window: int
    max_search_queries_per_window: int
    parallel_window_limit: int
    task_artifact_dir: str | Path | None
    task_id: str
    #: Fast mode's pre-rendered entry block: when set, no window renders its
    #: own and the knowledge machinery below only records what was injected.
    entry_details: str
    evidence_pack_mode: bool
    #: ``retrieval`` is not ``none``: only then does a window run a query round
    #: or receive harness-side search results.
    external_injection: bool
    #: How many knowledge keys the transfer chain may carry between windows.
    transfer_cap: int

    # --- what the run talks to -----------------------------------------
    client: RoleClient
    search_client: WebSearchClient | None
    token_counter: TokenCounter
    exchange_logger: ExchangeLogger | None
    session_checkpoint_store: SessionCheckpointStore
    planning_limits: ModelLimits
    content_filter_blacklist: set[str]
    streamer_index_text: str
    common_index_text: str

    # --- per-window helpers --------------------------------------------
    media: WindowMedia
    geometry: WindowGeometry
    ledger: ResumeLedger
    carried: CarriedContext

    # --- run identity, recorded with every committed window -------------
    task_fingerprint: str
    knowledge_version: str

    # --- what the windows accumulate ------------------------------------
    #: Query-round products keyed by *base* chunk id, so -a/-b split halves
    #: and validation retries reuse one round.
    query_round_cache: Dict[str, QueryRoundProduct] = field(default_factory=dict)
    token_rows: List[Dict[str, Any]] = field(default_factory=list)
    rendered_segments: List[TranslatedCsvSegment] = field(default_factory=list)
    #: One warning per window, not one per retry-loop pass.
    warned_parallel_replays: set[str] = field(default_factory=set)
    #: Set by the planner once the window list is known; both drivers count
    #: through it, and splits grow its denominator.
    progress: WindowProgress | None = None
    #: Two tallies for the closing summary, bumped where the thing happens:
    #: they are for the one-line report, not an audit trail --
    #: `correction-windows.jsonl` already holds the per-window truth. Bumped
    #: through the methods below: parallel lanes count from worker threads,
    #: and a bare `+=` is a read-modify-write that can drop counts.
    repair_rounds: int = 0
    content_filter_recoveries: int = 0
    _tally_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def count_repair_round(self) -> None:
        with self._tally_lock:
            self.repair_rounds += 1

    def count_content_filter_recovery(self) -> None:
        with self._tally_lock:
            self.content_filter_recoveries += 1

    def commit_window(
        self,
        current: SubtitleWindow,
        validation: CsvValidationResult,
        next_advice: str,
    ) -> None:
        """Fold one window's result into the accumulated output + advice ledger.

        Continuity for later windows is input-only since v13 (the read-only
        preceding-context block planned into each window), so committing no
        longer feeds any prompt state besides the advice ledger.
        """

        self.rendered_segments = merge_translated_csv_windows(
            self.rendered_segments, current.source_ids, validation.segments
        )
        if next_advice.strip():
            self.carried.note_advice(current.chunk_id, next_advice)
        # Here rather than in the drivers: every window that finishes -- freshly
        # corrected or replayed from the ledger, serial or parallel -- lands
        # exactly once on this line.
        if self.progress is not None:
            self.progress.unit_done(current.chunk_id)

"""Second-model verification evidence via Qwen3-ASR.

The tail ``vad-asr`` pass (where the energy track is still alive) produces
*evidence*, never decisions: suspect segments get a ``qwen_verify`` field
with what Qwen3-ASR-0.6B heard in their span, and speech-bearing coverage
gaps are recorded in the stage metadata. The same referee implementation is
also used by the inline language-vote-flip redecoder. Tail consumers live
downstream — the stabilize stage reads the evidence when deciding drops
(docs/asr-stabilize.md), and the LLM layer can read recovered text as a
correction candidate.

Validated on 67 adjudicated clips (2026-08-05, docs/wt-refine-handoff.md P1):
phrase suspects 11/11, real-EN vs translation-mode 22/22 separated by the
auto pass's output language, coverage-gap recovery with zero fabrication,
and a drop audit that caught two real shouts our energy legs deleted. Known
weakness: shouts/screams may come back empty or cross-lingually rendered, so
absence of Qwen text must never authorize deleting shout-shaped segments —
only the polysyllabic closing-phrase family uses absence as evidence.

Dependency note: this uses the ``-hf`` checkpoints through native
transformers (``transformers>=5.13,<6``, shipped inside ``[asr]`` so the
same command yields the same stable everywhere; its tokenizers 0.22-0.23 and
huggingface-hub 1.x requirements sit inside faster-whisper's declared
ranges). The alternative — the ``qwen-asr``
wrapper package with the non-hf checkpoints — was implemented first and
rejected: it pins transformers exactly, drags a gradio/flask web stack plus
nagisa/dyNET its inference path never uses, and needs a ``--no-deps``
install that pyproject cannot express. Both paths share the same weights;
parity was verified output-for-output (including identical mishearings) on
the adjudicated smoke clips.

Performance stance (measured 2026-09-01, `docs/bench-baselines.md` 二十一):
bf16 on GPU (float32 on CPU), one lazy load per referee. The decode step is
**launch-bound** -- ~5 ms of GPU work inside a ~60 ms step, ~2000 kernel
launches per token -- so a step costs the same for eight clips as for one.
Hence the lever here: clips are decoded in padded batches of `BATCH_CLIPS`
(16 production clips: 12.7 s -> 1.5 s; every Whisper segment of a 23.5 min
file: 40 s, about the ASR pass itself). A second lever -- a static KV cache so
transformers compiles the step into a CUDA graph, 65 ms -> 9 ms per step --
is `qwen_decode.FixedShapeDecoder`, taken automatically once a call carries
enough audio (`COMPILE_MIN_AUDIO_SEC`, VRAM permitting). It exists because
transformers' own static-cache compile grows the 2-D attention mask by one
column per token and records one CUDA graph per mask length (hundreds per
file, no faster than eager, ~11 GiB of pools that outlive `close()`); the
decoder pins every step tensor to one shape per batch size, so one graph
serves all lengths: a full re-check of a 23.5 min file 31.6 s -> 16.3 s
(0.70 s per file-minute) within 3.3 GiB. Neither
lever is bit-exact against the sequential eager path (bf16 near-ties flip on
1-2 clips in 16); the substitute acceptance metric is the referee-verdict
agreement recorded in that section. flash-attn would touch the 8% that is
kernel time and is not worth its build.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...reporting import current_reporter
from ...text import normalized_compact
from ..preprocessing.audio import (
    get_audio_info_stream,
    load_audio_slice_stream,
    resample_if_needed,
    to_mono,
)
from ..postprocessing import stabilization as asr_stabilize
from ..preprocessing import energy as vad_energy
from ..runtime import hf_weights, phase_timing
from ..recognition.segments import coerce_optional_float
from .qwen_decode import FixedShapeDecoder

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
TARGET_SR = 16000
MAX_NEW_TOKENS = 256

# Clips per padded generate. The step cost is flat in the batch dimension
# (62 ms at B=1, 84 ms at B=8 on a 5070 Ti; 16 is twice as fast again on short
# clips), so the ceiling is memory, not speed -- and that is the cap below.
BATCH_CLIPS = 16
# The compiled path adds a `B x cache` static cache to that. Since the decoder
# encodes audio one clip at a time and sizes the cache to the need (512 for
# any batch under the padded cap), sixteen 15 s clips peak at 2.6 GiB and a
# full re-check at 3.3 GiB, so the compiled batch can be the eager one.
COMPILED_BATCH_CLIPS = BATCH_CLIPS
# ...and the memory is set by `B x longest clip`, not by B alone: `generate`
# runs the audio encoder over every padded row at once, so a batch of eight
# 30 s clips peaks at 3.2 GiB where eight 4 s clips peak at 1.8. A batch is
# therefore also capped by its padded audio; 120 s keeps the eager path at
# ~2.3 GiB, inside the entry tier's 3 GiB once Whisper is released.
BATCH_MAX_PADDED_SEC = 120.0
# The compiled path holds a static cache and CUDA-graph pools on top of that,
# so `auto` takes it only when the caller's VRAM budget covers the peak
# measured at the batch cap (docs/bench-baselines.md 二十一); a referee built
# without a budget does not compile unless told `accel="on"`. Measured peak
# of the compiled path at `COMPILED_BATCH_CLIPS`: 3.3 GiB on a full re-check,
# so the entry tier (3 GiB) never compiles and the standard tier may, even
# beside the resident Whisper pool (6.5 - 2.07 = 4.4 spare).
COMPILE_MIN_VRAM_GIB = 3.5
# One static KV cache length for every compiled call. It must not follow the
# prompt: sizing the cache per call re-allocates it and recompiles the step
# for every new clip length (~20 s each, measured), which is what made the
# first compile attempt useless. ~13 prompt tokens per audio second, so with
# `MAX_NEW_TOKENS` this holds a clip of ~55 s; a batch whose prompt does not
# fit takes the eager path for that call.
MAX_CACHE_LEN = 1024
# Compile only pays past this much *step-pacing* audio in one call (the sum
# over batches of each batch's longest clip): the batched eager step is ~75 ms
# and the compiled one ~9 ms, so ~66 ms saved per step against the ~25 s a
# warm inductor cache still costs per process and batch size -- about 380
# steps, or 130 s of Japanese. Cold (this batch size never compiled on this
# machine) it is ~90 s, so the floor is raised until the shape is recorded; a
# run that far past the line is long enough not to notice.
COMPILE_MIN_AUDIO_SEC = 150.0
COMPILE_MIN_AUDIO_SEC_COLD = 600.0
# `FINESUB_REFEREE_ACCEL=0` keeps the referee eager, same shape as the
# separator's `FINESUB_SEPARATOR_ACCEL`. Not a CLI option: the eager path is
# the same model and prompt, so this is an escape hatch, not a choice a user
# should have to make.
ACCEL_ENV = "FINESUB_REFEREE_ACCEL"
# Part of the compile-cache key, so a change in how the step is compiled lands
# in a fresh directory instead of reading stale artefacts.
ACCEL_BUILD_FORMAT = "3"

# Segment-level evidence field: {"text": str, "language": str | None}.
VERIFY_KEY = "qwen_verify"
# Stage-metadata list of {"start", "end", "text"} for uncovered spans where
# Qwen heard speech. Evidence only — nothing is inserted into the subtitle
# stream here (cue construction without word timestamps is a separate step).
GAP_RECOVERY_KEY = "qwen_gap_recoveries"

# Uncovered VAD spans shorter than this are not probed (interjection-sized
# gaps are dominated by Qwen's shout blindness; long gaps are where whole
# missed lines live).
GAP_MIN_SEC = 3.0
# Context added around a suspect segment's span. Tight on purpose: wide pads
# bleed neighboring speech into the clip and dilute the evidence.
SEGMENT_PAD_SEC = 0.1
# Latin-run suspect shape (mirrors the stabilize lang-switch gate, minus the
# confidence condition — evidence is cheap, decisions stay downstream).
SUSPECT_MIN_LETTERS = 8
SUSPECT_MIN_LATIN_RATIO = 0.7


def _segment_span(segment: Dict[str, object]) -> Optional[Tuple[float, float]]:
    start = coerce_optional_float(segment.get("start"))
    end = coerce_optional_float(segment.get("end"))
    if (
        start is None
        or end is None
        or not math.isfinite(start)
        or not math.isfinite(end)
        or end <= start
    ):
        return None
    return start, end


def _is_closing_phrase_shape(text: str) -> bool:
    compact = normalized_compact(text)
    return any(
        phrase in compact
        and len(compact)
        <= len(phrase) + asr_stabilize.CLOSING_GHOST_MAX_EXTRA_CHARS
        for phrase in asr_stabilize.CLOSING_GHOST_PHRASES
    )


def _is_latin_run(text: str) -> bool:
    letters = [c for c in str(text) if c.isalpha()]
    if len(letters) < SUSPECT_MIN_LETTERS:
        return False
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters) >= SUSPECT_MIN_LATIN_RATIO


def collect_suspect_indices(
    segments: Sequence[Dict[str, object]],
) -> List[int]:
    """Indices of segments worth second-model evidence.

    Four families: whole-segment closing phrases (any rate — the rate-ghost
    subset is already deletable without evidence, the normal-rate rest is
    only deletable WITH it; the English family is entirely in that second
    group, so this probe is the ONLY thing that reaches it), Latin runs
    inside a CJK-dominant output
    (real-EN vs translation-mode), and segments the stabilize noise legs
    would currently tag for dropping (so the drop can be vetoed when Qwen
    hears speech there).
    """

    run_cjk_dominant = asr_stabilize._run_is_cjk_dominant(list(segments))
    out: List[int] = []
    for index, segment in enumerate(segments):
        if _segment_span(segment) is None:
            continue
        text = str(segment.get("text") or "")
        if _is_closing_phrase_shape(text):
            out.append(index)
            continue
        if run_cjk_dominant and _is_latin_run(text):
            out.append(index)
            continue
        # The absolute-level suspect tier (`preprocessing/energy.py`): quiet
        # enough in true dBFS that a second opinion is worth having, loud
        # enough that dropping it unseen would be reckless. The tier is decided
        # upstream and only TAGGED there -- this is the line that turns the tag
        # into inference, so a run with the referee off pays nothing for it.
        if segment.get(vad_energy.SEGMENT_LEVEL_TIER_FIELD) == vad_energy.LEVEL_TIER_SUSPECT:
            out.append(index)
            continue
        prospective = asr_stabilize._profile_2_tags(
            segment, run_cjk_dominant=run_cjk_dominant
        )
        if (
            asr_stabilize.TAG_HIGHLY_SUSPECTED_HALLUCINATION in prospective
            or asr_stabilize.TAG_HIGHLY_SUSPECTED_FILLER in prospective
        ):
            out.append(index)
    return out


def collect_gaps(
    vad_intervals: Sequence[Dict[str, object]],
    segments: Sequence[Dict[str, object]],
    *,
    min_sec: float = GAP_MIN_SEC,
) -> List[Tuple[float, float]]:
    """Uncovered VAD spans of at least ``min_sec`` seconds."""

    covered = sorted(
        span for span in (_segment_span(s) for s in segments) if span
    )
    gaps: List[Tuple[float, float]] = []
    for interval in vad_intervals:
        span = _segment_span(interval)
        if span is None:
            continue
        pieces = [span]
        for c_start, c_end in covered:
            next_pieces: List[Tuple[float, float]] = []
            for p_start, p_end in pieces:
                if c_end <= p_start or c_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                if c_start > p_start:
                    next_pieces.append((p_start, c_start))
                if c_end < p_end:
                    next_pieces.append((c_end, p_end))
            pieces = next_pieces
        gaps.extend(p for p in pieces if p[1] - p[0] >= min_sec)
    return gaps


def _ensure_referee_weights(model_name: str) -> hf_weights.HfLoad:
    """Fetch the referee's weights with the mirror routing, if they are absent.

    Same shape as the ASR stage's own prefetch and, since both go through
    `hf_weights.prepare`, literally the same code: on the CLI these 1.5 GB used
    to arrive through `from_pretrained`, which knows nothing about this
    project's endpoint routing or its per-class failure counter.

    All this adds is the gate: only the default model, because the manifest
    describes no other, so a referee pointed elsewhere must not pay for weights
    it will never load.
    """

    if model_name != DEFAULT_QWEN_MODEL:
        return hf_weights.UNMANAGED
    return hf_weights.prepare("qwen-referee")


def plan_batches(
    lengths: Sequence[int], *, max_clips: Optional[int] = None
) -> List[List[int]]:
    """Index batches for one call: length-sorted, at most `max_clips`
    (default `BATCH_CLIPS`, read at call time so a bench probe can vary it),
    and no batch whose padded audio (`count x longest`) exceeds
    `BATCH_MAX_PADDED_SEC`. A clip longer than the cap by itself still runs,
    alone -- exactly what the sequential path did with it.

    Sorting keeps the padding small and makes the batches' shapes few: a call
    typically yields one or two, which matters for the compiled path since
    every distinct batch size is compiled once.
    """

    if max_clips is None:
        max_clips = BATCH_CLIPS
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    cap = BATCH_MAX_PADDED_SEC * TARGET_SR
    batches: List[List[int]] = []
    current: List[int] = []
    for index in order:
        longest = lengths[index]  # ascending, so the newest is the longest
        if current and (
            len(current) >= max_clips or (len(current) + 1) * longest > cap
        ):
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def _accel_cache_dir(model_name: str) -> Path | None:
    """Where this machine keeps the compiled step's Inductor cache.

    A checkout keeps it in its own ignored `cache/` (a worktree resolves to
    the main checkout, which is fine: the key below is what tells artefacts
    apart, not the tree); a packaged or managed install uses its cache root;
    a bare wheel install falls back to the user's home. The key carries every
    version the artefacts are bound to, so a new torch, CUDA, card,
    transformers or model lands in a fresh directory and the old one is never
    read again. `None` when the key cannot be formed (no CUDA build of torch)
    -- there is nothing to compile then anyway.
    """

    try:
        import torch
        import transformers

        if not torch.cuda.is_available():
            return None
        parts = (
            ACCEL_BUILD_FORMAT,
            torch.__version__,
            str(torch.version.cuda),
            torch.cuda.get_device_name(0),
            transformers.__version__,
            model_name,
        )
    except Exception:  # noqa: BLE001 - no cache dir is a valid answer
        return None
    key = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    root: Path | None = None
    try:
        from finesub.paths import resolve_checkout_root, resolve_managed_app_paths

        checkout = resolve_checkout_root()
        if checkout is not None:
            root = Path(checkout) / "cache"
        else:
            managed = resolve_managed_app_paths()
            if managed is not None:
                root = Path(managed.cache)
    except Exception:  # noqa: BLE001 - fall through to the home directory
        root = None
    if root is None:
        root = Path.home() / ".cache" / "finesub"
    return root / "qwen-referee-accel" / key / "inductor"


_SHAPES_FILE = "shapes.json"


def _compiled_shapes(model_name: str) -> set[tuple[int, int]]:
    """`(batch, cache_len)` shapes this machine has compiled before (the
    Inductor cache makes them warm), as recorded by `_record_compiled_shape`.
    Best effort; the record lives beside the cache directory, so it is bound
    to the same build key."""

    cache = _accel_cache_dir(model_name)
    if cache is None:
        return set()
    try:
        data = json.loads((cache.parent / _SHAPES_FILE).read_text(encoding="utf-8"))
        return {(int(batch), int(length)) for batch, length in data.get("shapes", [])}
    except Exception:  # noqa: BLE001 - absent or unreadable both mean cold
        return set()


def _record_compiled_shape(model_name: str, shape: tuple[int, int]) -> None:
    cache = _accel_cache_dir(model_name)
    if cache is None:
        return
    try:
        shapes = sorted(_compiled_shapes(model_name) | {(int(shape[0]), int(shape[1]))})
        cache.parent.mkdir(parents=True, exist_ok=True)
        (cache.parent / _SHAPES_FILE).write_text(
            json.dumps({"shapes": [list(item) for item in shapes]}), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 - a lost record only costs a cold floor
        pass


class QwenReferee:
    """One lazily loaded Qwen3-ASR model per referee use.

    The model (1.5 GB of weights; ~2.3 GB peak with a batched decode at the
    120 s cap, ~2.9 GB compiled) is loaded on first use and
    freed via ``close()``. The post-run verification pass loads it after the
    stage has closed the Whisper pool; the inline lang-redecode referee
    co-resides with the pool instead, on CUDA only when the GPU profile has
    the spare VRAM for it (docs/asr-align.md). Auto-language
    transcription via the native transformers path; no forced-aligner, no
    accelerate.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        *,
        device: str = "cuda",
        context: str = "",
        accel: str = "auto",
        vram_budget_gib: Optional[float] = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None
        # Free VRAM the caller can vouch for (the tier's usable figure, minus
        # the Whisper pool while it is resident). Gates the compiled path's
        # extra footprint; None means "unknown", which `auto` reads as no.
        self._vram_budget_gib = vram_budget_gib
        # Plain text, assembled by whoever knows where names come from. This
        # module never learns: `speech` must not import `llm`, so the knowledge
        # base reaches the recogniser as a string and nothing else.
        self._context = str(context or "").strip()
        # `auto` compiles past `COMPILE_MIN_AUDIO_SEC` when the VRAM budget
        # allows; `on` always (bench probes); `off` never. CUDA only: the
        # compiled step is a CUDA graph.
        if accel not in ("auto", "on", "off"):
            raise ValueError(f"accel must be auto|on|off, got {accel!r}")
        self._accel = accel
        self._accel_prepared = False
        self._accel_failed = False
        # The fixed-shape decoder, built with the model when the compiled
        # path is prepared; its `compiled_shapes` says which (batch, cache
        # length) pairs this process has already paid for (the first call of
        # each is timed as `qwen.compile`).
        self._decoder: Optional[FixedShapeDecoder] = None
        # The stage may warm the model from a helper thread while Whisper is
        # still decoding; the inline referee can be asked for evidence in the
        # meantime and must wait for that load rather than start a second.
        self._lock = threading.RLock()

    def set_vram_budget(self, vram_budget_gib: Optional[float]) -> None:
        """Re-vouch after the Whisper pool is gone: a referee warmed beside
        the pool was told the spare-beside-Whisper figure, and the tail pass
        may raise it to the whole tier budget."""

        self._vram_budget_gib = vram_budget_gib

    def set_context(self, context: str) -> None:
        """Attach (or clear) the context after construction.

        Exists for one caller: the stage reuses the inline language referee for
        the verification pass to avoid a second model load, and that instance
        was built context-free on purpose -- the language phases must not be
        biased by a list of names. Setting it afterwards is what keeps the
        reuse from silently dropping the feature.
        """

        self._context = str(context or "").strip()

    def _transcription_inputs(self, clips, processor):
        """Processor inputs for one padded batch, carrying the run's context.

        `apply_transcription_request` builds the conversations itself and has
        no slot for anything but a language hint, so a context has to go
        through `apply_chat_template` with the system turn written out. The
        no-context path deliberately stays on the processor's own helper: it
        is the shape the model card documents, and a run that injects nothing
        should not start decoding through a hand-built conversation. Both pad
        on the left (the processor's default), which is what a batched decode
        needs.
        """

        audios = [np.asarray(clip, dtype=np.float32) for clip in clips]
        if not self._context:
            return processor.apply_transcription_request(audio=audios)
        conversations = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._context}],
                },
                {"role": "user", "content": [{"type": "audio", "audio": audio}]},
            ]
            for audio in audios
        ]
        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

    @property
    def requested_device(self) -> str:
        return self._device

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                # Separated from inference on purpose. The referee's cost reads
                # like a per-suspect price and is not: on a 23.5 min asset with
                # two suspects it was almost entirely this load, so "batch the
                # suspects" would optimise the wrong half. See
                # `docs/bench-baselines.md` -> P8/A4.
                with phase_timing.phase("qwen.load"):
                    self._model = self._load_model()
            return self._model

    def _load_model(self):
        """Fetch the weights and put the model on its device. Not cached here."""

        plan = _ensure_referee_weights(self._model_name)
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        wants_cuda = self._device.startswith("cuda")

        def fetch(load: hf_weights.HfLoad):
            # Both calls under one plan: the processor is the one that lists
            # chat templates over the network, and retrying only it would leave
            # the weights to fail the same way a moment later.
            self._processor = AutoProcessor.from_pretrained(
                self._model_name,
                revision=load.revision,
                local_files_only=load.local_files_only,
            )
            # bf16 on GPU; float32 on CPU for speed, not correctness: CPU
            # bf16/fp16 halve the footprint but decode 2.2-2.5x slower with
            # byte-identical output (docs/asr-align.md, referee device).
            return AutoModelForMultimodalLM.from_pretrained(
                self._model_name,
                revision=load.revision,
                local_files_only=load.local_files_only,
                dtype=torch.bfloat16 if wants_cuda else torch.float32,
            )

        model = hf_weights.offline_first(fetch, plan, what="qwen referee")
        if wants_cuda:
            try:
                model = model.to(self._device)
            except Exception as exc:
                current_reporter().warning(
                    "cpu-fallback",
                    f"Qwen referee falling back to CPU ({exc})",
                    impact="第二模型校验会明显变慢",
                )
                # ``Module.to`` mutates parameters as it walks them. A
                # CUDA OOM can therefore leave a partially moved module;
                # an explicit device move is required, not just a dtype
                # cast, before inference can safely continue on CPU.
                model = model.to(device="cpu", dtype=torch.float32)
                torch.cuda.empty_cache()
        model.eval()
        return model

    def warm(self) -> None:
        """Load the processor and weights without transcribing anything.

        The load and the first clip are separable, so a caller that already
        knows it will need the referee can pay for the load while something
        else is still running. Idempotent and thread-safe -- `_ensure_model`
        caches under the referee's lock.

        `vad_asr_stage` calls this from a helper thread during the Whisper
        decode, but only when the GPU profile says the referee fits beside the
        pool (`lang_redecode.referee_device`): on the entry tier the load still
        waits for the pool to be released, or it would put both on a 4 GB card
        at once.
        """

        self._ensure_model()

    def transcribe_batch(
        self,
        clips: Sequence[np.ndarray],
        *,
        max_new_tokens: Optional[int] = None,
        on_batch: Optional[Callable[[int, int], None]] = None,
    ) -> List[Tuple[str, Optional[str]]]:
        """(text, detected language) per 16 kHz mono clip, auto language.

        Clips are sorted by length and decoded in padded batches of
        `BATCH_CLIPS` (results come back in input order); a call with enough
        audio takes the compiled static-cache path. The whole call is one
        `qwen.infer` span -- the clip count is already recorded as `suspects`,
        and what the A4 question needs is inference-vs-load, not per-clip
        variance -- with the first compiled call per step shape additionally
        under `qwen.compile`.

        ``on_batch(done, total)`` is called with the running clip count after
        each batch. It exists because this call is the longest unbroken
        silence in the stage -- on a contended card it is minutes, and every
        second of it looks exactly like a hung run to whoever is watching.
        The batch is the finest grain that is *real*: a `generate` cannot be
        interrupted from here, so a per-clip counter would be a lie, and it is
        also what bounds the event count at the call site, which
        `docs/reporting.md` asks for rather than leaving to the renderer.

        ``max_new_tokens`` exists for callers that need the language prelude
        rather than the transcript, and it bounds the worst case for them.
        ⚠ It is a smaller lever than it looks: dropping the default 256 to 48
        moved the language audit from 52.3s to 54.9s -- nothing. The prefill
        is a flat ~0.1 s whatever the clip length; what costs is the number of
        tokens the model decides to emit, ~3 per second of Japanese, and that
        is not a knob. Note also that the text has to come back non-empty for
        the language to be trusted, so this cannot go to zero.
        """

        if not clips:
            return []
        budget = int(max_new_tokens or MAX_NEW_TOKENS)
        results: List[Tuple[str, Optional[str]]] = [("", None)] * len(clips)
        lengths = [len(clip) for clip in clips]
        batches = plan_batches(lengths)
        # The decode loop of a batch is paced by its longest clip, which is
        # also what the compile threshold is measured against. The compiled
        # path runs smaller batches (its static caches cost VRAM), so it has
        # its own plan; the pacing is the same to within one batch.
        pacing_sec = sum(len(clips[batch[-1]]) for batch in batches) / TARGET_SR
        compiled_plan = plan_batches(lengths, max_clips=COMPILED_BATCH_CLIPS)
        with self._lock, phase_timing.phase("qwen.infer"):
            self._ensure_model()
            # A step graph is keyed by (batch, cache length) and the cache
            # length follows the prompt, so the prompts are built first (CPU,
            # milliseconds) and the gate sees the exact shapes it would need.
            prepared = self._prepare(clips, compiled_plan)
            shapes = {
                shape
                for inputs in prepared
                if (shape := self._static_shape(inputs, budget)) is not None
            }
            compiled = self._compile_wanted(pacing_sec, shapes=shapes)
            if compiled:
                batches = compiled_plan
            elif batches != compiled_plan:
                prepared = self._prepare(clips, batches)
            done = 0
            for batch, inputs in zip(batches, prepared):
                replies = self._generate(inputs, budget, compiled=compiled)
                for index, reply in zip(batch, replies):
                    results[index] = reply
                done += len(batch)
                if on_batch is not None:
                    on_batch(done, len(clips))
        return results

    def _prepare(self, clips: Sequence[np.ndarray], plan: List[List[int]]) -> list:
        return [
            self._transcription_inputs([clips[i] for i in batch], self._processor)
            for batch in plan
        ]

    @staticmethod
    def _static_shape(inputs, budget: int) -> Optional[tuple[int, int]]:
        """The `(batch, cache_len)` the compiled step would run this batch
        with, or None when the prompt does not fit the static cache."""

        batch, prompt_len = (int(n) for n in inputs["input_ids"].shape)
        if prompt_len + budget > MAX_CACHE_LEN:
            return None
        return batch, FixedShapeDecoder.cache_len_for(prompt_len, budget, MAX_CACHE_LEN)

    def _generate(
        self, inputs, budget: int, *, compiled: bool
    ) -> List[Tuple[str, Optional[str]]]:
        """One padded generate over prepared processor inputs; `compiled`
        asks for the static-cache path."""

        model = self._model
        processor = self._processor
        inputs = inputs.to(model.device, model.dtype)
        prompt_len = int(inputs["input_ids"].shape[1])
        shape = self._static_shape(inputs, budget) if compiled else None
        decoder = self._decoder if shape is not None else None
        use_static = decoder is not None
        try:
            if use_static and shape not in decoder.compiled_shapes:
                # Includes this batch's own decode; the compile is not
                # separable from the first run through the graph.
                with phase_timing.phase("qwen.compile"):
                    generated = self._run_compiled(decoder, inputs, budget)
                _record_compiled_shape(self._model_name, shape)
            elif use_static:
                generated = self._run_compiled(decoder, inputs, budget)
            else:
                generated = self._run_generate(inputs, budget)
        except Exception as exc:  # noqa: BLE001 - eager is always available
            if not use_static:
                raise
            self._accel_failed = True
            current_reporter().warning(
                "referee-accel-disabled",
                f"compiled referee step failed, staying eager ({exc})",
                impact="第二模型校验会慢一些",
                action=f"{ACCEL_ENV}=0 可跳过再次尝试",
            )
            generated = self._run_generate(inputs, budget)
        tail = generated[:, prompt_len:]
        texts = processor.decode(tail, return_format="transcription_only")
        # The raw output carries a "language <name>" prelude before the
        # transcript; best-effort parse, evidence-only.
        raws = processor.decode(tail)
        out: List[Tuple[str, Optional[str]]] = []
        for text, raw in zip(texts, raws):
            text = str(text or "").strip()
            match = re.search(r"language\s+([A-Za-z_]+)", str(raw))
            out.append((text, match.group(1) if match and text else None))
        return out

    def _run_generate(self, inputs, budget: int):
        import torch

        with torch.no_grad():
            return self._model.generate(**inputs, max_new_tokens=budget)

    def _run_compiled(self, decoder, inputs, budget: int):
        config = self._model.generation_config
        eos = config.eos_token_id
        return decoder.generate(
            inputs,
            max_new_tokens=budget,
            eos_token_ids=list(eos) if isinstance(eos, (list, tuple)) else [int(eos)],
            pad_token_id=int(config.pad_token_id),
        )

    def _compile_wanted(
        self, pacing_sec: float, *, shapes: set[tuple[int, int]]
    ) -> bool:
        """Whether this call should take the compiled step, and set it up.

        `shapes` are the `(batch, cache_len)` pairs the call would run: each
        is its own step graph, and the threshold depends on whether this
        machine has already built them (`_compiled_shapes` on disk) -- a warm
        shape costs ~25 s per process, a cold one ~90 s. An empty set means
        no batch fits the static cache, so there is nothing to compile.
        """

        if not shapes:
            return False
        if self._accel == "off" or self._accel_failed:
            return False
        if os.environ.get(ACCEL_ENV, "").strip() == "0":
            return False
        if not str(getattr(self._model, "device", self._device)).startswith("cuda"):
            return False
        if self._accel == "auto":
            budget = self._vram_budget_gib
            if budget is None or budget < COMPILE_MIN_VRAM_GIB:
                return False
            done = self._decoder.compiled_shapes if self._decoder is not None else set()
            pending = shapes - done
            if pending:
                warm = pending <= _compiled_shapes(self._model_name)
                floor = COMPILE_MIN_AUDIO_SEC if warm else COMPILE_MIN_AUDIO_SEC_COLD
                if pacing_sec < floor:
                    return False
        try:
            self._prepare_accel()
        except Exception as exc:  # noqa: BLE001 - eager is always available
            self._accel_failed = True
            current_reporter().warning(
                "referee-accel-disabled",
                f"could not set up the compiled referee step ({exc})",
                impact="第二模型校验会慢一些",
            )
            return False
        return True

    def _prepare_accel(self) -> None:
        """Process-wide compile settings, applied once per referee.

        Two are Windows-specific and match the separator's JIT path: the
        static CUDA launcher overflows a C long, and Dynamo's C++ shape guards
        need MSVC on PATH. The inductor cache is redirected only when nothing
        has claimed it yet -- the separator may already have pointed it at its
        own directory, and Inductor reads the variable once. The decoder that
        owns the fixed-shape step and its static caches is built here too.
        """

        if self._accel_prepared:
            return
        import torch
        import torch._dynamo
        import torch._inductor.config as inductor_config

        cache = _accel_cache_dir(self._model_name)
        if cache is not None and not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
        inductor_config.use_static_cuda_launcher = False
        torch._dynamo.config.enable_cpp_symbolic_shape_guards = False
        torch._dynamo.config.cache_size_limit = max(
            int(torch._dynamo.config.cache_size_limit), 64
        )
        self._decoder = FixedShapeDecoder(
            self._model, max_cache_len=MAX_CACHE_LEN, compile_step=True
        )
        self._accel_prepared = True

    def close(self) -> None:
        with self._lock:
            if self._model is None:
                return
            decoder = self._decoder
            self._decoder = None
            self._model = None
            self._processor = None
            self._accel_prepared = False
            try:
                import torch

                if decoder is not None:
                    # Resets Dynamo's cache of the step and the graph pools,
                    # which otherwise keep the model and the KV caches alive
                    # (measured: 3 GiB per instance left on the card).
                    decoder.close()
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class _SpanReader:
    """Sequential 16 kHz mono reader over the stage's audio file."""

    def __init__(self, audio_path: str) -> None:
        self._path = str(audio_path)
        sr, total = get_audio_info_stream(self._path)
        if sr <= 0:
            raise RuntimeError(f"Invalid sample rate for audio: {self._path}")
        self._sr = int(sr)
        self._total = max(0, int(total))

    def read(self, start: float, end: float) -> np.ndarray:
        first = max(0, int(start * self._sr))
        last = min(self._total, int(end * self._sr))
        if last <= first:
            return np.zeros(0, dtype=np.float32)
        chunk, sr = load_audio_slice_stream(self._path, first, last - first)
        mono = to_mono(chunk)
        resampled, _ = resample_if_needed(mono.unsqueeze(0), int(sr), TARGET_SR)
        return resampled.squeeze(0).cpu().numpy().astype(np.float32)


#: The stage this pass reports under. Not one of its own: it *is* the tail of
#: the ASR stage, and a second stage name appearing after `aligned` reached
#: 100% would read as a new stage starting, which is not what happens -- the
#: same stage is still running, on its second model. Restarting the counter
#: under the same name is already an anticipated case: both renderers key
#: their progress de-duplication on `(step, total)` precisely because a
#: stage's denominator can change mid-run (`FileReporter.progress`).
_VERIFY_STAGE = "aligned"


def _verify_progress(total: int) -> Callable[[int, int], None]:
    """Clip-level progress for the verification pass, announced before it starts.

    The zero is emitted here rather than after the first batch because the
    load and the first `generate` *are* the slow part: what a watcher needs
    first is what is running and how much of it there is, and only then the
    count moving. Before this, the tail pass was the longest unbroken silence
    in the pipeline -- 168 s of a 197 s stage on a contended card, with no
    events at all, which is indistinguishable from a hang.
    """

    reporter = current_reporter()
    reporter.progress(
        _VERIFY_STAGE, completed=0, total=total, unit="clips", detail="第二模型校验"
    )

    def report(done: int, count: int) -> None:
        reporter.progress(
            _VERIFY_STAGE,
            completed=done,
            total=count,
            unit="clips",
            detail="第二模型校验",
        )

    return report


def apply_verification(
    segments: List[Dict[str, object]],
    *,
    vad_intervals: Sequence[Dict[str, object]],
    audio_path: str,
    referee: QwenReferee,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Attach ``qwen_verify`` evidence and collect gap recoveries.

    Returns (segments-with-evidence, stats). ``stats[GAP_RECOVERY_KEY]``
    carries the recovered-gap evidence for the stage metadata.
    """

    suspect_indices = collect_suspect_indices(segments)
    gaps = collect_gaps(vad_intervals, segments)
    stats: Dict[str, object] = {
        "model": referee._model_name,
        # Where this ran, because nobody chose it: `lang_redecode.referee_device`
        # derives it from the tier's VRAM minus the resident Whisper pool, so a
        # run that quietly verified on the CPU is otherwise indistinguishable
        # from one that had the card. Every other stage already records its
        # device (docs/plans/stage-device-plan.md 2.6); this was the one that did not.
        "device": referee.requested_device,
        "suspects": len(suspect_indices),
        "gaps_probed": len(gaps),
    }
    if not suspect_indices and not gaps:
        stats[GAP_RECOVERY_KEY] = []
        return segments, stats

    reader = _SpanReader(audio_path)
    clips: List[np.ndarray] = []
    for index in suspect_indices:
        start, end = _segment_span(segments[index])  # type: ignore[misc]
        clips.append(
            reader.read(start - SEGMENT_PAD_SEC, end + SEGMENT_PAD_SEC)
        )
    for start, end in gaps:
        clips.append(reader.read(start, end))

    # Spans clipped away entirely by the audio bounds (or degenerate ones)
    # must not reach the model; they read as "no speech heard".
    min_samples = int(0.05 * TARGET_SR)
    usable = [i for i, clip in enumerate(clips) if len(clip) >= min_samples]
    replies = referee.transcribe_batch(
        [clips[i] for i in usable],
        # Nothing survived the bounds check: there is no work to announce, and
        # `0/0` would be the one progress line that never moves.
        on_batch=_verify_progress(len(usable)) if usable else None,
    )
    results: List[Tuple[str, Optional[str]]] = [("", None)] * len(clips)
    for position, reply in zip(usable, replies):
        results[position] = reply

    out = list(segments)
    for position, index in enumerate(suspect_indices):
        text, language = results[position]
        item = dict(out[index])
        item[VERIFY_KEY] = {"text": text, "language": language}
        out[index] = item

    recoveries: List[Dict[str, object]] = []
    for position, (start, end) in enumerate(gaps):
        text, language = results[len(suspect_indices) + position]
        if normalized_compact(text):
            recoveries.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "language": language,
                }
            )
    stats[GAP_RECOVERY_KEY] = recoveries
    stats["gaps_recovered"] = len(recoveries)
    return out, stats

"""The recognition stage of a joint-ASR task: the source's mix -> one provider -> `-aligned.json`.

Takes the place of separation, VAD and Whisper for a task whose `--asr` is a joint backend. For a cloud backend the
local work is light: decoding the audio, Silero frame probabilities (the MOSS segments' unit times and MOSS Pro's
gap gate), the sentence-boundary model, production's DP and the timeline contract. Local MOSS decodes
on this machine's GPU (`moss_local`). `moss-diarization-aligner` takes a word backend's lines (`--asr-rows`: its API,
the task's cache, or a transcript file) and has local MOSS choose each line's speaker (`_diarization_aligned`). The
aligned artifact records the backend (`metadata.joint_asr`) and is what makes a task a joint-ASR task from then on
(`task_backend`).
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ... import api_keys, settings
from ...reporting import current_reporter
from ..postprocessing import segmentation as seg
from . import (GEMINI_35T, LOCAL, MAI2, MOSS_DIAR_ALIGNER, MOSS_LOCAL, MOSS_PRO, ROWS_BACKENDS, SCRIBE,
               SEGMENT_BACKENDS, resolve_backend)
from .cache import IDENTITY_FILE, CacheMiss, audio_fingerprint, recorded
from . import lines as joint_lines
from . import whisper_align
from . import repair as cloud_repair

if TYPE_CHECKING:
    from .mosspro import SpeechSpans

#: `metadata` key of a joint-ASR task's aligned and stable artifacts.
METADATA_KEY = "joint_asr"
#: The S5b / S6b names `docs/joint-asr.md` uses, recorded with the MOSS segments' lines.
TIMING = {False: "S5b", True: "S6b"}
#: `--asr-aligner` (owner 2026-11-20). `auto`: the forced aligner in a language it was trained on (`--language` not
#: given counts as one; local MOSS takes 50-odd), on the GPU where this run may use one, else on the CPU (2.6-3.1 min
#: an audio hour); on a GPU, Whisper then re-times the MOSS segments' unsure boundaries, except in English, where the
#: aligner alone is closer. `on`: the aligner in any language, and Whisper on a GPU in English too. 35t's repairs
#: take the aligner alone: Whisper's offset is one material's, and a repair aligns one spot at a time.
ALIGNER_MODES = ("auto", "on", "off")
#: Qwen3-ForcedAligner's languages, as `--language` codes (transformers' `FORCED_ALIGNER_LANGUAGES`).
ALIGNER_LANGUAGES = frozenset({"zh", "yue", "en", "ja", "ko", "fr", "de", "it", "pt", "ru", "es"})

AZURE_ENDPOINT_ENV = "FINESUB_AZURE_SPEECH_ENDPOINT"
AZURE_KEY_ENV = "FINESUB_KEY_AZURE_SPEECH"
#: MAI2's price per audio hour on every route (OpenRouter and Azure 2026-10, Vercel 2026-11), for the estimate it
#: prints.
MAI2_USD_PER_HOUR = 0.10
#: What the repair's re-asks add to MAI2's audio, by the survey (about 7 re-asks of 30 s per hour of source).
MAI2_REPAIR_SHARE = 0.06

_PROVIDER_NAMES = {MAI2: "MAI-Transcribe-2", GEMINI_35T: "Gemini 3.5 Transcribe", MOSS_PRO: "MOSS Pro",
                   SCRIBE: "ElevenLabs Scribe"}


class JointAsrError(RuntimeError):
    pass


def recorded_backend(payload: Mapping[str, Any]) -> str | None:
    """The joint backend an aligned or stable payload records, else None."""
    record = (payload.get("metadata") or {}).get(METADATA_KEY)
    return record.get("backend") if isinstance(record, Mapping) else None


def _read_backend(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return recorded_backend(payload) if isinstance(payload, dict) else None


def task_backend(stable_json: Path, aligned_json: Path, requested: str | None) -> str:
    """The task's ASR backend: what its products record, not this run's `--asr` (a task keeps its own
    products and choices). The stable first (profile -1 copies the aligned's metadata, and the aligned may
    be gone), then the aligned, then the request, then `[asr] backend`, then local."""
    for path in (stable_json, aligned_json):
        if path.exists():
            # A product that does not parse reads as local: a corrupt product is outside the resume
            # contract (delete it and its downstream to rerun), so this only picks which error comes first.
            return _read_backend(path) or LOCAL
    return resolve_backend(settings.ASR_BACKEND.resolve(requested))


@dataclass(frozen=True)
class AlignerUse:
    """How a run that aligns does it: `device` (`cuda` or `cpu`) for the forced aligner; `whisper`, whether Whisper
    re-times the unsure boundaries (`whisper_align`), and `english`, English segments too."""

    device: str
    whisper: bool = False
    english: bool = False

    def retime(self, language: str | None) -> whisper_align.Retime | None:
        """The MOSS segments' Whisper step for a run in `language` (None: not re-timed)."""
        return whisper_align.Retime(language, self.english) if self.whisper else None


def uses_aligner(value: str, backend: str, language: str | None, *, gpu_tier: str | None = None,
                 device: str | None = None) -> AlignerUse | None:
    """How `--asr-aligner` `value` times this backend's units (None: Silero, no aligner); `gpu_tier` and `device`
    are the run's (whether the aligner and Whisper may use a GPU)."""
    import importlib.util

    value = value.strip().lower()
    if value not in ALIGNER_MODES:
        raise ValueError(f"--asr-aligner must be one of {', '.join(ALIGNER_MODES)} (got {value!r})")
    if backend not in (*SEGMENT_BACKENDS, GEMINI_35T) or value == "off":
        return None
    lang = (language or "").strip().lower().split("-")[0]
    if value == "auto" and lang and lang not in ALIGNER_LANGUAGES:
        return None
    if value == "auto" and backend != MOSS_LOCAL and importlib.util.find_spec("transformers") is None:
        return None  # a `[cloud]`-only install has no transformers: off, not a warning every run
    if backend != MOSS_LOCAL and not gpu_allowed(gpu_tier, device):
        # asked for the CPU: there, quietly; otherwise (no usable card, an Intel card) the aligner asks for the
        # device and `resolve_device` / its xpu branch say it runs on the CPU
        asked_cpu = str(device or "").strip().lower() == "cpu" or str(gpu_tier or "").strip().lower() == "cpu"
        return AlignerUse("cpu" if asked_cpu else str(device or "cuda"))
    english = value == "on"
    whisper = (backend in SEGMENT_BACKENDS and (english or lang != "en")
               and importlib.util.find_spec("faster_whisper") is not None)
    return AlignerUse("cuda", whisper=whisper, english=english and whisper)


def gpu_allowed(gpu_tier: str | None, device: str | None) -> bool:
    """Whether this run may put the aligner on a GPU: not `--device cpu`, not `--gpu-tier cpu`, a card torch can use
    (`device.cuda_usable`), and transformers installed (a `[cloud]`-only install has none: off, not a warning a run)."""
    import importlib.util

    if not str(device or "cuda").strip().lower().startswith("cuda") or importlib.util.find_spec("transformers") is None:
        return False
    from ..runtime.device import cuda_usable
    from ..runtime.resources import get_resource_profile

    return bool(get_resource_profile(gpu_tier, device=device).gpu) and cuda_usable()


# --- calling ---------------------------------------------------------------------------------------------


class _Uploads:
    """Wraps callers so that the first upload through any of them says once that the audio leaves this machine;
    `sent`: whether one did (a transcript served wholly from the cache sent nothing)."""

    def __init__(self, backend: str):
        self.backend = backend
        self.sent = False
        self._lock = threading.Lock()  # Live's first calls come from several threads

    def __call__(self, caller: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                first, self.sent = not self.sent, True
            if first:
                current_reporter().warning("cloud-asr-upload", f"音频发给 {_PROVIDER_NAMES[self.backend]} 转写",
                                           impact="音频离开本机")
            return caller(*args, **kwargs)

        return call


def _single_key(env_name: str, env: Mapping[str, str]) -> str | None:
    entries = api_keys.key_entries(env_name, env)
    return entries[0].key if entries else None


def _mai2_route(env: Mapping[str, str]):
    """(via, caller) of the first route configured, in this order: Vercel AI Gateway, OpenRouter, the user's Azure
    resource; None when none is. Vercel leads (owner 2026-11-19): it comes with free credit and past that charges
    what OpenRouter does, its fee a little lower, so it is never the worse choice."""
    from . import mai2

    for via, env_name in (("vercel", mai2.VERCEL_KEY_ENV), ("openrouter", mai2.OPENROUTER_KEY_ENV)):
        key = _single_key(env_name, env)
        if key:
            return via, mai2.Caller(via, key)
    endpoint = os.getenv(AZURE_ENDPOINT_ENV) or env.get(AZURE_ENDPOINT_ENV)
    key = _single_key(AZURE_KEY_ENV, env)
    if endpoint and key:
        return "azure", mai2.Caller("azure", key, endpoint)
    return None


def _mai2_caller(env: Mapping[str, str]):
    """`_mai2_route`, or the error that says which keys MAI2 takes."""
    from . import mai2

    route = _mai2_route(env)
    if route is None:
        raise JointAsrError(
            f"MAI2 needs {mai2.VERCEL_KEY_ENV} (Vercel AI Gateway), {mai2.OPENROUTER_KEY_ENV} (OpenRouter), or "
            f"{AZURE_ENDPOINT_ENV} and {AZURE_KEY_ENV} (your Azure Speech resource), in .env")
    return route


class _Mai2Bill:
    """MAI2's caller for one source: the route's key read, and the bill estimated, when the first request goes out
    -- a source the cache holds needs neither. `announce` is called before each request, once the key is in hand."""

    def __init__(self, duration: float, repair: str, announce: Callable[[], None] = lambda: None):
        self.duration, self.repair, self._announce = duration, repair, announce
        self._route = None

    def __call__(self, audio: Path, *, locale: str | None) -> dict:
        if self._route is None:
            self._route = _mai2_caller(api_keys.read_dotenv())
            hours = self.duration / 3600
            more = (f", about {MAI2_REPAIR_SHARE:.0%} more for re-asking gaps (--asr-repair off: none)"
                    if self.repair != "off" else "")
            print(  # product output
                f"Warning: MAI2 ({self._route[0]}) bills per audio hour: {hours:.2f} h, about "
                f"${hours * MAI2_USD_PER_HOUR:.2f} at ${MAI2_USD_PER_HOUR:.2f}/h (2026-10 price){more}, less what is "
                "cached", file=sys.stderr)
        self._announce()
        return self._route[1](audio, locale=locale)


def transcribe(backend: str, source: Path, *, cache_root: Path, language: str | None, duration: float,
               repair: str, aligner: AlignerUse | None, gpu_tier: str | None = None, device: str | None = None,
               speech: SpeechSpans | None = None, offline: bool = False,
               answer: dict | None = None, uploads: _Uploads | None = None):
    """(transcript, what the metadata records about the route). `gpu_tier` and `device` are the run's, for local
    MOSS; `speech` the source's PCM and Silero's verdicts, shared with the caller's line timing (the MOSS backends').
    The word backends also take `offline` (the cache only: a miss is `CacheMiss`) and `answer` (one answer over the
    whole source, from a transcript file; no cache, no call). `uploads` says whether a request went out."""
    uploads = uploads or _Uploads(backend)
    if backend == MAI2:
        from . import mai2

        # the route a request would take; with none configured, the one the cache was made with (a source the cache
        # holds needs no key), else Vercel's
        made = None if answer is not None else recorded(cache_root, MAI2, audio_fingerprint(source)).get("via")
        route = None if offline or answer is not None else _mai2_route(api_keys.read_dotenv())
        via = made if offline else None if answer is not None else route[0] if route else made or "vercel"
        transcript = mai2.transcribe(source, cache_root=cache_root, language=language, via=via or "", repair=repair,
                                     caller=_Mai2Bill(duration, repair, uploads(lambda: None)), duration=duration,
                                     offline=offline, answer=answer, speech=speech)
        return transcript, ({"via": via} if via else {})
    if backend == GEMINI_35T:
        from . import gemini35t
        from .live import Live

        return gemini35t.transcribe(source, cache_root=cache_root, language=language, repair=repair,
                                    caller=gemini35t.Caller(announce=uploads(lambda: None)),
                                    live=Live(announce=uploads(lambda: None)), duration=duration,
                                    aligner=aligner, offline=offline, answer=answer, speech=speech), {}
    if backend == MOSS_PRO:
        from . import mosspro

        caller = mosspro.Caller(announce=uploads(lambda: None))
        return mosspro.transcribe(source, cache_root=cache_root, language=language, caller=caller,
                                  duration=duration, speech=speech), {}
    if backend == SCRIBE:
        from . import scribe

        from .scribe_keys import KeyPool

        more = (", plus a re-ask where its timestamps collapse or it returns a stretch of talk as one word "
                "(--asr-repair off: none)" if repair != "off" else "")
        return scribe.transcribe(source, cache_root=cache_root, language=language, repair=repair,
                                 caller=KeyPool(note=more, announce=uploads(lambda: None)), duration=duration,
                                 offline=offline, answer=answer, speech=speech), {}
    if backend == MOSS_LOCAL:
        from . import moss_local

        return moss_local.transcribe(source, language=language, duration=duration, gpu_tier=gpu_tier,
                                     device=device, speech=speech)
    raise ValueError(f"not a joint ASR backend: {backend!r}")


# --- moss-diarization-aligner's lines ---------------------------------------------------------------------

#: A transcript file whose words start past the audio's end in more than this share is probably another source's.
LATE_WORDS = 0.05


def rows_backend(value: str | None) -> str:
    """`--asr-rows` (resolved through `[asr] rows`) as a backend name: one of ROWS_BACKENDS, aliases taken."""
    if not value:
        raise JointAsrError(f"--asr {MOSS_DIAR_ALIGNER} needs --asr-rows {'|'.join(ROWS_BACKENDS)} (or [asr] rows): "
                            "the backend its lines come from")
    try:
        name = resolve_backend(value)
    except ValueError:
        name = value
    if name not in ROWS_BACKENDS:
        raise JointAsrError(f"--asr-rows must be one of {', '.join(ROWS_BACKENDS)} (got {value!r})")
    return name


@dataclass(frozen=True)
class RowsInput:
    """Where the word backend's transcript is read from: the task's cache or its API (no `path`), another task's
    cache directory in place (`offline`), or one answer from a file (`answer`). `source` is what the metadata's
    `rows_source` records for a path."""

    path: Path | None = None
    cache_root: Path | None = None
    offline: bool = False
    answer: dict | None = None
    source: dict | None = None


def rows_input(path: str | Path | None, rows: str, source: Path) -> RowsInput:
    """`--asr-transcript` read for `rows`: a directory is an earlier task's `<stem>-cloud/` (or its `<rows>/` inside
    it), read where it is and never written; a file is one answer in `rows`' shape. A directory holding no answers of
    this audio fails here, before anything else runs."""
    if path is None:
        return RowsInput()
    given = Path(path).expanduser()
    if not given.exists():
        raise JointAsrError(f"--asr-transcript {given} does not exist")
    if given.is_dir():
        root = given.parent if given.name == rows and (given / IDENTITY_FILE).exists() else given
        if not recorded(root, rows, audio_fingerprint(source)):
            what = ("holds no answers of " + rows) if not (root / rows / IDENTITY_FILE).exists() else \
                   f"holds {rows}'s answers for another recording (its audio fingerprint differs)"
            raise JointAsrError(f"--asr-transcript {given} {what}; give the <stem>-cloud directory of a task on this "
                                "same source, or a transcript file")
        return RowsInput(given, root, offline=True, source={"dir": str(given)})
    try:
        answer = json.loads(given.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise JointAsrError(f"--asr-transcript {given} cannot be read as JSON: {error}") from error
    if not isinstance(answer, dict):
        raise JointAsrError(f"--asr-transcript {given} is not a JSON object: not {rows}'s answer")
    return RowsInput(given, answer=answer, source={"file": str(given), "fingerprint": audio_fingerprint(given)})


# --- the stage -------------------------------------------------------------------------------------------


def _transcribed_lines(backend: str, src: Path, *, cache_root: Path, language: str | None, duration: float,
                       params: seg.SplitParams, repair: str, aligner: AlignerUse | None, gpu_tier: str | None,
                       device: str | None, speech: SpeechSpans | None,
                       given: RowsInput = RowsInput(),
                       uploads: _Uploads | None = None) -> tuple[list[dict], list, dict[str, Any]]:
    """The lines and VAD built from the transcript, and what `metadata.joint_asr` records. `given`: where a word
    backend's transcript is read from (`rows_input`)."""
    try:
        transcript, route = transcribe(backend, src, cache_root=given.cache_root or cache_root, language=language,
                                       duration=duration, repair=repair, aligner=aligner, gpu_tier=gpu_tier,
                                       device=device, speech=speech,
                                       offline=given.offline, answer=given.answer, uploads=uploads)
    except CacheMiss as error:
        if not given.offline:
            raise
        raise JointAsrError(f"--asr-transcript {given.path} lacks an answer this run needs ({error}): it does not "
                            f"hold {backend}'s whole transcript of this source") from error
    except Exception as error:  # noqa: BLE001 -- reading a file makes no call: what fails is its shape
        if given.answer is None:
            raise
        raise JointAsrError(f"--asr-transcript {given.path} does not read as {backend}'s answer: {error!r}") from error
    units = [u.as_dict() for u in transcript.units()]
    if given.answer is not None:
        if not any(u["text"].strip() for u in units):  # another backend's shape reads as words with no text
            raise JointAsrError(f"--asr-transcript {given.path} holds no {backend} words: is --asr-rows the "
                                "backend whose answer it is?")
        late = sum(u["start"] >= duration for u in units)
        if late > LATE_WORDS * len(units):
            print(f"Warning: {late} of the {len(units)} words in --asr-transcript {given.path} start after the "  # product output
                  f"audio's end ({duration:.0f} s): is it this source's transcript?", file=sys.stderr)
    record: dict[str, Any] = {"backend": backend, "model": transcript.model, **route}
    # absent: off, so an `off` product is what it was before the option; the MOSS backends have no repair
    if backend not in SEGMENT_BACKENDS and repair != "off":
        record["repair"] = repair
    from . import sbd

    boundary = sbd.loaded()  # None: switched off, or it could not load (warned)
    sentence_ends = boundary and functools.partial(sbd.sentence_ends, boundary)
    record["sentence_boundary_model"] = boundary is not None
    clauses = joint_lines.clause_bonus(language)
    if clauses:
        record["clause_bonus"] = clauses
    if backend in SEGMENT_BACKENDS:
        from . import moss_timing

        pcm = speech.pcm()
        align = None
        failed: list[str] = []
        if aligner:
            from .aligner import aligned_units

            def align(long_ones: list[dict], every: list[dict]) -> dict[int, list[dict]]:
                try:
                    return aligned_units(long_ones, all_segments=every, pcm16k=pcm, device=aligner.device,
                                         retime=aligner.retime(language), record=record)
                except Exception as error:  # noqa: BLE001 -- S5b, as for a segment the aligner gives up on
                    failed.append(str(error)[:300])
                    current_reporter().warning("joint-aligner", f"强制对齐失败（{error}），长段改用 Silero 定时")
                    return {}

        timed, counts = moss_timing.timed(units, speech.probs(),
                                          duration=duration, params=params, align=align)
        segments, vad = joint_lines.segment_lines(timed, duration=duration, params=params, language=language,
                                                  sentence_ends=sentence_ends, clauses=clauses)
        record.update(timing=TIMING[bool(aligner) and not failed], **counts)
        if failed:
            record["aligner_error"] = failed[0]
    else:
        segments, vad = joint_lines.word_lines(units, duration=duration, params=params, language=language,
                                               per_turn=backend == GEMINI_35T, grid=backend == GEMINI_35T,
                                               sentence_ends=sentence_ends, clauses=clauses)
        pads = {MAI2: (joint_lines.MAI2_EDGE_PAD, joint_lines.MAI2_EDGE_PAD)}
        if backend == SCRIBE:
            from .scribe import START_PAD

            pads[SCRIBE] = (START_PAD, 0.0)
        if backend in pads:
            start, end = pads[backend]
            segments = joint_lines.widened(segments, pad=start, end_pad=end, duration=duration)
            vad = joint_lines.derived_vad(sorted((w for s in segments for w in s.get("words") or []),
                                                 key=lambda w: w["start"]))
            record["edge_pad"] = [start, end]
    if transcript.log:
        record["log"] = transcript.log
    return segments, vad, record


def run(
    source: str | Path,
    *,
    backend: str,
    output_path: str | Path,
    cache_root: str | Path,
    language: str | None,
    aligner: str | None = None,
    repair: str | None = None,
    split_length_scale: float | None = None,
    gpu_tier: str | None = None,
    device: str | None = None,
    rows: str | None = None,
    transcript: str | Path | None = None,
    overlap_cue: bool | None = None,
    diar_record: str | Path | None = None,
) -> Path:
    """Transcribe `source` with `backend` and write the aligned artifact to `output_path`. `aligner` and `repair`
    are `--asr-aligner` and `--asr-repair` (None: `[asr] aligner` / `[asr] repair`); `gpu_tier` and `device` are the
    run's (local MOSS).
    `moss-diarization-aligner` only: `rows`, `transcript` and `overlap_cue` are `--asr-rows`, `--asr-transcript` and
    `--asr-overlap-cue` (None: `[asr] rows` / none / `[asr] overlap_cue`); `diar_record` is where its
    `<stem>-moss-diar.json` goes (None: not written)."""
    from ...media.ffmpeg import probe_media_duration
    from .mosspro import SpeechSpans

    src, out = Path(source), Path(output_path)
    forced = backend == MOSS_DIAR_ALIGNER
    words_from = rows_backend(settings.ASR_ROWS.resolve(rows)) if forced else backend
    given = rows_input(transcript, words_from, src) if forced else RowsInput()
    if forced:  # before step 1: rows from an API would be paid for and then not labelled
        from . import moss_local

        if (reason := moss_local.unusable_reason(backend, gpu_tier, device)) is not None:
            raise JointAsrError(reason)
    duration = probe_media_duration(src)
    params = seg.split_params_for_length_scale(settings.SPLIT_LENGTH_SCALE.resolve(split_length_scale))
    # a transcript file makes no call: the repairs would
    repair = "off" if given.answer is not None else cloud_repair.mode(settings.ASR_REPAIR.resolve(repair))
    aligner = uses_aligner(settings.ASR_ALIGNER.resolve(aligner), words_from, language, gpu_tier=gpu_tier,
                           device=device)
    # Decoded once, on first use: MOSS Pro's gap check and the line timing below read the same samples; the forced
    # decode's parts and voiceprints the same as its word backend's repairs. Otherwise the word backends' repairs
    # keep their own, freed when they return.
    speech = SpeechSpans(src) if backend in SEGMENT_BACKENDS or forced else None
    uploads = _Uploads(words_from)
    try:
        segments, vad, record = _transcribed_lines(
            words_from, src, cache_root=Path(cache_root), language=language, duration=duration, params=params,
            repair=repair, aligner=aligner, gpu_tier=gpu_tier, device=device, speech=speech,
            given=given, uploads=uploads)
    finally:
        if aligner:  # its GPU memory back before local MOSS (this task's or the next batch item's) counts it
            from .aligner import release

            release()
    if forced:
        segments, record = _diarization_aligned(
            src, segments, record, given=given, uploads=uploads, language=language, duration=duration, params=params,
            gpu_tier=gpu_tier, device=device, speech=speech,
            overlap_cue=bool(settings.ASR_OVERLAP_CUE.resolve(overlap_cue)),
            diar_record=Path(diar_record) if diar_record else None)
    payload = {
        "segments": segments,
        "vad_timeline": {"intervals": vad, "pause_hints": {"scorer": [], "padding": []}},
        "metadata": {
            "vad": {"backend": f"{words_from}-units", "merge_gap_sec": joint_lines.VAD_GAP},
            "asr_align": {"backend": backend, "language": language, "audio_duration": duration,
                          "segment_split": seg.split_params_metadata(params),
                          "seam_pause_sec": joint_lines.SEAM_PAUSE},
            METADATA_KEY: record,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    current_reporter().debug("joint asr", {"backend": backend, **joint_lines.summary(segments)})
    return out


def _diarization_aligned(src: Path, lines: list[dict], rows_record: dict[str, Any], *, given: RowsInput,
                         uploads: _Uploads, language: str | None, duration: float, params: seg.SplitParams,
                         gpu_tier: str | None, device: str | None, speech: SpeechSpans, overlap_cue: bool,
                         diar_record: Path | None) -> tuple[list[dict], dict[str, Any]]:
    """`moss-diarization-aligner`'s steps 2 and 3 over the word backend's `lines`: local MOSS's speaker for each
    line (`moss_local.label_rows`), the lines relabelled (`lines.relabelled`) and, with `overlap_cue`, one person's
    pieces around an interjection joined (`lines.overlap_cues`). -> (lines, `metadata.joint_asr`). `diar_record`
    gets the steps' record (`<stem>-moss-diar.json`): written when the decode is done and again with the merges;
    nothing reads it back."""
    from . import moss_local

    got = moss_local.label_rows(src, lines, language=language, gpu_tier=gpu_tier, device=device, speech=speech)
    rows = dict(rows_record)
    record: dict[str, Any] = {
        "backend": MOSS_DIAR_ALIGNER, "model": moss_local.MODEL_NAME, "revision": got.revision,
        "rows": rows.pop("backend"), "rows_model": rows.pop("model"),
        "rows_source": given.source or ("api" if uploads.sent else "cache"), **rows,
        "parts": [[p["lo"], p["hi"], p["kv"]] for p in got.parts], "unlabelled": got.unlabelled,
    }
    if got.unlabelled:
        current_reporter().warning("moss-diar-unlabelled", f"{got.unlabelled} 行没有落进任何一段，说话人记作未知",
                                   impact="这些行没有说话人")
    kept = {"rows": lines, "parts": got.parts, "voice_merges": [list(m) for m in got.merges],
            "labels": [{"row": i, "speaker": s, "margin": m} for i, (s, m) in enumerate(zip(got.speakers, got.margins))]}
    _write_record(diar_record, kept)
    out = joint_lines.relabelled([dict(line, _row=i) for i, line in enumerate(lines)], got.speakers, got.margins,
                                 duration=duration)
    merges: list[list[int]] = []
    if overlap_cue:
        order = [line["_row"] for line in out]
        out, joined = joint_lines.overlap_cues(out, params=params)
        merges = [[order[k] for k in ids] for ids in joined]
    record["overlap_cues"] = len(merges)
    for line in out:
        line.pop("_row", None)
    _write_record(diar_record, {**kept, "overlap_cues": merges})
    return out, record


def _write_record(path: Path | None, data: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)

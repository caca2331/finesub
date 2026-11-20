"""The recognition stage of a joint-ASR task: the source's mix -> one provider -> `-aligned.json`.

Takes the place of separation, VAD and Whisper for a task whose `--asr` is a joint backend. For a cloud backend the
local work is light: decoding the audio, Silero frame probabilities (the MOSS segments' unit times and MOSS Pro's
gap gate), the word backends' sentence-boundary model, production's DP and the timeline contract. Local MOSS decodes
on this machine's GPU (`moss_local`). The aligned artifact records the backend (`metadata.joint_asr`) and is what
makes a task a joint-ASR task from then on (`task_backend`).
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ... import api_keys, settings
from ...reporting import current_reporter
from ..postprocessing import segmentation as seg
from . import GEMINI_35T, LOCAL, MAI2, MOSS_LOCAL, MOSS_PRO, SCRIBE, SEGMENT_BACKENDS, resolve_backend
from . import lines as joint_lines
from . import relink
from . import repair as cloud_repair

if TYPE_CHECKING:
    from .mosspro import SpeechSpans

#: `metadata` key of a joint-ASR task's aligned and stable artifacts.
METADATA_KEY = "joint_asr"
#: The S5b / S6b names `docs/joint-asr.md` uses, recorded with the MOSS segments' lines.
TIMING = {False: "S5b", True: "S6b"}
#: `--asr-aligner`. `auto` (owner 2026-11-11): local MOSS always, its runs already hold the card; MOSS Pro and 35t's
#: repairs when this run may use a GPU, where it costs 20-30 s an audio hour (about 3 min on the CPU). Either only in a
#: language the aligner was trained on (`--language` not given counts as one); local MOSS takes 50-odd.
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


def uses_aligner(value: str, backend: str, language: str | None, *, gpu_tier: str | None = None,
                 device: str | None = None) -> bool:
    """Whether `--asr-aligner` `value` times this backend's units with the forced aligner; `gpu_tier` and `device`
    are the run's (`auto` on MOSS Pro and 35t asks whether they let it use a GPU)."""
    value = value.strip().lower()
    if value not in ALIGNER_MODES:
        raise ValueError(f"--asr-aligner must be one of {', '.join(ALIGNER_MODES)} (got {value!r})")
    if backend not in (*SEGMENT_BACKENDS, GEMINI_35T):
        return False
    if value == "auto":
        if language and language.lower().split("-")[0] not in ALIGNER_LANGUAGES:
            return False
        return backend == MOSS_LOCAL or gpu_allowed(gpu_tier, device)
    return value == "on"


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


def _announcer(backend: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wraps callers so that the first upload through any of them says once that the audio leaves this machine."""
    said = False
    lock = threading.Lock()  # Live's first calls come from several threads

    def wrap(caller: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            nonlocal said
            with lock:
                first, said = not said, True
            if first:
                current_reporter().warning("cloud-asr-upload", f"音频发给 {_PROVIDER_NAMES[backend]} 转写",
                                           impact="音频离开本机")
            return caller(*args, **kwargs)

        return call

    return wrap


def _announcing(backend: str, caller: Callable[..., Any]) -> Callable[..., Any]:
    return _announcer(backend)(caller)


def _single_key(env_name: str, env: Mapping[str, str]) -> str | None:
    entries = api_keys.key_entries(env_name, env)
    return entries[0].key if entries else None


def _mai2_caller(env: Mapping[str, str]):
    """The first route configured, in this order: Vercel AI Gateway, OpenRouter, the user's Azure resource. Vercel
    leads (owner 2026-11-19): it comes with free credit and past that charges what OpenRouter does, its fee a little
    lower, so it is never the worse choice."""
    from . import mai2

    for via, env_name in (("vercel", mai2.VERCEL_KEY_ENV), ("openrouter", mai2.OPENROUTER_KEY_ENV)):
        key = _single_key(env_name, env)
        if key:
            return via, mai2.Caller(via, key)
    endpoint = os.getenv(AZURE_ENDPOINT_ENV) or env.get(AZURE_ENDPOINT_ENV)
    key = _single_key(AZURE_KEY_ENV, env)
    if endpoint and key:
        return "azure", mai2.Caller("azure", key, endpoint)
    raise JointAsrError(
        f"--asr mai2 needs {mai2.VERCEL_KEY_ENV} (Vercel AI Gateway), {mai2.OPENROUTER_KEY_ENV} (OpenRouter), or "
        f"{AZURE_ENDPOINT_ENV} and {AZURE_KEY_ENV} (your Azure Speech resource), in .env")


def transcribe(backend: str, source: Path, *, cache_root: Path, language: str | None, duration: float,
               repair: str, aligner: bool, gpu_tier: str | None = None, device: str | None = None,
               speech: SpeechSpans | None = None, merge_speakers: bool = False):
    """(transcript, what the metadata records about the route). `gpu_tier` and `device` are the run's, for local
    MOSS; `speech` the source's PCM and Silero's verdicts, shared with the caller's line timing (the MOSS backends');
    `merge_speakers` is `--asr-speaker-merge` resolved, read by MAI2 only."""
    if backend == MAI2:
        from . import mai2

        via, caller = _mai2_caller(api_keys.read_dotenv())
        hours = duration / 3600
        more = (f", about {MAI2_REPAIR_SHARE:.0%} more for re-asking gaps (--asr-repair off: none)"
                if repair != "off" else "")
        print(  # product output
            f"Warning: MAI2 ({via}) bills per audio hour: {hours:.2f} h, about "
            f"${hours * MAI2_USD_PER_HOUR:.2f} at ${MAI2_USD_PER_HOUR:.2f}/h (2026-10 price){more}, unless cached",
            file=sys.stderr)
        transcript = mai2.transcribe(source, cache_root=cache_root, language=language, via=via, repair=repair,
                                     caller=_announcing(backend, caller), duration=duration,
                                     merge_speakers=merge_speakers)
        return transcript, {"via": via}
    if backend == GEMINI_35T:
        from . import gemini35t
        from .live import Live

        announced = _announcer(backend)
        return gemini35t.transcribe(source, cache_root=cache_root, language=language, repair=repair,
                                    caller=announced(gemini35t.Caller()), live=announced(Live()),
                                    duration=duration, aligner=aligner), {}
    if backend == MOSS_PRO:
        from . import mosspro

        caller = mosspro.Caller(announce=_announcing(backend, lambda: None))
        return mosspro.transcribe(source, cache_root=cache_root, language=language, caller=caller,
                                  duration=duration, speech=speech), {}
    if backend == SCRIBE:
        from . import scribe

        from .scribe_keys import KeyPool

        more = (", plus a re-ask where its timestamps collapse or it returns a stretch of talk as one word "
                "(--asr-repair off: none)" if repair != "off" else "")
        return scribe.transcribe(source, cache_root=cache_root, language=language, repair=repair,
                                 caller=KeyPool(note=more, announce=_announcing(backend, lambda: None)),
                                 duration=duration), {}
    if backend == MOSS_LOCAL:
        from . import moss_local

        return moss_local.transcribe(source, language=language, duration=duration, gpu_tier=gpu_tier,
                                     device=device, speech=speech)
    raise ValueError(f"not a joint ASR backend: {backend!r}")


# --- the stage -------------------------------------------------------------------------------------------


def _transcribed_lines(backend: str, src: Path, *, cache_root: Path, language: str | None, duration: float,
                       params: seg.SplitParams, repair: str, aligner: bool, gpu_tier: str | None, device: str | None,
                       speech: SpeechSpans | None, merge_speakers: bool) -> tuple[list[dict], list, dict[str, Any]]:
    """The lines and VAD built from the transcript, and what `metadata.joint_asr` records."""
    merge_speakers = merge_speakers and backend == MAI2
    transcript, route = transcribe(backend, src, cache_root=cache_root, language=language, duration=duration,
                                   repair=repair, aligner=aligner, gpu_tier=gpu_tier, device=device, speech=speech,
                                   merge_speakers=merge_speakers)
    units = [u.as_dict() for u in transcript.units()]
    record: dict[str, Any] = {"backend": backend, "model": transcript.model, **route}
    # absent: off, so an `off` product is what it was before the option; the MOSS backends have no repair
    if backend not in SEGMENT_BACKENDS and repair != "off":
        record["repair"] = repair
    if merge_speakers:  # absent: not merged, as before the option
        record["speaker_merge"] = "on"
    if backend in SEGMENT_BACKENDS:
        from . import moss_timing

        pcm = speech.pcm()
        align = None
        failed: list[str] = []
        if aligner:
            from .aligner import aligned_units

            def align(long_ones: list[dict], every: list[dict]) -> dict[int, list[dict]]:
                try:
                    return aligned_units(long_ones, all_segments=every, pcm16k=pcm)
                except Exception as error:  # noqa: BLE001 -- S5b, as for a segment the aligner gives up on
                    failed.append(str(error)[:300])
                    current_reporter().warning("joint-aligner", f"强制对齐失败（{error}），长段改用 Silero 定时")
                    return {}

        timed, counts = moss_timing.timed(units, speech.probs(),
                                          duration=duration, params=params, align=align)
        segments, vad = joint_lines.segment_lines(timed, duration=duration, params=params, language=language)
        record.update(timing=TIMING[aligner and not failed], **counts)
        if failed:
            record["aligner_error"] = failed[0]
    else:
        from . import sbd

        boundary = sbd.loaded()  # None: switched off, or it could not load (warned)
        clauses = joint_lines.clause_bonus(language)
        segments, vad = joint_lines.word_lines(units, duration=duration, params=params, language=language,
                                               per_turn=backend == GEMINI_35T, grid=backend == GEMINI_35T,
                                               sentence_ends=boundary and functools.partial(sbd.sentence_ends, boundary),
                                               clauses=clauses)
        if clauses:
            record["clause_bonus"] = clauses
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
        record["sentence_boundary_model"] = boundary is not None
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
    speaker_merge: str | None = None,
    split_length_scale: float | None = None,
    gpu_tier: str | None = None,
    device: str | None = None,
) -> Path:
    """Transcribe `source` with `backend` and write the aligned artifact to `output_path`. `aligner`, `repair` and
    `speaker_merge` are `--asr-aligner`, `--asr-repair` and `--asr-speaker-merge` (None: `[asr] aligner` /
    `[asr] repair` / `[asr] speaker_merge`); `gpu_tier` and `device` are the run's (local MOSS)."""
    from ...media.ffmpeg import probe_media_duration
    from .mosspro import SpeechSpans

    src, out = Path(source), Path(output_path)
    duration = probe_media_duration(src)
    params = seg.split_params_for_length_scale(settings.SPLIT_LENGTH_SCALE.resolve(split_length_scale))
    repair = cloud_repair.mode(settings.ASR_REPAIR.resolve(repair))
    merge_speakers = relink.speaker_merge(settings.ASR_SPEAKER_MERGE.resolve(speaker_merge))
    aligner = uses_aligner(settings.ASR_ALIGNER.resolve(aligner), backend, language, gpu_tier=gpu_tier, device=device)
    # Decoded once, on first use: MOSS Pro's gap check and the line timing below read the same samples. Only
    # the segment backends time lines from it; the word backends' repairs keep their own, freed when they return.
    speech = SpeechSpans(src) if backend in SEGMENT_BACKENDS else None
    try:
        segments, vad, record = _transcribed_lines(
            backend, src, cache_root=Path(cache_root), language=language, duration=duration, params=params,
            repair=repair, aligner=aligner, gpu_tier=gpu_tier, device=device, speech=speech,
            merge_speakers=merge_speakers)
    finally:
        if aligner:  # its GPU memory back before the next batch item's local MOSS counts it against its cap
            from .aligner import release

            release()
    payload = {
        "segments": segments,
        "vad_timeline": {"intervals": vad, "pause_hints": {"scorer": [], "padding": []}},
        "metadata": {
            "vad": {"backend": f"{backend}-units", "merge_gap_sec": joint_lines.VAD_GAP},
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

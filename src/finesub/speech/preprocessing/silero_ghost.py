"""Opt-in silero assist for the energy VAD (--vad-silero-assist).

Energy alone cannot tell residual separation noise from quiet speech -- the
accumulator has already spent that signal (tools/vad_tuning/FINDINGS.md,
appendix K4) -- and it cannot tell an honestly raised floor from creep. A
second, voicing-shaped signal can, and every rule here keeps the two-signal
AND discipline (appendix B1): silero alone never removes speech.

`assist_segments` composes four verdicts over the base detector's output
(calibrated on 144 human-labeled snippets, appendices V2/Z/AB): the
voicing-gated cap un-suppresses creep-suppressed loud speech, the ghost drop
removes intervals with no voicing that are neither loud nor long, the
unvoiced-span carve trims noise prefixes/tails/bridges inside intervals, and
seam restoration gives back base gaps a merge swallowed.

Default off. With the flag off the ASR stage is byte-identical to this module
not existing.

The probabilities normally ride along on the energy VAD's own streaming blocks
(`SileroProbCollector` as an `energy.WaveformObserver`), so the assist adds no
decode pass and no full-waveform buffer. `assist_segments` without `probs=`
still works standalone; it just pays for the audio a second time.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from . import energy
from ..runtime.device import resolve_device

SILERO_HOP = 512  # samples; silero's native frame step at 16 kHz
SILERO_CONTEXT = 64  # samples of the previous frame silero prepends to each one
SILERO_CHUNK_FRAMES = 8192  # extractor batch; result is chunk-size independent

GHOST_SILERO_PEAK_MAX = 0.3
GHOST_PEAK_DB_MAX = 0.0
GHOST_MAX_DROP_SEC = 12.0

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        # silero_vad/model.py calls torch.set_num_threads(1) at import time.
        # That is a process-global setting and it never restores it, so every
        # later torch op in this run would be single-threaded.
        threads = torch.get_num_threads()
        try:
            from silero_vad import load_silero_vad

            _MODEL = load_silero_vad()
        finally:
            torch.set_num_threads(threads)
    return _MODEL


@contextlib.contextmanager
def _tf32_disabled():
    """TF32 costs the extractor's convolutions ~3 mantissa bits, which moved 16
    of 112500 frames across CAP_SIL_THR and 17 across SIL_EVID on a 60-minute
    track. The thresholds were calibrated on the full-precision signal, so the
    reduced precision is not ours to spend. Scoped, because the separator wants
    TF32 on. CUDA-only; the CPU path never touches the CUDA backend flags."""
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    prev = (cudnn.allow_tf32, matmul.allow_tf32)
    cudnn.allow_tf32 = matmul.allow_tf32 = False
    try:
        yield
    finally:
        cudnn.allow_tf32, matmul.allow_tf32 = prev


def resolve_silero_device(device: str) -> str:
    return resolve_device(device or "cpu", context="the silero assist")


class SileroProbStream:
    """Vectorized silero speech probability, sequential over the audio.

    The shipped JIT model is STFT -> 4 conv blocks -> LSTMCell(128) ->
    conv+sigmoid, and only the LSTMCell carries state between frames. So the
    frame-independent half runs as one batch over many frames, and the cell's
    four weight tensors drive a full-sequence ``nn.LSTM`` -- the same recurrence,
    one call instead of one call per 32 ms. Against the per-frame loop that is
    ~21x on CPU with a max probability difference of 1.4e-05.

    ``feed`` may be called repeatedly to score a long file piece by piece. Each
    call carries the LSTM state, the trailing SILERO_CONTEXT samples and any
    sub-hop remainder forward, which makes the concatenated result identical to
    scoring the whole file at once (measured 4.7e-10 over 60 minutes) whatever
    the piece boundaries are. Scoring pieces *independently*, with a fresh state
    per piece, is a different -- and materially worse -- signal; that is what
    this class exists to avoid.

    A remainder shorter than SILERO_HOP at the very end of the audio is dropped,
    as the per-frame loop also dropped the file's tail. Pieces that are whole
    multiples of SILERO_HOP (as the streaming blocks are) avoid one buffer copy.
    """

    def __init__(self, device: str = "cpu", chunk_frames: int = SILERO_CHUNK_FRAMES):
        device = resolve_silero_device(device)
        # deepcopy: .to(device) on the JIT submodules mutates them in place, and
        # _MODEL is a process-global cache.
        inner = copy.deepcopy(_model()._model).to(device).eval()
        self._stft = inner.stft
        self._encoder = inner.encoder
        self._head = inner.decoder.decoder
        cell = inner.decoder.rnn.state_dict()
        self._lstm = torch.nn.LSTM(128, 128).to(device).eval()
        with torch.no_grad():
            self._lstm.weight_ih_l0.copy_(cell["weight_ih"])
            self._lstm.weight_hh_l0.copy_(cell["weight_hh"])
            self._lstm.bias_ih_l0.copy_(cell["bias_ih"])
            self._lstm.bias_hh_l0.copy_(cell["bias_hh"])
        self._device = device
        self._chunk = max(1, int(chunk_frames))
        self.reset()

    def reset(self) -> None:
        """Start a new stream: zero LSTM state and zero context, as silero's own
        ``reset_states`` does."""
        self._state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._ctx = torch.zeros(SILERO_CONTEXT)
        self._pending = torch.zeros(0)

    @torch.no_grad()
    def feed(self, wav: torch.Tensor) -> np.ndarray:
        flat = wav.reshape(-1)
        if self._pending.numel():
            flat = torch.cat([self._pending, flat])
        n = (flat.numel() // SILERO_HOP) * SILERO_HOP
        if n <= 0:
            self._pending = flat.clone()
            return np.zeros(0, dtype=np.float32)

        head = self._ctx
        n_frames = n // SILERO_HOP
        parts: list[torch.Tensor] = []
        guard = _tf32_disabled() if self._device != "cpu" else contextlib.nullcontext()
        with guard:
            for i in range(0, n_frames, self._chunk):
                lo = i * SILERO_HOP
                hi = min(i + self._chunk, n_frames) * SILERO_HOP
                # Every frame is [SILERO_CONTEXT previous | SILERO_HOP new];
                # only a piece's first chunk has no previous samples on hand.
                piece = (
                    torch.cat([head, flat[:hi]])
                    if lo == 0
                    else flat[lo - SILERO_CONTEXT : hi]
                )
                frames = piece.unfold(0, SILERO_HOP + SILERO_CONTEXT, SILERO_HOP)
                frames = frames.contiguous().to(self._device)
                feats = self._encoder(self._stft(frames)).squeeze(-1).unsqueeze(1)
                hidden, self._state = self._lstm(feats, self._state)
                probs = self._head(hidden.permute(1, 2, 0))
                parts.append(probs.reshape(-1).float().cpu())

        self._ctx = flat[n - SILERO_CONTEXT : n].clone()
        self._pending = flat[n:].clone()
        return torch.cat(parts).numpy()


def frame_probs(wav: torch.Tensor, *, device: str = "cpu") -> np.ndarray:
    """Silero speech probability per 512-sample frame of 16 kHz mono audio."""
    return SileroProbStream(device).feed(wav)


class SileroProbCollector:
    """A `energy.WaveformObserver` that scores the VAD's own normalized blocks.

    The energy pass already decodes, resamples and normalizes the file; running
    silero from the same blocks makes the probabilities free of a second pass
    over the audio and keeps the assist inside the streaming = bounded-RAM
    contract.

    STREAM_CORE_SEC * TARGET_SR is a whole number of SILERO_HOP, so blocks land
    on frame boundaries; the stream would carry a ragged seam anyway.
    """

    def __init__(self, device: str = "cpu"):
        self._stream = SileroProbStream(device)
        self._parts: list[np.ndarray] = []
        # Wall time spent scoring, so the caller can report it separately
        # instead of letting it disappear into the VAD's own total.
        self.seconds = 0.0

    def reset(self) -> None:
        self._stream.reset()
        self._parts.clear()

    def feed(self, block: torch.Tensor) -> None:
        started = time.perf_counter()
        self._parts.append(self._stream.feed(block))
        self.seconds += time.perf_counter() - started

    def probs(self) -> np.ndarray:
        if not self._parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._parts)


def drop_ghost_segments(
    segments: Sequence[dict],
    probs: np.ndarray,
    track: energy.VadEnergyTrack,
    *,
    silero_peak_max: float = GHOST_SILERO_PEAK_MAX,
    peak_db_max: float = GHOST_PEAK_DB_MAX,
    max_drop_sec: float = GHOST_MAX_DROP_SEC,
) -> tuple[list[dict], dict]:
    """Pure decision: split ``segments`` into kept and ghosts.

    A segment outside the probability track (audio tail shorter than one silero
    frame) is kept -- absence of evidence never drops speech.
    """
    hop_sil = SILERO_HOP / float(energy.TARGET_SR)
    edb = track.energy_db
    kept: list[dict] = []
    dropped: list[dict] = []
    for seg in segments:
        s, e = float(seg["start"]), float(seg["end"])
        p0 = int(s / hop_sil)
        p1 = max(p0 + 1, int(e / hop_sil))
        window = probs[p0:min(p1, len(probs))]
        sil_peak = float(window.max()) if window.size else 1.0
        i0 = int(s / track.hop_sec)
        i1 = max(i0 + 1, int(e / track.hop_sec))
        frame_slice = edb[i0:min(i1, len(edb))]
        peak_db = float(frame_slice.max().item()) if len(frame_slice) else -100.0
        if (
            (e - s) <= max_drop_sec
            and sil_peak < silero_peak_max
            and peak_db <= peak_db_max
        ):
            dropped.append(
                {"start": round(s, 3), "end": round(e, 3),
                 "silero_peak": round(sil_peak, 3), "peak_db": round(peak_db, 1)}
            )
        else:
            kept.append(seg)
    stats = {
        "dropped": len(dropped),
        "dropped_sec": round(sum(d["end"] - d["start"] for d in dropped), 3),
        "silero_peak_max": silero_peak_max,
        "peak_db_max": peak_db_max,
        "max_drop_sec": max_drop_sec,
        "dropped_intervals": dropped,
    }
    return kept, stats


# ---------------------------------------------------------------------------
# Full assist: everything the two-signal AND buys beyond the ghost drop
# (tools/vad_tuning/FINDINGS.md appendices Z and later; accepted end-to-end
# inside the calibrated churn band).
# ---------------------------------------------------------------------------

CAP_DB = 10.0                 # floor may bind at rolling-min anchor + this
CAP_SIL_THR = 0.5             # ...only where silero sees voicing
CAP_DILATE_RIGHT_SEC = 0.3    # extend the gate after voicing ends. NEVER left:
                              # a symmetric dilation opened the gate before
                              # onset and dragged noise prefixes into segments
ANCHOR_WIN_SEC = 10.0
ANCHOR_BIAS_DB = 3.0
ANCHOR_SMOOTH_ALPHA = 0.1

SIL_EVID = 0.3                # carve evidence: voiced above this...
SIL_EVID_DILATE_SEC = 0.10    # ...with a small symmetric dilation
CARVE_CEILING_DB = 0.0        # spans quieter than this with no evidence carve

SEAM_LOUD_KEEP_DB = -5.0      # a swallowed base gap stays merged only when its
                              # content peaks at least this loud (rescued creep
                              # speech); breath and fillers give the seam back


def _rolling_min_anchor(e: np.ndarray, hop: float) -> np.ndarray:
    """Lightly smoothed rolling minimum + bias: a level speech cannot raise."""
    sm = np.empty_like(e)
    prev = e[0]
    for i in range(len(e)):
        prev += ANCHOR_SMOOTH_ALPHA * (e[i] - prev)
        sm[i] = prev
    w = max(1, int(ANCHOR_WIN_SEC / max(hop, 1e-9)))
    out = np.empty_like(sm)
    from collections import deque

    dq: deque = deque()  # indices, values increasing
    for i in range(len(sm)):
        while dq and sm[dq[-1]] >= sm[i]:
            dq.pop()
        dq.append(i)
        lo = i - w + 1
        while dq[0] < lo:
            dq.popleft()
        out[i] = sm[dq[0]]
    return out + ANCHOR_BIAS_DB


def _voiced_frames(probs: np.ndarray, n: int, hop: float, thr: float,
                   dilate_left: float, dilate_right: float) -> np.ndarray:
    hop_sil = SILERO_HOP / float(energy.TARGET_SR)
    idx = np.clip((np.arange(n) * hop / hop_sil).astype(int),
                  0, max(len(probs) - 1, 0))
    voiced = (probs[idx] >= thr) if len(probs) else np.zeros(n, bool)
    vd = voiced.copy()
    for s in range(1, max(0, int(dilate_right / hop)) + 1):
        vd[s:] |= voiced[:-s]
    for s in range(1, max(0, int(dilate_left / hop)) + 1):
        vd[:-s] |= voiced[s:]
    return vd


def _carve_unvoiced_spans(non_speech: list, e_np: np.ndarray, evid: np.ndarray,
                          hop: float, duration: float) -> list:
    """v33's silero-evidence carve, on the non-speech list convention."""
    lead_in = energy.CARVE_LEAD_IN_SEC
    lead_out = energy.CARVE_LEAD_OUT_SEC
    n = len(e_np)
    speech = []
    prev = 0.0
    for a, b in non_speech:
        if a > prev:
            speech.append((prev, a))
        prev = b
    if prev < duration:
        speech.append((prev, duration))
    extra = []
    for s, e_ in speech:
        i0 = max(0, int(math.ceil(s / hop - 1e-9)))
        i1 = min(n, int(math.ceil(e_ / hop - 1e-9)))
        if i1 <= i0:
            continue
        noise = (~evid[i0:i1]) & (e_np[i0:i1] < CARVE_CEILING_DB)
        j = 0
        while j < len(noise):
            if not noise[j]:
                j += 1
                continue
            k = j
            while k < len(noise) and noise[k]:
                k += 1
            t0, t1 = s + j * hop, s + k * hop
            dur = t1 - t0
            head, tail = j == 0, k == len(noise)
            if head and tail:
                pass
            elif head and dur >= energy.CARVE_MIN_TRIM_SEC + lead_in:
                extra.append((s, t1 - lead_in))
            elif tail and dur >= energy.CARVE_MIN_TRIM_SEC + lead_out:
                extra.append((t0 + lead_out, e_))
            elif not head and not tail and dur >= energy.CARVE_INTERIOR_RUN_SEC:
                c0, c1 = t0 + lead_out, t1 - lead_in
                if c1 - c0 >= energy.CARVE_MIN_SEC:
                    extra.append((c0, c1))
            j = k
    if not extra:
        return list(non_speech)
    merged = sorted(list(non_speech) + extra)
    out = []
    for a, b in merged:
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _restore_seams(intervals: list, base: list, e_np: np.ndarray,
                   hop: float) -> tuple[list, int]:
    """Give the base detector's swallowed gaps back, at their exact bounds."""
    n = len(e_np)
    gaps = [(base[i][1], base[i + 1][0]) for i in range(len(base) - 1)
            if base[i + 1][0] > base[i][1]]
    out = []
    restored = 0
    for s, e_ in intervals:
        cuts = []
        for g0, g1 in gaps:
            if g0 <= s or g1 >= e_:
                continue
            i0 = max(0, int(math.ceil(g0 / hop - 1e-9)))
            i1 = min(n, int(math.ceil(g1 / hop - 1e-9)))
            if i1 <= i0 or float(e_np[i0:i1].max()) >= SEAM_LOUD_KEEP_DB:
                continue
            cuts.append((g0, g1))
            restored += 1
        cur = s
        for c0, c1 in sorted(cuts):
            if c0 > cur:
                out.append((cur, c0))
            cur = max(cur, c1)
        if e_ > cur:
            out.append((cur, e_))
    return out, restored


def assist_segments(
    audio_path: str | Path,
    segments: Sequence[dict],
    track: energy.VadEnergyTrack,
    duration_sec: float,
    *,
    device: str = "cpu",
    probs: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """The full silero assist over the base detector's output.

    1. voicing-gated cap: re-run the scorer with the floor capped at the
       rolling-min anchor + CAP_DB, only under (right-dilated) voicing --
       un-suppresses creep-suppressed loud speech; silero missing = base
       behavior, it can add speech but never remove it
    2. ghost drop: intervals with no voicing, not loud, not long
    3. unvoiced-span carve: noise prefixes/tails/bridges inside intervals
    4. seam restore: base gaps swallowed by the cap come back at their exact
       bounds unless the merge holds rescued loud speech
    """
    t0 = time.perf_counter()
    if track.frame_dbfs is None:
        raise ValueError("silero assist needs a track carrying frame_dbfs")
    device = resolve_silero_device(device)  # once: the fallback warns
    if probs is None:
        # No collector rode along with the VAD, so pay for the audio twice.
        wav = energy._load_asr_audio_streamed(str(audio_path))
        wav = energy.light_normalize(wav, energy.TARGET_SR)
        probs = frame_probs(wav, device=device)
        del wav

    e_t = track.energy_db
    e_np = e_t.detach().cpu().numpy().astype(np.float64)
    n = len(e_np)
    hop = float(track.hop_sec)
    starts = torch.arange(n, dtype=torch.float64) * hop
    ends = starts + float(track.frame_sec)

    floor = energy.estimate_noise_floor_db_local(
        e_t, starts, duration_sec,
        local_window_sec=energy.NOISE_LOCAL_WINDOW_SEC,
        local_hop_sec=energy.NOISE_LOCAL_HOP_SEC,
        local_percentile=energy.NOISE_INIT_PERCENTILE,
        track_gate_db=energy.NOISE_TRACK_GATE_DB,
        follow_alpha=energy.NOISE_TRACK_FOLLOW_ALPHA,
        rise_alpha=energy.NOISE_TRACK_RISE_ALPHA,
        local_blend=energy.NOISE_LOCAL_BLEND,
    ).detach().cpu().numpy().astype(np.float64)

    anchor = _rolling_min_anchor(e_np, hop)
    gate = _voiced_frames(probs, n, hop, CAP_SIL_THR, 0.0, CAP_DILATE_RIGHT_SEC)
    gate &= floor > anchor + CAP_DB
    gated = floor.copy()
    gated[gate] = anchor[gate] + CAP_DB

    raw = energy._score_to_non_speech_intervals(
        e_t, torch.from_numpy(gated.astype(np.float32)), track.frame_dbfs,
        starts, ends, duration_sec,
        enter_margin_db=energy.NON_SPEECH_MARGIN_DB_ENTER,
        weighted=bool(energy.WEIGHTED_INTERVAL),
    )
    ns = energy._apply_negative_padding(raw, duration_sec)
    ns = energy._absorb_low_peak_speech(ns, e_t, duration_sec)
    ns = energy._carve_low_peak_speech(ns, e_t, duration_sec)
    evid = (_voiced_frames(probs, n, hop, SIL_EVID,
                           SIL_EVID_DILATE_SEC, SIL_EVID_DILATE_SEC)
            | (e_np >= CARVE_CEILING_DB))
    ns = _carve_unvoiced_spans(ns, e_np, evid, hop, duration_sec)
    speech = [(float(a), float(b))
              for a, b in energy.invert_intervals(ns, duration_sec) if b > a]

    seg_dicts = [{"start": s, "end": e_} for s, e_ in speech]
    kept, ghost_stats = drop_ghost_segments(seg_dicts, probs, track)
    speech = [(float(x["start"]), float(x["end"])) for x in kept]

    base = [(float(x["start"]), float(x["end"])) for x in segments]
    speech, restored = _restore_seams(speech, base, e_np, hop)

    stats = {
        "backend": "energy+silero-assist",
        "device": device,
        "cap_db": CAP_DB,
        "base_intervals": len(base),
        "intervals": len(speech),
        "ghost_dropped": ghost_stats["dropped"],
        "ghost_dropped_sec": ghost_stats["dropped_sec"],
        "seams_restored": restored,
        "speech_sec": round(sum(b - a for a, b in speech), 1),
        "base_speech_sec": round(sum(b - a for a, b in base), 1),
        "silero_sec": round(time.perf_counter() - t0, 3),
    }
    return [{"start": s, "end": e_} for s, e_ in speech], stats

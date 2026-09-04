"""What the separation stage spends outside the model forward.

E9 left the compiled forward at 278ms/chunk and noted that on a 270s task the
forward is only 9.5s of 22.4s -- from which the queue inferred that "the non-GPU
part is now the big one" and put a dedicated demix runner near the top. That
inference silently equates "not forward" with "per-second CPU work that a better
runner could remove". This module tests that equation.

It needs no GPU and never loads the checkpoint: every step around the forward is
shape-determined, so replaying them on synthetic audio of the production shape
measures the real thing. Three groups are timed separately:

``input``     what finesub and ``prepare_mix`` do before the first chunk
``current``   the post-forward chain exactly as audio-separator + finesub ship it
``proposed``  the same result from a dedicated runner: one stem, no window/counter
              on non-overlapping steps, and no encode/decode round-trip per block

Run it for one block core (600s) and for the 270s protocol clip:

    python -m tools.separator_accel_bench.cpu_chain 270
    python -m tools.separator_accel_bench.cpu_chain 600

Numbers and the conclusion they support are in
``docs/separator-optimization.md`` E13.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from contextlib import contextmanager

import numpy as np
import soundfile as sf
import torch
from scipy import signal

#: The model's own rate; audio-separator resamples any input to it (E12).
SR = 44100
#: ``stft_hop_length * (dim_t - 1)`` from the checkpoint yaml, in samples.
CHUNK = 352800
#: ``req_shape`` uses ``len(training.instruments)``, not ``num_stems`` -- so the
#: CPU buffers are sized for two stems even though this checkpoint declares
#: ``target_instrument: Vocals`` and the model returns one.
STEMS = 2

_RESULTS: list[tuple[str, str, float]] = []
_GROUP = ""


@contextmanager
def step(name: str):
    started = time.perf_counter()
    yield
    elapsed = time.perf_counter() - started
    _RESULTS.append((_GROUP, name, elapsed))
    print(f"  {name:<52} {elapsed:8.3f}s", flush=True)


@contextmanager
def group(name: str):
    global _GROUP
    _GROUP = name
    print(f"\n[{name}]", flush=True)
    yield
    _GROUP = ""


def normalize(wave: np.ndarray, max_peak: float = 0.9) -> np.ndarray:
    """``spec_utils.normalize``, same passes over the same array."""

    peak = np.abs(wave).max()
    if peak > max_peak:
        wave *= max_peak / peak
    return wave


def input_side(mix: np.ndarray, tmpdir: str) -> None:
    """finesub writes a PCM_16 wav; ``prepare_mix`` reads it back with librosa."""

    wav = os.path.join(tmpdir, "block_in.wav")
    with step("sf.write PCM_16 wav (finesub)"):
        sf.write(wav, mix.T, SR, subtype="PCM_16")

    import librosa

    # Split deliberately: librosa is lazily imported and numba-backed, so the
    # first call in a process pays a one-time cost that does not scale with the
    # material and cannot be removed by a better runner.
    with step("librosa.load, first call in process"):
        librosa.load(wav, sr=SR, mono=False)
    with step("librosa.load, steady state"):
        librosa.load(wav, sr=SR, mono=False)
    with step("sf.read for comparison"):
        sf.read(wav, dtype="float32", always_2d=True)


def current_chain(mix: np.ndarray, model_out: torch.Tensor, tmpdir: str) -> None:
    """audio-separator's roformer branch plus ``_append_separated_block``."""

    frames = mix.shape[1]
    with step("normalize(mix) before demix"):
        mix = normalize(mix.copy())
    with step("torch.tensor(mix)  (float32 copy)"):
        torch.tensor(mix, dtype=torch.float32)

    with step("zeros result + counter  (2 stems, full length)"):
        req = (STEMS, 2, frames)
        result = torch.zeros(req, dtype=torch.float32)
        counter = torch.zeros(req, dtype=torch.float32)

    window = torch.tensor(signal.windows.hamming(CHUNK), dtype=torch.float32)
    with step("per-chunk overlap_add + counter"):
        for i in range(0, frames, CHUNK):
            if i + CHUNK > frames:
                start, length = frames - CHUNK, CHUNK
            else:
                start, length = i, min(CHUNK, frames - i)
            result[..., start : start + length] += model_out[..., :length] * window[:length]
            counter[..., start : start + length] += window[:length]

    with step("result / counter.clamp()  (full size)"):
        inferenced = result / counter.clamp(min=1e-10)
    del result, counter

    with step("take stem + .numpy() + .T"):
        primary = inferenced[0].numpy().T
    del inferenced
    with step("normalize(source) after demix"):
        primary = normalize(np.ascontiguousarray(primary.T)).T

    block_out = os.path.join(tmpdir, "block.flac")
    with step("write block: int16 + interleave + ffmpeg flac"):
        _write_audio_pydub(block_out, primary)
    with step("read block back + append to merged flac"):
        _append_block(block_out, os.path.join(tmpdir, "merged.flac"))


def proposed_chain(mix: np.ndarray, model_out: torch.Tensor, tmpdir: str) -> None:
    """One stem, direct writes, block handed over in memory."""

    frames = mix.shape[1]
    with step("normalize(mix) before demix"):
        mix = normalize(mix.copy())
    with step("torch.from_numpy(mix)  (no copy)"):
        torch.from_numpy(mix)
    with step("zeros out  (1 stem, full length)"):
        out = torch.zeros((2, frames), dtype=torch.float32)
    with step("per-chunk direct write"):
        for i in range(0, frames, CHUNK):
            if i + CHUNK > frames:
                out[..., frames - CHUNK :] = model_out
            else:
                out[..., i : i + CHUNK] = model_out[..., : min(CHUNK, frames - i)]
    with step("normalize(source) after demix"):
        arr = normalize(out.numpy())
    with step("write merged flac directly (soundfile)"):
        sf.write(os.path.join(tmpdir, "merged2.flac"), arr.T, SR,
                 format="FLAC", subtype="PCM_16")


def _write_audio_pydub(path: str, stem_source: np.ndarray) -> None:
    """``common_separator.write_audio_pydub``: the operations, not a call into it."""

    from pydub import AudioSegment

    if stem_source.dtype != np.int16:
        stem_source = (stem_source * 32767).astype(np.int16)
    interleaved = np.empty((2 * stem_source.shape[0],), dtype=np.int16)
    interleaved[0::2] = stem_source[:, 0]
    interleaved[1::2] = stem_source[:, 1]
    segment = AudioSegment(interleaved.tobytes(), frame_rate=SR, sample_width=2, channels=2)
    segment.export(path, format="flac", parameters=["-sample_fmt", "s16"])


def _append_block(block_out: str, merged: str, chunk_frames: int = 1 << 20) -> None:
    with sf.SoundFile(block_out, mode="r") as in_f:
        with sf.SoundFile(merged, mode="w", samplerate=in_f.samplerate,
                          channels=in_f.channels, format="FLAC") as out_f:
            remaining = len(in_f)
            while remaining > 0:
                frames = in_f.read(min(remaining, chunk_frames),
                                   dtype="float32", always_2d=True)
                if frames.size == 0:
                    break
                out_f.write(frames)
                remaining -= frames.shape[0]


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 270.0
    frames = int(duration * SR)
    print(f"torch {torch.__version__}  threads {torch.get_num_threads()}")
    print(f"duration {duration:.0f}s  frames {frames}  chunks {-(-frames // CHUNK)}")

    rng = np.random.default_rng(0)
    mix = (rng.standard_normal((2, frames)) * 0.05).astype(np.float32)
    # Real amplitude, not silence: FLAC on digital silence is far too cheap to
    # stand in for a separated vocal track.
    model_out = torch.from_numpy(
        (rng.standard_normal((2, CHUNK)) * 0.05).astype(np.float32)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with group("input"):
            input_side(mix, tmpdir)
        with group("current"):
            current_chain(mix, model_out, tmpdir)
        with group("proposed"):
            proposed_chain(mix, model_out, tmpdir)

    print()
    for name in ("input", "current", "proposed"):
        total = sum(dt for grp, _, dt in _RESULTS if grp == name)
        print(f"{name:<10} {total:8.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

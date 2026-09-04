"""The batched prefetch in front of the alignment loop (A1's assembly).

No model, no GPU: a fake decoder records what it was asked to decode and
returns results the sequential fake would also return, so these tests pin the
contract that matters -- the loop's outcome does not depend on whether a
window came from the batch or from the sequential path, batches are the next
groups in order, misses fall through, and leftovers are counted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.speech.recognition import transcribe as asr_align

SR = 16000


def interval(start: float, end: float) -> dict:
    return {"start": start, "end": end}


class FakeModel:
    """`transcribe_wt` labels a window by its sample count; the fake batched
    decoder below produces the identical label, so a hit and a miss agree."""

    def __init__(self) -> None:
        self.sequential: list[int] = []

    @staticmethod
    def result(audio, language):
        # One distinct word every half second, so every interval of the
        # combined window gets words and no rescue ladder fires; the label
        # carries the window length so a hit and a miss are told apart.
        seconds = len(audio) / SR
        words = []
        cursor = 0.0
        index = 0
        while cursor + 0.5 <= seconds + 1e-6:
            words.append(
                {
                    "word": f"w{len(audio)}_{index}",
                    "start": round(cursor, 3),
                    "end": round(cursor + 0.4, 3),
                    "confidence": 0.9,
                }
            )
            cursor += 0.5
            index += 1
        return {
            "segments": [
                {
                    "text": " ".join(w["word"] for w in words),
                    "start": 0.0,
                    "end": seconds,
                    "words": words,
                    "confidence": 0.9,
                    "no_speech_prob": 0.0,
                    "avg_logprob": -0.1,
                }
            ],
            "language": language or "ja",
        }

    def transcribe_wt(self, audio, **options):
        self.sequential.append(len(audio))
        return self.result(audio, options.get("language"))


class FakeBatchDecoder:
    def __init__(self, fail=False, drop_index=None) -> None:
        self.batches: list[list[int]] = []
        self.languages: list[object] = []
        self.fail = fail
        self.drop_index = drop_index

    def __call__(self, model, audios, **options):
        if self.fail:
            raise RuntimeError("no batch today")
        self.batches.append([len(a) for a in audios])
        self.languages.append(options.get("language"))
        out = [FakeModel.result(a, options.get("language")) for a in audios]
        if self.drop_index is not None and self.drop_index < len(out):
            out[self.drop_index] = None
        return out


def intervals(count: int, *, seconds: float = 2.0, gap: float = 1.0) -> list[dict]:
    items = []
    cursor = 0.0
    for _ in range(count):
        items.append(interval(cursor, cursor + seconds))
        cursor += seconds + gap
    return items


def audio_for(items: list[dict]) -> np.ndarray:
    total = float(items[-1]["end"]) + 5.0
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(total * SR)) * 0.01).astype(np.float32)


def run(items, *, decode_batch, decoder=None, language="ja"):
    model = FakeModel()
    audio = audio_for(items)
    with asr_align.collecting_stats() as stats:
        # Inject the fake driver through the class, the way the loop builds it.
        original = asr_align.DecodePrefetch.__init__

        def init(self, model_, batch_size, decode_fn=None):
            original(self, model_, batch_size, decode_fn=decoder)

        asr_align.DecodePrefetch.__init__ = init
        try:
            out = asr_align.align_segments(
                items,
                audio,
                SR,
                model=model,
                gap_sec=0.3,
                language=language,
                decode_batch=decode_batch,
            )
        finally:
            asr_align.DecodePrefetch.__init__ = original
    return out, model, dict(stats)


def strip(segments):
    return [(round(s["start"], 3), round(s["end"], 3), s["text"]) for s in segments]


class TestPrefetchInTheLoop:
    def test_batched_and_sequential_runs_agree_and_the_batch_is_the_next_groups(self):
        items = intervals(6)
        sequential, model_seq, _ = run(items, decode_batch=1)
        decoder = FakeBatchDecoder()
        batched, model_bat, stats = run(items, decode_batch=4, decoder=decoder)

        assert strip(batched) == strip(sequential)
        # Every group was answered from a batch: the sequential decoder saw nothing.
        assert model_bat.sequential == []
        assert len(model_seq.sequential) == stats["prefetch_hits"]
        assert stats.get("prefetch_misses", 0) == 0
        assert stats.get("prefetch_wasted", 0) == 0
        # Waves of at most 4 groups, in order, and the same windows the
        # sequential path decoded.
        assert all(len(b) <= 4 for b in decoder.batches)
        assert [n for b in decoder.batches for n in b] == model_seq.sequential

    def test_the_explicit_language_reaches_the_batch(self):
        decoder = FakeBatchDecoder()
        run(intervals(3), decode_batch=4, decoder=decoder, language="en")
        assert decoder.languages == ["en"]

    def test_a_failed_batch_falls_through_to_the_sequential_path(self):
        items = intervals(5)
        sequential, _, _ = run(items, decode_batch=1)
        batched, model, stats = run(items, decode_batch=4, decoder=FakeBatchDecoder(fail=True))
        assert strip(batched) == strip(sequential)
        assert len(model.sequential) == 5 or len(model.sequential) >= 1
        assert stats["prefetch_batch_failures"] >= 1
        assert stats["prefetch_misses"] == len(model.sequential)

    def test_an_item_the_driver_could_not_replay_is_decoded_sequentially(self):
        items = intervals(4)
        # `decode_batch=1`, like every other baseline here: the reference this
        # compares against has to be the sequential path. At 4 with no fake
        # driver the baseline pulls in the real `transcribe_batch`, so the run
        # needs the [asr] extra to be green -- and the comparison stops being
        # against anything sequential.
        sequential, _, _ = run(items, decode_batch=1)
        batched, model, stats = run(items, decode_batch=4, decoder=FakeBatchDecoder(drop_index=0))
        assert strip(batched) == strip(sequential)
        assert len(model.sequential) >= 1, "the dropped window decoded sequentially"
        assert stats["prefetch_misses"] >= 1

    def test_batch_of_one_means_no_prefetch(self):
        _, model, stats = run(intervals(3), decode_batch=1, decoder=FakeBatchDecoder())
        assert "prefetch_hits" not in stats and len(model.sequential) >= 1


class TestPrefetchUnit:
    def test_take_is_keyed_on_audio_bytes_and_options(self):
        prefetch = asr_align.DecodePrefetch(FakeModel(), 4, decode_fn=FakeBatchDecoder())
        audio = np.zeros(SR, dtype=np.float32)
        key = prefetch._key(audio, {"language": "ja", "beam_size": None})
        prefetch._results[key] = {"segments": []}
        with asr_align.collecting_stats() as stats:
            assert prefetch.take(audio, {"language": "ja", "beam_size": None}) == {"segments": []}
            assert prefetch.take(audio, {"language": "ja", "beam_size": 5}) is None
            assert prefetch.take(np.ones(SR, dtype=np.float32), {"language": "ja", "beam_size": None}) is None
        assert stats["prefetch_hits"] == 1 and stats["prefetch_misses"] == 2

    def test_leftovers_are_counted_as_waste_on_close(self):
        prefetch = asr_align.DecodePrefetch(FakeModel(), 4, decode_fn=FakeBatchDecoder())
        prefetch._results[("a", ())] = {}
        prefetch._results[("b", ())] = {}
        with asr_align.collecting_stats() as stats:
            prefetch.close()
        assert stats["prefetch_wasted"] == 2 and prefetch._results == {}

    def plan(self, prefetch, groups, audio):
        with asr_align.collecting_stats() as stats:
            prefetch.plan(
                groups,
                remaining=[i for g in groups for i in g],
                successor_start=None,
                audio=audio,
                sr=SR,
                gap_sec=0.3,
                language="ja",
                auto_language_history=[],
                audio_loader=None,
            )
        return dict(stats)

    def test_groups_longer_than_one_window_stay_sequential(self):
        """Measured: batching a long group's first window moves its seek and
        the rest of the group re-slices -- one 97 s group changed 24 segments.
        The gate prices per-window noise, not that cascade."""
        decoder = FakeBatchDecoder()
        prefetch = asr_align.DecodePrefetch(FakeModel(), 4, decode_fn=decoder)
        long_group = [interval(0.0, 45.0)]
        short_group = [interval(50.0, 52.0)]
        audio = np.zeros(int(60 * SR), dtype=np.float32)
        stats = self.plan(prefetch, [long_group, short_group], audio)
        assert len(decoder.batches) == 1 and len(decoder.batches[0]) == 1
        assert stats["prefetch_too_long"] == 1
        # Both are "covered": the loop must not re-plan when it reaches the
        # long one, it simply misses and decodes sequentially.
        assert prefetch.covers(long_group) and prefetch.covers(short_group)

    def test_a_replan_keeps_unchanged_groups_instead_of_decoding_them_twice(self):
        decoder = FakeBatchDecoder()
        prefetch = asr_align.DecodePrefetch(FakeModel(), 4, decode_fn=decoder)
        a = [interval(0.0, 2.0)]
        b = [interval(5.0, 7.0)]
        c = [interval(10.0, 12.0)]
        # Distinct audio per group: the key is the audio bytes.
        audio = audio_for(a + b + c)
        self.plan(prefetch, [a, b], audio)
        stats = self.plan(prefetch, [b, c], audio)  # `a` fell out, `c` came in
        assert [len(batch) for batch in decoder.batches] == [2, 1]
        assert stats["prefetch_wasted"] == 1
        assert len(prefetch._results) == 2

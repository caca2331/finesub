"""Scoring a speech-interval list.

The two errors are not symmetric, so they are never collapsed into one number:

  lost speech   a word the VAD would hide from the ASR. Subtitles disappear. This
                is the failure the user hit with silero, and it is the binding
                constraint -- report `words_lost` before anything else.
  kept silence  audio handed to the ASR that is not speech. Costs decode time and
                drags filled pauses into word timings, but never deletes a line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from refs import Interval, PauseRef, Word, covered


@dataclass
class Score:
    duration: float
    speech_frac: float
    words: int
    words_lost: int          # >=90% of the word inside non-speech
    words_clipped: int       # 10-90% inside non-speech
    clipped_head: int        # of those, the loss is at the word's start
    clipped_tail: int        # of those, the loss is at the word's end
    word_sec_lost: float
    word_sec_total: float
    word_recall: float       # fraction of word-seconds kept
    pause_excluded: float    # fraction of filled_pause blocks mostly non-speech
    onset_excluded: float    # same for word_onset blocks -- must stay near 0
    n_intervals: int

    def line(self, name: str) -> str:
        return (f"{name:<26} speech={self.speech_frac:>5.1%} "
                f"lost={self.words_lost:>4d} clipH={self.clipped_head:>4d} "
                f"clipT={self.clipped_tail:>4d} "
                f"recall={self.word_recall:>6.3%} "
                f"pause_excl={self.pause_excluded:>5.1%} "
                f"onset_excl={self.onset_excluded:>5.1%} "
                f"n={self.n_intervals:>4d}")


def _block_excluded(speech: Sequence[Interval], blocks: Sequence[Interval],
                    frac: float = 0.5) -> float:
    if not blocks:
        return 0.0
    hit = 0
    for s, e in blocks:
        if e <= s:
            continue
        if covered(speech, s, e) / (e - s) < (1.0 - frac):
            hit += 1
    return hit / len(blocks)


def score(speech: Sequence[Interval], words: Sequence[Word], duration: float,
          pause_ref: PauseRef | None = None) -> Score:
    speech = sorted(speech)
    lost = clipped = head = tail = 0
    sec_lost = sec_total = 0.0
    for w in words:
        dur = w.end - w.start
        if dur <= 0:
            continue
        cov = covered(speech, w.start, w.end)
        miss = dur - cov
        sec_total += dur
        sec_lost += miss
        r = miss / dur
        if r >= 0.9:
            lost += 1
        elif r >= 0.1:
            clipped += 1
            # Which end lost the audio. Head clipping is ambiguous evidence: the
            # reference word starts come from a run whose starts are known to be
            # early at pauses, so trimming a pause registers here as damage.
            probe = min(0.05, dur / 3)
            if covered(speech, w.start, w.start + probe) < probe * 0.5:
                head += 1
            elif covered(speech, w.end - probe, w.end) < probe * 0.5:
                tail += 1
    return Score(
        duration=duration,
        speech_frac=sum(e - s for s, e in speech) / max(duration, 1e-9),
        words=len(words),
        words_lost=lost,
        words_clipped=clipped,
        clipped_head=head,
        clipped_tail=tail,
        word_sec_lost=sec_lost,
        word_sec_total=sec_total,
        word_recall=(1.0 - sec_lost / sec_total) if sec_total else 1.0,
        pause_excluded=_block_excluded(speech, pause_ref.filled_pause) if pause_ref else 0.0,
        onset_excluded=_block_excluded(speech, pause_ref.word_onset) if pause_ref else 0.0,
        n_intervals=len(speech),
    )

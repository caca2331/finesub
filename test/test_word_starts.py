"""Word-start correction: [*] block rules and VAD-anchored clamps."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_playground.speech.preprocessing.energy import VadEnergyTrack
from asr_playground.speech.recognition import word_starts


HOP = 0.01


def make_track(
    duration_sec: float,
    quiet_spans: list[tuple[float, float]],
    *,
    loud_db: float = 0.0,
    quiet_db: float = -40.0,
) -> VadEnergyTrack:
    n = int(duration_sec / HOP)
    energy = [loud_db] * n
    for start, end in quiet_spans:
        for i in range(int(start / HOP), min(n, int(end / HOP))):
            energy[i] = quiet_db
    return VadEnergyTrack(
        energy_db=torch.tensor(energy, dtype=torch.float32),
        hop_sec=HOP,
        frame_sec=0.025,
        energy_mode="weighted",
    )


def word(text: str, start: float, end: float, **extra) -> dict:
    return {"word": text, "start": start, "end": end, **extra}


def segment(words: list[dict], **extra) -> dict:
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
        "text": "".join(str(w["word"]) for w in words),
        **extra,
    }


def apply_rules(segments, track):
    return word_starts.apply_disfluency_rules(segments, energy_track=track)


class TestDisfluencyBlocks:
    def test_short_block_merges_into_next_word(self) -> None:
        track = make_track(5.0, [])
        segs = [segment([word("[*]", 1.0, 1.08), word("あ", 1.08, 1.5)])]
        out, stats = apply_rules(segs, track)
        words = out[0]["words"]
        assert [w["word"] for w in words] == ["あ"]
        assert words[0]["start"] == 1.0
        assert words[0]["disfluency_span"] == [1.0, 1.08]
        assert words[0]["disfluency_action"] == "merge_short"
        assert stats == {"merge_short": 1}

    def test_segment_first_quiet_block_deletes_to_energy_onset(self) -> None:
        # Block 1.0-1.5 quiet through 1.4: onset lands at the quiet-run end.
        track = make_track(5.0, [(1.0, 1.4)])
        segs = [segment([word("[*]", 1.0, 1.5), word("あ", 1.5, 2.0)])]
        out, stats = apply_rules(segs, track)
        words = out[0]["words"]
        assert stats == {"delete": 1}
        assert words[0]["disfluency_action"] == "delete"
        assert abs(words[0]["start"] - 1.4) < 0.02
        assert out[0]["start"] == words[0]["start"]
        assert out[0]["text"] == "あ"

    def test_loud_block_merges_even_at_segment_start(self) -> None:
        track = make_track(5.0, [])
        segs = [segment([word("[*]", 1.0, 1.5), word("あ", 1.5, 2.0)])]
        out, stats = apply_rules(segs, track)
        assert stats == {"merge": 1}
        assert out[0]["words"][0]["start"] == 1.0

    def test_mid_phrase_quiet_block_deletes_position_independently(self) -> None:
        # No decode gap, no pause hint: the energy gate alone decides
        # (gold: 0/25 word-onset deletions across all positions).
        track = make_track(5.0, [(1.5, 1.9)])
        segs = [
            segment(
                [
                    word("あ", 1.0, 1.5),
                    word("[*]", 1.5, 2.0),
                    word("い", 2.0, 2.5),
                ]
            )
        ]
        out, stats = apply_rules(segs, track)
        assert stats == {"delete": 1}
        assert abs(out[0]["words"][1]["start"] - 1.9) < 0.02
        assert out[0]["words"][1]["disfluency_span"] == [1.5, 2.0]

    def test_long_move_deletes_without_position_evidence(self) -> None:
        # Mid-phrase, no gap, no hint: the energy gate alone decides even for
        # a >1s move (audited: such blocks were previous-word residuals).
        track = make_track(6.0, [(1.5, 2.7)])
        segs = [
            segment(
                [
                    word("あ", 1.0, 1.5),
                    word("[*]", 1.5, 2.8),
                    word("い", 2.8, 3.3),
                ]
            )
        ]
        out, stats = apply_rules(segs, track)
        assert stats == {"delete": 1}
        assert abs(out[0]["words"][1]["start"] - 2.7) < 0.02

    def test_pathological_move_is_capped_at_three_seconds(self) -> None:
        # Quiet span kept a minority of the ±2s reference window so the local
        # median stays at speech level.
        track = make_track(8.0, [(1.5, 4.9)])
        segs = [segment([word("[*]", 1.0, 5.0), word("あ", 5.0, 5.5)])]
        out, stats = apply_rules(segs, track)
        assert stats == {"delete": 1}
        assert out[0]["words"][0]["start"] == 4.0

    def test_mid_phrase_loud_block_merges(self) -> None:
        track = make_track(5.0, [])
        segs = [
            segment(
                [
                    word("あ", 1.0, 1.5),
                    word("[*]", 1.5, 2.0),
                    word("い", 2.0, 2.5),
                ]
            )
        ]
        out, stats = apply_rules(segs, track)
        assert stats == {"merge": 1}
        assert out[0]["words"][1]["start"] == 1.5

    def test_consecutive_blocks_collapse_into_one_span(self) -> None:
        track = make_track(5.0, [(1.0, 1.9)])
        segs = [
            segment(
                [
                    word("[*]", 1.0, 1.4),
                    word("[*]", 1.4, 2.0),
                    word("あ", 2.0, 2.5),
                ]
            )
        ]
        out, stats = apply_rules(segs, track)
        words = out[0]["words"]
        assert [w["word"] for w in words] == ["あ"]
        assert words[0]["disfluency_span"] == [1.0, 2.0]
        assert stats == {"delete": 1}

    def test_trailing_block_is_dropped_untouched(self) -> None:
        track = make_track(5.0, [])
        segs = [segment([word("あ", 1.0, 1.5), word("[*]", 1.5, 2.0)])]
        out, stats = apply_rules(segs, track)
        assert [w["word"] for w in out[0]["words"]] == ["あ"]
        assert out[0]["words"][0]["start"] == 1.0
        assert out[0]["end"] == 1.5
        assert stats == {"orphan_dropped": 1}

    def test_first_word_merge_is_bounded_by_previous_segment(self) -> None:
        track = make_track(5.0, [])
        segs = [
            segment([word("あ", 0.5, 1.2)]),
            segment([word("[*]", 1.0, 1.5), word("い", 1.5, 2.0)]),
        ]
        out, _ = apply_rules(segs, track)
        assert out[1]["words"][0]["start"] == 1.2
        assert out[1]["start"] == 1.2

    def test_no_energy_track_merges_every_long_block(self) -> None:
        # Standalone asr-align has no track: the gate can never pass, every
        # block (any length) merges back — and nothing crashes.
        segs = [
            segment(
                [word("[*]", 1.0, 2.5), word("あ", 2.5, 3.0)],
                alignment_events=[
                    {
                        "type": "disfluency_candidate",
                        "original_start": 0.5,
                        "refined_start": 1.0,
                        "is_leading_word": True,
                    }
                ],
            )
        ]
        out, stats = word_starts.apply_disfluency_rules(segs, energy_track=None)
        assert stats == {"merge": 1}
        assert out[0]["words"][0]["start"] == 1.0

    def test_segments_without_blocks_pass_through_unchanged(self) -> None:
        track = make_track(5.0, [])
        segs = [segment([word("あ", 1.0, 1.5)], confidence=0.9)]
        out, stats = apply_rules(segs, track)
        assert out[0] is segs[0]
        assert stats == {}


class TestLeadingCandidateGate:
    def event(self, original: float, refined: float) -> dict:
        return {
            "type": "disfluency_candidate",
            "original_start": original,
            "refined_start": refined,
            "is_leading_word": True,
        }

    def test_quiet_leading_gap_keeps_refined_start(self) -> None:
        track = make_track(5.0, [(1.0, 1.45)])
        segs = [
            segment(
                [word("あ", 1.5, 2.0), word("い", 2.0, 2.5)],
                alignment_events=[self.event(1.0, 1.5)],
            )
        ]
        out, stats = apply_rules(segs, track)
        assert stats == {"leading_delete": 1}
        first = out[0]["words"][0]
        assert abs(first["start"] - 1.45) < 0.02
        assert first["disfluency_action"] == "leading_delete"
        assert first["disfluency_span"] == [1.0, 1.5]

    def test_loud_leading_gap_reverts_to_original_start(self) -> None:
        track = make_track(5.0, [])
        segs = [
            segment(
                [word("あ", 1.5, 2.0)],
                alignment_events=[self.event(1.0, 1.5)],
            )
        ]
        out, stats = apply_rules(segs, track)
        assert stats == {"leading_merge": 1}
        assert out[0]["words"][0]["start"] == 1.0
        assert out[0]["start"] == 1.0

    def test_revert_is_bounded_by_previous_segment_end(self) -> None:
        track = make_track(5.0, [])
        segs = [
            segment([word("あ", 0.5, 1.3)]),
            segment(
                [word("い", 1.5, 2.0)],
                alignment_events=[self.event(1.0, 1.5)],
            ),
        ]
        out, _ = apply_rules(segs, track)
        assert out[1]["words"][0]["start"] == 1.3

    def test_stale_event_not_matching_word_start_is_ignored(self) -> None:
        track = make_track(5.0, [])
        segs = [
            segment(
                [word("あ", 1.8, 2.3)],
                alignment_events=[self.event(1.0, 1.5)],
            )
        ]
        out, stats = apply_rules(segs, track)
        assert stats == {}
        assert out[0]["words"][0]["start"] == 1.8


class TestAnchorClamps:
    def clamp(self, segments, intervals, hints=()):
        return word_starts.clamp_word_starts(
            segments,
            vad_intervals=[{"start": s, "end": e} for s, e in intervals],
            pause_hints=hints,
        )

    def test_interval_first_word_clamps_to_start_plus_lead(self) -> None:
        segs = [segment([word("あ", 0.9, 2.0)])]
        out, stats = self.clamp(segs, [(1.0, 3.0)])
        assert stats == {"clamp_interval": 1}
        assert out[0]["words"][0]["start"] == 1.1
        assert out[0]["start"] == 1.1

    def test_word_already_past_lead_is_untouched(self) -> None:
        segs = [segment([word("あ", 1.2, 2.0)])]
        out, stats = self.clamp(segs, [(1.0, 3.0)])
        assert stats == {}
        assert out[0]["words"][0]["start"] == 1.2

    def test_word_starting_too_early_is_skipped(self) -> None:
        segs = [segment([word("あ", 0.4, 2.0)])]
        _, stats = self.clamp(segs, [(1.0, 3.0)])
        assert stats == {}

    def test_word_ending_too_soon_is_skipped(self) -> None:
        segs = [segment([word("あ", 0.9, 1.1)])]
        _, stats = self.clamp(segs, [(1.0, 3.0)])
        assert stats == {}

    def test_close_previous_word_blocks_clamp(self) -> None:
        segs = [segment([word("あ", 0.2, 0.9), word("い", 0.95, 2.0)])]
        _, stats = self.clamp(segs, [(1.0, 3.0)])
        assert stats == {}

    def test_barely_long_enough_word_still_clamps_to_lead(self) -> None:
        # end > S+0.15 (guard) implies end-0.05 > S+0.1, so the minimum-span
        # bound can never undercut the lead; it stays a pure safety net.
        segs = [segment([word("あ", 0.9, 1.12)])]
        out, stats = self.clamp(segs, [(0.95, 3.0)])
        assert stats == {"clamp_interval": 1}
        assert abs(out[0]["words"][0]["start"] - 1.05) < 1e-9

    def test_pause_hint_clamps_mid_interval_word(self) -> None:
        segs = [
            segment([word("あ", 1.3, 1.5), word("い", 1.9, 3.0)])
        ]
        out, stats = self.clamp(segs, [(1.15, 4.0)], hints=[2.3])
        assert stats == {"clamp_hint": 1}
        assert out[0]["words"][1]["start"] == 2.3

    def test_hint_duplicating_interval_start_is_skipped(self) -> None:
        segs = [segment([word("あ", 0.9, 2.0)])]
        _, stats = self.clamp(segs, [(1.0, 3.0)], hints=[1.05])
        # The interval clamp fires; the hint maps to the same anchor and must
        # not double-apply with its tighter zero lead.
        assert stats == {"clamp_interval": 1}

    def test_hint_never_moves_start_earlier(self) -> None:
        segs = [segment([word("あ", 2.5, 3.5)])]
        out, stats = self.clamp(segs, [(0.0, 4.0)], hints=[2.0])
        assert stats == {}
        assert out[0]["words"][0]["start"] == 2.5

    def test_inputs_are_not_mutated(self) -> None:
        segs = [segment([word("あ", 0.9, 2.0)])]
        self.clamp(segs, [(1.0, 3.0)])
        assert segs[0]["words"][0]["start"] == 0.9

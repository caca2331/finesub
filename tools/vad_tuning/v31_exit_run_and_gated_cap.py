"""Routes (b) and (c) from FINDINGS Y3, swept together on the same guards.

  (c) exit-run accumulator: a speech-like frame's debit is banked and only hits
      the non-speech score once the speech-like run reaches N consecutive
      frames; shorter bursts are forgiven. Jitter bursts have median run 2-5
      frames, real words 17-24 (measured), so N in 3..8 absorbs blips without
      touching word-driven closes. Interval head/end frames come from quiet
      frames only, so boundaries do not move -- only whether an accumulation
      survives a blip.

  (b) voicing-gated cap: floor' = min(floor, anchor + cap) applied only on
      frames where silero (dilated +/-0.3 s) sees voicing. Silero failing means
      the cap does not bind -- behavior falls back to production; it can add
      speech, never remove it.

Guards per clip: interval count, speech seconds, empty-interval count/seconds
(no valid-word overlap), and on the annotated clip the hand-word lost count
(coverage < 0.1) plus the 25 word onsets (must keep) and 32 filled pauses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from backends import SILERO_HOP_SEC, silero_probs  # noqa: E402
from energy_sweep import Tracks, cached_tracks, speech_from_tracks  # noqa: E402
from floor_decomposed import Decomposed  # noqa: E402
from refs import covered, load_pause_ref, load_valid_words, load_word_srt  # noqa: E402
from asr_playground.speech.preprocessing import energy as E  # noqa: E402

Interval = Tuple[float, float]


# ---------------------------------------------------------------- (c) scorer
def score_exit_run(tr: Tracks, exit_min_run: int, floor: np.ndarray = None) -> List[Interval]:
    """Production scorer with a banked speech debit; N=0 reproduces production."""
    e = tr.energy_db.numpy().astype(np.float64)
    n_ = (tr.noise_floor.numpy().astype(np.float64) if floor is None else floor)
    dbfs = tr.frame_dbfs.numpy().astype(np.float64)
    starts = tr.frame_starts.numpy().astype(np.float64)
    ends = tr.frame_ends.numpy().astype(np.float64)
    hop = float(starts[1] - starts[0])
    margin = 6.0
    cap_score = max(1.0, (E.MIN_NON_SPEECH_MS / 1000.0) / hop)
    ratio = float(E.MIN_NON_SPEECH_MS) / max(float(E.MERGE_GAP_MS), 1e-6)

    out: List[Interval] = []
    score = head = end = 0.0
    is_iv = False
    run = 0
    debit = 0.0
    for i in range(len(e)):
        quiet = (e[i] <= n_[i] + margin) and (dbfs[i] <= E.ABS_NON_SPEECH_MAX_DBFS_ENTER)
        speech = (e[i] > n_[i] + margin) or (dbfs[i] >= E.ABS_NON_SPEECH_MAX_DBFS_EXIT)
        if quiet:
            run = 0
            debit = 0.0
            if score <= 0.0 and not is_iv:
                head = starts[i]
            score += 2.0 + min(0.0, (n_[i] - e[i]) / margin)
        elif speech:
            term = max(0.1, min(1.0, ((e[i] - n_[i] - margin) / 10) ** 2))
            if exit_min_run <= 1:
                score -= ratio * term
            else:
                run += 1
                debit += ratio * term
                if run >= exit_min_run:
                    score -= debit
                    debit = 0.0
        if score >= cap_score:
            score = cap_score
            end = ends[i]
            is_iv = True
        last = i == len(e) - 1
        if score < 0.0 or last:
            if is_iv and end > head:
                out.append((max(0.0, head), min(end, tr.duration)))
            score, is_iv = 0.0, False
            run, debit = 0, 0.0
            head = starts[i + 1] if not last else min(tr.duration, ends[i])
            end = head
    ns = E._apply_negative_padding(out, tr.duration)
    ns = E._absorb_low_peak_speech(ns, tr.energy_db, tr.duration)
    return [(float(a), float(b)) for a, b in E.invert_intervals(ns, tr.duration) if b > a]


# ------------------------------------------------------------- (b) gated cap
def gated_cap_speech(tr: Tracks, sil: np.ndarray, cap: float,
                     dilate_sec: float = 0.3, thr: float = 0.5) -> List[Interval]:
    e = tr.energy_db.numpy().astype(np.float64)
    starts = tr.frame_starts.numpy().astype(np.float64)
    hop = float(starts[1] - starts[0])
    anchor = Decomposed()._anchor(e, hop)
    idx = np.clip((starts / SILERO_HOP_SEC).astype(int), 0, len(sil) - 1)
    voiced = sil[idx] >= thr
    k = max(1, int(dilate_sec / hop))
    vd = voiced.copy()
    for s in range(1, k + 1):
        vd[s:] |= voiced[:-s]
        vd[:-s] |= voiced[s:]
    floor = tr.noise_floor.numpy().astype(np.float64).copy()
    gate = vd & (floor > anchor + cap)
    floor[gate] = anchor[gate] + cap
    raw = E._score_to_non_speech_intervals(
        tr.energy_db, torch.from_numpy(floor.astype(np.float32)), tr.frame_dbfs,
        tr.frame_starts, tr.frame_ends, tr.duration,
        enter_margin_db=6.0, weighted=bool(E.WEIGHTED_INTERVAL))
    ns = E._apply_negative_padding(raw, tr.duration)
    ns = E._absorb_low_peak_speech(ns, tr.energy_db, tr.duration)
    return [(float(a), float(b)) for a, b in E.invert_intervals(ns, tr.duration) if b > a]


# ------------------------------------------------------------------- scoring
def eval_arm(name, sp, words, pause_ref, joined_clip, prod, sil, duration):
    sp_sec = sum(e - s for s, e in sp)
    wiv = [(w.start, w.end) for w in words]
    empty = [iv for iv in sp if not any(a < iv[1] and b > iv[0] for a, b in wiv)]
    lost = sum(1 for a, b in wiv if covered(sp, a, b) / max(b - a, 1e-9) < 0.1)
    row = {"arm": name, "n_iv": len(sp), "speech_s": round(sp_sec, 1),
           "empty_n": len(empty), "empty_s": round(sum(b - a for a, b in empty), 1),
           "lost": lost}
    if pause_ref is not None:
        row["onset_cut"] = sum(1 for a, b in pause_ref.word_onset
                               if covered(sp, a, b) / max(b - a, 1e-9) < 0.5)
        row["pause_excl"] = sum(1 for a, b in pause_ref.filled_pause
                                if covered(sp, a, b) / max(b - a, 1e-9) < 0.5)
    if joined_clip is not None and prod is not None:
        rescue = joined_clip[(joined_clip["kind"] == "added")
                             & (joined_clip["never_decoded_frac"] > 0.5)
                             & (joined_clip["label"].isin(["真语音", "听不清"]))]
        row["rescue"] = sum(1 for _, r in rescue.iterrows()
                            if covered(sp, r["start"], r["end"]) /
                            max(r["end"] - r["start"], 1e-9) > 0.5)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=("b", "c", "both"))
    args = ap.parse_args()

    B = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
    cache = B / "tmp/vad-tuning-cache"
    clips = {
        "gold": (B / "out/reference/BV1cqLR6hEp3/BV1cqLR6hEp3-vocal.flac",
                 B / "data/disfluency-gold/BV1cqLR6hEp3/BV1cqLR6hEp3-fixed.srt", "srt"),
        "yingtao": (B / "out/yingtao/yingtao-vocal.flac",
                    B / "out/yingtao/yingtao-stable.json", "stable"),
        "kaguya60": (B / "out/kaguya60/kaguya60-vocal.flac",
                     B / "out/kaguya60/kaguya60-stable.json", "stable"),
        "miyako": (B / "out/miyako-overlap-removed-overlap-only/"
                       "miyako-overlap-removed-overlap-only-vocal.ogg",
                   B / "tmp/vad-step0/e2e/miyako-base-stable.json", "stable"),
    }
    joined = pd.read_csv(B / "tmp/vad-step0/step0-joined.csv", encoding="utf-8-sig")
    gold_pause = load_pause_ref(
        B.parent / "asr-playground-onset-gap-energy/tools/wt_refine_validation/disfluency_gold.json"
        if (B.parent / "asr-playground-onset-gap-energy/tools/wt_refine_validation/disfluency_gold.json").exists()
        else B / "tools/wt_refine_validation/disfluency_gold.json")

    for name, (vocal, ref, kind) in clips.items():
        tr = cached_tracks(vocal, cache)
        words = load_word_srt(ref) if kind == "srt" else load_valid_words(ref)[0]
        pr = gold_pause if name == "gold" else None
        jc = joined[joined["clip"] == ("BV1cqLR6hEp3" if name == "gold" else name)]
        sil = silero_probs(vocal, cache / f"silero-{vocal.stem}.npz")
        prod = speech_from_tracks(tr)
        rows = [eval_arm("prod", prod, words, pr, jc, prod, sil, tr.duration)]
        if args.which in ("c", "both"):
            for n in (3, 4, 6, 8):
                rows.append(eval_arm(f"exitrun{n}", score_exit_run(tr, n),
                                     words, pr, jc, prod, sil, tr.duration))
        if args.which in ("b", "both"):
            for cap in (10.0, 15.0, 20.0):
                rows.append(eval_arm(f"gcap{cap:.0f}", gated_cap_speech(tr, sil, cap),
                                     words, pr, jc, prod, sil, tr.duration))
        df = pd.DataFrame(rows)
        print(f"\n== {name}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

"""Per-line confidence marks: which labelled lines to check first, and a track for lines nobody claimed.

Three outcomes per subtitle line, all riding on its `speaker` object as `mark`:

* **No mark** -- the acoustic layer is sure. On the material the model was fitted on (Japanese two- to
  four-person streams) 74% of ordinary lines land here and 1.0% of them are wrong.
* **`low`** -- a logistic score over eleven signals production already has (`P_WRONG` and up). When the
  best other candidate track scores `ALTERNATIVE` or more it is named as `alternative`: where the label
  was wrong that track was the right person 85% of the time, but most `low` lines are not wrong at all,
  so it is a hint, never a relabel. Back-channel lines (`BACKCHANNEL_CHARS` or fewer characters, owner
  2026-10-07) get no `low` mark: they are cheap to delete and the reference often leaves them out.
* **`filled`** -- the line reached no track (`assign.min_share`). Its most active Nemotron slot's track
  when that slot is pure enough (`FILL_PURITY`), else the nearest labelled line's track: right on 58% of
  the missed lines the reference has words on, against 46% for the nearest line alone. About a third of
  missed lines have no words in the reference at all (music, effects) and get a track too. ⚠ Written to `mark.track`, never to `track`: `track` means
  "the acoustic layer put it there", and the speaker judge, `same_speaker` and third-party readers rely on it.

⚠ **Fitted on Japanese only.** The heaviest of the eleven signals is the line's character count, which
roughly doubles for the same duration in English: doubling it alone moves the flagged share from 25% to
15%. Marks are written for every language (owner 2026-10-07); other languages are a follow-up.

⚠ **Blind to merged tracks.** When several similar voices share one track their lines look as certain
as any other; no line-level signal sees it (`docs/diarization-findings.md`).

Evidence: `docs/diarization-findings.md`; the experiment this ports is
`tools/diarization_explore/p2_features.py` + `marks_step1.py`, and `marks_export.py` prints the constants.
Needs the nemotron3 sidecar fields `slot_grid`, `slot_probs` and `hard`; anything else gets no marks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from .assign import track_coverage
from .labels import SegmentSpeaker
from .segments import TrackSegment

#: What `speaker.marks` in `-diar-rows.json` records when marks were computed. `None` there means "not computed".
VERSION = 1
FITTED_ON = "ja"

# --- constants: `python tools/diarization_explore/marks_export.py` (step 1's fit, P2 dev, 28 hina files) ---
ERROR_MEAN = (0.918383128316248, 0.7996624940803368, 0.8453929663595365, 0.18195063436580025, 0.057301364144035194, 0.3854528719830308, 2.2593090612858804, 14.966370054726992, 0.5700797927894056, 0.5689936082215816, 0.28578526289280987)
ERROR_SCALE = (0.13365945692126532, 0.2921176966804433, 0.15276702278166923, 0.22669950497732516, 0.11614366509335033, 0.15241811306653785, 1.3617868764339287, 8.631728903579894, 0.49506446311823843, 0.4952170049832178, 0.13490153998782167)
ERROR_COEF = (0.025484114961371013, -0.38896862119010905, -0.5302743082644585, 0.3859809697070284, 0.10574815502178894, -0.4105360292115305, -0.04550489825815373, -0.6223410954395941, 0.06860260599722376, 0.12377518073136815, 0.07766900039271406)
ERROR_INTERCEPT = -3.7298489821056937
CANDIDATE_MEAN = (0.4795528647669258, 0.4564292098652028, 1.4360168393928308)
CANDIDATE_STD = (0.4259170578371087, 0.3828720477243616, 0.2765091847949794)
CANDIDATE_WEIGHTS = (-0.11423195440550132, 1.7371112007548284, 0.9745189703053321)
#: P(wrong) at or above this is `low`: the dev quantile that flags 25% of ordinary lines.
P_WRONG = 0.04783493855236411
#: Best other candidate's share of the in-line softmax at or above this is named as `alternative`.
ALTERNATIVE = 0.211
# --- end of fitted constants ---

#: The order `ERROR_*` expects.
ROW_FEATURES = ("lead_share", "margin", "p_lead", "p_comp", "none_frac", "soft_margin", "seconds", "chars",
                "same_prev", "same_next", "track_share")

#: A missed line takes its slot's track only when the slot is at least this pure.
FILL_PURITY = 0.7
#: A slot this pure also claims its sub-threshold frames for its majority track (`slot_maps`).
SLOT_PURITY = 0.80
#: A track is a candidate when its mean mapped probability over the line exceeds this.
CANDIDATE_PROB = 0.05
#: A frame "has someone" above this probability.
ACTIVE = 0.5
#: At most this many characters, punctuation not counted: a back-channel line, never `low`.
BACKCHANNEL_CHARS = 4
#: The 20 ms grid of the native activity and of `slot_probs`.
STEP = 0.02


def metadata() -> dict[str, Any]:
    return {"version": VERSION, "fitted_on": FITTED_ON, "p": round(P_WRONG, 4), "q": ALTERNATIVE}


def characters(text: str) -> int:
    return len(re.sub(r"[\W_]", "", text))


def resample(probabilities, frame_seconds: float, frames: int):
    """Model-rate probabilities -> `frames` frames of `STEP`, each the overlap-weighted mean."""
    import numpy as np

    probabilities = np.asarray(probabilities)
    ratio = STEP / frame_seconds
    t = np.arange(frames, dtype=np.float64)
    lo, hi = t * ratio, (t + 1) * ratio
    first = np.floor(lo).astype(np.int64)
    width = int(math.ceil(ratio)) + 1
    index = first[:, None] + np.arange(width)[None, :]
    stop = np.minimum(np.ceil(hi).astype(np.int64), len(probabilities))
    weight = np.minimum(index + 1, hi[:, None]) - np.maximum(index, lo[:, None])
    weight = np.where((index < stop[:, None]) & (weight > 0), weight, 0.0)
    values = probabilities[np.clip(index, 0, len(probabilities) - 1)].astype(np.float64)
    total = weight.sum(axis=1)
    mean = (values * weight[:, :, None]).sum(axis=1) / np.where(total > 0, total, 1.0)[:, None]
    return np.where(total[:, None] > 0, mean, 0.0).astype(np.float32)


def slot_maps(local, hard, centroid_of_track, frames: int):
    """Per slot: the track each 20 ms frame belongs to (-1 = none) and the slot's majority purity.

    Plus each window's start frame and per-(window, slot) activity. Weighted by activity, masked by
    activity as `diarizen_assemble.assemble` is -- not by `hard`.
    """
    import numpy as np

    from . import diarizen_assemble
    from .nemotron3_backend import STEP_SECONDS

    chunks, per_window, slots = local.shape
    tracks = len(centroid_of_track)
    valid = np.array([centroid_of_track[k] >= 0 for k in range(tracks)])
    starts = np.array([diarizen_assemble.closest_frame(c * STEP_SECONDS + 0.5 * diarizen_assemble.FRAME_DURATION)
                       for c in range(chunks)])
    activity = local.sum(axis=1)
    maps, purity = np.full((slots, frames), -1, dtype=np.int32), np.zeros(slots)
    for j in range(slots):
        weight = np.zeros((frames, tracks), dtype=np.float32)
        for c in np.flatnonzero(activity[:, j] > 0):
            k = int(hard[c, j])
            if not (0 <= k < tracks) or not valid[k]:
                continue
            a, b = max(0, starts[c]), min(frames, starts[c] + per_window)
            if b > a:
                weight[a:b, k] += activity[c, j]
        totals = weight.sum(axis=0)
        purity[j] = totals.max() / totals.sum() if totals.sum() else 0.0
        covered = weight.sum(axis=1) > 0
        maps[j, covered] = weight[covered].argmax(axis=1)
        if purity[j] >= SLOT_PURITY and totals.sum():
            maps[j, ~covered] = int(totals.argmax())
    return maps, purity, starts, activity


def usable(evidence) -> bool:
    """Whether this sidecar carries what the marks need (nemotron3 writes it; DiariZen and old files do not)."""
    return evidence is not None and all(
        getattr(evidence, name, None) is not None for name in ("slot_grid", "slot_probs", "hard", "soft",
                                                                "centroid_of_track", "native_activity"))


def row_marks(
    subtitles: Sequence[Mapping[str, Any]],
    timeline: Sequence[TrackSegment],
    speakers: Sequence[SegmentSpeaker],
    evidence,
) -> "list[dict[str, Any] | None] | None":
    """One mark (or None) per subtitle line; None overall when the evidence cannot carry marks."""
    import numpy as np

    from .nemotron3_backend import windows

    if not usable(evidence) or len(subtitles) != len(speakers):
        return None
    native = np.asarray(evidence.native_activity)
    soft = np.asarray(evidence.soft)
    centroid_of_track = np.asarray(evidence.centroid_of_track)
    hard = np.asarray(evidence.hard)
    prob = np.asarray(evidence.slot_probs, dtype=np.float32)
    local = windows(np.asarray(evidence.slot_grid, dtype=np.float32))
    frames, tracks_n = native.shape
    if prob.shape[0] != frames or len(centroid_of_track) != tracks_n or hard.shape != local.shape[::2]:
        return None
    maps, purity, chunk_starts, _ = slot_maps(local, hard, centroid_of_track, frames)
    trackprob = np.zeros((frames, tracks_n), dtype=np.float32)
    for j in range(maps.shape[0]):
        ok = maps[j] >= 0
        np.maximum.at(trackprob, (np.flatnonzero(ok), maps[j][ok]), prob[ok, j])
    major = {}
    for j in range(maps.shape[0]):
        owned = maps[j][(maps[j] >= 0) & (prob[:, j] > ACTIVE)]
        major[j] = Counter(owned.tolist()).most_common(1)[0][0] if len(owned) else -1

    segments = sorted(timeline, key=lambda segment: segment.start)
    total_time: Counter = Counter()
    for segment in segments:
        total_time[segment.track] += segment.end - segment.start
    all_time = sum(total_time.values()) or 1.0
    per_window = local.shape[1]
    valid = centroid_of_track >= 0
    column = np.where(valid, centroid_of_track, 0)

    # Neighbours and the nearest labelled line come from the acoustic labels only: a filled line never
    # feeds another fill.
    leaders = [speaker.track for speaker in speakers]
    centres = np.array([(float(row["start"]) + float(row["end"])) / 2 for row in subtitles])
    labelled = np.array([i for i, k in enumerate(leaders) if k is not None], dtype=np.int64)
    previous, following = [None] * len(leaders), [None] * len(leaders)
    last = None
    for i, k in enumerate(leaders):
        previous[i] = last
        if k is not None:
            last = k
    last = None
    for i in range(len(leaders) - 1, -1, -1):
        following[i] = last
        if leaders[i] is not None:
            last = leaders[i]

    def nearest(i):
        pool = labelled[labelled != i]
        return leaders[int(pool[np.argmin(np.abs(centres[pool] - centres[i]))])] if len(pool) else None

    marks: "list[dict[str, Any] | None]" = []
    for index, (row, leader) in enumerate(zip(subtitles, leaders)):
        start, end = float(row["start"]), float(row["end"])
        lo = int(start / STEP)
        hi = min(max(lo + 1, int(np.ceil(end / STEP))), frames)
        if hi <= lo:
            marks.append(None)
            continue
        if leader is None:
            marks.append(_fill(prob[lo:hi], purity, major, nearest(index)))
            continue
        leader = int(leader)
        coverage = {int(k): float(v) for k, v in track_coverage(start, end, segments).items()}
        rows_prob = trackprob[lo:hi].mean(axis=0)
        none_frac = float((trackprob[lo:hi].max(axis=1) <= ACTIVE).mean())
        soft_sum, soft_weight, present = np.zeros(tracks_n), 0.0, set()
        own_sum = own_weight = 0.0
        for c in np.flatnonzero((chunk_starts < hi) & (chunk_starts + per_window > lo)):
            a, b = max(lo, chunk_starts[c]) - chunk_starts[c], min(hi, chunk_starts[c] + per_window) - chunk_starts[c]
            for j in range(local.shape[2]):
                act = float(local[c, a:b, j].sum())
                # ⚠ A slot active for too few frames got no voiceprint: its soft row is NaN and its `hard` an
                # arbitrary pick, not evidence.
                if act <= 0 or np.isnan(soft[c, j]).all():
                    continue
                k = int(hard[c, j])
                if 0 <= k < tracks_n and valid[k]:
                    present.add(k)
                scores = np.where(valid, soft[c, j, column], np.nan)
                soft_sum += np.nan_to_num(scores) * act
                soft_weight += act
                if k == leader and valid[leader]:
                    others = np.delete(scores, leader)
                    others = others[~np.isnan(others)]
                    own_sum += act * (scores[leader] - (others.max() if others.size else 0.0))
                    own_weight += act
        soft_k = soft_sum / soft_weight if soft_weight else np.zeros(tracks_n)
        shares = sorted(coverage.values(), reverse=True)
        chars = characters(str(row.get("text", "")))
        features = (
            shares[0] if shares else 0.0,
            (shares[0] - (shares[1] if len(shares) > 1 else 0.0)) / shares[0] if shares and shares[0] else 0.0,
            float(rows_prob[leader]),
            float(max([rows_prob[k] for k in range(tracks_n) if k != leader] or [0.0])),
            none_frac,
            own_sum / own_weight if own_weight else 0.0,
            end - start,
            chars,
            float(previous[index] == leader),
            float(following[index] == leader),
            total_time[leader] / all_time,
        )
        p_wrong = _p_wrong(features)
        if chars <= BACKCHANNEL_CHARS or p_wrong < P_WRONG:
            marks.append(None)
            continue
        mark: dict[str, Any] = {"tier": "low", "p_wrong": round(p_wrong, 3)}
        candidates = [k for k in range(tracks_n) if valid[k] and
                      (coverage.get(k, 0.0) > 0 or rows_prob[k] > CANDIDATE_PROB or k in present)]
        alternative = _alternative([(k, coverage.get(k, 0.0), float(rows_prob[k]), float(soft_k[k]))
                                    for k in candidates], leader)
        if alternative is not None:
            mark["alternative"] = alternative
        marks.append(mark)
    return marks


def _p_wrong(features) -> float:
    z = ERROR_INTERCEPT + sum((x - m) / s * w for x, m, s, w in zip(features, ERROR_MEAN, ERROR_SCALE, ERROR_COEF))
    return 1.0 / (1.0 + math.exp(-z))


def _alternative(candidates, leader: int) -> "int | None":
    """The best candidate other than the leader, when its in-line softmax share reaches `ALTERNATIVE`."""
    import numpy as np

    if not candidates:
        return None
    x = np.array([c[1:] for c in candidates], dtype=float)
    z = ((x - np.array(CANDIDATE_MEAN)) / np.array(CANDIDATE_STD)) @ np.array(CANDIDATE_WEIGHTS)
    z -= z.max()
    share = np.exp(z) / np.exp(z).sum()
    others = [(c[0], p) for c, p in zip(candidates, share) if c[0] != leader]
    if not others:
        return None
    track, best = max(others, key=lambda item: item[1])
    return int(track) if best >= ALTERNATIVE else None


def _fill(prob, purity, major, nearest) -> "dict[str, Any] | None":
    """A missed line's track: its most active slot's when that slot is pure, else the nearest labelled line's."""
    totals = prob.sum(axis=0)
    # ⚠ No activity at all (music, effects): argmax would pick slot 0 for no reason.
    if totals.sum() > 0:
        slot = int(totals.argmax())
        if major[slot] >= 0 and purity[slot] >= FILL_PURITY:
            return {"tier": "filled", "track": int(major[slot]), "by": "slot"}
    if nearest is None:
        return None
    return {"tier": "filled", "track": int(nearest), "by": "nearest"}


def apply(speakers: Sequence[SegmentSpeaker], marks: Sequence["dict[str, Any] | None"]) -> list[SegmentSpeaker]:
    """The speakers with marks attached; `review` becomes "has a `low` mark" (one trigger for stable and CSV)."""
    from dataclasses import replace

    result = []
    for speaker, mark in zip(speakers, marks):
        extras = {key: value for key, value in speaker.extras.items() if key != "mark"}
        if mark is None:
            result.append(replace(speaker, review=False, extras=extras))
        elif mark["tier"] == "filled":
            result.append(SegmentSpeaker(block=speaker.block, extras={**extras, "mark": mark}))
        else:
            result.append(replace(speaker, review=True, extras={**extras, "mark": mark}))
    return result


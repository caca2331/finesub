"""Apply the noise verdicts to sub-spans of speech intervals.

User review of the final-form tracks: a segment often *starts* with a noise
perturbation and then leads into real speech, or a perturbation *bridges* two
segments that should be separate. Whole-interval rules (the -45 floor, the
ghost drop) cannot touch these; this carves inside the interval.

Evidence frame  = silero voiced (>= SIL_EVID, dilated a little) OR loud
                  (energy_db >= LOUD_DB).
Noise span      = maximal evidence-free run whose own peak stays under a class
                  ceiling: CERTAIN (-45, energy-only, default-on candidate) or
                  SILERO (0 dB, needs silero, opt-in candidate).

Carving keeps the decoder's margins: a head trim leaves LEAD_IN (0.14 s,
mirroring NEGATIVE_PAD_RIGHT_MS) of real audio before the first evidence, a
tail trim leaves LEAD_OUT (0.04 s), and an interior split only lands if the
carved non-speech still spans MIN_CARVE after both margins -- so a real pause
gets a proper seam and a short dip is left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from backends import SILERO_HOP_SEC  # noqa: E402

Interval = Tuple[float, float]

SIL_EVID = 0.3
SIL_EVID_DILATE = 0.10
LOUD_DB = 0.0
LEAD_IN = 0.14
LEAD_OUT = 0.04
MIN_TRIM = 0.20     # a prefix/suffix shorter than this is not worth touching
MIN_CARVE = 0.20    # carved interior non-speech after margins must be at least this
INTERIOR_RUN = 0.40


# No minimum length: a production gap of any size is the detector's own full
# hysteresis verdict (its floor is MIN_KEEP_AFTER_SHRINK_MS); re-thresholding
# it here silently dropped real seams (the user's 38.75-38.85 example, 95 ms).
MIN_SEAM = 0.0
SEAM_LOUD_KEEP_DB = -5.0


def restore_seams(intervals: List[Interval], prod: List[Interval],
                  edb: np.ndarray, starts: np.ndarray,
                  sil: np.ndarray) -> Tuple[List[Interval], dict]:
    """Re-cut the original detector's gaps where a merge swallowed them.

    The gated cap fuses intervals; downstream, those seams are what grouping,
    the DP splitter and the decoder's own sentence breaks key on. Voiced-but-
    quiet content in a swallowed gap (breath, a filler) is no reason to lose
    the seam -- silero fires on it, which is why an evidence-frame test
    restores nothing. Only loud content (>= SEAM_LOUD_KEEP_DB peak in the gap
    core, i.e. actual rescued creep speech) justifies keeping the merge;
    everything else gets its seam back at the production gap's exact bounds.
    """
    hop = float(starts[1] - starts[0]) if len(starts) > 1 else 0.01
    n = len(edb)
    gaps = [(prod[i][1], prod[i + 1][0]) for i in range(len(prod) - 1)
            if prod[i + 1][0] > prod[i][1]]
    out: List[Interval] = []
    stats = {"seams_restored": 0, "removed_sec": 0.0}
    for s, e in intervals:
        cuts: List[Interval] = []
        for g0, g1 in gaps:
            if g0 <= s or g1 >= e:
                continue  # not strictly interior to this interval
            # The production gap already carries the negative padding -- its
            # bounds ARE the lead margins. Restore it exactly: adding margins
            # again both skipped short gaps and started the following interval
            # 0.14 s earlier than production.
            if g1 - g0 < MIN_SEAM:
                continue
            i0 = max(0, int(np.ceil(g0 / hop - 1e-9)))
            i1 = min(n, int(np.ceil(g1 / hop - 1e-9)))
            if i1 <= i0 or float(edb[i0:i1].max()) >= SEAM_LOUD_KEEP_DB:
                continue
            cuts.append((g0, g1))
            stats["seams_restored"] += 1
            stats["removed_sec"] += g1 - g0
        cur = s
        for c0, c1 in sorted(cuts):
            if c0 > cur:
                out.append((cur, c0))
            cur = max(cur, c1)
        if e > cur:
            out.append((cur, e))
    stats["removed_sec"] = round(stats["removed_sec"], 1)
    return out, stats


def carve_intervals(intervals: List[Interval], edb: np.ndarray, starts: np.ndarray,
                    sil: np.ndarray, *, ceiling_db: float,
                    use_silero_evidence: bool = True) -> Tuple[List[Interval], dict]:
    hop = float(starts[1] - starts[0]) if len(starts) > 1 else 0.01
    n = len(edb)
    if use_silero_evidence:
        idx = np.clip((starts / SILERO_HOP_SEC).astype(int), 0, max(len(sil) - 1, 0))
        voiced = (sil[idx] >= SIL_EVID) if len(sil) else np.zeros(n, bool)
        k = max(1, int(SIL_EVID_DILATE / hop))
        vd = voiced.copy()
        for s in range(1, k + 1):
            vd[s:] |= voiced[:-s]
            vd[:-s] |= voiced[s:]
    else:
        vd = np.zeros(n, bool)
    evidence = vd | (edb >= LOUD_DB)

    out: List[Interval] = []
    stats = {"head_trims": 0, "tail_trims": 0, "splits": 0, "removed_sec": 0.0}
    for s, e in intervals:
        i0 = max(0, int(np.ceil(s / hop - 1e-9)))
        i1 = min(n, int(np.ceil(e / hop - 1e-9)))
        if i1 <= i0:
            out.append((s, e))
            continue
        ev = evidence[i0:i1]
        pk_ok = edb[i0:i1] < ceiling_db
        noise = ~ev & pk_ok
        # maximal noise runs
        runs = []
        r0 = None
        for j, v in enumerate(noise):
            if v and r0 is None:
                r0 = j
            elif not v and r0 is not None:
                runs.append((r0, j)); r0 = None
        if r0 is not None:
            runs.append((r0, len(noise)))

        seg_s, seg_e = s, e
        cuts: List[Interval] = []
        for a, b in runs:
            t0, t1 = s + a * hop, s + b * hop
            dur = t1 - t0
            head = a == 0
            tail = b == len(noise)
            if head and tail:
                continue  # whole-interval rules own this case
            if head and dur >= MIN_TRIM + LEAD_IN:
                seg_s = t1 - LEAD_IN
                stats["head_trims"] += 1
                stats["removed_sec"] += seg_s - s
            elif tail and dur >= MIN_TRIM + LEAD_OUT:
                seg_e = t0 + LEAD_OUT
                stats["tail_trims"] += 1
                stats["removed_sec"] += e - seg_e
            elif not head and not tail and dur >= INTERIOR_RUN:
                c0, c1 = t0 + LEAD_OUT, t1 - LEAD_IN
                if c1 - c0 >= MIN_CARVE:
                    cuts.append((c0, c1))
                    stats["splits"] += 1
                    stats["removed_sec"] += c1 - c0
        cur = seg_s
        for c0, c1 in cuts:
            if c0 > cur:
                out.append((cur, c0))
            cur = c1
        if seg_e > cur:
            out.append((cur, seg_e))
    stats["removed_sec"] = round(stats["removed_sec"], 1)
    return out, stats

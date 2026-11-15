"""The speakers of a MOSS Pro continuation or gap re-request, relinked to the people already heard.

Each re-request is a fresh call that numbers its speakers anew: its `S01` is usually not the main
call's `S01`, and the two calls do not overlap in time, so they cannot be matched by speech heard
together. Each label is matched by voice instead: a WeSpeaker ResNet34 voiceprint (the model the
default speaker diarization uses, on the CPU) per segment, put in PLDA space; per label, the mean
weighted by clean seconds; then one-to-one (Hungarian) on cosine, a pair below THETA cut. A label
not matched -- too little speech, or nobody close enough -- is a new person, `c<k>:<label>` for the
k-th re-request. Without the model every label is a new person.

The numbers are the research's (branch `explore/moss-relink`; the validation is in `docs/joint-asr.md`): on
re-requests faked from main calls, 83% of continuation labels and 62% of gap labels right
(95% / 85% of the speech), 2-3% given to the wrong person, against 8% right when the labels are
read as the main call's.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable, Sequence

from ...reporting import current_reporter

SR = 16000
#: A segment shorter than this is not embedded; a longer one only its centre MAX_CROP.
MIN_SEGMENT = 1.0
MAX_CROP = 15.0
#: Frames other segments overlap are masked out, unless fewer than this would be left.
MIN_CLEAN_FRAMES = 50
#: A segment's weight in its label's mean: its clean seconds, capped.
WEIGHT_CAP = 10.0
#: A label with less embedded speech than this is not matched.
MIN_LABEL = 3.0
THETA = 0.8
PLDA_DIM = 128
#: `Voices` takes each label's voice from at most this many of its longest blocks (an hour has hundreds).
MAX_BLOCKS = 20

if TYPE_CHECKING:
    import numpy as np

#: piece (float32 at SR), mask (one per 10 ms frame) -> unit vector in PLDA space.
Embed = Callable[["np.ndarray", "np.ndarray"], "np.ndarray"]


class Voiceprints:
    """The voiceprint model, loaded on first use; `embed` is `None` when it cannot be had (not retried in this
    process; `error` says why)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self._embed: Embed | None = None
        self.error: str | None = None

    def embed(self) -> Embed | None:
        with self._lock:
            if not self._loaded:
                self._loaded = True
                try:
                    self._embed = _wespeaker()
                except Exception as error:  # noqa: BLE001 - the labels degrade, the transcript does not
                    self.error = str(error)
        return self._embed


def _wespeaker() -> Embed:
    import numpy as np
    import torch

    from ..diarization.backend import artifact_candidates
    from ..diarization.diarizen_cluster import load_plda
    from ..diarization.diarizen_features import compute_fbank
    from ..diarization.nemotron3_backend import HOSTED_EMBEDDING, PLDA_FILES
    from ..runtime.native_paths import load_torchscript

    files = ("wespeaker-cpu.ts", *PLDA_FILES)
    directory = next((c for c in artifact_candidates(None) if all((c / f).exists() for f in files)), None)
    if directory is None:
        from ..runtime import hf_weights

        directory = hf_weights.prepare_directory(HOSTED_EMBEDDING)
        if directory is None:
            raise RuntimeError("could not fetch the speaker embedding model")
    model = load_torchscript(directory / "wespeaker-cpu.ts", map_location="cpu").eval()
    xvec_tf, plda_tf, _ = load_plda(directory / "plda")

    def embed(piece: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            fbank = compute_fbank(torch.from_numpy(piece).reshape(1, 1, -1))
            raw = model(fbank, torch.from_numpy(mask).reshape(1, -1)).numpy().astype(np.float64)
        x = plda_tf(xvec_tf(raw), lda_dim=PLDA_DIM)[0]
        return x / np.linalg.norm(x)

    return embed


#: The process's one copy: batch items on the network lane share it.
VOICEPRINTS = Voiceprints()


class _Prints:
    """Per segment (by identity: a relabelled segment is the same dict), its voiceprint and clean seconds,
    computed once per transcript."""

    def __init__(self, audio: np.ndarray, embed: Embed):
        self.audio = audio
        self.embed = embed
        self._done: dict[int, tuple[dict, tuple[np.ndarray, float] | None]] = {}

    def of(self, seg: dict, others: Sequence[dict]) -> tuple[np.ndarray, float] | None:
        key = id(seg)
        if key not in self._done:  # the segment is kept alongside, so its id is not reused
            self._done[key] = (seg, self._compute(seg, others))
        return self._done[key][1]

    def _compute(self, seg: dict, others: Sequence[dict]) -> tuple[np.ndarray, float] | None:
        import numpy as np

        dur = seg["end"] - seg["start"]
        if dur < MIN_SEGMENT:
            return None
        mid = (seg["start"] + seg["end"]) / 2
        a, b = (mid - MAX_CROP / 2, mid + MAX_CROP / 2) if dur > MAX_CROP else (seg["start"], seg["end"])
        piece = self.audio[int(a * SR):int(b * SR)].astype(np.float32) / 32768
        if len(piece) < SR * MIN_SEGMENT:
            return None
        t = a + np.arange(int(len(piece) / SR * 100)) / 100
        mask = np.ones(len(t), np.float32)
        if "voiced" in seg:  # only the frames Silero hears speech in (a repaired span: not its laughter or music)
            mask[:] = 0
            for x, y in seg["voiced"]:
                mask[(t >= x) & (t < y)] = 1
        for o in others:
            if o is not seg and o["end"] > a and o["start"] < b:
                mask[(t >= o["start"]) & (t < o["end"])] = 0
        if mask.sum() < MIN_CLEAN_FRAMES:
            mask[:] = 1
        return self.embed(piece, mask), float(mask.sum()) / 100


def _centroids(segs: Sequence[dict], every: Sequence[dict], prints: _Prints) -> dict[str, np.ndarray]:
    import numpy as np

    sums: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    for s in segs:
        if s["speaker"] is None:
            continue
        got = prints.of(s, every)
        if got is None:
            continue
        vec, clean = got
        w = min(clean, WEIGHT_CAP)
        sums[s["speaker"]] = sums.get(s["speaker"], 0) + w * vec
        weights[s["speaker"]] = weights.get(s["speaker"], 0.0) + w
    return {lab: v / np.linalg.norm(v) for lab, v in sums.items() if weights[lab] >= MIN_LABEL}


class Relinker:
    """Relabels each re-request's segments against everything heard before it in the transcript."""

    def __init__(self, audio: Callable[[], np.ndarray], voiceprints: Voiceprints):
        self._audio = audio
        self._voiceprints = voiceprints
        self._prints: _Prints | None = None
        self._warned = False
        self.calls = 0

    def __call__(self, segs: Sequence[dict], extra: Sequence[dict]) -> dict[str, str]:
        """Relabels `extra` in place onto `segs`' labels (or prefixed as new people); returns the map."""
        self.calls += 1
        labels = sorted({s["speaker"] for s in extra if s["speaker"] is not None})
        mapping = {lab: f"c{self.calls}:{lab}" for lab in labels}
        embed = self._voiceprints.embed() if labels else None
        if labels and embed is None and not self._warned:  # once per transcript: each batch item's report says it
            self._warned = True
            current_reporter().warning(
                "moss-relink",
                f"声纹模型拿不到（{self._voiceprints.error}）：MOSS Pro 续调、补调回来的说话人都当新人")
        if embed is not None:
            if self._prints is None:
                self._prints = _Prints(self._audio(), embed)
            every = [*segs, *extra]
            try:
                mapping.update(_match(_centroids(segs, every, self._prints), _centroids(extra, every, self._prints)))
            except Exception as error:  # noqa: BLE001 - as for a model that cannot be had
                current_reporter().warning(
                    "moss-relink", f"声纹比对失败（{error}）：这次续调 / 补调回来的说话人都当新人")
        for s in extra:
            s["speaker"] = mapping.get(s["speaker"], s["speaker"])
        return mapping


class Voices:
    """Speakers by voice for the repairs of a word-level transcript (35t, MAI2; `repair.py`), against the labels of
    `known` -- blocks (`repair.blocks`) of the speech already heard. Unlike `Relinker`, `nearest` is not one-to-one:
    each piece takes the closest label at THETA or above, so several pieces can go to one person. Without the model
    (or if comparing fails) nothing is matched, said once per transcript."""

    def __init__(self, audio: Callable[[], np.ndarray], voiceprints: Voiceprints, known: Sequence[dict]):
        self._audio = audio
        self._voiceprints = voiceprints
        by: dict[str, list[dict]] = {}
        for b in known:
            by.setdefault(b["speaker"], []).append(b)
        self._known = [b for bs in by.values()
                       for b in sorted(bs, key=lambda b: b["start"] - b["end"])[:MAX_BLOCKS]]
        self._prints: _Prints | None = None
        self._centroids: dict[str, np.ndarray] | None = None
        self._failed = False

    def _ready(self) -> bool:
        if self._failed:
            return False
        if self._centroids is None:
            embed = self._voiceprints.embed()
            if embed is None:
                return self._fail(f"声纹模型拿不到（{self._voiceprints.error}）")
            self._prints = _Prints(self._audio(), embed)
            try:
                self._centroids = _centroids(self._known, self._known, self._prints)
            except Exception as error:  # noqa: BLE001 - the labels degrade, the transcript does not
                return self._fail(f"声纹比对失败（{error}）")
        return True

    def _fail(self, why: str) -> bool:
        self._failed = True
        current_reporter().warning("cloud-repair-voice", f"{why}：修补补回来的话都记成 unknown")
        return False

    def nearest(self, piece: dict) -> str | None:
        """The label of the closest centroid at THETA or above for `piece` (start, end; `voiced` spans to mask
        the rest), else None."""
        import numpy as np

        if not self._ready() or not self._centroids:
            return None
        try:
            got = self._prints.of(piece, [])
        except Exception as error:  # noqa: BLE001
            self._fail(f"声纹比对失败（{error}）")
            return None
        if got is None:
            return None
        labels = list(self._centroids)
        sims = np.array([float(got[0] @ self._centroids[k]) for k in labels])
        best = int(np.argmax(sims))
        return labels[best] if sims[best] >= THETA else None

    def match(self, new: Sequence[dict]) -> dict[str, str]:
        """A re-ask's labels (its blocks, `new`) one-to-one onto the known ones by voice (`Relinker`'s rule)."""
        if not new or not self._ready():
            return {}
        try:
            # Masked only by the re-ask's own other labels: the known blocks are the same speech in the padding.
            return _match(self._centroids, _centroids(new, new, self._prints))
        except Exception as error:  # noqa: BLE001
            self._fail(f"声纹比对失败（{error}）")
            return {}


def _match(known: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> dict[str, str]:
    if not known or not new:
        return {}
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    old, fresh = list(known), list(new)
    sim = np.array([[float(new[n] @ known[k]) for k in old] for n in fresh])
    rows, cols = linear_sum_assignment(-sim)
    return {fresh[r]: old[c] for r, c in zip(rows, cols) if sim[r, c] >= THETA}

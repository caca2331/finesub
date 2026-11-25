"""Speakers by voice, for the joint-ASR backends.

MOSS (Pro and local) numbers its speakers per call, and within one call often splits one person into several labels.
Once a MOSS transcript is put together -- its parts stitched (each part's labels kept apart, `stitch`), each
gap re-request's labels renamed apart (`renumber`) -- `merge_by_voice` merges, over the whole transcript, the
labels that are one person: a WeSpeaker ResNet34 voiceprint (the model the default speaker diarization uses; on
the CPU, local MOSS's on CUDA) per segment, put in PLDA space; per label, the mean weighted by clean seconds; then
`merge_labels` at MERGE_THETA. Before that, a re-request's label with too little voice to tell becomes unknown
(`unknown_rescues`). Without the model nothing is merged and every re-request's labels stay new people.

The research behind it is `docs/joint-asr.md`「设计取舍」: across parts, windows and re-requests this one step beat
matching re-requests one-to-one by voice and clustering windows with a same-window ban, since it also merges the
labels one call split a person into.

35t, MAI2 and Scribe are not merged: at no cosine were their merges right often enough (owner 2026-11-21,
`docs/joint-asr.md`「设计取舍」); the speaker judge merges them, the close pairs shown as a hint (`voices`).

`Voices` serves the word-level backends' repairs (35t, MAI2, Scribe).
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

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
#: `merge_by_voice`'s: of the merges at 0.90 and above about 1% joined two people, at 0.75-0.90 about 9% (owner
#: 2026-11-21, measured per backend on real seams; MOSS Pro's set as local MOSS's). The rest go to the speaker judge.
MERGE_THETA = 0.9
#: `Voices`' match of a repair's piece or label to a known one.
THETA = 0.8
#: A re-request's label (`renumber`'s `c<k>:`) with less voice than this (clean seconds, weighed as `centroids`
#: weighs them) is nobody it can be told to be: its segments become unknown, for the speaker judge to place. On
#: pieces cut short of known labels, a match at cosine >= 0.9 was wrong about 1% of the time from 5 s and 9% at
#: 3-5 s (owner 2026-11-19).
RESCUE_MIN = 5.0
#: The key of a re-request's segment holding the label of the segment it says again (local MOSS's gap windows,
#: `moss_local.rules.twin`): the label it takes instead of unknown.
TWIN = "twin"
_RESCUE = re.compile(r"(?:^|:)c\d+:")
#: What an unmerged MOSS transcript leaves: every re-request's labels are new people (`renumber`).
REUSE_UNMERGED = "，续调、补调回来的都当新人"
#: Two labels talking over each other for more than this share of the smaller one's time are two people.
MERGE_OVERLAP = 0.1
PLDA_DIM = 128
#: `Voices` takes each label's voice from at most this many of its longest blocks (an hour has hundreds).
MAX_BLOCKS = 20

if TYPE_CHECKING:
    import numpy as np

#: piece (float32 at SR), mask (one per 10 ms frame) -> unit vector in PLDA space.
Embed = Callable[["np.ndarray", "np.ndarray"], "np.ndarray"]


class Voiceprints:
    """The voiceprint model on `device`, loaded on first use; `embed` is `None` when it cannot be had (not retried
    until `release`; `error` says why). A GPU model that cannot be had falls back to the CPU one, with a warning."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._lock = threading.Lock()
        self._loaded = False
        self._embed: Embed | None = None
        self.error: str | None = None

    def embed(self) -> Embed | None:
        with self._lock:
            if not self._loaded:
                self._loaded = True
                try:
                    self._embed = _wespeaker(self._device)
                except Exception as error:  # noqa: BLE001 - the labels degrade, the transcript does not
                    self.error = str(error)
                    if self._device != "cpu":
                        try:
                            self._embed = _wespeaker("cpu")
                            current_reporter().warning(
                                "moss-relink", f"声纹模型在 {self._device} 上用不了（{error}），改在 CPU 上算")
                        except Exception as cpu_error:  # noqa: BLE001 - as above
                            self.error = f"{error}; on the CPU: {cpu_error}"
        return self._embed

    def release(self) -> None:
        """Drop the model (loaded again on next use). Local MOSS's GPU copy goes before the next long pass and
        before the GPU lock goes: `decoder.fits` budgets by `memory_allocated()`, so a resident model would make a
        pass's length depend on what ran before it."""
        with self._lock:
            self._loaded, self._embed, self.error = False, None, None


def _wespeaker(device: str) -> Embed:
    """On CUDA, as the default diarization embeds: TorchScript fusion and TF32 off (`diarizen_runtime`)."""
    import contextlib

    import numpy as np
    import torch

    from ..diarization.backend import artifact_candidates
    from ..diarization.diarizen_cluster import load_plda
    from ..diarization.diarizen_features import compute_fbank
    from ..diarization.diarizen_runtime import jit_fusion_disabled, tf32_disabled
    from ..diarization.nemotron3_backend import HOSTED_EMBEDDING, PLDA_FILES, embedding_file
    from ..runtime.native_paths import load_torchscript

    name = embedding_file(device)
    files = (name, *PLDA_FILES)
    directory = next((c for c in artifact_candidates(None) if all((c / f).exists() for f in files)), None)
    if directory is None:
        from ..runtime import hf_weights

        directory = hf_weights.prepare_directory(HOSTED_EMBEDDING)
        if directory is None:
            raise RuntimeError("could not fetch the speaker embedding model")
    model = load_torchscript(directory / name, map_location=device).eval()
    xvec_tf, plda_tf, _ = load_plda(directory / "plda")
    cuda = device.startswith("cuda")

    def embed(piece: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with (torch.inference_mode(), jit_fusion_disabled() if cuda else contextlib.nullcontext(),
              tf32_disabled() if cuda else contextlib.nullcontext()):
            fbank = compute_fbank(torch.from_numpy(piece).to(device).reshape(1, 1, -1))
            raw = model(fbank, torch.from_numpy(mask).to(device).reshape(1, -1)).cpu().numpy().astype(np.float64)
        x = plda_tf(xvec_tf(raw), lda_dim=PLDA_DIM)[0]
        return x / np.linalg.norm(x)

    return embed


#: The process's one copy: batch items on the network lane share it.
VOICEPRINTS = Voiceprints()
#: Local MOSS's: it embeds holding the GPU lock, between passes, while the card is otherwise idle (a cloud backend
#: runs beside other GPU work, so it stays on the CPU).
VOICEPRINTS_CUDA = Voiceprints("cuda")


class Prints:
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


def _weighed(segs: Sequence[dict], every: Sequence[dict], prints: Prints):
    """(label, voiceprint, weight) of each labelled segment of `segs` long enough to embed (masked by `every`)."""
    for s in segs:
        if s["speaker"] is None:
            continue
        got = prints.of(s, every)
        if got is not None:
            yield s["speaker"], got[0], min(got[1], WEIGHT_CAP)


def centroids(segs: Sequence[dict], every: Sequence[dict], prints: Prints,
              least: float = MIN_LABEL) -> dict[str, np.ndarray]:
    """Each label's mean voiceprint (unit vector), over labels with `least` s of embedded speech or more."""
    import numpy as np

    sums: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    for lab, vec, w in _weighed(segs, every, prints):
        sums[lab] = sums.get(lab, 0) + w * vec
        weights[lab] = weights.get(lab, 0.0) + w
    return {lab: v / np.linalg.norm(v) for lab, v in sums.items() if weights[lab] >= least}


def unknown_rescues(segs: list[dict], prints: Prints) -> list[tuple[str, float]]:
    """Sets the segments of each re-request's label with under RESCUE_MIN s of voice to unknown (None), in place,
    or to the label of the segment they say again (TWIN) unless that label is one of these or a re-request's label
    no segment carries any more (a later window replaced them all: its voice cannot be told either); returns those
    labels and their seconds."""
    seconds = {s["speaker"]: 0.0 for s in segs if s["speaker"] is not None and _RESCUE.search(s["speaker"])}
    for lab, _, w in _weighed([s for s in segs if s["speaker"] in seconds], segs, prints):
        seconds[lab] += w
    short = {lab: round(t, 1) for lab, t in seconds.items() if t < RESCUE_MIN}
    for s in segs:
        if s["speaker"] in short:
            twin = s.get(TWIN)
            gone = twin is not None and _RESCUE.search(twin) and twin not in seconds
            s["speaker"] = None if twin in short or gone else twin
    return sorted(short.items())


def renumber(extra: Sequence[dict], k: int) -> dict[str, str]:
    """Renames the k-th re-request's labels apart from the transcript's (`c<k>:<label>`), in place; returns the map.
    `merge_by_voice` later merges the ones that are someone already heard."""
    mapping = {lab: f"c{k}:{lab}" for lab in sorted({s["speaker"] for s in extra if s["speaker"] is not None})}
    for s in extra:
        s["speaker"] = mapping.get(s["speaker"], s["speaker"])
    return mapping


_PART = re.compile(r"^w(\d+):")


def _origin(label: str) -> tuple[bool, int]:
    """Where a label was first heard, earlier first: a re-ask's (`c<k>:`) after the main transcript's, a later
    part's (`w<k>:`) after an earlier one's."""
    part = _PART.match(label)
    return _RESCUE.search(label) is not None, int(part.group(1)) if part else 0


def merge_labels(segs: list[dict], centroids: Callable[[list[dict]], Mapping[str, np.ndarray]],
                 theta: float) -> list[tuple[str, str, float]]:
    """Relabels `segs` in place and returns the merges (absorbed, kept, cosine): repeatedly, of the labels'
    voiceprint `centroids` (unit vectors, recomputed after each merge), the closest pair at cosine >= `theta` goes
    into the one heard first (`_origin`: the main transcript's before a re-ask's, an earlier part's before a later
    one's; the judge's profile says where a label came from), the one with more speech on a tie -- unless the two
    talk over each other for more than MERGE_OVERLAP of the smaller one's time (one person does not overlap
    themselves)."""
    import itertools

    talk: dict[str, float] = {}
    for s in segs:
        talk[s["speaker"]] = talk.get(s["speaker"], 0.0) + s["end"] - s["start"]

    def overlap(a: str, b: str) -> float:
        xa = [(s["start"], s["end"]) for s in segs if s["speaker"] == a]
        xb = [(s["start"], s["end"]) for s in segs if s["speaker"] == b]
        return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in xa for s2, e2 in xb)

    merges = []
    while True:
        cents = centroids(segs)
        best = None
        for a, b in itertools.combinations(sorted(cents), 2):
            sim = float(cents[a] @ cents[b])
            if sim >= theta and (best is None or sim > best[2]) \
                    and overlap(a, b) <= MERGE_OVERLAP * min(talk[a], talk[b]):
                best = (a, b, sim)
        if best is None:
            return merges
        a, b, sim = best
        keep, gone = (a, b) if (_origin(a), -talk[a]) <= (_origin(b), -talk[b]) else (b, a)
        for s in segs:
            if s["speaker"] == gone:
                s["speaker"] = keep
        talk[keep] += talk.pop(gone)
        merges.append((gone, keep, round(sim, 3)))


def merge_by_voice(segs: list[dict], audio: Callable[[], np.ndarray], voiceprints: Voiceprints, backend: str,
                   theta: float = MERGE_THETA, unmerged: str = "",
                   ) -> tuple[list[tuple[str, str, float]], list[tuple[str, float]]]:
    """`unknown_rescues`, then `merge_labels` at `theta`, over the whole transcript `segs` (in place; `audio` gives
    its int16 PCM at SR); returns the merges and the labels made unknown. Without the model, or if comparing fails,
    nothing is merged and the report says so (`unmerged`: what that leaves, after the backend's name)."""
    labels = {s["speaker"] for s in segs if s["speaker"] is not None}
    if len(labels) < 2 and not any(_RESCUE.search(lab) for lab in labels):
        return [], []
    embed = voiceprints.embed()
    if embed is None:
        current_reporter().warning(
            "joint-voice-merge", f"声纹模型拿不到（{voiceprints.error}）：{backend} 的说话人不按声音合并{unmerged}")
        return [], []
    prints = Prints(audio(), embed)
    unknown: list[tuple[str, float]] = []
    try:
        unknown = unknown_rescues(segs, prints)
        return merge_labels(segs, lambda s: centroids(s, s, prints), theta), unknown
    except Exception as error:  # noqa: BLE001 - the labels degrade, the transcript does not
        current_reporter().warning("joint-voice-merge", f"声纹比对失败（{error}）：{backend} 的说话人不按声音合并")
        return [], unknown


def linked(parts: Sequence[tuple[float, float, list[dict]]], *, segments: bool, audio: Callable[[], np.ndarray],
           backend: str, unmerged: str = "", orphans: str | None = None,
           ) -> tuple[list[dict], list[tuple[str, str, float]], list[tuple[str, float]]]:
    """Local MOSS's speakers across its parts, for both backends that decode with it: the parts stitched (`stitch`:
    each part's labels kept apart), then `merge_by_voice` at MERGE_THETA over the whole, on CUDA, and the rescue twins
    (TWIN) dropped. -> (units, merges, the re-requests' labels made unknown). `segments`: `moss-local`'s free decodes
    (True: each seam reconciled by text) or the forced decode's one set of lines (False, with `orphans` its line
    field). How MOSS links people across parts is changed here, for both."""
    from .stitch import stitch

    units = parts[0][2] if len(parts) == 1 else stitch(parts, segments=segments, orphans=orphans)
    merges, unknown = merge_by_voice(units, audio, VOICEPRINTS_CUDA, backend, unmerged=unmerged)
    for u in units:
        u.pop(TWIN, None)
    return units, merges, unknown


class Voices:
    """Speakers by voice for the repairs of a word-level transcript (35t, MAI2, Scribe; `repair.py`), against the labels of
    `known` -- blocks (`repair.blocks`) of the speech already heard. `nearest` is not one-to-one:
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
        self._prints: Prints | None = None
        self._centroids: dict[str, np.ndarray] | None = None
        self._failed = False

    def _ready(self) -> bool:
        if self._failed:
            return False
        if self._centroids is None:
            embed = self._voiceprints.embed()
            if embed is None:
                return self._fail(f"声纹模型拿不到（{self._voiceprints.error}）")
            self._prints = Prints(self._audio(), embed)
            try:
                self._centroids = centroids(self._known, self._known, self._prints)
            except Exception as error:  # noqa: BLE001 - the labels degrade, the transcript does not
                return self._fail(f"声纹比对失败（{error}）")
        return True

    def _fail(self, why: str) -> bool:
        self._failed = True
        current_reporter().warning("cloud-repair-voice", f"{why}：修补补回来的话都记成 unknown")
        return False

    def nearest(self, piece: dict) -> str | None:
        """The label of the closest centroid at THETA or above for `piece` (start, end; `voiced` spans to mask
        the rest), else None; None too for a piece with under RESCUE_MIN s of voice (owner 2026-11-21)."""
        import numpy as np

        if not self._ready() or not self._centroids:
            return None
        try:
            got = self._prints.of(piece, [])
        except Exception as error:  # noqa: BLE001
            self._fail(f"声纹比对失败（{error}）")
            return None
        if got is None or got[1] < RESCUE_MIN:
            return None
        labels = list(self._centroids)
        sims = np.array([float(got[0] @ self._centroids[k]) for k in labels])
        best = int(np.argmax(sims))
        return labels[best] if sims[best] >= THETA else None

    def match(self, new: Sequence[dict]) -> dict[str, str]:
        """A re-ask's labels (its blocks, `new`) one-to-one onto the known ones by voice (Hungarian, THETA); a label
        with under RESCUE_MIN s of voice is matched to no one (owner 2026-11-21, as `unknown_rescues`)."""
        if not new or not self._ready():
            return {}
        try:
            # Masked only by the re-ask's own other labels: the known blocks are the same speech in the padding.
            return _match(self._centroids, centroids(new, new, self._prints, RESCUE_MIN))
        except Exception as error:  # noqa: BLE001
            self._fail(f"声纹比对失败（{error}）")
            return {}


    def voiced(self, new: Sequence[dict]) -> set[str] | None:
        """The labels of a re-ask's blocks (`new`) with RESCUE_MIN s of voice or more: who may be a new person. None
        when there are no voiceprints to tell (the model missing or failing): then any of them may be."""
        if not new:
            return set()
        if not self._ready():
            return None
        try:
            return set(centroids(new, new, self._prints, RESCUE_MIN))
        except Exception as error:  # noqa: BLE001
            self._fail(f"声纹比对失败（{error}）")
            return None


def _match(known: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> dict[str, str]:
    if not known or not new:
        return {}
    import numpy as np

    old, fresh = list(known), list(new)
    sim = np.array([[float(new[n] @ known[k]) for k in old] for n in fresh])
    rows, cols = _assign(sim)
    return {fresh[r]: old[c] for r, c in zip(rows, cols) if sim[r, c] >= THETA}


def _assign(sim: np.ndarray) -> tuple[Sequence[int], Sequence[int]]:
    """The one-to-one pairing of `sim`'s rows and columns with the largest total (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    return linear_sum_assignment(-sim)

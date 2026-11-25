"""How alike a cloud task's speaker labels sound: each `line_speaker` label's voiceprint, compared pairwise.

The speaker judge shows the close pairs as a hint, in its backend's tiers (`TIERS`), and its file-wide merges pass
a gate on them (`docs/llm_harness_behavior.md`「说话人裁判」). Made the first time the judge needs it, not by the joint stage: a run that
stops at the raw subtitles never reads it, and a task resumed from before it existed gets it too. Written beside the
stable as `<stem>-joint-voices.json`, keyed on the rows and their labels (`identity`); made again only when gone,
unreadable or made for other rows or another assignment of labels. It holds the tiers it was made with, so a
resumed task shows what it first showed.

The voiceprints are `relink`'s, per label over its lines (a line's 15 s middle at most, its frames that another line
overlaps masked, weighted by clean seconds; a label with under `relink.MIN_LABEL` clean seconds has none). The
overlap share is how long two labels talk at once over the shorter one's talk -- one person does not talk over
themselves, so `relink.MERGE_OVERLAP` of it rules a pair out whatever the voices say.

Version 3 (experiment v23, 2026-11-23) also keeps each label's voiceprint and each row's (its vector and clean
seconds, in the stable's row order; None for a row with no label or too short to embed), so that any set of rows --
a piece of a track the judge cut, a track it opened -- has a voiceprint of its own without decoding the audio again
(`piece`): the rows' prints weighted as `relink.centroids` weighs them. A label's print is its rows' (`compare`).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import GEMINI_35T, MAI2, MOSS_DIAR_ALIGNER, MOSS_LOCAL, MOSS_PRO, SCRIBE, relink
from .model import UNKNOWN_SPEAKER

VOICES_SUFFIX = "-joint-voices.json"
VERSION = 3
#: Decimals a stored voiceprint keeps
DIGITS = 4
#: The judge's tiers per backend, (高 from, 中 from or None: no 中): a pair at or above 高 is shown as 高, at or
#: above 中 as 中; lower pairs are not shown. Set per backend by how often its labels' merges joined two people on
#: real seams -- about 10% at most for 高, 35% for 中 (owner 2026-11-21, `docs/joint-asr.md`「设计取舍」). The MOSS
#: backends merge 0.9 and above before the judge (`relink.MERGE_THETA`); MOSS Pro's and the forced decode's
#: (`moss-diarization-aligner`, local MOSS's labels on another backend's lines) are set as local MOSS's, unmeasured.
TIERS = {MOSS_LOCAL: (0.75, None), MOSS_PRO: (0.75, None), MOSS_DIAR_ALIGNER: (0.75, None), MAI2: (0.90, 0.75),
         GEMINI_35T: (0.95, 0.75), SCRIBE: (0.80, None)}
#: Below this the merge gate refuses a pair whose labels both have a voiceprint (offline 2026-11-16: one person's
#: labels compared 0.66-0.99, so the floor sits under that, where a short label's print is still believed).
FLOOR = 0.5


def voices_path(stable_json: str | Path) -> Path:
    stable = Path(stable_json)
    return stable.with_name(f"{stable.name.removesuffix('.json').removesuffix('-stable')}{VOICES_SUFFIX}")


def compare(lines: Sequence[Mapping[str, Any]], pcm=None, embed: relink.Embed | None = None) -> dict[str, Any]:
    """`{"labels": {label: {"talk", "printed", "vector"?}}, "pairs": [{"a", "b", "cos", "overlap"}], "rows": [...]}`
    for the labelled lines (`start`, `end`, `line_speaker`; `unknown` is no one). `pcm` is the source's int16 PCM at
    `relink.SR`; `cos` is None when either label has no voiceprint, and for every pair without `pcm` and `embed`.
    `rows` holds each line's print (`row_prints`), all None without them."""
    segs = [{"start": float(line["start"]), "end": float(line["end"]), "speaker": str(line["line_speaker"])}
            if line.get("line_speaker") not in (None, UNKNOWN_SPEAKER) and float(line["end"]) > float(line["start"])
            else None for line in lines]
    labelled = [seg for seg in segs if seg is not None]
    talk: dict[str, float] = {}
    spans: dict[str, list[tuple[float, float]]] = {}
    for s in labelled:
        talk[s["speaker"]] = talk.get(s["speaker"], 0.0) + s["end"] - s["start"]
        spans.setdefault(s["speaker"], []).append((s["start"], s["end"]))
    prints = relink.Prints(pcm, embed) if labelled and embed is not None else None
    cents = relink.centroids(labelled, labelled, prints) if prints is not None else {}
    pairs = []
    for a, b in itertools.combinations(sorted(talk), 2):
        both = overlap_seconds(spans[a], spans[b])
        cos = round(float(cents[a] @ cents[b]), 3) if a in cents and b in cents else None
        pairs.append({"a": a, "b": b, "cos": cos, "overlap": round(both / max(1e-9, min(talk[a], talk[b])), 3)})
    return {"labels": {label: {"talk": round(talk[label], 1), "printed": label in cents,
                               **({"vector": _stored(cents[label])} if label in cents else {})}
                       for label in sorted(talk)},
            "pairs": pairs, "rows": _rows(segs, labelled, prints)}


def row_prints(lines: Sequence[Mapping[str, Any]], pcm, embed: relink.Embed) -> list[dict[str, Any] | None]:
    """Each line's voiceprint as `compare` stores it (`{"vector", "seconds"}`, None for a line with no label or too
    short to embed), its frames that another labelled line overlaps masked: what a caller whose labels' prints come
    from elsewhere stores for its own rows (the experiment's windows, a span of a whole source)."""
    return compare(lines, pcm, embed)["rows"]


def _rows(segs: Sequence[dict | None], labelled: Sequence[dict], prints: relink.Prints | None
          ) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    for seg in segs:
        got = prints.of(seg, labelled) if prints is not None and seg is not None else None
        out.append(None if got is None else {"vector": _stored(got[0]), "seconds": round(got[1], 2)})
    return out


def _stored(vector) -> list[float]:
    return [round(float(x), DIGITS) for x in vector]


def label_print(voices: Mapping[str, Any] | None, label: str):
    """A label's stored voiceprint (unit length), None when it has none."""
    import numpy as np

    vector = (((voices or {}).get("labels") or {}).get(label) or {}).get("vector")
    return None if vector is None else _unit(np.asarray(vector, dtype=np.float32))


def piece(voices: Mapping[str, Any] | None, indices: Iterable[int]):
    """The voiceprint of a set of rows (indices into the stable's rows): their prints weighted by clean seconds
    (capped at `relink.WEIGHT_CAP`), as `relink.centroids` weighs a label's; None under `relink.MIN_LABEL` clean
    seconds, or without stored rows."""
    import numpy as np

    rows = (voices or {}).get("rows") or []
    total, weight = None, 0.0
    for index in indices:
        got = rows[index] if 0 <= index < len(rows) else None
        if got is None:
            continue
        w = min(float(got["seconds"]), relink.WEIGHT_CAP)
        vector = np.asarray(got["vector"], dtype=np.float32) * w
        total = vector if total is None else total + vector
        weight += w
    return None if total is None or weight < relink.MIN_LABEL else _unit(total)


def _unit(vector):
    import numpy as np

    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None


def cosine(a, b) -> float | None:
    """Two voiceprints' cosine (both unit length), None when either is missing."""
    return None if a is None or b is None else round(float(a @ b), 3)


def overlap_seconds(a: Iterable[tuple[float, float]], b: Iterable[tuple[float, float]]) -> float:
    """How long spans `a` and spans `b` run at once, each span pair's overlap summed (as a double loop would), in
    one sweep over both sorted."""
    a, b = sorted(a), sorted(b)
    total, start = 0.0, 0
    for s1, e1 in a:
        while start < len(b) and b[start][1] <= s1:  # ends before this span and every later one starts
            start += 1
        for s2, e2 in b[start:]:
            if s2 >= e1:
                break
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def identity(segments: Sequence[Mapping[str, Any]]) -> str:
    """What a comparison was made for: the rows (`judgments.rows_hash`, which leaves the speakers out) and each row's
    label -- the same rows labelled again sound different per label."""
    from ..diarization.judgments import rows_hash

    labels = json.dumps([str(segment.get("line_speaker") or "") for segment in segments], ensure_ascii=False)
    return f"{rows_hash(segments)}:{hashlib.sha256(labels.encode('utf-8')).hexdigest()[:16]}"


def read(stable_json: str | Path, segments: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The stored comparison when it was made for these rows and labels (`identity`), else None."""
    key = identity(segments)
    path = voices_path(stable_json)
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict) and payload.get("version") == VERSION and payload.get("key") == key:
        return payload
    return None


def ensure(stable_json: str | Path, audio: str | Path | None, *, warn: Callable[[str], None],
           voiceprints: relink.Voiceprints = relink.VOICEPRINTS) -> dict[str, Any]:
    """The comparison for this stable: the stored one, else made from `audio` (the source the task transcribed) and
    written. Without audio or the voiceprint model, or when reading the audio or embedding fails, it is written all
    the same, with no cosines and why (`unavailable`), and a warning: the judge then gets no hints and its merges no
    voice gate -- a hint never stops the judge. Written either way so that a resumed task reads what it first read: a
    merge near the floor must not flip because the model came back. On the CPU, as the joint stage's own voice
    merges (`relink.VOICEPRINTS`)."""
    from ...subtitles.stable_rows import read_rows
    from .stage import recorded_backend

    segments = read_rows(stable_json)
    key = identity(segments)
    stored = read(stable_json, segments)
    if stored is not None:
        return stored
    high, mid = TIERS[recorded_backend(json.loads(Path(stable_json).read_text(encoding="utf-8")))]
    made = {"version": VERSION, "key": key, "tiers": {"high": high, "mid": mid}}
    embed = voiceprints.embed() if audio is not None else None
    why = "没有源音频" if audio is None else None if embed is not None else f"声纹模型拿不到（{voiceprints.error}）"
    payload = None
    if why is None:
        import numpy as np

        from ...media.ffmpeg import decode_pcm16k

        try:
            pcm = np.frombuffer(decode_pcm16k(Path(audio)), np.int16)
            payload = {**made, **compare(segments, pcm, embed)}
        except Exception as error:  # noqa: BLE001 - a hint: the judge goes on without it
            why = f"声纹没算成（{type(error).__name__}: {error}）"
    if payload is None:
        warn(f"{why}：说话人裁判这个任务不给轨之间的声音相似度，合并也不经声纹把关（删掉 {voices_path(stable_json).name} 可重算）")
        payload = {**made, "unavailable": why, **compare(segments)}
    voices_path(stable_json).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def pairs(voices: Mapping[str, Any] | None) -> Callable[[str, str], Mapping[str, Any] | None]:
    """A lookup of the stored pair of two labels, either order; None when not compared."""
    index = {frozenset((pair["a"], pair["b"])): pair for pair in (voices or {}).get("pairs") or ()}
    return lambda a, b: index.get(frozenset((a, b)))


def tier(pair: Mapping[str, Any] | None, tiers: Mapping[str, float | None]) -> str | None:
    """`high`, `mid` or None (not shown): by the cosine against `tiers` (a comparison's, `TIERS`), never for two
    labels that talk at once."""
    if pair is None or pair.get("cos") is None or pair["overlap"] > relink.MERGE_OVERLAP:
        return None
    if pair["cos"] >= tiers["high"]:
        return "high"
    return "mid" if tiers["mid"] is not None and pair["cos"] >= tiers["mid"] else None


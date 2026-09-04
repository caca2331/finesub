"""P18: does a text-side sentence-boundary model tell `T(k)` anything new?

`docs/plans/crispasr-followups.md` -> P18 approves ONE thing: an offline feasibility
check. Not wiring, not a knob, not a calibration. The question is narrow and
was written down BEFORE any number existed:

    In the `g = 0` bucket, how much does `p_sbd` add ON TOP OF the word-level
    pause dimension we already have?

`g = 0` is 97.9% of intra-segment boundaries (`segmentation-split.md`), and in
that bucket `T(k)` has six discrete values plus one continuous dimension --
pause. So "does it correlate with `must`" is the wrong question: punctuation
already decides the boundaries where punctuation exists, and those we already
cut correctly. Only the increment counts.

Two modes:

  features  gold labels + frozen worksheets + the substrate word stream ->
            one row per boundary in the window: pause, vad, label, the ASR
            seam, where production actually cuts, and `p_sbd`. Runs the ONNX
            model on CPU.
  gate      that table -> the three pre-registered numbers, plus prerequisite
            5 (the comparison against the prior we already own). No model.

Three things the first version of this file got wrong, all of them silent:

* the negative class was the *written-out* `never` items rather than the
  presumed `never` the gold contract defines, which shrank the negatives to
  the positions that already have a boundary signal;
* gate 3's fixed FPR was a Youden point on the baseline -- an operating point
  read off the evaluation data, i.e. a function of what was being tested;
* the `seam` prior was taken from substrates `split_segments` had already
  rewritten, which leaks a segmenter's decisions into the baseline.

`test_sbd_split.py` pins the first two; `segment_seams` returns `None` for the
third rather than answering with a contaminated feature.

⚠ **The model publishes flattering numbers that do not transfer.** Its card
reports Japanese SBD F1 99.47, but: "For measuring true-casing and sentence
boundary detection, reference punctuation tokens were used for conditioning"
and "This model was trained on news data, and may not perform well on
conversational or informal data". We feed it spoken ASR output and it must
condition on its OWN predicted punctuation. Both gaps are load-bearing, which
is why the gate is local and pre-registered.

⚠ **We never touch reference punctuation.** The input is stripped of ASR
punctuation (the pre-registered `strip` form) and the SBD head conditions on
the model's own `post_punc` embedding inside the graph. Nothing in this file
reads the gold text, the reference SRT, or the substrate's punctuation as a
model input -- `test_sbd_split.py` asserts it.

    python -m tools.bench.probe_sbd_split features --json tmp/bench/p18-v2.json
    python -m tools.bench.probe_sbd_split gate --features tmp/bench/p18-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.segmentation_gold import gold

REPO = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
#: Pinned. A `main` that moves under us would silently restate every number in
#: `bench-baselines.md` 第十九节 as if it had been measured on this model. The two
#: digests are checked as well: a revision pin trusts the hub's resolution, a
#: digest trusts nothing.
REVISION = "d1769a597ce8dfaa070d436bc67d4ee761f58884"
FILE_SHA256 = {
    "model.onnx": "c43ca686dabc237c3b06be834b9423c07580fef7e2b1a6c09976f7d60caa5d89",
    "sp.model": "7f944d0be93b275f62e1913fd409f378ddbba108e57fe4a9cb47e8c047f6bef1",
}
#: The SBD probability inside the exported graph. The published ONNX only
#: returns `seg_preds`, a BOOL -- `p > 0.05` already thresholded -- and a
#: boolean cannot feed an AUC. This tensor is its pre-threshold input, so
#: exposing it as a graph output adds no computation and changes no weights;
#: `test_sbd_split.py` asserts `seg_preds == (p > SBD_THRESHOLD)` still holds.
SBD_PROB_TENSOR = "/Gather_output_0"
SBD_THRESHOLD = 0.05
#: `max_length: 256` in the model's own config, minus <s> and </s>.
MAX_TOKENS = 254
#: Chunking deviates from CrispASR's `pcs.cpp`, which cuts every 254 tokens
#: with no overlap. That is fine for restoring punctuation and fatal here: the
#: last token of a chunk always looks like the end of the text and reads
#: p_sbd = 1.0, which would plant one fake positive per chunk. Overlapping
#: windows and taking each token's prediction from the window where it sits
#: furthest from both edges removes the artifact.
CHUNK_STRIDE = 127

#: The pre-registered input form (P18 prerequisite 2). The model was trained on
#: punctuation-stripped lowercased text, and the segments this is meant to help
#: carry no punctuation anyway (13/16 of the long ones). `as-is` would be a
#: second stratum, pre-registered separately -- not a choice made after seeing
#: numbers.
INPUT_FORM = "strip"

LABELS_DIR = pathlib.Path("tools/segmentation_gold/labels")
WORKSHEETS_DIR = pathlib.Path("tools/segmentation_gold/worksheets")
#: `must` is the positive class, `never` the negative. `ok` and `unknown` are
#: excluded entirely -- "could cut here" is not evidence either way, and
#: `unknown` marks text nobody could read.
#:
#: ⚠ The negative class is **presumed** never, which is NOT the same as "an item
#: labelled `never`". `segmentation-gold.md` §2.2: `never` is the default and the
#: labeller's duty is to declare `must`/`ok`/`unknown`; a written-out `never` is
#: *evidence*, not the class definition. `gold.evaluate_by_index` scores exactly
#: that way. So the universe below is every word boundary inside a labelled
#: window, minus the declared `must`/`ok`/`unknown` -- undeclared positions are
#: negatives too, and they are the majority.
POSITIVE, NEGATIVE = "must", "never"


def strip_punctuation(text: str) -> str:
    """The `strip` input form: no punctuation, no spaces, lowercased.

    Lowercasing is a no-op for Japanese but the model was trained on lowercased
    text, so a latin run inside a Japanese line should not look novel.
    """

    return "".join(
        ch.lower()
        for ch in text
        if not unicodedata.category(ch).startswith("P") and not ch.isspace()
    )


# --- features ------------------------------------------------------------------
_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
                  r"\s*([\d.]+)\s*\|\s*(\S*)\s*\|")


def worksheet_rows(path: pathlib.Path) -> dict[int, dict]:
    """`i -> {k, t, pause, vad, punct}` from a FROZEN worksheet.

    Read rather than recomputed on purpose: the labels reference `i` in this
    exact table, and recomputing `pause`/`vad` from the substrate would silently
    re-derive the very columns the labeller saw.
    """

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        i, k, t, pause, vad, punct = m.groups()
        out[int(i)] = {
            "k": int(k),
            "t": float(t),
            "pause": float(pause),
            "vad": float(vad),
            "punct": bool(punct.strip()),
        }
    return out


def find_worksheet(clip: str, window) -> pathlib.Path | None:
    a, b = int(window[0]), int(window[1])
    for name in (f"ws-{clip}-{a}-{b}.md", f"ws-{clip}.md"):
        path = WORKSHEETS_DIR / name
        if path.exists():
            return path
    return None


def load_boundaries() -> list[dict]:
    """One row per boundary in the pre-registered universe.

    `must` -> positive; `ok` / `unknown` / anything inside an `unknown` span ->
    dropped; **everything else in the window -> negative**, declared or not
    (see the POSITIVE/NEGATIVE note above). The window is `[min k, max k]` over
    the labelled items, the same range `gold.evaluate_by_index` scores in.

    Features for a declared position come from the FROZEN worksheet -- the
    columns the labeller actually saw. An undeclared position is by
    construction not a candidate, and the candidate rule
    (`gold.candidates`: `pause >= CAND_PAUSE or vad > 0 or punct`) then forces
    `vad = 0`, `punct = False`, `pause < CAND_PAUSE`. Only the pause is
    recomputed from the substrate, and the loader checks the implication rather
    than assuming it.

    Rows are deduplicated by `(clip, k)`: `BV1nxje63ERi` and `yingtao` have
    overlapping gold windows, and the same physical boundary appearing in two
    groups would put it on both sides of a fold.
    """

    rows = []
    seen: set[tuple[str, int]] = set()
    duplicates = 0
    off_rule = 0
    for label_path in sorted(LABELS_DIR.glob("*.json")):
        g = json.loads(label_path.read_text(encoding="utf-8"))
        substrate = pathlib.Path(g["substrate_path"])
        if not substrate.exists():
            print(f"  skip {label_path.name}: substrate missing")
            continue
        digest = hashlib.sha256(substrate.read_bytes()).hexdigest()[:12]
        if digest != g["substrate_sha"]:
            print(f"  skip {label_path.name}: substrate sha {digest} != {g['substrate_sha']}")
            continue
        sheet = find_worksheet(g["clip"], g["window"])
        if sheet is None:
            print(f"  skip {label_path.name}: no frozen worksheet")
            continue
        table = worksheet_rows(sheet)
        by_k = {entry["k"]: entry for entry in table.values()}
        words = gold.load_words(substrate)
        seams = segment_seams(substrate)
        cuts, cuts_stale = production_cuts(substrate, g["clip"])
        window = label_path.stem

        declared: dict[int, str] = {}
        spans: list[tuple[int, int]] = []
        for item in g["items"]:
            if "span_k" in item:
                a, b = item["span_k"]
                spans.append((int(a), int(b)))
                continue
            k = item.get("k")
            if k is None and "i" in item:
                k = (table.get(item["i"]) or {}).get("k")
            if k is not None:
                declared[int(k)] = item.get("label")
        if not declared:
            print(f"  skip {label_path.name}: no positional items")
            continue

        lo, hi = min(declared), max(declared)
        kept = presumed = 0
        for k in range(lo, hi + 1):
            label = declared.get(k)
            if label in ("ok", "unknown") or any(a <= k <= b for a, b in spans):
                continue
            if k + 1 >= len(words):
                continue
            entry = by_k.get(k)
            if entry is None:
                pause = max(0.0, words[k + 1]["start"] - words[k]["end"])
                t = (words[k]["end"] + words[k + 1]["start"]) / 2
                vad, punct = 0.0, False
                # The implication that lets us fill vad/punct in without the
                # VAD track. If it ever fails the candidate set and the label
                # file disagree about what the substrate is.
                off_rule += pause >= gold.CAND_PAUSE
                presumed += 1
            else:
                pause, t = entry["pause"], entry["t"]
                vad, punct = entry["vad"], entry["punct"]
            key = (g["clip"], k)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append({
                "group": g["clip"],
                "window": window,
                "clip": g["clip"],
                "k": k,
                "t": round(t, 2),
                "pause": round(pause, 3),
                "vad": vad,
                "punct": punct,
                "seam": None if seams is None else int((k + 1) in seams),
                "prod_cut": int((k + 1) in cuts),
                "prod_stale": cuts_stale,
                "label": POSITIVE if label == POSITIVE else NEGATIVE,
                "declared": label is not None,
                "substrate": str(substrate),
            })
            kept += 1
        print(f"  {window}: {kept} rows in [{lo},{hi}] "
              f"({presumed} presumed never, seam="
              f"{'raw' if seams is not None else 'contaminated'})")
    if duplicates:
        print(f"  deduplicated {duplicates} boundaries shared by overlapping windows")
    if off_rule:
        print(f"  ⚠ {off_rule} undeclared positions have pause >= CAND_PAUSE "
              f"-- the candidate rule and the label file disagree")
    return rows


def _segment_starts(segments) -> set[int]:
    """Word indices that start a segment, counted over the same non-empty word
    stream `gold.load_words` produces."""

    starts, index = set(), 0
    for segment in segments:
        words = [
            w for w in (segment.get("words") or [])
            if str(w.get("word", "")).strip()
            and not w.get("synthetic_from_segment")
        ]
        if words:
            starts.add(index)
            index += len(words)
    return starts


def _split_ran(substrate: pathlib.Path) -> bool:
    """Whether `split_segments` has already been applied to this artifact.

    It records itself in `metadata.asr_align.segment_split` (`vad_asr_stage`),
    so this is the artifact's own statement, not an inference from its shape.
    """

    data = json.loads(substrate.read_text(encoding="utf-8"))
    return "segment_split" in (data.get("metadata", {}).get("asr_align") or {})


def segment_seams(substrate: pathlib.Path) -> set[int] | None:
    """Word indices carrying the ASR segment seam -- the prior production
    already owns as `whisper_segment_bonus`, or `None` if this artifact cannot
    say.

    `WHISPER_SEGMENT_WORD_TAG` marks the first word of every segment that was
    **handed to** `split_segments`. The tag itself postdates these substrates,
    but on an artifact `split_segments` has not touched, its segment starts ARE
    that input -- so the feature here is the prior itself, not a stand-in.

    ⚠ On an artifact that HAS been split, the segment starts are the DP's
    output: mostly seams, plus its interior cuts, minus the seams it swallowed.
    Those cuts were decided partly from pause and VAD, i.e. from the baseline's
    own features, so such a substrate would leak the answer into the prior.
    Nine of the fourteen gold windows (the eight `BV*` clips) are un-split;
    `yui`/`yingtao`/`kaguya60` are not, and return `None`.
    """

    if _split_ran(substrate):
        return None
    data = json.loads(substrate.read_text(encoding="utf-8"))
    return _segment_starts(data["segments"])


def _vad_intervals(substrate: pathlib.Path, clip: str):
    path = substrate.parent / f"{clip}-vad.json"
    if not path.exists():
        raise SystemExit(f"no VAD intervals for {clip} at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"start": float(a), "end": float(b)} for a, b in data["intervals"]]


def production_cuts(substrate: pathlib.Path, clip: str) -> tuple[set[int], bool]:
    """`(cuts, stale)` -- where the segmenter cuts this window, and whether
    that is TODAY'S segmenter.

    Gate 3's operating point is production's own mis-cut rate, so the cuts have
    to come from the code that ships now:

    * un-split substrate -> run `split_segments` with `DEFAULT_SPLIT_PARAMS`
      over its own VAD intervals. That is the production call
      (`vad_asr_stage.py:1147`), not a reimplementation. `stale = False`.
    * already split, and its recorded params match today's schema -> its
      segment starts ARE production's output. `stale = False`.
    * already split by an OLDER segmenter -> `stale = True`. The stored cuts
      are a historical system's, and today's code cannot be re-run on them
      either: regrouping keys off `WHISPER_SEGMENT_WORD_TAG`, which those
      artifacts predate, so the pre-split segments are unrecoverable. The rows
      are kept and flagged; `cmd_gate` reports gate 3 with and without them.

    ⚠ The schema comparison is the whole check. `yui`/`yingtao`/`kaguya60`
    record `no_gap_penalty` and carry no `whisper_segment_bonus` --
    i.e. they were cut before the global DP with the seam discount existed,
    which is exactly the prior this probe is measuring against.
    """

    data = json.loads(substrate.read_text(encoding="utf-8"))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
    from finesub.speech.postprocessing.segmentation import (
        DEFAULT_SPLIT_PARAMS, split_params_metadata, split_segments,
    )
    stored = (data.get("metadata", {}).get("asr_align") or {}).get("segment_split")
    if stored is not None:
        current = set(split_params_metadata(DEFAULT_SPLIT_PARAMS))
        stale = set(stored) != current
        if stale:
            missing = ", ".join(sorted(current - set(stored))) or "-"
            extra = ", ".join(sorted(set(stored) - current)) or "-"
            print(f"  ⚠ {substrate.name}: cuts are from an older segmenter "
                  f"(missing {missing}; obsolete {extra})")
        return _segment_starts(data["segments"]), stale
    pieces = split_segments(
        data["segments"], _vad_intervals(substrate, clip),
        params=DEFAULT_SPLIT_PARAMS,
    )
    return _segment_starts(pieces), False


def fetch_pinned(name: str, download) -> str:
    """One model file, at the pinned revision, verified byte for byte.

    `download(repo_id=, filename=, revision=)` is `hf_hub_download`; passed in
    so the pin itself is testable without the network. Both halves matter: the
    revision keeps the hub from resolving `main` somewhere new, the digest
    keeps a cache from lying about what that resolved to.
    """

    path = download(repo_id=REPO, filename=name, revision=REVISION)
    digest = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if digest != FILE_SHA256[name]:
        raise SystemExit(
            f"{name} is {digest}, pinned {FILE_SHA256[name]} -- the numbers in "
            f"bench-baselines.md 第十九节 are not about this file"
        )
    return path


class SbdModel:
    """The ONNX graph, with its pre-threshold SBD probability exposed."""

    def __init__(self) -> None:
        import onnx
        import onnxruntime as ort
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download

        def fetch(name: str) -> str:
            return fetch_pinned(name, hf_hub_download)

        model = onnx.load(fetch("model.onnx"))
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                SBD_PROB_TENSOR, onnx.TensorProto.FLOAT, None
            )
        )
        self._session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        self._names = [o.name for o in self._session.get_outputs()]
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(fetch("sp.model"))

    def tokenize(self, text: str):
        """(ids, spans). ⚠ SentencePiece reports **character** offsets, not
        byte offsets. Measuring boundaries in bytes and matching them against
        these silently produced a ~3x shift on Japanese and made every number
        in the first run meaningless -- `test_sbd_split.py` pins the unit."""

        pieces = self._sp.encode(text, out_type="immutable_proto").pieces
        return [p.id for p in pieces], [(p.begin, p.end) for p in pieces]

    def probabilities(self, ids: list[int]):
        """`p_sbd` per input token, from overlapping windows (see CHUNK_STRIDE)."""

        import numpy as np

        best = np.zeros(len(ids), dtype=np.float32)
        margin = np.full(len(ids), -1.0, dtype=np.float32)
        starts = list(range(0, max(1, len(ids)), CHUNK_STRIDE))
        for start in starts:
            chunk = ids[start : start + MAX_TOKENS]
            if not chunk:
                continue
            arr = np.array([[0] + chunk + [2]], dtype=np.int64)
            out = dict(zip(self._names, self._session.run(None, {"input_ids": arr})))
            probs = out[SBD_PROB_TENSOR][0]
            probs = probs[1 : 1 + len(chunk)]
            for offset, value in enumerate(probs):
                index = start + offset
                # Distance to the nearer edge of THIS window: the winner is the
                # window that saw the most context on both sides.
                edge = min(offset, len(chunk) - 1 - offset)
                if edge > margin[index]:
                    margin[index], best[index] = edge, value
            if start + MAX_TOKENS >= len(ids):
                break
        return best


def map_boundaries(words, boundaries, model: SbdModel):
    """Boundary -> `p_sbd`, under the pre-registered mapping contract.

    * text = every word's `strip`ped surface, concatenated in order;
    * boundary `k` sits at the CHARACTER offset ending word `k` -- the unit
      SentencePiece reports its spans in;
    * its probability is the token that ENDS exactly there -- "the token that
      just finished before this boundary";
    * if no token ends there the boundary is INSIDE a token (two ASR words in
      one subword). Those are marked unevaluable and dropped rather than made
      to share a neighbour's probability.
    """

    surfaces = [strip_punctuation(w["word"]) for w in words]
    text = "".join(surfaces)
    offsets, running = [], 0
    for surface in surfaces:
        running += len(surface)
        offsets.append(running)  # character offset just after word i
    ids, spans = model.tokenize(text)
    ends = {end: index for index, (_begin, end) in enumerate(spans)}
    probs = model.probabilities(ids)
    out = {}
    for k in boundaries:
        if k >= len(offsets):
            continue
        index = ends.get(offsets[k])
        out[k] = None if index is None else float(probs[index])
    return out


def cmd_features(out_json) -> int:
    rows = load_boundaries()
    print(f"\nlabelled must/never boundaries: {len(rows)}")
    model = SbdModel()
    by_substrate: dict[str, list[dict]] = {}
    for row in rows:
        by_substrate.setdefault(row["substrate"], []).append(row)
    for substrate, group in by_substrate.items():
        words = gold.load_words(pathlib.Path(substrate))
        mapping = map_boundaries(words, sorted({r["k"] for r in group}), model)
        for row in group:
            row["p_sbd"] = mapping.get(row["k"])
        unevaluable = sum(1 for r in group if r["p_sbd"] is None)
        print(f"  {pathlib.Path(substrate).name}: {len(group)} rows, "
              f"{unevaluable} unevaluable (boundary inside a subword)")
    _write(out_json, rows)
    return cmd_gate(rows)


# --- gate ----------------------------------------------------------------------
def cmd_gate(rows) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    total = len(rows)
    rows = [r for r in rows if r.get("p_sbd") is not None]
    dropped_map = total - len(rows)
    #: THE bucket. Everything else is decided by punctuation, and those we
    #: already cut correctly -- averaging them in would inflate every column.
    rows = [r for r in rows if r["vad"] == 0.0]
    print("\n" + "=" * 74)
    print(f"P18 offline gate -- {len(rows)} boundaries in the g = 0 bucket "
          f"({dropped_map} dropped as unevaluable)")
    print("=" * 74)
    if not rows:
        print("nothing to score")
        return 1

    y = np.array([1 if r["label"] == POSITIVE else 0 for r in rows])
    pause = np.array([[r["pause"]] for r in rows], dtype=float)
    sbd = np.array([[r["p_sbd"]] for r in rows], dtype=float)
    prod = np.array([r["prod_cut"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    declared = sum(1 for r in rows if r["declared"] and r["label"] == NEGATIVE)
    print(f"  positives (must) {int(y.sum())}   negatives (never) {int((1 - y).sum())}"
          f"   of which declared {declared}   clips {len(set(groups))}")

    def out_of_fold(features):
        return _oof(features, y, groups)

    raw_auc = roc_auc_score(y, sbd[:, 0])
    baseline = out_of_fold(pause)
    combined = out_of_fold(np.hstack([pause, sbd]))
    base_auc, comb_auc = roc_auc_score(y, baseline), roc_auc_score(y, combined)

    print(f"\n  [1] p_sbd alone, AUC                 {raw_auc:.3f}"
          f"   (gate >= 0.70)  {'PASS' if raw_auc >= 0.70 else 'FAIL'}")
    print(f"      pause-only baseline, out-of-fold   {base_auc:.3f}")
    print(f"      pause + p_sbd,      out-of-fold    {comb_auc:.3f}")
    delta = comb_auc - base_auc
    print(f"  [2] increment                        {delta:+.3f}"
          f"   (gate >= +0.05) {'PASS' if delta >= 0.05 else 'FAIL'}")

    # [3] The operating point is PRODUCTION'S, read off today's segmenter on
    # these same windows -- not a point optimized on this data. The gate as
    # written is "must-recall must not fall below the current system's" at the
    # rate the current system mis-cuts presumed-never positions.
    fpr_now = float(prod[y == 0].mean())
    recall_now = float(prod[y == 1].mean())
    recall_base = _recall_at_fpr(y, baseline, fpr_now)
    recall_new = _recall_at_fpr(y, combined, fpr_now)
    print(f"  [3] production cuts {int(prod.sum())} of these boundaries: "
          f"FPR {fpr_now:.3f}, must-recall {recall_now:.3f}")
    print(f"      at that FPR: pause {recall_base:.3f} -> pause+p_sbd {recall_new:.3f}"
          f"   (gate >= {recall_now:.3f}) "
          f"{'PASS' if recall_new >= recall_now else 'FAIL'}")

    # Sensitivity, because gate 3 is the cell that decides: three clips were cut
    # by an OLDER segmenter and cannot be re-cut (`production_cuts`), so their
    # operating point is a historical system's. If dropping them flipped the
    # cell, the verdict would be an artifact of stale artifacts rather than a
    # result. It is printed unconditionally -- a sensitivity check you only look
    # at when you dislike the answer is not a sensitivity check.
    fresh = [r for r in rows if not r.get("prod_stale")]
    if fresh and len(fresh) != len(rows):
        fy = np.array([1 if r["label"] == POSITIVE else 0 for r in fresh])
        fprod = np.array([r["prod_cut"] for r in fresh])
        fgroups = np.array([r["group"] for r in fresh])
        ffpr, frecall = float(fprod[fy == 0].mean()), float(fprod[fy == 1].mean())
        farm = _oof(np.hstack([
            np.array([[r["pause"]] for r in fresh], dtype=float),
            np.array([[r["p_sbd"]] for r in fresh], dtype=float),
        ]), fy, fgroups)
        fnew = _recall_at_fpr(fy, farm, ffpr)
        print(f"      ⚠ {len(rows) - len(fresh)} rows come from clips an older "
              f"segmenter cut; on the {len(set(fgroups))} re-cuttable clips only:")
        print(f"        FPR {ffpr:.3f}, production {frecall:.3f}, "
              f"pause+p_sbd {fnew:.3f}   "
              f"{'PASS' if fnew >= frecall else 'FAIL'} (same cell, same arm)")

    verdict = raw_auc >= 0.70 and delta >= 0.05 and recall_new >= recall_now
    print("\n  VERDICT: " + ("PASS -- 值得再取一批语料确认" if verdict
                             else "FAIL -- 按 P18 结案，不进 T(k)"))

    _prerequisite_five(rows, y, pause, sbd, groups)
    return 0


def _prerequisite_five(rows, y, pause, sbd, groups) -> None:
    """Not part of the gate: does `p_sbd` add over the prior we ALREADY own?

    Restricted to the windows whose substrate `split_segments` never touched,
    where `seam` is the real `whisper_segment_bonus` input rather than a later
    segmenter's output (see `segment_seams`). Everything is recomputed on that
    subset -- folds, operating point, the lot -- so no number here is mixed
    across two populations.
    """

    import numpy as np
    from sklearn.metrics import roc_auc_score

    keep = np.array([r["seam"] is not None for r in rows])
    print("\n  --- 前置 5：与我们已有的边界先验比（不属于门槛） ---")
    if keep.sum() < 2 or len(set(y[keep])) < 2:
        print("      no un-split windows in this table")
        return
    sub = [r for r in rows if r["seam"] is not None]
    ys, ps, ss = y[keep], pause[keep], sbd[keep]
    seam = np.array([[float(r["seam"])] for r in sub])
    prod = np.array([r["prod_cut"] for r in sub])
    gs = groups[keep]
    print(f"      {len(sub)} boundaries, {int(ys.sum())} must, "
          f"{len(set(gs))} clips (substrates split_segments never ran on)")

    seam_auc = roc_auc_score(ys, seam[:, 0])
    arm_pause = _oof(ps, ys, gs)
    arm_seam = _oof(np.hstack([ps, seam]), ys, gs)
    arm_all = _oof(np.hstack([ps, seam, ss]), ys, gs)
    print(f"      pause,               out-of-fold   {roc_auc_score(ys, arm_pause):.3f}")
    print(f"      segment seam alone,  AUC           {seam_auc:.3f}")
    print(f"      pause + seam,        out-of-fold   {roc_auc_score(ys, arm_seam):.3f}")
    print(f"      pause + seam + p_sbd,out-of-fold   {roc_auc_score(ys, arm_all):.3f}"
          f"   increment {roc_auc_score(ys, arm_all) - roc_auc_score(ys, arm_seam):+.3f}")

    # The increment as TP/FP, between the two OUT-OF-FOLD arms at one shared
    # operating point -- production's. Comparing thresholded feature sets
    # instead (seam vs `p_sbd > 0.5`) would answer a different question and let
    # a threshold nobody registered pick the answer.
    fpr = float(prod[ys == 0].mean())
    tp_seam, fp_seam = _counts_at_fpr(ys, arm_seam, fpr)
    tp_all, fp_all = _counts_at_fpr(ys, arm_all, fpr)
    # Both arms spend the same FP budget by construction -- that is what
    # "at a fixed FPR" means -- so the comparison is on TP alone.
    print(f"\n      at production's FPR {fpr:.3f} "
          f"({int(prod.sum())} cuts, must-recall {float(prod[ys == 1].mean()):.3f}), "
          f"shared FP budget {fp_seam}/{int((1 - ys).sum())}:")
    print(f"      pause + seam          TP {tp_seam:3d} / {int(ys.sum())}"
          f"   (FP {fp_seam})")
    print(f"      pause + seam + p_sbd  TP {tp_all:3d} / {int(ys.sum())}"
          f"   (FP {fp_all})   net {tp_all - tp_seam:+d} TP")


def _oof(features, y, groups):
    """Out-of-fold predictions for one arm, grouped by clip."""

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    pred = np.zeros(len(y), dtype=float)
    splitter = fold_splitter(min(len(set(groups)), 5))
    for train, test in splitter.split(features, y, groups):
        if len(set(y[train])) < 2:
            pred[test] = y[train].mean()
            continue
        mean, std = features[train].mean(0), features[train].std(0)
        std[std == 0] = 1.0
        model = LogisticRegression(max_iter=1000)
        model.fit((features[train] - mean) / std, y[train])
        pred[test] = model.predict_proba((features[test] - mean) / std)[:, 1]
    return pred


def fold_splitter(n_splits: int):
    """Grouped by window, always.

    Adjacent boundaries are highly correlated -- a random split would leak a
    neighbour's answer into the test fold and inflate every column. Its own
    function so the property is testable rather than asserted on source text:
    a mutation to `KFold` leaves the word "GroupKFold" sitting in the imports.
    """

    from sklearn.model_selection import GroupKFold

    return GroupKFold(n_splits=n_splits)


def _counts_at_fpr(y, score, target_fpr):
    """(TP, FP) at the largest threshold whose FPR does not exceed the target."""

    import numpy as np

    order = np.argsort(-np.asarray(score, dtype=float))
    negatives = int((1 - y).sum())
    budget = int(np.floor(target_fpr * negatives + 1e-9))
    tp = fp = 0
    for index in order:
        if y[index] == 0:
            if fp + 1 > budget:
                break
            fp += 1
        else:
            tp += 1
    return tp, fp


def _recall_at_fpr(y, score, target_fpr):
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, score)
    index = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    return float(tpr[max(0, index)])


def _write(path, payload) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("features", "gate"))
    parser.add_argument("--features", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    if args.mode == "features":
        return cmd_features(args.json)
    if args.features is None:
        parser.error("gate needs --features")
    return cmd_gate(json.loads(args.features.read_text(encoding="utf-8")))


if __name__ == "__main__":
    raise SystemExit(main())

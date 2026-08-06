"""Step 0 of the creep takeover: is the recall loss real, and what exactly is in the
regions the production floor suppresses?

Two region families, mirrored questions:

  added   speech under a creep-free floor (Decomposed a=8, the S1 pick) that
          production calls non-speech. If ASR pulls real words out of one of these,
          production's creep genuinely lost subtitles there.
  empty   production speech intervals in which the full production run produced no
          valid word. If such a region is also low-energy and snippet ASR stays
          silent, it is background jitter that production wrongly kept.

Auto-verdict rules (user-specified):
  - snippet ASR yields real text                       -> recall loss ("real")
  - no real text and peak energy < CERTAIN_DB (-50)    -> jitter, near-certain
  - no real text, low energy (< DELEGATE_DB), sustained (>= DELEGATE_DUR) and
    silero finds voicing                               -> "delegate": cannot be
    separated from a true whisper by level alone; per the user's call this class is
    handed to silero/ASR downstream, not decided by the energy VAD
  - the rest                                           -> ambiguous, cut for human
    annotation

Every region gets a full feature row (duration, energy peak/p90, SNR over the
decomposed background level, silero peak/mean, spectral flatness, ASR text +
no_speech_prob) so a better jitter-vs-whisper separator can be hunted afterwards --
long-duration jitter exists, and level alone does not separate it.

Decoder visibility: a region already covered by production speech plus the 0.7 s
gap-keep tail after each interval (GAP_KEEP_REAL_MAX_SEC) was decoded anyway --
words there are NOT lost (FINDINGS I2: 93.5% of interval-level "lost" words live
there). Head-side audio is replaced by synthetic silence and is never seen. The
never-decoded fraction is recorded per region and the summary is split by it.

Controls (method check, FINDINGS H taught us not to trust an unvalidated probe):
  - positive: regions inside production speech that contain valid words; snippet
    ASR must recover text there, else the probe under-reports "real".
  - negative: regions both floors call non-speech with peak < CERTAIN_DB; snippet
    ASR must stay silent there, else it hallucinates on silence.

Usage (paths are per-machine, see docs/data-index.md):
  python v26_step0.py --clip yingtao=... --stable yingtao=... \
      --clip BV1cqLR6hEp3=... --stable BV1cqLR6hEp3=... \
      --word-srt BV1cqLR6hEp3=...fixed.srt \
      --cache-dir ../../tmp/vad-tuning-cache --outdir ../../tmp/vad-step0
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402

from backends import intersect, invert, silero_probs, union, SILERO_HOP_SEC  # noqa: E402
from energy_sweep import cached_tracks, speech_from_tracks, Tracks  # noqa: E402
from floor_decomposed import Decomposed  # noqa: E402
from refs import load_valid_words, load_word_srt, Word  # noqa: E402

Interval = Tuple[float, float]

SR = 16000
CERTAIN_DB = -50.0     # below this peak, jitter near-certain (user rule)
DELEGATE_DB = -30.0    # low-energy band whose voiced members go to silero/ASR
DELEGATE_DUR = 0.2
MIN_DUR = 0.12         # shorter than any word start we care about
SNIP_PAD_SIL = 0.3     # synthetic silence around each snippet, like group assembly
GAP_KEEP = 0.7         # transcribe.GAP_KEEP_REAL_MAX_SEC, restated deliberately:
                       # importing transcribe pulls the whole ASR stack

# silero voicing gates for the no-text classes
SILERO_VOICED = 0.5
SILERO_UNVOICED = 0.2

FILLERS = {
    "あ", "あー", "ああ", "え", "えー", "ええ", "えっ", "えっと", "えーと", "えとー",
    "うん", "うんうん", "ん", "んー", "んん", "はぁ", "はあ", "は", "はい", "ふう",
    "ふー", "ふふ", "ふふふ", "はは", "ははは", "へえ", "へー", "ほう", "ほー",
    "まあ", "まぁ", "ね", "ねえ", "ねー", "よし", "うわ", "うわあ", "わあ", "わー",
    "おお", "おー", "お", "おっ", "や", "やあ", "よ", "さあ", "さー", "そう",
    "うーん", "むー", "む", "ふん", "ヤホ", "ヤホッ", "あれ", "あら", "おや",
}
HALLU_PATTERNS = (
    "ご視聴ありがとう", "チャンネル登録", "ご清聴", "おやすみなさい", "また明日",
)


def _strip_punct(s: str) -> str:
    return "".join(ch for ch in s
                   if not unicodedata.category(ch).startswith("P")
                   and not ch.isspace() and ch not in "…‥ー~〜!?！？。、・")


def classify_text(raw: str) -> str:
    """'' | 'filler' | 'hallucination' | 'real'"""
    t = raw.strip()
    if not t:
        return ""
    if any(p in t for p in HALLU_PATTERNS):
        return "hallucination"
    core = _strip_punct(t)
    if not core:
        return ""
    # decoder loop: one token (or two) repeated to fill the segment
    m = re.fullmatch(r"(.{1,3}?)\1{2,}", core)
    if m:
        return "hallucination"
    if core in FILLERS or _strip_punct(t.replace("ー", "")) in FILLERS:
        return "filler"
    # strings of pure long-vowel laughter/vowel runs
    if re.fullmatch(r"[あーはぁふふんうわおえヘヘへよォオぉ]{1,}", core) and len(set(core)) <= 2:
        return "filler"
    return "real"


# --------------------------------------------------------------------------
# interval helpers
# --------------------------------------------------------------------------

def subtract(a: Sequence[Interval], b: Sequence[Interval], duration: float) -> List[Interval]:
    if not b:
        return [tuple(x) for x in a]
    return intersect(a, invert(b, duration))


def with_tails(intervals: Sequence[Interval], duration: float) -> List[Interval]:
    """Production speech plus the <=0.7 s real-audio tail the decoder also sees."""
    ext = [(s, min(duration, e + GAP_KEEP)) for s, e in intervals]
    return union(ext, [])


def covered_frac(iv: Interval, cover: Sequence[Interval]) -> float:
    s, e = iv
    if e <= s:
        return 0.0
    tot = 0.0
    for a, b in cover:
        if b <= s:
            continue
        if a >= e:
            break
        tot += min(b, e) - max(a, s)
    return tot / (e - s)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

@dataclass
class Region:
    kind: str                  # added | empty | ctrl_pos | ctrl_neg
    start: float
    end: float
    peak_db: float = -100.0
    p90_db: float = -100.0
    med_db: float = -100.0
    snr_peak: float = 0.0      # peak_db - decomposed background level (median)
    silero_peak: float = 0.0
    silero_mean: float = 0.0
    flatness: float = 0.0      # mean spectral flatness (1.0 = white-noise-like)
    never_decoded_frac: float = 1.0
    gap_prev: float = 999.0    # to previous production speech interval
    gap_next: float = 999.0
    asr_text: str = ""
    asr_nsp: float = -1.0      # max no_speech_prob over segments
    asr_lp: float = 0.0        # mean avg_logprob
    text_class: str = ""       # '', filler, hallucination, real
    verdict: str = ""
    note: str = ""

    @property
    def dur(self) -> float:
        return self.end - self.start


def energy_stats(tr: Tracks, s: float, e: float) -> Tuple[float, float, float]:
    starts = tr.frame_starts.numpy() if hasattr(tr.frame_starts, "numpy") else np.asarray(tr.frame_starts)
    edb = tr.energy_db.numpy() if hasattr(tr.energy_db, "numpy") else np.asarray(tr.energy_db)
    i0, i1 = np.searchsorted(starts, [s, e])
    if i1 <= i0:
        i1 = min(i0 + 1, len(edb))
    seg = edb[i0:i1]
    if seg.size == 0:
        return -100.0, -100.0, -100.0
    return float(seg.max()), float(np.quantile(seg, 0.9)), float(np.median(seg))


def level_median(lvl: np.ndarray, starts: np.ndarray, s: float, e: float) -> float:
    i0, i1 = np.searchsorted(starts, [s, e])
    if i1 <= i0:
        i1 = min(i0 + 1, len(lvl))
    seg = lvl[i0:i1]
    return float(np.median(seg)) if seg.size else -100.0


def silero_stats(probs: np.ndarray, s: float, e: float) -> Tuple[float, float]:
    i0 = int(s / SILERO_HOP_SEC)
    i1 = max(i0 + 1, int(e / SILERO_HOP_SEC))
    seg = probs[i0:min(i1, len(probs))]
    if seg.size == 0:
        return 0.0, 0.0
    return float(seg.max()), float(seg.mean())


def flatness_of(wav: np.ndarray, s: float, e: float) -> float:
    import librosa

    a = wav[int(s * SR):int(e * SR)]
    if a.size < 256:
        return 0.0
    fl = librosa.feature.spectral_flatness(y=a, n_fft=512, hop_length=128)
    return float(np.mean(fl))


def gaps_to_speech(iv: Interval, speech: Sequence[Interval]) -> Tuple[float, float]:
    s, e = iv
    prev = min((s - b for a, b in speech if b <= s + 1e-9), default=999.0)
    nxt = min((a - e for a, b in speech if a >= e - 1e-9), default=999.0)
    return max(prev, 0.0), max(nxt, 0.0)


# --------------------------------------------------------------------------
# snippet ASR
# --------------------------------------------------------------------------

class SnippetAsr:
    def __init__(self, model_name: str = "large-v3-turbo"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device="cuda", compute_type="float16")

    def run(self, wav: np.ndarray, s: float, e: float) -> Tuple[str, float, float]:
        pad = np.zeros(int(SNIP_PAD_SIL * SR), dtype=np.float32)
        a = np.concatenate([pad, wav[int(s * SR):int(e * SR)], pad])
        segs, _info = self.model.transcribe(
            a, language="ja", beam_size=5, temperature=0.0,
            condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True)
        texts, nsps, lps = [], [], []
        for seg in segs:
            texts.append(seg.text.strip())
            nsps.append(seg.no_speech_prob)
            lps.append(seg.avg_logprob)
        return ("".join(texts), max(nsps) if nsps else -1.0,
                float(np.mean(lps)) if lps else 0.0)


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

def decide(r: Region) -> str:
    if r.text_class == "real":
        return "real"
    if r.peak_db < CERTAIN_DB:
        return "jitter_certain"          # user rule: this quiet is never speech
    if r.silero_peak < SILERO_UNVOICED:
        return "jitter_silero"           # sustained or not, nothing voiced in it
    if (r.peak_db < DELEGATE_DB and r.dur >= DELEGATE_DUR
            and r.silero_peak >= SILERO_VOICED):
        return "delegate"                # low, sustained, voiced: silero/ASR's call
    return "ambiguous"


# --------------------------------------------------------------------------
# per-clip pipeline
# --------------------------------------------------------------------------

@dataclass
class ClipResult:
    name: str
    regions: List[Region] = field(default_factory=list)
    n_prod: int = 0
    n_decomp: int = 0
    added_sec: float = 0.0
    never_sec: float = 0.0


def load_wave(path: Path) -> np.ndarray:
    from asr_playground.speech.preprocessing import energy as E

    wav = E._load_asr_audio_streamed(str(path))
    wav = E.light_normalize(wav, E.TARGET_SR)
    return wav.numpy().astype(np.float32).reshape(-1)


def decomp_speech(tr: Tracks, floor_np: np.ndarray) -> List[Interval]:
    from asr_playground.speech.preprocessing import energy as E

    raw = E._score_to_non_speech_intervals(
        tr.energy_db, torch.from_numpy(floor_np.astype(np.float32)), tr.frame_dbfs,
        tr.frame_starts, tr.frame_ends, tr.duration,
        enter_margin_db=6.0, weighted=bool(E.WEIGHTED_INTERVAL))
    ns = E._apply_negative_padding(raw, tr.duration)
    return [(float(a), float(b))
            for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def run_clip(name: str, vocal: Path, stable: Optional[Path], word_srt: Optional[Path],
             cache_dir: Path, outdir: Path, asr: Optional[SnippetAsr],
             n_controls: int, rng: np.random.Generator) -> ClipResult:
    print(f"\n=== {name} ===", flush=True)
    tr = cached_tracks(vocal, cache_dir)
    starts = tr.frame_starts.numpy()
    edb = tr.energy_db.numpy()
    prod = speech_from_tracks(tr)

    # the S1/S2 pick: level + 8, no dwell -- dwell would absorb filled pauses and
    # hide exactly the added regions this probe wants to look at
    dec = Decomposed(a=8.0, b=0.0, dwell_max=0.0)
    thr, lvl = dec.diagnostics(edb.astype(np.float64), starts.astype(np.float64), tr.duration)
    dspeech = decomp_speech(tr, thr - 6.0)

    added_all = [iv for iv in subtract(dspeech, prod, tr.duration)
                 if iv[1] - iv[0] >= MIN_DUR]
    visible = with_tails(prod, tr.duration)

    words: List[Word] = []
    if word_srt is not None:
        words = load_word_srt(word_srt)
    elif stable is not None:
        words, _stats = load_valid_words(stable)

    # empty production intervals: no valid word overlaps
    empties: List[Interval] = []
    if words:
        wiv = [(w.start, w.end) for w in words]
        for iv in prod:
            if covered_frac(iv, wiv) <= 0.0:
                empties.append(iv)

    wav = load_wave(vocal)
    sil = silero_probs(vocal, cache_dir / f"silero-{vocal.stem}.npz")

    res = ClipResult(name=name, n_prod=len(prod), n_decomp=len(dspeech))
    res.added_sec = sum(e - s for s, e in added_all)

    def featurize(kind: str, s: float, e: float) -> Region:
        r = Region(kind=kind, start=round(s, 3), end=round(e, 3))
        r.peak_db, r.p90_db, r.med_db = energy_stats(tr, s, e)
        r.snr_peak = r.peak_db - level_median(lvl, starts, s, e)
        r.silero_peak, r.silero_mean = silero_stats(sil, s, e)
        r.flatness = flatness_of(wav, s, e)
        r.never_decoded_frac = 1.0 - covered_frac((s, e), visible)
        r.gap_prev, r.gap_next = gaps_to_speech((s, e), prod)
        return r

    for s, e in added_all:
        res.regions.append(featurize("added", s, e))
    res.never_sec = sum(r.dur * r.never_decoded_frac
                        for r in res.regions if r.kind == "added")
    for s, e in empties:
        res.regions.append(featurize("empty", s, e))

    # controls
    if words and n_controls > 0:
        speech_words = [w for w in words if w.end - w.start >= 0.15]
        for w in rng.choice(len(speech_words), size=min(n_controls, len(speech_words)),
                            replace=False):
            ww = speech_words[int(w)]
            res.regions.append(featurize("ctrl_pos", max(0.0, ww.start - 0.05),
                                         min(tr.duration, ww.end + 0.05)))
        both_ns = subtract(invert(prod, tr.duration), dspeech, tr.duration)
        deep = []
        for s, e in both_ns:
            t = s
            while t + 0.5 <= e:
                pk, _, _ = energy_stats(tr, t, t + 0.5)
                if pk < CERTAIN_DB:
                    deep.append((t, t + 0.5))
                t += 0.5
        if deep:
            for i in rng.choice(len(deep), size=min(n_controls, len(deep)),
                                replace=False):
                res.regions.append(featurize("ctrl_neg", *deep[int(i)]))

    if asr is not None:
        for i, r in enumerate(res.regions):
            r.asr_text, r.asr_nsp, r.asr_lp = asr.run(wav, r.start, r.end)
            r.text_class = classify_text(r.asr_text)
            if (i + 1) % 50 == 0:
                print(f"  asr {i + 1}/{len(res.regions)}", flush=True)

    for r in res.regions:
        if r.kind in ("added", "empty"):
            r.verdict = decide(r)
        elif r.kind == "ctrl_pos":
            r.verdict = "ok" if r.text_class in ("real", "filler") else "MISS"
        elif r.kind == "ctrl_neg":
            r.verdict = "ok" if r.text_class in ("", "hallucination") else "FALSE_TEXT"

    write_outputs(name, res, wav, outdir)
    return res


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

FIELDS = ["idx", "kind", "start", "end", "dur", "peak_db", "p90_db", "med_db",
          "snr_peak", "silero_peak", "silero_mean", "flatness",
          "never_decoded_frac", "gap_prev", "gap_next",
          "asr_text", "asr_nsp", "asr_lp", "text_class", "verdict", "label"]


def _fmt_ts(t: float) -> str:
    # total-ms first, like subtitles.rendering.format_srt_time: deriving ms from
    # the fraction rounds 0.9996 to ",1000"
    ms = max(0, int(round(t * 1000.0)))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(name: str, res: ClipResult, wav: np.ndarray, outdir: Path) -> None:
    import soundfile as sf

    d = outdir / name
    (d / "snips").mkdir(parents=True, exist_ok=True)

    with open(d / "regions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for i, r in enumerate(res.regions):
            w.writerow([i, r.kind, f"{r.start:.3f}", f"{r.end:.3f}", f"{r.dur:.3f}",
                        f"{r.peak_db:.1f}", f"{r.p90_db:.1f}", f"{r.med_db:.1f}",
                        f"{r.snr_peak:.1f}", f"{r.silero_peak:.2f}",
                        f"{r.silero_mean:.2f}", f"{r.flatness:.3f}",
                        f"{r.never_decoded_frac:.2f}", f"{r.gap_prev:.2f}",
                        f"{r.gap_next:.2f}", r.asr_text, f"{r.asr_nsp:.2f}",
                        f"{r.asr_lp:.2f}", r.text_class, r.verdict, ""])

    # snippets for human ears: everything not auto-settled, plus the evidence class
    review = [(i, r) for i, r in enumerate(res.regions)
              if r.verdict in ("real", "ambiguous", "delegate")
              or (r.kind in ("added", "empty") and r.verdict == "jitter_silero"
                  and r.dur >= 0.5)]
    for i, r in review:
        pad = int(0.15 * SR)
        a0 = max(0, int(r.start * SR) - pad)
        a1 = min(len(wav), int(r.end * SR) + pad)
        fn = (f"{i:04d}-{r.kind}-{r.verdict}-t{r.start:07.1f}"
              f"-d{r.dur:.2f}-pk{r.peak_db:.0f}.wav")
        sf.write(d / "snips" / fn, wav[a0:a1], SR)

    for kind in ("added", "empty"):
        rows = [r for r in res.regions if r.kind == kind]
        if not rows:
            continue
        with open(d / f"{name}-step0-{kind}.srt", "w", encoding="utf-8") as f:
            for j, r in enumerate(rows, 1):
                txt = f"[{r.verdict}] pk{r.peak_db:.0f} sil{r.silero_peak:.2f} {r.asr_text}"
                f.write(f"{j}\n{_fmt_ts(r.start)} --> {_fmt_ts(r.end)}\n{txt.strip()}\n\n")


def summarize(results: List[ClipResult], outdir: Path) -> None:
    lines: List[str] = ["# step0 探针汇总\n"]
    for res in results:
        lines.append(f"\n## {res.name}\n")
        lines.append(f"- 生产区间 {res.n_prod} / 拆分floor区间 {res.n_decomp};"
                     f" 新增语音 {res.added_sec:.1f}s,其中从未被解码 {res.never_sec:.1f}s\n")
        for kind in ("added", "empty", "ctrl_pos", "ctrl_neg"):
            rows = [r for r in res.regions if r.kind == kind]
            if not rows:
                continue
            lines.append(f"- **{kind}** n={len(rows)}:\n")
            counts: Dict[str, int] = {}
            for r in rows:
                counts[r.verdict] = counts.get(r.verdict, 0) + 1
            for v, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                sec = sum(r.dur for r in rows if r.verdict == v)
                nd = sum(r.dur * r.never_decoded_frac for r in rows if r.verdict == v)
                extra = f",从未解码 {nd:.1f}s" if kind == "added" else ""
                lines.append(f"    - {v}: {c} 段 / {sec:.1f}s{extra}\n")
            reals = [r for r in rows if r.verdict == "real" and
                     (kind != "added" or r.never_decoded_frac > 0.5)]
            if reals and kind == "added":
                lines.append(f"    - ⚠️ 真语音且从未解码: {len(reals)} 段 —— 丢 recall 的直接证据\n")
                for r in reals[:12]:
                    lines.append(f"        - {r.start:.1f}s d{r.dur:.2f} pk{r.peak_db:.0f} "
                                 f"sil{r.silero_peak:.2f}: {r.asr_text}\n")
    (outdir / "step0-summary.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=VOCAL")
    ap.add_argument("--stable", action="append", default=[], metavar="NAME=STABLE_JSON")
    ap.add_argument("--word-srt", action="append", default=[], metavar="NAME=SRT")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    stables = dict(x.split("=", 1) for x in args.stable)
    srts = dict(x.split("=", 1) for x in args.word_srt)
    args.outdir.mkdir(parents=True, exist_ok=True)

    asr = None if args.no_asr else SnippetAsr()
    rng = np.random.default_rng(args.seed)
    results = []
    for name, vocal in clips.items():
        results.append(run_clip(
            name, Path(vocal),
            Path(stables[name]) if name in stables else None,
            Path(srts[name]) if name in srts else None,
            args.cache_dir, args.outdir, asr, args.controls, rng))
    summarize(results, args.outdir)


if __name__ == "__main__":
    main()

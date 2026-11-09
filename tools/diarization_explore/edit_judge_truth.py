"""Japanese-text truth for the edit judge (owner 2026-10-07): rows to split, lines ASR missed, characters ASR got wrong.

Sources (only material with a Japanese transcript and a speaker per line):
  hina      the bilingual refined subtitles: every Chinese line has a `<person>-日文` twin, a fansub's Japanese transcript
            (hina-60..74; the eleven with a P2 cache are used, because production's Nemotron-3 tracks come from it).
            `p2_review_video.reference_lines` drops these twins; they are read here. The fansub cleans speech up: fillers
            (うーん, まあ) and many back-channels are left out; timing is approximate (about 0.5 s). PV narration and other
            lines whose style names no person keep a person of None (never scored).
  manosaba  the 魔裁 clips' OCR text boxes with nameplate speakers (`manosaba_truth.cues`); narration (no plate) is not
            voiced and is left out; the SNS scene of clip b (cues carrying the chat overlay「退出しました」) is OCR junk
            and its time is excluded from scoring.
  callhome  CALLHOME Japanese telephone calls (TalkBank), the twenty with a production run in
            `tmp/diarization_explore/callhome-ja/` (pipeline `--stage stable --diarize acoustics`; that run's tracks are the old
            Sortformer backend, not Nemotron-3). Verbatim research transcripts with back-channels, speakers A / B (B1).
姐姐咚咚 has Chinese only, so it gives no text truth and is not used.

Matching is local and at reading level (Sudachi readings in hiragana, digits read as numbers), so kanji / kana choice
does not count: a row is compared with the reference lines within NEAR seconds of it, a line with the rows within NEAR
seconds of it, by longest common subsequence; only common runs of >= RUN characters count as matched.
  row           `split`: one word boundary splits the row into a prefix matched by person A's lines and a suffix matched
                by person B's, each with >= MIN_HALF characters only that person's lines match, scoring >= MIN_GAIN more
                matched characters than the best single person, and B's line starting within ONSET_TOL s of the boundary;
                its truth is the range of word indices the second half may start at (ties). `single` (one person
                explains it best), `multi` (a better two-person split that fails the onset check), `unclear` (less than
                MIN_ALIGNED of the row matched, or in an excluded span).
  missing line  a reference line with a person, Japanese (has kana), >= 2 characters, < MISSING_COVER of it matched by
                the rows near it, away from the file edges: `lexical` when >= LEXICAL characters are left once
                back-channel and filler tokens are removed, else back-channel; `overlapped` when ASR rows cover >=
                OVERLAPPED of its time. Anchors: the rows an insertion may be placed after (the last row ending before
                it and every row within 1 s of it).
  edit target   for a `single` row, the stretch of its person's nearby lines it aligns to (semi-global alignment); edits
                are scored by the reading distance to it with fillers removed (`row_distance`).

The readings need sudachipy + sudachidict_core, which no extra lists on its own: `pip install sudachipy
sudachidict_core` (`pip install -e ".[kb-phonetic]"` brings both too).

    python tools/diarization_explore/edit_judge_truth.py survey     # counts per file, no model
    python tools/diarization_explore/edit_judge_truth.py list       # every split / missing line, for reading
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402

OUT = lr.OUT / "edit"
CALLHOME_RUNS = lr.n3.MAIN / "tmp" / "diarization_explore" / "callhome-ja"
CALLHOME_CHA = lr.n3.MAIN / "data" / "diarization" / "public" / "callhome-jpn" / "transcripts"
HINA = ("hina-61", "hina-63", "hina-64", "hina-65", "hina-67", "hina-68", "hina-70", "hina-71", "hina-72", "hina-73", "hina-74")
MANOSABA = ("manosaba-a", "manosaba-b", "manosaba-c")
CALLHOME = tuple(f"callhome-{p.name}" for p in sorted(CALLHOME_RUNS.glob("[0-9]*")) if (CALLHOME_CHA / f"{p.name}.cha").exists())
FILES = HINA + MANOSABA + CALLHOME  # ⚠ hina first (P2's loaders); nothing here configures onetonton
NEAR = 1.5           # seconds around a row / line within which the other side's text is compared
RUN = 2              # shortest common run that counts as matched
ONSET_TOL = 1.5      # seconds between a split's first B word and the start of a B line it matches
MIN_HALF = 2         # characters only A's (B's) lines match, in the prefix (suffix) of a split
MIN_GAIN = 2         # matched characters a split must add over the best single person
EDGE_SHARE = 0.25    # of a line's length: see `edge`
MIN_ALIGNED = 0.6    # share of a row's characters matched for it to be scored at all
MISSING_COVER = 0.3  # a reference line with less than this share matched is missing from ASR
OVERLAPPED = 0.3     # share of a missing line's time under ASR rows for it to count as `overlapped`
LEXICAL = 4          # characters left after back-channels / fillers for a missing line to be `lexical`
EDGE = 2.0           # seconds: missing lines this close to the first / last ASR row are left out
#: filler and back-channel tokens (readings, hiragana)
FILLERS = ("えーと", "えっと", "ええと", "あのー", "あのう", "うーん", "うーむ", "んー", "えー", "あー", "まあ", "あの")
BACKCHANNEL = FILLERS + ("うんうん", "うん", "ええ", "はい", "ああ", "あっ", "へえ", "へー", "ふーん", "ほう", "そうそう", "そう", "えっ",
                         "あはは", "えへへ", "うふふ", "はは", "ふふ", "へへ", "はあ", "ね", "ん", "あ", "お", "う", "え", "ー", "っ")
#: a line left with only these characters (after back-channels are removed) is a cry or a laugh, not lexical
INTERJECTION = set("ぁぃぅぇぉっゃゅょゎーあいうえおはひふへほんわやゆよ")
PRESENT = 60         # rapidfuzz partial_ratio (reading or surface) at or above which a line counts as written by ASR
_DIGITS = "〇一二三四五六七八九"


def _kanji_number(match):
    n = int(match.group(0))
    if n == 0 or n >= 100000:
        return match.group(0)
    out = ""
    for value, unit in ((10000, "万"), (1000, "千"), (100, "百"), (10, "十")):
        q, n = divmod(n, value)
        if q:
            out += ("" if q == 1 and unit != "万" else _DIGITS[q]) + unit
    return out + (_DIGITS[n] if n else "")


def surface(text):
    """NFKC, letters and digits only, katakana folded to hiragana, lower case."""
    import jaconv
    text = jaconv.kata2hira(unicodedata.normalize("NFKC", str(text))).lower()
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN")


_SUDACHI = None


@lru_cache(maxsize=None)
def reading(text):
    """Hiragana reading (Sudachi) of `surface(text)`, digits read as numbers."""
    global _SUDACHI
    import jaconv
    text = re.sub(r"\d+", _kanji_number, surface(text))
    if not text:
        return ""
    if _SUDACHI is None:
        from sudachipy import dictionary
        _SUDACHI = dictionary.Dictionary().create()
    return surface(jaconv.kata2hira("".join(m.reading_form() for m in _SUDACHI.tokenize(text))))


def strip_tokens(text, tokens):
    for tok in sorted(tokens, key=len, reverse=True):
        text = text.replace(tok, "")
    return text


def has_kana(text):
    return any("぀" <= c <= "ヿ" for c in text)


def is_lexical(text):
    left = strip_tokens(reading(text), BACKCHANNEL)
    return len(left) >= LEXICAL and not set(left) <= INTERJECTION


def morpheme_starts(text):
    """Character offsets in `text` where a Sudachi morpheme starts (and len(text))."""
    global _SUDACHI
    if _SUDACHI is None:
        reading("あ")
    out, pos = set(), 0
    for m in _SUDACHI.tokenize(text):
        out.add(pos)
        pos += len(m.surface())
    return out | {pos}


def pairs(a, b):
    """{position in a: position in b} over common runs of >= RUN characters of an LCS of a and b."""
    from rapidfuzz.distance import Indel
    out = {}
    for op in Indel.opcodes(a, b):
        if op.tag == "equal" and op.src_end - op.src_start >= RUN:
            out.update({op.src_start + k: op.dest_start + k for k in range(op.src_end - op.src_start)})
    return out


def matched(a, b):
    """Positions of `a` in common runs of >= RUN characters of an LCS of a and b."""
    return set(pairs(a, b))


def edge(length):
    """How many characters at a line's end (start) a split's prefix (suffix) may leave unmatched."""
    return max(3, round(EDGE_SHARE * length))


def semiglobal(a, b):
    """(distance, b[start:end]): the stretch of b that a aligns to best, a aligned whole (b's ends free)."""
    n, m = len(a), len(b)
    if not n:
        return 0, ""
    prev = list(range(n + 1))  # column for b[:0]: a[:i] against nothing
    best = (prev[n], 0, 0)
    starts = [0] * (n + 1)
    for j in range(1, m + 1):
        cur, cs = [0] * (n + 1), [j] * (n + 1)  # free start: a[:0] against b[..j] costs 0, starting at j
        for i in range(1, n + 1):
            options = ((prev[i - 1] + (a[i - 1] != b[j - 1]), starts[i - 1]), (prev[i] + 1, starts[i]), (cur[i - 1] + 1, cs[i - 1]))
            cur[i], cs[i] = min(options)
        prev, starts = cur, cs
        if prev[n] < best[0]:
            best = (prev[n], starts[n], j)
    return best[0], b[best[1]:best[2]]


# --- references and production ------------------------------------------------------------------

def reference_jp(ident):
    """([(start, end, person or None, text)] in time order, [(start, end)] spans excluded from scoring)."""
    if ident.startswith("manosaba-"):
        import manosaba_truth as mt
        import manosaba_window_judge as mw
        names = mw.display_names()
        cues = mt.cues(ident, mt.names())
        junk = [(a, b) for a, b, t, _ in cues if "退出しました" in t]
        lines = [(a, b, names[w], t.replace("\n", "")) for a, b, t, w in cues if w is not None and "退出しました" not in t]
        return sorted(lines, key=lambda x: (x[0], x[1])), junk
    if ident.startswith("callhome-"):
        import s4_callhome as sc
        rows = sc.utterances(CALLHOME_CHA / f"{ident.split('-', 1)[1]}.cha")
        begin = max(0.0, min(s for _, s, _, _ in rows) - sc.EDGE)  # the run's audio was cut there (s4_callhome --prepare)
        lines = [(s - begin, e - begin, who, re.sub(r"[⇘⇗→↗↘]", "、", text).replace(" ", "")) for who, s, e, text in rows if text and e > s]
        return sorted(lines, key=lambda x: (x[0], x[1])), []
    import p2_features as pf
    pf._configure("hina")
    import refined_multi as rm
    out = []
    for line in rm.sources()[ident].read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        style = parts[3].strip()
        if not re.search(r"-日文\d*$", style):
            continue
        text = re.sub(r"\{[^}]*\}", "", parts[9]).replace(r"\N", " ").strip()
        if not text or text.startswith("※"):
            continue
        a, b = rm.seconds(parts[1]), rm.seconds(parts[2])
        if b > a:
            out.append((a, b, rm.speaker(style), text))
    return sorted(out, key=lambda x: (x[0], x[1])), []


def production(ident):
    """(stable, speakers): rounds 3-5's loader for hina / 魔裁; the CALLHOME run's own stable + diar.json."""
    if ident.startswith("callhome-"):
        from finesub.speech.diarization.assign import assign_all
        from finesub.speech.diarization.segments import TrackSegment
        call = ident.split("-", 1)[1]
        stable = json.loads((CALLHOME_RUNS / call / f"{call}-stable.json").read_text(encoding="utf-8"))["segments"]
        tracks = json.loads((CALLHOME_RUNS / call / f"{call}-diar.json").read_text(encoding="utf-8"))["tracks"]
        return stable, assign_all(stable, [TrackSegment(track=int(t["track"]), start=float(t["start"]), end=float(t["end"])) for t in tracks])
    import window_judge_r3 as r3
    m = r3.load(ident)
    return m.stable, m.speakers


# --- truth ---------------------------------------------------------------------------------------

@dataclass
class Truth:
    ident: str
    stable: list
    lines: list                      # reference_jp
    junk: list                       # spans excluded from scoring
    rows: dict = field(default_factory=dict)      # row -> {"kind", "people", "split": [lo, hi] or None, ...}
    missing: list = field(default_factory=list)  # [{"line", "start", "end", "person", "text", "lexical", "overlapped", "anchors"}]

    def span(self, i):
        return float(self.stable[i]["start"]), float(self.stable[i]["end"])

    def words(self, i):
        return self.stable[i].get("words") or []

    def in_junk(self, s, e):
        return any(min(e, b) > max(s, a) for a, b in self.junk)

    def near_lines(self, s, e, person=None):
        return [n for n, (a, b, w, _) in enumerate(self.lines) if b > s - NEAR and a < e + NEAR and w is not None
                and (person is None or w == person)]

    def row_person(self, i):
        info = self.rows.get(i) or {}
        return info["people"][0] if info.get("kind") == "single" else None

    def target(self, i, text=None):
        """(reading distance with fillers removed, target reading) of row i's text (or `text` in its place) against the
        stretch of its person's nearby lines it aligns to; None for a row that is not `single`."""
        person = self.row_person(i)
        if person is None:
            return None
        s, e = self.span(i)
        ref = strip_tokens("".join(reading(self.lines[n][3]) for n in self.near_lines(s, e, person)), FILLERS)
        mine = strip_tokens(reading(self.stable[i].get("text", "") if text is None else text), FILLERS)
        return semiglobal(mine, ref)


def _row_reading(t, i):
    """(reading of row i, reading character -> word index, {reading position: word index} where a split may fall).

    The row is read as a whole (Whisper's words are often single kanji, read wrongly alone): each Sudachi morpheme's
    reading goes to the word its first character is in; a split may fall where a morpheme starts at a word start."""
    reading("あ")  # loads Sudachi
    words = [str(x.get("word", "")) for x in t.words(i)]
    starts, pos = {}, 0
    for w, word in enumerate(words):
        starts[pos] = w
        pos += len(word)
    raw = "".join(words)
    text, owner, bounds, word = "", [], {}, 0
    for m in _SUDACHI.tokenize(raw):
        while word + 1 < len(words) and sum(len(x) for x in words[:word + 1]) <= m.begin():
            word += 1
        r = reading(m.surface())
        if r and m.begin() in starts and text:
            bounds[len(text)] = starts[m.begin()]
        text += r
        owner += [word] * len(r)
    return text, owner, bounds


def _row(t, i):
    s, e = t.span(i)
    text, owner, bounds = _row_reading(t, i)
    info = {"people": [], "split": None, "chars": len(text)}
    if not text or t.in_junk(s, e):
        return {**info, "kind": "unclear", "aligned": 0.0}
    by_person, maps = {}, []
    for n in t.near_lines(s, e):
        line = reading(t.lines[n][3])
        found = pairs(text, line)
        by_person.setdefault(t.lines[n][2], set()).update(found)
        maps.append((t.lines[n][2], len(line), found))

    def ends_line(person, chars):
        """Do `chars` (a prefix) reach the end of one of person's lines?"""
        return any(p == person and any(c in f for c in chars) and max(f[c] for c in chars if c in f) >= size - edge(size)
                   for p, size, f in maps)

    def starts_line(person, chars):
        """Do `chars` (a suffix) begin one of person's lines?"""
        return any(p == person and any(c in f for c in chars) and min(f[c] for c in chars if c in f) < edge(size)
                   for p, size, f in maps)

    single = max(((len(m), p) for p, m in by_person.items()), default=(0, None))
    best_split = None
    for a, ma in by_person.items():
        for b, mb in by_person.items():
            if a == b:
                continue
            for k in sorted(bounds):
                only_a = {c for c in ma - mb if c < k}
                only_b = {c for c in mb - ma if c >= k}
                if len(only_a) < MIN_HALF or len(only_b) < MIN_HALF:
                    continue
                if not (ends_line(a, {c for c in ma if c < k}) and starts_line(b, {c for c in mb if c >= k})):
                    continue
                score = len({c for c in ma if c < k}) + len({c for c in mb if c >= k})
                if best_split is None or score > best_split[0]:
                    best_split = (score, a, b, [bounds[k]])
                elif score == best_split[0] and (a, b) == best_split[1:3]:
                    best_split[3].append(bounds[k])
    aligned = max(single[0], best_split[0] if best_split else 0) / len(text)
    info["aligned"] = round(aligned, 3)
    if aligned < MIN_ALIGNED:
        return {**info, "kind": "unclear"}
    if best_split and best_split[0] >= single[0] + MIN_GAIN:
        _, a, b, ks = best_split
        onsets = [float(t.words(i)[k]["start"]) for k in ks]
        near = min(abs(t.lines[n][0] - x) for n in t.near_lines(s, e, b) for x in onsets)
        if near <= ONSET_TOL:
            return {**info, "kind": "split", "people": [a, b], "split": [min(ks), max(ks)]}
        return {**info, "kind": "multi", "people": [a, b], "why": f"onset {near:.1f} s from B's line"}
    return {**info, "kind": "single", "people": [single[1]]}


def _missing(t):
    first, last = t.span(0)[0], t.span(len(t.stable) - 1)[1]
    for n, (a, b, who, text) in enumerate(t.lines):
        line = reading(text)
        if who is None or len(line) < 2 or not has_kana(text) or a < first + EDGE or b > last - EDGE or t.in_junk(a, b):
            continue
        near = [i for i in range(len(t.stable)) if t.span(i)[1] > a - NEAR and t.span(i)[0] < b + NEAR]
        from rapidfuzz import fuzz
        rows = "".join(reading(t.stable[i].get("text", "")) for i in near)
        cover = len(matched(line, rows)) / len(line)
        similar = max(fuzz.partial_ratio(line, rows) if rows else 0,
                      fuzz.partial_ratio(surface(text), "".join(surface(t.stable[i].get("text", "")) for i in near)) if near else 0)
        if cover >= MISSING_COVER or similar >= PRESENT:
            continue  # ASR wrote it (a homophone or a garbled proper noun is still written)
        if (t.ident, round(a, 1)) in EXCLUDE_LINES:
            continue
        under = sum(max(0.0, min(b, t.span(i)[1]) - max(a, t.span(i)[0])) for i in near)
        before = [i for i in range(len(t.stable)) if t.span(i)[1] <= a + 0.3]
        anchors = sorted(set([i for i in near if t.span(i)[1] > a - 1 and t.span(i)[0] < b + 1] + ([before[-1]] if before else [])))
        t.missing.append({"line": n, "start": a, "end": b, "person": who, "text": text, "lexical": is_lexical(text),
                          "overlapped": under / (b - a) >= OVERLAPPED, "anchors": anchors})


#: missing lines read as not missing by hand (text only, before any model run; 2026-10-07): the ASR row is there with
#: the words garbled beyond the similarity check, or the line is a chorus of several people
EXCLUDE_LINES = {
    ("hina-61", 2735.7): "chorus of three; ASR row 527「純正の例だ」",
    ("hina-61", 2840.7): "chorus; ASR row 553「バイバイ」",
    ("hina-72", 1077.5): "ASR row 164「スゴムは」",
    ("hina-73", 2417.3): "ASR row 520「さなたり!」",
    ("hina-63", 617.1): "ASR row 60「ん?ミスキさん?」",
    ("hina-63", 662.7): "ASR row 72「ミツキさん準備はできたか」",
    ("hina-63", 1548.9): "ASR row 275「指定って…」",
    ("hina-63", 1626.6): "ASR row 292「…フーヨンかわいいー!」",
    ("hina-70", 1564.4): "ASR row 306「炭鉱さん?」",
    ("hina-74", 2043.9): "ASR row 408「この他、…」",
    ("manosaba-a", 590.8): "ASR row 103「目立てめらん…」",
    ("manosaba-b", 78.6): "a yawn; ASR row 14「うわぁぁぁぁぁ」",
    ("callhome-1425", 208.4): "ASR row 82「なんだ、和吉君。」",
    ("callhome-1425", 209.5): "ASR row 83「和吉君?」",
}


def build(ident, stable=None):
    stable = stable if stable is not None else production(ident)[0]
    lines, junk = reference_jp(ident)
    t = Truth(ident, stable, lines, junk)
    t.rows = {i: _row(t, i) for i in range(len(stable))}
    _missing(t)
    return t


_TRUTH = {}


def load(ident):
    """The Truth of a file, built once per process (it is cheap enough not to cache on disk)."""
    if ident not in _TRUTH:
        _TRUTH[ident] = build(ident)
    return _TRUTH[ident]


# --- commands ----------------------------------------------------------------------------------------

def survey():
    total = Counter()
    for ident in FILES:
        t = load(ident)
        kinds = Counter(v["kind"] for v in t.rows.values())
        miss = Counter(("lexical" if m["lexical"] else "backch") + ("/ovl" if m["overlapped"] else "/gap") for m in t.missing)
        dist = [t.target(i) for i in t.rows]
        dist = [d for d in dist if d is not None]
        cer = sum(d for d, _ in dist) / max(1, sum(len(x) for _, x in dist))
        print(f"{ident}: rows {len(t.stable)}, {dict(kinds)}; missing {dict(miss)}; single rows' reading CER {cer:.1%}")
        group = ident.split("-")[0]
        total.update({(group, k): n for k, n in kinds.items()})
        total.update({(group, "missing " + k): n for k, n in miss.items()})
        total.update({(group, "rows"): len(t.stable), (group, "edit distance"): sum(d for d, _ in dist)})
    for key in sorted(total):
        print(key, total[key])


def listing():
    for ident in FILES:
        t = load(ident)
        for i, v in sorted(t.rows.items()):
            if v["kind"] in ("split", "multi"):
                words = [w["word"] for w in t.words(i)]
                cut = v.get("split")
                text = "".join(words[:cut[0]]) + " ‖ " + "".join(words[cut[0]:]) if cut else "".join(words)
                print(f"{ident} {i} {v['kind']} {cut} aligned {v['aligned']} {v['people']} {v.get('why', '')} | {text}")
        for m in t.missing:
            print(f"{ident} MISSING {'lexical' if m['lexical'] else 'backch'} {'overlapped' if m['overlapped'] else 'gap'} "
                  f"{m['start']:.1f} {m['person']}: {m['text']} | after {m['anchors']}")


if __name__ == "__main__":
    {"survey": survey, "list": listing}[sys.argv[1]]()

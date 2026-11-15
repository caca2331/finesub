"""Open MOSS passes with run-time rescues for its known failures, decoded by `decoder.Decoder`.

The model keeps its whole-file context throughout: a rescue rewinds the decode to a token boundary and continues, so
speaker labels stay those of the one pass. Each rule answers a failure seen in open MOSS's single passes or in MOSS
Pro's production patches (`mosspro.py`); what each does and why: `docs/joint-asr.md`「本地 MOSS 的救援」. In order of
where they act:

- in the pass: loops (re-decode with a repetition penalty, then sampled, then give the region up), whole-second
  timestamps, a skip ahead or a jump back over talk (forced to where the talk is), an early end with talk after it,
  the wrong language (re-decode with its first word, then its letters, banned);
- after the pass: holes it left (>= SHORT_GAP s with speech) decoded as their own windows, kept only where the
  window's clock agrees with the pass, their speakers mapped onto the pass's by voice; a window that hit its token
  budget is continued;
- on the finished transcript: a source longer than one pass holds is cut into parts and stitched as MOSS Pro's are;
  labels of one person merged by voice; ends running on past the speech pulled in.

Silero's lock is taken inside the GPU lock the caller holds (`joint.GPU_LOCK`), never the other way round.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from ....reporting import current_reporter
from . import decoder as D
from . import rules as R

FORCES_PER_GAP = 2
EOS_FORCES = 2
SKIP_AHEAD = 10.0
DENSE_WINDOW = 30.0
DENSE_VOICED = 9.0
#: talk that starts this little before the model's own start is its start, not a skip
MIN_SKIPPED = 5.0
SEGMENT_TOKENS = 400  # longest segment seen in any open or Pro run: 324 characters (~230 tokens)
EMPTY_RUN = 5
INT_STREAK = 2
PENALTY = 1.3
SAMPLE_T = 0.7
TOKENS_PER_SECOND = 12.0
#: first: ban the first word token the segment wrote; ban: all its letter tokens.
LANG_LEVERS = ("first", "ban")

_START = re.compile(r"^\]?\[(\d+\.\d\d)\]\[")
#: The tag is a speaker (`S01`) or an acoustic event (`音楽`): an event segment has no text and does not go into the
#: output, but the time it covers counts as heard.
_HEAD = re.compile(r"^\]?\[(\d+\.\d\d)\]\[([^\[\]]+)\]")
_DONE = re.compile(r"^\]?\[(\d+\.\d\d)\]\[([^\[\]]+)\](.*)\[(\d+\.\d\d)$", re.S)
_SPEAKER = re.compile(r"^S\d+$")
_DECIMAL1 = re.compile(r"\[\d+\.(\d)$")


@dataclass
class _Seg:
    boundary: int  # index of the token that opens it ('[' or '][')
    head_end: int  # index just after its `[Sxx]`
    start: float
    speaker: str
    text: str = ""
    end: float = 0.0
    end_token: int = 0  # index of the token that closes it (it also opens the next segment)


@dataclass
class Pass:
    """One guarded decode over one audio array (times relative to it). A gap `window` does not force skips or an
    early end: its audio is the hole itself."""

    model: object
    processor: object
    audio: np.ndarray
    spans: list[tuple[float, float]]
    prompt: str
    language: str | None
    budget_gib: float
    window: bool = False
    events: list[dict] = field(default_factory=list)

    def note(self, event: dict) -> None:
        self.events.append(event)
        current_reporter().debug("moss-local rescue", event)

    def __post_init__(self):
        tok = self.processor.tokenizer
        self.tok = tok
        self.dur = len(self.audio) / D.SR
        one = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
        self.t_open, self.t_close, self.t_join = one("[")[0], one("]")[0], one("][")[0]
        self.t_digit = {str(d): one(str(d))[0] for d in range(10)}
        self.t_dot = one(".")[0]
        #: never penalised: a re-decoded segment still has to write its end timestamp
        self.structural = {self.t_open, self.t_close, self.t_join, self.t_dot, *self.t_digit.values()}
        self.eos = self.model.generation_config.eos_token_id
        self.rng = torch.Generator(device="cuda").manual_seed(0)

    # --- Silero -------------------------------------------------------------------------------------------------
    def voiced(self, a: float, b: float) -> float:
        return sum(max(0.0, min(b, e) - max(a, s)) for s, e in self.spans)

    def onset(self, after: float, talk: bool = False) -> float | None:
        """Silero's first speech after `after`; with `talk`, the first one that starts a stretch of talk."""
        for s, e in self.spans:
            if e > after + 0.2:
                at = max(s, after)
                if not talk or self.voiced(at, at + DENSE_WINDOW) >= DENSE_VOICED:
                    return at
        return None

    # --- decode -------------------------------------------------------------------------------------------------
    def run(self) -> list[dict]:
        inputs = D.encode_inputs(self.processor, self.audio, self.prompt)
        n = inputs["input_ids"].shape[1]
        budget = int(TOKENS_PER_SECOND * self.dur + 512)
        dec = D.Decoder(self.model, inputs, budget, self.budget_gib)  # may get fewer positions than asked
        del inputs
        self.dec, self.n = dec, n
        self.ids: list[int] = []
        self.segs: list[_Seg] = []
        self.open: _Seg | None = None
        self.boundary = 0
        self.mode: tuple[str, int] | None = None  # (penalty | sample | first | ban, segment boundary) while re-decoding
        self.attempts: dict[int, int] = {}  # loop attempts per segment start (centiseconds)
        self.forces: dict[int, int] = {}  # forced onsets per jump target (the model's start, 10 s buckets)
        self.floor = 0.0  # audio given up on by a loop skip: no gap before it is a skip
        self._stop = False
        self.eos_forces = 0
        self.int_streak = 0
        self.pending: list[int] = []  # forced tokens still to append
        self.capped = False
        self.lang_on = True
        self.lang_from = 0  # first segment not yet tested for its language
        self.lang_first: dict[int, list[int]] = {}  # per segment start (centiseconds): its tokens as first decoded
        self.lang_attempts: dict[int, int] = {}
        self.banned: list[int] = []  # the tokens a wrong-language segment may not write while it is decoded again
        self.note({"kind": "kv", "format": dec.cache.kv, "positions": dec.max})  # the execution shape it ran in
        while not self._stop:
            if len(self.ids) >= budget or dec.pos >= dec.max - 1:
                self.capped = True
                self.note({"kind": "cap", "tokens": len(self.ids)})
                break
            if self.pending:
                self._feed()
                self.ids.append(self.pending.pop(0))
                self._after_token()
                continue
            logits = self._feed()
            nxt = self._choose(logits)
            if nxt == self.eos:
                if not self._at_eos():
                    break
                continue
            self.ids.append(nxt)
            self._after_token()
        self.acoustic = [{"start": s.start, "end": s.end, "tag": s.speaker} for s in self.segs
                         if not _SPEAKER.match(s.speaker)]
        return [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in self.segs
                if _SPEAKER.match(s.speaker)]

    def _feed(self) -> torch.Tensor:
        self.dec.pos = self.n - 1 + len(self.ids)
        return self.dec.logits(self.ids[-1] if self.ids else self.dec.prompt[-1])

    def _rewind(self, length: int) -> None:
        del self.ids[length:]
        self.segs = [s for s in self.segs if s.end_token < length]
        self.lang_from = min(self.lang_from, len(self.segs))
        self.open = None
        self.boundary = self.segs[-1].end_token if self.segs else 0
        self._reparse()

    def _reparse(self) -> None:
        """After a rewind: the open segment's state from the tokens since the last complete segment."""
        text = self._slice()
        m = _HEAD.match(text)
        if m:
            head = self._tokens_to(len(m.group(0)))
            self.open = _Seg(self.boundary, head, float(m.group(1)), m.group(2))

    def _slice(self) -> str:
        return self.tok.decode(self.ids[self.boundary:])

    def _tokens_to(self, chars: int) -> int:
        """Index just after the token that completes the first `chars` characters of the slice."""
        for k in range(self.boundary + 1, len(self.ids) + 1):
            if len(self.tok.decode(self.ids[self.boundary:k])) >= chars:
                return k
        return len(self.ids)

    def _choose(self, logits: torch.Tensor) -> int:
        if self.int_streak >= INT_STREAK:
            m = _DECIMAL1.search(self.tok.decode(self.ids[-6:])) if self.ids else None
            if m and m.group(1) == "0" and self.ids[-1] == self.t_digit["0"]:
                logits = logits.clone()
                logits[self.t_digit["0"]] = float("-inf")
        if self.mode and self.open:
            kind, _ = self.mode
            if kind in ("first", "ban"):
                logits = logits.clone()
                logits[torch.tensor(self.banned, dtype=torch.long, device="cuda")] = float("-inf")
                return int(logits.argmax())
            seen = torch.tensor(sorted(set(self.ids[self.open.head_end:]) - self.structural), dtype=torch.long,
                                device="cuda")
            if len(seen):
                logits = logits.clone()
                v = logits[seen]
                logits[seen] = torch.where(v > 0, v / PENALTY, v * PENALTY)
            if kind == "sample":
                p = torch.softmax(logits / SAMPLE_T, dim=-1)
                return int(torch.multinomial(p, 1, generator=self.rng))
        return int(logits.argmax())

    def _after_token(self) -> None:
        tok = self.ids[-1]
        if self.open is None:
            m = _START.match(self._slice())
            if m:
                self._started(float(m.group(1)))
            return
        o = self.open
        if tok in (self.t_close, self.t_join):
            m = _DONE.match(self.tok.decode(self.ids[self.boundary:-1]))
            if m:
                self._ended(float(m.group(4)), m.group(3))
                return
        body = self.ids[o.head_end:]
        if len(body) > SEGMENT_TOKENS or R.periodic(body):
            self._loop(o)

    def _started(self, start: float) -> None:
        reach = max([self.floor, *(s.end for s in self.segs)])
        first = not self.segs and self.floor == 0.0
        # a segment longer than GAP is a made-up end, not how far the pass got; the longest real one seen is 23 s
        written = max([self.floor, *(s.end for s in self.segs if s.end - s.start <= R.GAP)])
        if start < written - R.GAP:  # a jump back: within GAP it may be someone talking over the last segment
            key = -1 - int(written // 10)  # `forces` keys: >= 0 for a skip's place, < 0 for a jump back's
            at = self.onset(written) or written
            if self.forces.get(key, 0) < FORCES_PER_GAP:
                self.forces[key] = self.forces.get(key, 0) + 1
                self.note({"kind": "back", "reach": round(written, 2), "model_start": start, "forced": round(at, 2)})
                self._rewind(self.boundary + 1)
                self.pending = self._digits(at) + [self.t_join]
                return
        # A skip ahead -- but not before the first segment: a late first start is the model leaving out a pre-roll,
        # which the gap fill decodes as its own window, not a drift.
        if (not self.window and not first and start - reach >= R.GAP
                and self.voiced(reach, start) >= R.GATE_VOICED):
            key = int(start // 10)
            at = self.onset(reach, talk=True)
            if self.forces.get(key, 0) < FORCES_PER_GAP and at is not None and at < start - MIN_SKIPPED:
                self.forces[key] = self.forces.get(key, 0) + 1
                self.note({"kind": "skip", "reach": round(reach, 2), "model_start": start, "forced": round(at, 2),
                           "voiced": round(self.voiced(reach, start), 1)})
                self._rewind(self.boundary + 1)
                self.pending = self._digits(at) + [self.t_join]
                return
        self.open = _Seg(self.boundary, len(self.ids), start, "")  # head_end moves past `[Sxx]` in `_ended`

    def _ended(self, end: float, text: str) -> None:
        o = self.open
        m = _HEAD.match(self.tok.decode(self.ids[o.boundary:]))
        o.speaker, o.text, o.end = (m.group(2) if m else "S??"), text, end
        o.head_end = self._tokens_to(len(m.group(0))) if m else o.head_end
        o.end_token = len(self.ids) - 1
        for t in (o.start, end):
            self.int_streak = self.int_streak + 1 if abs(t - round(t)) < 1e-6 else 0
        self.segs.append(o)
        self.open = None
        self.boundary = o.end_token
        if self.mode and self.mode[1] == o.boundary:
            self.mode = None
        elif self.lang_on and self.mode is None and self._wrong_language():
            return
        said = [s for s in self.segs if _SPEAKER.match(s.speaker)]
        run = [s.text.strip() for s in said[-EMPTY_RUN:]]
        wordless = all(not re.search(r"\w", x) for x in run)
        # the same line from several people is a crowd shouting it
        echo = len(set(run)) == 1 and len({s.speaker for s in said[-EMPTY_RUN:]}) == 1
        if len(run) == EMPTY_RUN and (wordless or echo):
            first = said[-EMPTY_RUN]
            self.note({"kind": "empty-run", "at": first.start, "text": run[0][:20]})
            self._give_up(first)

    def _loop(self, o: _Seg) -> None:
        key = int(o.start * 100)
        k = self.attempts.get(key, 0)
        self.attempts[key] = k + 1
        if k < 2:
            kind = ("penalty", "sample")[k]
            self.note({"kind": f"loop-{kind}", "at": o.start, "tokens": len(self.ids) - o.head_end})
            head = self._head_end(o)
            self._rewind(head)
            self.open = _Seg(o.boundary, head, o.start, o.speaker)
            self.mode = (kind, o.boundary)
            return
        self.note({"kind": "loop-skip", "at": o.start})
        self._give_up(o)

    def _wrong_language(self) -> bool:
        """Test the speaker segments since the last test, once they hold enough text; True when it rewound. Still
        wrong after both levers, the segment is put back as first decoded and the pass is not tested again (the
        source may well be in that language there)."""
        said = [s for s in self.segs[self.lang_from:] if _SPEAKER.match(s.speaker)]
        verdict = R.wrong_language(" ".join(s.text for s in said), self.language)
        if not verdict:
            if verdict is not None:
                self.lang_from = len(self.segs)
            return False
        o = said[0]
        key = int(o.start * 100)
        head = self._head_end(o)
        first = self.lang_first.setdefault(key, self.ids[head:o.end_token + 1])
        k = self.lang_attempts.get(key, 0)
        self.lang_attempts[key] = k + 1
        self._rewind(head)
        self.lang_from = len(self.segs)
        if k == len(LANG_LEVERS):
            self.note({"kind": "language-keep", "at": o.start})
            self.lang_on = False
            self.pending = list(first)
            return True
        kind = LANG_LEVERS[k]
        self.note({"kind": f"language-{kind}", "at": o.start, "text": o.text[:60]})
        letters = [t for t in first[:-1] if t not in self.structural and any(ch.isalpha() for ch in self.tok.decode([t]))]
        self.banned = sorted(set(letters[:1] if kind == "first" else letters))
        self.open = _Seg(o.boundary, head, o.start, o.speaker)
        self.mode = (kind, o.boundary)
        return True

    def _head_end(self, o: _Seg) -> int:
        text = self.tok.decode(self.ids[o.boundary:])
        m = _HEAD.match(text)
        if not m:
            return o.head_end
        saved, self.boundary = self.boundary, o.boundary
        try:
            return self._tokens_to(len(m.group(0)))
        finally:
            self.boundary = saved

    def _give_up(self, o: _Seg) -> None:
        """Abandon the audio from `o.start` to the next onset SKIP_AHEAD s on: the next segment starts there."""
        self.mode = None
        at = self.onset(o.start + SKIP_AHEAD)
        self._rewind(o.boundary + 1)
        if at is None:
            self.ids.pop()  # the '[' / '][' that opened `o`
            self.pending = []
            self._stop = True
            return
        self.floor = at
        self.pending = self._digits(at) + [self.t_join]

    def _at_eos(self) -> bool:
        """True when decoding goes on (a forced continuation)."""
        reach = max([self.floor, *(s.end for s in self.segs)])
        if (not self.window and self.eos_forces < EOS_FORCES and self.ids and self.ids[-1] == self.t_close
                and self.dur - reach >= R.GAP and self.voiced(reach, self.dur) >= R.GATE_VOICED):
            at = self.onset(reach, talk=True)
            if at is not None:
                self.eos_forces += 1
                self.note({"kind": "eos", "reach": round(reach, 2), "forced": round(at, 2)})
                self.ids.pop()  # ']' -> '][' + onset + ']['
                self.pending = [self.t_join] + self._digits(at) + [self.t_join]
                return True
        return False

    def _digits(self, t: float) -> list[int]:
        return [self.t_digit[c] if c != "." else self.t_dot for c in f"{t:.2f}"]


@dataclass
class Result:
    segments: list[dict]
    events: list[dict]
    #: per part's main pass: where it starts in the source, its K/V format and positions
    passes: list[dict]


def transcribe(model, processor, audio: np.ndarray, language: str | None, budget_gib: float) -> Result:
    """Segments (seconds into `audio`), the rescue events and the passes' execution shape. `language` picks the
    prompt and the language test. A source longer than one pass holds in the budget is cut into parts overlapping
    by PART_OVERLAP and stitched as production stitches MOSS Pro's (hand-over in a gap near the overlap's middle,
    speakers linked by co-speaking seconds there)."""
    from ..mosspro import PART_OVERLAP, _silero_spans
    from ..stitch import stitch, windows

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    spans = _silero_spans(pcm)
    dur = len(audio) / D.SR
    prompt = D.prompt_for(language)
    size = D.longest_pass(model, TOKENS_PER_SECOND, budget_gib)
    if size <= PART_OVERLAP and dur > size:
        raise RuntimeError(f"the VRAM budget holds {size:.0f} s a pass, no more than the parts' {PART_OVERLAP:.0f} s "
                           "overlap")
    parts, events, passes = [], [], []
    for lo, hi in windows(0.0, dur, size, PART_OVERLAP):
        a, b = int(lo * D.SR), int(hi * D.SR)
        segs, ev = _part(model, processor, audio[a:b], pcm[a:b],
                         [(s - lo, e - lo) for s, e in spans if e > lo and s < hi], language, prompt, budget_gib)
        parts.append((lo, hi, [{**s, "start": s["start"] + lo, "end": s["end"] + lo} for s in segs]))
        kv = next(e for e in ev if e["kind"] == "kv")
        passes.append({"start": round(lo, 1), "kv": kv["format"], "positions": kv["positions"]})
        events += ev if lo == 0.0 else [{"kind": "part", "at": round(lo, 1), "events": ev}]
    segs = parts[0][2] if len(parts) == 1 else stitch(parts, segments=True)
    merges = _merge_by_voice(segs, pcm)
    if merges:
        events.append({"kind": "merge", "labels": merges})
    segs, trimmed = R.trim_tails(segs, spans)
    if trimmed:
        events.append({"kind": "tail", "segments": trimmed})
    return Result(segs, events, passes)


def _merge_by_voice(segs: list[dict], pcm: np.ndarray) -> list[tuple[str, str, float]]:
    """`rules.merge_labels` on the voiceprint centroids MOSS Pro's relink builds (WeSpeaker ResNet34 in PLDA space,
    weighted by clean seconds, labels under MIN_LABEL s left alone), at its threshold. Without the voiceprint model
    nothing is merged."""
    from .. import relink as RL

    embed = RL.VOICEPRINTS.embed()
    if embed is None:
        return []
    prints = RL._Prints(pcm, embed)
    return R.merge_labels(segs, lambda s: RL._centroids(s, s, prints), RL.THETA)


def _part(model, processor, audio: np.ndarray, pcm: np.ndarray, spans: list[tuple[float, float]],
          language: str | None, prompt: str, budget_gib: float) -> tuple[list[dict], list[dict]]:
    """One pass and its gap fills over `audio` (times relative to it)."""
    from ..relink import VOICEPRINTS, Relinker

    p = Pass(model, processor, audio, spans, prompt, language, budget_gib)
    segs = p.run()
    events = p.events
    if p.acoustic:
        events.append({"kind": "acoustic", "segments": p.acoustic})
    p.dec = None  # free the pass's K/V buffers and graphs before any window
    torch.cuda.empty_cache()
    relink = Relinker(lambda: pcm, VOICEPRINTS)
    dur = len(audio) / D.SR
    # (start, end, kind): `gap` and `short` windows get a lead-in either side; `cont` continues a window that hit its
    # token budget from where its last segment ended, without a lead-in that would hand it the end of that segment again.
    todo = [(a, b, "gap" if b - a >= R.GAP else "short") for a, b in R.gaps(segs, dur, R.SHORT_GAP)]
    while todo:
        a, b, kind = todo.pop(0)
        if p.voiced(a, b) < R.GATE_VOICED:
            continue
        lead = {"gap": 1.0, "short": R.SHORT_LEAD, "cont": 0.0}[kind]
        lo, hi = max(0.0, a - lead), min(dur, b + 1.0 if kind == "cont" else b + lead)
        sub = Pass(model, processor, audio[int(lo * D.SR):int(hi * D.SR)],
                   [(s - lo, e - lo) for s, e in spans if e > lo and s < hi], prompt, language, budget_gib,
                   window=True)
        extra = [{**s, "start": s["start"] + lo, "end": s["end"] + lo} for s in sub.run()]
        capped = sub.capped
        sub.dec = None
        torch.cuda.empty_cache()
        extra, anchors, off = R.kept_from_window(extra, segs, lo, hi, a, b, short=kind == "short")
        mapping = relink(segs, extra) if extra else {}
        events.append({"kind": "fill" if kind != "short" else "short", "gap": [round(a, 2), round(b, 2)],
                       "segments": len(extra), "anchors": anchors, "anchor_off": round(off, 2), "map": mapping,
                       "events": sub.events})
        segs = sorted([*segs, *extra], key=lambda s: s["start"])
        reach = max((x["end"] for x in extra), default=a)
        if capped and b - reach >= 1.0 and reach > a + 5.0:
            todo.insert(0, (reach, b, "cont"))
    return segs, events

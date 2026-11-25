"""Open MOSS passes with run-time rescues for its known failures, decoded by `decoder.Decoder`.

Within a part the model keeps its whole context: a rescue rewinds the decode to a token boundary and continues, so
speaker labels stay those of the one pass. Each rule answers a failure seen in open MOSS's single passes or in MOSS
Pro's production patches (`mosspro.py`); what each does and why: `docs/joint-asr.md`「本地 MOSS 的救援」. In order of
where they act:

- in the pass: the token stream read loosely (whitespace and a stray `]` between segments), and where it still does
  not parse decoded again (a head that does not parse: its start forced, then the stretch given up; EOS inside a
  segment: masked); loops (re-decode with a repetition penalty, then sampled, then give the region up), whole-second
  timestamps, a skip ahead or a jump back over talk (forced to where the talk is), an early end with talk after it,
  the wrong language (re-decode with its first word, then its letters, banned);
- after the pass: holes it left (>= SHORT_GAP s with speech) decoded as their own windows with the transcript
  around them (`rules.window`), all they decode replacing the context they decoded again (`rules.replace`), their
  speakers renamed apart (`relink.renumber`); a long hole's window whose first line skips talk at the hole's start
  forced back to it, once; a window that got on but stopped short of its hole goes on from where it got (continued
  when it hit its token budget); a Japanese segment with a long run of characters and no mark decoded again on its
  own and given the window's commas (`punctuate`);
- on the finished transcript: a source longer than PART_SECONDS (or than one pass holds) is cut into parts at
  Silero pauses (`cuts.window`) and stitched as MOSS Pro's are, each part's speakers kept apart; labels of one person
  merged by voice (`relink.linked`, the forced decode's too; also across the parts and gap windows; a gap window's
  label with too little voice to tell made unknown, except on the lines that say a replaced segment again, which
  take its label: `rules.twin`); ends running on past the speech pulled in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from ....reporting import current_reporter
from .. import cuts
from .. import relink as RL
from ..lines import japanese
from . import decoder as D
from . import rules as R

if TYPE_CHECKING:
    from ..mosspro import SpeechSpans

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
#: A pass's output budget: this many tokens a second of its audio, + 512. The densest 10-minute part of 13 hina
#: episodes (several people talking over each other) wrote about 14.4 a second into its final text, a floor: output
#: dropped again (masked loops, rewound heads) is not counted; 12 cut 8 of 87 parts short (owner 2026-11-23: 18).
TOKENS_PER_SECOND = 18.0
#: A part's audio at most (owner 2026-11-23): 10-minute parts beat one pass over the whole source on later parts and
#: on the degradations and speakers (docs/joint-asr.md, 设计取舍「本地 MOSS 一段至多 10 分钟」); a card that holds less
#: gets what it holds.
PART_SECONDS = 600.0
#: first: ban the first word token the segment wrote; ban: all its letter tokens.
LANG_LEVERS = ("first", "ban")

#: A head that has not parsed this many tokens after the token closing the segment before is a structural failure.
#: A head takes 9-13 (`][1547.81][S04]`, with a space or two more when the model writes them).
HEAD_TOKENS = 24
UNPARSED_TRIES = 1
#: EOS where the grammar has no end (inside a segment or its head) is masked this many times per segment start.
EOS_MASKS = 3

# The grammar is read loosely, the same on every well-formed stream: `[start][TAG] text[end]`, the end's `]` (or a
# `][` that opens the next segment) closing it. Before a bracket that opens a segment, or its tag, there may be the
# segment before's `]` and whitespace -- the model sometimes writes a whole pass as `[5.44] [5.58][S01]`.
_T = r"(\d+\.\d\d)"
_LEAD = r"^[\]\s]*\["
_START = re.compile(rf"{_LEAD}{_T}\]\s*\[")
#: The tag is a speaker (`S01`) or an acoustic event (`音楽`), never a time: an event segment has no text and does not
#: go into the output, but the time it covers counts as heard.
_TAG = r"((?!\d+\.\d\d\])[^\[\]]+)"
_HEAD = re.compile(rf"{_LEAD}{_T}\]\s*\[{_TAG}\]")
_DONE = re.compile(rf"{_LEAD}{_T}\]\s*\[{_TAG}\](.*)\[{_T}$", re.S)
#: a time where the tag goes: a run of timestamps outside any segment
_TIME_TAG = re.compile(rf"{_LEAD}{_T}\]\s*\[\d+\.\d\d\]")
_SEPARATORS = re.compile(r"[\]\s]*")
_OPENER = re.compile(r"\s*\[")
_SPEAKER = re.compile(r"^S\d+$")
_DECIMAL1 = re.compile(r"\[\d+\.(\d)$")


@dataclass
class _Seg:
    boundary: int  # index of the token before it: the first '[', or the one that closed the segment before
    head_end: int  # index just after its `[Sxx]`
    start: float
    speaker: str
    text: str = ""
    end: float = 0.0
    end_token: int = 0  # index of the token that closes it (a '][' also opens the next segment)
    headed: bool = False  # its `[Sxx]` has parsed


@dataclass
class Pass:
    """One guarded decode over one audio array (times relative to it). A gap `window` does not force skips or an
    early end: its audio is the hole itself. A long hole's window gets the `hole` ((start, end) in its audio): its
    first speaker line starting at or after the hole's start is checked once for skipping talk there."""

    model: object
    processor: object
    audio: np.ndarray
    spans: list[tuple[float, float]]
    prompt: str
    language: str | None
    budget_gib: float
    window: bool = False
    hole: tuple[float, float] | None = None
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
        self.rng: torch.Generator | None = None  # made on the logits' device at the first sample
        self.closers: dict[int, bool] = {}

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
        self.unparsed: dict[int, int] = {}  # structural failures per place (where the pass had got to, centiseconds)
        self.eos_masked: dict[int, int] = {}  # EOS masked mid-segment, per segment start (its boundary)
        self.skipped_at = -1  # segments parsed when a stretch that did not parse was last given up
        self.opener = [self.t_open]  # how the model last opened a segment after a `]` (`[`, ` [`)
        self.spaced = False  # it opened its latest segment after a `]` (not with a `][`)
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
                if self._may_end():
                    if not self._at_eos():
                        break
                    continue
                k = self.eos_masked.get(self.boundary, 0)
                if k >= EOS_MASKS:
                    self.note({"kind": "eos-mid-end", "text": self._slice()[-20:]})
                    break
                self.eos_masked[self.boundary] = k + 1
                self.note({"kind": "eos-mid", "text": self._slice()[-20:]})
                nxt = self._choose(logits, end=False)
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
            self.open = _Seg(self.boundary, head, float(m.group(1)), m.group(2), headed=True)

    def _slice(self) -> str:
        return self.tok.decode(self.ids[self.boundary:])

    def _tokens_to(self, chars: int) -> int:
        """Index just after the token that completes the first `chars` characters of the slice."""
        for k in range(self.boundary + 1, len(self.ids) + 1):
            if len(self.tok.decode(self.ids[self.boundary:k])) >= chars:
                return k
        return len(self.ids)

    def _choose(self, logits: torch.Tensor, end: bool = True) -> int:
        """The next token; with `end` False, not EOS."""
        if not end:
            logits = logits.clone()
            logits[self.eos] = float("-inf")
        if self.int_streak >= INT_STREAK:
            m = _DECIMAL1.search(self.tok.decode(self.ids[-6:])) if self.ids else None
            if m and m.group(1) == "0" and self.ids[-1] == self.t_digit["0"]:
                logits = logits.clone()
                logits[self.t_digit["0"]] = float("-inf")
        if self.mode and self.open:
            kind, _ = self.mode
            if kind in ("first", "ban"):
                logits = logits.clone()
                logits[torch.tensor(self.banned, dtype=torch.long, device=logits.device)] = float("-inf")
                return int(logits.argmax())
            seen = torch.tensor(sorted(set(self.ids[self.open.head_end:]) - self.structural), dtype=torch.long,
                                device=logits.device)
            if len(seen):
                logits = logits.clone()
                v = logits[seen]
                logits[seen] = torch.where(v > 0, v / PENALTY, v * PENALTY)
            if kind == "sample":
                if self.rng is None:
                    self.rng = torch.Generator(device=logits.device).manual_seed(0)
                p = torch.softmax(logits / SAMPLE_T, dim=-1)
                return int(torch.multinomial(p, 1, generator=self.rng))
        return int(logits.argmax())

    def _closes(self, tok: int) -> bool:
        """Whether `tok` can close a segment: it starts with ']' (`]`, `][`, `]\\n`...)."""
        if tok not in self.closers:
            self.closers[tok] = self.tok.decode([tok]).startswith("]")
        return self.closers[tok]

    def _may_end(self) -> bool:
        """Whether the grammar lets the pass end here: before anything, or after a segment's closing `]` (and
        whitespace) -- not inside a segment or its head, nor after a `][` that opened the next one."""
        return self.open is None and _SEPARATORS.fullmatch(self._slice()) is not None

    def _after_token(self) -> None:
        tok = self.ids[-1]
        if self.open is None:
            m = _START.match(self._slice())
            if m:
                self._started(float(m.group(1)))
            elif len(self.ids) - self.boundary > HEAD_TOKENS:
                self._unparsed()
            return
        o = self.open
        if self._closes(tok):
            m = _DONE.match(self.tok.decode(self.ids[self.boundary:-1]))
            if m:
                self._ended(float(m.group(4)), m.group(3))
                return
        if not o.headed:
            text = self._slice()
            m = _HEAD.match(text)
            if m:
                o.headed = True
                # a long hole's window: checked once its first line in the hole has a speaker, before its text, so a
                # loop or an end in that line cannot lose the talk it skipped
                if self.hole is not None and o.start >= self.hole[0] and _SPEAKER.match(m.group(2)) \
                        and self._skipped_hole(o):
                    return
            elif _TIME_TAG.match(text) or len(self.ids) - self.boundary > HEAD_TOKENS:
                self._unparsed()
                return
        body = self.ids[o.head_end:]
        if len(body) > SEGMENT_TOKENS or R.periodic(body):
            self._loop(o)

    def _unparsed(self) -> None:
        """What follows the last segment does not parse as a segment's head (text with no `[start][Sxx]`, a run of
        timestamps): decode it again from that segment's end, its start forced to Silero's next speech; failing
        there again, give up the stretch as a loop does."""
        reach = max([self.floor, *(s.end for s in self.segs)])
        key = int(reach * 100)
        k = self.unparsed.get(key, 0)
        self.unparsed[key] = k + 1
        text = self._slice()[:40]
        self.mode = None
        if k >= UNPARSED_TRIES:
            if self.skipped_at == len(self.segs):  # no segment since the last stretch given up: the pass is stuck
                self.note({"kind": "unparsed-end", "reach": round(reach, 2), "text": text})
                self._rewind(self.boundary + 1 if self.segs else 0)
                self._stop = True
                return
            self.skipped_at = len(self.segs)
            self.note({"kind": "unparsed-skip", "reach": round(reach, 2), "text": text})
            self._give_up(self.boundary, reach)
            return
        at = self.onset(reach)
        self.note({"kind": "unparsed", "reach": round(reach, 2), "forced": None if at is None else round(at, 2),
                   "text": text})
        if at is None:  # no speech left: the pass ends at the last segment
            self._rewind(self.boundary + 1 if self.segs else 0)
            self._stop = True
            return
        self._restart(self.boundary, at)

    def _restart(self, boundary: int, at: float) -> None:
        """Rewind to just after the token at `boundary` (the one that closed the segment before, kept) and force a
        segment start at `at`, opened the way the model opens one: after a `][` only the time; after a `]`, the
        opener it wrote there (` [`), else the one it last wrote after a `]`. Given a `[` it does not write there
        (`] [` passes), it wrote a timestamp where the tag goes, every time."""
        keep = boundary + 1 if self.segs or (self.ids and self.ids[0] == self.t_open) else 0
        own = self._opener(keep - 1) if keep else None
        self._rewind(own or keep)
        opened = bool(self.ids) and self.tok.decode(self.ids[-1:]).endswith("[")
        self.pending = ([] if opened else list(self.opener)) + self._digits(at) + [self.t_join]

    def _opener(self, boundary: int) -> int | None:
        """Index just after the opener written after the token at `boundary` (whitespace and a token ending in
        '['), when that token does not open the segment itself."""
        if self.tok.decode(self.ids[boundary:boundary + 1]).endswith("["):
            return None
        for k in range(boundary + 1, min(len(self.ids), boundary + 4)):
            text = self.tok.decode(self.ids[boundary + 1:k + 1])
            if _OPENER.fullmatch(text):
                return k + 1
            if text.strip():
                return None
        return None

    def _started(self, start: float) -> None:
        if self.segs:
            own = self._opener(self.boundary)
            self.spaced = own is not None
            if own is not None:
                self.opener = self.ids[self.boundary + 1:own]
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
                self._restart(self.boundary, at)
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
                self._restart(self.boundary, at)
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
            self._give_up(first.boundary, first.start)

    def _skipped_hole(self, o: _Seg) -> bool:
        """A long hole's window whose first line at or after the hole's start (`o`, its head just written) comes
        MIN_SKIPPED s or more after the first stretch of talk in the hole after what the window has written so far
        (a lead-in line running into the hole has said that part), with GATE_VOICED s of speech between (up to the
        hole's end): rewound to before `o` and forced to start at that talk, as the pass's own skip check does; talk
        a loop already gave up on (`floor`) is not gone back to. Asked once per window; True when it forced."""
        (a, b), self.hole = self.hole, None
        reach = max([a, self.floor, *(s.end for s in self.segs)])
        at = self.onset(reach, talk=True)
        skipped = self.voiced(at, min(o.start, b)) if at is not None else 0.0  # the talk in the hole only
        if at is None or at >= b or o.start - at < MIN_SKIPPED or skipped < R.GATE_VOICED:
            return False
        self.note({"kind": "hole-skip", "reach": round(reach, 2), "model_start": o.start, "forced": round(at, 2),
                   "voiced": round(skipped, 1)})
        self.mode = None  # a re-decode of `o` (a loop's penalty or sampling) does not carry over to the forced start
        self._restart(o.boundary, at)
        return True

    def _loop(self, o: _Seg) -> None:
        key = int(o.start * 100)
        k = self.attempts.get(key, 0)
        self.attempts[key] = k + 1
        if k < 2:
            kind = ("penalty", "sample")[k]
            self.note({"kind": f"loop-{kind}", "at": o.start, "tokens": len(self.ids) - o.head_end})
            head = self._head_end(o)
            self._rewind(head)
            self.open = _Seg(o.boundary, head, o.start, o.speaker, headed=True)
            self.mode = (kind, o.boundary)
            return
        self.note({"kind": "loop-skip", "at": o.start})
        self._give_up(o.boundary, o.start)

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
        self.open = _Seg(o.boundary, head, o.start, o.speaker, headed=True)
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

    def _give_up(self, boundary: int, start: float) -> None:
        """Abandon the audio from `start` to the next onset SKIP_AHEAD s on: the segment after the token at
        `boundary` starts there."""
        self.mode = None
        at = self.onset(start + SKIP_AHEAD)
        if at is None:
            self._rewind(boundary + 1)
            self.ids.pop()  # the '[' / '][' that opened the abandoned stretch
            self.pending = []
            self._stop = True
            return
        self.floor = at
        self._restart(boundary, at)

    def _at_eos(self) -> bool:
        """True when decoding goes on (a forced continuation)."""
        reach = max([self.floor, *(s.end for s in self.segs)])
        if (not self.window and self.eos_forces < EOS_FORCES and self.segs
                and self.dur - reach >= R.GAP and self.voiced(reach, self.dur) >= R.GATE_VOICED):
            at = self.onset(reach, talk=True)
            if at is not None:
                self.eos_forces += 1
                self.note({"kind": "eos", "reach": round(reach, 2), "forced": round(at, 2)})
                close = self.segs[-1].end_token
                if self.spaced and not self.tok.decode(self.ids[close:close + 1]).endswith("["):
                    # opened the way it opens segments: its ']' + its opener + onset + ']['
                    del self.ids[close + 1:]
                    self.pending = list(self.opener) + self._digits(at) + [self.t_join]
                else:  # ']' (and whitespace) -> '][' + onset + ']['
                    del self.ids[close:]
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


def transcribe(model, processor, audio: np.ndarray, language: str | None, budget_gib: float,
               speech: SpeechSpans) -> Result:
    """Segments (seconds into `audio`), the rescue events and the passes' execution shape. `language` picks the
    prompt (a later part's main pass gets `D.MIDSTART` too) and the language test; `speech` is the same source's
    int16 samples and Silero spans (the voiceprints and the rescues' speech test), shared with the stage's line
    timing. A source longer than PART_SECONDS, or than one pass holds in the budget, is cut into parts at Silero
    pauses (`cuts.window`; a part's start where the previous part's segments break too, so each part is planned once
    the one before is decoded) and stitched as production stitches MOSS Pro's (hand-over in a gap near the overlap's
    middle, each part's speakers kept apart)."""
    pcm = speech.pcm()
    spans = speech.spans()
    dur = len(audio) / D.SR
    prompt = D.prompt_for(language)
    size = min(PART_SECONDS, D.longest_pass(model, TOKENS_PER_SECOND, budget_gib))
    if size <= cuts.OVERLAP[1] and dur > size:
        raise RuntimeError(f"the VRAM budget holds {size:.0f} s a pass, no more than the parts' {cuts.OVERLAP[1]:.0f} s "
                           "overlap")
    parts, events, passes = [], [], []
    # a small budget looks less far back for a pause, so each part still gets on
    reach = min(cuts.REACH, (size - cuts.OVERLAP[1]) / 2)
    pauses, after, breaks = cuts.pauses_of(spans), None, []
    while True:
        (lo, hi), last = cuts.window(0.0, dur, size, pauses, after, breaks=breaks, reach=reach)
        a, b = int(lo * D.SR), int(hi * D.SR)
        segs, ev = _part(model, processor, audio[a:b], pcm[a:b],
                         [(s - lo, e - lo) for s, e in spans if e > lo and s < hi], language, prompt, budget_gib,
                         D.prompt_for(language, later=bool(parts)))
        parts.append((lo, hi, [{**s, "start": s["start"] + lo, "end": s["end"] + lo} for s in segs]))
        kv = next(e for e in ev if e["kind"] == "kv")
        passes.append({"start": round(lo, 1), "kv": kv["format"], "positions": kv["positions"]})
        events += ev if lo == 0.0 else [{"kind": "part", "at": round(lo, 1), "events": ev}]
        if last:
            break
        after, breaks = hi, cuts.segment_breaks(parts[-1][2])
    segs, merges, unknown = RL.linked(parts, segments=True, audio=lambda: pcm, backend="本地 MOSS",
                                      unmerged=RL.REUSE_UNMERGED)
    if unknown:
        events.append({"kind": "unknown", "labels": unknown})
    if merges:
        events.append({"kind": "merge", "labels": merges})
    segs, trimmed = R.trim_tails(segs, spans)
    if trimmed:
        events.append({"kind": "tail", "segments": trimmed})
    return Result(segs, events, passes)


def _part(model, processor, audio: np.ndarray, pcm: np.ndarray, spans: list[tuple[float, float]],
          language: str | None, prompt: str, budget_gib: float, main_prompt: str) -> tuple[list[dict], list[dict]]:
    """One pass and its gap fills over `audio` (times relative to it): `main_prompt` for the pass, `prompt` for the
    windows; the comma re-decodes get the later-part prompt, whatever part this is (its example lends them more
    commas)."""
    p = Pass(model, processor, audio, spans, main_prompt, language, budget_gib)
    segs = p.run()
    events = p.events
    if p.acoustic:
        events.append({"kind": "acoustic", "segments": p.acoustic})
    p.dec = None  # free the pass's K/V buffers and graphs before any window
    torch.cuda.empty_cache()
    fills = 0
    dur = len(audio) / D.SR
    # (start, end, kind): a `cont` goes on from where a window that hit its token budget stopped. The gaps are the
    # main pass's: a window whose context covers the next gap fills it, and that gap's own window decodes it again
    # and replaces it.
    todo = [(a, b, "gap" if b - a >= R.GAP else "short") for a, b in R.gaps(segs, dur, R.SHORT_GAP)]
    while todo:
        a, b, kind = todo.pop(0)
        if p.voiced(a, b) < R.GATE_VOICED:
            continue
        lo, hi = R.window(segs, a, b, dur, cont=kind == "cont", onset=p.onset(a))
        RL.VOICEPRINTS_CUDA.release()  # a window's length is budgeted by `memory_allocated()`: not with the model in it
        sub = Pass(model, processor, audio[int(lo * D.SR):int(hi * D.SR)],
                   [(s - lo, e - lo) for s, e in spans if e > lo and s < hi], prompt, language, budget_gib,
                   window=True, hole=(a - lo, b - lo) if kind != "short" and b - a >= R.GAP else None)
        extra = [{**s, "start": s["start"] + lo, "end": s["end"] + lo} for s in sub.run()]
        capped = sub.capped
        sub.dec = None
        torch.cuda.empty_cache()
        anchors, off = R.clock_off(extra, segs, lo, hi)
        if R.CLOCK_CHECK and off > R.ANCHOR_TOL:
            extra = []
        if extra:
            fills += 1
        mapping = RL.renumber(extra, fills) if extra else {}
        segs, gone = R.replace(segs, extra, lo, hi)
        for s in extra:
            if (lab := R.twin(s, gone)) is not None:
                s[RL.TWIN] = lab
        events.append({"kind": "short" if kind == "short" else "fill", "gap": [round(a, 2), round(b, 2)],
                       "window": [round(lo, 2), round(hi, 2)], "segments": len(extra), "replaced": len(gone),
                       "twins": sum(RL.TWIN in s for s in extra), "anchors": anchors, "anchor_off": round(off, 2),
                       "map": mapping, "events": sub.events})
        # A window that got on but not to the gap's end goes on from where it got: one that hit its token budget
        # continues; one that stopped by itself leaves a hole, filled as any other.
        reach = max((x["end"] for x in extra), default=a)
        if reach > a + 5.0:
            if capped and b - reach >= 1.0:
                todo.insert(0, (reach, b, "cont"))
            elif not capped and b - reach >= R.SHORT_GAP:
                todo.insert(0, (reach, b, "gap" if b - reach >= R.GAP else "short"))
    # `punctuate` re-decodes Japanese segments only, so the Japanese example whatever `--language` said
    segs, marked = punctuate(model, processor, audio, spans, segs, language, D.prompt_for("ja", later=True),
                             budget_gib)
    events += marked
    RL.VOICEPRINTS_CUDA.release()  # nor in the next part's pass
    return segs, events


def punctuate(model, processor, audio: np.ndarray, spans: list[tuple[float, float]], segs: list[dict],
              language: str | None, prompt: str, budget_gib: float) -> tuple[list[dict], list[dict]]:
    """`segs` with the commas a Japanese segment lost (`rules.unmarked`) taken from a window over it alone, LEAD
    s either side out to a pause (`cuts.padded`; `rules.graft`: commas only, no character changes), and a `punct`
    event per segment tried (with the window's text over the segment, `heard`)."""
    if language and not language.lower().startswith("ja"):  # as `lines.japanese_segment` reads it
        return segs, []
    dur = len(audio) / D.SR
    pauses, breaks = cuts.pauses_of(spans), cuts.segment_breaks(segs)
    out, events = [], []
    for s in segs:
        if not (japanese(s["text"]) and R.unmarked(s["text"])):
            out.append(s)
            continue
        lo, hi = cuts.padded(s["start"], s["end"], R.LEAD, pauses, dur, breaks=breaks)
        RL.VOICEPRINTS_CUDA.release()
        sub = Pass(model, processor, audio[int(lo * D.SR):int(hi * D.SR)],
                   [(a - lo, b - lo) for a, b in spans if b > lo and a < hi], prompt, language, budget_gib, window=True)
        heard = "".join(x["text"] for x in sub.run() if s["start"] <= lo + (x["start"] + x["end"]) / 2 <= s["end"])
        sub.dec = None
        torch.cuda.empty_cache()
        text, added, similar = R.graft(s["text"], heard)
        events.append({"kind": "punct", "segment": [round(s["start"], 2), round(s["end"], 2)], "added": added,
                       "similarity": round(similar, 2), "heard": heard, "events": sub.events})
        out.append({**s, "text": text} if added else s)
    return out, events

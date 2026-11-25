"""The forced decode (`--asr moss-diarization-aligner`): another ASR's lines written into local MOSS's own output
stream, the model choosing only each line's speaker label.

The stream is written as MOSS writes it, `[0.00][S01]text[2.99][3.04][S01]text[5.02]...`: the times the lines' own
(relative to the part), the text the lines', lines in time order. At a line's label the candidates are the labels
used so far plus one new one (at most MAX_LABELS); the pick is the one the model gives the highest probability where
it stands, before the text (the `head` rule of the exploration, `docs/joint-asr.md`). The pick is committed into the
context and the next line goes on from it. Each line keeps every candidate's log-probability: the margin (best minus
second, None with one candidate) says how sure the pick was.

Known tokens go through `blocks.Blocks` a block at a time; the label's two digits are single steps. The K/V are bf16
only (the blocks write bf16). Parts as local MOSS's (`rescue.PART_SECONDS`, cut at Silero pauses by `cuts.window`),
each also held to what bf16 K/V fit for its own lines (a dense stretch gets a shorter part); a part with no line is
not decoded. The default prompt (`decoder.prompt_for`): the text is given, so the later-part example is not added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ....reporting import current_reporter
from .. import cuts
from . import decoder as D

#: K/V positions a line takes besides its text: `][`, two times of up to 7 digits, `]`, `[S`, two label digits, `]`,
#: `[` (21 past 1000 s), with room to spare
ROW_TOKENS = 24
#: label numbers have two digits
MAX_LABELS = 99
#: a part is shortened by this factor until its lines' K/V fit
SHRINK = 0.8


@dataclass
class Part:
    lo: float
    hi: float
    kv: str | None
    #: per line in it: (index into the lines, pick, {label: log-probability})
    picks: list[tuple[int, int, dict[int, float]]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"lo": round(self.lo, 3), "hi": round(self.hi, 3), "kv": self.kv,
                "rows": [{"row": i, "pick": n, "cands": {str(k): round(v, 3) for k, v in c.items()}}
                         for i, n, c in self.picks]}


def margin(cands: dict[int, float]) -> float | None:
    """Best candidate's log-probability minus the second's; None with one candidate."""
    if len(cands) < 2:
        return None
    top = sorted(cands.values(), reverse=True)
    return round(top[0] - top[1], 3)


class _Stream:
    def __init__(self, model, processor, pcm: np.ndarray, budget_tokens: int, budget_gib: float,
                 language: str | None):
        from .blocks import Blocks

        self.tok = processor.tokenizer
        inputs = D.encode_inputs(processor, pcm, D.prompt_for(language))
        self.dec = D.Decoder(model, inputs, budget_tokens, budget_gib, kv="bf16")
        self.blocks = Blocks(self.dec)
        self.cur = self.dec.logits(self.dec.prompt[-1]).clone()
        self._piece: dict[str, int] = {}

    def one(self, s: str) -> int:
        if s not in self._piece:
            ids = self.tok.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"{s!r} is not one token of local MOSS's tokenizer: {ids}")
            self._piece[s] = ids[0]
        return self._piece[s]

    def digits(self, x: str) -> list[int]:
        return [self.one(ch) for ch in x]

    def feed(self, ids: list[int]) -> None:
        if len(ids) > 1:
            self.cur = self.blocks.feed(ids).clone()
        else:
            for t in ids:
                self.cur = self.dec.logits(t).clone()

    def label_lps(self, used: int) -> dict[int, float]:
        """{n: log p(label n)} for labels 1 .. used+1 (at most MAX_LABELS) where the stream stands (after `[S`), one
        step per first digit; leaves the stream where it was."""
        import torch

        pos, cur = self.dec.pos, self.cur.clone()
        top = min(used + 1, MAX_LABELS)
        first = torch.log_softmax(cur, -1)
        out: dict[int, float] = {}
        for d in sorted({n // 10 for n in range(1, top + 1)}):
            self.dec.pos = pos
            lp = float(first[self.one(str(d))])
            second = torch.log_softmax(self.dec.logits(self.one(str(d))), -1)
            for n in range(max(1, 10 * d), min(top, 10 * d + 9) + 1):
                out[n] = lp + float(second[self.one(str(n % 10))])
        self.dec.pos, self.cur = pos, cur
        return out


def decode_part(model, processor, pcm: np.ndarray, lines: Sequence[dict], lo: float, budget_gib: float,
                language: str | None) -> list[tuple[int, dict[int, float]]]:
    """`lines` (absolute seconds, in time order, all inside the part) forced over `pcm` (16 kHz float from `lo` s);
    -> (pick, candidates) per line."""
    import torch

    tok = processor.tokenizer
    texts = [tok.encode(r["text"], add_special_tokens=False) for r in lines]
    f = _Stream(model, processor, pcm, sum(len(t) + ROW_TOKENS for t in texts) + 64, budget_gib, language)
    out, used, pending = [], 0, []
    for r, text in zip(lines, texts):
        f.feed(pending + [f.one("][" if pending else "[")] + f.digits(f"{max(0.0, r['start'] - lo):.2f}")
               + [f.one("]"), f.one("[S")])
        cands = f.label_lps(used)
        pick = max(cands, key=cands.get)
        out.append((pick, cands))
        used = max(used, pick)
        pending = (f.digits(f"{pick:02d}") + [f.one("]")] + text + [f.one("[")]
                   + f.digits(f"{max(0.0, r['end'] - lo):.2f}"))
    f = None  # frees the K/V buffers and the graphs
    torch.cuda.empty_cache()
    return out


def label(model, processor, audio: np.ndarray, lines: Sequence[dict], spans: list[tuple[float, float]],
          language: str | None, budget_gib: float, part_seconds: float) -> list[Part]:
    """Every part's picks for `lines` (absolute seconds, in time order) over the whole `audio`; a line goes into
    each part it lies wholly inside, the last part's end aside (the lines' times are clamped to the container's
    duration, which can be a few ms past the decoded samples')."""
    tok = processor.tokenizer
    dur = len(audio) / D.SR
    sizes = [len(tok.encode(r["text"], add_special_tokens=False)) + ROW_TOKENS for r in lines]
    per_second = sum(sizes) / dur if dur > 0 else 0.0
    size = min(part_seconds, D.longest_pass(model, per_second * 1.1, budget_gib, kv="bf16"))
    pauses, after, breaks, parts = cuts.pauses_of(spans), None, [], []
    while True:
        local = size
        while True:
            if local <= cuts.OVERLAP[1] and dur > local:  # parts no longer than their overlap make no headway
                raise RuntimeError(f"the VRAM budget holds no part of the lines from {after or 0.0:.0f} s on in "
                                   "bf16 K/V")
            reach = min(cuts.REACH, max(0.0, (local - cuts.OVERLAP[1]) / 2))
            (lo, hi), last = cuts.window(0.0, dur, local, pauses, after, breaks=breaks, reach=reach)
            mine = [i for i, r in enumerate(lines) if r["start"] >= lo and (last or r["end"] <= hi)]
            prompt = int(D.AUDIO_TOKENS_PER_SECOND * (hi - lo)) + D.PROMPT_TEXT_TOKENS
            want = prompt + sum(sizes[i] for i in mine) + 64 + D.BUCKET
            if not mine or D.fits(model, prompt, budget_gib)["bf16"] >= want:
                break
            local *= SHRINK
        part = Part(lo, hi, None)
        if mine:
            a, b = int(lo * D.SR), int(hi * D.SR)
            got = decode_part(model, processor, audio[a:b], [lines[i] for i in mine], lo, budget_gib, language)
            part.kv = "bf16"
            part.picks = [(i, n, c) for i, (n, c) in zip(mine, got)]
        current_reporter().debug("moss forced part", {"lo": round(lo, 1), "hi": round(hi, 1), "lines": len(mine)})
        parts.append(part)
        if last:
            break
        after, breaks = hi, cuts.segment_breaks([lines[i] for i in mine])
    return parts

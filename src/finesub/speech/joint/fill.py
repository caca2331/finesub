"""Which MOSS Pro key the next piece of a source goes on, and where that piece ends (owner 2026-11-09).

Keys are filled free hours first and cut as little as possible: a piece that fits one key's free room goes whole
into the key with the least room that holds it; one that fits none is cut across keys. Every candidate plan for
the rest of the source is costed on its real cuts and overlaps, and the plans are compared, in order, on:

1. the seconds not on free hours: billed to the paid key (its free room is filled and cut like any other key's;
   a rest no free hours hold may stay on it whole, billed, when it fits one call), or, without a paid key, not
   transcribed today;
2. the number of cuts;
3. the keys, by their room as planning began: the pieces before the last fill the keys with the least room first
   (a piece the balance pays comes after the free ones), the last goes into the key whose room fits it tightest;
   then the keys' order in `.env`;
4. the seams: longer Silero pauses at the cut and at the next piece's start, then later cuts.

A cut piece ends at a limit T or at the middle of a pause in the REACH s before it; the next starts OVERLAP[0] s
before that end or at the middle of a pause up to OVERLAP[1] s before it. T is the key's room or PART_SECONDS; where
that would leave the key under MIN_ROOM, also the point that leaves it exactly MIN_ROOM; for the paid key also the
hour (or the source's end) past its room, billed, and the points where the rest fits whole into another key's free
room (both owner 2026-11-10). A key with less than MIN_ROOM s of room takes only a piece that ends the source (rooms
are judged rounded to 0.1 s). A
later free end never costs more on rules 1-2; a billed one bills less the earlier it ends while the rest lands on
free hours, and a plan whose billed piece is followed by several free ones costs the same with those free pieces
first. So the limits keep the cheapest plans and pauses only move a cut for rule 4, with one known, bounded miss:
on a paid key whose own free room is under MIN_ROOM, the cheapest end can be a pause just past that room, which is
not tried (at most REACH s more billed). Only the plan's first piece is used: the next is planned again
against the account as it then stands (`mosspro.transcribe`). So only that piece tries pauses; the rest of the plan
is costed on its limits, which keeps planning inside the ledger lock to about a second for a 4 h source. Why each
rule: `docs/joint-asr.md`「设计取舍」.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

#: A call's longest audio, overlap included.
PART_SECONDS = 3600.0
#: The next piece starts this many seconds before the cut: at least, at most.
OVERLAP = (60.0, 90.0)
#: A cut looks for a pause this far before its limit.
REACH = 120.0
#: Room below this takes no cut piece (owner 2026-11-09).
MIN_ROOM = 600.0
#: Pauses tried per cut (besides the latest) and per start, for the piece about to go out: the longest ones.
TRIED_ENDS = 3
TRIED_STARTS = 2
#: Ranks a piece the balance pays, unless it is the last, after every free one (rule 3): spent first, the balance
#: is lost if the run stops or another task takes the free hours meanwhile.
BILLED_FIRST = 1e7


@dataclass(frozen=True)
class Key:
    room: float  # free audio seconds left today
    paid: bool = False  # past its room the balance pays


@dataclass(frozen=True)
class Piece:
    start: float
    end: float
    key: int
    billed: float  # seconds past the key's free room


def pauses_of(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """(middle, length) of the silences between Silero's speech spans."""
    out, reach = [], None
    for s, e in sorted(spans):
        if reach is not None and s > reach:
            out.append(((reach + s) / 2, s - reach))
        reach = e if reach is None else max(reach, e)
    return out


def next_starts(cut: float, pauses: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """(start, pause length) the piece after a cut at `cut` may begin at."""
    return [(cut - OVERLAP[0], 0.0), *_longest(pauses, cut - OVERLAP[1], cut - OVERLAP[0], TRIED_STARTS)]


def plan(total: float, starts: Sequence[tuple[float, float]], keys: Sequence[Key],
         pauses: Sequence[tuple[float, float]]) -> list[Piece]:
    """The best plan for [start, total], over the candidate `starts` ((start, pause length)); empty when no key
    can take a piece. Without a paid key the plan may stop short of `total`."""
    any_paid = any(k.paid for k in keys)

    def combine(piece: Piece, room: float, end_pause: float, rest):
        r_score, r_pieces = rest
        unfree, cuts, fits, seams, ends, order = r_score
        # The key's room when planning began: least first, the last piece into the tightest key that holds it.
        # Not the room as the plan leaves it, which moves with every cut and would outrank the seams.
        fit = room + (BILLED_FIRST if piece.billed and r_pieces else 0.0)  # the balance spent before free hours
        return ((round(piece.billed) + unfree, cuts + (1 if r_pieces else 0), (round(fit),) + fits,
                 seams - end_pause, (-round(piece.end, 1),) + ends, (piece.key,) + order),
                (piece,) + r_pieces)

    def stop(at: float):
        return (round(total - at), 0, (), 0.0, (), ()), ()

    def options(s: float, rooms: tuple[float, ...], seams: bool) -> list:
        out = []
        rest = total - s
        for k, key in enumerate(keys):
            room = rooms[k]
            if rest <= PART_SECONDS and (key.paid or rest <= room):
                piece = Piece(s, total, k, max(0.0, rest - room) if key.paid else 0.0)
                out.append(combine(piece, rooms0[k], 0.0, ((0, 0, (), 0.0, (), ()), ())))
                if rest <= room:  # whole on free hours: no cut can do better
                    continue
            limits = [s + min(room, PART_SECONDS)] if room >= MIN_ROOM else []  # the paid key's free room too
            if limits and room - PART_SECONDS < MIN_ROOM and room - MIN_ROOM > OVERLAP[1]:
                limits.append(s + room - MIN_ROOM)  # the key keeps MIN_ROOM for another piece
            if key.paid and room < min(PART_SECONDS, rest):
                billed_limit = s + min(PART_SECONDS, rest)
                limits.append(billed_limit)  # billed past its room
                # ... or billed only up to where the rest fits whole into another key's free room (owner 2026-11-10)
                limits += [f for j, other in enumerate(keys) if j != k and not other.paid and rooms[j] > 0
                           and s + max(room, OVERLAP[1]) < (f := total + OVERLAP[0] - rooms[j]) < billed_limit]
            for limit in limits:
                for end, length in _ends(limit, pauses, seams):
                    if end <= s + OVERLAP[1] or end >= total:
                        continue
                    used = end - s
                    piece = Piece(s, end, k, max(0.0, used - room) if key.paid else 0.0)
                    if not piece.billed and room < MIN_ROOM:  # the paid key's small free room: no free cut piece
                        continue
                    left = rooms[:k] + (round(max(0.0, room - used), 1),) + rooms[k + 1:]
                    nexts = next_starts(end, pauses) if seams else [(end - OVERLAP[0], 0.0)]
                    tails = [_with_start(rest_of(round(a, 2), left), p) for a, p in nexts]
                    tails = [t for t in tails if t is not None]
                    if not any_paid:
                        tails.append(stop(end))
                    if tails:
                        out.append(combine(piece, rooms0[k], length, min(tails, key=lambda t: t[0])))
        return out

    @lru_cache(maxsize=None)
    def rest_of(s: float, rooms: tuple[float, ...]):
        found = options(s, rooms, False)
        return min(found, key=lambda o: o[0]) if found else None

    rooms0 = tuple(round(k.room, 1) for k in keys)
    found = []
    for a, p in starts:
        here = options(round(a, 2), rooms0, True)
        if here:
            found.append(_with_start(min(here, key=lambda o: o[0]), p))
    found = [f for f in found if f[1]]
    return list(min(found, key=lambda f: f[0])[1]) if found else []


def _with_start(result, pause: float):
    if result is None or not result[1]:
        return result
    (unfree, cuts, fits, seams, ends, order), pieces = result
    return (unfree, cuts, fits, seams - pause, ends, order), pieces


def _ends(limit: float, pauses: Sequence[tuple[float, float]], seams: bool) -> list[tuple[float, float]]:
    """(end, pause length) a cut up to `limit` is tried at: the limit, and for the piece about to go out
    (`seams`) the latest pause before it and the TRIED_ENDS longest."""
    if not seams:
        return [(limit, 0.0)]
    inside = [(m, length) for m, length in pauses if limit - REACH <= m < limit]
    keep = set(_longest(inside, limit - REACH, limit, TRIED_ENDS)) | ({max(inside)} if inside else set())
    return [(limit, 0.0), *sorted(keep, key=lambda p: (-p[1], -p[0]))]


def _longest(pauses: Sequence[tuple[float, float]], lo: float, hi: float, n: int) -> list[tuple[float, float]]:
    inside = [(m, length) for m, length in pauses if lo <= m < hi]
    return sorted(inside, key=lambda p: (-p[1], -p[0]))[:n]

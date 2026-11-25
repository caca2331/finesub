"""The name check: a session that checks term lines -- their Chinese name first, their source word and
description too -- and corrects what is wrong (docs/knowledge.md「中文定名的核查」).

Deterministic halves around one session:

* **which lines** -- `due_lines` (the 未核 names `name_trust.check_due` says are owed a check, the scan of
  `knowledge verify`) or the lines a task searched (`name_check_step.py`, with the task's searches as clues);
  `build_claims` adds the lines whose name contains a checked one, owed a check or not;
* **the session** (`run_name_check_session`, `kb_name_check_v1.md`): per line one of confirmed / corrected /
  not_found; a correction sets any of the source word, the Chinese name and the description (`SET_FIELDS`), a
  not_found any but the name; confirmed and corrected carry the URL of the Chinese page the name stands on. Lines
  without a usable conclusion are asked again in the same conversation (`kb_name_check_repair_v1.md`), the
  correction windows' `max_retries_per_window` times. The session searches on its own where its route can
  (`name_check_route`); elsewhere the harness searches for it (the local path: the clue pages fetched ahead, the
  queries it lists in a first turn searched, the conclusions asked in a second), and a line it had nothing to read
  for is left unanswered;
* **landing** (`book_name_checks`): ``external`` evidence on ``payload.zh`` -- a confirmation of the current name
  (confirmed, or corrected without a new name), a refutation of a name replaced, ``unverifiable`` for a not_found
  -- and a harness revision per corrected line. A 人定 line is not changed (a new name refutes its name, the
  correction is listed for the user). A replaced name's successor gets no confirmation from the session that
  wrote it: it is 未核 and checked twice more like any other.

The URL is the session's word, not something the harness read: a name becomes 已核 only on two confirmations at
least `name_trust.RECHECK_GAP` apart, from two sessions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Collection, Mapping, Sequence

from ..output_tags import find_top_level_tag_blocks, parse_line_items
from ..prompt_compose import load_prompt_template
from .node.matching import scan_normalize
from .node.model import NodeVersion, digest, payload_group_hash
from .node.name_trust import (
    HUMAN,
    NOT_FOUND_KINDS,
    NOT_FOUND_TEMPORARY,
    ZH_FIELD,
    ZhStatus,
    check_due,
    zh_name,
    zh_statuses,
)
from .node.signals import record_evidence
from .node.store import KnowledgeStore

NAME_CHECK_ALGO_VERSION = "kb-name-check-2"
#: The revision note of what the check lands (one harness revision per line).
NAME_CHECK_REVISION_NOTE = "kb-name-check"
RESULTS_TAG = "name_check_results"
VERDICTS = ("confirmed", "corrected", "not_found")
#: What a correction may set: the answer's key -> the payload field. Aliases are not among them (an alias edit is
#: an item op, and a check has not needed one).
SET_FIELDS = {"source": "payload.surface", "zh": "payload.zh", "desc": "payload.desc"}
#: How the report names each field.
FIELD_LABELS = {"source": "原文", "zh": "定名", "desc": "描述"}
#: A checked name shorter than this brings no derived lines: 「散兵」「晶矿」 are inside half an entry.
DERIVED_MIN_CHARS = 3
DERIVED_MAX_LINES = 10
_URL_RE = re.compile(r"https?://\S+")
#: The local path (`CheckRoute` ``local``): the block of the first turn's queries, ``id|query`` a line.
QUERIES_TAG = "search_queries"
LOCAL_QUERIES_PER_LINE = 2
#: Clue pages fetched for a line before the first turn.
LOCAL_PAGES_PER_LINE = 2
LOCAL_RESULTS_PER_QUERY = 5
#: What a local batch of 8 lines reads at most: search results 8 × 2 queries × 5 results × 600 tokens
#: (`web_search.SEARCH_SNIPPET_MAX_TOKENS`) ≈ 48k, clue pages 8 × 2 × 1,800 (`EXTRACT_CONTENT_MAX_TOKENS`) ≈ 29k.
LOCAL_RESULT_TOKENS = 48_000
LOCAL_PAGE_TOKENS = 29_000


# ---- which lines -------------------------------------------------------------------------------------------------


def _subjects_of(store: KnowledgeStore, rev: int) -> dict[str, str]:
    """``node -> the surface of the first entry it hangs under``."""

    subject_of: dict[str, str] = {}
    for subject in store.subjects(rev):
        stack = [subject.local_id]
        while stack:
            parent = stack.pop()
            for membership in store.children(parent, rev):
                if membership.child_id not in subject_of:
                    subject_of[membership.child_id] = str(subject.payload.get("surface") or "")
                    stack.append(membership.child_id)
    return subject_of


def _checkable(store: KnowledgeStore, rev: int) -> list[NodeVersion]:
    """Term lines with a name that may be sent out at all: not tentative, and every placement lets it
    (`verify._externally_verifiable` -- the same data rule as `knowledge verify`'s other object)."""

    from .verify import _externally_verifiable, _placements

    placements = _placements(store, rev)
    return [
        node for node in store.nodes_of_kind("term", rev)
        if node.maturity != "tentative" and zh_name(node) is not None and _externally_verifiable(node, placements)
    ]


def due_lines(store: KnowledgeStore, rev: int | None = None, *, limit: int, now: datetime | None = None) -> list[str]:
    """The node ids the check is owed now (`name_trust.check_due`), at most ``limit``, longest unchecked first
    (never checked before all)."""

    at = store.current_rev() if rev is None else rev
    nodes = _checkable(store, at)
    statuses = zh_statuses(store, nodes, at)
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    due = [
        (statuses[node.local_id].last_checked or epoch, index, node.local_id)
        for index, node in enumerate(nodes)
        if check_due(statuses[node.local_id], now)
    ]
    return [node_id for *_, node_id in sorted(due)[:max(0, limit)]]


@dataclass
class Claim:
    """One line sent to the session."""

    claim_id: str
    node_id: str
    surface: str
    zh: str
    desc: str
    subject: str
    #: what a task's searches found on this line (`name_check_step.collect_clues`): ``found``, ``url``, ``quote``,
    #: ``query``, ``origin``
    clues: list[dict[str, str]] = field(default_factory=list)
    #: the claim whose name this one's contains (a derived line), None for a line sent for itself
    derived_from: str | None = None
    #: sent only because its name contains a sent one's: it needs an answer only when that one's name changes
    along: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id, "node_id": self.node_id, "surface": self.surface, "zh": self.zh,
            "desc": self.desc, "subject": self.subject, "clues": list(self.clues), "derived_from": self.derived_from,
            "along": self.along,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Claim":
        return cls(**{key: data[key] for key in ("claim_id", "node_id", "surface", "zh", "desc", "subject")},
                   clues=list(data.get("clues") or []), derived_from=data.get("derived_from"),
                   along=bool(data.get("along")))


def build_claims(
    store: KnowledgeStore, node_ids: Sequence[str], rev: int | None = None,
    *, clues: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
    derive_for: Collection[str] | None = None, skip: Collection[str] = (),
) -> list[Claim]:
    """The session's lines: each of ``node_ids``, then the lines of the same entry whose name contains its name (a
    name of at least `DERIVED_MIN_CHARS`, at most `DERIVED_MAX_LINES` of them), each line once. A derived line comes
    whether or not it is owed a check of its own: when the checked name changes, the session changes the names built
    on it in the same answer. A 人定 line does not come (the check changes none), nor one in ``skip``; only the lines
    in ``derive_for`` bring any (None: every line). Each line is marked as derived from the line with the longest name
    inside its own, a line of ``node_ids`` too; the lines brought along are marked ``along``."""

    at = store.current_rev() if rev is None else rev
    subject_of = _subjects_of(store, at)
    checkable = _checkable(store, at)
    by_id = {node.local_id: node for node in checkable}
    statuses = zh_statuses(store, checkable, at)
    claims: list[Claim] = []
    taken: set[str] = set()

    def add(node: NodeVersion, derived_from: str | None) -> Claim:
        claim = Claim(
            claim_id=f"n{len(claims) + 1}", node_id=node.local_id, surface=str(node.payload.get("surface") or ""),
            zh=zh_name(node) or "", desc=str(node.payload.get("desc") or ""),
            subject=subject_of.get(node.local_id, ""),
            clues=[dict(clue) for clue in (clues or {}).get(node.local_id, ())], derived_from=derived_from,
            along=derived_from is not None,
        )
        claims.append(claim)
        taken.add(node.local_id)
        return claim

    for node_id in node_ids:
        node = by_id.get(node_id)
        if node is None or node_id in taken:
            continue
        add(node, None)
    requested = {claim.node_id for claim in claims} | set(skip)
    for claim in list(claims):
        if len(claim.zh) < DERIVED_MIN_CHARS or (derive_for is not None and claim.node_id not in derive_for):
            continue
        # each checked line's own first ten, whatever the other checked lines took: the set does not hang on order
        derived = [
            other for other in checkable
            if other.local_id not in requested
            and subject_of.get(other.local_id) == claim.subject and claim.zh in (zh_name(other) or "")
            and statuses[other.local_id].state != HUMAN
        ][:DERIVED_MAX_LINES]
        for other in derived:
            if other.local_id not in taken:
                add(other, claim.claim_id)
    # Which line each one derives from, from the whole set at once: the longest name inside its own (「水晶矿脉」
    # follows 「水晶矿」, not 「水晶」), whatever order the lines came in. A 人定 line follows none.
    for claim in claims:
        parents = [
            other for other in claims
            if other is not claim and other.subject == claim.subject and len(other.zh) >= DERIVED_MIN_CHARS
            and other.zh in claim.zh and other.zh != claim.zh
        ]
        human = statuses[claim.node_id].state == HUMAN if claim.node_id in statuses else False
        # the longest; of equal length, the first by name (「天空城」 before 「空城堡」), not by the order lines came in;
        # two lines with that very name leave it no single parent to follow: it is checked on its own
        ranked = sorted(parents, key=lambda other: (-len(other.zh), other.zh))
        tied = len(ranked) > 1 and ranked[0].zh == ranked[1].zh
        claim.derived_from = None if human or not ranked or tied else ranked[0].claim_id
    return claims


# ---- the session ------------------------------------------------------------------------------------------------


def _phrases(table: str) -> dict[str, Any]:
    from string import Template

    from ..prompt_tables import prompt_phrases

    return {key: Template(value) for key, value in prompt_phrases(table).items()}


def _clue_text(clue: Mapping[str, str], phrases: Mapping[str, Any]) -> str:
    """One clue under its line: where it came from, then the spelling found, its page and quote -- or the query that
    found nothing."""

    origin = str(clue.get("origin") or "")
    where = (phrases["origin_prepare"].substitute() if origin == "prepare"
             else phrases["origin_window"].substitute(window=origin.removeprefix("window ").strip()) if origin
             else "")
    if not clue.get("found"):
        return phrases["clue_none"].substitute(origin=where, query=clue.get("query") or "")
    text = phrases["clue"].substitute(origin=where, found=clue["found"])
    if clue.get("url"):
        text += phrases["clue_source"].substitute(url=clue["url"])
    if clue.get("quote"):
        text += phrases["clue_quote"].substitute(quote=clue["quote"])
    return text


def _output_rule() -> str:
    return load_prompt_template("fragment_name_check_output_v1.md", strict=True, tag=RESULTS_TAG).strip()


def render_name_check_prompt(claims: Sequence[Claim], *, pages: str | None = None) -> str:
    """The session's prompt; the paragraph on clues only when a line carries one (`knowledge verify`'s never do).
    ``pages`` given -- the clue pages fetched, rendered, maybe empty -- makes it the local path's first turn: the
    session reads what it is given and lists queries (`QUERIES_TAG`) instead of searching and answering."""

    local = pages is not None
    phrases = _phrases("name_check")
    rows = []
    for claim in claims:
        row = phrases["row"].substitute(claim_id=claim.claim_id, subject=claim.subject, surface=claim.surface,
                                        zh=claim.zh, desc=claim.desc)
        if claim.derived_from:
            row += phrases["derived"].substitute(parent=claim.derived_from)
        row += "".join(_clue_text(clue, phrases) for clue in claim.clues)
        rows.append(row)
    clues = "fragment_name_check_clues_local_v1.md" if local else "fragment_name_check_clues_v1.md"
    clue_rule = load_prompt_template(clues, strict=True) if any(claim.clues for claim in claims) else ""
    output_rule = (load_prompt_template("fragment_name_check_queries_v1.md", strict=True, queries_tag=QUERIES_TAG,
                                        per_line=LOCAL_QUERIES_PER_LINE, results=LOCAL_RESULTS_PER_QUERY).strip()
                   if local else _output_rule())
    return load_prompt_template(
        "kb_name_check_v1.md", strict=True, claims_text="\n".join(rows), clue_rule=clue_rule,
        pages=load_prompt_template("fragment_name_check_pages_v1.md", strict=True, pages=pages) if pages else "",
        search_rule=phrases["search_local" if local else "search_native"].substitute(),
        search_failed_rule="" if local else phrases["search_failed_native"].substitute(), output_rule=output_rule)


def _claim_value(claim: Claim, key: str) -> str:
    return {"source": claim.surface, "zh": claim.zh, "desc": claim.desc}[key]


def _same(key: str, old: str, new: str) -> bool:
    """A name compares `scan_normalize`d (a traditional or full-width spelling of the same name is no change);
    a description compares trimmed."""

    if key == "desc":
        return old.strip() == new.strip()
    return scan_normalize(old) == scan_normalize(new)


def _parse_set(entry: Mapping[str, Any], claim: Claim, verdict: str) -> dict[str, str] | None:
    """The fields the row changes: known keys with a string value differing from the line's (a not_found sets no
    name). None when ``set`` is not an object, a corrected's is missing or names none of `SET_FIELDS`, or a value given is unusable -- not a string, a blank source word or name, a line break anywhere, a
    ``|`` in the source word or the name (the column separator; a description may hold one): the row is no usable
    conclusion, rather than a confirmation of what it meant to replace."""

    if "set" not in entry:
        return None if verdict == "corrected" else {}
    raw = entry["set"]
    if not isinstance(raw, Mapping) or (verdict == "corrected" and not any(key in raw for key in SET_FIELDS)):
        return None
    changes: dict[str, str] = {}
    for key in SET_FIELDS:
        if key not in raw or (key == "zh" and verdict == "not_found"):
            continue
        value = raw[key]
        if not isinstance(value, str):
            return None
        value = value.strip()
        if (not value and key != "desc") or "\n" in value or "\r" in value or (key != "desc" and "|" in value):
            return None
        if not _same(key, _claim_value(claim, key), value):
            changes[key] = value
    return changes


def parse_name_check_answer(text: str, claims: Sequence[Claim]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The session's ``<name_check_results>`` rows joined onto the claims, and ``{claim id: problem}`` for each line
    owed an answer that has none usable (`required_claims`): unanswered (``missing``), a verdict not among `VERDICTS`
    (``verdict``), a confirmed or corrected without a URL (``url``), a ``set`` that is no usable change (``set``).
    A row naming an unknown id, or an id already answered, is ignored. A confirmed changes nothing whatever ``set`` it
    carries; a corrected that changes nothing reads as confirmed. Raises ValueError when there is no single block of
    a JSON array (the reason in words the retry turn can show)."""

    unreadable = _phrases("name_check_retry")
    blocks = find_top_level_tag_blocks(text, RESULTS_TAG)
    if len(blocks) != 1:
        raise ValueError(unreadable["blocks"].substitute(tag=RESULTS_TAG, count=len(blocks)))
    try:
        data = json.loads(blocks[0].strip())
    except json.JSONDecodeError as exc:
        raise ValueError(unreadable["json"].substitute(tag=RESULTS_TAG, line=exc.lineno, column=exc.colno)) from exc
    if not isinstance(data, list):
        raise ValueError(unreadable["array"].substitute(tag=RESULTS_TAG))
    by_id = {claim.claim_id: claim for claim in claims}
    rows: list[dict[str, Any]] = []
    problems: dict[str, str] = {}
    seen: set[str] = set()
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        claim = by_id.get(str(entry.get("id") or ""))
        if claim is None or claim.claim_id in seen:
            continue
        verdict = str(entry.get("verdict") or "")
        url = str(entry.get("url") or "").strip()
        changes = None
        if verdict not in VERDICTS:
            problems[claim.claim_id] = "verdict"
        elif verdict != "not_found" and not _URL_RE.fullmatch(url):
            problems[claim.claim_id] = "url"
        else:
            changes = {} if verdict == "confirmed" else _parse_set(entry, claim, verdict)
            if changes is None:
                problems[claim.claim_id] = "set"
        if changes is None:
            continue
        problems.pop(claim.claim_id, None)
        if verdict == "corrected" and not changes:
            verdict = "confirmed"
        kind = str(entry.get("kind") or "")
        seen.add(claim.claim_id)
        rows.append({
            "claim_id": claim.claim_id,
            "verdict": verdict,
            **({"url": url} if verdict != "not_found" else {}),
            **({"set": changes} if changes else {}),
            **({"kind": kind if kind in (*NOT_FOUND_KINDS, SEARCH_FAILED) else NOT_FOUND_TEMPORARY}
               if verdict == "not_found" else {}),
            "note": str(entry.get("note") or "")[:300],
        })
    for claim_id in required_claims(claims, rows):
        if claim_id not in seen:
            problems.setdefault(claim_id, "missing")
    return rows, {claim_id: problems[claim_id] for claim_id in by_id if claim_id in problems}


#: A not-found the session itself says came from searching that failed (the tool erred, no connection, no search
#: to be had): no conclusion -- nothing is booked, the line stays owed a check and goes again with a later task,
#: where a 暂时 not-found would keep it back 30 days and more on one failed search.
SEARCH_FAILED = "search_failed"


def parse_name_check_results(text: str, claims: Sequence[Claim]) -> list[dict[str, Any]]:
    """The usable rows of an answer (`parse_name_check_answer`), its problems left aside."""

    return parse_name_check_answer(text, claims)[0]


def required_claims(claims: Sequence[Claim], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The claim ids an answer owes a conclusion: every line sent for itself, and a line sent along only when the
    line it follows (``derived_from``) sets a new name -- otherwise it has nothing to follow."""

    renamed = {row["claim_id"] for row in rows if "zh" in (row.get("set") or {})}
    return [claim.claim_id for claim in claims if not claim.along or claim.derived_from in renamed]


@dataclass
class SessionResult:
    rows: list[dict[str, Any]]
    #: the session's replies: on the local path its queries first, then the first answer and each repair's
    answers: list[str]
    #: what is still wrong once the repairs ran out: ``{claim id: problem}``; ``no_material`` for a line the local
    #: path had nothing to read for
    problems: dict[str, str]
    #: the local path's searching (`_LocalOpening.stats`); empty on the native one
    search: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckRoute:
    """How a batch searches: ``native``, the model's own search; ``local``, the harness's -- clue pages fetched
    ahead, then the queries the session lists (`web_search.WebSearchClient`)."""

    kind: str
    search: Any = None
    #: the token counter the local path budgets its material with (None: `token_budget.default_token_counter`)
    count_tokens: Callable[[str], int] | None = None


NATIVE_ROUTE = CheckRoute("native")


def name_check_route(client, *, search: Any = None) -> tuple[CheckRoute | None, str]:  # type: ignore[no-untyped-def]
    """The route a batch takes, or None and why there is none: native search when a ``GENERAL_CAPABLE`` route of
    the research group has it; else the local path, when a route answers at all and a search service holds a key
    (``search``: the client to use, default a `web_search.WebSearchClient` asking `LOCAL_RESULTS_PER_QUERY`
    results). Spends nothing: the same eligibility walk `complete` does, CLI probes included."""

    from ..routing.config import LLMRole

    def refused(native: bool) -> str | None:
        try:
            client.ensure_eligible_target(LLMRole.GENERAL_CAPABLE, native_search=native, task_group="research")
        except RuntimeError as exc:
            return str(exc)
        return None

    if refused(True) is None:
        return NATIVE_ROUTE, ""
    why = refused(False)
    if why is not None:
        return None, why
    if search is None:
        from ..web_search import WebSearchClient

        search = WebSearchClient(execution_settings=getattr(client, "execution_settings", None),
                                 max_results=LOCAL_RESULTS_PER_QUERY)
    if not search.has_provider():
        return None, "没有能自己联网的模型，也没有配好 key 的搜索服务（Exa、Tavily 或 Gemini 免费 key）"
    return CheckRoute("local", search), ""


def parse_check_queries(text: str, claims: Sequence[Claim]) -> tuple[dict[str, list[str]], int]:
    """The local first turn's queries by claim id (``id|query`` lines of `QUERIES_TAG`, at most
    `LOCAL_QUERIES_PER_LINE` a line), and how many lines were dropped: over the cap, an unknown id, no query. No block
    reads as no queries."""

    blocks = find_top_level_tag_blocks(text, QUERIES_TAG)
    ids = {claim.claim_id for claim in claims}
    queries: dict[str, list[str]] = {}
    dropped = 0
    for line in parse_line_items(blocks[0] if blocks else ""):
        claim_id, sep, query = line.replace("｜", "|").partition("|")
        claim_id, query = claim_id.strip(), query.strip()
        mine = queries.setdefault(claim_id, []) if sep and claim_id in ids and query else None
        if mine is None or len(mine) >= LOCAL_QUERIES_PER_LINE:
            dropped += 1
            continue
        if query not in mine:
            mine.append(query)
    return {claim_id: found for claim_id, found in queries.items() if found}, dropped


@dataclass
class _LocalOpening:
    """The local path's conversation up to the turn that asks for conclusions."""

    turns: list[dict[str, str]]
    #: the lines with nothing read -- sent for themselves with no page fetched and no search result shown, or sent
    #: along a line with nothing read: they may not be judged
    exempt: set[str]
    stats: dict[str, Any]


def _owners(owned: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """``{claim id: items}`` turned into ``{item: claim ids}``, in first-seen order."""

    out: dict[str, list[str]] = {}
    for claim_id, items in owned.items():
        for item in items:
            out.setdefault(item, []).append(claim_id)
    return out


def _owned_block(sections: Sequence[tuple[str, str]], owners: Sequence[Sequence[str]], phrase: Any, *,
                 limit: int, count_tokens: Callable[[str], int]) -> tuple[str, list[bool]]:
    """Rendered sections, each headed by the lines it belongs to, within ``limit`` tokens; and, by position, whether
    each section made it into the text (whole or cut short), not dropped for the budget."""

    from ..injection_budget import render_budgeted_block
    from ..routing.config import INJECTION_SECTION_MAX_TOKENS

    headed = [(label, phrase.substitute(ids="、".join(ids)) + "\n" + text)
              for (label, text), ids in zip(sections, owners)]
    # the labels are the requests' own identities (`request_label`), one per request, so they tell what was dropped
    block = render_budgeted_block(headed, count_tokens=count_tokens, section_limit=INJECTION_SECTION_MAX_TOKENS,
                                  block_limit=limit)
    return block.text, [label not in block.dropped for label, _text in headed]


def _local_opening(claims: Sequence[Claim], route: CheckRoute, ask: Callable[[list[dict[str, str]]], str],
                   answers: list[str]) -> _LocalOpening:
    """Fetch the clue pages (`LOCAL_PAGES_PER_LINE` a line), ask for queries, search them; the session's queries
    reply goes into ``answers``."""

    from ..web_search import ExtractRequest, extract_result_sections, search_result_sections

    search = route.search
    if route.count_tokens is not None:
        count_tokens = route.count_tokens
    else:
        from ..token_budget import default_token_counter

        count_tokens = default_token_counter(execution_settings=getattr(search, "execution_settings", None)).count_text
    phrases = _phrases("name_check")
    own = [claim for claim in claims if not claim.along]
    page_owners = _owners({claim.claim_id: list(dict.fromkeys(
        clue["url"] for clue in claim.clues if clue.get("url")))[:LOCAL_PAGES_PER_LINE] for claim in own})
    pages = search.extract_many([ExtractRequest(url=url) for url in page_owners]) if page_owners else []
    # one result per URL asked, in order (the URLs are distinct): paired by position, not by the address a provider
    # may give back rewritten
    owners_of_page = list(page_owners.values())
    pages_text, pages_shown = _owned_block(extract_result_sections(pages, count_tokens=count_tokens), owners_of_page,
                                           phrases["page_owner"], limit=LOCAL_PAGE_TOKENS, count_tokens=count_tokens)
    first = {"role": "user", "content": render_name_check_prompt(claims, pages=pages_text)}
    answers.append(ask([first]))
    queries, dropped = parse_check_queries(answers[-1], claims)
    # one request per query that differs only in case (`search_many` drops the repeat): every line it serves owns it
    query_owners: dict[str, list[str]] = {}
    spelled: dict[str, str] = {}
    for query, ids in _owners(queries).items():
        owners = query_owners.setdefault(spelled.setdefault(query.lower(), query), [])
        owners.extend(claim_id for claim_id in ids if claim_id not in owners)
    results = search.search_many(list(query_owners)) if query_owners else []
    result_owners = [query_owners.get(result.query, ()) for result in results]
    results_text, results_shown = (
        _owned_block(search_result_sections(results, count_tokens=count_tokens), result_owners,
                     phrases["result_owner"], limit=LOCAL_RESULT_TOKENS, count_tokens=count_tokens)
        if results else (phrases["no_queries"].substitute(), []))
    # read: a page or a result that came back and is in the prompt -- one the budget dropped was not read
    read = ({claim_id for page, ids, shown in zip(pages, owners_of_page, pages_shown) if page.ok and shown
             for claim_id in ids}
            | {claim_id for result, ids, shown in zip(results, result_owners, results_shown) if result.ok and shown
               for claim_id in ids})
    exempt = {claim.claim_id for claim in own if claim.claim_id not in read}
    # a line sent along reads its parent's material (it fetches none of its own): with nothing read for the parent,
    # through any chain, it may not be judged either -- unless results came back for it under a query of its own
    while True:
        more = {claim.claim_id for claim in claims if claim.along and claim.claim_id not in exempt
                and claim.claim_id not in read and claim.derived_from in exempt}
        if not more:
            break
        exempt |= more
    ordered = [claim.claim_id for claim in claims if claim.claim_id in exempt]
    follow = load_prompt_template(
        "kb_name_check_local_results_v1.md", strict=True, search_results=results_text, output_rule=_output_rule(),
        no_material=phrases["no_material"].substitute(ids="、".join(ordered)) if ordered else "")
    stats = {
        "pages": len(pages), "pages_failed": sum(not page.ok for page in pages),
        "queries": len(results), "queries_failed": sum(not result.ok for result in results),
        "queries_dropped": dropped, "no_material": ordered,
    }
    return _LocalOpening(turns=[first, {"role": "assistant", "content": answers[-1]},
                                {"role": "user", "content": follow}], exempt=exempt, stats=stats)


def _repair_prompt(problems: Mapping[str, str], claims: Sequence[Claim]) -> str:
    phrases = _phrases("name_check_repair")
    by_id = {claim.claim_id: claim for claim in claims}
    lines = [phrases[problem].substitute(id=claim_id) for claim_id, problem in problems.items() if claim_id in by_id]
    return load_prompt_template("kb_name_check_repair_v1.md", strict=True, problems="\n".join(lines), tag=RESULTS_TAG)


def run_name_check_session(
    claims: Sequence[Claim], *, client, retries: int | None = None,  # type: ignore[no-untyped-def]
    route: CheckRoute = NATIVE_ROUTE,
) -> SessionResult:
    """One conversation over the batch. On the ``native`` route the session searches and answers; on the ``local``
    one it is given the clue pages, lists queries, is given their results, then answers (`_local_opening`) -- and a
    line sent for itself with nothing read (no page fetched, no search result) may not be judged: its answer is
    dropped, it is left in ``problems`` as ``no_material`` (unanswered: a failed search is no not_found to back off
    on) and asked about no further; none left to ask, no answer is asked for. An answer that does not read is asked
    again whole; lines without a usable conclusion are asked again alone, in the same conversation, at most
    ``retries`` times (None: the correction windows' `max_retries_per_window`); what is still wrong then is left in
    ``problems`` (``unreadable`` for every line when no answer read at all) and the rest kept. Raises when an answer
    is cut off or a call fails."""

    from ...settings import MAX_RETRIES_PER_WINDOW
    from ..client import OutputCutOff, reply_cut_off
    from ..routing.config import SESSION_OUTPUT_MAX_TOKENS, LLMRole

    retries = MAX_RETRIES_PER_WINDOW.resolve(retries)

    def ask(messages: list[dict[str, str]]) -> str:
        # the local path's searching is the harness's: the model is asked for none of its own
        result = client.complete(LLMRole.GENERAL_CAPABLE, messages, output_reserve=SESSION_OUTPUT_MAX_TOKENS,
                                 retrieval="native" if route.kind == "native" else "none", task_group="research")
        if reply_cut_off(result):
            raise OutputCutOff(f"name check ({len(claims)} lines)", result.requested_output_tokens)
        return result.content

    answers: list[str] = []
    exempt: set[str] = set()
    stats: dict[str, Any] = {}
    if route.kind == "native":
        opening = [{"role": "user", "content": render_name_check_prompt(claims)}]
    else:
        local = _local_opening(claims, route, ask, answers)
        opening, exempt, stats = local.turns, local.exempt, local.stats

    def owed(rows: Sequence[Mapping[str, Any]]) -> list[str]:
        return [claim_id for claim_id in required_claims(claims, rows) if claim_id not in exempt]

    def parse(text: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
        rows, problems = parse_name_check_answer(text, claims)
        rows = [row for row in rows if row["claim_id"] not in exempt]
        return rows, {claim_id: problems[claim_id] for claim_id in owed(rows) if claim_id in problems}

    def result(rows: list[dict[str, Any]], problems: dict[str, str]) -> SessionResult:
        problems = {**problems, **{claim_id: "no_material" for claim_id in sorted(exempt)}}
        return SessionResult(rows=rows, answers=answers, problems=problems, search=stats)

    if not owed([]):  # nothing read for any line the batch was sent for
        return result([], {})
    answers.append(ask(opening))
    rows: list[dict[str, Any]] = []
    error = ""
    try:
        rows, problems = parse(answers[-1])
    except ValueError as exc:
        problems, error = {}, str(exc)
    for _attempt in range(retries):
        if not problems and not error:
            break
        if error:
            follow = load_prompt_template("kb_name_check_retry_v1.md", strict=True, error=error, tag=RESULTS_TAG)
        else:
            follow = _repair_prompt(problems, claims)
        # the opening, the reply that fell short and what to fix: each re-ask replaces the last one's turns; an empty
        # reply (thoughts only, or blocked) is no turn to send back, the opening goes again
        last = answers[-1]
        answers.append(ask([*opening, {"role": "assistant", "content": last}, {"role": "user", "content": follow}]
                           if last.strip() else opening))
        try:
            more, wrong = parse(answers[-1])
        except ValueError as exc:
            if error:  # still nothing that reads
                error = str(exc)
            continue
        if error or not last.strip():
            rows, error = more, ""  # a whole answer: it replaces whatever there was
        else:
            # the repair answers the lines asked about; a row for a line already answered is not taken
            rows += [row for row in more if row["claim_id"] in problems]
        answered = {row["claim_id"] for row in rows}
        problems = {claim_id: wrong.get(claim_id) or problems.get(claim_id, "missing")
                    for claim_id in owed(rows) if claim_id not in answered}
    if error:  # no answer read at all: every line is without a conclusion
        problems = {claim_id: "unreadable" for claim_id in owed([])}
    return result(rows, problems)


# ---- landing ----------------------------------------------------------------------------------------------------


@dataclass
class LandingReport:
    confirmed: list[dict[str, Any]] = field(default_factory=list)
    #: lines whose correction landed (a not_found's too: it may fix the source word or the description)
    corrected: list[dict[str, Any]] = field(default_factory=list)
    #: a correction that did not land: a 人定 line (``reason`` 人定), one the store refused (a new name then refutes
    #: the old one for good), or one not written this time (the lock held elsewhere, a race: ``retry``, the line
    #: stays owed a check)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    #: lines the session gave no usable conclusion for (unanswered, or a row the parser dropped)
    unanswered: list[dict[str, Any]] = field(default_factory=list)
    #: lines still built on a name this landing replaced (``parent``: that line), left as they were: sent along and
    #: left unanswered, or not brought at all (past `DERIVED_MAX_LINES`, or past the batch size)
    left_behind: list[dict[str, Any]] = field(default_factory=list)
    not_found: list[dict[str, Any]] = field(default_factory=list)
    #: a not-found the session put down to searching that failed (`SEARCH_FAILED`): nothing booked
    search_failed: list[dict[str, Any]] = field(default_factory=list)
    #: the line changed since the session read it: nothing booked
    stale: list[dict[str, Any]] = field(default_factory=list)
    rev: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()}


def _land_line(store: KnowledgeStore, claim: Claim, node: NodeVersion, changes: Mapping[str, str], *, task_id: str,
               read_rev: int, report: LandingReport) -> tuple[str, str]:
    """One line's harness revision: ``("landed", "")``, ``("final", why)`` (the validator refused the value) or
    ``("retry", why)`` (not written this time: a concurrent write rolled it back, or committed first so the update
    was dropped as stale). ``reject`` with one op writes nothing either way; it only tells a refusal from a race."""

    from .node.apply import apply_envelope
    from .node.envelope import Binding, Envelope

    # One envelope per line (the maintenance callers' rule, docs/knowledge.md「整批校验不过时的拒收」): a line the
    # store refuses does not hold the others back.
    result = apply_envelope(store, Envelope(
        # the rev the landing read the line at: a write since (a retirement too) is a stale update, not a refusal
        task_id=task_id, assignment_id="name-check", context_epoch=0, knowledge_read_rev=read_rev,
        ops=[{"op": "update", "id": "@n", "set": {SET_FIELDS[key]: value for key, value in changes.items()}}],
        handle_bindings=[Binding("@n", "node", claim.node_id, node.valid_from_rev)],
    ), revision_kind="harness", note=NAME_CHECK_REVISION_NOTE, invalid_ops="reject")
    if result.rejected_ops and not result.rolled_back:
        return "final", "; ".join(why for _, why in result.rejected_ops)
    if result.rolled_back or result.conflicts or result.rev is None:
        return "retry", result.rollback_reason or "; ".join(c.reason for c in result.conflicts) or "没写进去"
    report.rev = result.rev
    return "landed", ""


def _line_now(node: NodeVersion | None) -> dict[str, str]:
    if node is None:
        return {key: "" for key in SET_FIELDS}
    return {"source": str(node.payload.get("surface") or ""), "zh": zh_name(node) or "",
            "desc": str(node.payload.get("desc") or "")}


def book_name_checks(
    store: KnowledgeStore,
    rows: Sequence[Mapping[str, Any]],
    claims: Sequence[Claim],
    *,
    task_id: str,
    write: bool = True,
    hold: str | None = None,
) -> LandingReport:
    """Land the session's rows: evidence on ``payload.zh`` and the corrections, a harness revision per line that
    sets only `SET_FIELDS`. ``write`` False (the write lock is held elsewhere) or a ``hold`` reason (``rendered/`` has
    edits not taken in yet: a later harvest would read the corrected lines back as a person's edits) lands no revision:
    a row with a correction goes to ``blocked`` as ``retry`` and books nothing (it comes back whole), the other rows
    book their evidence.

    A replaced name is refuted when the correction lands, or cannot land for good (人定, refused by the validator);
    a correction not written this time books nothing on the old name, so the line stays owed a check
    (`name_trust.check_due` takes a refutation as final). The name a correction writes gets no confirmation.

    Replays book nothing new: evidence rows dedupe on their key, and a line already carrying every value a
    correction sets is not written again (a line that changed otherwise since the session read it is left alone)."""

    by_id = {claim.claim_id: claim for claim in claims}
    report = LandingReport()
    read_rev = store.current_rev()
    nodes = {claim.node_id: store.node(claim.node_id, read_rev) for claim in claims}
    statuses = zh_statuses(store, [node for node in nodes.values() if node is not None])
    edits: list[tuple[dict[str, Any], Claim]] = []
    answered = {row["claim_id"] for row in rows}
    owed = set(required_claims(claims, rows))  # a line sent along whose parent kept its name is owed nothing
    unanswered = [claim for claim in claims if claim.claim_id in owed and claim.claim_id not in answered]

    def book(node_id: str, name: str, verdict: str, *, source: str | None = None, span: str | None = None) -> None:
        record_evidence(store, node_id=node_id, field_path=ZH_FIELD, value_hash=digest(name), verdict=verdict,
                        evidence_kind="external", task_id=task_id, source_ref=source, span=span,
                        algo_version=NAME_CHECK_ALGO_VERSION)

    def book_row(row: Mapping[str, Any], claim: Claim, node: NodeVersion, item: dict[str, Any]) -> None:
        """The row's own evidence on the name it read: a not-found, or a confirmation when it sets no new name."""

        if row["verdict"] == "not_found":
            book(claim.node_id, claim.zh, "unverifiable", span=row["kind"])
            report.not_found.append(item)
        elif "zh" not in (row.get("set") or {}):
            book(claim.node_id, claim.zh, "confirmed", source=row["url"])
            if row["verdict"] == "confirmed":
                # the whole line stood the check: what `knowledge.report` reads for a line as a whole
                record_evidence(store, node_id=claim.node_id, field_path="payload:core",
                                value_hash=payload_group_hash("term", node.payload, "core"), verdict="confirmed",
                                evidence_kind="external", task_id=task_id, source_ref=row["url"],
                                algo_version=NAME_CHECK_ALGO_VERSION)
                report.confirmed.append(item)

    for row in rows:
        claim = by_id[row["claim_id"]]
        item = {**row, "line": f"{claim.surface}|{claim.zh}", "subject": claim.subject}
        if row["verdict"] == "not_found" and row.get("kind") == SEARCH_FAILED:
            report.search_failed.append(item)  # nothing was read: no evidence, and no change it might carry
            continue
        changes: Mapping[str, str] = row.get("set") or {}
        node = nodes.get(claim.node_id)
        now = _line_now(node)
        if node is None:  # retired since the session read it
            report.stale.append(item)
            continue
        if changes and all(now[key] == changes.get(key, _claim_value(claim, key)) for key in SET_FIELDS):
            # a replay after the correction landed: no revision left to write, and the evidence it may not have
            # reached is booked now (rows dedupe on their key, so a full replay books nothing new)
            book_row(row, claim, node, item)
            if "zh" in changes:
                book(claim.node_id, claim.zh, "refuted", source=row.get("url"))
            report.corrected.append(item)
            continue
        if any(now[key] != _claim_value(claim, key) for key in SET_FIELDS):
            report.stale.append(item)
            continue
        status: ZhStatus | None = statuses.get(claim.node_id)
        if not changes:
            book_row(row, claim, node, item)
            continue
        if status is not None and status.state == HUMAN:
            book_row(row, claim, node, item)
            if "zh" in changes:
                book(claim.node_id, claim.zh, "refuted", source=row["url"])
            report.blocked.append({**item, "reason": HUMAN})
            continue
        edits.append((item, claim))

    # A row with a correction books its evidence only once the correction's fate is settled: a correction left to
    # retry books nothing (a 根本 not-found booked ahead of it would end the line's scans before it ever lands).
    for item, claim in edits:
        changes = item["set"]
        node = nodes[claim.node_id]
        outcome, reason = "retry", hold or "知识库正被别的进程写"
        if write and hold is None:
            outcome, reason = _land_line(store, claim, node, changes, task_id=task_id, read_rev=read_rev, report=report)
        if outcome == "retry":
            report.blocked.append({**item, "reason": reason, "retry": True})
            continue
        book_row(item, claim, node, item)
        if "zh" in changes:
            book(claim.node_id, claim.zh, "refuted", source=item["url"])
        if outcome == "landed":
            report.corrected.append(item)
        else:
            report.blocked.append({**item, "reason": reason})
    renamed = [by_id[item["claim_id"]] for item in report.corrected if "zh" in (item.get("set") or {})]
    renamed_ids = {claim.claim_id for claim in renamed}
    report.unanswered = [{"line": f"{claim.surface}|{claim.zh}", "subject": claim.subject} for claim in unanswered
                         if not (claim.along and claim.derived_from in renamed_ids)]
    # answered, or owed for itself (unanswered, it stays due): only the rest can be left on the old name. A search
    # that failed is no answer: the line it was about is as unanswered as one the session skipped.
    concluded = {row["claim_id"] for row in rows if not (row["verdict"] == "not_found" and row.get("kind") == SEARCH_FAILED)}
    settled = {claim.node_id for claim in claims if claim.claim_id in concluded or not claim.along}
    report.left_behind = _left_behind(store, renamed, settled=settled)
    return report


def _left_behind(store: KnowledgeStore, renamed: Sequence[Claim], *, settled: Collection[str]) -> list[dict[str, Any]]:
    """The lines still built on a name the landing replaced: of the parent's entry, a name containing the old one
    (as `build_claims` brings them: at least `DERIVED_MIN_CHARS`), not ``settled``. A 已核 one is not due again and
    keeps that name: the person is told. Each line under the longest replaced name in its own."""

    if not renamed:
        return []
    at = store.current_rev()
    subject_of = _subjects_of(store, at)
    rows: list[dict[str, Any]] = []
    taken = set(settled) | {claim.node_id for claim in renamed}
    lines = _checkable(store, at)
    for parent in sorted(renamed, key=lambda claim: -len(claim.zh)):
        if len(parent.zh) < DERIVED_MIN_CHARS:
            continue
        for node in lines:
            name = zh_name(node) or ""
            if node.local_id in taken or subject_of.get(node.local_id) != parent.subject or parent.zh not in name:
                continue
            taken.add(node.local_id)
            rows.append({"line": f"{node.payload.get('surface') or ''}|{name}", "subject": parent.subject,
                         "parent": f"{parent.surface}|{parent.zh}"})
    return rows


def pending_edits_hold(repo: Any) -> str | None:
    """Why corrections wait this time, when ``rendered/`` holds edits the harvest has not taken in (it skipped them:
    the lock was busy, a file did not parse): landing now, the next harvest would read the corrected lines of those
    files back to their old text as a person's edits. None when there are none."""

    dirty = sorted(rel for rel, *_ in repo.rendered_dirty_files())
    return f"rendered/ 里有没收进库的编辑（{'、'.join(dirty[:3])}{' 等' if len(dirty) > 3 else ''}）" if dirty else None


def change_text(row: Mapping[str, Any]) -> str:
    """What a correction sets, for a person: ``原文 A → B；定名 X → Y；描述改为「…」``."""

    old = dict(zip(("source", "zh"), str(row.get("line") or "").split("|", 1)))
    parts = []
    for key, value in (row.get("set") or {}).items():
        if key == "desc":
            parts.append(f"描述改为「{value}」" if value else "描述清空")
        else:
            parts.append(f"{FIELD_LABELS[key]} {old.get(key, '')} → {value}")
    return "；".join(parts)


def landing_lines(report: Mapping[str, Any]) -> list[str]:
    """The landing as lines for a person (the task report, `knowledge verify`'s output)."""

    def where(row: Mapping[str, Any]) -> str:
        return f"「{row.get('subject', '')}」{row.get('line', '')}"

    def source(row: Mapping[str, Any]) -> str:
        return f"（{row['url']}）" if row.get("url") else ""

    lines: list[str] = []
    lines += [f"印证：{where(row)}{source(row)}" for row in report.get("confirmed", ())]
    lines += [f"修正：{where(row)}：{change_text(row)}{source(row)}" + (f"——{row['note']}" if row.get("note") else "")
              for row in report.get("corrected", ())]
    lines += [f"没改（{row.get('reason', '')}）：{where(row)}：核查要{change_text(row)}{source(row)}，"
              + ("下次核查会再来" if row.get("retry") else "要改请自己改库")
              for row in report.get("blocked", ())]
    lines += [f"查不到（{'根本' if row.get('kind') == 'permanent' else '暂时'}）：{where(row)}"
              + (f"——{row['note']}" if row.get("note") else "") for row in report.get("not_found", ())]
    lines += [f"搜索失败、没记：{where(row)}" + (f"——{row['note']}" if row.get("note") else "")
              for row in report.get("search_failed", ())]
    lines += [f"行已变、没记：{where(row)}" for row in report.get("stale", ())]
    unanswered = report.get("unanswered") or ()
    if unanswered:
        lines.append(f"没有可用结论（没答或答得不合格）{len(unanswered)} 行：" + "、".join(where(row) for row in unanswered))
    lines += [f"没跟着改：{where(row)}：它所随的 {row['parent']} 改了名，这一行这次没改，要改请自己改库"
              for row in report.get("left_behind") or ()]
    return lines

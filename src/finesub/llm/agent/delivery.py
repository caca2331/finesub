"""Did each harness reply reach the model? (docs/llm_agent_tool_protocol.md §6)

The sent side is the MCP server's frame log: every reply leaves through
`channels.fit_reply`, which gives it a reply id, and the log holds the reply
text as sent. The received side is a record the driver already keeps: the
model's own context where the vendor writes one (Claude Code and CodeBuddy's
stream-json, agy's brain transcript, a Codex rollout, dsh's session
transcript), or only the CLI's copy (Codex's ``--json`` stream). Nothing here
parses a vendor format: every string in a record is searched for reply ids,
so a record whose layout changes reads as "not found", never as an error.

Delivered means: the id is there and what carries it is not shorter than what
was sent -- a cut that keeps head and tail is still shorter. A spilled reply
is delivered only once its parts were read too: every `read_spill` part
delivered, or every part file opened. Files a driver names to the model
outright (agy's task parts) are checked the same way. "Opened" means a file
tool was called on the path; its result is not looked at (that would read a
vendor format), and parts are cut so one call reads one whole. Only a ``model``-level
record can say something did not arrive; ``cli`` and ``unknown`` are recorded
and never warned about.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from finesub import jsonl
#: A reply id (`channels.reply_id_for`).
_REPLY_ID = re.compile(r"r-[0-9a-f]{12}")
#: The text an `isError` reply starts with.
_ERROR_ID = re.compile(r"\[reply (r-[0-9a-f]{12})\]")
#: A record this large is not read (an unknown, not an error).
MAX_RECORD_BYTES = 64 * 1024 * 1024
#: A spill as a stub names it (`channels._stub`): its id, and its part count.
_SPILL_ID = re.compile(r'spill_id "([^"]+)"')
_SPILL_PARTS = re.compile(r"in (?:the )?(\d+) (?:parts|files)")
#: Codex's mark on an `exec` output it cut (code mode).
_TRUNCATED = "truncated output"
#: How Codex names a tool call made from inside code mode's JS `exec`
#: (`code_mode/delegate.rs`); a direct call is ``call_<hex>``.
_EXEC_CALL_PREFIX = "exec-"
_DIRECT_CALL_PREFIX = "call_"


@dataclass(frozen=True)
class SentReply:
    reply_id: str
    tool: str
    #: UTF-8 bytes of the reply text as sent.
    bytes: int
    #: Where the call came from: "direct", "exec" (Codex code mode) or "unknown".
    origin: str
    #: A `submit` the runtime accepted: exit-on-accept may hold it back by design.
    accepted: bool
    #: A stub's spill: (spill id, part count, part files or () for `read_spill`).
    spill: tuple[str, int, tuple[str, ...]] | None = None
    #: A `read_spill` reply: (spill id, part).
    reads_part: tuple[str, int] | None = None


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _reply_id(text: str) -> str:
    match = _ERROR_ID.match(text)
    if match:
        return match.group(1)
    try:
        body = json.loads(text)
    except ValueError:
        return ""
    reply_id = body.get("reply_id") if isinstance(body, Mapping) else None
    return reply_id if isinstance(reply_id, str) and _REPLY_ID.fullmatch(reply_id) else ""


def _origin(call_id: Any) -> str:
    call_id = str(call_id or "")
    if call_id.startswith(_EXEC_CALL_PREFIX):
        return "exec"
    if call_id.startswith(_DIRECT_CALL_PREFIX):
        return "direct"
    return "unknown"


def sent_replies(log_path: Path, session_id: str) -> list[SentReply]:
    """The tool replies one server session sent, in order, from its frame log."""

    try:
        rows = list(jsonl.iter_records(log_path))
    except OSError:
        return []
    calls: dict[Any, tuple[str, str]] = {}
    sent: list[SentReply] = []
    for row in rows:
        try:
            frame = json.loads(row["frame"])
        except (ValueError, KeyError, TypeError):
            continue
        if not isinstance(frame, Mapping) or row.get("session") != session_id:
            continue
        if row.get("dir") == "in":
            params = frame.get("params")
            if frame.get("method") == "tools/call" and isinstance(params, Mapping):
                meta = params.get("_meta") if isinstance(params.get("_meta"), Mapping) else {}
                arguments = params.get("arguments") if isinstance(params.get("arguments"), Mapping) else {}
                calls[frame.get("id")] = (
                    str(params.get("name") or ""), _origin(meta.get("callId")), arguments
                )
            continue
        if frame.get("id") not in calls:
            continue
        tool, origin, arguments = calls.pop(frame.get("id"))
        result = frame.get("result")
        content = result.get("content") if isinstance(result, Mapping) else None
        text = "".join(
            str(item.get("text") or "") for item in content or () if isinstance(item, Mapping)
        )
        reply_id = _reply_id(text)
        if not reply_id:
            continue
        accepted = False
        if tool == "submit" and not result.get("isError"):
            try:
                accepted = json.loads(text).get("status") == "accepted"
            except (ValueError, AttributeError):
                accepted = False
        reads_part = None
        # An error reply (a lapsed hold, a missing part) carries no part.
        if tool == "read_spill" and not result.get("isError"):
            try:
                reads_part = (str(arguments.get("spill_id") or ""), int(arguments.get("part")))
            except (TypeError, ValueError):
                reads_part = None
        sent.append(SentReply(
            reply_id, tool, _utf8_len(text), origin, accepted, _spill(text), reads_part
        ))
    return sent


def _spill(text: str) -> tuple[str, int, tuple[str, ...]] | None:
    """What a stub says about its spill: id, part count and the part files."""

    try:
        body = json.loads(text)
    except ValueError:
        body = None
    if isinstance(body, Mapping) and isinstance(body.get("spilled"), Mapping):
        spilled = body["spilled"]
        spill_id, count, listing = str(spilled.get("spill_id") or ""), int(spilled.get("parts") or 0), str(
            spilled.get("files") or ""
        )
    elif _ERROR_ID.match(text):  # an `isError` stub: the note, then the listing
        found_id, found_count = _SPILL_ID.search(text), _SPILL_PARTS.search(text)
        if not found_count:
            return None
        spill_id, count = (found_id.group(1) if found_id else ""), int(found_count.group(1))
        listing = text.split("\n", 1)[1] if "\n" in text else ""
    else:
        return None
    return spill_id, count, part_files(listing, count)


def part_files(listing: str, count: int) -> tuple[str, ...]:
    """The part paths a listing names (`channels.part_list`): one a line, or a
    directory with a range."""

    lines = [line.strip() for line in listing.splitlines() if line.strip()]
    if len(lines) == 1 and " to " in lines[0]:
        first = lines[0].split(" to ", 1)[0]
        directory = first.rsplit("\\", 1)[0] if "\\" in first else first.rsplit("/", 1)[0]
        separator = "\\" if "\\" in first else "/"
        return tuple(f"{directory}{separator}part-{index:02d}.md" for index in range(1, count + 1))
    return tuple(lines)


def _path_key(text: str) -> str:
    return text.replace("\\\\", "/").replace("\\", "/").lower()


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _carried_bytes(text: str, reply_id: str, depth: int = 0) -> int:
    """How much of reply ``reply_id`` a string that names it carries.

    A string that is (or wraps) the reply's JSON is measured as the server
    wrote it, so a CLI that re-serialises it is not shorter for that; any
    other string -- an error text, a reply under a header the CLI adds, or a
    JSON reply cut in the middle -- by its own UTF-8 bytes.
    """

    try:
        value = json.loads(text)
    except ValueError:
        return 0 if _TRUNCATED in text else _utf8_len(text)
    if isinstance(value, Mapping) and value.get("reply_id") == reply_id:
        return _utf8_len(json.dumps(value, ensure_ascii=False))
    if depth >= 3:
        return _utf8_len(text)
    inner = [item for item in _strings(value) if reply_id in item]
    return max((_carried_bytes(item, reply_id, depth + 1) for item in inner), default=0)


def received_bytes(
    paths: Iterable[Path], wanted: Sequence[str], files: Sequence[str] = ()
) -> tuple[dict[str, int], set[str], set[str]] | None:
    """Per reply id in ``wanted``, the most any record string carries of it;
    the ids that only came inside a string carrying other replies too (whose
    share cannot be told apart); and the ``files`` opened -- a record string
    that is exactly the path, as a file tool's argument is (the prompt that
    lists the paths holds them inside a longer string). None when no record
    could be read."""

    wanted_set = set(wanted)
    found: dict[str, int] = {}
    shared: set[str] = set()
    file_keys = {_path_key(path): path for path in files}
    named: set[str] = set()
    read_any = False
    for path in paths:
        try:
            if Path(path).stat().st_size > MAX_RECORD_BYTES:
                continue
            lines = jsonl.lines(Path(path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        read_any = True
        for line in lines:
            if "r-" not in line and not file_keys:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = line
            for text in _strings(row):
                if file_keys and _path_key(text) in file_keys:
                    named.add(file_keys[_path_key(text)])
                if "r-" not in text:
                    continue
                ids = set(_REPLY_ID.findall(text)) & wanted_set
                # A cut output names its replies as not carried, all of them
                # (`_carried_bytes`); otherwise several replies in one text
                # cannot be told apart.
                if len(ids) > 1 and not _is_json(text) and _TRUNCATED not in text:
                    shared |= ids
                    continue
                for reply_id in ids:
                    found[reply_id] = max(found.get(reply_id, 0), _carried_bytes(text, reply_id))
    return (found, shared - set(found), named) if read_any else None


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def check(
    sent: Sequence[SentReply],
    records: Sequence[tuple[Path, str]],
    *,
    named_files: Sequence[str] = (),
    ended_on_accept: bool = False,
) -> dict[str, Any]:
    """The `delivery` block of an execution attempt.

    ``records`` are (path, evidence level), most direct first; the first
    level whose records could be read is the one judged by. ``named_files``
    are files the driver named to the model outright.
    """

    spill_files = [path for reply in sent if reply.spill for path in reply.spill[2]]
    replies: list[dict[str, Any]] = []
    level, received = "unknown", None
    for candidate in ("model", "cli"):
        paths = [path for path, path_level in records if path_level == candidate]
        received = (
            received_bytes(paths, [reply.reply_id for reply in sent], [*named_files, *spill_files])
            if paths
            else None
        )
        if received is not None:
            level = candidate
            break
    found, shared, opened = received or ({}, set(), set())
    delivered_parts = set()
    for reply in sent:
        row: dict[str, Any] = {
            "reply_id": reply.reply_id, "tool": reply.tool, "sent_bytes": reply.bytes,
            "origin": reply.origin,
        }
        got = found.get(reply.reply_id)
        if got is not None:
            row["received_bytes"] = got
        if level == "unknown" or reply.reply_id in shared:
            row["status"] = "unknown"
        elif got is not None and got >= reply.bytes:
            row["status"] = "delivered"
            if reply.reads_part:
                delivered_parts.add(reply.reads_part)
        elif reply.accepted and ended_on_accept and got is None:
            # exit-on-accept: the driver ended the CLI while the server held it.
            row["status"] = "held"
        elif level == "model":
            row["status"] = "not_delivered"
        else:
            row["status"] = "unconfirmed"
        replies.append(row)
    for reply, row in zip(sent, replies):
        if not reply.spill or row["status"] != "delivered":
            continue
        spill_id, count, files = reply.spill
        read = (
            sum(path in opened for path in files)
            if files
            else sum((spill_id, part) in delivered_parts for part in range(1, count + 1))
        )
        row["parts_read"] = f"{read}/{count}"
        if read < count and level == "model":
            row["status"] = "parts_unread"
    booked: dict[str, Any] = {
        "evidence": level,
        "replies": replies,
        "not_delivered": sum(row["status"] in ("not_delivered", "parts_unread") for row in replies),
        "exec_calls": sum(reply.origin == "exec" for reply in sent),
    }
    if named_files:
        booked["named_files"] = {"named": len(named_files), "opened": len(set(named_files) & opened)}
        booked["files_unopened"] = (
            len(set(named_files) - opened) if level == "model" else 0
        )
    return booked

"""What each agent CLI can carry to its model, and how a tool reply is fitted.

docs/llm_agent_tool_protocol.md §6 "通道上限". Every driver config holds one
`ChannelLimits`: the limits left once the driver has widened what its CLI lets
it widen (`tool_output_token_limit`, `MAX_MCP_OUTPUT_TOKENS`, the dsh spill
policy), as data. The harness MCP server fits every reply to them in one
place (`fit_reply`): a reply that fits goes out whole, one that does not is
written into parts the model reads -- with the CLI's own file tool
(``overflow="files"``, agy) or with the harness's `read_spill` tool
(``overflow="tool"``). Nothing here depends on which tool produced the reply.

Imported by the MCP server process, so it stays light: the standard library
and the prompt template directory, nothing that pulls the LLM stack in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import re
from string import Template
from typing import Any, Callable, Iterable, Mapping, Sequence

#: The server environment key carrying `ChannelLimits` as JSON.
CHANNEL_ENV = "FINESUB_MCP_CHANNEL"
#: Dev switch: ``"<ok>,<error>"`` lowers both reply limits (bytes), so tests
#: and acceptance runs can force a spill without a production-sized reply.
#: Below about 1 KB a ``read_spill`` reply (half the limit plus its wrapper)
#: spills again; production limits are 10 KB and up.
REPLY_LIMIT_ENV = "FINESUB_MCP_REPLY_LIMIT"

#: Fields a spilled reply keeps inline: what the session acts on next. The
#: rest of the reply goes into the parts.
CONTROL_FIELDS = (
    "reply_id",
    "status",
    "task_id",
    "message",
    "repair_rounds_remaining",
    "owed_blocks",
    "required_blocks",
    "knowledge_read_rev",
    "result_digest",
)
SPILL_DIRNAME = "spill"
_SPILL_ID = re.compile(r"^[A-Za-z0-9-]{1,80}$")
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "prompt_templates"
# Room a stub or a `read_spill` reply needs around its part: field names,
# ids, the hint sentence, the CallToolResult frame.
_REPLY_FRAME_BYTES = 2_000


@dataclass(frozen=True)
class ChannelLimits:
    """One driver's channel to its model, after the driver widened it.

    ``reply_ok_bytes`` / ``reply_error_bytes``: the largest successful and
    `isError` reply that reaches the model whole (0 = not measured, treated
    as unlimited). Measured as ``measure`` says: "text" is the UTF-8 bytes of
    the reply text; "json" the CallToolResult as compact JSON (Codex direct
    calls, which is never smaller than the text). ``overflow`` is where a
    larger reply goes. ``part_bytes`` / ``part_lines`` bound one part a file
    tool reads whole (``overflow="files"``). ``opening_max_chars`` bounds the
    whole first message, in ``opening_unit`` (agy's sits on argv and counts
    UTF-16 units of the command line). ``batch_tool`` exposes `batch` (a CLI
    whose model cannot call tools in parallel); ``result_size_meta`` declares
    `_meta["anthropic/maxResultSizeChars"]` on every tool (Claude Code and its
    fork CodeBuddy only).
    """

    reply_ok_bytes: int = 0
    reply_error_bytes: int = 0
    overflow: str = "tool"
    measure: str = "text"
    part_bytes: int = 40_000
    part_lines: int = 700
    opening_max_chars: int = 150_000
    opening_unit: str = "chars"
    batch_tool: bool = False
    result_size_meta: int = 0

    def to_env(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_env(cls, text: str) -> "ChannelLimits":
        """The limits a server was started with; unlimited when absent."""

        try:
            raw = json.loads(text) if text else {}
        except ValueError:
            raw = {}
        known = {field.name for field in fields(cls)}
        limits = cls(**{key: value for key, value in dict(raw).items() if key in known})
        override = os.environ.get(REPLY_LIMIT_ENV, "")
        if override:
            ok, _, error = override.partition(",")
            try:
                limits = cls(**{
                    **asdict(limits),
                    "reply_ok_bytes": int(ok),
                    "reply_error_bytes": int(error or ok),
                })
            except ValueError:
                pass
        return limits

    @property
    def reads_spills_by_tool(self) -> bool:
        """Whether a reply can spill into parts read with `read_spill`."""

        return self.overflow == "tool" and bool(self.reply_ok_bytes or self.reply_error_bytes)

    @property
    def spill_part_budget(self) -> int:
        """How large one part may be, in the unit the reader is limited by."""

        if self.overflow == "files":
            return self.part_bytes
        # A part goes back as a successful `read_spill` reply, whichever
        # reply it was spilled from.
        limit = self.reply_ok_bytes or 10**9
        return max(limit - _REPLY_FRAME_BYTES, limit // 2)


def channel_limits(driver: Any) -> ChannelLimits:
    """The driver's `ChannelLimits`; unlimited for a driver that declares none."""

    limits = getattr(getattr(driver, "config", None), "channel", None)
    return limits if isinstance(limits, ChannelLimits) else ChannelLimits()


def server_channel(driver: Any, root: Path) -> tuple[dict[str, str], list[str], list[str]]:
    """What the MCP server of one invocation needs from the driver's channel.

    Returns the server environment, the roots the CLI's file tool may read
    (the assignment root, where spilled parts land, for ``overflow="files"``)
    and the extra tools the server exposes (`read_spill`, `batch`). The same
    for every window of a driver, so the tool table -- part of the cached
    prefix -- does not move.
    """

    limits = channel_limits(driver)
    env = {CHANNEL_ENV: limits.to_env()}
    if os.environ.get(REPLY_LIMIT_ENV):
        env[REPLY_LIMIT_ENV] = os.environ[REPLY_LIMIT_ENV]
    view_roots = [str(Path(root).resolve())] if limits.overflow == "files" else []
    return env, view_roots, extra_tool_names(limits)


def extra_tool_names(limits: ChannelLimits) -> list[str]:
    names = []
    if limits.reads_spills_by_tool:
        names.append("read_spill")
    if limits.batch_tool:
        names.append("batch")
    return names


# -- measuring -----------------------------------------------------------


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def reply_size(result: Mapping[str, Any], measure: str) -> int:
    """A CallToolResult's size where ``measure`` says the CLI's limit bites."""

    if measure == "json":
        return _utf8_len(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return sum(
        _utf8_len(str(item.get("text") or ""))
        for item in result.get("content") or []
        if isinstance(item, Mapping)
    )


def _escaped_len(text: str, times: int) -> int:
    """UTF-8 bytes of ``text`` once it is a JSON string ``times`` times over."""

    for _ in range(times):
        text = json.dumps(text, ensure_ascii=False)[1:-1]
    return _utf8_len(text)


def part_cost(limits: ChannelLimits) -> Callable[[str], int]:
    """What one line of a part costs against `spill_part_budget`.

    A file part is read raw. A part read through `read_spill` travels as a
    string inside the JSON reply text -- escaped once, and once more where the
    CLI measures the whole result as JSON -- so its escapes count.
    """

    if limits.overflow == "files":
        return _utf8_len
    times = 2 if limits.measure == "json" else 1
    return lambda line: _escaped_len(line, times)


# -- parts -----------------------------------------------------------------


def split_parts(
    text: str, *, budget: int, max_lines: int = 0, cost: Callable[[str], int] = _utf8_len
) -> list[str]:
    """``text`` cut into parts of at most ``budget`` (by ``cost``) and
    ``max_lines`` lines, on line breaks; a line longer than a part is cut on
    character boundaries. Joined back, the parts are ``text``."""

    parts: list[str] = []
    current: list[str] = []
    size = 0

    def close() -> None:
        nonlocal current, size
        if current:
            parts.append("".join(current))
        current, size = [], 0

    for line in text.splitlines(keepends=True):
        line_cost = cost(line)
        if line_cost > budget:
            close()
            # Each character's cost stands alone (UTF-8 bytes, JSON escapes),
            # so the line is cut in one pass.
            start, chunk_cost, costs = 0, 0, {}
            for index, char in enumerate(line):
                char_cost = costs.get(char)
                if char_cost is None:
                    char_cost = costs[char] = cost(char)
                if index > start and chunk_cost + char_cost > budget:
                    parts.append(line[start:index])
                    start, chunk_cost = index, 0
                chunk_cost += char_cost
            current, size = [line[start:]], chunk_cost
            continue
        if current and (size + line_cost > budget or (max_lines and len(current) >= max_lines)):
            close()
        current.append(line)
        size += line_cost
    close()
    return parts or [""]


def part_name(index: int) -> str:
    """The file name of part ``index`` (1-based): ``part-01.md`` and on."""

    return f"part-{index:02d}.md"


def write_parts(directory: Path, parts: Sequence[str]) -> list[tuple[str, int]]:
    """Write ``parts`` as ``part-01.md`` ... into ``directory``; (name, bytes)."""

    directory.mkdir(parents=True, exist_ok=True)
    names = {part_name(index) for index in range(1, len(parts) + 1)}
    for stale in directory.glob("part-*.md"):
        if stale.name not in names:
            stale.unlink()
    written = []
    for index, text in enumerate(parts, start=1):
        name = part_name(index)
        (directory / name).write_text(text, encoding="utf-8", newline="")
        written.append((name, _utf8_len(text)))
    return written


def write_document_parts(directory: Path, document: str, limits: ChannelLimits) -> list[str]:
    """``document`` written into ``directory`` as parts ``limits`` carries
    whole -- one file-tool read each for ``overflow="files"``, one
    `read_spill` reply each otherwise; returns the part names, in order. The
    one writer for spilled replies and agy's task bodies, so a text is cut the
    same way wherever it travels."""

    parts = split_parts(
        document,
        budget=limits.spill_part_budget,
        max_lines=limits.part_lines if limits.overflow == "files" else 0,
        cost=part_cost(limits),
    )
    return [name for name, _bytes in write_parts(directory, parts)]


def part_list(directory: Path, names: Sequence[str], *, max_units: int = 0, unit: str = "chars") -> str:
    """How parts are named to the model: one absolute path a line, or -- when
    that would pass ``max_units`` -- the directory once and the names as a
    range (``part-01.md`` to ``part-NN.md``), which the model can still open in
    one step."""

    directory = Path(directory).resolve()
    full = "\n".join(str(directory / name) for name in names)
    if not max_units or text_units(full, unit) <= max_units or len(names) <= 1:
        return full
    return f"{directory}{os.sep}{names[0]} to {names[-1]} ({len(names)} files, numbered in order)"


def text_units(text: str, unit: str) -> int:
    """``text``'s length in ``unit``: characters, or UTF-16 code units (what
    a Windows command line is limited by)."""

    if unit == "utf16":
        return len(text.encode("utf-16-le")) // 2
    return len(text)


# -- the generic renderer ----------------------------------------------------


def render_spill(payload: Mapping[str, Any], *, skip: Iterable[str] = ()) -> str:
    """A reply's fields as one document a model reads straight through.

    No rule per tool: every field gets a heading line with its path, type
    and size; text is written raw (no JSON escapes), a list one item a line as
    compact JSON, an object field by field. Nothing is dropped, and the
    heading's character count marks where a text ends.
    """

    skipped = set(skip)
    out: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            out.append(f"@@ {path} · object · {len(value)} fields\n")
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, str):
            out.append(f"@@ {path} · text · {len(value)} chars\n")
            out.append(value if value.endswith("\n") else value + "\n")
        elif isinstance(value, (list, tuple)):
            out.append(f"@@ {path} · list · {len(value)} items\n")
            for item in value:
                out.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            out.append(f"@@ {path} · {json.dumps(value, ensure_ascii=False)}\n")

    for key, value in payload.items():
        if key in skipped:
            continue
        walk(value, str(key))
    return "".join(out)


# -- fitting a reply ---------------------------------------------------------


def fragment(name: str, **values: object) -> str:
    """A prompt fragment from `prompt_templates/`, with ``values`` filled in."""

    text = (_TEMPLATE_DIR / name).read_text(encoding="utf-8")
    return Template(text).safe_substitute({key: str(value) for key, value in values.items()}).strip()


def _result(text: str, *, error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": bool(error)}


def reply_id_for(request_id: str) -> str:
    """The short id a reply carries, so a CLI's record of what the model saw
    can be matched to what the server sent (`delivery_check`)."""

    return "r-" + hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:12]


def fit_reply(
    *,
    payload: Mapping[str, Any] | None,
    error: str = "",
    reply_id: str,
    limits: ChannelLimits,
    spill_root: Path | Callable[[], Path],
    spill_id: str,
    task_id: str = "",
) -> dict[str, Any]:
    """The CallToolResult for one tool reply, fitted to ``limits``.

    ``payload`` for a successful reply (sent as one JSON document), ``error``
    for an `isError` one. Either way the reply carries ``reply_id``. When it
    does not fit, the reply is written into parts under
    ``spill_root/<spill_id>/`` (``spill_root`` may be a callable, asked only
    then) and the model gets a stub: the control fields,
    the parts and one sentence on how to read them. Replaying the same
    ``spill_id`` writes nothing new.
    """

    is_error = payload is None
    if is_error:
        result = _result(f"[reply {reply_id}] {error}", error=True)
    else:
        result = _result(json.dumps({"reply_id": reply_id, **payload}, ensure_ascii=False), error=False)
    limit = limits.reply_error_bytes if is_error else limits.reply_ok_bytes
    size = reply_size(result, limits.measure)
    if not limit or size <= limit:
        return result
    if is_error:
        document = error
        control: dict[str, Any] = {}
    else:
        control = {key: payload[key] for key in CONTROL_FIELDS if key in payload}
        # Control fields too: the stub keeps them only while they fit.
        document = render_spill(payload)
    directory = Path(spill_root() if callable(spill_root) else spill_root) / spill_id
    meta_path = directory / "meta.json"
    if meta_path.exists():
        names = [str(name) for name in json.loads(meta_path.read_text(encoding="utf-8"))["parts"]]
    else:
        names = write_document_parts(directory, document, limits)
        meta_path.write_text(
            json.dumps({"task_id": task_id, "parts": names, "bytes": size}, ensure_ascii=False),
            encoding="utf-8",
        )
    return _stub(
        control=control,
        reply_id=reply_id,
        size=size,
        names=names,
        directory=directory,
        spill_id=spill_id,
        limits=limits,
        limit=limit,
        is_error=is_error,
    )


def _stub(
    *,
    control: Mapping[str, Any],
    reply_id: str,
    size: int,
    names: Sequence[str],
    directory: Path,
    spill_id: str,
    limits: ChannelLimits,
    limit: int,
    is_error: bool,
) -> dict[str, Any]:
    if limits.overflow == "files":
        note = fragment("fragment_reply_spill_files_v1.md", total_bytes=size, count=len(names))
        listings = [part_list(directory, names), part_list(directory, names, max_units=1)]
    else:
        # A CLI given `batch` makes no parallel calls, and `batch` does not run
        # `read_spill` (two parts together would spill again).
        name = "fragment_reply_spill_tool_serial_v1.md" if limits.batch_tool else "fragment_reply_spill_tool_v1.md"
        note = fragment(name, total_bytes=size, count=len(names), spill_id=spill_id)
        listings = [""]

    def build(listing: str, kept: Mapping[str, Any]) -> dict[str, Any]:
        if is_error:
            text = f"[reply {reply_id}] {note}" + (f"\n{listing}" if listing else "")
            return _result(text, error=True)
        spilled: dict[str, Any] = {"spill_id": spill_id, "parts": len(names)}
        if listing:
            spilled["files"] = listing
        body = {"reply_id": reply_id, **kept, "spilled": spilled, "note": note}
        return _result(json.dumps(body, ensure_ascii=False), error=False)

    # Control fields and the full list first; then the list as a range; then
    # only the fields the session cannot go on without.
    essential = {key: control[key] for key in ("status", "task_id") if key in control}
    for kept in (control, essential):
        for listing in listings:
            stub = build(listing, kept)
            if reply_size(stub, limits.measure) <= limit:
                return stub
    return build(listings[-1], essential)


def read_spill_part(spill_root: Path, spill_id: str, part: Any, *, task_id: str) -> dict[str, Any]:
    """One part of a spilled reply of this task, for `read_spill`."""

    spill_id = str(spill_id or "").strip()
    if not _SPILL_ID.match(spill_id):
        raise ValueError(f"unknown spill_id {spill_id!r}")
    meta_path = Path(spill_root) / spill_id / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"unknown spill_id {spill_id!r}") from None
    if str(meta.get("task_id") or "") != task_id:
        raise ValueError(f"spill {spill_id!r} belongs to another task")
    names = list(meta.get("parts") or [])
    try:
        index = int(part)
    except (TypeError, ValueError):
        raise ValueError("part must be an integer") from None
    if not 1 <= index <= len(names):
        raise ValueError(f"part must be 1 to {len(names)}")
    with open(Path(spill_root) / spill_id / names[index - 1], encoding="utf-8", newline="") as handle:
        text = handle.read()
    return {"spill_id": spill_id, "part": index, "parts": len(names), "text": text}

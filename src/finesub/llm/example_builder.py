"""Generate the correction prompt's worked examples from curated material.

The examples used to be three hand-written fragment files that shared one
43-line scene, kept in sync by hand. Every derived number in them -- durations,
gaps, char counts, the "merged span would be ~5.2s" teaching asides -- was
also computed by hand, and the local ids they referenced had gone stale in at
least three places (the current guarantees live in ``docs/llm_prompts.md``).

Material files under ``prompt_templates/examples/`` carry only what a human
should be writing: the scene, the corrected text, the translation, and the
teaching point. Everything a machine can check is a machine's job here:

- input rows carry ``start`` and ``duration``; **gap is computed** from the row
  before (the pause before each row, empty on the first), so a scene edit cannot
  leave a stale gap behind;
- rows are addressed by stable ``label``, never by number; the builder assigns
  the local ids and resolves ``{ref:label}`` in the prose, so "stale line
  number" stops being a representable state;
- ``{thr:name}`` resolves against the window's :class:`~finesub.llm.prompt_constants.Thresholds`,
  so a threshold lives in exactly one place;
- the input rows' durations follow the thresholds' length ratio (the pauses between rows do
  not, and each start is recomputed from them), so a teaching aside that compares a span with
  a threshold holds at every tier;
- ``{calc:...}`` computes a teaching aside (a merged span, a merged char count,
  the latter through the same :func:`weighted_char_count` the runtime validator
  uses) from the rows it is talking about.

The column layout comes from ``output_protocol.output_csv_header``, which is what
retires ``_add_start_to_example_outputs``: a start-bearing variant gets its
start column generated from the row data rather than spliced into finished text
by a regex.

Rendering is deterministic: material files load in filename order, nothing
iterates a set, and the same inputs render byte-identical twice (test-enforced).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from string import Template
import re
from typing import Dict, List, Mapping, Sequence, Tuple

from finesub.subtitles.metrics import weighted_char_count

from .chunking import Seam
from .output_protocol import output_csv_header, output_row_spec
from .prompt_constants import Thresholds
from .routing.profiles import MEDIA, TranslationProfile

EXAMPLES_DIR = Path(__file__).resolve().parent / "prompt_templates" / "examples"
# a correct-only task's examples (`CorrectionVariant.correct_only`): Chinese material, its second text column the
# subtitle line (`polished`)
EXAMPLES_ZH_DIR = EXAMPLES_DIR.with_name("examples_zh")

ASR_EXAMPLE_HEADER = "local_id|start|duration|gap|text"
ASR_EXAMPLE_HEADER_WITH_TURN = "local_id|start|duration|gap|turn|text"


class ExampleMaterialError(RuntimeError):
    """A material file is malformed, or a placeholder cannot be resolved."""


@dataclass(frozen=True)
class InputRow:
    label: str
    start: float
    duration: float
    text: str

    @property
    def end(self) -> float:
        return round(self.start + self.duration, 3)


@dataclass(frozen=True)
class OutputRow:
    kind: str  # sub | discard | (anything output_protocol accepts)
    labels: Tuple[str, ...]
    corrected: str = ""
    translation: str = ""
    conf: str = ""
    note: str = ""


@dataclass(frozen=True)
class ExampleBlock:
    """One ``<asr_result>`` + one ``<translated>`` pair."""

    inputs: Tuple[InputRow, ...]
    outputs: Tuple[OutputRow, ...]
    # Header repetition is per side and deliberate: a short example may still
    # need the input header to be readable while omitting the output one (they
    # cost tokens and teach nothing the oneshot has not).
    input_headers: bool = False
    output_headers: bool = False
    # Prose written between the <input> and the <output> (a bad-output example's 「错误输出：」), rendered
    # right before the output block.
    output_lead: str = ""


@dataclass(frozen=True)
class OverlayBlock:
    """One output block of an overlay group."""

    rows: Tuple[OutputRow, ...]
    lead: str = ""
    headers: bool = True


# Which overlay groups a variant reads, the first an example has winning. BasicA merges only a word cut in two,
# so the main scene has its own one-row-per-source output for it. capableC judges who is speaking itself and
# merges by pass 2's wording, so an example whose output or prose would read wrong there carries a `single`
# group; where it has none, the `nosingles` one (the unmarked block) reads right.
OVERLAY_GROUPS_FOR_VARIANT: Mapping[str, Tuple[str, ...]] = {
    "capableC": ("single", "nosingles"),
    "basicA": ("basic",),
}
# The group three passes' pass 2 reads where an example has one: its input is grouped and its merge wording
# differs, so an example whose output or prose would read wrong there carries its own.
GROUPED_GROUP = "grouped"
# frontmatter `grouped:` -- an example shown only in single pass (`no`) or only in three passes' pass 2 (`only`)
_GROUPED_MODES = ("", "no", "only")


@dataclass(frozen=True)
class Example:
    id: str
    kind: str  # main | mini | bad-output
    # Variant names this example is written for. Empty means "any".
    applies: Tuple[str, ...]
    # Switch predicates the example needs, as ``axis=value`` terms ANDed
    # together (``requires: media=audio`` etc.). This is the M4 form: an example
    # that talks about re-listening belongs to a vector that has audio, and one
    # that cites search results belongs to a vector that receives them.
    requires_switches: Tuple[str, ...]
    teach: str
    lead: str = ""
    trailer: str = ""
    preceding: Tuple[InputRow, ...] = ()
    blocks: Tuple[ExampleBlock, ...] = ()
    # Closing prose per overlay group ("对照要点"), which differs by variant
    # because the merge decisions it walks through do.
    notes: Mapping[str, str] = field(default_factory=dict)
    # Per-group output overlays for the shared main scene: one scene, one
    # input block, and the group's own <translated> output.
    overlays: Mapping[str, Tuple["OverlayBlock", ...]] = field(default_factory=dict)
    # `grouped:` in the frontmatter, one of _GROUPED_MODES
    grouped: str = ""
    # The labels whose `turn` cell reads 可并 in three passes' pass 2: every row the example merges into the
    # one before it (a voided merge too), and any it keeps apart only for length.
    may: Tuple[str, ...] = ()
    # `single_pass_skip:` -- the variants the example is not shown to in single pass (it still is in three
    # passes' pass 2, which reads the variant's examples): the filler sandwich teaches the one three-source
    # shape, which capableC, merging up to four lines, does not have.
    single_pass_skip: Tuple[str, ...] = ()


# --- Material parsing --------------------------------------------------------

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_BLOCK = re.compile(
    # A quoted attribute may itself contain ">" -- lead prose routinely names a
    # tag like `<translated>` -- so quotes are matched before the closing ">".
    r"<(?P<tag>input|output|preceding|notes)(?P<attrs>(?:\"[^\"]*\"|[^>])*)>"
    r"\n(?P<body>.*?)\n</(?P=tag)>",
    re.DOTALL,
)


def _parse_frontmatter(text: str, *, source: str) -> Tuple[Dict[str, str], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        raise ExampleMaterialError(f"{source}: missing frontmatter")
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ExampleMaterialError(f"{source}: bad frontmatter line {line!r}")
        meta[key.strip()] = value.strip()
    return meta, text[match.end() :]


def _parse_list(value: str) -> Tuple[str, ...]:
    stripped = (value or "").strip().strip("[]")
    return tuple(item.strip() for item in stripped.split(",") if item.strip())


def _parse_float(value: str, *, source: str, what: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        raise ExampleMaterialError(f"{source}: {what} is not a number: {value!r}")


def _parse_input_rows(body: str, *, source: str) -> Tuple[InputRow, ...]:
    rows: List[InputRow] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4:
            raise ExampleMaterialError(
                f"{source}: input row needs label|start|duration|text, got {line!r}"
            )
        rows.append(
            InputRow(
                label=parts[0],
                start=_parse_float(parts[1], source=source, what="start"),
                duration=_parse_float(parts[2], source=source, what="duration"),
                text=parts[3],
            )
        )
    return tuple(rows)


def _parse_output_rows(body: str, *, source: str) -> Tuple[OutputRow, ...]:
    rows: List[OutputRow] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split("|")]
        kind = parts[0]
        if kind == "discard":
            if len(parts) != 3:
                raise ExampleMaterialError(
                    f"{source}: discard row needs discard|label|note, got {line!r}"
                )
            rows.append(
                OutputRow(
                    kind="discard",
                    labels=(parts[1],),
                    note=parts[2],
                )
            )
            continue
        if len(parts) != 6:
            raise ExampleMaterialError(
                f"{source}: output row needs 6 fields "
                f"(type|position|corrected|translation|conf|note), got {len(parts)} in {line!r}"
            )
        rows.append(
            OutputRow(
                kind=kind,
                labels=tuple(part.strip() for part in parts[1].split(",")),
                corrected=parts[2],
                translation=parts[3],
                conf=parts[4],
                note=parts[5],
            )
        )
    return tuple(rows)


def _attr(attrs: str, name: str) -> str:
    # Quoted attributes (the lead prose) are stripped first so text like
    # `lead="... variant=..."` can never satisfy a bare-attribute lookup.
    unquoted = re.sub(r'\w+="[^"]*"', "", attrs or "")
    match = re.search(rf"{name}=([\w-]+)", unquoted)
    return match.group(1) if match else ""


def _attr_text(attrs: str, name: str) -> str:
    """A quoted attribute -- the prose that introduces an overlay block."""

    match = re.search(rf'{name}="([^"]*)"', attrs or "")
    return match.group(1) if match else ""


def parse_example(text: str, *, source: str) -> Example:
    meta, body = _parse_frontmatter(text, source=source)
    for required in ("id", "kind"):
        if required not in meta:
            raise ExampleMaterialError(f"{source}: frontmatter needs {required!r}")

    preceding: Tuple[InputRow, ...] = ()
    blocks: List[ExampleBlock] = []
    overlays: Dict[str, List[OverlayBlock]] = {}
    notes: Dict[str, str] = {}
    pending_inputs: Tuple[InputRow, ...] | None = None
    spans: List[Tuple[int, int]] = []
    for match in _BLOCK.finditer(body):
        tag = match.group("tag")
        attrs = match.group("attrs")
        block_body = match.group("body")
        # Prose between two blocks renders only where it is written before a plain <output> (that block's lead);
        # anywhere else it would have nowhere to go but after the whole example, out of order.
        gap = body[spans[-1][1] : match.start()].strip() if spans else ""
        spans.append(match.span())
        if gap and (tag != "output" or pending_inputs is None or _attr(attrs, "variant")):
            raise ExampleMaterialError(
                f"{source}: prose before <{tag}> between two blocks has no place to render: {gap[:40]!r}"
            )
        if tag == "notes":
            notes[_attr(attrs, "variant") or "nosingles"] = block_body.strip()
        elif tag == "preceding":
            preceding = _parse_input_rows(block_body, source=source)
        elif tag == "input":
            pending_inputs = _parse_input_rows(block_body, source=source)
        else:
            rows = _parse_output_rows(block_body, source=source)
            variant_group = _attr(attrs, "variant")
            if variant_group:
                overlays.setdefault(variant_group, []).append(
                    OverlayBlock(
                        rows=rows,
                        lead=_attr_text(attrs, "lead"),
                    )
                )
                continue
            if pending_inputs is None:
                raise ExampleMaterialError(
                    f"{source}: <output> without a preceding <input>"
                )
            blocks.append(
                ExampleBlock(
                    inputs=pending_inputs,
                    outputs=rows,
                    input_headers=meta.get("headers", "").strip().lower() == "true",
                    output_headers=(
                        meta.get("output_headers", "").strip().lower() == "true"
                    ),
                    output_lead=gap,
                )
            )
            pending_inputs = None
    if overlays and pending_inputs is not None:
        # Overlay files declare the shared input once, then one output per group.
        blocks.append(
            ExampleBlock(
                inputs=pending_inputs,
                outputs=(),
                input_headers=meta.get("headers", "").strip().lower() == "true",
            )
        )

    lead, trailer = (body[: spans[0][0]], body[spans[-1][1] :]) if spans else (body, "")
    grouped = meta.get("grouped", "").strip()
    if grouped not in _GROUPED_MODES:
        raise ExampleMaterialError(f"{source}: grouped must be one of {_GROUPED_MODES}, got {grouped!r}")
    may = _parse_list(meta.get("may", ""))
    labels = {row.label for block in blocks for row in block.inputs}
    if not set(may) <= labels:
        raise ExampleMaterialError(f"{source}: may names labels with no input row: {sorted(set(may) - labels)}")

    return Example(
        id=meta["id"],
        kind=meta["kind"],
        applies=_parse_list(meta.get("applies", "")),
        requires_switches=_parse_list(meta.get("requires", "")),
        teach=meta.get("teach", ""),
        lead=lead.strip(),
        trailer=trailer.strip(),
        preceding=preceding,
        blocks=tuple(blocks),
        overlays={key: tuple(value) for key, value in overlays.items()},
        notes=notes,
        grouped=grouped,
        may=may,
        single_pass_skip=_parse_list(meta.get("single_pass_skip", "")),
    )


def load_examples(directory: Path = EXAMPLES_DIR) -> Tuple[Example, ...]:
    """Every material file, in filename order (determinism, §4)."""

    return tuple(
        parse_example(path.read_text(encoding="utf-8"), source=path.name)
        for path in sorted(directory.glob("*.md"))
    )


# --- Rendering ---------------------------------------------------------------


def _fmt_time(value: float) -> str:
    """Seconds always carry one decimal, matching the runtime CSV."""

    return f"{round(float(value) + 1e-9, 1):.1f}"


def _fmt_count(value: float) -> str:
    """Weighted char counts drop a trailing ``.0`` (7, not 7.0)."""

    rounded = round(float(value) + 1e-9, 1)
    if abs(rounded - int(rounded)) < 1e-9:
        return str(int(rounded))
    return f"{rounded:.1f}"


def _gap_before(rows: Sequence[InputRow], index: int) -> float | None:
    """The pause before row ``index``, the row before it's end to its start (`chunking.render_segments_as_csv`);
    None for the first row."""

    if index == 0:
        return None
    return round(max(0.0, rows[index].start - rows[index - 1].end), 3)


def _fmt_gap(value: float | None) -> str:
    return "" if value is None else _fmt_time(value)


def _tenths(value: float) -> int:
    return int((Decimal(str(value)) * 10).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def scaled_rows(rows: Sequence[InputRow], scale: float) -> Tuple[InputRow, ...]:
    """``rows`` at the length ratio ``scale``: each duration scaled and kept to 0.1 s, the pause after each row
    as it was, each start following from them (the first stays where it is). A pause is not a length: the
    rule that a pause of about a second keeps two lines apart does not scale, so neither does a pause here."""

    out: List[InputRow] = []
    for index, row in enumerate(rows):
        if index:
            previous = rows[index - 1]
            start = _tenths(out[-1].start) + _tenths(out[-1].duration) + _tenths(row.start) - _tenths(previous.end)
            if start < _tenths(out[-1].start):
                # a row that overlapped the one before by more than the scaling took off would now open before it
                raise ExampleMaterialError(
                    f"row {row.label!r} would start before {previous.label!r} at length ratio {scale}"
                )
        else:
            start = _tenths(row.start)
        duration = int((Decimal(_tenths(row.duration)) * Decimal(str(scale))).quantize(Decimal("1"),
                                                                                      rounding=ROUND_HALF_UP))
        out.append(replace(row, start=start / 10, duration=duration / 10))
    return tuple(out)


def _scaled_example(example: Example, scale: float) -> Example:
    """``example`` at the length ratio ``scale``. The preceding context is scaled anchored on its last row's end,
    each block's rows on their first row's start: the two still meet where they did, so the pause into the
    window is kept at every tier (no pause scales), no window row goes below 0, and the context, rendered once,
    sits at one place whichever block follows it."""

    preceding = scaled_rows(example.preceding, scale)
    if preceding:
        shift = _tenths(example.preceding[-1].end) - _tenths(preceding[-1].end)
        preceding = tuple(replace(row, start=(_tenths(row.start) + shift) / 10) for row in preceding)
    blocks = []
    for block in example.blocks:
        rows = scaled_rows(block.inputs, scale)
        if preceding and rows and _tenths(rows[0].start) < _tenths(preceding[-1].start):
            raise ExampleMaterialError(
                f"example {example.id}: row {rows[0].label!r} would start before the preceding "
                f"{preceding[-1].label!r} at length ratio {scale}"
            )
        blocks.append(replace(block, inputs=rows))
    return replace(example, preceding=preceding, blocks=tuple(blocks))


def _render_input_block(
    rows: Sequence[InputRow],
    ids: Mapping[str, int],
    *,
    headers: bool,
    turns: bool = False,
    may: frozenset[str] = frozenset(),
) -> str:
    """``turns``: the task's input carries the `turn` column, so the example does too -- empty, since no
    example is about speakers and the examples must read like the input they teach, except that it reads
    可并 on the labels in ``may`` (three passes' pass 2)."""

    lines = [ASR_EXAMPLE_HEADER_WITH_TURN if turns else ASR_EXAMPLE_HEADER] if headers else []
    for index, row in enumerate(rows):
        lines.append(
            "|".join(
                (
                    str(ids[row.label]),
                    _fmt_time(row.start),
                    _fmt_time(row.duration),
                    _fmt_gap(_gap_before(rows, index)),
                    *([Seam.MAY.value if row.label in may else ""] if turns else []),
                    row.text,
                )
            )
        )
    return "\n".join(lines)


def _render_preceding_block(rows: Sequence[InputRow]) -> str:
    """Read-only context lines, numbered backwards from 0."""

    lines = []
    for index, row in enumerate(rows):
        local_id = index - (len(rows) - 1)
        lines.append(
            "|".join(
                (
                    str(local_id),
                    _fmt_time(row.start),
                    _fmt_time(row.duration),
                    _fmt_gap(_gap_before(rows, index)),
                    row.text,
                )
            )
        )
    return "\n".join(lines)


def _render_output_block(
    block: ExampleBlock,
    outputs: Sequence[OutputRow],
    ids: Mapping[str, int],
    *,
    with_start: bool,
    thresholds: Thresholds,
    polished: bool = False,
    headers: bool = False,
) -> str:
    """``polished``: a correct-only task's examples (`EXAMPLES_ZH_DIR`), whose second text column is headed
    `polished` (`CorrectionVariant.correct_only`)."""

    def resolve(text: str) -> str:
        return _resolve(text, ids=ids, block=block, thresholds=thresholds, outputs=outputs)

    by_label = {row.label: index for index, row in enumerate(block.inputs)}
    header = output_csv_header(start=with_start, polished=polished)
    lines = [header] if headers else []
    for row in outputs:
        if row.kind == "discard":
            lines.append(f"discard|{ids[row.labels[0]]}|{resolve(row.note)}")
            continue
        position = ",".join(str(ids[label]) for label in row.labels)
        fields = [row.kind, position]
        if with_start:
            fields.append(
                _fmt_time(block.inputs[by_label[row.labels[0]]].start)
            )
        cells = {
            "corrected": row.corrected,
            "translation": row.translation,
            "conf": row.conf,
            "note": resolve(row.note),
        }
        fields.extend(cells[name] for name in output_row_spec()[2:])
        lines.append("|".join(fields))
    return "\n".join(lines)


# --- Placeholder resolution --------------------------------------------------

_PLACEHOLDER = re.compile(r"\{(ref|thr|calc):([^}]+)\}")
# A *malformed* placeholder -- "{calc span}" or "{thr_hard_chars}" -- does not
# match the resolver above, so without this it would ship to the model verbatim
# (docs/llm_prompts.md requires residue to be an assembly error, not a leak).
_PLACEHOLDER_RESIDUE = re.compile(r"\{\s*(?:ref|thr|calc)[^}]*\}")


def _calc(
    expression: str,
    block: ExampleBlock,
    ids: Mapping[str, int],
    outputs: Sequence[OutputRow] = (),
) -> str:
    """Compute a teaching aside from the rows it is talking about.

    ``span``   -- first row's start to the last row's end
    ``dur``    -- one input row's own duration
    ``gap``    -- the pause before one input row (an error on the block's first row, which has none)
    ``cc``     -- the weighted char count of the output row starting at a label; given several labels,
                  the sum over the rows starting at each (about what merging them would make)

    Every one of these used to be typed by hand into an example comment, which
    is where the shipped arithmetic errors came from.
    """

    op, _, arg = expression.partition(":")
    labels = [label.strip() for label in arg.split(",") if label.strip()]
    by_label = {row.label: index for index, row in enumerate(block.inputs)}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ExampleMaterialError(f"calc references unknown labels: {missing}")
    rows = [block.inputs[by_label[label]] for label in labels]
    if op == "span":
        return _fmt_time(round(rows[-1].end - rows[0].start, 3))
    if op == "dur":
        return _fmt_time(rows[0].duration)
    if op == "gap":
        gap = _gap_before(block.inputs, by_label[labels[0]])
        if gap is None:
            raise ExampleMaterialError(f"calc gap: {labels[0]!r} is the block's first row, with no pause before it")
        return _fmt_time(gap)
    if op == "cc":
        total = 0.0
        for label in labels:
            row = next((row for row in outputs if row.labels and row.labels[0] == label), None)
            if row is None:
                raise ExampleMaterialError(f"no output row starts at label {label!r}")
            total += weighted_char_count(row.translation)
        return _fmt_count(total)
    raise ExampleMaterialError(f"unknown calc op {op!r}")


def _resolve(
    text: str,
    *,
    ids: Mapping[str, int],
    block: ExampleBlock | None,
    thresholds: Thresholds,
    outputs: Sequence[OutputRow] = (),
) -> str:
    def replace(match: re.Match[str]) -> str:
        kind, body = match.group(1), match.group(2)
        if kind == "ref":
            if body not in ids:
                raise ExampleMaterialError(f"unknown row label in {{ref:{body}}}")
            return str(ids[body])
        if kind == "thr":
            if body not in thresholds.values:
                raise ExampleMaterialError(f"unknown threshold {{thr:{body}}}")
            return str(thresholds.values[body])
        if block is None:
            raise ExampleMaterialError(f"{{calc:{body}}} outside an example block")
        return _calc(body, block, ids, outputs)

    rendered = _PLACEHOLDER.sub(replace, text or "")
    residue = _PLACEHOLDER_RESIDUE.findall(rendered)
    if residue:
        raise ExampleMaterialError(
            "unresolved placeholder residue (check the ':' separator and the "
            f"name): {sorted(set(residue))}"
        )
    return rendered


def profile_satisfies(
    example: Example, profile: "TranslationProfile | None"
) -> bool:
    """Whether this example's switch requirements hold for ``profile``.

    ``None`` (a caller with no profile in hand) satisfies everything -- the
    variant filter still applies.
    """

    if profile is None:
        return True
    for term in example.requires_switches:
        axis, operator, value = _split_requirement(term)
        # Examples live in the correction prompt, so their ``media`` axis is
        # the correction window's switch (plan v2 D20). The example files keep
        # the short name.
        attr = "correction_media" if axis == "media" else axis
        actual = getattr(profile, attr, None)
        if actual is None:
            raise ExampleMaterialError(
                f"{example.id}: requires unknown switch axis {axis!r}"
            )
        if operator == ">=":
            # media is a ladder, so "needs audio" is satisfied by video.
            if axis != "media":
                raise ExampleMaterialError(
                    f"{example.id}: '>=' only applies to the media ladder, got {term!r}"
                )
            if MEDIA.index(str(actual)) < MEDIA.index(value):
                return False
        elif str(actual) != value:
            return False
    return True


def _split_requirement(term: str) -> Tuple[str, str, str]:
    for operator in (">=", "="):
        axis, found, value = term.partition(operator)
        if found:
            return axis.strip(), operator, value.strip()
    raise ExampleMaterialError(f"malformed requires term {term!r}")


def render_example(
    example: Example,
    *,
    variant: str,
    with_start: bool,
    thresholds: Thresholds,
    polished: bool = False,
    profile: "TranslationProfile | None" = None,
    turns: bool = False,
    grouped: bool = False,
) -> str:
    """Render one example ("" when its variant, switch vector, pass or task mode rules it out). ``grouped``: three
    passes' pass 2 -- the input carries `turn` with the example's 可并 marks, and the example's
    `grouped` overlay and notes stand in for its variant group's where it has them. ``thresholds``: the
    window's, whose numbers the prose cites and whose ratio the input rows' durations follow."""

    if example.applies and variant not in example.applies:
        return ""
    if not profile_satisfies(example, profile):
        return ""
    if example.grouped == ("no" if grouped else "only"):
        return ""
    if not grouped and variant in example.single_pass_skip:
        return ""
    example = _scaled_example(example, thresholds.scale)
    turns = turns or grouped
    may = frozenset(example.may) if grouped else frozenset()
    groups = (GROUPED_GROUP, "nosingles") if grouped else OVERLAY_GROUPS_FOR_VARIANT.get(variant, ("nosingles",))
    overlay_group = next((group for group in groups if group in example.overlays), groups[-1])
    notes_group = next((group for group in groups if group in example.notes), groups[-1])
    parts: List[str] = []
    ids: Dict[str, int] = {}
    block_for_prose: ExampleBlock | None = None
    # The closing prose may cite a rendered row's computed values, so it needs
    # the rows this variant actually got.
    rows_for_prose: List[OutputRow] = []
    for block in example.blocks:
        ids = {row.label: index + 1 for index, row in enumerate(block.inputs)}
        block_for_prose = block
        overlay_blocks: Tuple[OverlayBlock, ...] = ()
        if not block.outputs and example.overlays:
            overlay_blocks = example.overlays.get(overlay_group, ())
        if example.preceding:
            parts.append(
                "<preceding_context>\n"
                + _render_preceding_block(example.preceding)
                + "\n</preceding_context>"
            )
        parts.append(
            "<asr_result>\n"
            + _render_input_block(block.inputs, ids, headers=block.input_headers, turns=turns, may=may)
            + "\n</asr_result>"
        )
        if not block.outputs and example.overlays and not overlay_blocks:
            # An overlay example that has no block for this variant's group
            # would otherwise render an empty <translated></translated> while
            # the surrounding prose still describes a worked example. A plain
            # block with its own outputs inside an overlay file is fine.
            raise ExampleMaterialError(
                f"{example.id}: no output overlay for group {overlay_group!r} "
                f"(variant {variant}); groups present: {sorted(example.overlays)}"
            )
        emitted = overlay_blocks or (
            OverlayBlock(rows=block.outputs, lead=block.output_lead, headers=block.output_headers),
        )
        if not emitted[0].rows and not block.outputs:
            raise ExampleMaterialError(
                f"{example.id}: would render an empty <translated> block"
            )
        for overlay in emitted:
            if grouped and example.kind != "bad-output":
                unmarked = sorted({label for row in overlay.rows for label in row.labels[1:]} - may)
                if unmarked:
                    raise ExampleMaterialError(
                        f"{example.id}: merges rows not marked may in three passes' pass 2: {unmarked}"
                    )
            rows_for_prose.extend(overlay.rows)
            if overlay.lead:
                parts.append(
                    _resolve(
                        overlay.lead, ids=ids, block=block, thresholds=thresholds, outputs=overlay.rows
                    )
                )
            parts.append(
                "<translated>\n"
                + _render_output_block(
                    block,
                    overlay.rows,
                    ids,
                    with_start=with_start,
                    thresholds=thresholds,
                    polished=polished,
                    headers=overlay.headers,
                )
                + "\n</translated>"
            )
    body = "\n".join(part for part in parts if part)
    lead = _resolve(
        example.lead, ids=ids, block=block_for_prose, thresholds=thresholds, outputs=rows_for_prose
    )
    trailer = _resolve(
        example.trailer, ids=ids, block=block_for_prose, thresholds=thresholds, outputs=rows_for_prose
    )
    notes = _resolve(
        example.notes.get(notes_group, ""),
        ids=ids,
        block=block_for_prose,
        thresholds=thresholds,
        outputs=rows_for_prose,
    )
    return "\n".join(part for part in (lead, body, trailer, notes) if part.strip())


def build_examples(
    *,
    variant: str,
    with_start: bool,
    thresholds: Thresholds,
    polished: bool = False,
    profile: "TranslationProfile | None" = None,
    params: Mapping[str, str] | None = None,
    directory: Path = EXAMPLES_DIR,
    turns: bool = False,
    grouped: bool = False,
) -> str:
    """Every material example that applies to ``variant``, in filename order.

    ``params`` are the composer's modal substitutions. Material prose may still
    carry ``$noisy_span_handling`` and friends; they are filled in here because
    ``safe_substitute`` does not recurse into an already-substituted value, so
    a ``$`` left in the built block would ship to the model verbatim.
    """

    rendered = [
        render_example(
            example,
            variant=variant,
            with_start=with_start,
            thresholds=thresholds,
            polished=polished,
            profile=profile,
            turns=turns,
            grouped=grouped,
        )
        for example in load_examples(directory)
    ]
    text = "\n\n".join(part for part in rendered if part.strip())
    if params:
        text = Template(text).safe_substitute(dict(params))
    return text

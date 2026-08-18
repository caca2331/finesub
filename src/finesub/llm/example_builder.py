"""Generate the correction prompt's worked examples from curated material.

The examples used to be three hand-written fragment files that shared one
43-line scene, kept in sync by hand. Every derived number in them -- durations,
gaps, ``char_count``, the "merged span would be ~5.2s" teaching asides -- was
also computed by hand, and the local ids they referenced had gone stale in at
least three places (the current guarantees live in ``docs/llm_prompts.md``).

Material files under ``prompt_templates/examples/`` carry only what a human
should be writing: the scene, the corrected text, the translation, and the
teaching point. Everything a machine can check is a machine's job here:

- input rows carry ``start`` and ``duration``; **gap is computed** from the next
  row, so a scene edit cannot leave a stale gap behind;
- rows are addressed by stable ``label``, never by number; the builder assigns
  the local ids and resolves ``{ref:label}`` in the prose, so "stale line
  number" stops being a representable state;
- ``AUTO`` in a derived output column is computed -- ``duration`` and ``gap``
  from the source rows, ``char_count`` through the same
  :func:`weighted_char_count` the runtime validator uses;
- ``{thr:name}`` resolves against :mod:`finesub.llm.prompt_constants`, so a threshold
  lives in exactly one place;
- ``{calc:...}`` computes a teaching aside (a merged span, a merged char count)
  from the rows it is talking about.

The column layout comes from ``output_protocol``' own header constants, which is what
retires ``_add_start_to_example_outputs``: a start-bearing variant gets its
start column generated from the row data rather than spliced into finished text
by a regex.

Rendering is deterministic: material files load in filename order, nothing
iterates a set, and the same inputs render byte-identical twice (test-enforced).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from string import Template
import re
from typing import Dict, List, Mapping, Sequence, Tuple

from finesub.subtitles.metrics import weighted_char_count

from .output_protocol import OUTPUT_CSV_HEADER, OUTPUT_CSV_HEADER_WITH_START
from .prompt_constants import THRESHOLDS
from .routing.profiles import MEDIA, TranslationProfile

EXAMPLES_DIR = Path(__file__).resolve().parent / "prompt_templates" / "examples"

ASR_EXAMPLE_HEADER = "local_id|start|duration|gap|text"

# Columns a material file may leave as AUTO, in output-row order.
_AUTO = "AUTO"


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
    comment: str = ""  # capableC's leading "# ..." reasoning line
    # Explicit overrides; None means AUTO.
    duration: float | None = None
    gap: float | None = None
    char_count: float | None = None


@dataclass(frozen=True)
class ExampleBlock:
    """One ``<asr_result>`` + one ``<translated>``/``<singles>`` pair."""

    inputs: Tuple[InputRow, ...]
    outputs: Tuple[OutputRow, ...]
    tag: str = "translated"
    # Header repetition is per side and deliberate: a short example may still
    # need the input header to be readable while omitting the output one (they
    # cost tokens and teach nothing the oneshot has not).
    input_headers: bool = False
    output_headers: bool = False


@dataclass(frozen=True)
class OverlayBlock:
    """One output block of an overlay group."""

    tag: str
    rows: Tuple[OutputRow, ...]
    lead: str = ""
    headers: bool = True


# Which overlay group a variant reads. BasicA is the only two-stage prompt set;
# everything else takes the judgment-merge <translated> straight.
OVERLAY_GROUP_FOR_VARIANT: Mapping[str, str] = {
    "capableB": "nosingles",
    "capableC": "nosingles",
    "basicA": "basic",
    "basicB": "nosingles",
}


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
    # input block, and as many output blocks as a group needs (BasicA emits
    # <singles> then <translated>, the capable variants only <translated>).
    overlays: Mapping[str, Tuple["OverlayBlock", ...]] = field(default_factory=dict)


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
    pending_comment = ""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            pending_comment = stripped.lstrip("#").strip()
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
                    comment=pending_comment,
                )
            )
            pending_comment = ""
            continue
        if len(parts) != 9:
            raise ExampleMaterialError(
                f"{source}: output row needs 9 fields "
                f"(type|position|duration|gap|corrected|translation|conf|"
                f"char_count|note), got {len(parts)} in {line!r}"
            )
        rows.append(
            OutputRow(
                kind=kind,
                labels=tuple(part.strip() for part in parts[1].split(",")),
                duration=None
                if parts[2] == _AUTO
                else _parse_float(parts[2], source=source, what="duration"),
                gap=None
                if parts[3] == _AUTO
                else _parse_float(parts[3], source=source, what="gap"),
                corrected=parts[4],
                translation=parts[5],
                conf=parts[6],
                char_count=None
                if parts[7] == _AUTO
                else _parse_float(parts[7], source=source, what="char_count"),
                note=parts[8],
                comment=pending_comment,
            )
        )
        pending_comment = ""
    return tuple(rows)


def _attr(attrs: str, name: str) -> str:
    # Quoted attributes (the lead prose) are stripped first so text like
    # `lead="... tag=..."` can never satisfy a bare-attribute lookup.
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
        spans.append(match.span())
        tag = match.group("tag")
        attrs = match.group("attrs")
        block_body = match.group("body")
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
                        tag=_attr(attrs, "tag") or "translated",
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
                    tag=_attr(attrs, "tag") or "translated",
                    input_headers=meta.get("headers", "").strip().lower() == "true",
                    output_headers=(
                        meta.get("output_headers", "").strip().lower() == "true"
                    ),
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

    prose = body
    for start, end in reversed(spans):
        prose = prose[:start] + "\x00" + prose[end:]
    lead, _, trailer = prose.partition("\x00")
    trailer = trailer.replace("\x00", "").strip()

    return Example(
        id=meta["id"],
        kind=meta["kind"],
        applies=_parse_list(meta.get("applies", "")),
        requires_switches=_parse_list(meta.get("requires", "")),
        teach=meta.get("teach", ""),
        lead=lead.strip(),
        trailer=trailer,
        preceding=preceding,
        blocks=tuple(blocks),
        overlays={key: tuple(value) for key, value in overlays.items()},
        notes=notes,
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


def _gap_after(rows: Sequence[InputRow], index: int) -> float:
    if index + 1 >= len(rows):
        return 0.0
    return round(rows[index + 1].start - rows[index].end, 3)


def _render_input_block(
    rows: Sequence[InputRow], ids: Mapping[str, int], *, headers: bool
) -> str:
    lines = [ASR_EXAMPLE_HEADER] if headers else []
    for index, row in enumerate(rows):
        lines.append(
            "|".join(
                (
                    str(ids[row.label]),
                    _fmt_time(row.start),
                    _fmt_time(row.duration),
                    _fmt_time(_gap_after(rows, index)),
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
                    _fmt_time(_gap_after(rows, index)),
                    row.text,
                )
            )
        )
    return "\n".join(lines)


def _row_metrics(
    row: OutputRow, inputs: Sequence[InputRow], by_label: Mapping[str, int]
) -> Tuple[float, float, float]:
    """(duration, gap, char_count) for one output row, AUTO resolved."""

    indices = [by_label[label] for label in row.labels]
    first, last = inputs[indices[0]], inputs[indices[-1]]
    duration = (
        row.duration if row.duration is not None else round(last.end - first.start, 3)
    )
    gap = row.gap if row.gap is not None else _gap_after(inputs, indices[-1])
    char_count = (
        row.char_count
        if row.char_count is not None
        else weighted_char_count(row.translation)
    )
    return duration, gap, char_count


def _render_output_block(
    block: ExampleBlock,
    outputs: Sequence[OutputRow],
    ids: Mapping[str, int],
    *,
    with_start: bool,
    with_comments: bool,
    headers: bool = False,
) -> str:
    def resolve(text: str) -> str:
        return _resolve(text, ids=ids, block=block, outputs=outputs)

    by_label = {row.label: index for index, row in enumerate(block.inputs)}
    header = OUTPUT_CSV_HEADER_WITH_START if with_start else OUTPUT_CSV_HEADER
    lines = [header] if headers else []
    for row in outputs:
        if with_comments and row.comment:
            lines.append(f"# {resolve(row.comment)}")
        if row.kind == "discard":
            lines.append(f"discard|{ids[row.labels[0]]}|{resolve(row.note)}")
            continue
        duration, gap, char_count = _row_metrics(row, block.inputs, by_label)
        position = ",".join(str(ids[label]) for label in row.labels)
        fields = [row.kind, position]
        if with_start:
            fields.append(
                _fmt_time(block.inputs[by_label[row.labels[0]]].start)
            )
        fields.extend(
            (
                _fmt_time(duration),
                _fmt_time(gap),
                row.corrected,
                row.translation,
                row.conf,
                _fmt_count(char_count),
                resolve(row.note),
            )
        )
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
    ``gap``    -- one input row's gap to the next
    ``cc``     -- the weighted char count of the output row starting at a label

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
        return _fmt_time(_gap_after(block.inputs, by_label[labels[0]]))
    if op == "cc":
        for row in outputs:
            if row.labels and row.labels[0] == labels[0]:
                _duration, _gap, char_count = _row_metrics(row, block.inputs, by_label)
                return _fmt_count(char_count)
        raise ExampleMaterialError(f"no output row starts at label {labels[0]!r}")
    raise ExampleMaterialError(f"unknown calc op {op!r}")


def _resolve(
    text: str,
    *,
    ids: Mapping[str, int],
    block: ExampleBlock | None,
    outputs: Sequence[OutputRow] = (),
) -> str:
    def replace(match: re.Match[str]) -> str:
        kind, body = match.group(1), match.group(2)
        if kind == "ref":
            if body not in ids:
                raise ExampleMaterialError(f"unknown row label in {{ref:{body}}}")
            return str(ids[body])
        if kind == "thr":
            if body not in THRESHOLDS:
                raise ExampleMaterialError(f"unknown threshold {{thr:{body}}}")
            return str(THRESHOLDS[body])
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
    with_comments: bool,
    profile: "TranslationProfile | None" = None,
) -> str:
    """Render one example ("" when its variant or switch vector rules it out)."""

    if example.applies and variant not in example.applies:
        return ""
    if not profile_satisfies(example, profile):
        return ""
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
            group = OVERLAY_GROUP_FOR_VARIANT.get(variant, "nosingles")
            overlay_blocks = example.overlays.get(group, ())
        if example.preceding:
            parts.append(
                "<preceding_context>\n"
                + _render_preceding_block(example.preceding)
                + "\n</preceding_context>"
            )
        parts.append(
            "<asr_result>\n"
            + _render_input_block(block.inputs, ids, headers=block.input_headers)
            + "\n</asr_result>"
        )
        if not block.outputs and example.overlays and not overlay_blocks:
            # An overlay example that has no block for this variant's group
            # would otherwise render an empty <translated></translated> while
            # the surrounding prose still describes a worked example. A plain
            # block with its own outputs inside an overlay file is fine.
            raise ExampleMaterialError(
                f"{example.id}: no output overlay for group "
                f"{OVERLAY_GROUP_FOR_VARIANT.get(variant, 'nosingles')!r} "
                f"(variant {variant}); groups present: {sorted(example.overlays)}"
            )
        emitted = overlay_blocks or (
            OverlayBlock(
                tag=block.tag, rows=block.outputs, headers=block.output_headers
            ),
        )
        if not emitted[0].rows and not block.outputs:
            raise ExampleMaterialError(
                f"{example.id}: would render an empty {emitted[0].tag!r} block"
            )
        for overlay in emitted:
            rows_for_prose.extend(overlay.rows)
            if overlay.lead:
                parts.append(
                    _resolve(
                        overlay.lead, ids=ids, block=block, outputs=overlay.rows
                    )
                )
            parts.append(
                f"<{overlay.tag}>\n"
                + _render_output_block(
                    block,
                    overlay.rows,
                    ids,
                    with_start=with_start,
                    with_comments=with_comments,
                    headers=overlay.headers,
                )
                + f"\n</{overlay.tag}>"
            )
    body = "\n".join(part for part in parts if part)
    lead = _resolve(
        example.lead, ids=ids, block=block_for_prose, outputs=rows_for_prose
    )
    trailer = _resolve(
        example.trailer, ids=ids, block=block_for_prose, outputs=rows_for_prose
    )
    group = OVERLAY_GROUP_FOR_VARIANT.get(variant, "nosingles")
    notes = _resolve(
        example.notes.get(group, ""),
        ids=ids,
        block=block_for_prose,
        outputs=rows_for_prose,
    )
    return "\n".join(part for part in (lead, body, trailer, notes) if part.strip())


def build_examples(
    *,
    variant: str,
    with_start: bool,
    with_comments: bool,
    profile: "TranslationProfile | None" = None,
    params: Mapping[str, str] | None = None,
    directory: Path = EXAMPLES_DIR,
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
            with_comments=with_comments,
            profile=profile,
        )
        for example in load_examples(directory)
    ]
    text = "\n\n".join(part for part in rendered if part.strip())
    if params:
        text = Template(text).safe_substitute(dict(params))
    return text

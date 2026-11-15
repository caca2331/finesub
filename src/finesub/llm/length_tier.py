"""The task's length ratio: which one a correction task runs at, and where it came from.

The merge thresholds follow one ratio per task (`prompt_constants.Thresholds`). It resolves once, after the
task's style is settled and before pass 1's lines are grouped, from: an explicit `--split-length-scale`, the
style's length tier (`[字幕长度]` in its 档案 section), `config.toml`'s `[segmentation] length_scale`, 1.0 --
the same setting (`settings.SPLIT_LENGTH_SCALE`) the ASR splitter reads, with the style's tier between the
flag and the file. It is recorded with the task's choice of passes (`correction_mode.PassChoice`), and a resumed
task keeps the recorded one (`docs/llm_harness_behavior.md`「合并门槛与任务的长度比例」).

Which style's tier counts: the task-level selection only, read from the revision actually injected. Stacked
styles (`--style A,B`) take the first in the named order that has a tier, a later different one warned about. A
fast window that picks its style itself does not move the ratio (its prompt already states the numbers): a pick
whose tier differs from the task's ratio is warned about (`warn_window_pick`). A tier this module does not know
is warned about and not used.

A style's tier changes only on the user's words (owner 2026-11-05): the flows that maintain a style on them
(`ingest`, `repair`, a human edit) write the 档案 line; a knowledge update after a task writes or changes it only
when the user's own `--task-summary` asks, and holds the line back otherwise (`tier_line_gate`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence

from finesub import settings
from finesub.reporting import current_reporter

from .knowledge.base import _match_normalize
from .knowledge.node.proposals import LineChange, LineGate
from .knowledge.style import STYLE_CATEGORY, load_style_entries, profile_fields

#: The three tiers and their ratios (owner 2026-11-04); the names are the style preset's `[字幕长度]` values,
#: which the proposal check holds a written tier to (a test pins the two the same).
TIER_RATIOS: Mapping[str, float] = MappingProxyType({"偏短": 0.8, "标准": 1.0, "偏长": 1.25})

#: The style preset's label for the tier, and the section it counts in.
TIER_LABEL, PROFILE_SECTION = "字幕长度", "档案"

#: Where a task's ratio came from.
EXPLICIT, STYLE, CONFIG, DEFAULT, RECORD = "explicit", "style", "config", "default", "record"
#: Said beside a dry run's ratio when a model still has to pick the style.
STYLE_UNDECIDED = "风格未定，档位未计入"


@dataclass(frozen=True)
class TaskLengthScale:
    """A task's length ratio, its source (`EXPLICIT` / `STYLE` / `CONFIG` / `DEFAULT` / `RECORD`), and for a
    style's tier the tier and the style it came from. ``note``: what the ratio does not account for yet."""

    value: float
    source: str
    tier: str = ""
    style: str = ""
    note: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"value": self.value, "source": self.source}
        out.update({key: value for key, value in (("tier", self.tier), ("style", self.style), ("note", self.note))
                    if value})
        return out


def resolve_task_length_scale(explicit: float | None, *, tier: str = "", style: str = "", note: str = "",
                              config_path: str | Path | None = None) -> TaskLengthScale:
    """The ratio a task runs at, when it has none recorded: ``explicit`` (`--split-length-scale`), else the
    style's ``tier`` (named by ``style``), else `config.toml`, else the default 1.0 -- checked against the
    setting's range whichever wins (`settings.SPLIT_LENGTH_SCALE.resolve`)."""

    if explicit is not None:
        return TaskLengthScale(settings.SPLIT_LENGTH_SCALE.resolve(explicit), EXPLICIT, note=note)
    if tier:
        return TaskLengthScale(settings.SPLIT_LENGTH_SCALE.resolve(TIER_RATIOS[tier]), STYLE, tier=tier,
                               style=style, note=note)
    loaded = settings.read(config_path)
    source = CONFIG if settings.SPLIT_LENGTH_SCALE.configured(config=loaded) is not None else DEFAULT
    return TaskLengthScale(settings.SPLIT_LENGTH_SCALE.resolve(config=loaded), source, note=note)


def _entry_tier(key: str, text: str) -> str:
    """The tier an injected style entry states, "" when it states none or one this module does not know."""

    written = profile_fields(text).length
    if not written or written in TIER_RATIOS:
        return written
    current_reporter().warning(
        "style-length-tier-unknown",
        f"风格 {key} 的 [字幕长度] 写的是「{written}」，只认 {' / '.join(TIER_RATIOS)}",
        impact="这套风格的档位不参与，合并门槛按命令行、config.toml 或 1.0 定",
        action=f"把那一行改成 {' / '.join(TIER_RATIOS)} 之一",
    )
    return ""


def styles_tier(knowledge_root: str | Path, names: Sequence[str], *, rev: int | None) -> tuple[str, str]:
    """``(tier, style)``: the first style of ``names`` (in the order named) whose injected text at ``rev``
    states a known tier, ``("", "")`` when none does. A later style stating another tier is warned about."""

    found = ("", "")
    for key, text in load_style_entries(knowledge_root, names, rev=rev):
        tier = _entry_tier(key, text)
        if not tier:
            continue
        if not found[0]:
            found = (tier, key)
        elif tier != found[0]:
            current_reporter().warning(
                "style-length-tier-stacked",
                f"叠用的风格档位不同：{found[1]} 是{found[0]}，{key} 是{tier}",
                impact=f"按点名顺序里第一套写了档位的 {found[1]}（{found[0]}）定合并门槛",
            )
    return found


def task_length_scale(explicit: float | None, recorded: float | None, *, knowledge_root: str | Path | None,
                      names: Sequence[str] | None, rev: int | None) -> TaskLengthScale:
    """The ratio a task runs at. ``recorded``: the ratio the task recorded with its choice of passes, which a
    resumed task keeps over everything else -- an explicit one that differs is warned about. ``names``: the
    task-level style selection as injected (at ``rev``); ``None`` while a model still has to pick it (a dry run
    under `auto`, a fast window that picks itself), when the tier is not counted."""

    if recorded is not None:
        if explicit is not None and settings.SPLIT_LENGTH_SCALE.resolve(explicit) != recorded:
            current_reporter().warning(
                "length-scale-recorded",
                f"--split-length-scale {explicit:g} 与这个任务记下的比例 {recorded:g} 不同，续跑沿用记下的",
                impact="合并门槛照任务开始时的比例；ASR 分句若重跑，按这次的值",
                action="要换比例就删掉这个任务的 LLM 产物重跑",
            )
        return TaskLengthScale(recorded, RECORD)
    if explicit is not None:
        # The flag wins over any tier: reading the styles would only warn about tiers that do not decide.
        return resolve_task_length_scale(explicit)
    if names is None:
        return resolve_task_length_scale(None, note=STYLE_UNDECIDED)
    tier, style = styles_tier(knowledge_root, names, rev=rev) if names and knowledge_root is not None else ("", "")
    return resolve_task_length_scale(None, tier=tier, style=style)


def warn_window_pick(knowledge_root: str | Path | None, names: Sequence[str], *, rev: int | None,
                     task_scale: float | None) -> None:
    """A fast window picked its style itself: the tier does not take part (the window's prompt already states
    the task's numbers), so a picked tier whose ratio differs from the task's is said once."""

    if not names or knowledge_root is None or task_scale is None:
        return
    tier, style = styles_tier(knowledge_root, names, rev=rev)
    if tier and TIER_RATIOS[tier] != task_scale:
        current_reporter().warning(
            "style-length-tier-window-pick",
            f"窗口选中的风格 {style} 是{tier}（比例 {TIER_RATIOS[tier]:g}），这个任务按 {task_scale:g} 跑",
            impact="窗口自选风格时档位不参与：合并门槛照任务的比例",
            action="要按这一档并，用 --style 点名这套风格或给 --split-length-scale 重跑",
        )


#: Why a knowledge update skips a `[字幕长度]` change: the user gave no `--task-summary` to ask for one (not
#: fed back to the model: it may not change the tier at all), or the change moves the line or puts it outside 档案
#: (fed back, like any other rejection).
TIER_HELD = "用户没有写本任务的说明（`<task_summary>`），本任务不加、不改、不删 `[字幕长度]`"
TIER_PROFILE_ONLY = "`[字幕长度]` 只在「档案」节里加、改、删，不挪动、不写进别的节"


def tier_line_gate(user_summary: bool) -> LineGate:
    """A knowledge update's hold on a style's `[字幕长度]` line (owner 2026-11-05): without the user's own task
    summary (``user_summary`` False) every change that adds, rewrites, removes or moves such a line is skipped;
    with it, the model decides whether the summary asks for a tier, and only a move or a line landing outside
    the 档案 section is skipped (a move's `LineChange.section` is where the line is, not where it goes). The flows that maintain a style on the user's words (`ingest`, `repair`, a human edit)
    apply without this gate."""

    def gate(change: LineChange) -> str:
        if change.category != STYLE_CATEGORY or not any(
            payload is not None and _match_normalize(str(payload.get("label") or "")) == _match_normalize(TIER_LABEL)
            for payload in (change.before, change.after)
        ):
            return ""
        if not user_summary:
            return TIER_HELD
        if change.op == "move_line" or (change.after is not None and change.section != PROFILE_SECTION):
            return TIER_PROFILE_ONLY
        return ""

    return gate

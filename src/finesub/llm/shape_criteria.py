"""Who is speaking and the shape of the subtitles: the criteria two prompts share.

Pass 1 of three passes (the speaker judge's shape section, `stages/speaker_judge/prompt.py`) answers per row
`join`, `may_join` or `drop` in JSON; the single pass prompt (capableC, `prompt_compose`) makes the same calls
itself, in one call with the correction, and acts on them in its merges and discards
(`docs/llm_prompts.md`「单 pass 的拼法」). Both read the same criteria, kept in the
`shape_criteria_*` templates: when a line is the same person going on with a fragment to join, the same person
going on and mergeable, or a line to leave out. How to tell who is speaking (`who`, `promo`) only the single pass
reads from here; the judge's own prompt words it for naming tracks. The few words that differ
-- what only holds where pass 1 hands its answer to another call ("the next step corrects it", "joins over a
dropped line"), and that the single pass keeps a line only from the speaker calls -- are the phrase layer below.
"""

from __future__ import annotations

from typing import Dict

from .prompt_constants import PASS1, Thresholds

#: The criteria templates, by the slot each fills.
TEMPLATES = {
    "who": "shape_criteria_who_v1.md",
    "promo": "shape_criteria_promo_v1.md",
    "join": "shape_criteria_join_v1.md",
    "may_join": "shape_criteria_may_join_v1.md",
    "drop": "shape_criteria_drop_v1.md",
}

#: The phrases that depend on whether the calls are made in a step of their own (`pass1`) or within the
#: correction (`single`): pass 1 joins back over a dropped line, has its long joins held back by the script,
#: and hands every kept line to a correction that fixes it. The lines the criteria keep are kept by pass 1
#: outright; the single pass keeps them only from the speaker calls, its correction may still discard a filler
#: that carries nothing (its prompt says so after the criteria). Template files; None leaves the slot empty.
WORDS: Dict[str, Dict[str, str | None]] = {
    "pass1": {
        "across_drop": "shape_criteria_across_drop_v1.md",
        "auto_guard": "shape_criteria_auto_guard_v1.md",
        "strict_reason": "shape_criteria_strict_reason_v1.md",
        "fix_later": "shape_criteria_fix_later_v1.md",
        "keep_lead": "shape_criteria_keep_lead_v1.md",
    },
    "single": {
        "across_drop": None,
        "auto_guard": None,
        "strict_reason": None,
        "fix_later": "shape_criteria_fix_later_single_v1.md",
        "keep_lead": "shape_criteria_keep_lead_single_v1.md",
    },
}


def shape_criteria(
    words: str, *, same_person: str, correct_only: bool, thresholds: Thresholds | None = None
) -> Dict[str, str]:
    """The criteria, rendered for `words` (`pass1` or `single`), each stripped. ``same_person`` is the sentence
    that says how this prompt tells one person from another (with or without the acoustic tracks).
    ``correct_only``: the task's mode, which picks the language of the cue words the criteria quote (a
    connective a line stops on, a filler, a backchannel) from `prompt_templates/mode_texts_v1.toml`.
    ``thresholds``: the single pass's, the window's (required there). Pass 1's prompt is the same at every
    length ratio, at the standard tier's numbers (`prompt_constants.PASS1`): the script's checks before pass 2
    follow the task's ratio instead (`grouping.group_lines`)."""

    if words == "pass1":
        if thresholds is not None:
            raise ValueError("pass 1's criteria are at the standard tier's numbers; pass no thresholds")
        thresholds = PASS1
    elif thresholds is None:
        raise ValueError(f"the {words!r} criteria follow the window's length ratio: pass its thresholds")

    from .prompt_compose import load_prompt_template, task_mode_texts  # that module composes the single pass from these

    def phrase(name: str | None) -> str:
        return load_prompt_template(name).strip() if name else ""

    values = {
        **task_mode_texts(correct_only),
        **{key: str(value) for key, value in thresholds.values.items()},
        **{key: phrase(name) for key, name in WORDS[words].items()},
        "same_person": same_person,
    }
    return {slot: load_prompt_template(name, strict=True, **values).strip() for slot, name in TEMPLATES.items()}

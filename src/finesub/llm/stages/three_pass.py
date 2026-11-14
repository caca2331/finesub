"""Three passes in a correction task: which way a task runs, and the lines pass 1 hands pass 2.

A task runs one of two ways (`docs/llm_harness_behavior.md`「三个 pass」):

- `single`: the correction windows on the ASR lines as they are (capableC's single pass sorts out who is
  speaking and the shape of the lines itself, reading the acoustic 换人 in the `turn` column where there are labels);
- `three-pass`: pass 1 (the speaker judge with its shape section) groups the lines, pass 2 corrects the grouped
  lines and merges only where pass 1 marked 可并, pass 3 proofreads the finished subtitles.

The speaker judge decides it: a task with the judge on (`--diarize judge-*`) runs the three-pass split with the
tracked pass 1 -- also when the acoustic layer left no labels, where the judge skips and every window gets the
single pass prompt -- and a task with the judge off runs one pass, acoustic labels or not (owner 2026-10-30).
The split needs a difficulty for capable models (`THREE_PASS_DIFFICULTIES`: quality and intermediate; owner D1)
and a correction prompt set with a grouped version for every model the cell may answer with (only capableC has
one: `prompt_variants.has_grouped_variant`); the judge on without them is refused (`refuse_judge`; owner
2026-10-30: there is no "judge + one pass"). Pass 3 follows `--llm-proofread`: `auto` runs it at quality
(`PROOFREAD_DIFFICULTIES`, D2) whichever way the task runs, `on` at every difficulty, `off` never. The owner's
decisions cited as D1–D4 and their reasons: `docs/llm_design_notes.md`「三个 pass 的决策记录」.

The choice is written once into `correction-mode.json` in the task's artifact directory, with the speaker judge
the task ran with (the pipeline's `stages.task_diarize` reads it back when `--diarize` is unset), and a resumed task
follows it whatever its switches say now (README_DEV「复用的依据是任务身份」) -- so does the refusal: a judge the
task recorded runs on at efficiency too; a task that already has correction windows from before the file
existed runs one pass. `--no-resume` chooses again. Pass 1 failing for a window does not change the mode: that
window gets the single pass prompt (`grouping.pass2_windows`); so does every window of a task recorded as three
passes and resumed with the judge off, where there is no pass 1 to run (`pass1_shape`).

The same record holds the task's mode: a Chinese source only corrects (`task_correct_only`, decided first of all
and kept on resume; docs/llm_harness_behavior.md「只纠错」), and pass 3 then runs only when asked.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...speech.diarization import judgments as sidecar
from ...speech.diarization.labels import different_speakers
from ...subtitles import line_groups
from ..chunking import load_product_segments
from ..grouping import Grouping, line_acts, pass2_windows
from ..prompt_constants import Thresholds
from ..correction_mode import (  # noqa: F401 - the task's record, re-exported for its readers here
    MODE_FILENAME, MODE_VERSION, PROOFREAD_DIFFICULTIES, SINGLE, THREE_PASS, THREE_PASS_DIFFICULTIES, PassChoice,
    record_choice, recorded_choice, started,
)
from ..correction_request import CorrectionRequestError


def _no_split(difficulty: str, grouped_prompts: bool) -> str | None:
    """Why a task at `difficulty` cannot run the three-pass split, None when it can."""
    if difficulty not in THREE_PASS_DIFFICULTIES:
        return f"difficulty {difficulty} runs one pass"
    if not grouped_prompts:
        return "the correction cell's prompt set has no grouped version"
    return None


def _judge_refused(speaker_judge: str, why: str) -> CorrectionRequestError:
    return CorrectionRequestError(
        f"the speaker judge ({speaker_judge}) runs as pass 1 of the three-pass split, and {why}. Turn the judge "
        "off (pipeline: --diarize acoustics or off; correction module: --speaker-judge off), or use "
        "--llm-difficulty quality / intermediate with a prompt set that has a grouped version (capableC)."
    )


def choose(*, difficulty: str, grouped_prompts: bool, speaker_judge: str, proofread: str,
           correct_only: bool = False) -> PassChoice:
    """The way a new task runs (see the module docstring); a judge on where the split cannot run is refused.
    A correct-only task runs pass 3 only when asked (`on`): its prompt proofreads a translation."""
    if speaker_judge == "off":
        mode, reason = SINGLE, "speaker judge off"
    else:
        no_split = _no_split(difficulty, grouped_prompts)
        if no_split:
            raise _judge_refused(speaker_judge, no_split)
        mode, reason = THREE_PASS, f"{difficulty}, speaker judge {speaker_judge}"
    run_proofread = proofread == "on" or (
        proofread == "auto" and difficulty in PROOFREAD_DIFFICULTIES and not correct_only
    )
    return PassChoice(mode, run_proofread, reason, speaker_judge)


def refuse_judge(profile: Any, speaker_judge: str, recorded: PassChoice | None) -> None:
    """Refuse the speaker judge where the task cannot run the three-pass split it needs: efficiency, or a
    correction cell without a grouped prompt set (owner 2026-10-30). A resumed task whose record names this
    same judge passes: it follows the way it recorded. `recorded` is that record, None for a new task or
    `--no-resume`. The pipeline asks before ASR when the run reaches the LLM stage; the correction task asks again
    before its finished-task shortcuts (`run_correction_task`), and `task_choice` once more."""
    if speaker_judge == "off" or (recorded is not None and recorded.speaker_judge == speaker_judge):
        return
    no_split = _no_split(profile.difficulty, profile.difficulty in THREE_PASS_DIFFICULTIES
                         and grouped_prompts(profile))
    if no_split:
        raise _judge_refused(speaker_judge, no_split)


def grouped_prompts(profile: Any) -> bool:
    """Whether every prompt set the correction cell may answer with has a grouped version: the cell's variant
    and each model's override. False when there is no route declaration to ask."""
    from ..prompt_variants import correct_only_name, has_grouped_variant
    from ..routing.capabilities import correction_task_group
    from ..routing.config import role_config_for

    try:
        config = role_config_for(correction_task_group(profile), profile.difficulty)
    except Exception:  # noqa: BLE001 - no declaration: the single pass needs none
        return False
    names = [config.variant, *config.variant_overrides.values()]
    if profile.correct_only:
        names = [correct_only_name(name) for name in names]
    return all(name and has_grouped_variant(name) for name in names)


def source_language(stable_json: str | Path) -> str | None:
    """The language of the task's source: what ASR was told, else the segments' `lang` weighted by their
    characters (`recall_repair.episode_language`, the recall repair's own reading); None without either."""
    from ...speech.postprocessing.recall_repair import episode_language
    from ...subtitles.stable_rows import rows

    payload = json.loads(Path(stable_json).read_text(encoding="utf-8"))
    lines = rows(payload, None)  # speakers and cut lines do not change the language
    return episode_language(payload if isinstance(payload, dict) else {}, lines if isinstance(lines, list) else [])


def task_correct_only(
    request: Any, artifact_dir: str | Path | None, *, warn: Callable[[str], None]
) -> tuple[bool, str | None]:
    """Whether this task only corrects (its source is Chinese, `zh`), and the source language it read. A resumed
    task follows its record; one whose windows started before the record existed, or recorded before this
    field, translated and goes on translating. A source whose language cannot be told translates, said once
    through `warn`. Decided before anything the mode changes: the style, the finished-task shortcuts, the plan."""
    resuming = request.resume and artifact_dir is not None
    recorded = recorded_choice(artifact_dir) if resuming else None
    if recorded is not None:
        return recorded.correct_only, recorded.source_language
    if resuming and started(artifact_dir):
        return False, None
    try:
        language = source_language(request.stable_json)
    except (OSError, ValueError):
        return False, None  # no stable to read: the step that needs it says so
    if language is None:
        warn("判断不出素材的语言（ASR 结果没记语言），按翻译处理；中文素材在转写时加 --language zh 就只纠错、不翻译")
    return language == "zh", language


def task_choice(request: Any, artifact_dir: str | Path | None, rows: Sequence[Mapping[str, Any]], *,
                source_language: str | None = None) -> PassChoice:
    """The way this task runs: the recorded choice when resuming; one pass for a task whose windows started
    before choices were recorded; else a new choice (`record_choice` writes it), carrying the task's mode
    (`request.profile.correct_only`) and ``source_language``. A judge `refuse_judge` turns down is refused
    first."""
    resuming = request.resume and artifact_dir is not None
    recorded = recorded_choice(artifact_dir) if resuming else None
    refuse_judge(request.profile, request.speaker_judge, recorded)
    if recorded is not None:
        return recorded
    if resuming and started(artifact_dir):
        return PassChoice(SINGLE, request.proofread == "on", "the task started before three passes")
    choice = choose(difficulty=request.profile.difficulty, grouped_prompts=grouped_prompts(request.profile),
                    speaker_judge=request.speaker_judge, proofread=request.proofread,
                    correct_only=request.profile.correct_only)
    return replace(choice, correct_only=request.profile.correct_only, source_language=source_language)


def pass1_shape(speaker_judge: str, *, warn: Callable[[str], None]) -> str | None:
    """The pass 1 a three-pass task runs: the judge with the shape section. A task recorded as three passes and
    resumed with the judge off has none (a new task without the judge runs one pass, `choose`): every window
    it still has to correct gets the single pass prompt, said once through `warn`."""
    if speaker_judge != "off":
        return sidecar.SHAPE_TRACKED
    warn("这个任务记的是三个 pass，这次没开说话人裁判，没有 pass 1 可跑：还没纠完的窗按单 pass 纠错")
    return None


def pass1_groups(shape: str | None, rows: Sequence[Mapping[str, Any]]) -> bool:
    """Whether pass 1 can group any line: it has a shape and the judge can run on these rows
    (`judgments.judgeable`). Else the judge skips and every window gets the single pass prompt, which is what
    the dry run previews then."""
    return shape is not None and sidecar.judgeable(rows)


def _different(layer: Any, segments: Sequence[Any]) -> Callable[[str, str], bool]:
    """Whether two product lines are two people, as the correction's 换人 reads it: the judged speakers where the
    judge's verdicts hold, else the acoustic ones -- and two lines of different group keys that overlap in time
    (a cloud task's two people talking at once, or a local moved row on the overlap lane over the row it covers),
    whatever the labels say."""
    speakers = layer.speakers() or {}
    by_id = {segment.id: segment for segment in segments}
    position = {id_: i for i, id_ in enumerate(by_id)}
    at_once = line_groups.at_once(list(by_id.values()))

    def apart(a: str, b: str) -> bool:
        return a in position and b in position and at_once(position[a], position[b])

    return lambda a, b: (a in speakers and b in speakers and different_speakers(speakers[a], speakers[b])) \
        or apart(a, b)


def pass2_plan(windows: Sequence[Any], stable_json: str | Path, shape: str | None, *,
               warn: Callable[[str], None], thresholds: Thresholds,
               replayable: Callable[[Any], bool] | None = None) -> tuple[list[Any], Grouping, dict[str, Any]]:
    """Pass 2's windows, the grouping that maps their rows back, and a record of both: run once pass 1 is
    over, on `windows` already spelled as the product lines (`chunking.product_windows`); pass 1's acts, given
    on the judge's rows, carry over to them (`grouping.line_acts`). Without a pass 1 (`shape` None) or without
    a usable sidecar every window keeps the single pass prompt; so does every window when grouping fails
    (pass 1 is optional: warned, never the correction's failure). A join across one of this run's window edges
    (the judge cut its windows on an earlier run) is broken there, with one warning. ``thresholds``: the task's,
    at its length ratio, which the script's checks on pass 1's joins hold to (`grouping.group_lines`).

    `replayable` is for a resumed three-pass task whose correction cell no longer has a grouped prompt set
    (switched to another difficulty or variant since): only the grouped windows it says the ledger can replay
    stay grouped, every other window gets the single pass prompt (no grouped prompt could be built for it),
    with one warning."""
    segments, layer = load_product_segments(stable_json, warn=warn)
    different = _different(layer, segments)
    try:
        shaped = sidecar.read_acts(stable_json, layer.source_rows, shape=shape, warn=warn) if shape else None
        acts, missing = line_acts(shaped, layer)
        out, grouping = pass2_windows(windows, segments, acts, missing, different=different, thresholds=thresholds)
        if replayable is not None:
            cores = {window.chunk_id: window.segments[len(window.overlap_segments):] for window in windows}
            lost = [window.chunk_id for window in out if window.grouped and not replayable(window)]
            if lost:
                warn(f"这一格的 prompt 没有分组版（续跑时换了难度或 variant），{len(lost)} 个还没做完的窗按单 pass 纠错："
                     + ", ".join(lost))
                missing = set(missing) | {line.id for chunk in lost for line in cores[chunk]}
                out, grouping = pass2_windows(windows, segments, acts, missing, different=different,
                                              thresholds=thresholds)
    except Exception as error:  # noqa: BLE001 - see the docstring
        warn(f"pass 1 的结果没能分组（{type(error).__name__}: {error}），每窗按单 pass 纠错")
        out, grouping = pass2_windows(windows, segments, {}, None, different=different, thresholds=thresholds)
    edges = [key for key, why in grouping.refused if why == "window"]
    if edges:
        warn(f"pass 1 的 {len(edges)} 处合并跨过了这次纠错窗的边界（裁判的窗是上一次切的），在边界处断开："
             + ", ".join(edges))
    without = [window.chunk_id for window in out if not window.grouped]
    refused: dict[str, int] = {}
    for _key, why in grouping.refused:
        refused[why] = refused.get(why, 0) + 1
    return out, grouping, {
        "shape": shape, "windows": len(out), "without_pass1": without,
        "skipped": [window.chunk_id for window in windows if window.chunk_id not in {w.chunk_id for w in out}],
        "dropped": len(grouping.dropped), "joined": sum(len(group) - 1 for group in grouping.groups),
        "refused": refused,
    }


def preview_windows(windows: Sequence[Any]) -> list[Any]:
    """The dry run's pass 2 windows: pass 1 has not run, so the ASR lines as they are, under the grouped prompt
    (no line reads 可并)."""
    return [replace(window, grouped=True) for window in windows]

"""Three passes in a correction task: which way a task runs, and the lines pass 1 hands pass 2.

A task runs one of three ways (`docs/llm_harness_behavior.md`「三个 pass」):

- `single`: the correction windows on the ASR lines as they are (capableC's single pass sorts out who is
  speaking and the shape of the lines itself, reading the acoustic 换人 in the `turn` column where there are labels);
- `single-judged`: the same single pass that also names who speaks on each row (`capableC-tracked`): the speaker
  judge folded into the window, its answers written as the judge's sidecar (`correction/speaker_book.py`);
- `three-pass`: pass 1 (the speaker judge with its shape section) groups the lines, pass 2 corrects the grouped
  lines and merges only where pass 1 marked 可并, pass 3 proofreads the finished subtitles.

The speaker judge decides it: a task with the judge off runs one pass, acoustic labels or not (owner 2026-10-30).
With the judge on, an intermediate task runs `single-judged` (`ONE_PASS_JUDGE_DIFFICULTIES`; owner 2026-11-11)
whichever judge it asks for -- the window hears what `--llm-media` gives it, and an audio judge on a text window is
said once (`task_choice`) -- where its cell has a tracked set for every model it may answer with
(`prompt_variants.has_tracked_variant`: only capableC, in either mode) and the rows can be judged by
it (`window_speakers.tracked_rows`: acoustic labels in one block, not a cloud task's); every other task with the
judge on runs the three-pass split with the tracked pass 1 -- also when the acoustic layer left no labels, where
the judge skips and every window gets the single pass prompt. The split needs a difficulty for capable models
(`THREE_PASS_DIFFICULTIES`: quality and intermediate; owner D1) and a correction prompt set with a grouped version
for every model the cell may answer with (only capableC has one: `prompt_variants.has_grouped_variant`); the judge
on without them is refused (`refuse_judge`; owner 2026-10-30: there is no "judge + one pass" run as two steps).
Pass 3 follows `--llm-proofread`: `auto` runs it at quality (`PROOFREAD_DIFFICULTIES`, D2) whichever way the task
runs, `on` at every difficulty, `off` never. The owner's decisions cited as D1–D4 and their reasons:
`docs/llm_design_notes.md`「三个 pass 的决策记录」.

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
    MODE_FILENAME, MODE_VERSION, ONE_PASS_JUDGE_DIFFICULTIES, PROOFREAD_DIFFICULTIES, SINGLE, SINGLE_JUDGED,
    THREE_PASS, THREE_PASS_DIFFICULTIES, PassChoice, record_choice, recorded_choice, started,
)
from ..correction_request import CorrectionRequestError
from ..window_speakers import tracked_rows


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


def folds_judge(difficulty: str, speaker_judge: str, tracked_prompts: bool) -> bool:
    """Whether the speaker judge folds into the single pass (`SINGLE_JUDGED`): either judge, at a difficulty that
    folds it (the window's modality is `--llm-media`'s, owner 2026-11-11), with a tracked set for the cell; the rows
    are the caller's to check (`window_speakers.tracked_rows`)."""
    return speaker_judge in ("text", "audio") and difficulty in ONE_PASS_JUDGE_DIFFICULTIES and tracked_prompts


def choose(*, difficulty: str, grouped_prompts: bool, speaker_judge: str, proofread: str,
           correct_only: bool = False, tracked_prompts: bool = False, inline_proofread: bool = False) -> PassChoice:
    """The way a new task runs (see the module docstring); a judge on where neither the folded single pass nor the
    split can run is refused. ``tracked_prompts``: the cell has a tracked set and the rows can be judged by it.
    A correct-only task runs pass 3 only when asked (`on`): its prompt proofreads a translation.
    ``inline_proofread``: each window is proofread in its own conversation before it is committed, as resolved by
    `inline_proofread_choice`; never in three passes, whose pass 2 it is not written for."""
    if speaker_judge == "off":
        mode, reason = SINGLE, "speaker judge off"
    elif folds_judge(difficulty, speaker_judge, tracked_prompts):
        mode, reason = SINGLE_JUDGED, f"{difficulty}, speaker judge {speaker_judge} folded into the single pass"
    else:
        no_split = _no_split(difficulty, grouped_prompts)
        if no_split:
            raise _judge_refused(speaker_judge, no_split)
        mode, reason = THREE_PASS, f"{difficulty}, speaker judge {speaker_judge}"
    run_proofread = proofread == "on" or (
        proofread == "auto" and difficulty in PROOFREAD_DIFFICULTIES and not correct_only
    )
    return PassChoice(mode, run_proofread, reason, speaker_judge, inline_proofread=inline_proofread
                      and mode != THREE_PASS)


def refuse_judge(profile: Any, speaker_judge: str, recorded: PassChoice | None,
                 stable_json: str | Path | None = None) -> None:
    """Refuse the speaker judge where the task can run neither the folded single pass nor the three-pass split it
    needs: efficiency, or a correction cell without a grouped prompt set (owner 2026-10-30). A resumed task whose
    record names this same judge passes the first check: it follows the way it recorded. `recorded` is that record,
    None for a new task or `--no-resume`. The pipeline asks before ASR when the run reaches the LLM stage; the
    correction task asks again before its finished-task shortcuts (`run_correction_task`), and `task_choice` once more.

    A task that runs (or would run) single-judged is refused too when a sidecar its windows did not write is beside
    the stable (``stable_json``; `speaker_book.foreign_sidecar`, owner Q11): that is another judge's spent result,
    which the windows would overwrite."""
    if speaker_judge == "off":
        return
    if recorded is None or recorded.speaker_judge != speaker_judge:
        folds = folds_judge(profile.difficulty, speaker_judge, tracked_prompts(profile))
        no_split = None if folds else _no_split(profile.difficulty, profile.difficulty in THREE_PASS_DIFFICULTIES
                                                and grouped_prompts(profile))
        if no_split:
            raise _judge_refused(speaker_judge, no_split)
    single_judged = recorded.mode == SINGLE_JUDGED if recorded is not None else \
        folds_judge(profile.difficulty, speaker_judge, tracked_prompts(profile))
    if single_judged and stable_json is not None:
        from ...subtitles.stable_rows import read_rows
        from .correction.speaker_book import foreign_sidecar

        foreign = foreign_sidecar(stable_json)
        # A new task whose rows the single pass cannot judge runs three passes (`task_choice`), which reuse it;
        # rows that do not read (the stable gone) leave the refusal, which names the ways out.
        try:
            judgeable = foreign is None or recorded is not None or tracked_rows(read_rows(stable_json))
        except (OSError, ValueError):
            judgeable = True
        if foreign is not None and judgeable:
            raise CorrectionRequestError(
                f"{foreign} holds another speaker judge's results (three passes' pass 1, or a separate judge). At "
                "intermediate the text speaker judge folds into the correction windows, which write that same file. "
                "Either delete it, or use --llm-difficulty quality (three passes, which reuse it)."
            )


def grouped_prompts(profile: Any) -> bool:
    """Whether every prompt set the correction cell may answer with has a grouped version (`_every_cell_set`)."""
    from ..prompt_variants import has_grouped_variant

    return _every_cell_set(profile, has_grouped_variant)


def tracked_prompts(profile: Any) -> bool:
    """Whether every prompt set the correction cell may answer with has a tracked version, one that also names who
    speaks (`_every_cell_set`)."""
    from ..prompt_variants import has_tracked_variant

    return _every_cell_set(profile, has_tracked_variant)


def _every_cell_set(profile: Any, has: Callable[[str], bool]) -> bool:
    """Whether ``has`` holds for every prompt set the correction cell may answer with -- the cell's variant and each
    model's override, in the task's mode (a correct-only task's are the correct-only sets). False when there is no
    route declaration to ask."""
    from ..prompt_variants import correct_only_name
    from ..routing.capabilities import correction_task_group
    from ..routing.config import role_config_for

    try:
        config = role_config_for(correction_task_group(profile), profile.difficulty)
    except Exception:  # noqa: BLE001 - no declaration: the single pass needs none
        return False
    names = [config.variant, *config.variant_overrides.values()]
    if profile.correct_only:
        names = [correct_only_name(name) for name in names]
    return all(name and has(name) for name in names)


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
                source_language: str | None = None, warn: Callable[[str], None] = lambda _message: None) -> PassChoice:
    """The way this task runs: the recorded choice when resuming; one pass for a task whose windows started
    before choices were recorded; else a new choice (`record_choice` writes it), carrying the task's mode
    (`request.profile.correct_only`) and ``source_language``. A judge `refuse_judge` turns down is refused
    first. Said once through ``warn``: a resumed task whose switches would now choose another way with the judge on
    (it follows its record), an inline proofread asked for where it does not run (`inline_proofread_choice`), an audio
    judge folded into a window that reads text only (`--llm-media` decides, owner 2026-11-11), and a correct-only
    task folding the judge (`_say_single_judged`)."""
    resuming = request.resume and artifact_dir is not None
    recorded = recorded_choice(artifact_dir) if resuming else None
    refuse_judge(request.profile, request.speaker_judge, recorded, request.stable_json)
    tracked = tracked_prompts(request.profile) and tracked_rows(rows)
    if recorded is not None:
        now = SINGLE_JUDGED if folds_judge(request.profile.difficulty, request.speaker_judge, tracked) else THREE_PASS
        if request.speaker_judge != "off" and recorded.mode != SINGLE and now != recorded.mode:
            warn(f"这个任务记的是 {recorded.mode}（说话人裁判 {recorded.speaker_judge}），这次的开关会走 {now}；"
                 "续跑照记录走，要按这次的开关重选就加 --no-resume")
        if recorded.mode == SINGLE_JUDGED and not tracked_prompts(request.profile):
            from ..prompt_variants import fallback_tracked

            fallback = fallback_tracked(request.profile.correct_only).name
            warn("纠错这一格没有兼认人的 prompt（续跑换了档位或模型）：这个任务照记录一步纠错兼认人，"
                 f"没做完的窗口仍用 {fallback}，由这一格的模型答")
        _say_single_judged(recorded, recorded.speaker_judge, request.profile, warn)
        return recorded
    if resuming and started(artifact_dir):
        return PassChoice(SINGLE, request.proofread == "on", "the task started before three passes")
    wanted = inline_proofread_choice(request, warn=warn)
    choice = choose(difficulty=request.profile.difficulty, grouped_prompts=grouped_prompts(request.profile),
                    speaker_judge=request.speaker_judge, proofread=request.proofread,
                    correct_only=request.profile.correct_only, tracked_prompts=tracked, inline_proofread=wanted)
    if wanted and not choice.inline_proofread and request.inline_proofread == "on":
        warn("三个 pass 不做窗内校对（--llm-proofread-inline on 只用于单 pass）；要校对用 --llm-proofread on")
    _say_single_judged(choice, request.speaker_judge, request.profile, warn)
    return replace(choice, correct_only=request.profile.correct_only, source_language=source_language)


def _say_single_judged(choice: PassChoice, speaker_judge: str | None, profile: Any, warn: Callable[[str], None]) -> None:
    """What a single-judged task is told once a run, new task or resumed: an audio judge folded into windows that
    read text only (`--llm-media text`, which decides; owner 2026-11-11); a judge, text or audio, folded into windows
    that hear (it judges hearing too, no better than reading; owner 2026-11-15); and, for a correct-only task, that
    its tracked set trailed three passes on the Chinese acceptance -- the route is kept so that intermediate means one
    pass in either mode, with quality the way to three passes (owner 2026-11-12; docs/llm_design_notes.md
    「intermediate 单 pass 兼认人与窗内校对的决策记录」)."""
    if choice.mode != SINGLE_JUDGED:
        return
    if profile.correct_only:
        warn("中文素材（只纠错）走一步纠错兼认人：量过的中文素材上认人不如三个 pass，会议素材上有的模型比不开说话人裁判"
             "还差；要更稳用 --llm-difficulty quality（走三个 pass）")
    if speaker_judge == "audio" and profile.correction_media == "text":
        warn("--diarize judge-audio 与 --llm-media text 不一致：intermediate 的一步纠错兼认人以 --llm-media 为准，"
             "这次按纯文本认人")
    elif speaker_judge in ("text", "audio") and profile.correction_media != "text":
        warn("intermediate 的一步纠错兼认人这次带音频认人，不一定比纯文本好；要纠错带音频、认人用纯文本，"
             "改用 --llm-difficulty quality --diarize judge-text（三个 pass）")


#: The difficulties whose `--llm-proofread-inline auto` proofreads each window (owner 2026-11-11, on the corrected
#: cost readout): intermediate, whose `auto` runs no pass 3. Quality's `auto` runs pass 3 and stays without it.
INLINE_PROOFREAD_DIFFICULTIES = frozenset({"intermediate"})


def inline_proofread_choice(request: Any, *, warn: Callable[[str], None]) -> bool:
    """Whether a new task asks each window for an inline proofread (`--llm-proofread-inline`): `on` where the cell
    serves capableC alone (owner Q6: `on` opens on capableC's single pass only; a correct-only task's set is not
    capableC's) -- elsewhere it is said through ``warn`` and does not run; `auto` the same at
    `INLINE_PROOFREAD_DIFFICULTIES`, silently nowhere else; `off` never. Three passes drop it either way (`choose`)."""
    asked = getattr(request, "inline_proofread", "off")
    if asked == "auto":
        return request.profile.difficulty in INLINE_PROOFREAD_DIFFICULTIES and inline_proofread_cell(request.profile)
    if asked != "on":
        return False
    if not inline_proofread_cell(request.profile):
        warn("窗内校对只用于 capableC 的单 pass（quality / intermediate 的出厂单元格）；这一格不是，--llm-proofread-inline on 不起作用")
        return False
    return True


def inline_proofread_cell(profile: Any) -> bool:
    """Whether every prompt set the correction cell may answer with is capableC (the set the inline proofread was
    measured on; its tracked set is served on a single-judged task's windows). False for a correct-only task."""
    from ..routing.capabilities import correction_task_group
    from ..routing.config import role_config_for

    if profile.correct_only:
        return False
    try:
        config = role_config_for(correction_task_group(profile), profile.difficulty)
    except Exception:  # noqa: BLE001 - no declaration: nothing to ask
        return False
    return {config.variant, *config.variant_overrides.values()} == {"capableC"}


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

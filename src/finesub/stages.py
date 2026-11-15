"""The conversion itself: one source, stage by stage.

    source audio -> vocal separation -> VAD + ASR alignment -> stabilization -> SRT
    (-> LLM correction/translation, when the target stage asks for it)

`run_pipeline` is the whole contract: it derives every artifact path from the
final SRT, skips a stage whose output already exists, and records what it did in
the run metadata sidecar. It knows nothing about command lines, manifests or how
many sources are in flight -- `pipeline.py` is the front end that builds requests
and presents results. It imports this module, never the reverse: that direction
is also what keeps `python -m finesub.pipeline` from executing its own module a
second time under a different name.

Not to be confused with `llm/stages/`, which are the stages *inside* the LLM
correction step -- one stage of this one.
"""

from __future__ import annotations

from collections.abc import Callable
import glob
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence

from . import settings
from .reporting import current_reporter
from .speech.joint import stage as joint_stage
from .speech.postprocessing import stabilization as asr_stabilize
from .speech.runtime.resources import AUTO_GPU_TIER
from .run_metadata import (
    metadata_path_for_output,
    stage_record,
    summarize_llm_rounds,
    update_run_metadata,
)
from .subtitles import rendering as to_srt
from .speech.recognition.asr_backend import DEFAULT_MODEL
from .speech.recognition.segments import DEFAULT_GAP_SEC


class PipelinePaths(NamedTuple):
    vocal_audio: Path
    vad_json: Path
    vad_energy_npz: Path
    aligned_json: Path
    stable_json: Path
    #: The speaker timeline. Its own artifact because it is the one thing here
    #: that a rerun may not reproduce: the same audio down the same numerical
    #: path is bit-reproducible, but a changed input or a changed path gives a
    #: different-but-equally-valid answer, because a hard argmax turns last-bit
    #: float differences into flipped labels (`docs/diarization-findings.md`
    #: 5.1 and 5.2). So it is cached rather than recomputed, and `metadata.backend`
    #: records which backend made it -- provenance, not a compatibility key.
    #: It reads only the vocal audio; the labels derived from it live in `diar_rows_json`.
    diar_json: Path
    #: What the timeline says about each line of `stable_json`: speakers, confidence marks, the row
    #: splitting's cuts and the task's choice (`subtitles/stable_rows.py`). A record: written once, and
    #: again only when gone or made for other lines; `stable_json` itself is never rewritten.
    diar_rows_json: Path
    #: The LLM speaker judge's verdicts (`speech/diarization/judgments.py`): a
    #: record like `diar_json`, read over the acoustic labels by the `spk` column
    #: and the per-speaker exports, never folded into `stable_json` or `diar_rows_json`.
    speaker_judge_json: Path
    raw_srt: Path
    translated_srt: Path
    final_srt: Path
    task_artifact_dir: Path
    metadata_json: Path
    #: A cloud backend's raw answers, one file per call (`speech/joint/cache.py`): kept, since they cost quota.
    cloud_dir: Path

    @property
    def srt(self) -> Path:
        return self.final_srt

    def resolve_vocal_audio(self) -> Path:
        """Return the existing vocal file, preferring the ASR delivery.

        The pipeline always asks separation for `.ogg` (16 kHz mono, what every
        reader of it resamples to anyway). A `.flac` beside it is the lossless
        delivery -- separation's other mode, produced by a direct call -- and
        reading it is equally valid, just larger.
        """
        if self.vocal_audio.exists():
            return self.vocal_audio
        flac_fallback = self.vocal_audio.with_suffix(".flac")
        if flac_fallback.exists():
            return flac_fallback
        return self.vocal_audio


#: `--diarize`: off; the acoustic labels (and confidence marks); those plus the LLM speaker judge,
#: reading the rows or also hearing them (owner 2026-10-07, 2026-10-09: one switch, four levels).
DIARIZE_LEVELS = ("off", "acoustics", "judge-text", "judge-audio")


def judge_of(diarize: str) -> str:
    """The speaker judge a `--diarize` level asks for: `off`, `text` or `audio`."""
    return diarize.removeprefix("judge-") if diarize.startswith("judge-") else "off"


def split_choice(paths: "PipelinePaths", diarize_split: bool) -> bool:
    """The choice `-diar-rows.json` records when it is written: `--diarize-split`, unless the task already
    has products counting its current lines -- a raw SRT, a correction finished or under way (window
    checkpoints, research context), a speaker-judge sidecar -- which would no longer line up."""
    annotated = paths.final_srt.with_name(f"{paths.final_srt.stem}-annotated.csv")
    made = (paths.raw_srt, paths.translated_srt, paths.final_srt, annotated, paths.speaker_judge_json)
    started = paths.task_artifact_dir.is_dir() and any(paths.task_artifact_dir.iterdir())
    return diarize_split and not started and not any(path.exists() for path in made)


def default_diarize(stage: str, profile: Any) -> str:
    """`--diarize` for a run that leaves it unset (owner 2026-10-17): a run that reaches the LLM stage at
    `quality` (`DEFAULT_JUDGE_DIFFICULTIES`) gets the speaker judge in the correction's modality -- `judge-text` on
    text, `judge-audio` otherwise, which judges the rows when no model in the group hears -- and every other
    run `off`, so a run that stops at raw-srt downloads no model and its SRT is unchanged. There is no
    one-speaker step down to `acoustics`: the acoustic layer splits one voice into several tracks (one
    streamer with game audio came out as three), and the judge is what can hear it is one."""
    if PIPELINE_STAGE_ORDER[stage] < PIPELINE_STAGE_ORDER["translated-srt"]:
        return "off"
    from .llm.correction_mode import DEFAULT_JUDGE_DIFFICULTIES

    if profile.difficulty not in DEFAULT_JUDGE_DIFFICULTIES:
        return "off"
    return "judge-text" if profile.correction_media == "text" else "judge-audio"


def task_diarize(paths: "PipelinePaths", diarize: str | None, default: str, *, resume: bool = True) -> str:
    """The level a run uses: the one asked for, else what the task's LLM stage ran with -- the speaker judge
    recorded in `correction-mode.json` (`llm.correction_mode`), acoustic labels standing for themselves -- else
    `default`. A task whose LLM stage got as far as its windows or its output without that record started
    before the default: `off`, so a new default never adds spending to it (README_DEV「复用的依据是任务身份」).
    `--no-resume` chooses again, as the correction does.

    `run_pipeline` takes `default` from a batch item when the item gave one (`_diarize_default`): the
    item's ASR half stops at raw-srt and must still diarize when the item goes on to the LLM stage."""
    from .llm.correction_mode import recorded_choice, started

    if diarize is not None:
        if diarize not in DIARIZE_LEVELS:
            raise ValueError(f"--diarize takes {'/'.join(DIARIZE_LEVELS)}, got {diarize!r}")
        return diarize
    if not resume:
        return default
    recorded = recorded_choice(paths.task_artifact_dir)
    if recorded is not None and recorded.speaker_judge is not None:
        if recorded.speaker_judge != "off":
            return f"judge-{recorded.speaker_judge}"
        return "acoustics" if paths.diar_json.exists() else "off"
    if recorded is not None or started(paths.task_artifact_dir) \
            or paths.translated_srt.exists() or paths.final_srt.exists():
        return "off"
    return default


PIPELINE_STAGE_ORDER = {
    "vocal": 1,
    "aligned": 2,
    "stable": 3,
    "raw-srt": 4,
    "translated-srt": 5,
    "final-srt": 6,
}

_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts",
})


def profile_asr_workers(items: Sequence[Mapping[str, Any]]) -> int:
    """One ASR task at a time; the file itself supplies the parallelism.

    Concurrency used to come from the asr bin running ``wt_instances`` files
    side by side, each pinned to one shard. That kept the GPU busy but put N
    files' worth of non-model state in one process at once, and the 8GB profile
    was already measured over its RAM budget with a single file. Running one
    file with full shard and separator width bounds the live state to that file
    while keeping the same number of models resident.

    Reverses an earlier file-level design; docs/wt-parallelism.md records that
    trade, and the intra-file sharding it refers to has since been removed too
    (ASR is single-worker).

    ``items`` is unused: the answer no longer depends on the profile mix. It
    stays in the signature because callers pass the merged rows.
    """

    return 1


def default_output_path(input_path: Path) -> Path:
    # Group every artifact of one input under out/<stem>/ so a run's outputs
    # stay together instead of scattering across out/.
    return Path("out") / input_path.stem / f"{input_path.stem}.srt"


def resolve_llm_media_for_source(
    source_path: Path,
    *,
    stage: str,
    llm_media: str,
    llm_video: str | Path | None,
) -> tuple[str, str | Path | None, str]:
    """Pick the effective LLM media switch / video file for a local input.

    A video switch needs a video track: a local video file becomes the default
    ``llm_video``. When the input is audio-only, ``--llm-media video``
    downgrades to audio with a notice. Only meaningful once an LLM stage
    actually runs, so earlier stages are left untouched. Returns
    ``(llm_media, llm_video, notice)``; an empty notice means silence.
    """
    if PIPELINE_STAGE_ORDER[stage] < PIPELINE_STAGE_ORDER["translated-srt"]:
        return llm_media, llm_video, ""
    if llm_media != "video":
        return llm_media, llm_video, ""
    if source_path.suffix.lower() in _VIDEO_EXTENSIONS:
        return llm_media, llm_video or source_path, ""
    if llm_video:
        # Audio-only ASR source, but the caller supplied the video separately
        # -- that is exactly what --llm-video is for (transcribe the .wav,
        # show the .mp4 to the LLM). Availability is about the video *source*,
        # not about the ASR input's suffix.
        return llm_media, llm_video, ""
    return (
        "audio",
        llm_video,
        "Note: input is audio-only; downgrading --llm-media video → audio.",
    )


def default_pipeline_paths(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> PipelinePaths:
    input_path = Path(input_path).expanduser()
    srt_path = Path(output_path).expanduser() if output_path else default_output_path(input_path)
    if srt_path.suffix == "":
        srt_path = srt_path.with_suffix(".srt")
    base = srt_path.with_suffix("")
    return PipelinePaths(
        vocal_audio=base.with_name(f"{base.name}-vocal.ogg"),
        vad_json=base.with_name(f"{base.name}-vad.json"),
        vad_energy_npz=base.with_name(f"{base.name}-vad-energy.npz"),
        aligned_json=base.with_name(f"{base.name}-aligned.json"),
        stable_json=base.with_name(f"{base.name}-stable.json"),
        diar_json=base.with_name(f"{base.name}-diar.json"),
        diar_rows_json=base.with_name(f"{base.name}-diar-rows.json"),
        speaker_judge_json=base.with_name(f"{base.name}-speaker-judge.json"),
        raw_srt=base.with_name(f"{base.name}-raw.srt"),
        translated_srt=base.with_name(f"{base.name}-translated.srt"),
        final_srt=srt_path,
        task_artifact_dir=base.with_name(f"{base.name}.llm-artifacts"),
        metadata_json=metadata_path_for_output(srt_path),
        cloud_dir=base.with_name(f"{base.name}-cloud"),
    )


#: How long a stage's temporary file may sit unchanged before it counts as left
#: behind by a run that was killed outright. A live writer keeps touching its
#: file; the margin is for one long write between touches.
STALE_TEMPORARY_SECONDS = 3600.0


def _sweep_stale_temporaries(path: Path) -> None:
    """Remove temporaries of `path` that a killed run left behind.

    Their names are per writer, so a new run cannot just reuse and clear the
    one name; it would not know whether another process is still writing.
    Age decides instead. Reached on the skip path too: a run that crashed and
    was rerun at once finds its leftover still fresh.
    """

    cutoff = time.time() - STALE_TEMPORARY_SECONDS
    pattern = f".{glob.escape(path.stem)}.part-*{glob.escape(path.suffix)}"
    for leftover in path.parent.glob(pattern):
        try:
            if leftover.stat().st_mtime < cutoff:
                leftover.unlink()
        except OSError:
            pass  # gone already, or still open on Windows: not a leftover


def _temporary_path(path: Path) -> Path:
    """Where one writer builds `path`: unique per writer.

    Two runs of the same source (two terminals, a raw-srt run beside a
    final-srt one) shared one fixed name: each start unlinked the other's
    file, and one run published the other's half-written one. Now each writes
    its own and the last replace wins, whole either way.
    """

    return path.with_name(
        f".{path.stem}.part-{os.getpid()}-{secrets.token_hex(4)}{path.suffix}"
    )


def _use_or_create(
    path: Path,
    step_name: str,
    create: Callable[[Path], str | Path],
) -> Path:
    _sweep_stale_temporaries(path)
    if path.exists():
        # The stage line already says "已有结果，跳过"; the path itself is
        # detail for someone diagnosing which artifact was picked up.
        current_reporter().debug(
            f"skipping {step_name}", {"existing": str(path)}
        )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        produced = Path(create(temporary))
        if not produced.is_file():
            raise RuntimeError(
                f"{step_name} did not create its expected output: {produced}"
            )
        os.replace(produced, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


class _StageClock(NamedTuple):
    """Wall and CPU time taken at the same instant.

    Both, because the interesting quantity is the pair. See
    `run_metadata.stage_record` for why the ratio is telemetry and not a
    stall verdict.
    """

    wall: float
    cpu: float


def _stage_clock() -> _StageClock:
    return _StageClock(time.perf_counter(), time.process_time())


def _record_stage_time(
    paths: "PipelinePaths",
    stage_timing: dict[str, Any],
    name: str,
    started: _StageClock,
    *,
    status: str,
) -> None:
    """Write one stage's elapsed time to the sidecar, and say it under verbose.

    Every stage, not only the ones that happened to grow their own timing
    block: `verbose` promises timing, and a run where two of four stages report
    none leaves the reader to work out the difference by subtraction.

    Reused stages are recorded without an elapsed time rather than with zero --
    "did not run" and "ran instantly" are different facts.

    `skipped` is a third fact and not a synonym for either: the stage's real
    work was not asked for (`--no-separate`), but something still ran in its
    place, so it keeps its elapsed time. Asking the sidecar "did this run
    separate?" is the whole reason that answer may not be folded into
    `reused` -- that is the field a bug report is read for.
    """

    ran = status != "reused"
    now = _stage_clock()
    elapsed = now.wall - started.wall if ran else None
    cpu = now.cpu - started.cpu if ran else None
    update_run_metadata(
        paths.metadata_json,
        {
            "timing": {
                "stages": {
                    name: _stage_record_for_current_run(
                        stage_timing,
                        name,
                        stage_record(
                            status=status,
                            elapsed_sec=elapsed,
                            cpu_sec=cpu,
                        ),
                    )
                }
            }
        },
    )
    if elapsed is not None:
        current_reporter().debug(
            "stage timing", {"stage": name, "elapsed_sec": f"{elapsed:.3f}"}
        )


def _stage_record_for_current_run(
    prior_stages: dict[str, Any],
    name: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Keep an executed stage when a later pass of the same run reuses it.

    `skipped` counts as having run for this purpose: it is a pass that did
    work and left an artifact behind, so a later pass finding that artifact
    must not overwrite the record of how it got there.
    """

    prior = prior_stages.get(name)
    if (
        record.get("status") == "reused"
        and isinstance(prior, Mapping)
        and prior.get("status") in ("executed", "skipped")
    ):
        chosen = dict(prior)
    else:
        chosen = record
    prior_stages[name] = chosen
    return chosen


def _validate_llm_configuration(
    *,
    knowledge_root: str | Path | None,
    style: str | None,
    style_mode: str | None,
    llm_difficulty: str,
) -> None:
    """Everything about the LLM stage that is knowable before ASR runs.

    Not an optimisation. Every check below is a configuration fact -- a
    malformed `[chunking]` cap, a typo in `--style`, a bound model group whose
    members cannot hold a window -- and the correction stage finds them only
    after the expensive half of the run has already happened. A run that was
    going to fail should fail in the first second.
    """

    from finesub.llm.knowledge.base import DEFAULT_KNOWLEDGE_ROOT
    from finesub.llm.knowledge.style import resolve_style_keys, resolve_style_selection
    from finesub.llm.routing.capabilities import check_model_group_windows

    # Read again where they are used; checked here only for the early failure.
    settings.MAX_WINDOW_SUBTITLE_TOKENS.resolve()
    settings.RESEARCH_TRANSCRIPT_MAX_TOKENS.resolve()
    check_model_group_windows()
    root = knowledge_root or DEFAULT_KNOWLEDGE_ROOT
    names = resolve_style_selection(
        style, style_mode, knowledge_root=root, difficulty=llm_difficulty
    ).names
    if names:
        resolve_style_keys(root, names)


def _refuse_judge_without_split(paths: PipelinePaths, profile: Any, speaker_judge: str, *, resume: bool) -> None:
    """A speaker judge the correction cannot run the three-pass split for (`three_pass.refuse_judge`), refused
    before diarization spends GPU time and model downloads on it. Only for a run that reaches the LLM stage:
    the correction routing it reads is never loaded for one that stops at raw-srt (a batch item's ASR half
    included). A resume whose recorded judge is this one passes, as the correction task does."""
    if speaker_judge == "off":
        return
    from finesub.llm.correction_mode import recorded_choice
    from finesub.llm.stages.three_pass import refuse_judge

    refuse_judge(profile, speaker_judge, recorded_choice(paths.task_artifact_dir) if resume else None)


def _correction_request(
    paths: PipelinePaths,
    source_path: Path,
    stage: str,
    *,
    axes: tuple[str, str, str, str, float],
    llm_video: str | Path | None,
    refined_srt: str | Path | None,
    task_id: str,
    postprocess_profile: int,
    joint: bool,
    **options: Any,
) -> Any:
    """``run_pipeline``'s LLM keys as the one value the correction task takes,
    checked. The keys that pass through unchanged arrive in ``options`` under
    ``CorrectionRequest``'s own names, so a new one costs a line here only if
    the pipeline has to translate it."""

    from finesub.llm.correction_request import CorrectionRequest, resolved_request
    from finesub.llm.routing.profiles import resolve_profile

    media, retrieval, difficulty, continuity, output_scale = axes
    profile = resolve_profile(media, retrieval, difficulty, continuity, output_scale=output_scale)
    if llm_video and not profile.uses_video:
        raise ValueError("--llm-video only applies when a media switch is video")
    if profile.uses_video and not llm_video:
        raise ValueError("--llm-video is required when a media switch is video")
    request = resolved_request(
        CorrectionRequest(
            stable_json=paths.stable_json,
            output_path=paths.final_srt,
            audio_path=source_path,
            video_path=Path(llm_video) if llm_video else None,
            profile=profile,
            refined_srt=Path(refined_srt) if refined_srt else None,
            task_id=task_id or paths.final_srt.stem,
            task_artifact_dir=paths.task_artifact_dir,
            # What the speaker judge listens to (separation may not have run yet); a joint-ASR task has no vocal
            # track, and its labels came from the mix.
            vocal_audio=source_path if joint else paths.resolve_vocal_audio(),
            # Targeting translated-srt stops before the final postprocess.
            postprocess_profile=None if stage == "translated-srt" else postprocess_profile,
            **options,
        )
    )
    _refuse_judge_without_split(paths, request.profile, request.speaker_judge, resume=request.resume)
    return request


def _run_speaker_export_stage(
    paths: PipelinePaths,
    *,
    speaker_tracks: str,
    empty_cues: bool,
    announce: Callable[..., None],
) -> None:
    """The lossless exports: one SRT per speaker, and/or one ASS holding all.

    Explicitly requested, never automatic -- the single-track SRT is the product
    and these are additions to it (owner, 2026-09-05; the one way diarization
    changes that SRT is acoustic row splitting, owner 2026-10-14). A single track
    has to make a choice about overlapping speech; these do not.

    The acceptance here is the completeness invariant, not the labelling: a line
    written into one track too many costs a deletion, a line that reached no
    track has to be noticed, found and retyped. So a broken invariant is a
    warning on every run, not a silent shortfall.

    The speaker judge's verdicts apply when its sidecar is there and still
    belongs to this stable: the lines are the correction's (`speech.diarization.product`),
    a moved line goes to its new track, a split row's second half and a line the
    judge heard missing to theirs, and a track the judge opened gets its own file. Called after
    raw-srt and again after the LLM stage, so the files always follow the
    sidecar (the second call is `run_pipeline`'s, after the task whose judge
    the raw-srt export could not see yet); every file an earlier export wrote goes first, whichever format
    it was in, so a track that no longer exists leaves nothing behind.
    """

    if speaker_tracks not in ("off", "srt", "ass", "both"):
        raise ValueError(f"--speaker-tracks takes off/srt/ass/both, got {speaker_tracks!r}")
    if speaker_tracks == "off":
        return

    from .speech.diarization import export as speaker_export
    from .speech.diarization.labels import filled_track, speaker_of
    from .speech.diarization.product import read_product

    if not paths.stable_json.exists():
        return
    layer = read_product(paths.stable_json)
    subtitles = layer.rows()
    speakers = [speaker_of(segment) for segment in subtitles]
    if not any(speaker.track is not None or filled_track(speaker) is not None for speaker in speakers):
        print(  # product output
            "Warning: --speaker-tracks needs speaker labels; run with --diarize acoustics (or a judge level)",
            file=sys.stderr,
        )
        announce("speaker-tracks", "skipped", detail="没有说话人标签")
        return

    announce("speaker-tracks", "running")
    speakers = [layer.judged.get(line.id, speaker) for line, speaker in zip(layer.lines, speakers)]
    destination = paths.stable_json.with_name(f"{paths.stable_json.name.removesuffix('-stable.json')}-speakers")
    destination.mkdir(parents=True, exist_ok=True)
    for stale in (*destination.glob(f"{destination.name}-*.srt"), destination / f"{destination.name}.ass"):
        stale.unlink(missing_ok=True)
    groups = speaker_export.group_by_speaker(subtitles, speakers)
    missing = speaker_export.check_completeness(subtitles, groups)
    if missing:
        print(  # product output
            f"Warning: {len(missing)} subtitle line(s) reached no speaker track "
            f"(first at index {missing[0]}); the tracks are incomplete",
            file=sys.stderr,
        )

    written = []
    if speaker_tracks in ("srt", "both"):
        for name, text in speaker_export.render_speaker_srts(subtitles, speakers, empty_cues=empty_cues).items():
            target = destination / f"{destination.name}-{name}.srt"
            target.write_text(text, encoding="utf-8")
            written.append(target.name)
    if speaker_tracks in ("ass", "both"):
        target = destination / f"{destination.name}.ass"
        target.write_text(speaker_export.render_ass(subtitles, speakers), encoding="utf-8")
        written.append(target.name)
    current_reporter().debug(
        "speaker tracks", {"directory": str(destination), "files": len(written)}
    )


def _planned_stages(target_order: int, *, diarize: str) -> list[str]:
    """Which stage lines a person should expect to see, reused ones included.

    Diarization is not a rung on the `--stage` ladder -- that ladder names which
    *product* to stop at, and speaker labels are an addition to the subtitles
    rather than another kind of them. It still gets its own line when it runs,
    because it costs GPU minutes and a watcher should see where they went.
    """

    planned = [name for name, order in PIPELINE_STAGE_ORDER.items() if order <= target_order]
    if diarize != "off" and "stable" in planned:
        planned.insert(planned.index("stable") + 1, "diarize")
    return planned


def _warn_on_backend_mismatch(diar_json, requested, diarization) -> None:
    """Say so when the cached timeline came from a different backend.

    Never fatal and never a reason to recompute -- see the call site. A bad read
    is not worth failing over either: the point is the label, and an unreadable
    cache produces its own error where it is actually used.

    ⚠ **Only a `Warning:` when a backend was NAMED.** Compared against the
    resolved backend either way, but the default has moved (Sortformer, then a
    probe, then DiariZen on 2026-09-21, then Nemotron-3 on 2026-09-25), and every
    older task's resume would otherwise print this for a run where nothing on the
    command line named a backend. That
    is the case the resume rule covers -- current parameters do not retroactively
    redefine finished stages -- so an implicit mismatch is reported where run
    detail lives, not on stderr.
    """
    try:
        resolved = diarization.resolve_backend(requested).name
    except ValueError:  # an unknown name is reported where it is resolved for real
        return
    try:
        cached = diarization.read_timeline(diar_json).backend
    except Exception:  # the consumer reports an unreadable timeline properly
        return
    if cached == resolved:
        return
    note = (f"{diar_json.name} was produced by the {cached} backend, not {resolved}; "
            "it is reused as-is. Delete it to recompute.")
    if requested is None:
        current_reporter().debug("diarization backend differs", {"cached": cached,
                                                                 "resolved": resolved})
        return
    print(f"Warning: {note}", file=sys.stderr)  # product output


def _recall_repair(paths: PipelinePaths, *, squeeze: str, gap: str, qwen_verify: str, device: str | None,
                   gpu_tier: str, profile: int, row_fill: str):
    """The stable step's recall repair hook (`speech/verification/recall_decode`), or None.

    Built only when the stable step runs (a reused `-stable.json` is kept as it
    is), and the devices asked for only once the hook is about to load a model.
    Whisper goes where the ASR step would put it -- CTranslate2 answers, under
    the run's tier -- and the referee follows it as the ASR step's tail referee
    does: the stable step may be reusing an aligned another run made, so the
    ASR step's resolved device is not at hand.
    """
    if profile != 0:
        return None
    from .speech.verification.recall_decode import recall_repair

    def devices() -> tuple[str, str]:
        from .speech.recognition.vad_asr_stage import tail_verify_device
        from .speech.runtime.device import resolve_asr_device
        from .speech.runtime.resources import get_resource_profile

        whisper = resolve_asr_device(device or "cuda", gpu_allowed=bool(get_resource_profile(gpu_tier).gpu))
        return whisper, tail_verify_device(whisper)

    vocal = paths.resolve_vocal_audio()
    return recall_repair(audio_path=vocal if vocal.exists() else None, squeeze=squeeze, gap=gap,
                         row_fill=row_fill, qwen_verify=qwen_verify, devices=devices, vad_prefix=paths.vad_json)


def _diarization_device(device: str | None, gpu_tier: str) -> str:
    """Where the timeline is computed: the run's device, unless the tier or the card says CPU.

    Every backend runs on CPU, so a run without a usable GPU still gets labels -- slowly. Measured
    once for the default (2026-09-25, 8 threads): Nemotron-3 + vbx RTF 0.20 on CPU (10 min of
    hina-04) against 0.009 on CUDA (all 59 min), and on another file the CPU chain differed from
    NeMo on the GPU by 0.19% of frames. Said on stderr, because an hour of audio then costs about
    twelve minutes. (DiariZen, research only: RTF 0.86.)
    """
    from .speech.runtime.device import resolve_device
    from .speech.runtime.resources import get_resource_profile

    requested = device or "cuda"
    if not get_resource_profile(gpu_tier).gpu:
        requested = "cpu"
    chosen = resolve_device(requested, context="speaker diarization")
    if not str(chosen).startswith("cuda"):
        current_reporter().warning(
            "diarization-on-cpu",
            "说话人分离在 CPU 上跑：约为音频时长的五分之一（1 小时音频约 12 分钟）",
            impact="只影响这一阶段的耗时",
        )
    return chosen


def warn_if_the_judge_cannot_run(diarize: str, stage: str) -> None:
    """A judge level whose item stops before the LLM stage keeps its labels unjudged: say so. Called once
    per item with its target stage (`pipeline._build_item`), never from `run_pipeline`: an item past raw-srt
    runs it twice, the first time only up to raw-srt."""
    if judge_of(diarize) == "off" or PIPELINE_STAGE_ORDER[stage] >= PIPELINE_STAGE_ORDER["translated-srt"]:
        return
    print(  # product output
        f"Warning: --diarize {diarize}: the LLM speaker judge runs in the LLM stage, and this run "
        f"stops at {stage}; the labels stay unjudged this time",
        file=sys.stderr,
    )
    current_reporter().warning("speaker-judge-not-run", f"--diarize {diarize} 的说话人裁判在 LLM 阶段，这次只跑到 {stage}",
                               impact="标签这次没经过裁判")


def _write_raw_srt(paths: PipelinePaths, *, word: bool, postprocess_profile: int, empty_cues: bool) -> None:
    """`-raw.srt` from the stable's lines as the task chose them (`subtitles.stable_rows`), skipped when it
    exists; the other version is rebuilt from the stable whenever wanted (owner 2026-10-14: one raw SRT)."""
    from .subtitles.model import warn_on_invalid_srt
    from .subtitles.postprocess import (
        TIMELINE_POSTPROCESS_PROFILES,
        postprocess_srt_file,
    )

    stable_audio_end = asr_stabilize.recorded_audio_end(paths.stable_json)

    def _create_raw_srt(temporary: Path) -> Path:
        # Every write below lands in `temporary`, and there are up to three
        # of them. Validating each would report one finding once per pass,
        # against a `.part` path the reader cannot open -- so the passes
        # stay quiet and the finished artifact is checked once, under the
        # name it will actually have.
        # Each cue's group key rides along: the passes keep cues apart only within one speaker.
        produced, keys = to_srt.convert_json_to_srt(
            paths.stable_json,
            output_path=temporary,
            word=word,
            validate=False,
            empty_cues=empty_cues,
        )
        if postprocess_profile in (0, 1):
            # Timeline half of the final profile; the ASR text stays as is.
            for timeline_profile in TIMELINE_POSTPROCESS_PROFILES:
                postprocess_srt_file(
                    produced, profile=timeline_profile, validate=False,
                    audio_end=stable_audio_end, keys=keys,
                )
        # Raw text is before layout: only lines far over the budget are
        # worth a mention, as one banded summary (see the validator).
        warn_on_invalid_srt(
            produced.read_text(encoding="utf-8"), where=str(paths.raw_srt),
            budget_stage="raw-srt", budget_above=35, keys=keys,
        )
        return produced

    _use_or_create(
        paths.raw_srt,
        "raw SRT export",
        _create_raw_srt,
    )


def _note_task_backend(paths: PipelinePaths, backend: str, requested: str | None, aligner: bool | None) -> None:
    """Say where a resumed task's recognition came from when this run asked for something else -- either way
    round: local products under `--asr <joint>` as much as joint products under `--asr local` -- and that
    an explicit `--asr-aligner` does nothing for this backend (`[asr] aligner` is a standing preference for
    the MOSS and 35t tasks, not worth a line on every other one). A batch item's two halves each call this, so a run
    past raw-srt says it twice; that is the price of saying it wherever `run_pipeline` is entered."""
    if aligner is True and backend not in (*joint_stage.SEGMENT_BACKENDS, joint_stage.GEMINI_35T):
        print(f"Warning: --asr-aligner only applies to moss-pro, moss-local and gemini-35t; ignored for {backend}",  # product output
              file=sys.stderr)
    if requested is None or not (paths.stable_json.exists() or paths.aligned_json.exists()):
        return
    if joint_stage.resolve_backend(requested) == backend:
        return
    print(  # product output
        f"Warning: this task's recognition came from {backend} and is reused; --asr {requested} applies only "
        "after deleting -aligned.json, -stable.json and everything after them",
        file=sys.stderr,
    )


def _backend_diarize(
    paths: PipelinePaths, backend: str, requested: str | None, diarize: str, aligner: bool | None,
) -> str:
    """The task's `--diarize` level for its backend, having said where a resumed task's recognition came from."""
    _note_task_backend(paths, backend, requested, aligner)
    # A joint-ASR task's own transcription labels its lines; writing them costs nothing (`docs/joint-asr.md`).
    return "acoustics" if backend != joint_stage.LOCAL and diarize == "off" else diarize


def _run_joint_aligned(
    paths: PipelinePaths, source_path: Path, *, backend: str, language: str | None, aligner: bool | None,
    repair: str | None, split_length_scale: float | None, gpu_tier: str, device: str | None,
    stage_timing: dict[str, Any], announce: Callable[..., None],
) -> None:
    """A joint-ASR task's `-aligned.json`: the joint transcription through production's sentence DP."""
    aligner = settings.ASR_ALIGNER.resolve(aligner)
    aligned_existed = paths.aligned_json.exists()
    announce("aligned", "reused" if aligned_existed else "running")
    asr_t0 = _stage_clock()
    _use_or_create(
        paths.aligned_json,
        "joint ASR",
        lambda temporary: joint_stage.run(
            source_path, backend=backend, output_path=temporary, cache_root=paths.cloud_dir, language=language,
            aligner=aligner, repair=repair, split_length_scale=split_length_scale, gpu_tier=gpu_tier, device=device,
        ),
    )
    _record_stage_time(paths, stage_timing, "asr", asr_t0, status="reused" if aligned_existed else "executed")


def _run_joint_labels(
    paths: PipelinePaths, *, backend: str, announce: Callable[..., None], diarize_backend: str | None,
) -> None:
    """A joint-ASR task's speaker labels: its own transcription's, from each stable line's
    `line_speaker`, at every `--diarize` level. `-diar.json` is neither read nor written; one already in the
    directory stays for the task's local recognition."""
    from .speech.diarization import stage as diarization

    if diarize_backend is not None:
        print(  # product output
            f"Warning: --diarize-backend {diarize_backend} is ignored: a joint-ASR task's speakers come from "
            f"{backend}'s own labels", file=sys.stderr)
    if diarization.rows_current(paths.stable_json):
        announce("diarize", "reused")
        return
    unused = "；目录里的声学时间线不用" if paths.diar_json.exists() else ""
    announce("diarize", "running", detail=f"说话人来自 {backend} 的标签{unused}")
    record = ((json.loads(paths.stable_json.read_text(encoding="utf-8")).get("metadata") or {})
              .get(joint_stage.METADATA_KEY) or {})
    diarization.write_joint_rows(paths.stable_json, backend=backend, model=str(record.get("model", backend)))


def _run_diarization_stage(
    paths: PipelinePaths,
    *,
    diarize: str,
    device: str | None,
    gpu_tier: str,
    announce: Callable[..., None],
    backend: str | None = None,
    joint_backend: str | None = None,
    max_speakers: int | None = None,
    diarize_split: bool,
) -> None:
    """Speaker timeline after stabilization, and what it says about each line, when asked for. A joint-ASR
    task (`joint_backend`) takes its own transcription's labels instead, at every level (`_run_joint_labels`).

    Two files with opposite properties, so two existence checks rather than one: the timeline
    (`-diar.json`) is cached because it cannot be reproduced; `-diar-rows.json` is cheap to compute
    from it, but it is written once too -- the products after it count its lines, and jitter is no
    reason to change them. It is written again only when gone or made for other lines (the stable was
    redone). A run that already has both does nothing at all.

    Failure never takes the run down. The subtitles are the product; speaker
    labels are an addition to them, and a missing model or an unusable GPU should
    cost the labels, not the subtitles -- with a `Warning:`. Labelling from a cached
    timeline needs neither the GPU nor the model. (There used to be an `auto` that
    ran "whenever this machine can"; owner 2026-09-21 removed it.)
    """

    if diarize not in DIARIZE_LEVELS:
        raise ValueError(f"--diarize takes {'/'.join(DIARIZE_LEVELS)}, got {diarize!r}")
    if joint_backend is not None:
        _run_joint_labels(paths, backend=joint_backend, announce=announce, diarize_backend=backend)
        return
    if diarize == "off":
        return

    from .speech.diarization import stage as diarization

    timeline_exists = paths.diar_json.exists()
    if timeline_exists:
        # Owner 2026-09-19: a cached timeline does NOT expire because the run
        # asked for a different backend. It is neither corrupt nor unreadable,
        # so backend identity is provenance and only warns -- the way to get the
        # other backend's answer is to delete the artifact, which is the rule
        # every stage already has. Warning rather than silence, because serving
        # X while the command line said Y is the part nobody would notice.
        _warn_on_backend_mismatch(paths.diar_json, backend, diarization)
        diarization.seed_legacy(paths.stable_json, paths.diar_json,  # ⚠ compatibility, to delete at 1.0
                                split_choice=split_choice(paths, diarize_split))
    if timeline_exists and diarization.rows_current(paths.stable_json):
        announce("diarize", "reused")
        return
    announce("diarize", "running")
    reporter = current_reporter()
    try:
        if not timeline_exists:
            from .speech.diarization.evidence import evidence_path

            result = diarization.run_diarization(
                paths.resolve_vocal_audio(), device=_diarization_device(device, gpu_tier),
                backend=backend,
                evidence_path=evidence_path(paths.diar_json), max_speakers=max_speakers,
            )
            diarization.write_timeline(result, paths.diar_json)
            reporter.debug(
                "diarization timeline",
                {
                    "backend": result.backend,
                    "tracks": len({segment.track for segment in result.tracks}),
                    "segments": len(result.tracks),
                    "segment_overlap_ratio": round(result.overlap, 4),
                },
            )
        diarization.write_rows(paths.stable_json, paths.diar_json, split_choice=split_choice(paths, diarize_split))
    except diarization.DiarizationUnavailable as error:
        # Documented degradation, not a crash: `Warning:` to stderr is the
        # repository's convention for "the run continues without this".
        print(f"Warning: speaker diarization skipped: {error}", file=sys.stderr)  # product output
        if judge_of(diarize) != "off":
            print(  # product output
                f"Warning: no speaker labels, so the LLM speaker judge ({diarize}) will not run either",
                file=sys.stderr,
            )
        announce("diarize", "skipped", detail="模型或依赖不可用")


def _run_vocal_stage(
    paths: "PipelinePaths",
    *,
    stage: str,
    target_order: int,
    aligned_required: bool,
    source_path: Path,
    separate: bool,
    gpu_tier: str,
    device: str,
    separator_sample_rate: int | None,
    stage_timing: dict[str, Any],
    announce: Callable[..., None],
    joint: bool = False,
) -> None:
    """The vocal stage: decide whether it is wanted, then deliver or skip it.

    Lifted out of `run_pipeline` when `--no-separate` landed (2026-09-03): the
    branch answers three separate questions -- is the track wanted at all, is
    one already here, and does this run separate or merely transcode -- and
    reading them inside a five-hundred-line caller made the last of the three
    look like a special case of the first two.

    `announce` is the caller's, because the `[n/N]` prefix belongs to the run's
    plan rather than to any one stage.
    """

    # Here, not at the top: the separator imports torch, and `pipeline --help`
    # imports this module.
    from .speech.preprocessing.separator import separation as vocal_separation

    reporter = current_reporter()
    if joint:
        # A joint backend transcribes the original mix: no vocal track is made.
        if target_order >= PIPELINE_STAGE_ORDER["vocal"]:
            announce("vocal", "skipped", detail="一体转写，读原始混音")
        return
    if target_order >= PIPELINE_STAGE_ORDER["vocal"]:
        # Vocal audio only feeds VAD-ASR. A stable JSON satisfies every later
        # consumer without requiring the aligned artifact to be backfilled.
        # When stable is absent, an existing aligned JSON likewise avoids both
        # vocal separation and VAD-ASR.
        vocal_needed = stage == "vocal" or (
            aligned_required and not paths.aligned_json.exists()
        )
        if vocal_needed:
            # Either delivery counts as done, which is why this asks
            # `resolve_vocal_audio` rather than looking for the shape this stage
            # happens to write. `.ogg` is what it asks separation for; a `.flac`
            # beside it is separation's lossless mode, and every reader below
            # takes it. Checking only the `.ogg` meant a run that already held
            # the lossless track paid for the whole GPU stage again to end up
            # with a smaller copy of what it had.
            existing_vocal = paths.resolve_vocal_audio()
            vocal_existed = existing_vocal.exists()
            if vocal_existed:
                vocal_status = "reused"
            elif separate:
                vocal_status = "executed"
            else:
                vocal_status = "skipped"
            announce(
                "vocal",
                vocal_status,
                # Something does run in the skipped case -- the same 16 kHz mono
                # delivery, made by transcoding instead of separating -- so this
                # is not `reused` and the line says which of the two it is.
                detail="" if vocal_status != "skipped" else "跳过分离，直接转出 ASR 轨",
            )
            vocal_t0 = _stage_clock()
            separator_metadata: dict[str, Any] = {}
            if vocal_existed:
                reporter.debug(
                    "skipping vocal separation", {"existing": str(existing_vocal)}
                )
            elif not separate:
                # README_DEV.md, 「Pipeline 复用规则」: the
                # stage's *contract* is honoured (a 16 kHz mono track at the
                # path everything downstream reads), only the separation itself
                # is not run. Every consumer, the existence-based skip and
                # resume therefore need no special case for "no vocal file".
                reporter.debug(
                    "vocal separation not requested; transcoding the source",
                    {"source": str(source_path)},
                )
                _use_or_create(
                    paths.vocal_audio,
                    "vocal delivery",
                    lambda temporary: vocal_separation.encode_asr_delivery(
                        source_path,
                        temporary,
                        # A compressed or video source is decoded to a
                        # temporary flac first, and a *failed* encode keeps it
                        # on purpose so a retry need not decode again. Only
                        # this record can name it for the cleanup afterwards.
                        run_metadata_path=paths.metadata_json,
                    ),
                )
            else:
                _use_or_create(
                    paths.vocal_audio,
                    "vocal separation",
                    lambda temporary: vocal_separation.run_vocal_separation(
                        source_path,
                        output_path=temporary,
                        gpu_tier=gpu_tier,
                        # The request, not a resolved device: separation
                        # decides for itself (torch), the ASR stage asks
                        # CTranslate2. What they share is the user's intent.
                        device=device,
                        separator_sample_rate=separator_sample_rate,
                        metadata_sink=separator_metadata,
                        run_metadata_path=paths.metadata_json,
                    ),
                )
            _record_stage_time(
                paths,
                stage_timing,
                "vocal_separation",
                vocal_t0,
                status=vocal_status,
            )
            if separator_metadata:
                update_run_metadata(
                    paths.metadata_json,
                    {"workers": {"vocal_separation": separator_metadata}},
                )
        elif not paths.resolve_vocal_audio().exists() and aligned_required:
            # Not "reused" in the strict sense -- the vocal audio is not there
            # and is not wanted. For the progress list the two are the same
            # thing: nothing runs, and the stage is behind us.
            announce("vocal", "reused")
            reporter.debug(
                "skipping vocal separation; a later artifact already exists",
                {
                    "existing": str(
                        paths.aligned_json
                        if paths.aligned_json.exists()
                        else paths.stable_json
                    )
                },
            )
        elif not vocal_needed:
            announce("vocal", "reused")
            update_run_metadata(
                paths.metadata_json,
                {
                    "timing": {
                        "stages": {
                            "vocal_separation": _stage_record_for_current_run(
                                stage_timing,
                                "vocal_separation",
                                stage_record(status="reused"),
                            )
                        }
                    }
                },
            )


def run_pipeline(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    device: str | None = "cuda",
    language: Optional[str] = None,
    gap_sec: float = DEFAULT_GAP_SEC,
    # `auto` = ask the card. A tier names what CLASS of card this is, not a
    # cap on what the pipeline may use; `resolve_gpu_tier` owns the answer.
    gpu_tier: str = AUTO_GPU_TIER,
    # None = `[asr] backend`, then local; a task's own products win over both (`joint_stage.task_backend`).
    asr: str | None = None,
    # None = `[asr] aligner`, then off. MOSS Pro only.
    asr_aligner: bool | None = None,
    asr_repair: str | None = None,
    separator_sample_rate: int | None = None,
    # None = follow `[separator] enabled`, then the stage's default (on).
    # `--no-separate` is for input that is already a clean vocal track: the
    # stage still delivers its 16 kHz mono artifact, it just transcodes rather
    # than separating.
    separate: bool | None = None,
    # None = follow config.toml, then the stage's default. The stage owns
    # that resolution so every front end lands on the same answer.
    vad_silero_assist: bool | None = None,
    qwen_verify: str = "auto",
    lang_redecode: str = "auto",
    squeeze_repair: str = "auto",  # the stable step's recall repair, `_recall_repair`
    gap_fill: str = "auto",
    row_fill: str = "off",
    # None = the default (`default_diarize`), through `task_diarize`.
    diarize: str | None = None,
    # None = the stage's own `DEFAULT_BACKEND`; never a second copy here.
    diarize_backend: str | None = None,
    diarize_max_speakers: int | None = None,
    # Whether a task diarized here reads acoustic row splitting's halves (`split_choice`).
    diarize_split: bool = True,
    # Lossless extras, asked for explicitly: see `_run_speaker_export_stage`.
    speaker_tracks: str = "off",
    # None = `[postprocess] empty_cues`, then off: a line with no text is left out of the rendered SRTs.
    empty_cues: bool | None = None,
    split_length_scale: float | None = None,
    asr_decode_batch: int | str | None = None,
    asr_context: str | None = None,
    word: bool = False,
    asr_stabilize_profile: int = asr_stabilize.DEFAULT_ASR_STABILIZE_PROFILE,
    stage: str = "raw-srt",
    # None = the active preset's defaults, then the built-in ones
    # (`resolve_llm_switches`); so is `knowledge` below.
    llm_media: str | None = None,
    llm_retrieval: str | None = None,
    llm_difficulty: str | None = None,
    llm_continuity: str | None = None,
    llm_parallel_windows: int | None = None,
    llm_fast: str = "auto",
    llm_proofread: str = "auto",
    llm_output_scale: float = 1.0,
    llm_video: str | Path | None = None,
    extra_info: str = "",
    extra_style: str = "",
    #: Named style entries for the LLM correction (`--style`). `None` falls
    #: through to `[llm] style` in the config; the backend resolver decides,
    #: not argparse (README_DEV.md 开发原则's option-default chain).
    style: str | None = None,
    #: none / read / update — see `knowledge/style.py`. `None` reads
    #: `[llm] style_mode`, then defaults to `read`.
    style_mode: str | None = None,
    download_video_source: bool = True,
    knowledge: str | None = None,
    refined_srt: str | Path | None = None,
    knowledge_root: str | Path | None = None,
    task_artifact_dir: str | Path | None = None,
    task_id: str = "",
    task_summary: str = "",
    test_profile: bool = False,
    postprocess_profile: int = 0,
    max_retries_per_window: int | None = None,
    max_replacements_per_window: int | None = None,
    resume: bool = True,
    _run_started_monotonic: float | None = None,
    _prior_timing: Mapping[str, Any] | None = None,
    _batch_workers: Mapping[str, int] | None = None,
    _diarize_default: str | None = None,
) -> PipelinePaths:
    # Taken from the caller's binding rather than passed in: the stages deep
    # inside this call tree read it the same way, and two mechanisms for one
    # thing is how they drift apart. A front end wraps its run in
    # `reporting_to(...)`; nothing bound means nothing shown.
    reporter = current_reporter()
    run_t0 = (
        float(_run_started_monotonic)
        if _run_started_monotonic is not None
        else time.perf_counter()
    )
    stage_timing: dict[str, Any] = dict(_prior_timing or {})
    if stage not in PIPELINE_STAGE_ORDER:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    if word and PIPELINE_STAGE_ORDER[stage] > PIPELINE_STAGE_ORDER["raw-srt"]:
        raise ValueError("--word can only be used through the raw-srt stage.")
    separate = settings.SEPARATE.resolve(separate)
    # Their stage resolves these itself; they are checked here too so that a
    # bad value -- a manifest row, a typo in config.toml -- fails the run
    # before separation, the heaviest stage, rather than after it.
    settings.VAD_SILERO_ASSIST.resolve(vad_silero_assist)
    settings.SPLIT_LENGTH_SCALE.resolve(split_length_scale)
    empty_cues = settings.EMPTY_CUES.resolve(empty_cues)
    # `None` is "not chosen" -- a front end sends it so that a bare `cpu` tier
    # can be told apart from an explicit `cuda` -- and it means the code
    # default. Normalised here, once, so no stage ever sees None and reads it
    # as something other than "the user asked for the card".
    device = str(device or "cuda")
    switches, profile = resolve_llm_switches(stage, llm_media, llm_retrieval, llm_difficulty, llm_continuity,
                                       knowledge, llm_output_scale, llm_parallel_windows)
    llm_media, llm_retrieval, llm_difficulty, llm_continuity, knowledge = switches.vector()
    asr_context = resolve_asr_context(asr_context, knowledge=knowledge)
    if PIPELINE_STAGE_ORDER[stage] > PIPELINE_STAGE_ORDER["raw-srt"]:
        _validate_llm_configuration(
            knowledge_root=knowledge_root,
            style=style,
            style_mode=style_mode,
            llm_difficulty=llm_difficulty,
        )
    # Normalize "auto" to None: whisper uses None for auto-detection;
    # the string "auto" is not a valid language code and would raise.
    if language and language.strip().lower() == "auto":
        language = None
    source_arg = str(input_path)
    input_is_url = _is_media_url(source_arg)
    if input_is_url:
        # Wall only, no CPU: a download is network wait by construction, so a
        # wall:CPU ratio here would be a large number that means nothing.
        download_t0 = time.perf_counter()
        source_path, paths, llm_video, source_extra_info = prepare_url_input(
            source_arg,
            output_path=output_path,
            llm_video=llm_video,
            download_video_source=download_video_source,
        )
        extra_info = compose_url_extra_info(
            source_arg,
            source_extra_info,
            extra_info,
            stage=stage,
            asr_context=asr_context,
        )
        stage_timing["download"] = stage_record(
            status="executed",
            elapsed_sec=time.perf_counter() - download_t0,
        )
    else:
        source_path = Path(input_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Input not found: {source_path}")
        paths = default_pipeline_paths(source_path, output_path)
        llm_media, llm_video, media_notice = resolve_llm_media_for_source(
            source_path,
            stage=stage,
            llm_media=llm_media,
            llm_video=llm_video,
        )
        if media_notice:
            # Says the LLM stage will not get the video it was asked for, which
            # changes the result rather than merely describing the run.
            current_reporter().warning("media-downgraded", media_notice)
        # A video source goes on to the stages as it is, like the URL branch
        # above. Separation decodes its own lossless copy when soundfile cannot
        # read the container, and the LLM clip cutter runs ffmpeg either way;
        # narrowing the audio here used to cost a lossy generation and a mono
        # 16 kHz downmix before separation -- which runs a 44.1 kHz stereo
        # model -- for an artifact nothing needed.
    paths.final_srt.parent.mkdir(parents=True, exist_ok=True)
    resolved_task_artifact_dir = (
        Path(task_artifact_dir).expanduser()
        if task_artifact_dir is not None
        else paths.task_artifact_dir
    )
    if resolved_task_artifact_dir != paths.task_artifact_dir:
        paths = paths._replace(task_artifact_dir=resolved_task_artifact_dir)
    diarize = task_diarize(paths, diarize, _diarize_default or default_diarize(stage, profile), resume=resume)
    backend = joint_stage.task_backend(paths.stable_json, paths.aligned_json, asr)
    joint = backend != joint_stage.LOCAL
    diarize = _backend_diarize(paths, backend, asr, diarize, asr_aligner)
    # The LLM keys become the one value the correction task takes, here and not
    # at the LLM stage: building it is where a contradictory request is found
    # (a speaker judge at efficiency, a video switch without a video), and
    # finding that after ASR is finding it at the most expensive moment.
    llm_request = None
    if PIPELINE_STAGE_ORDER[stage] >= PIPELINE_STAGE_ORDER["translated-srt"]:
        llm_request = _correction_request(
            paths, source_path, stage,
            axes=(llm_media, llm_retrieval, llm_difficulty, llm_continuity, llm_output_scale),
            llm_video=llm_video, fast=llm_fast, parallel_windows=switches.parallel_windows,
            extra_info=extra_info, extra_style=extra_style, style=style, style_mode=style_mode,
            knowledge=knowledge, knowledge_root=knowledge_root, refined_srt=refined_srt,
            switch_sources=switches.sources,
            task_id=task_id, task_summary=task_summary, postprocess_profile=postprocess_profile,
            max_retries_per_window=max_retries_per_window,
            max_replacements_per_window=max_replacements_per_window,
            speaker_judge=judge_of(diarize), proofread=llm_proofread, resume=resume, test_profile=test_profile,
            split_length_scale=split_length_scale, joint=joint,
        )

    from .media.ffmpeg import describe_ffmpeg
    from .speech.recognition import vad_asr_stage as vad_asr  # torch; not at the top (`pipeline --help`)

    update_run_metadata(
        paths.metadata_json,
        {
            "task_id": task_id or paths.final_srt.stem,
            "source": source_arg,
            "timing": {"stages": stage_timing},
            # The launchers may reuse a system ffmpeg instead of the managed
            # one, so which binary ran is no longer implied by the install.
            "tools": {"ffmpeg": describe_ffmpeg()},
            **(
                {"workers": {"batch": dict(_batch_workers)}}
                if _batch_workers
                else {}
            ),
        },
    )
    def announce(stage_key: str, status: str, *, detail: str = "") -> None:
        """Report which stage the run is entering, and whether it does work.

        Deliberately at the branch level rather than inside `_use_or_create`:
        a stage skipped outright never reaches that helper (see the `elif`
        arms below), the LLM stage does not go through it at all, and one of
        its call sites is not a pipeline stage in the first place. Reporting
        from the branches is also what keeps this in step with the
        executed/reused/skipped status written into the run metadata.

        ✱ `skipped` is deliberately *not* folded into `reused`: only `reused`
        means nothing ran, and the one stage that can be skipped still produces
        its artifact by another route. `detail` says which route, which is why
        this needed no third state in the reporting contract.
        """

        reporter.stage_started(stage_key, reused=status == "reused", detail=detail)

    target_order = PIPELINE_STAGE_ORDER[stage]
    # Every stage at or below the target may be announced, reused ones
    # included -- which is exactly the denominator a person watching wants.
    reporter.planned(_planned_stages(target_order, diarize=diarize))
    aligned_required = stage == "aligned" or (
        target_order >= PIPELINE_STAGE_ORDER["stable"]
        and not paths.stable_json.exists()
    )
    _run_vocal_stage(
        paths,
        stage=stage,
        target_order=target_order,
        aligned_required=aligned_required,
        source_path=source_path,
        separate=separate,
        gpu_tier=gpu_tier,
        device=device,
        separator_sample_rate=separator_sample_rate,
        stage_timing=stage_timing,
        announce=announce,
        joint=joint and stage != "vocal",
    )
    if joint and target_order >= PIPELINE_STAGE_ORDER["aligned"] and aligned_required:
        _run_joint_aligned(paths, source_path, backend=backend, language=language, aligner=asr_aligner,
                           repair=asr_repair, split_length_scale=split_length_scale, gpu_tier=gpu_tier,
                           device=device, stage_timing=stage_timing, announce=announce)
    elif target_order >= PIPELINE_STAGE_ORDER["aligned"] and aligned_required:
        aligned_existed = paths.aligned_json.exists()
        announce("aligned", "reused" if aligned_existed else "running")
        # Built here, not inside the stage: the selection needs the knowledge
        # base and `speech` must not import `llm`, so what crosses is a string.
        asr_context_text, asr_context_report = build_asr_context(
            asr_context,
            knowledge_root=knowledge_root,
            # The extra-info block already carries the scraped title when
            # there is one (`compose_url_extra_info`), plus whatever the user
            # wrote -- a user who names the subject is at least as good a
            # signal as a title, and costs nothing to read.
            text=extra_info,
        )
        if asr_context_text:
            update_run_metadata(
                paths.metadata_json, {"asr_context": asr_context_report}
            )
        asr_t0 = _stage_clock()
        _use_or_create(
            paths.aligned_json,
            "VAD-ASR",
            lambda temporary: vad_asr.run_vad_asr(
                paths.resolve_vocal_audio(),
                output_path=temporary,
                model_name=model_name,
                device=device,
                language=language,
                gap_sec=gap_sec,
                gpu_tier=gpu_tier,
                vad_silero_assist=vad_silero_assist,
                qwen_verify=qwen_verify,
                lang_redecode=lang_redecode,
                split_length_scale=split_length_scale,
                asr_decode_batch=asr_decode_batch,
                asr_context=asr_context_text,
                run_metadata_path=paths.metadata_json,
                vad_prefix_path=paths.vad_json,
            ),
        )
        _record_stage_time(
            paths,
            stage_timing,
            "asr",
            asr_t0,
            status="reused" if aligned_existed else "executed",
        )
    elif target_order >= PIPELINE_STAGE_ORDER["aligned"]:
        announce("aligned", "reused")
        update_run_metadata(
            paths.metadata_json,
            {
                "timing": {
                    "stages": {
                        "asr": _stage_record_for_current_run(
                            stage_timing,
                            "asr",
                            stage_record(status="reused"),
                        )
                    }
                },
            },
        )
    if target_order >= PIPELINE_STAGE_ORDER["stable"]:
        stable_existed = paths.stable_json.exists()
        announce("stable", "reused" if stable_existed else "running")
        stable_t0 = _stage_clock()
        # Stabilization and its recall repair are written for Whisper's failures: a joint-ASR task's stable is its aligned.
        stable_profile = -1 if joint else asr_stabilize_profile
        _use_or_create(
            paths.stable_json,
            "ASR stabilization",
            lambda temporary: asr_stabilize.run_asr_stabilize(
                paths.aligned_json,
                output_path=temporary,
                profile=stable_profile,
                repair=_recall_repair(paths, squeeze=squeeze_repair, gap=gap_fill, row_fill=row_fill,
                                      qwen_verify=qwen_verify, device=device, gpu_tier=gpu_tier,
                                      profile=stable_profile),
            ),
        )
        _record_stage_time(
            paths,
            stage_timing,
            "stabilize",
            stable_t0,
            status="reused" if stable_existed else "executed",
        )
        _run_diarization_stage(paths, diarize=diarize, device=device, gpu_tier=gpu_tier,
                               announce=announce, backend=diarize_backend, joint_backend=backend if joint else None,
                               max_speakers=diarize_max_speakers, diarize_split=diarize_split)
    if target_order >= PIPELINE_STAGE_ORDER["raw-srt"]:
        raw_srt_existed = paths.raw_srt.exists()
        announce("raw-srt", "reused" if raw_srt_existed else "running")
        raw_srt_t0 = _stage_clock()
        _write_raw_srt(paths, word=word, postprocess_profile=postprocess_profile, empty_cues=empty_cues)
        _run_speaker_export_stage(
            paths, speaker_tracks=speaker_tracks, empty_cues=empty_cues, announce=announce
        )
        _record_stage_time(
            paths,
            stage_timing,
            "raw_srt",
            raw_srt_t0,
            status="reused" if raw_srt_existed else "executed",
        )
    if target_order >= PIPELINE_STAGE_ORDER["translated-srt"]:
        llm_existed = paths.translated_srt.exists() or paths.final_srt.exists()
        # One call covers what the progress list shows as two stages, and which
        # of them actually runs depends on what is already on disk -- targeting
        # final-srt from nothing produces it in a single correction pass, with
        # no translated-srt in between.
        final_done = paths.final_srt.exists()
        translated_done = final_done or paths.translated_srt.exists()
        wants_final = target_order >= PIPELINE_STAGE_ORDER["final-srt"]
        announce("translated-srt", "reused" if translated_done else "running")
        if wants_final and translated_done:
            # Nothing but the post-processing tail is left, and that is what
            # the call below will do -- so this one is announced up front.
            announce("final-srt", "reused" if final_done else "running")
        llm_t0 = _stage_clock()
        from finesub.llm.correction_translation import run_correction_task

        # The task skips what is on disk itself (and still owes the knowledge
        # update for an output an earlier run left behind).
        run_correction_task(llm_request)
        if judge_of(diarize) != "off":
            _run_speaker_export_stage(paths, speaker_tracks=speaker_tracks, empty_cues=empty_cues, announce=announce)
        if wants_final and not translated_done:
            # The single correction pass above produced the final SRT itself,
            # post-processing included. This marks the transition; there is no
            # work left behind it.
            announce("final-srt", "running")
        _record_stage_time(
            paths,
            stage_timing,
            "llm_harness",
            llm_t0,
            status="reused" if llm_existed else "executed",
        )
        update_run_metadata(
            paths.metadata_json,
            {"llm_rounds": summarize_llm_rounds(paths.task_artifact_dir)},
        )
    completed = _stage_output(paths, stage)
    total_sec = time.perf_counter() - run_t0
    update_run_metadata(
        paths.metadata_json,
        {
            "timing": {"total_sec": round(max(0.0, total_sec), 3)},
            "completed_stage": stage,
            "completed_output": str(completed),
        },
    )
    if target_order >= PIPELINE_STAGE_ORDER["translated-srt"]:
        from finesub.llm.task_report import write_task_report

        write_task_report(
            paths.task_artifact_dir,
            task_id=task_id or paths.final_srt.stem,
            outputs={
                "raw_srt": str(paths.raw_srt),
                "translated_srt": str(paths.translated_srt),
                **(
                    {"final_srt": str(paths.final_srt)}
                    if paths.final_srt.exists()
                    else {}
                ),
            },
            run_metadata_path=paths.metadata_json,
        )
    reporter.completed(completed, total_sec)
    return paths


def _is_media_url(value: str) -> bool:
    from .media.source import is_url

    return is_url(value)


def source_title_line(url: str) -> str:
    """The scraped title as one extra-info line, or ``""``.

    It sits directly under `视频来源 URL:` because it describes the same thing,
    and it is **labelled rather than pasted bare**: everything else in the
    extra-info block was either computed by us or typed by the user, and this
    one line was scraped off someone else's page. A reader (human or model)
    that cannot tell those apart will treat a wrong title as a fact about the
    recording.

    The episode subject is what the term-injection experiment found missing:
    knowing it raised agreement with the corrector from 8.3% to 33.3% against
    an oracle ceiling of 53.3%, and the pipeline recorded nothing that said
    which subject a recording belonged to (`crispasr-followups.md` P9). A title
    is the cheapest thing that does.

    ⚠ Two things this line is NOT. The measured gain came from using the
    subject to pick a term list -- injecting the subject *name* alone scored
    the same as the control -- and the number is agreement with the corrector,
    not recall or accuracy. And this line only makes the subject available:
    the term list is picked from the whole extra-info block, this line
    included, by `build_asr_context` for the Qwen referee and by the
    correction layer's own retrieval.

    The video *description* is deliberately not here: url extraction usually
    picks it up anyway, and descriptions carry sponsor blurbs and channel
    boilerplate that would pre-inject entries with nothing to do with the
    episode.
    """

    from .media.source import resolve_video_title
    from .paths import resolve_reference_data_root

    title = resolve_video_title(url, resolve_reference_data_root())
    return f"视频标题（yt-dlp 自动抓取，未经人工核对）: {title}" if title else ""


#: How much of the knowledge base the recogniser's referee is told about.
#:
#: ``off``    inject nothing.
#: ``terms``  every NAME and nothing else: labelled entries, term lines trimmed
#:            to source form / canonical / aliases. The strings a recogniser
#:            has to get right, without the prose a corrector reads. The
#:            default (owner 2026-10-14). What was measured is agreement with
#:            the corrector, not accuracy; the per-segment listening review P9
#:            asked for (`docs/plans/crispasr-followups.md`) was closed without
#:            being run, because no negative effect has been seen in use.
#: ``full``   what the correction layer would see: entry bodies whole.
ASR_CONTEXT_LEVELS = ("off", "terms", "full")


def resolve_asr_context(level: str | None, *, knowledge: str | None = None) -> str:
    """The injection depth a run uses -- the one place the default lives.

    An explicit level wins, even over ``--knowledge none``: asking for context
    by name is a clearer statement than a switch about the LLM layer. Unset
    means ``terms``, except that ``--knowledge none`` promises the base "is not
    read or injected at all", and a default must not break that promise.
    ``knowledge`` is the RESOLVED switch (``resolve_llm_switches``); callers
    without one get the plain default.

    An unknown level degrades to ``off``: a typo must not silently inject a
    different depth.
    """

    chosen = str(level or "").strip().lower()
    if not chosen:
        return "off" if knowledge == "none" else "terms"
    return chosen if chosen in ASR_CONTEXT_LEVELS else "off"


def build_asr_context(
    level: str | None,
    *,
    knowledge_root: str | Path | None,
    text: str,
) -> tuple[str, dict[str, object]]:
    """Knowledge names for the ASR referee, and a report of what was picked.

    ⚠ This function is the whole reason the layering holds. `speech` must not
    import `llm`, and the referee lives in `speech`; so the selection happens
    HERE -- in the one module that already knows both sides -- and what crosses
    the boundary is a plain string. No new dependency in either direction.

    Selection reuses the retrieval the correction layer already runs rather
    than inventing a second one: entry-level keyword matching plus the
    sub-entry (term) matcher, over the same free text. ``text`` is the scraped
    title AND the user's extra-info -- a user who names the episode subject is
    at least as good a signal as a scraped title, and costs nothing to read.
    """

    chosen = resolve_asr_context(level)
    report: dict[str, object] = {"level": chosen}
    if chosen == "off" or not (text or "").strip():
        return "", report

    from .llm.knowledge.base import (
        DEFAULT_KNOWLEDGE_ROOT,
        load_preinjected_entries,
        match_terms,
        render_term_matches,
        term_lines_only,
    )

    root = knowledge_root or DEFAULT_KNOWLEDGE_ROOT
    try:
        entries, matches = load_preinjected_entries(root, text)
        terms = match_terms(
            root, text, exclude_subjects=[match.key for match in matches]
        )
    except Exception as exc:  # a missing or unreadable store is not fatal here
        current_reporter().debug("no asr context", {"error": str(exc)})
        return "", report

    names_only = chosen == "terms"
    blocks = []
    for key, body in entries.items():
        rendered = term_lines_only(body) if names_only else str(body).strip()
        if rendered:
            blocks.append(f"# {key}\n{rendered}")
    term_block = render_term_matches(terms, names_only=names_only)
    if term_block:
        blocks.append(term_block)

    report["entries"] = list(entries)
    report["terms"] = [match.to_dict() for match in terms]
    return "\n".join(blocks), report


def stage_consumes_extra_info(stage: str, *, asr_context: str | None = None) -> bool:
    """Does a run stopping at `stage` have anything that reads extra-info?

    Two consumers, and they sit at different depths. The correction task is
    the obvious one, so a run stopping at `vocal` through `raw-srt` normally
    reads it never -- and that matters, because one line of extra-info costs a
    network probe for the title.

    ⚠ `--asr-context` is the second, and it reads extra-info at **`aligned`**.
    Miss that and the flag looks wired while a default `finesub URL` (context
    `terms`) sees only the URL and a file path: the title it is supposed to
    match subjects from was never fetched. The gate therefore asks about the
    context switch too, not only the stage.

    But it asks about BOTH. A `--stage vocal` run stops before the referee
    exists, so the switch buys nothing there and the network probe would be
    paid for a consumer that never runs -- the same waste the stage half of
    this gate exists to avoid.
    """

    reached = PIPELINE_STAGE_ORDER.get(stage, PIPELINE_STAGE_ORDER["raw-srt"])
    if (
        reached >= PIPELINE_STAGE_ORDER["aligned"]
        and resolve_asr_context(asr_context) != "off"
    ):
        return True
    return reached >= PIPELINE_STAGE_ORDER["translated-srt"]


def compose_url_extra_info(
    url: str,
    source_info: str,
    user_extra_info: str,
    *,
    stage: str,
    asr_context: str | None = None,
) -> str:
    """The extra-info block for a URL input, in one place.

    One place because there are two callers -- the pipeline and the batch
    runner's download stage -- and they had drifted into two copies of the same
    join. A field added to one copy simply would not exist in batch runs, with
    nothing to say so.
    """

    return "\n".join(
        part
        for part in (
            f"视频来源 URL: {url}",
            source_title_line(url)
            if stage_consumes_extra_info(stage, asr_context=asr_context)
            else "",
            source_info,
            user_extra_info,
        )
        if part
    )


def resolve_llm_switches(
    stage: str,
    llm_media: str | None,
    llm_retrieval: str | None,
    llm_difficulty: str | None,
    llm_continuity: str | None,
    knowledge: str | None,
    llm_output_scale: float,
    llm_parallel_windows: int | None = None,
) -> tuple[Any, Any]:
    """The run's LLM switches (``correction_request.resolve_switches``) and the
    profile they make, which also validates them.

    Called before anything is downloaded or separated -- by ``run_pipeline``
    and by the batch runner's download stage -- so a conflicting vector costs
    nothing, and every later use reads the resolved values.
    A run that stops at ``raw-srt`` skips the preset layers: its switches are
    only validated, and a route configuration it never uses must not fail it.
    """

    from finesub.llm.correction_request import resolve_switches
    from finesub.llm.routing.profiles import resolve_profile

    switches = resolve_switches(
        {
            "media": llm_media,
            "retrieval": llm_retrieval,
            "difficulty": llm_difficulty,
            "continuity": llm_continuity,
            "knowledge": knowledge,
            "parallel_windows": llm_parallel_windows,
        },
        presets=PIPELINE_STAGE_ORDER[stage] > PIPELINE_STAGE_ORDER["raw-srt"],
    )
    profile = resolve_profile(
        switches.media,
        switches.retrieval,
        switches.difficulty,
        switches.continuity,
        output_scale=llm_output_scale,
    )
    return switches, profile


def prepare_url_input(
    url: str,
    *,
    output_path: str | Path | None,
    llm_video: str | Path | None,
    download_video_source: bool = True,
    on_destination: Callable[[PipelinePaths], None] | None = None,
) -> tuple[Path, PipelinePaths, str | Path | None, str]:
    """Resolve a media URL into (audio_path, paths, llm_video, extra_info).

    Public so the batch runner's download stage can reuse it; derives the
    artifact paths from the resolved video id.

    ``on_destination`` is shown those paths as soon as the id is known --
    before anything is downloaded into them -- and again if the download
    names the video differently; raising stops the download. The batch claims
    the destination there: two spellings of one video would otherwise run
    yt-dlp into one directory at once and collide only afterwards.

    A URL fetches the **video** by default, at every stage. The subtitles this
    pipeline produces are usually going to be burned into that video, so the
    file is wanted whether or not an LLM stage runs and whether or not a media
    switch asks to *show* it to a model -- and re-fetching it later means
    resolving the same URL a second time, which is exactly what an id map
    exists to avoid. ``download_video_source=False`` (``--no-download-video``)
    takes the audio-only path for a run that really only wants a transcript.
    """
    from .media.source import (
        download_audio,
        download_video,
        resolve_video_id,
    )
    from .paths import resolve_reference_data_root

    # The switches are validated by the caller (`resolve_llm_switches`),
    # before this downloads anything.
    map_dir = resolve_reference_data_root()
    video_id = resolve_video_id(url, map_dir)
    paths = default_pipeline_paths(Path(video_id), output_path)
    claim = on_destination or (lambda _paths: None)
    claim(paths)
    media_dir = paths.final_srt.parent
    if download_video_source and not llm_video:
        resolved_id, video_path = download_video(
            url, map_dir, video_id=video_id, target_dir=media_dir
        )
        # The video is the source: separation makes its own lossless copy when it
        # needs one, and the LLM clip cutter runs ffmpeg either way. Extracting a
        # narrowed audio track here would only cost a generation.
        if resolved_id != video_id:
            paths = default_pipeline_paths(Path(resolved_id), output_path)
            claim(paths)
        return (
            video_path.resolve(),
            paths,
            video_path,
            "媒体文件: " + str(video_path) + "\nLLM 视频文件: " + str(video_path),
        )
    resolved_id, audio_path = download_audio(
        url, map_dir, video_id=video_id, target_dir=media_dir
    )
    if resolved_id != video_id:
        paths = default_pipeline_paths(Path(resolved_id), output_path)
        claim(paths)
    return audio_path.resolve(), paths, llm_video, "媒体文件: " + str(audio_path)


def _stage_output(paths: PipelinePaths, stage: str) -> Path:
    if stage == "vocal":
        return paths.vocal_audio
    if stage == "aligned":
        return paths.aligned_json
    if stage == "stable":
        return paths.stable_json
    if stage == "raw-srt":
        return paths.raw_srt
    if stage == "translated-srt":
        return paths.translated_srt
    return paths.final_srt

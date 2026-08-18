"""What a finished task leaves on disk, and what may be removed.

What a run produces is the pipeline's business; what happens to it afterwards
is the front end's, and the pipeline never deletes anything it made.

The lists are shared. The desktop reads all of them -- it tidies a finished
task down to what a rerun needs. The CLI runs where the user pointed it and
takes nothing out of their folder, so it reads the rest: which two files are
worth copying into the task directory as the record of this work, which file a
stage delivers, and what that file is called in a record both front ends read.

Naming is the pipeline's, restated. `finesub.pipeline` derives every
artifact from the delivered SRT, and this module has to name the same files
without importing that package -- the CLI shell resolves them before it has an
interpreter that could import it, and `finesub_bootstrap` must not depend on
the pipeline. `test_paths.py` pins the two lists against each other, which is
what keeps a restatement from becoming a divergence.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

from .fsops import remove_tree


#: Suffixes appended to the delivered SRT's stem, by role.
#:
#: `-stable.json` is deliberately absent from both: it is neither bulky nor
#: reproducible cheaply. Delete it and a rerun redoes separation and
#: recognition from the audio, and a standalone correction pass has no input at
#: all -- it is small next to the vocal track, which is what actually fills a
#: disk. `-annotated.csv` stays for the same reason: the knowledge base reads
#: it, and nothing regenerates it but another LLM run.
REMOVABLE_SUFFIXES = (
    "-vocal.ogg",
    "-vocal.flac",  # Separation's lossless delivery mode, and older runs.
    "-aligned.json",
    "-raw.srt",
    "-translated.srt",
    "-metadata.json",
    # Only catches the case where the source media and the subtitle share a
    # stem. A decoded copy is named after the *source*, so the sidecar below is
    # what actually finds it; this stays as the cheap half of the answer.
    "-decoded.flac",
)

#: Where a run lists the scratch files nothing else could name. Restated here
#: rather than imported for the same reason the suffixes above are: this module
#: must not depend on the pipeline (see the note at the top of the file).
_SCRATCH_FILES_KEY = "scratch_files"
_METADATA_SUFFIX = "-metadata.json"

#: What is worth keeping once a run is over, and the exact complement of the
#: list above. The ASR result is what a rerun reads instead of redoing
#: separation and recognition; the annotated CSV is what the knowledge base
#: reads, and only another LLM pass could produce it again. Everything else is
#: reproducible from these two.
RECORD_SUFFIXES = (
    "-stable.json",
    "-annotated.csv",
)

#: Appended to the stem to name the LLM harness's working directory. Removed
#: whole: session and window checkpoints exist to survive a *failure*, and a
#: task that reached cleanup had none.
ARTIFACT_DIR_SUFFIX = ".llm-artifacts"


#: Which file a run actually delivers, by the stage it was asked for. `-o`
#: names the *final* SRT, so for every earlier stage the deliverable is a
#: sibling of it rather than the path itself -- a caller that copies `-o` after
#: a default `raw-srt` run copies a file that was never written.
DELIVERABLE_SUFFIX_BY_STAGE = {
    "raw-srt": "-raw.srt",
    "translated-srt": "-translated.srt",
    "final-srt": ".srt",
}


#: What each stage's subtitle is called in a task record's `outputs`. The
#: desktop's history reads these three names and nothing else, so a run filed
#: under any other key is one whose subtitle the UI cannot offer to open --
#: which is the whole point of the two front ends sharing an index.
DELIVERABLE_KEY_BY_STAGE = {
    "raw-srt": "rawSrt",
    "translated-srt": "translatedSrt",
    "final-srt": "finalSrt",
}


def deliverable(output_srt: str | Path, stage: str) -> Path | None:
    """The subtitle `stage` produces, or None for a stage producing none.

    `final-srt` delivers the output path itself, extension included. The
    pipeline only supplies `.srt` when the path has no suffix at all, so
    forcing one here would name `out.srt` for a run the user asked to write to
    `out.txt` -- and then fail to find the file it just produced.
    """

    if stage not in DELIVERABLE_SUFFIX_BY_STAGE:
        return None
    delivered = Path(output_srt).expanduser().resolve()
    if stage == "final-srt":
        return delivered if delivered.suffix else delivered.with_suffix(".srt")
    stem = delivered.with_suffix("")
    return stem.with_name(f"{stem.name}{DELIVERABLE_SUFFIX_BY_STAGE[stage]}")


def removable_artifacts(output_srt: str | Path) -> tuple[Path, ...]:
    """Files a successful task may delete, deliverable included.

    The delivered subtitle is in here because which subtitle was delivered
    depends on the stage the task asked for; the caller says what to keep.
    """

    delivered = Path(output_srt).expanduser().resolve()
    stem = delivered.with_suffix("")
    return (
        *(stem.with_name(f"{stem.name}{suffix}") for suffix in REMOVABLE_SUFFIXES),
        delivered,
    )


def recorded_scratch_files(output_srt: str | Path) -> tuple[Path, ...]:
    """Scratch files this run wrote down, because nothing else can name them.

    A decoded copy of the input is named after the *source* media and lands in
    whatever directory the stage was writing to, so it cannot be derived from
    the delivered subtitle. The run records the path as soon as the file
    exists; this reads that record back. An unreadable or absent sidecar simply
    yields nothing -- it is one more file left behind, not a failure.
    """

    delivered = Path(output_srt).expanduser().resolve()
    stem = delivered.with_suffix("")
    sidecar = stem.with_name(f"{stem.name}{_METADATA_SUFFIX}")
    try:
        body = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return ()
    recorded = body.get(_SCRATCH_FILES_KEY) if isinstance(body, dict) else None
    if not isinstance(recorded, list):
        return ()
    # Confined to the directory the run wrote into, which is where both stages
    # put a decoded copy. The record is our own, but it is a list of paths in a
    # JSON file that survives the run, and what reads it goes on to delete
    # them: a hand-edited or half-written sidecar should cost a leftover file,
    # never a deletion somewhere else on the disk.
    within = delivered.parent
    return tuple(
        candidate
        for candidate in (
            Path(item).expanduser().resolve()
            for item in recorded
            if isinstance(item, str) and item
        )
        if candidate.is_relative_to(within)
    )


def artifact_directory(output_srt: str | Path) -> Path:
    delivered = Path(output_srt).expanduser().resolve()
    stem = delivered.with_suffix("")
    return stem.with_name(f"{stem.name}{ARTIFACT_DIR_SUFFIX}")


def cleanup_intermediate(
    output_srt: str | Path,
    *,
    preserve: Iterable[str | Path] = (),
) -> None:
    """Remove the bulky private artifacts of one successful task.

    Only ever called once the whole pipeline returned and the subtitle was
    published: a task that fails part way through skips this, so every
    intermediate it produced is still there for the rerun to resume from.

    Removing nothing that was asked to be preserved is the caller's only
    obligation, and the run directory follows the same rule implicitly -- the
    final `rmdir` succeeds on an empty directory only, so whatever survives
    keeps the directory too.

    One case this deliberately does not chase: a run given an explicit
    `--task-artifact-dir` keeps its harness directory somewhere this cannot
    derive, so that directory is left alone. Neither front end passes the flag;
    a person who does gets less tidying, never a deletion somewhere else.
    """

    kept = {Path(path).expanduser().resolve() for path in preserve}
    # Read before the sweep: the sidecar holding the record is itself removable.
    scratch = recorded_scratch_files(output_srt)
    for path in (*removable_artifacts(output_srt), *scratch):
        if path not in kept:
            path.unlink(missing_ok=True)

    directory = artifact_directory(output_srt)
    # A published subtitle never lives in there, but the caller decides what
    # `preserve` means -- so ask rather than assume.
    if not any(path.is_relative_to(directory) for path in kept):
        # Best effort on purpose. This runs after the subtitle was published,
        # inside the block that marks a task failed, so one open handle -- an
        # antivirus scan, an editor holding an artifact -- would turn a finished
        # task into a failed one. Tidying is not what the task was for.
        try:
            remove_tree(directory)
        except OSError:
            pass

    try:
        Path(output_srt).expanduser().resolve().parent.rmdir()
    except OSError:
        pass

"""Which on-demand tools a run needs, from what the run is asking for.

Two entry points ask this question about the same pipeline -- the desktop from
a TaskRequest, the CLI from a command line -- so the rule lives once here and
each adapts its own inputs to it. Getting them out of step would mean a task
that starts on one and is refused on the other.

`uv` and `ffmpeg` are not here: every run needs them, so they are provisioned
up front rather than derived from the request.

Two lists, and the difference between them is load-bearing: a missing
*required* tool refuses the run, a missing *preferred* one only makes it
slower.
"""

from __future__ import annotations

from collections.abc import Sequence


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


#: Stages that run the LLM layer, and therefore the knowledge update.
LLM_STAGES = ("translated-srt", "final-srt")


def required_capabilities(
    *,
    knowledge: str = "none",
    source: str = "",
    stage: str = "raw-srt",
) -> tuple[str, ...]:
    """On-demand resource ids for a run with these properties."""

    needed: list[str] = []
    # The knowledge base is a SQLite store (docs/plans/knowledge-node-plan.md); no
    # run needs git any more. `knowledge`/`stage` stay in the signature so the
    # callers' intent is still expressed and a future on-demand resource can
    # key on them without touching every call site.
    del knowledge, stage
    if is_url(source):
        needed.append("yt-dlp")
    return tuple(needed)


def preferred_capabilities(*, stage: str = "raw-srt") -> tuple[str, ...]:
    """Resource ids this run is better off with, but must never wait on.

    Deliberately not part of `required_capabilities`: both front ends gate a
    task on that list, and everything here has a working fallback in the
    pipeline. Gating on an optimisation turns a slower run into no run at all.

    `tokcount` is the local tokenizer binary. The LLM layer counts tokens
    through the free `countTokens` endpoint without it, so the run still works
    -- it just needs the network and a key to plan, even for a dry run.
    """

    return ("tokcount",) if stage in LLM_STAGES else ()


def _run_properties(arguments: Sequence[str]) -> dict[str, str]:
    """The few pipeline flags the questions above are asked about.

    Deliberately literal: it reads the flags that matter rather than
    reimplementing the pipeline's parser, and errs toward installing (a missed
    flag would fail the run later, a spurious install merely costs a download).
    """

    knowledge = "none"
    source = ""
    stage = ""
    correct_translate = False
    for index, argument in enumerate(arguments):
        if argument == "--knowledge" and index + 1 < len(arguments):
            knowledge = arguments[index + 1]
        elif argument.startswith("--knowledge="):
            knowledge = argument.split("=", 1)[1]
        elif argument == "--stage" and index + 1 < len(arguments):
            stage = arguments[index + 1]
        elif argument.startswith("--stage="):
            stage = argument.split("=", 1)[1]
        elif argument == "--llm-correct-translate":
            correct_translate = True
        elif is_url(argument) and not source:
            source = argument
    # `pipeline.main`: `args.stage or ("final-srt" if correct_translate else
    # "raw-srt")`. An explicit stage always wins, whichever order they were
    # typed in -- reading them as one value in argument order made
    # `--stage raw-srt --llm-correct-translate` look like an LLM run to
    # everything here while the pipeline stopped at the transcript.
    if not stage:
        stage = "final-srt" if correct_translate else "raw-srt"
    return {"knowledge": knowledge, "source": source, "stage": stage}


def capabilities_from_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    """`required_capabilities`, asked of a pipeline command line."""

    return required_capabilities(**_run_properties(arguments))


def preferred_capabilities_from_arguments(
    arguments: Sequence[str],
) -> tuple[str, ...]:
    """`preferred_capabilities`, asked of a pipeline command line."""

    return preferred_capabilities(stage=_run_properties(arguments)["stage"])

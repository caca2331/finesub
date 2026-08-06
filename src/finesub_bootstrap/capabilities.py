"""Which on-demand tools a run needs, from what the run is asking for.

Two entry points ask this question about the same pipeline -- the desktop from
a TaskRequest, the CLI from a command line -- so the rule lives once here and
each adapts its own inputs to it. Getting them out of step would mean a task
that starts on one and is refused on the other.

`uv` and `ffmpeg` are not here: every run needs them, so they are provisioned
up front rather than derived from the request.
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
    if knowledge == "update" and stage in LLM_STAGES:
        # The knowledge base is an embedded git repository and auto-apply
        # commits to it -- but the update only runs inside the correction and
        # translation stage. Asking for git on a plain transcription would make
        # the default settings demand a download nothing is going to use.
        needed.append("git")
    if is_url(source):
        needed.append("yt-dlp")
    return tuple(needed)


def capabilities_from_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    """Same question, asked of a pipeline command line.

    Deliberately literal: it reads the two flags that matter rather than
    reimplementing the pipeline's parser, and errs toward installing (a missed
    flag would fail the run later, a spurious install merely costs a download).
    """

    knowledge = "none"
    source = ""
    stage = "raw-srt"
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
            # pipeline.main: this flag alone selects final-srt.
            stage = "final-srt"
        elif is_url(argument) and not source:
            source = argument
    return required_capabilities(knowledge=knowledge, source=source, stage=stage)

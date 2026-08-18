"""Nothing on the pipeline's path may write to a stream directly.

The log-shape test cannot see this: it fakes every stage, so a `print` inside
one never runs. That is exactly how the first pass of this migration left
`Wrote ...` lines in the separator and reuse notices in the downloader -- green
tests, unchanged output. A static check has no such blind spot.

Module-level CLIs are exempt. Each of these files is also a standalone dev
entry point, and `main()` printing to its own terminal is the thing the
reporter exists to keep *out* of the library path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "finesub"

#: Everything `run_pipeline` reaches. Kept as an explicit list rather than a
#: glob so that adding a module to the pipeline is a decision someone makes
#: here too.
#: Everything under `finesub` is checked; only these are exempt. An
#: allowlist was the first shape of this test and it was wrong: it listed ten
#: modules, and eight others on the pipeline path -- the silero assist's CPU
#: fallback, the referee's, checkpoint write failures, trimmed overlapping
#: cues -- kept printing, unseen, because nobody had thought to add them.
#: Forgetting to exempt something fails loudly; forgetting to include it did
#: not fail at all.
EXEMPT_MODULES = {
    # Front ends. They *supply* reporters to the pipelines they run; their own
    # per-item status and closing summaries are that front end's output.
    "batch.py",
    "workflows/reference_ingest.py",
    # The stall watchdog dumps stacks when a run has stopped responding. It
    # writes to its own sink on purpose: a reporter takes a lock, which is the
    # one thing that must not happen in the code that runs when the process is
    # already wedged. See docs/wt-parallelism.md on its GIL hazards.
    "speech/runtime/stall_watchdog.py",
}

#: The LLM harness, exempt only because it always was: it lived in its own
#: top-level package until `llm` moved under `finesub` (2026-08), and this
#: rule's subject has been the pipeline layer since it was written. Twelve of
#: its modules print today. They do sit on `run_pipeline`'s path once the
#: translate stages are opted into, so bringing them in is worth doing -- but
#: that is twelve behaviour changes, not a rename, and it needs its own change.
EXEMPT_PREFIXES = ("llm/",)

#: Functions whose body is allowed to print: the standalone CLI entry points.
CLI_FUNCTIONS = {"main", "parse_args"}


def _printing_lines(source: Path) -> list[int]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in CLI_FUNCTIONS:
            exempt.update(
                range(node.lineno, (node.end_lineno or node.lineno) + 1)
            )
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == "print":
            if node.lineno not in exempt:
                offenders.append(node.lineno)
    return sorted(offenders)


def _checked_modules() -> list[str]:
    return sorted(
        relative
        for path in SOURCE_ROOT.rglob("*.py")
        if (relative := path.relative_to(SOURCE_ROOT).as_posix())
        not in EXEMPT_MODULES
        and not relative.startswith(EXEMPT_PREFIXES)
    )


@pytest.mark.parametrize("relative", _checked_modules())
def test_a_pipeline_module_reports_instead_of_printing(relative: str) -> None:
    source = SOURCE_ROOT / relative
    offenders = _printing_lines(source)

    assert offenders == [], (
        f"{relative} prints outside its CLI entry point at lines {offenders}; "
        "use current_reporter() so the desktop and the task log see it too"
    )


def test_every_front_end_binds_a_reporter_and_quiets_libraries() -> None:
    """A front end that forgets either one is silent, or drowned out.

    reference_ingest built its own batch items and called run_batch directly,
    so a binding that lived in batch's item builder never reached it -- the
    pipeline reported to a reporter that shows nothing, warnings included.
    """

    for relative, expected in (
        # run_batch binds a per-item reporter for whoever calls it.
        ("batch.py", ("quieted_libraries", "reporting_to")),
        ("workflows/reference_ingest.py", ("quieted_libraries",)),
        ("pipeline.py", ("quieted_libraries", "reporting_to")),
        ("speech/preprocessing/separator/separation.py", ("quieted_libraries", "reporting_to")),
        ("speech/recognition/vad_asr_stage.py", ("reporting_to",)),
        ("speech/recognition/transcribe.py", ("reporting_to",)),
        ("speech/postprocessing/stabilization.py", ("reporting_to",)),
        ("subtitles/rendering.py", ("reporting_to",)),
    ):
        source = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        for name in expected:
            assert f"{name}(" in source, f"{relative} never calls {name}"


def test_every_exemption_still_names_a_real_module() -> None:
    """An exemption for a file that moved would silently stop protecting it."""

    missing = [name for name in EXEMPT_MODULES if not (SOURCE_ROOT / name).is_file()]

    assert missing == []


def test_the_check_would_notice_a_print_that_came_back(tmp_path: Path) -> None:
    """A guard nobody has seen fail is a guard nobody should trust."""

    module = tmp_path / "regressed.py"
    module.write_text(
        "def run():\n"
        "    print('Wrote something')\n"
        "\n"
        "def main():\n"
        "    print('this one is fine')\n",
        encoding="utf-8",
    )

    assert _printing_lines(module) == [2]

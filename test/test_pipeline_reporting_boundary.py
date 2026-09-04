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
    # `pipeline.py` is one since the batch CLI merged into it (2026-08-30);
    # `scheduler.py` is the runner behind it and prints the per-item lines a
    # caller has not taken over with `on_item_error`.
    "pipeline.py",
    "scheduler.py",
    # Split out of `pipeline.py` (2026-08-31) and exempt on the same grounds as
    # `scheduler.py`: it is behind that one CLI, and its three prints are that
    # CLI's `[batch]` channel -- the registry could not be written, the control
    # file already had lines acted on, a control line was skipped. They are the
    # batch front end talking about its own bookkeeping, not pipeline stages
    # bypassing a reporter. ⚠ Anything here that ever reports on the CONVERSION
    # belongs in `stages`/`speech` and under the reporter, not under this line.
    "batch_state.py",
    "workflows/reference_ingest.py",
    # The stall watchdog dumps stacks when a run has stopped responding. It
    # writes to its own sink on purpose: a reporter takes a lock, which is the
    # one thing that must not happen in the code that runs when the process is
    # already wedged. See docs/wt-parallelism.md on its GIL hazards.
    "speech/runtime/stall_watchdog.py",
}

#: No prefix is exempt. `llm/` used to be, only because it always had been --
#: it lived in its own top-level package until it moved under `finesub`
#: (2026-08), and this rule's subject had been the pipeline layer since it was
#: written. Its 42 sites across 14 modules were converted in 2026-08-19; they
#: sit on `run_pipeline`'s path once the translate stages are opted into, and
#: the user-visible symptom was a run log whose LLM section was empty.
EXEMPT_PREFIXES: tuple[str, ...] = ()

#: Functions whose body is allowed to print: the standalone CLI entry points.
#: `_main_impl` is part of one: an entry point that has to own a resource for
#: the whole run (`correction_translation.main` opens the agent session scope
#: and books what it spent afterwards) splits into a wrapper plus the body
#: that used to be `main`. The wrapper is the boundary; the body is still the
#: CLI. `_main` is the same split done for a second reason (2026-09-03):
#: `main` now only binds the terminal reporter around the run, because the
#: thread-local default is silent and these entry points were dropping every
#: warning the library handed them.
#: Library functions are unaffected -- they never had these names.
CLI_FUNCTIONS = {"main", "_main", "_main_impl", "parse_args"}

#: A print that *is* the deliverable rather than a log line: a dry run's prompt
#: text, a measurement table. Marking one is a deliberate, greppable act --
#: unlike a module-level exemption, which also covers every print added later.
PRODUCT_OUTPUT_MARKER = "# product output"


def _printing_lines(source: Path) -> list[int]:
    text = source.read_text(encoding="utf-8")
    lines = text.split("\n")
    tree = ast.parse(text, filename=str(source))
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
            if node.lineno in exempt:
                continue
            if PRODUCT_OUTPUT_MARKER in lines[node.lineno - 1]:
                continue
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

    reference_ingest built its own items and called run_batch directly, so a
    binding that lived in the item builder never reached it -- the pipeline
    reported to a reporter that shows nothing, warnings included.
    """

    for relative, expected in (
        # run_batch binds a per-item reporter for whoever calls it.
        ("scheduler.py", ("reporting_to",)),
        ("workflows/reference_ingest.py", ("quieted_libraries",)),
        # The CLI does not bind one itself any more -- it HANDS one to the
        # runner (`item_reporter=`), which is what makes a single foreground
        # run keep in-place progress and its log file while going through the
        # same code path as a batch. Losing that argument would silently drop
        # the run back to the shared-terminal line renderer.
        ("pipeline.py", ("quieted_libraries", "item_reporter")),
        ("speech/preprocessing/separator/separation.py", ("quieted_libraries", "reporting_to")),
        ("speech/recognition/vad_asr_stage.py", ("reporting_to",)),
        ("speech/recognition/transcribe.py", ("reporting_to",)),
        ("speech/postprocessing/stabilization.py", ("reporting_to",)),
        ("subtitles/rendering.py", ("reporting_to",)),
    ):
        source = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        for name in expected:
            # Called, or handed over as a keyword: a front end either binds the
            # reporter itself or gives the runner one to bind.
            assert f"{name}(" in source or f"{name}=" in source, (
                f"{relative} never calls or passes {name}"
            )


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


def _unbound_pools(source: Path) -> list[int]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name != "ThreadPoolExecutor":
            continue
        if not any(keyword.arg == "initializer" for keyword in node.keywords):
            offenders.append(node.lineno)
    return sorted(offenders)


@pytest.mark.parametrize(
    "relative",
    sorted(
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in (SOURCE_ROOT / "llm").rglob("*.py")
    ),
)
def test_a_worker_pool_carries_the_reporter_into_its_threads(relative: str) -> None:
    """A pool without an `initializer` reports into the void.

    The binding is thread-local, so a pool started without one leaves its
    workers on the silent default: the run looks normal and the log is just
    missing things. The correction driver runs whole windows in such threads,
    which is where this would be least visible and most missed.

    Scoped to `llm/` because that is where the reporting call sites in worker
    threads are. `speech/` has two unbound pools -- `energy.py`, `spectral.py`
    -- whose bodies say nothing today; widen this when they do.
    """

    offenders = _unbound_pools(SOURCE_ROOT / relative)

    assert offenders == [], (
        f"{relative} starts a ThreadPoolExecutor without initializer= at lines "
        f"{offenders}; pass bind_reporter with current_reporter() so work in "
        "those threads reaches the same log its parent does"
    )


def _timeline_postprocess_calls() -> list[tuple[str, int, ast.Call]]:
    """Every `postprocess_srt_file(...)` inside a loop over the timeline profiles.

    Two profiles, one file: whatever these calls do, they do it twice.
    """

    found: list[tuple[str, int, ast.Call]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(SOURCE_ROOT.parent).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            iterated = node.iter
            name = (
                iterated.id
                if isinstance(iterated, ast.Name)
                else getattr(iterated, "attr", "")
            )
            if name != "TIMELINE_POSTPROCESS_PROFILES":
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and getattr(inner.func, "id", getattr(inner.func, "attr", ""))
                    == "postprocess_srt_file"
                ):
                    found.append((relative, inner.lineno, inner))
    return found


def test_the_timeline_passes_do_not_report_the_same_finding_twice() -> None:
    """`validate=False` on every intermediate pass over one file.

    The 2026-08 fix for "行超长的提示重复三遍" changed `stages.py` and missed the
    copy of the same loop in `llm/stages/correction/run.py`, so a `--stage
    final-srt` run printed each finding three times: once from the ASR half,
    then once per timeline profile from the LLM half -- the last two against an
    absolute path the first one did not use. Found by running the pipeline for
    real on 2026-09-03, not by a test, which is why this one exists.

    The deliverable is still validated: that call is not in this loop.
    """

    calls = _timeline_postprocess_calls()
    assert calls, "找不到任何一处时间轴 postprocess 循环——筛选面坏了"

    offenders = [
        f"{relative}:{line}"
        for relative, line, call in calls
        if not any(
            keyword.arg == "validate"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in call.keywords
        )
    ]
    assert offenders == [], (
        "这些 postprocess_srt_file 调用在时间轴循环里，却没有 validate=False，"
        "同一条提示会按 profile 数重复：" + ", ".join(offenders)
    )

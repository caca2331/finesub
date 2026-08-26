from __future__ import annotations

from pathlib import Path

import pytest

# Domain markers, declared per file. `test_llm_*.py` is not listed -- the
# prefix is matched below. Every other test file must appear in exactly one of
# these tuples; `test_packaging.py` asserts it, because a file in none of them
# is invisible to all three `-m` selectors while still reading as green.
#
# Keys are paths relative to `test/`, in posix form, not bare file names: the
# suite has subdirectories now, and two directories may hold a `test_paths.py`
# apiece. Under bare names one of them would silently inherit the other's
# marker -- or, worse, register as covered while never being marked.

# Production pipeline orchestration, plus the runtime and provisioning
# infrastructure a run stands on: GPU/thread budgets, device resolution, the
# separator's own plumbing, terminal reporting, config, secrets and state.
_PIPELINE_FILES: tuple[str, ...] = (
    # The provisioning layer both front ends stand on. Its docstring above says
    # why it counts as pipeline: a run cannot start without paths, a runtime and
    # the models fetched into them.
    "bootstrap/test_archive.py",
    "bootstrap/test_asset_resolve.py",
    "bootstrap/test_download_routes.py",
    "bootstrap/test_downloader.py",
    "bootstrap/test_fsops.py",
    "bootstrap/test_hf_verify.py",
    "bootstrap/test_migrations.py",
    "bootstrap/test_model_caches.py",
    "bootstrap/test_model_ensure.py",
    "bootstrap/test_paths.py",
    "bootstrap/test_resource_manager.py",
    "bootstrap/test_runtime_environment.py",
    "bootstrap/test_runtime_regional_lock.py",
    "bootstrap/test_shell_activity.py",
    "bootstrap/test_shell_commands.py",
    "bootstrap/test_shell_first_run.py",
    "bootstrap/test_system_tools.py",
    "bootstrap/test_task_lease.py",
    "bootstrap/test_task_output.py",
    "test_batch_runner.py",
    "test_config.py",
    "test_cuda_libs.py",
    "test_config_file.py",
    "test_doc_links.py",
    "test_gpu_stage_gate.py",
    "test_import_boundaries.py",
    "test_packaging.py",
    "test_paths.py",
    "test_pipeline_log_shape.py",
    "test_pipeline_refactor.py",
    "test_pipeline_reporting_boundary.py",
    "test_publish_filter.py",
    "test_reporting.py",
    "test_resource_budget_pipeline.py",
    "test_resource_profiles.py",
    "test_resource_usage.py",
    "test_run_metadata.py",
    "test_runtime_device.py",
    "test_secrets.py",
    "test_agy_records.py",
    "test_separation_blocks.py",
    "test_separator_accel.py",
    "test_separator_progress.py",
    "test_stall_watchdog.py",
    "test_state_store.py",
    "test_thread_budget.py",
    "test_vocal_separation_pool.py",
)

# Everything on the audio -> words -> subtitle path: VAD, decoding, alignment,
# stabilization, segmentation, and the text utilities they use.
_ASR_FILES: tuple[str, ...] = (
    "test_asr_and_text_utils.py",
    "test_asr_progress_reporting.py",
    "test_asr_stabilize.py",
    "test_decodable_input.py",
    "test_fw_refine.py",
    "test_intervals.py",
    "test_lang_redecode.py",
    "test_qwen_verify.py",
    "test_segment_split.py",
    "test_srt_rendering.py",
    "test_vad_carve_hints.py",
    "test_vad_low_peak_absorb.py",
    "test_vad_segment_energy.py",
    "test_vad_silero_ghost.py",
    "test_vad_silero_probs.py",
    "test_vad_streaming.py",
    "test_word_starts.py",
    "test_wt_refine_validation.py",
)

_FILE_MARKERS: dict[str, tuple[str, ...]] = {
    **{name: ("pipeline",) for name in _PIPELINE_FILES},
    **{name: ("asr",) for name in _ASR_FILES},
}


TEST_ROOT = Path(__file__).resolve().parent


def setattr_correction(monkeypatch, name: str, value) -> None:
    """Patch a name wherever the correction loop or its orchestrator binds it.

    The window loop is a package now, and several of the names a test wants to
    control are looked up in more than one of its modules (`load_entry_texts`
    in four, `_output_limit_check` in two). Patching only the module a
    particular call happens to sit in leaves the rest answering for real --
    which is not a crash, just a test that quietly stops testing. Patching
    every module that binds the name is what the single-module patch used to
    mean.
    """

    import finesub.llm.correction_translation as correction_translation
    from finesub.llm.stages.correction import (
        attempts,
        commit,
        context,
        metadata,
        parallel,
        query_round,
        run,
        serial,
    )

    # The package `__init__` is deliberately absent: patching a re-export
    # rebinds a name nothing reads, and having it here would let the assertion
    # below pass on a patch that did nothing.
    modules = (
        run,
        serial,
        parallel,
        attempts,
        query_round,
        context,
        commit,
        metadata,
        correction_translation,
    )
    targets = [module for module in modules if hasattr(module, name)]
    assert targets, f"test patch target {name!r} no longer exists"
    for module in targets:
        monkeypatch.setattr(module, name, value, raising=True)


@pytest.fixture
def file_markers() -> dict[str, tuple[str, ...]]:
    """The per-file marker table, for the test that guards its coverage."""

    return dict(_FILE_MARKERS)


def _marker_key(path: Path) -> str | None:
    """A test file's key in the table, or None if it lives outside `test/`.

    `testpaths` also names a file under `desktop/scripts/`, and
    `relative_to` raises rather than returning None -- during collection that
    is a ValueError with no test attached to it.
    """

    resolved = path.resolve()
    if not resolved.is_relative_to(TEST_ROOT):
        return None
    return resolved.relative_to(TEST_ROOT).as_posix()


def _is_linked_worktree_checkout(repository_root: Path) -> bool:
    """Distinguish a linked worktree from a normal checkout or submodule."""

    pointer = repository_root / ".git"
    if not pointer.is_file():
        return False
    try:
        gitdir = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return "/worktrees/" in gitdir.replace("\\", "/")


@pytest.fixture(autouse=True)
def managed_data_root(tmp_path_factory, monkeypatch):
    """Keep the developer's real FineSub install out of the suite.

    Two separate hazards, one temporary directory. Personal-data paths fall
    back to the managed layout under `%LOCALAPPDATA%`, so a test that resolves
    one would otherwise point at -- and could write into -- the machine's own
    installation. The agent quota ledger is durable and shared with every other
    FineSub on the box, so a developer whose Codex plan happens to be spent
    would watch routing tests fail for a reason that has nothing to do with
    their change; that is exactly what happened the first time it went in.

    They share one `mktemp` because these are autouse: two directories per test
    is two directory creations per test, paid before the first line of it runs.
    """

    root = tmp_path_factory.mktemp("finesub-data")
    local_app_data = root / "LocalAppData"
    local_app_data.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    # `%LOCALAPPDATA%` is the Windows half of `default_data_root`; everywhere
    # else it resolves from the home directory and this fixture used to do
    # nothing at all. On Linux CI that meant every bootstrap test shared the
    # runner's real `~/.finesub` -- and xdist workers overwrote each other's
    # `locations.json`, so failures named a path belonging to another test.
    home = root / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # The developer machine may opt out of .env protection globally
    # (FINESUB_ENV_PROTECT=0, a transition hatch); tests need the default.
    monkeypatch.delenv("FINESUB_ENV_PROTECT", raising=False)
    # A checkout's data root is the checkout itself, so a `model_catalog.psv`
    # a developer wrote to reach their own endpoint is layered onto the
    # packaged one -- and the assertions that pin the packaged targets then
    # describe that machine instead of the package. Pointed at a path that
    # does not exist, `resolve_model_catalog_override` returns None, which is
    # what CI sees. A test that wants an override passes it explicitly.
    monkeypatch.setenv("FINESUB_MODEL_CATALOG", str(root / "no-catalog-override.psv"))
    # The shipped source table names real country endpoints, so anything that
    # resolves a download region would reach the network. Forcing it keeps the
    # suite offline.
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "global")

    from finesub.llm.agent import agent_quota

    ledger = agent_quota.AgentQuotaLedger(root / "agent-quota" / ".state")
    monkeypatch.setattr(agent_quota, "default_ledger", lambda: ledger)
    return ledger


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-heavy-resource",
        action="store_true",
        default=False,
        help="Run tests that may load models, process media, or use significant GPU/RAM.",
    )
    parser.addoption(
        "--regenerate-goldens",
        action="store_true",
        default=False,
        help="Rewrite the tracked prompt snapshots and their manifest entry.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    for item in items:
        key = _marker_key(item.path)
        if key is None:
            continue
        markers = _FILE_MARKERS.get(key)
        if markers is None and item.path.name.startswith("test_llm_"):
            markers = ("llm",)
        if markers:
            for name in markers:
                item.add_marker(getattr(pytest.mark, name))

    # A linked worktree's `.git` pointer targets `.git/worktrees/<name>`; the
    # main checkout has the real `.git` directory, while a submodule's pointer
    # targets `.git/modules/...`. Some tests intentionally exercise the active
    # checkout's own paths or perform knowledge-base apply/commit. Product code
    # redirects those operations to the main checkout (and refuses writes), so
    # running their ordinary-checkout assertions here is either a failure or a
    # false pass that never reaches the behavior under test.
    repository_root = Path(__file__).resolve().parents[1]
    if _is_linked_worktree_checkout(repository_root):
        skip_worktree = pytest.mark.skip(
            reason="requires the main git checkout; linked worktrees redirect shared data"
        )
        for item in items:
            if "requires_main_checkout" in item.keywords:
                item.add_marker(skip_worktree)

    if config.getoption("--run-heavy-resource"):
        return
    skip_heavy = pytest.mark.skip(
        reason="requires --run-heavy-resource to run significant GPU/RAM tests"
    )
    for item in items:
        if "heavy_resource" in item.keywords:
            item.add_marker(skip_heavy)




class RecordedWarning:
    """One `warning()` call, kept whole so a test can assert on any part."""

    def __init__(self, code: str, message: str, impact: str, action: str) -> None:
        self.code = code
        self.message = message
        self.impact = impact
        self.action = action

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.message, self.impact, self.action) if part)


class _Recorder:
    """Records what a stage reported, ignoring everything else.

    Before the LLM layer moved onto the reporter its warnings went to stderr,
    and tests read them with `capsys`. That stopped working the moment nothing
    binds a renderer in a unit test -- the default reporter is silent by
    design -- so the assertion has to move to the events themselves.
    """

    def __init__(self) -> None:
        self.warnings: list[RecordedWarning] = []
        self.debugs: list[tuple[str, dict]] = []
        self.progress_calls: list[dict] = []

    def warning(self, code, message, *, impact="", action="") -> None:
        self.warnings.append(RecordedWarning(code, message, impact, action))

    def debug(self, message, fields=None) -> None:
        self.debugs.append((message, dict(fields or {})))

    def progress(self, stage, **kwargs) -> None:
        self.progress_calls.append({"stage": stage, **kwargs})

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None

    def codes(self) -> list[str]:
        return [item.code for item in self.warnings]

    def joined(self) -> str:
        return "\n".join(item.text for item in self.warnings)


@pytest.fixture
def reported():
    """Bind a recording reporter for the duration of a test."""

    from finesub.reporting import reporting_to

    recorder = _Recorder()
    with reporting_to(recorder):
        yield recorder

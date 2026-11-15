from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# The knowledge root default is read once, at import (`DEFAULT_KNOWLEDGE_ROOT`,
# also bound into many default arguments), so per-test monkeypatching cannot
# reach it. Point it at an empty directory before anything imports finesub:
# otherwise a checkout run resolves to the developer's real knowledge base and
# CI (which has none) runs a different shape. A test that wants a base builds
# one and passes it explicitly -- so a value exported in the shell is
# overridden too, or it would bring the real base back without a word.
_empty_knowledge = Path(tempfile.mkdtemp(prefix="finesub-test-")) / "knowledge"
_empty_knowledge.mkdir()
atexit.register(shutil.rmtree, _empty_knowledge.parent, ignore_errors=True)
os.environ["FINESUB_KNOWLEDGE_ROOT"] = str(_empty_knowledge)

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
    "bootstrap/test_cn_lock.py",
    "bootstrap/test_asset_resolve.py",
    "bootstrap/test_download_routes.py",
    "bootstrap/test_downloader.py",
    "bootstrap/test_fsops.py",
    "bootstrap/test_fsops_links.py",
    "bootstrap/test_hf_verify.py",
    "bootstrap/test_migrations.py",
    "bootstrap/test_model_caches.py",
    "bootstrap/test_model_ensure.py",
    "bootstrap/test_model_fetch.py",
    "bootstrap/test_paths.py",
    "bootstrap/test_resource_manager.py",
    "bootstrap/test_shell.py",
    "bootstrap/test_runtime_environment.py",
    "bootstrap/test_runtime_health.py",
    "bootstrap/test_shell_runtime_health.py",
    "bootstrap/test_runtime_regional_lock.py",
    "bootstrap/test_shell_activity.py",
    "bootstrap/test_shell_commands.py",
    "bootstrap/test_shell_first_run.py",
    "bootstrap/test_system_tools.py",
    "bootstrap/test_task_lease.py",
    "bootstrap/test_task_output.py",
    "bootstrap/test_update_check.py",
    "test_config.py",
    "test_settings.py",
    "test_cuda_libs.py",
    "test_native_paths.py",
    "test_doc_facts.py",
    "test_doc_links.py",
    "test_doc_style.py",
    "test_feedback_pack.py",
    "test_run_audit_digest.py",
    "test_function_size.py",
    "test_gpu_stage_gate.py",
    "test_import_boundaries.py",
    "test_jsonl.py",
    "test_subprocess_text_encoding.py",
    "test_subprocess_console_window.py",
    "test_suite_isolation.py",
    "test_packaging.py",
    "test_paths.py",
    "test_pipeline_log_shape.py",
    "test_pipeline_items.py",
    "test_pipeline_refactor.py",
    "test_pipeline_reporting_boundary.py",
    "test_pre_push.py",
    "test_publish_filter.py",
    "test_reporting.py",
    "test_resource_budget_pipeline.py",
    "test_resource_profiles.py",
    "test_resource_usage.py",
    "test_run_metadata.py",
    "test_run_telemetry.py",
    "test_batch_state.py",
    "test_scheduler.py",
    "test_runtime_device.py",
    "test_support_levels.py",
    "test_secrets.py",
    "test_agy_records.py",
    "test_separation_blocks.py",
    "test_source_title.py",
    "test_separator_accel.py",
    "test_separator_demix.py",
    "test_separator_progress.py",
    "test_stall_watchdog.py",
    "test_state_store.py",
    "test_thread_budget.py",
    "test_vocal_separation_pool.py",
)

# Everything on the audio -> words -> subtitle path: VAD, decoding, alignment,
# stabilization, segmentation, and the text utilities they use.
_ASR_FILES: tuple[str, ...] = (
    "test_asr_backend_contract.py",
    "test_speech_joint.py",
    "test_speech_joint_stage.py",
    "test_speech_joint_moss_local.py",
    "test_speech_joint_repair.py",
    "test_vad_stage_guards.py",
    "test_phase_timing.py",
    "test_encoder_cache.py",
    "test_asr_and_text_utils.py",
    "test_asr_progress_reporting.py",
    "test_align_sentinel.py",
    "test_asr_stabilize.py",
    "test_squeeze_repair.py",
    "test_recall_repair.py",
    "test_row_fill.py",
    "test_collapse_ladder.py",
    "test_recall_decode.py",
    "test_cjk_repeat_folding.py",
    "test_decodable_input.py",
    "test_fw_refine.py",
    "test_hf_weights.py",
    "test_intervals.py",
    "test_lang_audit.py",
    "test_lang_redecode.py",
    "test_qwen_verify.py",
    "test_referee_accel.py",
    "test_decode_prefetch.py",
    "test_diarization_assign.py",
    "test_annotated_speaker_column.py",
    "test_speaker_judge_results.py",
    "test_spk_cell_example.py",
    "test_speaker_judge_stage.py",
    "test_speaker_judge_shape.py",
    "test_speaker_judge_repair.py",
    "test_product_rows.py",
    "test_llm_grouped_pass2.py",
    "test_diarization_annotate.py",
    "test_diarization_chunking.py",
    "test_diarization_export.py",
    "test_diarization_backends.py",
    "test_diarization_nemotron3.py",
    "test_diarizen_port.py",
    "test_diarizen_chunking.py",
    "test_diarization_evidence.py",
    "test_diarization_features.py",
    "test_diarization_many_tracks.py",
    "test_diarization_row_split.py",
    "test_stable_rows.py",
    "test_diarization_stage.py",
    "test_diarization_labels.py",
    "test_diarization_marks.py",
    "test_diarization_segments.py",
    "test_segment_split.py",
    "test_option_defaults.py",
    "test_srt_rendering.py",
    "test_subtitle_time_order.py",
    "test_subtitle_compliance.py",
    "test_line_groups.py",
    "test_overlap_local_golden.py",
    "test_vad_carve_hints.py",
    "test_vad_prefix.py",
    "test_vad_level_tiers.py",
    "test_asr_decode_batch.py",
    "test_asr_context.py",
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
        request_turn,
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
        request_turn,
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

    `cli/tests` has its own suite, but one command may name files from both
    (`pytest test/x.py cli/tests/y.py`), and this rootdir conftest then sees its
    items too; `relative_to` raises rather than returning None -- during
    collection that is a ValueError with no test attached to it.
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
    # Windows reads `USERPROFILE`, not `HOME`, so there `Path.home()` is still
    # the developer's. It is not moved wholesale: tests that touch a model read
    # its weights from the home caches (the Hugging Face hub, audio-separator),
    # and with an empty home they download them instead. What else lives under
    # the home is agy's, and FineSub both reads it (transcripts) and writes it
    # (project records, which agent cleanup also deletes), so those two move.
    monkeypatch.setattr(
        "finesub_bootstrap.agy_records.agy_project_records_dir",
        lambda: home / ".gemini" / "config" / "projects",
    )
    monkeypatch.setattr(
        "finesub.llm.agent.agy_transcript.agy_data_dir",
        lambda: home / ".gemini" / "antigravity-cli",
    )
    # The developer machine may opt out of .env protection globally
    # (FINESUB_ENV_PROTECT=0, a transition hatch); tests need the default.
    monkeypatch.delenv("FINESUB_ENV_PROTECT", raising=False)
    # A checkout's data root is the checkout itself, so a `model_catalog.toml`
    # a developer wrote to reach their own endpoint is layered onto the
    # packaged one -- and the assertions that pin the packaged targets then
    # describe that machine instead of the package. Pointed at a path that
    # does not exist, `resolve_model_catalog_override` returns None, which is
    # what CI sees. A test that wants an override passes it explicitly.
    monkeypatch.setenv("FINESUB_MODEL_CATALOG", str(root / "no-catalog-override.toml"))
    # The same for the checkout's own `config.toml`: a developer's `[postprocess]`
    # or `[vad]` would otherwise move the defaults every assertion below pins.
    # A test that wants a config writes one and points FINESUB_CONFIG_FILE at it.
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(root / "no-config.toml"))
    # The Codex driver carries the user's connection settings from
    # `$CODEX_HOME/config.toml`; a developer's proxy there would put a real
    # provider table into every Codex argv and a connect to their port into
    # every call. A test that wants a config writes one under CODEX_HOME.
    monkeypatch.setenv("CODEX_HOME", str(root / "codex-home"))
    # Machine state a checkout resolves to itself: the limiters' ledger
    # (`.state`, with its `.lock`), which key-pool tests wrote cooldowns into,
    # and the `.env` the key pools read -- the developer's real keys, plus a
    # `.env.lock` beside them. Pointed at a path that does not exist,
    # `resolve_env_file` returns None, which is what CI sees.
    monkeypatch.setenv("FINESUB_STATE_FILE", str(root / "state"))
    monkeypatch.setenv("FINESUB_ENV_FILE", str(root / "no.env"))
    # From a checkout an agent episode publishes its activity lease in the
    # checkout itself (`.task-activity.lock`, `.task-activity/`). The complete
    # location tuple a launcher hands a managed worker moves it, and the
    # capsules, into tmp -- shaped as an explicit parent, so retained evidence
    # resolves back to the same place. A test about resolving the location
    # clears these.
    from finesub.llm.agent import agent_paths
    from finesub_bootstrap import locks

    capsules = root / "agent" / "capsules"
    monkeypatch.setenv(locks.AGENT_CAPSULE_ROOT_VARIABLE, str(capsules))
    monkeypatch.setenv(locks.AGENT_ACTIVITY_ROOT_VARIABLE, str(capsules.parent))
    monkeypatch.setenv(locks.AGENT_IDENTITY_ANCHOR_VARIABLE, str(capsules))
    monkeypatch.setenv(locks.AGENT_LOCATOR_KIND_VARIABLE, agent_paths.LOCATOR_EXPLICIT)
    from finesub import config as app_config

    app_config.clear_config_cache()
    # The shipped source table names real country endpoints, so anything that
    # resolves a download region would reach the network. Forcing it keeps the
    # suite offline.
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "global")

    from finesub.llm.agent import agent_quota

    ledger = agent_quota.AgentQuotaLedger(root / "agent-quota" / ".state")
    monkeypatch.setattr(agent_quota, "default_ledger", lambda: ledger)
    return ledger


@pytest.fixture(autouse=True)
def _no_installed_cards(monkeypatch: pytest.MonkeyPatch):
    """Tests must not read this machine's graphics hardware.

    `device.installed_gpu_vendors()` asks the OS what is plugged in, so a test
    that simulates "no CUDA" and lands on the CPU tier warns on a developer
    machine with a card and stays quiet in CI -- and since the warning is once
    per process, which tests see it depends on how xdist split them. Tests that
    want the warning install their own vendors on top of this.

    The once-flag is reset with it: a process-global "already said it" must not
    leak from one test into the next.

    Without torch there is nothing to patch: `device` imports it, so no test
    can reach the probe. The CI Windows job installs no torch and runs only
    such tests.
    """

    try:
        from finesub.speech.runtime import device
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return

    monkeypatch.setattr(device, "installed_gpu_vendors", frozenset)
    monkeypatch.setattr(device, "_WARNED_UNUSED_CARD", False)


@pytest.fixture(autouse=True)
def _no_hosted_diarizen(monkeypatch: pytest.MonkeyPatch):
    """Tests must not fetch the hosted DiariZen build, nor see a copy this machine already has.

    `--diarize acoustics` defaults to DiariZen and downloads about 300 MB on first use. Without this a
    test that reaches `ensure` would start that download on a clean machine -- and on a developer
    machine that ran it once would find the cached copy, so "a half export is refused" passed or
    failed depending on what was in `~/.cache`. Tests about the fetch stub it themselves.
    """

    from finesub.speech.diarization import diarizen_backend
    from finesub.speech.runtime import hf_weights

    monkeypatch.setattr(hf_weights, "prepare_directory", lambda manifest_id: None)
    monkeypatch.setattr(diarizen_backend, "_already_fetched", lambda build: True)


@pytest.fixture(autouse=True)
def _fresh_agent_slot_pools():
    """Process-global agent slot budgets must not survive a test.

    `local_agent._IN_FLIGHT_POOLS` is one budget per vendor, and by design the
    limit is fixed by **whichever driver builds it first** -- a later driver
    asking for a different `max_parallel` is reported, not honoured
    (`local_agent.py`, `_shared_in_flight_pool`). That is right for production,
    where the budget is physical: one CLI, one subscription, one machine.

    It is wrong across tests, which share a process. A test constructing
    `CodexLocalAgentDriver()` with the default config pins the codex pool at
    `max_parallel=4` for every later test in that worker, so a test that asks
    for 2 silently runs with 4. That is what made
    `test_driver_admits_only_max_parallel_calls_at_once` flaky: with a limit of
    4 and six threads its `peak` lands on 2, 3 or 4 by timing, and under xdist
    whether the earlier test shares its worker varies run to run.

    Only touched when something already imported the module -- a run that never
    loads the agent layer has no pools to clear and should not pay the import.
    Sharing WITHIN one test is untouched, which is what
    `test_all_models_of_one_vendor_share_the_in_flight_budget` asserts.
    """

    module = sys.modules.get("finesub.llm.agent.local_agent")
    if module is None:
        yield
        return
    for pools in (module._IN_FLIGHT_POOLS, module._TOOL_SLOT_POOLS):
        pools.clear()
    yield
    for pools in (module._IN_FLIGHT_POOLS, module._TOOL_SLOT_POOLS):
        pools.clear()


#: Names a module binds that resolve to the checkout itself when a test runs
#: from one and that no environment variable moves: the batch root (relative
#: `out/batch`), the logs directory (the data root, whose parent also holds the
#: batch registry `batches.json`) and `data/reference` (downloaded media and
#: the URL -> id / title cache `url-info.json`). Each module imported the name
#: itself, so each binding is patched; `finesub.paths` is listed for the
#: callers that import it at call time.
_CHECKOUT_BOUND = {
    "DEFAULT_BATCH_ROOT": ("finesub.scheduler", "finesub.pipeline"),
    "resolve_logs_dir": ("finesub.pipeline", "finesub.batch_state", "finesub.llm.http"),
    "resolve_reference_data_root": (
        "finesub.paths",
        "finesub.media.source",
        "finesub.workflows.reference_ingest",
    ),
}


@pytest.fixture(autouse=True)
def _checkout_paths_in_tmp(request, monkeypatch):
    """What a run writes into the checkout lands in the test's tmp dir instead.

    `pipeline.main` takes the batch lock before handing over to `run_batch`,
    opens a run log per item, and `batch_state.record_batch` adds a row to the
    registry beside the logs directory -- so a test that ran `main` on several
    sources, or with `--batch-id`, left `out/batch/<id>/.batch.lock`, an empty
    `logs/run-*.log` and a `batches.json` row in the checkout even with
    `run_batch` faked; each full run added more. A URL source does the same to
    `data/reference/url-info.json`, which is maintainer data a rerun never
    clears. Tests that point these somewhere themselves still win: their
    `setattr` comes after this one.

    Only modules something already imported are touched, as for the agent
    pools above.
    """

    targets = [
        (module, name)
        for name, owners in _CHECKOUT_BOUND.items()
        for module in map(sys.modules.get, owners)
        if module is not None and hasattr(module, name)
    ]
    if targets:
        base = request.getfixturevalue("tmp_path")
        replacements = {
            "DEFAULT_BATCH_ROOT": base / "out" / "batch",
            "resolve_logs_dir": lambda: base / "data-root" / "logs",
            "resolve_reference_data_root": lambda: base / "data-root" / "data" / "reference",
        }
        for module, name in targets:
            monkeypatch.setattr(module, name, replacements[name])
    yield


def _leak_locations() -> tuple[Path, Path | None, Path | None]:
    """Where an unpatched run would write, resolved the way production does.

    Different roots, and in a linked worktree they are not the same checkout:
    the batch directory is `out/batch` relative to the working directory (the
    slot), while the logs directory -- and the registry beside it -- and the
    checkout data root resolve to the MAIN checkout. Taking them from this
    file's location missed every leak into the main checkout from a run in a
    slot. Called at session start, before any fixture has patched anything.
    """

    from finesub.paths import checkout_data_enabled, resolve_checkout_root, resolve_logs_dir

    checkout = resolve_checkout_root() if checkout_data_enabled() else None
    return Path.cwd() / "out" / "batch", resolve_logs_dir(), checkout


#: Machine state a checkout keeps at its root, which the suite must never
#: create (`managed_data_root` moves each of them to tmp).
_CHECKOUT_STATE_NAMES = (
    ".state", ".state.lock", ".state.tmp", ".env.lock", ".task-activity", ".task-activity.lock",
)
#: Hosts no real source has: the reserved example domains and TLDs (RFC 2606)
#: and loopback, which is what the suite's URL sources use.
_TEST_HOSTS = ("example.com", "example.org", "example.net", "localhost", "127.0.0.1")
_TEST_TLDS = (".test", ".example", ".invalid", ".localhost")


def _is_test_url(url: str) -> bool:
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").lower()
    return host.endswith(_TEST_TLDS) or any(
        host == name or host.endswith("." + name) for name in _TEST_HOSTS
    )


def _checkout_leaks(batch_root: Path, logs: Path | None, checkout: Path | None) -> set[str]:
    """What a test leaves behind, recognisable as a test's.

    Each shape is one a real run never produces, so a batch running beside the
    suite in the same checkout is not mistaken for a leak: a real batch writes
    its queue view the moment it starts, a real run log has content from its
    first line, a real registry row never points into pytest's temp tree, and
    a real source is never on a reserved example host. The checkout's state
    files are the exception: a real run writes them too, so they count only
    when the run created them, which a developer's checkout -- where they
    already exist -- never shows; CI and fresh clones do.

    Scan surface: `out/batch` under the working directory; `run-*.log` and
    `batches.json` in the logs directory; at the checkout data root, the names
    in `_CHECKOUT_STATE_NAMES` and `data/reference/url-info.json`.
    """

    import json

    found: set[str] = set()
    if checkout is not None:
        found |= {
            str(checkout / name) for name in _CHECKOUT_STATE_NAMES if (checkout / name).exists()
        }
        url_info = checkout / "data" / "reference" / "url-info.json"
        if url_info.is_file():
            try:
                cached = json.loads(url_info.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached = {}
            if not isinstance(cached, dict):
                cached = {}
            found |= {f"{url_info} entry {url}" for url in cached if _is_test_url(str(url))}
    if batch_root.is_dir():
        found |= {
            str(entry)
            for entry in batch_root.iterdir()
            if entry.is_dir() and [child.name for child in entry.iterdir()] == [".batch.lock"]
        }
    if logs is None:
        return found
    if logs.is_dir():
        found |= {
            str(entry)
            for entry in logs.glob("run-*.log")
            if entry.is_file() and entry.stat().st_size == 0
        }
    registry = logs.parent / "batches.json"
    if registry.is_file():
        try:
            rows = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = []
        # Valid JSON need not be a list (`null`, a number); production's reader
        # treats that as empty, and so must a check that runs before any test.
        if not isinstance(rows, list):
            rows = []
        found |= {
            f"{registry} row {row.get('batch_id')} -> {row.get('queue')}"
            for row in rows
            if isinstance(row, dict) and "pytest-of-" in str(row.get("queue", ""))
        }
    return found


def pytest_sessionstart(session: pytest.Session) -> None:
    if not hasattr(session.config, "workerinput"):  # the xdist controller, or no xdist
        locations = _leak_locations()
        session.config._leak_locations = locations
        session.config._checkout_leaks_before = _checkout_leaks(*locations)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run if the suite left batch state, run logs or checkout state behind.

    The fixture above closes the leaks it knows about; this catches the next
    one, which would otherwise pile up unnoticed the way these did.
    """

    before = getattr(session.config, "_checkout_leaks_before", None)
    if before is None:
        return
    leaked = sorted(_checkout_leaks(*session.config._leak_locations) - before)
    if not leaked:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            "The test run left these in the checkout (a test reached a checkout path "
            "that conftest does not move to tmp): " + "; ".join(leaked),
            red=True,
        )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config: pytest.Config) -> None:
    """Tests must not route through this machine's proxy.

    Several tests serve HTTP on 127.0.0.1 and fetch from it. Where the
    environment sets `HTTPS_PROXY` (cloud sandboxes do; so does a developer
    behind one) and `NO_PROXY` misses loopback, those requests reach the proxy,
    which answers 405, and the tests fail for a reason unrelated to the code.
    Cleared here, before collection, rather than in a per-test fixture: a
    module that builds its urllib opener at import reads the proxies then.
    Tests about proxy handling set the variables they need.
    """

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        os.environ.pop(name, None)
        os.environ.pop(name.lower(), None)
    # The same for the checkout's own model catalog and config: a test module
    # that builds parameters at import (`test_llm_correct_only_prompts`) loads
    # the routes during collection, before the per-test fixture below points
    # these away -- and the cached catalog then describes the developer's
    # machine for every later test in that worker.
    os.environ["FINESUB_MODEL_CATALOG"] = os.path.join(
        str(config.rootpath), "no-catalog-override.toml"
    )
    os.environ["FINESUB_CONFIG_FILE"] = os.path.join(str(config.rootpath), "no-config.toml")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-heavy-resource",
        action="store_true",
        default=False,
        help="Run tests that may load models, process media, or use significant GPU/RAM.",
    )
    parser.addoption(
        "--run-network-mock",
        action="store_true",
        default=False,
        help="Run tests that bind real local sockets (e.g. the LLM proxy route proof).",
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
    # checkout's own paths. Product code redirects those to the main checkout,
    # so running their ordinary-checkout assertions here is either a failure or
    # a false pass that never reaches the behavior under test.
    repository_root = Path(__file__).resolve().parents[1]
    # FINESUB_TEST_ALLOW_WORKTREE=1 runs them anyway.
    if _is_linked_worktree_checkout(repository_root) and not os.environ.get("FINESUB_TEST_ALLOW_WORKTREE"):
        skip_worktree = pytest.mark.skip(
            reason="requires the main git checkout; linked worktrees redirect shared data"
        )
        for item in items:
            if "requires_main_checkout" in item.keywords:
                item.add_marker(skip_worktree)

    # Real sockets and server threads: correct, quick, and still not something
    # every `pytest -q` should be binding ports for. Opt in when the thing under
    # test is the route itself.
    if not config.getoption("--run-network-mock"):
        skip_sockets = pytest.mark.skip(
            reason="requires --run-network-mock to bind local sockets"
        )
        for item in items:
            if "network_mock" in item.keywords:
                item.add_marker(skip_sockets)

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


@pytest.fixture(autouse=True)
def _fresh_host_breaker():
    """The process-wide reachability record must not survive a test.

    `api_http.HOST_BREAKER` judges a host unreachable after 30 s of connect
    failures, measured on the real clock. Tests that fake a refused connection
    each add one; across a long suite they would add up to a verdict, and every
    later test talking to that host would see a skipped or single-shot call
    depending on which tests ran before it.
    """

    from finesub.api_http import HOST_BREAKER

    HOST_BREAKER.reset()
    yield
    HOST_BREAKER.reset()

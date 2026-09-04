from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from finesub_bootstrap import model_caches
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.models import ResourceStatus
from desktop.backend.common.models import TaskRequest
from desktop.backend.resources.model_prefetch import ModelPrefetchFailed
from desktop.backend.resources.desktop_service import (
    DesktopResourceService,
    capability_requirements,
)
from finesub_bootstrap.system_tools import SystemTool
from finesub_bootstrap.environment import WorkerContext


class FakeBootstrap:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.installed: list[str] = []
        self.states = {
            resource_id: ResourceStatus(
                id=resource_id, version="1.0", state="missing"
            )
            for resource_id in ("uv", "ffmpeg", "git", "yt-dlp", "tokcount")
        }

    def status(self, resource_id: str) -> ResourceStatus:
        return self.states[resource_id]

    def active_version(self, resource_id: str) -> str | None:
        state = self.states[resource_id]
        return state.version if state.state == "ready" else None

    def install_path(self, resource_id: str) -> Path:
        return self.root / resource_id

    def cache_path(self, resource_id: str) -> Path:
        return self.root / "cache" / resource_id

    def install(self, resource_id: str, progress) -> ResourceStatus:
        self.installed.append(resource_id)
        self.states[resource_id] = self.states[resource_id].model_copy(
            update={"state": "ready"}
        )
        return self.states[resource_id]

    # Mirrors runtime-manifest.json: the required file's path inside the
    # archive is what decides which directory ends up on PATH, so flattening it
    # here would let a wrong parent computation pass.
    LAYOUT = {
        "ffmpeg": Path("bin/ffmpeg.exe"),
        "git": Path("cmd/git.exe"),
        "yt-dlp": Path("yt_dlp/__init__.py"),
        "tokcount": Path("tokcount.exe"),
    }

    def active_file(self, resource_id: str, filename: str) -> Path | None:
        if self.states[resource_id].state != "ready":
            return None
        relative = self.LAYOUT.get(resource_id, Path(filename))
        if relative.name.casefold() != filename.casefold():
            return None
        target = self.root / resource_id / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"binary")
        return target


class FakeRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths = AppPaths.for_root(root)
        self.ready = False
        self.installs = 0

    def status(self) -> ResourceStatus:
        return ResourceStatus(
            id="uv",
            version="Python 3.12",
            state="ready" if self.ready else "missing",
        )

    def install(self) -> ResourceStatus:
        self.installs += 1
        self.ready = True
        return self.status()

    def worker_context(
        self,
        *,
        ffmpeg_bin,
        extra_env,
        extra_path_dirs=(),
        extra_python_path=(),
    ) -> WorkerContext:
        return WorkerContext(
            python_executable=self.root / "python.exe",
            working_directory=self.root / "app",
            environment={
                **extra_env,
                "FFMPEG_BIN": str(ffmpeg_bin) if ffmpeg_bin else "",
                "PATH_DIRS": os.pathsep.join(str(p) for p in extra_path_dirs),
                "PYTHONPATH_EXTRA": os.pathsep.join(
                    str(p) for p in extra_python_path
                ),
            },
        )


def test_python_install_bootstraps_uv_before_activating_runtime(
    tmp_path: Path,
) -> None:
    bootstrap = FakeBootstrap(tmp_path)
    runtime = FakeRuntime(tmp_path)
    service = DesktopResourceService(
        bootstrap=bootstrap, runtime=runtime, system_tool_finders={}
    )

    result = service.install("uv", lambda event: None)

    assert result.state == "ready"
    assert bootstrap.installed == ["uv"]
    assert runtime.installs == 1


def test_a_task_waits_for_every_managed_tool(tmp_path: Path) -> None:
    bootstrap = FakeBootstrap(tmp_path)
    runtime = FakeRuntime(tmp_path)
    service = DesktopResourceService(
        bootstrap=bootstrap, runtime=runtime, system_tool_finders={}
    )

    for resource_id in ("uv", "ffmpeg", "git"):
        assert service.task_ready() is False
        service.install(resource_id, lambda event: None)
    assert service.task_ready() is False
    service.install("yt-dlp", lambda event: None)
    # tokcount is still missing here, and the task starts anyway: it only makes
    # token counting offline, and the pipeline counts through the free
    # countTokens endpoint without it.
    assert service.status("tokcount").usable is False
    assert service.task_ready() is True
    assert service.task_ready(TaskRequest(input="a.wav", stage="final-srt")) is True


def test_worker_context_uses_active_ffmpeg_and_user_settings(
    tmp_path: Path,
) -> None:
    bootstrap = FakeBootstrap(tmp_path)
    runtime = FakeRuntime(tmp_path)
    service = DesktopResourceService(
        bootstrap=bootstrap, runtime=runtime, system_tool_finders={}
    )
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)

    context = service.worker_context({"GEMINI_FREE": "user-key"})

    assert context.environment["GEMINI_FREE"] == "user-key"
    assert context.environment["FFMPEG_BIN"] == str(tmp_path / "ffmpeg" / "bin")


def test_worker_context_pins_the_knowledge_base_to_user_data(
    tmp_path: Path, monkeypatch
) -> None:
    # Left to resolve itself, the knowledge root walks up from the worker's
    # source directory and lands in app/versions/<version>/knowledge -- which
    # the next app update replaces, silently taking the knowledge base with it.
    # The CLI already pins it; the desktop has to agree, or the two disagree
    # about where the same user's knowledge lives.
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    service = DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders={},
    )

    context = service.worker_context({})

    assert context.environment["FINESUB_KNOWLEDGE_ROOT"] == str(
        AppPaths.for_root(tmp_path).user_data / "knowledge"
    )


def test_explicit_knowledge_root_beats_the_launcher_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINESUB_KNOWLEDGE_ROOT", "explicit")
    service = DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders={},
    )

    assert "FINESUB_KNOWLEDGE_ROOT" not in service.worker_context({}).environment


def _service(tmp_path: Path, **finders):
    return DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders=finders,
    )


def test_a_capable_system_ffmpeg_makes_the_download_unnecessary(
    tmp_path: Path,
) -> None:
    found = SystemTool(path=Path("C:/tools/ffmpeg.exe"), version="ffmpeg 7.1")
    service = _service(tmp_path, ffmpeg=lambda: found)

    status = service.status("ffmpeg")
    service.install("ffmpeg", lambda event: None)

    assert status.state == "ready"
    assert "C:/tools/ffmpeg.exe" in status.detail.replace("\\", "/")
    # install() must be a no-op, not a 146MB second copy.
    assert service.bootstrap.installed == []


def test_the_system_probe_runs_once_per_resource(tmp_path: Path) -> None:
    calls: list[str] = []

    def finder():
        calls.append("ffmpeg")
        return None

    service = _service(tmp_path, ffmpeg=finder)
    for _ in range(4):
        service.status("ffmpeg")

    assert calls == ["ffmpeg"]


def test_git_and_yt_dlp_are_required_before_any_task(tmp_path: Path) -> None:
    # 40MB together, and the desktop would rather charge that once than ship an
    # install whose abilities depend on what the user happened to fetch.
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)

    assert service.task_ready() is False
    service.install("git", lambda event: None)
    service.install("yt-dlp", lambda event: None)
    assert service.task_ready() is True


def test_what_a_task_can_start_without_is_what_stays_optional(
    tmp_path: Path,
) -> None:
    # Two ways to be optional, and neither is about size. The weights a task
    # downloads for itself during the run; tokcount it never needs at all,
    # because the LLM layer counts tokens through the free countTokens endpoint
    # without it. Everything else has to be there before the run starts.
    service = _service(tmp_path)

    required = [s.id for s in service.check_all() if not s.optional]
    optional = [s.id for s in service.check_all() if s.optional]

    assert required == ["uv", "ffmpeg", "git", "yt-dlp"]
    assert optional == ["tokcount", "models"]


def test_the_shared_capability_rule_still_names_what_a_request_needs() -> None:
    # The desktop now installs these up front, so `task_ready` cannot tell the
    # cases apart any more -- but the rule is shared with the CLI, which still
    # fetches on demand, and it is the only place the mapping is written down.
    # The knowledge base is a SQLite store now: a knowledge update needs no git.
    assert capability_requirements(
        TaskRequest(input="a.wav", knowledge="update", stage="final-srt")
    ) == ()
    assert capability_requirements(
        TaskRequest(input="https://example.test/watch?v=1")
    ) == ("yt-dlp",)
    # knowledge defaults to "update" and stage to raw-srt; the update only runs
    # in the correction stage, so a default request needs neither.
    assert capability_requirements(TaskRequest(input="a.wav")) == ()


def test_a_url_task_is_refused_until_yt_dlp_is_there(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)
    service.install("git", lambda event: None)
    request = TaskRequest(input="https://example.test/watch?v=1")

    assert service.task_ready(request) is False

    service.install("yt-dlp", lambda event: None)
    assert service.task_ready(request) is True


def test_a_system_git_satisfies_the_requirement_without_downloading_one(
    tmp_path: Path,
) -> None:
    found = SystemTool(path=Path("C:/Program Files/Git/cmd/git.exe"), version="2.44")
    service = _service(tmp_path, git=lambda: found)
    service.install("uv", lambda event: None)
    service.install("ffmpeg", lambda event: None)
    service.install("yt-dlp", lambda event: None)

    assert (
        service.task_ready(
            TaskRequest(input="a.wav", knowledge="update", stage="final-srt")
        )
        is True
    )
    assert service.bootstrap.installed == ["uv", "ffmpeg", "yt-dlp"]


def test_git_goes_on_path_and_yt_dlp_goes_on_pythonpath(tmp_path: Path) -> None:
    # Different injection because they are found differently: git is executed,
    # yt-dlp is imported.
    service = _service(tmp_path)
    service.install("git", lambda event: None)
    service.install("yt-dlp", lambda event: None)

    environment = service.worker_context({}).environment

    assert environment["PATH_DIRS"] == str(tmp_path / "git" / "cmd")
    assert environment["PYTHONPATH_EXTRA"] == str(tmp_path / "yt-dlp")


def test_an_uninstalled_yt_dlp_adds_nothing_to_pythonpath(tmp_path: Path) -> None:
    service = _service(tmp_path)

    assert service.worker_context({}).environment["PYTHONPATH_EXTRA"] == ""


def test_an_installed_tokcount_is_named_for_the_llm_layer(
    tmp_path: Path, monkeypatch
) -> None:
    # Executed by name rather than found on PATH: the pipeline reads the
    # variable first, so pointing at it is what keeps token counting offline.
    monkeypatch.delenv("GEMINI_TOKEN_COUNTER_EXE", raising=False)
    service = _service(tmp_path)
    service.install("tokcount", lambda event: None)

    environment = service.worker_context({}).environment

    assert environment["GEMINI_TOKEN_COUNTER_EXE"] == str(
        tmp_path / "tokcount" / "tokcount.exe"
    )


def test_without_tokcount_the_worker_is_told_nothing_at_all(
    tmp_path: Path, monkeypatch
) -> None:
    # Not an empty string: that would override a counter the pipeline could
    # otherwise have found for itself, and turn "no local binary" into "look
    # for one here, where there is nothing".
    monkeypatch.delenv("GEMINI_TOKEN_COUNTER_EXE", raising=False)
    service = _service(tmp_path)

    assert "GEMINI_TOKEN_COUNTER_EXE" not in service.worker_context({}).environment


def test_a_configured_token_counter_beats_the_managed_one(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("GEMINI_TOKEN_COUNTER_EXE", "C:/mine/tokcount.exe")
    service = _service(tmp_path)
    service.install("tokcount", lambda event: None)

    assert "GEMINI_TOKEN_COUNTER_EXE" not in service.worker_context({}).environment


def test_a_system_token_counter_makes_the_download_unnecessary(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("GEMINI_TOKEN_COUNTER_EXE", raising=False)
    found = SystemTool(path=Path("C:/tools/tokcount.exe"), version="unknown")
    service = _service(tmp_path, tokcount=lambda: found)

    assert service.status("tokcount").state == "ready"
    assert service.worker_context({}).environment[
        "GEMINI_TOKEN_COUNTER_EXE"
    ] == str(found.path)
    assert service.bootstrap.installed == []


def test_tokcount_is_offered_but_never_required(tmp_path: Path) -> None:
    service = _service(tmp_path)

    rows = {status.id: status for status in service.check_all()}

    assert rows["tokcount"].optional is True
    assert rows["yt-dlp"].optional is False


def _isolate_shared_caches(monkeypatch, root: Path) -> tuple[Path, Path]:
    """Point the conventional caches somewhere empty.

    Without this the model row's answer depends on whether the machine running
    the test happens to have downloaded these weights for something else.
    """

    hf = root / "conventional-hf"
    separator = root / "conventional-separator"
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(model_caches, "default_hf_home", lambda: hf)
    monkeypatch.setattr(model_caches, "default_separator_dir", lambda: separator)
    return hf, separator


_CACHE_DIR_BY_MODEL = {
    "whisper": model_caches.WHISPER_CACHE_DIR,
    "qwen-referee": model_caches.QWEN_REFEREE_CACHE_DIR,
}


def test_whisper_cache_name_is_derived_from_the_prefetched_repository() -> None:
    assert TaskRequest.model_fields["model_name"].default == "large-v3-turbo"
    assert model_caches.WHISPER_REPO_ID == (
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    )
    assert model_caches.WHISPER_CACHE_DIR == (
        "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    )


def _install_managed_weights(models_root: Path, *model_ids: str) -> None:
    managed_hf, managed_separator = model_caches.managed_model_dirs(models_root)
    for model_id in model_ids:
        if model_id == "separator":
            managed_separator.mkdir(parents=True, exist_ok=True)
            (managed_separator / model_caches.SEPARATOR_CHECKPOINT).write_bytes(
                b"weights"
            )
        else:
            # A finished snapshot_download: a revision directory under
            # `snapshots/` and no blob left marked `.incomplete`.
            directory = _CACHE_DIR_BY_MODEL[model_id]
            revision = managed_hf / "hub" / directory / "snapshots" / "abc123"
            revision.mkdir(parents=True, exist_ok=True)
            (revision / "config.json").write_text("{}", encoding="utf-8")


def test_the_model_row_reports_how_many_weights_are_still_missing(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_shared_caches(monkeypatch, tmp_path)
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    models_root = service.runtime.paths.models

    assert service.status("models").state == "missing"
    assert "3" in service.status("models").detail

    _install_managed_weights(models_root, "separator", "whisper")
    assert service.status("models").detail.startswith("还需下载 1/3")

    _install_managed_weights(models_root, "qwen-referee")
    assert service.status("models").state == "ready"


def test_weights_the_machine_already_has_elsewhere_count_as_present(
    tmp_path: Path, monkeypatch
) -> None:
    # The launcher points the pipeline at a shared cache when it already holds
    # these weights, so offering to download them again would be a lie.
    _, conventional_separator = _isolate_shared_caches(monkeypatch, tmp_path)
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    _install_managed_weights(
        service.runtime.paths.models, "whisper", "qwen-referee"
    )
    conventional_separator.mkdir(parents=True, exist_ok=True)
    (conventional_separator / model_caches.SEPARATOR_CHECKPOINT).write_bytes(
        b"weights"
    )

    assert service.status("models").state == "ready"


def test_models_cannot_be_fetched_before_the_managed_interpreter(
    tmp_path: Path, monkeypatch
) -> None:
    # There would be no interpreter to run the prefetch with.
    _isolate_shared_caches(monkeypatch, tmp_path)
    calls: list[tuple[str, ...]] = []
    service = DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders={},
        model_prefetch=lambda ids, **kwargs: calls.append(tuple(ids)),
    )

    status = service.status("models")

    assert status.state == "missing"
    assert status.blocked_by == "uv"
    with pytest.raises(RuntimeError):
        service.install("models", lambda event: None)
    assert calls == []


def test_installing_models_only_fetches_the_missing_ones(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_shared_caches(monkeypatch, tmp_path)
    calls: list[tuple[str, ...]] = []

    def prefetch(model_ids, **kwargs):
        calls.append(tuple(model_ids))
        _install_managed_weights(
            service.runtime.paths.models, *model_ids
        )

    service = DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders={},
        model_prefetch=prefetch,
    )
    service.install("uv", lambda event: None)
    _install_managed_weights(service.runtime.paths.models, "separator")

    result = service.install("models", lambda event: None)

    # One process per model: the download endpoint is read at import time, so
    # falling back to the official source means starting again -- and a batch
    # would re-download whatever had already finished.
    assert calls == [("whisper",), ("qwen-referee",)]
    assert result.state == "ready"


def test_the_fallback_attempt_does_not_carry_the_mirror_back_in(
    tmp_path: Path, monkeypatch
) -> None:
    """`worker_context` fills in the configured endpoint for an ordinary run.

    Applied to the fallback attempt as well, it would put the mirror back into
    the very attempt that exists to avoid it -- so a "retry against the
    official source" would silently retry the host that just failed. The unit
    test for `fetch_with_fallback` cannot see this: it never builds a context.
    """

    _isolate_shared_caches(monkeypatch, tmp_path)
    table = tmp_path / "sources.json"
    table.write_text(
        json.dumps({"hfEndpoint": "https://mirror.example"}), encoding="utf-8"
    )
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(table))
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    class InjectingRuntime(FakeRuntime):
        """Applies the endpoint the way the real worker_context does."""

        def worker_context(self, *, ffmpeg_bin, extra_env, **kwargs):
            context = super().worker_context(
                ffmpeg_bin=ffmpeg_bin, extra_env=extra_env, **kwargs
            )
            from finesub_bootstrap import model_fetch

            model_fetch.apply_hf_endpoint(
                context.environment, data_root=self.paths.data_root, region="cn"
            )
            return context

    endpoints: list[str] = []

    def prefetch(model_ids, **kwargs):
        endpoints.append(kwargs["context"].environment.get("HF_ENDPOINT", ""))
        if len(endpoints) == 1:
            # The shape run_model_prefetch really raises: the download happens
            # in a subprocess, so what reaches us is its output tail wrapped in
            # ModelPrefetchFailed, never the original exception.
            raise ModelPrefetchFailed("模型下载失败：Connection reset by peer")
        _install_managed_weights(service.runtime.paths.models, *model_ids)

    service = DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=InjectingRuntime(tmp_path),
        system_tool_finders={},
        model_prefetch=prefetch,
    )
    service.install("uv", lambda event: None)
    _install_managed_weights(service.runtime.paths.models, "separator", "qwen-referee")

    service.install("models", lambda event: None)

    assert endpoints == ["https://mirror.example", ""]


def test_a_successful_prefetch_must_also_pass_the_final_cache_check(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_shared_caches(monkeypatch, tmp_path)
    service = DesktopResourceService(
        bootstrap=FakeBootstrap(tmp_path),
        runtime=FakeRuntime(tmp_path),
        system_tool_finders={},
        model_prefetch=lambda _ids, **_kwargs: None,
    )
    service.install("uv", lambda event: None)

    with pytest.raises(RuntimeError, match="完整性检查"):
        service.install("models", lambda event: None)


def test_an_outdated_tool_does_not_stop_a_task(tmp_path: Path) -> None:
    # The whole reason the state exists: yt-dlp needs regular bumps, and a bump
    # must offer an upgrade, not wall off everyone who already has a copy.
    service = _service(tmp_path)
    for resource_id in ("uv", "ffmpeg", "git", "yt-dlp"):
        service.install(resource_id, lambda event: None)
    service.bootstrap.states["yt-dlp"] = ResourceStatus(
        id="yt-dlp",
        version="2026.08.01",
        installed_version="2026.05.02",
        state="outdated",
    )

    assert service.status("yt-dlp").state == "outdated"
    assert service.task_ready() is True
    assert service.ensure(("yt-dlp",)) == []


def test_an_interrupted_download_does_not_count_as_installed(
    tmp_path: Path, monkeypatch
) -> None:
    # snapshot_download leaves the repository directory behind when it is cut
    # short. Reading that as "installed" told the user the weights were ready
    # while their first task quietly fetched the rest.
    _isolate_shared_caches(monkeypatch, tmp_path)
    service = _service(tmp_path)
    service.install("uv", lambda event: None)
    models_root = service.runtime.paths.models
    _install_managed_weights(models_root, "separator", "whisper", "qwen-referee")
    managed_hf, _ = model_caches.managed_model_dirs(models_root)
    blobs = managed_hf / "hub" / _CACHE_DIR_BY_MODEL["qwen-referee"] / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / "deadbeef.incomplete").write_bytes(b"half a tensor")

    assert model_caches.missing_pipeline_models(models_root) == ("qwen-referee",)
    assert service.status("models").state == "missing"


def test_a_repository_directory_without_a_snapshot_is_not_installed(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_shared_caches(monkeypatch, tmp_path)
    service = _service(tmp_path)
    models_root = service.runtime.paths.models
    _install_managed_weights(models_root, "separator", "qwen-referee")
    managed_hf, _ = model_caches.managed_model_dirs(models_root)
    (managed_hf / "hub" / _CACHE_DIR_BY_MODEL["whisper"]).mkdir(parents=True)

    assert model_caches.missing_pipeline_models(models_root) == ("whisper",)


def test_an_empty_snapshot_revision_is_not_installed(
    tmp_path: Path, monkeypatch
) -> None:
    # snapshot_download creates the revision directory before linking files
    # into it, and no blob is marked .incomplete yet inside that window.
    _isolate_shared_caches(monkeypatch, tmp_path)
    service = _service(tmp_path)
    models_root = service.runtime.paths.models
    _install_managed_weights(models_root, "separator", "whisper")
    managed_hf, _ = model_caches.managed_model_dirs(models_root)
    empty = (
        managed_hf / "hub" / _CACHE_DIR_BY_MODEL["qwen-referee"] / "snapshots" / "abc"
    )
    empty.mkdir(parents=True)

    assert model_caches.missing_pipeline_models(models_root) == ("qwen-referee",)

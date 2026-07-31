from __future__ import annotations

from pathlib import Path

from desktop.backend.common.models import ResourceStatus
from desktop.backend.launcher.bridge import DesktopBridge
from desktop.backend.runtime.environment import WorkerContext
from desktop.backend.settings.store import SettingsStore


class FakeJobs:
    def __init__(self) -> None:
        self.requests = []
        self.worker_env: dict[str, str] = {}

    def start(self, request):
        self.requests.append(request)
        return {
            "taskId": "task-1",
            "state": "running",
            "request": request.model_dump(mode="json"),
            "events": [],
        }

    def snapshot(self):
        return None

    def history(self):
        return []

    def events_after(self, after_cursor=0):
        return [], max(0, int(after_cursor))

    def cancel(self, task_id: str):
        return {"taskId": task_id, "state": "cancelled", "events": []}


class FakeResources:
    ready = True

    def check_all(self):
        return [
            ResourceStatus(
                id="ffmpeg",
                version="7.1",
                state="ready",
            )
        ]

    def task_ready(self):
        return self.ready

    def install(self, resource_id, progress):
        self.ready = True
        return ResourceStatus(id=resource_id, version="1", state="ready")

    def worker_context(self, extra_env):
        return WorkerContext(
            python_executable=Path("C:/FineSub/runtime/python/python.exe"),
            working_directory=Path("C:/FineSub/app/current"),
            environment={**extra_env, "FINESUB_MODEL_DIR": "C:/FineSub/models"},
        )


class FakeUpdates:
    def check(self):
        return {
            "available": True,
            "version": "1.1.0",
            "releaseUrl": "https://github.com/caca2331/finesub/releases/tag/v1.1.0",
        }

    def release_url(self):
        return "https://github.com/caca2331/finesub/releases/tag/v1.1.0"


def _bridge(tmp_path: Path) -> tuple[DesktopBridge, FakeJobs]:
    jobs = FakeJobs()
    bridge = DesktopBridge(
        jobs=jobs,
        resources=FakeResources(),
        settings=SettingsStore(tmp_path / "user-data"),
    )
    return bridge, jobs


def test_bridge_rejects_unknown_task_fields(tmp_path: Path) -> None:
    bridge, _ = _bridge(tmp_path)

    result = bridge.start_task({"input": "a.wav", "command": "calc.exe"})

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_request"


def test_bridge_blocks_translation_without_key_but_allows_raw(
    tmp_path: Path,
) -> None:
    bridge, jobs = _bridge(tmp_path)

    translated = bridge.start_task({"input": "a.wav", "stage": "final-srt"})
    raw = bridge.start_task({"input": "a.wav", "stage": "raw-srt"})

    assert translated["ok"] is False
    assert translated["error"]["code"] == "api_key_required"
    assert translated["error"]["action"] == "open_settings"
    assert raw["ok"] is True
    assert jobs.requests[-1].stage == "raw-srt"


def test_bridge_requests_runtime_install_before_worker_launch(
    tmp_path: Path,
) -> None:
    jobs = FakeJobs()
    resources = FakeResources()
    resources.ready = False
    bridge = DesktopBridge(
        jobs=jobs,
        resources=resources,
        settings=SettingsStore(tmp_path / "user-data"),
    )

    result = bridge.start_task({"input": "a.wav", "stage": "raw-srt"})

    assert result["ok"] is False
    assert result["error"]["code"] == "runtime_required"
    assert result["error"]["action"] == "open_resources"
    assert jobs.requests == []


def test_save_api_keys_returns_only_configuration_status(tmp_path: Path) -> None:
    bridge, jobs = _bridge(tmp_path)

    result = bridge.save_api_keys(
        {"gemini": "private-gemini-key", "exa": "", "tavily": ""}
    )

    assert result["ok"] is True
    assert result["data"]["api_keys"]["gemini"] == "configured"
    assert "private-gemini-key" not in str(result)
    assert jobs.worker_env["GEMINI_FREE"] == "private-gemini-key"


def test_bootstrap_state_reports_resources_and_optional_capabilities(
    tmp_path: Path,
) -> None:
    bridge, _ = _bridge(tmp_path)

    result = bridge.get_bootstrap_state()

    assert result["ok"] is True
    assert result["data"]["app_version"] == "development"
    assert result["data"]["resources"][0]["state"] == "ready"
    assert result["data"]["capabilities"]["raw_srt"] is True
    assert result["data"]["capabilities"]["translation"] is False


def test_resource_install_refreshes_worker_context(tmp_path: Path) -> None:
    bridge, jobs = _bridge(tmp_path)

    result = bridge.install_resource("uv")

    assert result["ok"] is True
    assert jobs.python_executable.endswith("python.exe")
    assert jobs.working_directory == "C:\\FineSub\\app\\current"
    assert jobs.worker_env["FINESUB_MODEL_DIR"] == "C:/FineSub/models"


def test_update_check_opens_release_page_without_installing(
    tmp_path: Path,
) -> None:
    opened: list[str] = []
    jobs = FakeJobs()
    bridge = DesktopBridge(
        jobs=jobs,
        resources=FakeResources(),
        settings=SettingsStore(tmp_path / "user-data"),
        updates=FakeUpdates(),
        url_opener=opened.append,
    )

    checked = bridge.check_updates()
    opened_result = bridge.open_update_page()

    assert checked["data"]["available"] is True
    assert opened_result["ok"] is True
    assert opened == [
        "https://github.com/caca2331/finesub/releases/tag/v1.1.0"
    ]


def test_open_output_accepts_any_saved_task_but_rejects_other_paths(
    tmp_path: Path,
) -> None:
    older_output = tmp_path / "older" / "subtitle.srt"
    current_output = tmp_path / "current" / "subtitle.srt"
    opened: list[Path] = []

    class JobsWithHistory(FakeJobs):
        def snapshot(self):
            return {"outputs": {"rawSrt": str(current_output)}}

        def history(self):
            return [
                {"outputs": {"rawSrt": str(current_output)}},
                {"outputs": {"finalSrt": str(older_output)}},
            ]

    bridge = DesktopBridge(
        jobs=JobsWithHistory(),
        resources=FakeResources(),
        settings=SettingsStore(tmp_path / "user-data"),
        output_opener=opened.append,
    )

    accepted = bridge.open_output(str(older_output))
    rejected = bridge.open_output(str(tmp_path / "not-a-task-output.txt"))

    assert accepted["ok"] is True
    assert opened == [older_output.resolve()]
    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "invalid_output"

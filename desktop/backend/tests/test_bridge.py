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
    assert jobs.worker_env["GEMINI_API_KEY"] == "private-gemini-key"


def test_bootstrap_state_reports_resources_and_optional_capabilities(
    tmp_path: Path,
) -> None:
    bridge, _ = _bridge(tmp_path)

    result = bridge.get_bootstrap_state()

    assert result["ok"] is True
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

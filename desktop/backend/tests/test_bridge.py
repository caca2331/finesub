from __future__ import annotations

from pathlib import Path

from finesub_bootstrap.models import ResourceStatus
from desktop.backend.common.models import TaskRequest
from desktop.backend.jobs.launch import WorkerLaunchContext
from desktop.backend.launcher.bridge import DesktopBridge
from finesub_bootstrap.environment import WorkerContext
from desktop.backend.settings.store import SettingsStore


class FakeJobs:
    def __init__(self) -> None:
        self.requests = []
        self.worker_context = WorkerLaunchContext(
            python_executable="python.exe",
            working_directory=None,
            environment={},
        )

    def set_worker_context(self, context: WorkerLaunchContext) -> None:
        self.worker_context = context

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

    def open_task_directory(self, task_id, opener):
        target = Path(task_id or ".").resolve()
        opener(target)
        return target

    def open_owned_output(self, output_path, opener):
        path = Path(output_path).expanduser().resolve()
        owned = {
            Path(value).expanduser().resolve()
            for snapshot in self.history()
            for value in snapshot.get("outputs", {}).values()
        }
        if path not in owned:
            raise ValueError("Output path is not owned by a saved task")
        opener(path)
        return path

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

    def task_ready(self, request=None):
        return self.ready

    def ensure(self, resource_ids):
        # The bridge now asks which of *these* are missing, so the fake has to
        # answer per resource rather than with one global flag.
        return [] if self.ready else list(resource_ids)

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


class FakeTray:
    def __init__(self) -> None:
        self.hidden = False

    def hide_window(self) -> None:
        self.hidden = True


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
    assert jobs.worker_context.environment["GEMINI_FREE"] == "private-gemini-key"


def test_reveal_api_keys_returns_plaintext_entries(tmp_path: Path) -> None:
    bridge, _ = _bridge(tmp_path)
    bridge.save_api_keys({"gemini": "private-gemini-key-123"})

    result = bridge.reveal_api_keys()

    assert result["ok"] is True
    entries = result["data"]["gemini"]
    assert entries[0]["key"] == "private-gemini-key-123"
    assert entries[0]["masked"] == "priv…-123"
    assert result["data"]["exa"] == []


def test_preferences_round_trip_through_the_bridge(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "config.toml"))
    bridge, _ = _bridge(tmp_path)

    saved = bridge.save_preferences(
        {"ui": {"language": "en"}, "task_defaults": {"gpu_budget_gb": 8}}
    )
    assert saved["ok"] is True

    loaded = bridge.get_preferences()

    assert loaded["ok"] is True
    assert loaded["data"]["preferences"]["ui"] == {"language": "en"}
    assert loaded["data"]["preferences"]["task_defaults"]["gpu_budget_gb"] == 8
    # The shared half travels with it: one call is what the panel needs to draw.
    assert loaded["data"]["shared"]["split_length_scale"] is None
    assert loaded["data"]["config_path"].endswith("config.toml")


def test_unset_task_defaults_never_reach_the_wire(tmp_path: Path, monkeypatch) -> None:
    # The front end spreads task_defaults straight over the task request, so a
    # null here is not "unset" -- it overwrites a real default and every task
    # start then fails validation. Assert on the serialized payload, not on the
    # model: that is where the nulls used to appear.
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "config.toml"))
    bridge, _ = _bridge(tmp_path)
    bridge.save_preferences({"task_defaults": {"gpu_budget_gb": 8}})

    defaults = bridge.get_bootstrap_state()["data"]["preferences"]["task_defaults"]

    assert defaults == {"gpu_budget_gb": 8}
    assert TaskRequest.model_validate({"input": "a.mp4", **defaults}).gpu_budget_gb == 8


def test_saving_one_preference_section_leaves_the_other_alone(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "config.toml"))
    bridge, _ = _bridge(tmp_path)
    bridge.save_preferences({"ui": {"language": "en"}})

    bridge.save_preferences({"task_defaults": {"gpu_budget_gb": 8}})
    # null clears one setting without touching the rest of its section.
    bridge.save_preferences({"ui": {"closeWindowAction": "close"}})
    bridge.save_preferences({"ui": {"closeWindowAction": None}})

    preferences = bridge.get_preferences()["data"]["preferences"]
    assert preferences["ui"] == {"language": "en"}
    assert preferences["task_defaults"]["gpu_budget_gb"] == 8


def test_saving_a_shared_setting_refreshes_the_worker_environment(
    tmp_path: Path, monkeypatch
) -> None:
    # The worker learns config.toml's location from its launch environment,
    # which only names the file if it already existed when the launcher started.
    # Creating it from the panel therefore has to rebuild that environment, or
    # the next task runs at the default and the panel looks broken.
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "config.toml"))
    bridge, jobs = _bridge(tmp_path)
    before = jobs.worker_context

    result = bridge.save_shared_settings({"split_length_scale": 0.8})

    assert result["ok"] is True
    assert jobs.worker_context is not before


def test_an_invalid_shared_setting_is_refused_with_a_reason(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "config.toml"))
    bridge, _ = _bridge(tmp_path)

    result = bridge.save_shared_settings({"split_length_scale": 4.0})

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_settings"
    assert "length scale" in result["error"]["message"]
    assert not (tmp_path / "config.toml").exists()


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
    assert jobs.worker_context.python_executable.endswith("python.exe")
    assert jobs.worker_context.working_directory == "C:\\FineSub\\app\\current"
    assert jobs.worker_context.environment["FINESUB_MODEL_DIR"] == "C:/FineSub/models"


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


def test_minimize_to_tray_hides_window_through_tray_controller(
    tmp_path: Path,
) -> None:
    tray = FakeTray()
    bridge = DesktopBridge(
        jobs=FakeJobs(),
        resources=FakeResources(),
        settings=SettingsStore(tmp_path / "user-data"),
        tray=tray,
    )

    result = bridge.minimize_to_tray()

    assert result["ok"] is True
    assert tray.hidden is True


def test_maximize_button_toggles_between_maximized_and_normal(
    tmp_path: Path,
) -> None:
    class NativeWindow:
        WindowState = "Normal"

    class FakeWindow:
        native = NativeWindow()

        def maximize(self) -> None:
            self.native.WindowState = "Maximized"

        def restore(self) -> None:
            self.native.WindowState = "Normal"

    window = FakeWindow()
    bridge = DesktopBridge(
        jobs=FakeJobs(),
        resources=FakeResources(),
        settings=SettingsStore(tmp_path / "user-data"),
        window=window,
    )

    assert bridge.maximize_window()["ok"] is True
    assert window.native.WindowState == "Maximized"
    assert bridge.maximize_window()["ok"] is True
    assert window.native.WindowState == "Normal"


def test_the_missing_resource_is_named_rather_than_guessed(tmp_path: Path) -> None:
    # "请先安装 Python 运行环境和 FFmpeg" was the whole message regardless of what
    # was actually missing. With git and yt-dlp installed on demand, a user told
    # to install FFmpeg when yt-dlp is what is missing has no way to act on it.
    class OnlyYtDlpMissing:
        def check_all(self):
            return []

        def ensure(self, resource_ids):
            return [item for item in resource_ids if item == "yt-dlp"]

    bridge = DesktopBridge(
        jobs=FakeJobs(),
        resources=OnlyYtDlpMissing(),
        settings=SettingsStore(tmp_path / "user-data"),
    )

    result = bridge.start_task(
        {"input": "https://example.test/v", "stage": "raw-srt"}
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "runtime_required"
    assert "yt-dlp" in result["error"]["message"]
    assert "FFmpeg" not in result["error"]["message"]

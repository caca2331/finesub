from __future__ import annotations

from pathlib import Path

import pytest

from finesub_bootstrap.downloader import DownloadPaused

from desktop.backend.resources.model_prefetch import (
    ModelPrefetchFailed,
    run_model_prefetch,
)


class FakeContext:
    def __init__(self, root: Path) -> None:
        self.python_executable = root / "python.exe"
        self.working_directory = root / "app"
        self.environment = {"HF_HOME": str(root / "models" / "huggingface")}


class FakeProcess:
    def __init__(
        self,
        lines: list[str],
        returncode: int = 0,
        running_polls: int = 0,
    ) -> None:
        self.stdout = iter(f"{line}\n" for line in lines)
        self._returncode = returncode
        self._running_polls = running_polls
        self.terminated = False
        self.killed = False

    @property
    def returncode(self) -> int | None:
        return None if self._running_polls > 0 else self._returncode

    def poll(self) -> int | None:
        if self._running_polls > 0:
            self._running_polls -= 1
            return None
        return self._returncode

    def wait(self, timeout=None) -> int:
        self._running_polls = 0
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._running_polls = 0

    def kill(self) -> None:
        self.killed = True
        self._running_polls = 0


def _spawn(lines: list[str], returncode: int = 0, running_polls: int = 0):
    captured: dict[str, object] = {}

    def factory(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        captured["process"] = FakeProcess(lines, returncode, running_polls)
        return captured["process"]

    return factory, captured


def test_the_managed_interpreter_runs_the_prefetch_module(tmp_path: Path) -> None:
    factory, captured = _spawn(["Whisper 识别模型已就绪"])

    run_model_prefetch(
        ["whisper"],
        context=FakeContext(tmp_path),
        process_factory=factory,
    )

    assert captured["command"] == [
        str(tmp_path / "python.exe"),
        "-m",
        "desktop.backend.worker.prefetch",
        "whisper",
    ]
    # The launcher already resolved cache reuse into this environment; losing it
    # would download weights the machine has into a directory nothing reads.
    assert captured["env"]["HF_HOME"] == str(tmp_path / "models" / "huggingface")
    assert captured["cwd"] == str(tmp_path / "app")


def test_stage_lines_become_stage_callbacks_and_the_rest_are_logs(
    tmp_path: Path,
) -> None:
    factory, _ = _spawn(
        [
            "STAGE 1/2 正在获取人声分离模型",
            "downloading checkpoint",
            "STAGE 2/2 正在获取 Whisper 识别模型",
        ]
    )
    stages: list[tuple[str, str]] = []
    logs: list[str] = []

    run_model_prefetch(
        ["separator", "whisper"],
        context=FakeContext(tmp_path),
        stage=lambda phase, message: stages.append((phase, message)),
        log=logs.append,
        process_factory=factory,
    )

    assert stages == [
        ("downloading", "正在获取人声分离模型（1/2）"),
        ("downloading", "正在获取 Whisper 识别模型（2/2）"),
    ]
    # A STAGE line is an announcement, not output; duplicating it into the log
    # would double every heading in the transcript.
    assert logs == ["downloading checkpoint"]


def test_a_failing_prefetch_reports_the_last_thing_it_said(tmp_path: Path) -> None:
    factory, _ = _spawn(
        ["STAGE 1/1 正在获取 Qwen 校验模型", "Qwen 校验模型（qwen-referee）获取失败：OSError: no route"],
        returncode=1,
    )

    with pytest.raises(ModelPrefetchFailed) as failure:
        run_model_prefetch(
            ["qwen-referee"],
            context=FakeContext(tmp_path),
            process_factory=factory,
        )

    assert "qwen-referee" in str(failure.value)


def test_pausing_raises_the_pause_signal_the_installer_understands(
    tmp_path: Path,
) -> None:
    # A plain return means "finished" to ResourceInstallManager, which would
    # mark the row 已安装 with most of the weights still missing. Each
    # downloader resumes by itself next time, so stopping is the whole pause
    # protocol -- but it has to arrive as this exception.
    factory, captured = _spawn(
        ["downloading", "still downloading"],
        returncode=1,
        running_polls=10,
    )

    with pytest.raises(DownloadPaused):
        run_model_prefetch(
            ["whisper"],
            context=FakeContext(tmp_path),
            should_pause=lambda: True,
            process_factory=factory,
        )

    assert captured["process"].terminated is True


def test_pause_is_checked_even_when_the_child_prints_nothing(tmp_path: Path) -> None:
    factory, captured = _spawn([], returncode=1, running_polls=10)

    with pytest.raises(DownloadPaused):
        run_model_prefetch(
            ["whisper"],
            context=FakeContext(tmp_path),
            should_pause=lambda: True,
            process_factory=factory,
        )

    assert captured["process"].terminated is True

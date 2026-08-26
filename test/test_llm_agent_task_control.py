from __future__ import annotations

from io import StringIO
import json

from finesub.llm.agent.agent_task_control import main
from finesub.llm.agent.agent_task_runtime import AgentTaskRuntime, AgentTaskSpec


def _runtime(tmp_path) -> AgentTaskRuntime:
    return AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="finish",
        tasks=[
            AgentTaskSpec(
                task_id="task-1",
                session_type="research",
                input_hash="sha256:input",
                goal="answer",
            )
        ],
    )


def test_control_cli_emits_one_json_object_for_status_and_claim(tmp_path, capsys) -> None:
    runtime = _runtime(tmp_path)
    common = [
        "--root",
        str(runtime.root),
        "--assignment",
        "assignment-1",
    ]
    assert main([*common, "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "ready"

    assert main(
        [
            *common,
            "next-task",
            "--kind",
            "headless",
            "--worker",
            "worker-1",
            "--request-id",
            "claim",
            "--control-generation",
            str(status["control_generation"]),
        ]
    ) == 0
    claimed = json.loads(capsys.readouterr().out)
    assert claimed["status"] == "task"
    assert claimed["task"]["task_id"] == "task-1"


def test_control_cli_reads_submit_json_from_stdin(tmp_path, capsys, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    monkeypatch.setattr("sys.stdin", StringIO('{"answer":"ok"}'))
    code = main(
        [
            "--root",
            str(runtime.root),
            "--assignment",
            "assignment-1",
            "submit",
            "--worker",
            "worker-1",
            "--request-id",
            "submit",
            "--task",
            "task-1",
            "--lease-generation",
            str(claimed["task"]["lease_generation"]),
            "--input-hash",
            "sha256:input",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "assignment_complete"


def test_control_cli_keeps_errors_off_stdout(tmp_path, capsys) -> None:
    runtime = _runtime(tmp_path)
    assert main(
        [
            "--root",
            str(runtime.root),
            "--assignment",
            "wrong",
            "status",
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "error"

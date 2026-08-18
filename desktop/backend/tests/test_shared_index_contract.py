"""What the desktop promises about the index both front ends write.

The rest of the shared naming and placement rules moved to the root suite
(`test/bootstrap/test_task_output.py`); this one stays because it is the
desktop's half of the contract, and asserting it needs the desktop's model.
"""

from __future__ import annotations

from pathlib import Path

from desktop.backend.jobs.history import JobSnapshot
from finesub_bootstrap import task_index


def test_the_desktop_can_read_a_shared_index_entry(tmp_path: Path) -> None:
    """The schema claim that makes one shared index possible.

    The desktop validates entries into a model that forbids extras. If an
    entry failed that, it would be skipped on read -- and an older desktop's
    next write could make it vanish rather than merely look sparse.
    """

    index = tmp_path / "tasks.json"
    task_index.merge_write(
        index,
        [
            {
                "task_id": "clip-260811-2205-abc123",
                "state": "completed",
                "request": {"input": "D:/media/clip.wav", "stage": "raw-srt"},
                "outputs": {},
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        ],
        tmp_path / "tasks",
    )

    stored = task_index.read(index, tmp_path / "tasks")
    snapshot = JobSnapshot.model_validate(stored[0])

    assert snapshot.task_id == "clip-260811-2205-abc123"
    assert snapshot.request.input == "D:/media/clip.wav"
    # This fixture omits the optional fields; the shared shape still accepts it.
    assert snapshot.request.model_name == "large-v3-turbo"

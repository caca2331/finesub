"""Session kind registry for ``tools.session_replay``."""

from __future__ import annotations

from typing import Callable, Mapping, Protocol

from .sessions.correction import CorrectionSessionAdapter
from .sessions.task_flow import PrepareTaskSessionAdapter, WindowRequestSessionAdapter
from .sessions.three_pass import ProofreadSessionAdapter, SpeakerJudgeSessionAdapter


class SessionAdapter(Protocol):
    """Extensible hook: load/build fixture, assemble messages, sample replies."""

    name: str

    def ensure_fixture(self, **kwargs):  # noqa: ANN003
        ...

    def run(self, **kwargs):  # noqa: ANN003
        ...


SESSIONS: dict[str, Callable[[], SessionAdapter]] = {
    "correction": CorrectionSessionAdapter,
    "prepare-task": PrepareTaskSessionAdapter,
    "correction-request": WindowRequestSessionAdapter,
    "speaker-judge": SpeakerJudgeSessionAdapter,
    "proofread": ProofreadSessionAdapter,
}


def get_session(name: str) -> SessionAdapter:
    key = (name or "").strip().lower()
    factory = SESSIONS.get(key)
    if factory is None:
        known = ", ".join(sorted(SESSIONS))
        raise ValueError(f"Unknown session {name!r}; known: {known}")
    return factory()


def list_sessions() -> Mapping[str, str]:
    return {
        "correction": "纠错窗口的交付轮（复用冻结的 search/extract 等注入）",
        "prepare-task": "准备任务的首条消息（请求批或直接交付；--agentic 看工具会话形态）",
        "correction-request": "纠错窗口的首条消息（retrieval=local 带请求通道；--agentic 看工具会话形态）",
        "speaker-judge": "说话人裁判的一个单元（三个 pass 的 pass 1 也是它；--chunk 是单元 id）",
        "proofread": "三个 pass 的 pass 3：校对一个窗",
    }

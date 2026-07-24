"""Session kind registry for ``tools.session_replay``."""

from __future__ import annotations

from typing import Callable, Mapping, Protocol

from .sessions.correction import CorrectionSessionAdapter
from .sessions.fast_round import FastRound1SessionAdapter
from .sessions.query import QuerySessionAdapter
from .sessions.research import ResearchR1SessionAdapter, ResearchR2SessionAdapter
from .sessions.search_judge import SearchJudgeSessionAdapter


class SessionAdapter(Protocol):
    """Extensible hook: load/build fixture, assemble messages, sample replies."""

    name: str

    def ensure_fixture(self, **kwargs):  # noqa: ANN003
        ...

    def run(self, **kwargs):  # noqa: ANN003
        ...


SESSIONS: dict[str, Callable[[], SessionAdapter]] = {
    "correction": CorrectionSessionAdapter,
    "query": QuerySessionAdapter,
    "research-r1": ResearchR1SessionAdapter,
    "research-r2": ResearchR2SessionAdapter,
    "search-judge": SearchJudgeSessionAdapter,
    "fast-round1": FastRound1SessionAdapter,
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
        "correction": "纠错窗口 R2（复用冻结的 R1 search/extract 等注入）",
        "query": "每窗查询轮（search queries / window notes / entry requests）",
        "research-r1": "背景调查 R1（queries / notes / 词条请求）",
        "research-r2": "背景调查 R2（background context 组装）",
        "search-judge": "搜索循环 judge 轮（停机 / 消化 / 词条决策）",
        "fast-round1": "fast 模式合一轮（query + research 融合）",
    }

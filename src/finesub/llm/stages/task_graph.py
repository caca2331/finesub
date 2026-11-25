"""Task graph: which business tasks one correction run dispatches, with what grants.

Owner doc: ``docs/llm_harness_behavior.md`` -- 「开关组合 → 实际会跑的会话」 is the
main matrix (execution strategy × normal/fast) and the capability
matrix (search × knowledge). Stage A of the 2026-09 task-flow rework landed here: the graph is a
pure function of the strategy and the fast decision, and a capability only
changes what a task may *call* -- never which tasks exist. Two consequences
the tests pin:

* the graph never sees a knowledge root. An empty or disabled base is an
  empty tool result inside a task, not another run shape (§4: "KB 开启但为空，
  只是空结果，不是另一条执行形状");
* ``batched`` and ``agentic`` dispatch the same task list. The strategy says
  how a task gets its inputs (one batch of tool requests continued in the same
  logical conversation, or a tool session driven by the executor), not what
  the task is (§2: "一个任务契约，两种执行策略").

The graph is recorded as the ``task_graph`` artifact next to
``fast_decision`` so every run can be audited against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from finesub.reporting import current_reporter

from ..correction_request import KNOWLEDGE_MODES
from ..routing.profiles import TranslationProfile

#: How a task obtains what it lacks. ``batched``: the executor asks for every
#: independent tool result in one go, the harness fills them in and the same
#: conversation continues to the deliverable (request-count-bound executors,
#: chiefly free Gemini). ``agentic``: the executor calls tools as it goes
#: inside one tool session (local agents; an API with a real tool loop later).
EXECUTION_STRATEGIES = ("batched", "agentic")
DEFAULT_EXECUTION_STRATEGY = "batched"

#: The three business tasks of §3. ``prepare`` is the global preparation
#: (material understanding, sources, cross-window conventions); ``translate``
#: is one correction/translation deliverable; ``knowledge_update`` checks the
#: names the task searched, then proposes into the base after the product
#: exists, and is the only write channel.
TASK_KINDS = ("prepare", "translate", "knowledge_update")

#: ``global``: once per run. ``window``: once per planned correction window
#: (the count is a planner fact, not part of the graph). ``whole``: the fast
#: shape -- one task takes the entire input and folds preparation into it.
TASK_SCOPES = ("global", "window", "whole")

SEARCH_MODES = ("none", "local", "native")

#: Tool vocabulary of the grants. Search names its implementation because the
#: user chose it (§4: no running ``local`` under a ``native`` request or the
#: reverse); the two never appear together.
TOOL_SEARCH_LOCAL = "search:local"
TOOL_SEARCH_NATIVE = "search:native"
TOOL_KB_READ = "kb:read"
TOOL_KB_PROPOSE = "kb:propose"
TOOL_STYLE_READ = "style:read"


@dataclass(frozen=True)
class CapabilityGrant:
    """The six-cell matrix of §4 plus the independent style authorization.

    ``search`` is ``--retrieval`` verbatim, ``knowledge`` is the resolved
    ``--knowledge`` switch, ``style`` says a style entry was selected for this
    run. ``knowledge="none"`` does not grant the base *and* does not revoke the
    style: the two are authorized separately (§4).
    """

    search: str = "none"
    knowledge: str = "none"
    style: bool = False

    def __post_init__(self) -> None:
        if self.search not in SEARCH_MODES:
            raise ValueError(f"unknown search mode {self.search!r}; use none|local|native")
        if self.knowledge not in KNOWLEDGE_MODES:
            raise ValueError(
                f"unknown knowledge mode {self.knowledge!r}; use none|read|update"
            )

    @property
    def reads_knowledge(self) -> bool:
        return self.knowledge != "none"

    @property
    def updates_knowledge(self) -> bool:
        return self.knowledge == "update"

    def read_tools(self) -> tuple[str, ...]:
        """What a preparation or translation task may call."""

        tools: list[str] = []
        if self.search == "local":
            tools.append(TOOL_SEARCH_LOCAL)
        elif self.search == "native":
            tools.append(TOOL_SEARCH_NATIVE)
        if self.reads_knowledge:
            tools.append(TOOL_KB_READ)
        if self.style:
            tools.append(TOOL_STYLE_READ)
        return tuple(tools)

    def to_metadata(self) -> dict[str, Any]:
        return {"search": self.search, "knowledge": self.knowledge, "style": self.style}


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    scope: str
    tools: tuple[str, ...] = ()
    #: How *this* task gets what it lacks. Per task, because the chains differ
    #: and so do their transports: the preparation task is text and may run a
    #: tool session where the media-bearing window cannot -- ``agy`` plus the
    #: production default ``--media audio`` is exactly that split. ``""`` = not
    #: resolved (the knowledge update runs on its own chain and has no request
    #: channel either way).
    strategy: str = ""

    def __post_init__(self) -> None:
        if self.kind not in TASK_KINDS:
            raise ValueError(f"unknown task kind {self.kind!r}")
        if self.scope not in TASK_SCOPES:
            raise ValueError(f"unknown task scope {self.scope!r}")
        if self.strategy and self.strategy not in EXECUTION_STRATEGIES:
            raise ValueError(f"unknown execution strategy {self.strategy!r}")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scope": self.scope,
            "tools": list(self.tools),
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class TaskGraph:
    #: The **window** task's strategy -- the run's dispatch strategy. Each task
    #: carries its own; read those when a chain split matters.
    strategy: str
    fast: bool
    grant: CapabilityGrant
    tasks: tuple[TaskSpec, ...]

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(task.kind for task in self.tasks)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fast": self.fast,
            "grant": self.grant.to_metadata(),
            "tasks": [task.to_metadata() for task in self.tasks],
        }


def plan_task_graph(
    *,
    strategy: str = DEFAULT_EXECUTION_STRATEGY,
    fast_enabled: bool,
    grant: CapabilityGrant,
    prepare_strategy: str | None = None,
    correct_only: bool = False,
) -> TaskGraph:
    """The four-cell main matrix (§3) over the six-cell grant (§4).

    Normal: one global ``prepare`` then ``translate`` per window. Fast: one
    ``translate`` over the whole input that folds preparation in -- because
    the whole input fits one deliverable, not because of any search backend.
    ``knowledge_update`` follows the product whenever the switch says
    ``update``, in both columns, and is never folded into the fused task
    (§3: "fast 不把知识更新塞进翻译任务"). It opens with the name check
    (docs/knowledge.md「中文定名的核查」), which searches; a correct-only
    task's stage is that check alone, no proposals.

    Deliberately takes no knowledge root and no index: with the base enabled
    but empty, the read tool simply returns nothing.
    """

    if strategy not in EXECUTION_STRATEGIES:
        raise ValueError(
            f"unknown execution strategy {strategy!r}; use {'|'.join(EXECUTION_STRATEGIES)}"
        )
    read_tools = grant.read_tools()
    tasks: list[TaskSpec] = []
    if fast_enabled:
        tasks.append(
            TaskSpec(kind="translate", scope="whole", tools=read_tools, strategy=strategy)
        )
    else:
        tasks.append(
            TaskSpec(
                kind="prepare",
                scope="global",
                tools=read_tools,
                # Its own chain answers it, so its own strategy: recording the
                # window's here would call a run "batched" whose preparation
                # task fetched for itself.
                strategy=prepare_strategy or strategy,
            )
        )
        tasks.append(
            TaskSpec(kind="translate", scope="window", tools=read_tools, strategy=strategy)
        )
    if grant.updates_knowledge:
        # the check's search is the stage's own (the route's native search, else local), not the run's grant
        tasks.append(TaskSpec(kind="knowledge_update", scope="global", tools=() if correct_only else (TOOL_KB_PROPOSE,)))
    return TaskGraph(strategy=strategy, fast=fast_enabled, grant=grant, tasks=tuple(tasks))


def grant_for_run(
    profile: TranslationProfile, *, knowledge: str, style_selected: bool
) -> CapabilityGrant:
    """Today's switches → the grant. ``retrieval`` is the search axis verbatim."""

    return CapabilityGrant(
        search=profile.retrieval, knowledge=knowledge, style=bool(style_selected)
    )


def strategy_for_executors(
    backends: Iterable[str],
    *,
    tool_session: bool = True,
    override: str | None = None,
) -> str:
    """Which strategy a set of candidate executors can carry.

    ``agentic`` needs every executor that may answer to drive a tool session,
    so a chain with any API member (free or paid) resolves to ``batched`` --
    the bounded strategy is the one both kinds can execute. ``tool_session``
    is the second half of the same question: an all-agent chain whose calls
    take the **capsule** transport (media, or the ``api``/``resume`` session
    modes) has no tools either, and picking ``agentic`` there would stand the
    harness down while telling the model to fetch for itself -- i.e. turn
    ``retrieval=local`` into ``none`` in silence
    (``capabilities.chain_serves_tool_session``). An explicit ``override``
    (the future execution-profile key of plan §9.3) wins after validation; the
    model *name* never decides (§2: no ``free``-in-the-name or HTTP-vs-CLI
    inference).
    """

    if override is not None:
        if override not in EXECUTION_STRATEGIES:
            raise ValueError(
                f"unknown execution strategy {override!r}; use {'|'.join(EXECUTION_STRATEGIES)}"
            )
        return override
    kinds = tuple(backends)
    if kinds and tool_session and all(kind == "local_agent" for kind in kinds):
        return "agentic"
    return DEFAULT_EXECUTION_STRATEGY


#: Pins already warned about, so a run that resolves the same task group four
#: times says it once. Process-global on purpose: both callers are dev-only
#: diagnostics, and the pin does not change within a process.
_WARNED_FORCED: set[tuple[str, str, str]] = set()


def _warn_if_forced_past_the_gate(
    profile: TranslationProfile,
    strategy: str,
    *,
    task_group: str,
    source: str,
    knowledge_readable: bool = False,
) -> None:
    """Say so when a forced ``agentic`` lands somewhere the tools cannot follow.

    The pin exists to hold a strategy fixed for an A/B (plan §11), so it keeps
    its power -- forcing ``batched`` onto an agent is exactly what the
    acceptance runs did. But forcing ``agentic`` where the derivation would
    have said ``batched`` recreates the very failure that derivation exists to
    prevent: the harness stands its request turn down and the executor has no
    tools, so ``retrieval=local`` quietly becomes ``none``. A pinned run that
    measures *that* would be measuring the wrong thing without knowing.

    The judgement is the derivation itself, re-run and compared -- not a copy
    of half of it. Asking only ``chain_serves_tool_session`` missed both shapes
    that have no agent at all: it filters to agent members and then takes
    ``all()``, so a pure-API chain answers ``all([]) is True`` and a mixed
    chain is judged by its agent members alone.
    """

    # Only where there is something to lose: harness search, or -- since the
    # window request channel covers knowledge too -- a readable base, whose
    # entry requests a forced agentic strands the same way.
    if strategy != "agentic" or not (profile.external_injection or knowledge_readable):
        return
    from ..routing.capabilities import chain_backends, chain_serves_tool_session

    backends = chain_backends(profile, task_group=task_group)
    all_agents = bool(backends) and all(kind == "local_agent" for kind in backends)
    # Short-circuit before the transport question: it may probe a driver, and
    # a chain that is not all agents is already the answer.
    if all_agents and strategy_for_executors(
        backends, tool_session=chain_serves_tool_session(profile, task_group=task_group)
    ) == "agentic":
        return
    key = (source, task_group, profile.profile_id)
    if key in _WARNED_FORCED:
        return
    _WARNED_FORCED.add(key)
    why = (
        "这条链上有不是 local_agent 的成员，它们没有 harness 工具"
        if not all_agents
        else "这条链的调用拿不到 harness 工具"
        "（带媒体、或会话档位是 api/resume、或 driver 不接受 per-invocation MCP server）"
    )
    current_reporter().warning(
        "task-strategy-pin-without-tools",
        f"{source} 把「{task_group}」钉成 agentic，但{why}",
        impact="harness 不再代取，模型也没有检索与查库工具——这一格的搜索与知识库词条请求都拿不到",
        action="去掉这个钉法，或换一条全是 agent、纯文本、per-window 且认 MCP 的链",
    )


def resolve_execution_strategy(
    profile: TranslationProfile,
    *,
    test_profile: bool = False,
    override: str | None = None,
    task_group: str = "correction",
    knowledge_readable: bool = False,
) -> tuple[str, str]:
    """``(strategy, source)`` for one task, resolved once at dispatch.

    The executors consulted are the usable endpoints of the chain that
    serves ``task_group`` -- ``correction`` (the window / whole-input task) or
    ``research`` (the preparation task) -- i.e. the members the router may
    actually hand that deliverable to, **and** whether that task's agent calls
    would really reach the harness tools. A ``test_profile`` run pins one stub
    endpoint that says nothing about tools, so it takes the default.
    ``source`` names where the answer came from (``override`` /
    ``test_profile`` / ``<task_group>_chain``) so the recorded artifact can
    be read without re-deriving it.
    """

    if override is not None:
        forced = strategy_for_executors((), override=override)
        _warn_if_forced_past_the_gate(
            profile, forced, task_group=task_group, source="override",
            knowledge_readable=knowledge_readable,
        )
        return forced, "override"
    pinned = task_flow_strategy_override()
    if pinned is not None:
        _warn_if_forced_past_the_gate(
            profile, pinned, task_group=task_group, source=TASK_STRATEGY_ENV,
            knowledge_readable=knowledge_readable,
        )
        return pinned, "env_pin"
    if test_profile:
        return DEFAULT_EXECUTION_STRATEGY, "test_profile"
    from ..routing.capabilities import chain_backends, chain_serves_tool_session

    backends = chain_backends(profile, task_group=task_group)
    # Lazily: the transport question may probe a driver (`--help`), and a chain
    # that is not all agents already answers `batched`. The warning path next
    # to this one short-circuits the same way.
    tool_session = bool(backends) and all(kind == "local_agent" for kind in backends) and (
        chain_serves_tool_session(profile, task_group=task_group)
    )
    return strategy_for_executors(backends, tool_session=tool_session), f"{task_group}_chain"


#: Dev-only strategy pin (plan §11 acceptance runs; owner 2026-09-09): an
#: agent executor standing in for the request-bound API path, or the reverse.
#: Unset = the production rule, derived per task from the executor chain.
TASK_STRATEGY_ENV = "FINESUB_TASK_STRATEGY"


def window_requests(profile: TranslationProfile, *, knowledge_readable: bool) -> bool:
    """Whether a correction window has anything to ask for: harness search
    (``retrieval=local``) or a readable knowledge base. Strategy-free, so the
    fast gate -- which deliberately keeps routing out of the planner -- can
    read it without a driver probe."""

    return profile.external_injection or knowledge_readable


@dataclass(frozen=True)
class WindowFlow:
    """How one run's correction windows get the inputs they lack.

    The one owner of these answers: the run, its dry-run and the fast
    `--style auto` hand-off read them here, so no caller re-derives half of
    the expression.

    Two independent reasons to talk to the model once more before it
    translates: ``window_requests`` (search or entries to ask for) and
    ``style_pick`` (a fast window that must pick its style first,
    auto-style-gaps plan A). The channels open for either; the transfer chain
    and the preparation task's downstream caps read ``window_requests``
    itself -- a window that can only pick a style cannot ask for a dropped
    entry back, and pruning the chain on its account would lose the entry for
    the rest of the run.
    """

    #: Something to ask for at all: harness search (`retrieval=local`) or a
    #: readable knowledge base. Independent of the execution strategy, and
    #: "a dropped entry can be asked for again" -- by the request turn or by
    #: the executor itself.
    window_requests: bool
    #: Batched: the harness request turn (one request batch, one fill) --
    #: for requests, a style pick, or both.
    request_channel: bool
    #: Agentic: the executor's own tool session fetches -- requests, or the
    #: picked style's full text, or both.
    agent_fetches: bool
    #: The window must pick its `--style auto` style before it translates:
    #: fast, auto, and a candidate besides `default_style` (the caller's
    #: answer -- `StyleCatalog.has_choice` on a task with no recorded pick).
    style_pick: bool = False


def window_flow(
    profile: TranslationProfile,
    *,
    knowledge_readable: bool,
    test_profile: bool = False,
    style_pick: bool = False,
) -> WindowFlow:
    """The correction windows' flow for this run.

    ``knowledge_readable`` is "the base is on and its index is non-empty"
    (`research.knowledge_index_available`), computed once per run by the caller.
    ``style_pick`` is the caller's too: only it knows whether the run is fast,
    `--style auto`, and still without a recorded pick.
    """

    requests = window_requests(profile, knowledge_readable=knowledge_readable)
    if not (requests or style_pick):
        return WindowFlow(window_requests=False, request_channel=False, agent_fetches=False)
    strategy, _source = resolve_execution_strategy(
        profile, test_profile=test_profile, knowledge_readable=knowledge_readable
    )
    return WindowFlow(
        window_requests=requests,
        request_channel=strategy == "batched",
        agent_fetches=strategy == "agentic",
        style_pick=style_pick,
    )


def task_flow_strategy_override() -> str | None:
    import os

    raw = os.environ.get(TASK_STRATEGY_ENV, "").strip().lower()
    if not raw:
        return None
    if raw in EXECUTION_STRATEGIES:
        return raw
    raise ValueError(f"{TASK_STRATEGY_ENV} must be batched|agentic, got {raw!r}")


def task_graph_metadata(
    graph: TaskGraph, *, strategy_source: str, prepare_strategy_source: str = ""
) -> Mapping[str, Any]:
    """Payload of the ``task_graph`` artifact.

    ``strategy``/``strategy_source`` are the window task's -- the run's. The
    preparation task's own answer is on its ``TaskSpec``; its source rides
    alongside when the two chains were resolved separately.
    """

    return {
        **graph.to_metadata(),
        "strategy_source": strategy_source,
        **({"prepare_strategy_source": prepare_strategy_source} if prepare_strategy_source else {}),
    }

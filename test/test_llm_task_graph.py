"""Contract tests for the task graph (plan §8, first bullet; stage A).

Four-cell main matrix × six-cell capability matrix: the task list is the
same in every cell that shares ``fast`` and the update switch; capabilities
only change tool grants; an empty or disabled knowledge base cannot change
the graph because the graph never sees one.
"""

from __future__ import annotations

import inspect
import itertools
import json
from pathlib import Path

import pytest

from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages import task_graph as tg


STRATEGIES = tg.EXECUTION_STRATEGIES
FASTS = (False, True)
SEARCHES = tg.SEARCH_MODES
KNOWLEDGES = tg.KNOWLEDGE_MODES
STYLES = (False, True)


def _every_cell():
    for strategy, fast, search, knowledge, style in itertools.product(
        STRATEGIES, FASTS, SEARCHES, KNOWLEDGES, STYLES
    ):
        grant = tg.CapabilityGrant(search=search, knowledge=knowledge, style=style)
        yield tg.plan_task_graph(strategy=strategy, fast_enabled=fast, grant=grant)


def test_task_list_depends_only_on_fast_and_the_update_switch() -> None:
    """Both strategies and all six grant cells dispatch the same tasks."""

    for graph in _every_cell():
        expected = ("translate",) if graph.fast else ("prepare", "translate")
        if graph.grant.updates_knowledge:
            expected += ("knowledge_update",)
        assert graph.kinds == expected, graph.to_metadata()
        scopes = tuple(task.scope for task in graph.tasks)
        assert scopes[: 1 if graph.fast else 2] == (
            ("whole",) if graph.fast else ("global", "window")
        )


def test_strategy_never_changes_the_task_list() -> None:
    for fast, search, knowledge, style in itertools.product(
        FASTS, SEARCHES, KNOWLEDGES, STYLES
    ):
        grant = tg.CapabilityGrant(search=search, knowledge=knowledge, style=style)
        batched = tg.plan_task_graph(strategy="batched", fast_enabled=fast, grant=grant)
        agentic = tg.plan_task_graph(strategy="agentic", fast_enabled=fast, grant=grant)
        # Same tasks, same grants; only the recorded "how does this one fetch"
        # differs (§2: the strategy is not what the task *is*).
        assert batched.kinds == agentic.kinds
        assert [t.scope for t in batched.tasks] == [t.scope for t in agentic.tasks]
        assert [t.tools for t in batched.tasks] == [t.tools for t in agentic.tasks]


def test_each_task_records_the_chain_that_answers_it() -> None:
    """Strategy is resolved per task, so it is recorded per task: the
    preparation task runs on the research chain and the window on the
    correction chain, and `agy` + `--media audio` really is "prepare agentic,
    window batched" (the window's clip forces a capsule). One graph-level
    number would file that run as batched and lose the split."""

    graph = tg.plan_task_graph(
        strategy="batched",
        prepare_strategy="agentic",
        fast_enabled=False,
        grant=tg.CapabilityGrant(search="local", knowledge="update"),
    )
    by_kind = {task.kind: task.strategy for task in graph.tasks}
    assert by_kind == {"prepare": "agentic", "translate": "batched", "knowledge_update": ""}
    # `strategy` stays the run's dispatch strategy: the window's.
    assert graph.strategy == "batched"
    # Absent, it follows the window's -- one chain, one answer.
    assert all(
        task.strategy in ("agentic", "")
        for task in tg.plan_task_graph(
            strategy="agentic", fast_enabled=False, grant=tg.CapabilityGrant()
        ).tasks
    )


def test_capabilities_only_move_tool_grants() -> None:
    for graph in _every_cell():
        grant = graph.grant
        for task in graph.tasks:
            if task.kind == "knowledge_update":
                assert task.tools == (tg.TOOL_KB_PROPOSE,)
                continue
            assert (tg.TOOL_SEARCH_LOCAL in task.tools) is (grant.search == "local")
            assert (tg.TOOL_SEARCH_NATIVE in task.tools) is (grant.search == "native")
            assert not (
                tg.TOOL_SEARCH_LOCAL in task.tools and tg.TOOL_SEARCH_NATIVE in task.tools
            )
            assert (tg.TOOL_KB_READ in task.tools) is grant.reads_knowledge
            # Style is authorized on its own: knowledge=none does not revoke it.
            assert (tg.TOOL_STYLE_READ in task.tools) is grant.style
            assert tg.TOOL_KB_PROPOSE not in task.tools


def test_knowledge_update_is_its_own_task_even_under_fast() -> None:
    graph = tg.plan_task_graph(
        fast_enabled=True, grant=tg.CapabilityGrant(knowledge="update")
    )
    assert graph.kinds == ("translate", "knowledge_update")
    fused = graph.tasks[0]
    assert fused.scope == "whole"
    assert tg.TOOL_KB_PROPOSE not in fused.tools


def test_the_graph_never_sees_the_knowledge_base() -> None:
    """No root, no index, no loader: an empty base is an empty tool result."""

    params = inspect.signature(tg.plan_task_graph).parameters
    assert not any("knowledge_root" in name or "index" in name for name in params)
    source = Path(tg.__file__).read_text(encoding="utf-8")
    for forbidden in ("load_index_texts", "index_texts_present", "knowledge.base"):
        assert forbidden not in source, forbidden


def test_grant_for_run_takes_retrieval_verbatim() -> None:
    for retrieval in SEARCHES:
        profile = resolve_profile("text", retrieval, "quality")
        grant = tg.grant_for_run(profile, knowledge="read", style_selected=True)
        assert grant.search == retrieval
        assert grant.knowledge == "read"
        assert grant.style is True


def test_strategy_for_executors() -> None:
    assert tg.strategy_for_executors(["local_agent", "local_agent"]) == "agentic"
    assert tg.strategy_for_executors(["local_agent", "gemini_rest"]) == "batched"
    assert tg.strategy_for_executors(["gemini_rest"]) == "batched"
    assert tg.strategy_for_executors([]) == "batched"
    # The override wins regardless of the chain, and is validated.
    assert tg.strategy_for_executors(["gemini_rest"], override="agentic") == "agentic"
    with pytest.raises(ValueError):
        tg.strategy_for_executors([], override="fast")


def test_resolve_execution_strategy_reads_the_correction_chain(monkeypatch) -> None:
    from finesub.llm.routing import capabilities

    class Endpoint:
        def __init__(self, backend: str) -> None:
            self.backend = backend
            self.provider_tier = "LOCAL_STUB"
            self.api_model_id = "stub"

    chain = capabilities.ChainRequirement(
        label="correction", endpoints=(Endpoint("local_agent"),)  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        capabilities, "required_chains", lambda profile: [chain]
    )
    monkeypatch.setattr(capabilities, "usable_endpoints", lambda req: list(req.endpoints))
    monkeypatch.setattr(capabilities, "_driver_takes_mcp_server", lambda *a: True)
    profile = resolve_profile("text", "none", "quality")
    assert tg.resolve_execution_strategy(profile) == (
        "agentic",
        "correction_chain",
    )
    assert tg.resolve_execution_strategy(profile, test_profile=True) == (
        "batched",
        "test_profile",
    )
    assert tg.resolve_execution_strategy(
        profile, test_profile=True, override="agentic"
    ) == ("agentic", "override")


def test_an_all_agent_chain_without_tools_is_not_agentic(monkeypatch) -> None:
    """`agentic` means "the executor fetches for itself", which only exists on
    the tool-session transport. A media call and the `api`/`resume` session
    modes are `capsule` by definition (`agent_transports.agent_transport_for`):
    picking `agentic` there would stand the harness's own request turn down and
    then tell a tool-less model to search -- `retrieval=local` silently becoming
    `none`."""

    from finesub.llm.routing import capabilities

    class Endpoint:
        def __init__(self, backend: str) -> None:
            self.backend = backend
            self.provider_tier = "LOCAL_STUB"
            self.api_model_id = "stub"

    chain = capabilities.ChainRequirement(
        label="correction", endpoints=(Endpoint("local_agent"),)  # type: ignore[arg-type]
    )
    monkeypatch.setattr(capabilities, "required_chains", lambda profile: [chain])
    monkeypatch.setattr(capabilities, "usable_endpoints", lambda req: list(req.endpoints))
    takes_mcp = [True]
    monkeypatch.setattr(capabilities, "_driver_takes_mcp_server", lambda *a: takes_mcp[0])

    assert tg.resolve_execution_strategy(resolve_profile("text", "local", "quality"))[0] == "agentic"
    # A clip rides on the correction call, so that call is a capsule.
    assert tg.resolve_execution_strategy(resolve_profile("audio", "local", "quality"))[0] == "batched"

    # A driver whose CLI takes no per-invocation MCP server falls back to
    # capsule too. This one is not decided by a switch the user set, which is
    # why it was first left to the transport's own warning -- but the warning
    # cannot put the request turn back, so it has to be known here.
    takes_mcp[0] = False
    assert tg.resolve_execution_strategy(resolve_profile("text", "local", "quality"))[0] == "batched"
    takes_mcp[0] = True

    class Config:
        agent_session_mode = "resume"

    monkeypatch.setattr(capabilities, "role_config_for", lambda *a, **k: Config())
    assert tg.resolve_execution_strategy(resolve_profile("text", "local", "quality"))[0] == "batched"
    assert tg.strategy_for_executors(["local_agent"], tool_session=False) == "batched"


def test_a_pin_that_forces_agentic_past_the_gate_says_so(monkeypatch, reported) -> None:
    """The pin holds a strategy fixed for an A/B (plan §11) and keeps that
    power -- the acceptance runs forced `batched` onto an agent, which is safe.
    Forcing `agentic` the other way recreates the failure the gate was added
    for: the harness stands its request turn down and the executor has no
    tools. A run that measured *that* would be measuring the wrong thing."""

    from finesub.llm.routing import capabilities

    monkeypatch.setattr(tg, "_WARNED_FORCED", set())
    monkeypatch.setattr(capabilities, "chain_backends", lambda *a, **k: ["local_agent"])
    monkeypatch.setattr(capabilities, "chain_serves_tool_session", lambda *a, **k: False)
    monkeypatch.setenv(tg.TASK_STRATEGY_ENV, "agentic")
    profile = resolve_profile("audio", "local", "quality")

    assert tg.resolve_execution_strategy(profile) == ("agentic", "env_pin")
    # Said once per (pin, task group, profile), not once per resolution.
    tg.resolve_execution_strategy(profile)
    assert reported.codes().count("task-strategy-pin-without-tools") == 1

    # A chain with no agent at all is the same failure and used to be silent:
    # the transport question filters to agent members and then takes `all()`,
    # so a pure-API chain answered `all([]) is True`. The judgement is the
    # derivation itself, so "not every executor is an agent" counts too.
    monkeypatch.setattr(tg, "_WARNED_FORCED", set())
    monkeypatch.setattr(capabilities, "chain_backends", lambda *a, **k: ["gemini_rest"])
    monkeypatch.setattr(capabilities, "chain_serves_tool_session", lambda *a, **k: True)
    tg.resolve_execution_strategy(resolve_profile("text", "local", "quality"))
    assert reported.codes().count("task-strategy-pin-without-tools") == 2
    assert "不是 local_agent" in reported.joined()
    # Mixed is judged the same way -- its agent half passing proves nothing.
    monkeypatch.setattr(tg, "_WARNED_FORCED", set())
    monkeypatch.setattr(
        capabilities, "chain_backends", lambda *a, **k: ["local_agent", "gemini_rest"]
    )
    tg.resolve_execution_strategy(resolve_profile("text", "local", "quality"))
    assert reported.codes().count("task-strategy-pin-without-tools") == 3

    # An all-agent chain that really does get a tool session agrees with the
    # derivation, so the pin is saying nothing new.
    monkeypatch.setattr(tg, "_WARNED_FORCED", set())
    monkeypatch.setattr(capabilities, "chain_backends", lambda *a, **k: ["local_agent"])
    tg.resolve_execution_strategy(resolve_profile("text", "local", "quality"))
    assert reported.codes().count("task-strategy-pin-without-tools") == 3

    # Forcing batched is not a warning (that is what the acceptance runs did),
    # and neither is a pin on a run with no harness retrieval to lose.
    monkeypatch.setattr(tg, "_WARNED_FORCED", set())
    monkeypatch.setattr(capabilities, "chain_backends", lambda *a, **k: ["gemini_rest"])
    monkeypatch.setenv(tg.TASK_STRATEGY_ENV, "batched")
    tg.resolve_execution_strategy(profile)
    tg.resolve_execution_strategy(
        resolve_profile("audio", "none", "quality"), override="agentic"
    )
    assert reported.codes().count("task-strategy-pin-without-tools") == 3


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        tg.CapabilityGrant(search="web")
    with pytest.raises(ValueError):
        tg.CapabilityGrant(knowledge="collect")
    with pytest.raises(ValueError):
        tg.plan_task_graph(strategy="serial", fast_enabled=False, grant=tg.CapabilityGrant())
    with pytest.raises(ValueError):
        tg.TaskSpec(kind="research", scope="global")


def test_metadata_round_trips_through_json() -> None:
    graph = tg.plan_task_graph(
        strategy="agentic",
        fast_enabled=False,
        grant=tg.CapabilityGrant(search="local", knowledge="update", style=True),
    )
    payload = json.loads(json.dumps(tg.task_graph_metadata(graph, strategy_source="override")))
    assert payload["strategy"] == "agentic"
    assert payload["strategy_source"] == "override"
    assert [task["kind"] for task in payload["tasks"]] == [
        "prepare",
        "translate",
        "knowledge_update",
    ]
    assert payload["tasks"][0]["tools"] == ["search:local", "kb:read", "style:read"]


def test_window_flow_asks_for_search_or_a_readable_base(monkeypatch) -> None:
    """window-kb-request-channel plan: one owner for the windows' flow. A
    window has something to ask for under harness search *or* a readable base;
    the strategy only decides who asks."""

    none = resolve_profile("text", "none", "quality")
    monkeypatch.setattr(tg, "resolve_execution_strategy", lambda profile, **kw: ("batched", "t"))
    assert tg.window_flow(none, knowledge_readable=False) == tg.WindowFlow(False, False, False)
    batched = tg.window_flow(none, knowledge_readable=True)
    assert batched == tg.WindowFlow(True, True, False)
    local = tg.window_flow(resolve_profile("text", "local", "quality"), knowledge_readable=False)
    assert local.window_requests and local.request_channel

    monkeypatch.setattr(tg, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "t"))
    agentic = tg.window_flow(resolve_profile("text", "native", "quality"), knowledge_readable=True)
    assert agentic == tg.WindowFlow(True, False, True)


def test_a_forced_agentic_on_a_readable_base_says_so_too(monkeypatch, reported) -> None:
    """Entry requests are lost the same way search is; with nothing to ask for
    the pin costs nothing and stays silent."""

    from finesub.llm.routing import capabilities

    monkeypatch.setattr(tg, "_WARNED_FORCED", set())
    monkeypatch.setattr(capabilities, "chain_backends", lambda *a, **k: ["gemini_rest"])
    none = resolve_profile("audio", "none", "quality")
    tg.resolve_execution_strategy(none, override="agentic")
    assert "task-strategy-pin-without-tools" not in reported.codes()
    tg.resolve_execution_strategy(none, override="agentic", knowledge_readable=True)
    assert reported.codes().count("task-strategy-pin-without-tools") == 1


def test_the_capsule_transport_pin_takes_the_tools_away(monkeypatch) -> None:
    """`FINESUB_AGENT_TRANSPORT=capsule` puts a per-window agent call on a
    transport with no harness tools, so the task must not be told to fetch
    for itself: the strategy has to fall back to the request turn."""

    from finesub.llm.routing import capabilities

    monkeypatch.setattr(capabilities, "_driver_takes_mcp_server", lambda *a: True)
    profile = resolve_profile("text", "local", "quality")
    monkeypatch.delenv("FINESUB_AGENT_TRANSPORT", raising=False)
    assert capabilities.chain_serves_tool_session(profile) is True
    monkeypatch.setenv("FINESUB_AGENT_TRANSPORT", "capsule")
    assert capabilities.chain_serves_tool_session(profile) is False

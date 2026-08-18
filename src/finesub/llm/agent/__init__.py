"""Running a task on a local agent instead of spending an API quota.

`local_agent` is the one-shot episode transport and its three drivers (Codex,
Claude Code, Antigravity); `agent_transports` and `agent_task_runtime` are the
durable task protocol built on top of it; `agent_quota` books a spent
subscription per quota pool and `agent_ping` probes one; `agent_paths` and
`agent_cleanup` own where episodes live and how they are cleared;
`agent_retrieval` is the searches an agent runs on the harness's behalf;
`agent_task_control` is the `finesub agent-task` CLI.

Module names keep their `agent_` prefix on purpose: they are named in
`finesub_bootstrap.shell` as strings (`python -m finesub.llm.agent.agent_cleanup`),
and a rename there fails at run time rather than at import.

docs/llm_local_agent.md is the single entry point for what is wired into
production and what is still groundwork (§12.5).
"""

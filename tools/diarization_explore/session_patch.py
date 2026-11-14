"""Agent calls in one process run on agy's continued-conversation path (D5 in docs/reports/2026-10-17-three-pass.md):
every call takes the `resume` session mode, so the run keeps one conversation; `seed` makes the first call continue
an earlier process's conversation (pass 2 after pass 1) instead of opening a new one, with `bridge` put in front of
its last user message. Each call's handle, whether it inherited history, and its usage go to `log` (jsonl).

Both arms of D5 run pass 2 through here, seeded or not, so they share a transport (the production pass 2's default
tool session is another one, whose cost would not compare).
"""
from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path


def install(log: Path, seed: str = "", bridge: str = "") -> None:
    from finesub.llm.client import RoleClient
    from finesub.llm.routing.execution_policy import normalized_tier

    original_config, original_run = RoleClient._config_for, RoleClient._run_local_agent
    lock, state = threading.Lock(), {"seeded": not seed}

    def config_for(self, *args, **kwargs):
        return dataclasses.replace(original_config(self, *args, **kwargs), agent_session_mode="resume")

    def run_local_agent(self, driver, messages, *, repair_session_key="", provider_tier="", model="",
                        session_mode="", **kwargs):
        with lock:
            first, state["seeded"] = not state["seeded"], True
        if first:
            key = self._conversation_key(session_mode, repair_session_key)
            with self._agent_conversation_lock:
                self._agent_repair_conversations[(normalized_tier(provider_tier), model, key)] = seed
            if bridge:
                messages = [dict(m) for m in messages]
                last = max(n for n, m in enumerate(messages) if m.get("role") == "user")
                content = messages[last]["content"]
                messages[last]["content"] = (f"{bridge}\n\n{content}" if isinstance(content, str)
                                             else [{"type": "text", "text": bridge}, *content])
        result, discarded, inherited = original_run(
            self, driver, messages, repair_session_key=repair_session_key, provider_tier=provider_tier,
            model=model, session_mode=session_mode, **kwargs)
        with lock, open(log, "a", encoding="utf-8") as out:
            out.write(json.dumps({"handle": getattr(result, "conversation_handle", ""), "inherited": inherited,
                                  "seeded": first and bool(seed), "rebuilt": bool(discarded),
                                  "asked": str(messages[-1].get("content"))[:120],
                                  "usage": dict(getattr(result, "usage", None) or {})},
                                 ensure_ascii=False) + "\n")
        return result, discarded, inherited

    RoleClient._config_for, RoleClient._run_local_agent = config_for, run_local_agent


def last_handle(log: Path) -> str:
    """The conversation the logged process ended in."""
    rows = [json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines() if line.strip()]
    return next((row["handle"] for row in reversed(rows) if row["handle"]), "")

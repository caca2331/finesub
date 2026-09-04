"""Local knowledge base and its update pipeline.

Modules:

- ``base``      — Markdown knowledge base primitives (index + per-entry files,
                  embedded git, proposal apply) and the task-artifact JSONL store.
- ``style``     — selecting (`--style`) and rendering a style entry.
- ``feedback``  — ``task_update_feedback`` v2 parsing and aggregation.
- ``entries``   — hint-driven knowledge-entry prefetch/rendering.
- ``materials`` — per-window CSV evidence packs and 100k chunk planning.
- ``update``    — the unified knowledge-update runner and CLI
                  (``python -m finesub.llm.knowledge.update``).

Import from the submodules directly (``from finesub.llm.knowledge.base import ...``);
this package intentionally re-exports nothing.
"""

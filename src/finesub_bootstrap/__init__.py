"""Shared provisioning layer for FineSub front ends.

Everything needed to stand up and use a managed FineSub runtime on an end-user
machine lives here: application directory layout, verified downloads, archive
extraction, the uv-managed Python environment, and worker process context.
The desktop app and the CLI shell both build on this package; nothing in it
may import from ``desktop`` or depend on a UI.

Requires ``pydantic`` and ``httpx`` (the ``[desktop]`` extra provides both) --
with one deliberate exception: ``secrets`` (the ``.env`` key-protection layer)
is stdlib-only and is imported by ``finesub.llm.llm_runtime``, so plain
``[asr]``/``[harness]`` installs do import that module. Two permanent
constraints keep this working: ``secrets.py`` stays stdlib-only, and this
``__init__`` stays free of imports (``test_secrets.py`` guards both). Tests
live in ``desktop/backend/tests`` and run under the desktop CI, which is the
only Windows lane; ``secrets`` is tested in the root suite.
"""

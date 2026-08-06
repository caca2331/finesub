"""Shared provisioning layer for FineSub front ends.

Everything needed to stand up and use a managed FineSub runtime on an end-user
machine lives here: application directory layout, verified downloads, archive
extraction, the uv-managed Python environment, and worker process context.
The desktop app and the CLI shell both build on this package; nothing in it
may import from ``desktop`` or depend on a UI.

Requires ``pydantic`` and ``httpx`` (the ``[desktop]`` extra provides both);
plain ``[asr]``/``[harness]`` installs never import it. Tests live in
``desktop/backend/tests`` and run under the desktop CI, which is the only
Windows lane.
"""

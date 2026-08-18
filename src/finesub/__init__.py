"""finesub production package.

Keep it import side-effect free. Since `llm` moved in here (2026-08) this file
runs ahead of every `import finesub.llm.…`, including on installs that have no
ASR stack: a plain `[harness]` one, and the thin CLI's own Python 3.10 running
`python -m finesub.llm.agent.agent_cleanup` without the managed runtime.
`test_import_boundaries` holds the line.
"""

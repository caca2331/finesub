"""FineSub CLI shell: a thin launcher over a uv-managed runtime.

The wheel carries no heavy dependencies. On first use the launcher provisions
`%LOCALAPPDATA%\\FineSub` (Python 3.12 + the locked ASR stack + FFmpeg), then
re-executes the pipeline inside that runtime with the vendored sources on
PYTHONPATH. An installed FineSub Desktop shares the same personal-data
directory, so API keys configured in either product work in both.
"""

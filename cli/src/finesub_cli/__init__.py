"""FineSub CLI shell: a thin launcher over a uv-managed runtime.

The wheel carries no heavy dependencies. On first use the launcher provisions
`%LOCALAPPDATA%\\FineSub` (Python 3.12 + the locked ASR stack + FFmpeg), then
re-executes the pipeline inside that runtime with the vendored sources on
PYTHONPATH. Personal data -- settings, API keys, the knowledge base -- lives in
`%LOCALAPPDATA%\\FineSub\\user-data`, shared with a source checkout that opts in.
"""

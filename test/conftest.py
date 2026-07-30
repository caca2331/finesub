from __future__ import annotations

import pytest

_FILE_MARKERS: dict[str, tuple[str, ...]] = {
    "test_pipeline_refactor.py": ("pipeline",),
    "test_batch_runner.py": ("pipeline",),
    "test_resource_profiles.py": ("pipeline",),
    "test_resource_budget_pipeline.py": ("pipeline",),
    "test_vocal_separation_pool.py": ("pipeline",),
    "test_gpu_stage_gate.py": ("pipeline",),
    "test_import_boundaries.py": ("pipeline",),
    "test_packaging.py": ("pipeline",),
    "test_paths.py": ("pipeline",),
    "test_run_metadata.py": ("pipeline",),
    "test_asr_and_text_utils.py": ("asr",),
    "test_asr_stabilize.py": ("asr",),
    "test_intervals.py": ("asr",),
    "test_segment_split.py": ("asr",),
    "test_srt_rendering.py": ("asr",),
    "test_vad_streaming.py": ("asr", "slow"),
    "test_vad_segment_energy.py": ("asr",),
    "test_wt_shard.py": ("asr",),
    "test_wt_sharding.py": ("asr",),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-heavy-resource",
        action="store_true",
        default=False,
        help="Run tests that may load models, process media, or use significant GPU/RAM.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    for item in items:
        filename = item.path.name
        markers = _FILE_MARKERS.get(filename)
        if markers is None and filename.startswith("test_llm_"):
            markers = ("llm",)
        if markers:
            for name in markers:
                item.add_marker(getattr(pytest.mark, name))

    if config.getoption("--run-heavy-resource"):
        return
    skip_heavy = pytest.mark.skip(
        reason="requires --run-heavy-resource to run significant GPU/RAM tests"
    )
    for item in items:
        if "heavy_resource" in item.keywords:
            item.add_marker(skip_heavy)

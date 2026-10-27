"""Separator reporting and device workarounds without loading a checkpoint.

The pieces under test are the adapter's own: library output, block progress,
and applying the Windows ROCm attention fallback before inference.
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
import types

import pytest


@pytest.fixture
def separation():
    """The separator adapter.

    Importing it is cheap: `audio_separator` itself is only reached lazily from
    `_build_separator`, so nothing here needs the optional extra installed.
    """

    from finesub.speech.preprocessing.separator import separation

    return separation


@pytest.fixture
def reporting():
    """Where library quieting lives now.

    It moved out of the separator once a real run showed transformers drawing
    its own bar during verification and the separator's *logger* -- untouched
    by any of this -- producing more lines than the pipeline itself.
    """

    from finesub import reporting

    return reporting


class _FakeTqdm:
    """Stands in for `tqdm.std.tqdm`: records whether it was disabled."""

    instances: list[bool] = []

    def __init__(self, *args, disable=False, **kwargs):
        self.disable = disable
        _FakeTqdm.instances.append(disable)


@pytest.fixture
def fake_tqdm(monkeypatch):
    _FakeTqdm.instances = []
    std = types.ModuleType("tqdm.std")
    std.tqdm = _FakeTqdm
    package = types.ModuleType("tqdm")
    package.tqdm = _FakeTqdm
    package.std = std
    monkeypatch.setitem(sys.modules, "tqdm", package)
    monkeypatch.setitem(sys.modules, "tqdm.std", std)
    return _FakeTqdm


def test_third_party_bars_are_disabled_inside_the_scope(reporting, fake_tqdm) -> None:
    # The library holds the class itself (`from tqdm import tqdm`), so the
    # patch has to survive that binding.
    held_by_library = fake_tqdm

    with reporting.quieted_libraries("normal"):
        held_by_library(total=10)
        held_by_library(total=10, disable=False)

    assert fake_tqdm.instances == [True, True]

    held_by_library(total=10)
    assert fake_tqdm.instances[-1] is False, "the patch must not outlive the scope"


def test_nested_scopes_do_not_unsilence_the_outer_one(reporting, fake_tqdm) -> None:
    with reporting.quieted_libraries("normal"):
        with reporting.quieted_libraries("normal"):
            fake_tqdm(total=1)
        fake_tqdm(total=1)

    assert fake_tqdm.instances == [True, True]


def test_a_missing_tqdm_is_not_an_error(reporting, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.setitem(sys.modules, "tqdm.std", None)

    with reporting.quieted_libraries("normal"):
        pass


def test_a_chatty_library_is_raised_to_warning_but_not_silenced(
    reporting,
) -> None:
    """A 3-minute run had 25 of its 45 stderr lines from one library's INFO.

    Silencing it outright would take its warnings too, so the floor moves to
    WARNING rather than to nothing.
    """

    import logging

    logger = logging.getLogger("audio_separator")
    logger.setLevel(logging.INFO)
    try:
        with reporting.quieted_libraries("normal"):
            assert logger.level == logging.WARNING
            assert logger.isEnabledFor(logging.WARNING)
        assert logger.level == logging.INFO, "the level must be given back"
    finally:
        logger.setLevel(logging.NOTSET)


def test_verbose_leaves_the_libraries_alone(reporting, fake_tqdm) -> None:
    """Someone who asked for everything gets everything."""

    import logging

    logger = logging.getLogger("audio_separator")
    logger.setLevel(logging.INFO)
    try:
        with reporting.quieted_libraries("verbose"):
            assert logger.level == logging.INFO
            fake_tqdm(total=1)
        assert fake_tqdm.instances == [False]
    finally:
        logger.setLevel(logging.NOTSET)


def test_the_separator_places_its_files_before_the_library_loads_them(
    separation, monkeypatch
) -> None:
    """The CLI has no prefetch, so this is where it gets the proxy and the
    digest checks the desktop used to get on its own."""

    placed: list[str] = []
    monkeypatch.setattr(
        "finesub_bootstrap.model_fetch.fetch_fixed_files",
        lambda entry, directory, **kwargs: placed.append(entry.model_id),
    )
    # Make the dependency state deterministic: developer machines may have the
    # optional package installed, while the ordering contract must hold either
    # way. A None entry makes Python's lazy import fail at the same boundary as
    # an absent package without loading a real model.
    monkeypatch.setitem(sys.modules, "audio_separator.separator", None)

    with pytest.raises(RuntimeError, match="audio-separator is required"):
        # The build fails at the import after placement: the files must be
        # present before the library looks for them.
        separation._build_separator("out", "ogg", 1, use_cuda=False)

    assert placed == ["separator"]


def test_a_placement_failure_never_stops_the_run(separation, monkeypatch) -> None:
    """Unreachable proxy, unlisted model, unwritable dir -- the library still
    downloads them itself, exactly as before any of this existed."""

    def explode(*args, **kwargs):
        raise OSError("model directory is read-only")

    monkeypatch.setattr(
        "finesub_bootstrap.model_fetch.fetch_fixed_files", explode
    )

    separation.place_separator_files()


class _RecordingReporter:
    def __init__(self) -> None:
        self.progress_calls: list[dict] = []

    def progress(self, stage, **kwargs) -> None:
        self.progress_calls.append({"stage": stage, **kwargs})

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def test_progress_counts_completions_not_merges(separation, monkeypatch) -> None:
    """Blocks merge in order; counting on merge shows nothing until the first.

    With four workers in flight that reads as a stall at exactly the moment the
    most work is happening.
    """
    reporter = _RecordingReporter()
    monkeypatch.setattr(separation, "current_reporter", lambda: reporter)

    progress = separation._BlockProgress(4, workers=2)
    progress.report()

    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lambda: None) for _ in range(4)]
        for future in futures:
            future.add_done_callback(progress.block_finished)
        for future in futures:
            future.result()

    completions = [call["completed"] for call in reporter.progress_calls]
    assert completions[0] == 0
    assert completions[-1] == 4
    assert sorted(completions) == completions, "the count must only go up"
    assert {call["stage"] for call in reporter.progress_calls} == {"vocal"}
    assert {call["total"] for call in reporter.progress_calls} == {4}


def test_a_cancelled_block_is_not_counted(separation, monkeypatch) -> None:
    reporter = _RecordingReporter()
    monkeypatch.setattr(separation, "current_reporter", lambda: reporter)
    progress = separation._BlockProgress(2, workers=1)

    future: cf.Future = cf.Future()
    future.cancel()
    progress.block_finished(future)

    failed: cf.Future = cf.Future()
    failed.set_exception(RuntimeError("block died"))
    progress.block_finished(failed)

    assert reporter.progress_calls == []


def test_concurrency_shows_up_in_the_detail(separation, monkeypatch) -> None:
    reporter = _RecordingReporter()
    monkeypatch.setattr(separation, "current_reporter", lambda: reporter)

    separation._BlockProgress(4, workers=3, clock=lambda: 0.0).report()
    separation._BlockProgress(4, workers=1, clock=lambda: 0.0).report()

    assert "3 并发" in reporter.progress_calls[0]["detail"]
    assert "并发" not in reporter.progress_calls[1]["detail"]


def test_the_onnx_provider_notice_is_dropped_and_other_warnings_survive() -> None:
    """The Roformer runs on torch; "acceleration will NOT be enabled" is false for it.

    Driven through the library's own emit point (`configure_cuda`) so a wrong
    logger name in the filter cannot pass by agreeing with itself.
    """

    import logging

    pytest.importorskip("audio_separator.separator.separator")
    from audio_separator.separator import Separator

    from finesub.speech.preprocessing.separator import separation

    instance = Separator.__new__(Separator)
    instance.logger = logging.getLogger(Separator.__module__)
    seen: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    handler = _Collect()
    instance.logger.addHandler(handler)
    try:
        separation._mute_onnx_provider_notice()
        separation._mute_onnx_provider_notice()  # idempotent
        assert (
            sum(isinstance(f, separation._DropOnnxProviderNotice) for f in instance.logger.filters)
            == 1
        )
        instance.configure_cuda([])  # onnxruntime without CUDAExecutionProvider
        instance.logger.warning("something else went wrong")
    finally:
        instance.logger.removeHandler(handler)
        for f in [
            f for f in instance.logger.filters if isinstance(f, separation._DropOnnxProviderNotice)
        ]:
            instance.logger.removeFilter(f)

    assert not any("acceleration will NOT" in message for message in seen)
    assert "something else went wrong" in seen


@pytest.fixture
def roformer_attend(separation, monkeypatch):
    """The dependency boundary, available even without audio-separator installed."""

    torch = separation.torch

    class Attend(torch.nn.Module):
        def __init__(self, *, flash=True, dropout=0.0, scale=None):
            super().__init__()
            self.flash = flash
            self.dropout = dropout
            self.scale = scale

        def forward(self, value):
            assert not self.flash, "inference reached the incompatible SDPA path"
            return value

    names = (
        "audio_separator",
        "audio_separator.separator",
        "audio_separator.separator.uvr_lib_v5",
        "audio_separator.separator.uvr_lib_v5.roformer",
        "audio_separator.separator.uvr_lib_v5.roformer.attend",
    )
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, module)
    sys.modules[names[-1]].Attend = Attend
    return Attend


def test_rocm_attention_fallback_changes_only_the_loaded_model(
    separation, roformer_attend, monkeypatch
) -> None:
    torch = separation.torch
    monkeypatch.setattr(separation.sys, "platform", "win32")
    monkeypatch.setattr(separation, "is_rocm", lambda: True)
    first = roformer_attend(dropout=0.2, scale=0.7)
    nested = roformer_attend()
    unrelated = roformer_attend()
    original_init = roformer_attend.__init__
    model = torch.nn.Sequential(first, torch.nn.Sequential(nested))
    separator = types.SimpleNamespace(
        model_instance=types.SimpleNamespace(model_run=model)
    )

    separation._disable_roformer_flash_attention(separator, use_cuda=True)
    separation._disable_roformer_flash_attention(separator, use_cuda=True)

    assert first.flash is False and nested.flash is False
    assert first.dropout == 0.2 and first.scale == 0.7
    assert unrelated.flash is True
    assert roformer_attend().flash is True
    assert roformer_attend.__init__ is original_init


@pytest.mark.parametrize("scale", [None, 0.37])
def test_real_roformer_einsum_attention_matches_scaled_dot_product_on_cpu(
    separation, monkeypatch, scale
) -> None:
    attend_module = pytest.importorskip(
        "audio_separator.separator.uvr_lib_v5.roformer.attend"
    )
    torch = separation.torch
    # Construct without accelerator probing, then represent the checkpoint's
    # flash=True choice. Every tensor and operation in this test stays on CPU.
    attention = attend_module.Attend(flash=False, dropout=0.0, scale=scale).eval()
    attention.flash = True
    separator = types.SimpleNamespace(
        model_instance=types.SimpleNamespace(model_run=attention)
    )
    monkeypatch.setattr(separation.sys, "platform", "win32")
    monkeypatch.setattr(separation, "is_rocm", lambda: True)

    def reject_sdpa(*args, **kwargs):
        raise AssertionError("the fallback must bypass Attend.flash_attn")

    monkeypatch.setattr(attention, "flash_attn", reject_sdpa)
    separation._disable_roformer_flash_attention(separator, use_cuda=True)

    generator = torch.Generator(device="cpu").manual_seed(47)
    q = torch.randn(2, 3, 4, 5, generator=generator, device="cpu")
    k = torch.randn(2, 3, 6, 5, generator=generator, device="cpu")
    v = torch.randn(2, 3, 6, 7, generator=generator, device="cpu")
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

    with torch.inference_mode():
        actual = attention(q, k, v)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "platform,rocm,use_cuda",
    [
        ("win32", True, False),  # explicit CPU, even with ROCm installed
        ("win32", False, True),  # NVIDIA CUDA
        ("win32", False, False),  # CPU-only build
        ("linux", True, True),  # ROCm stacks without the Windows failure
        ("darwin", False, False),
    ],
)
def test_other_devices_do_not_import_or_change_roformer_attention(
    separation, roformer_attend, monkeypatch, platform, rocm, use_cuda
) -> None:
    attention = roformer_attend()
    model = separation.torch.nn.Sequential(attention)
    separator = types.SimpleNamespace(
        model_instance=types.SimpleNamespace(model_run=model)
    )
    monkeypatch.setattr(separation.sys, "platform", platform)
    monkeypatch.setattr(separation, "is_rocm", lambda: rocm)
    monkeypatch.setitem(
        sys.modules,
        "audio_separator.separator.uvr_lib_v5.roformer.attend",
        None,
    )

    separation._disable_roformer_flash_attention(separator, use_cuda=use_cuda)

    assert attention.flash is True


def test_separator_build_applies_rocm_fallback_after_load_and_keeps_warning_filter(
    separation, roformer_attend, monkeypatch, caplog, tmp_path
) -> None:
    import logging

    torch = separation.torch
    events = []
    logger = logging.getLogger("audio_separator.separator.separator")
    # Restore the logger's filter list after the test as well as fake modules.
    monkeypatch.setattr(logger, "filters", list(logger.filters))
    monkeypatch.setattr(separation.sys, "platform", "win32")
    monkeypatch.setattr(separation, "is_rocm", lambda: True)
    monkeypatch.setattr(separation, "place_separator_files", lambda: None)
    monkeypatch.setattr(separation, "resolve_separator_model_dir", lambda: tmp_path)

    class Separator:
        def __init__(self, **kwargs):
            events.append("construct")
            self.model_instance = None
            logger.warning("CUDAExecutionProvider not available in ONNXruntime")
            logger.warning("an unrelated separator warning")

        def load_model(self, *, model_filename):
            events.append("load")
            assert model_filename == separation.MODEL_NAME
            self.model_instance = types.SimpleNamespace(
                model_run=torch.nn.Sequential(
                    roformer_attend(), torch.nn.Sequential(roformer_attend())
                )
            )

    sys.modules["audio_separator.separator"].Separator = Separator
    with caplog.at_level(logging.WARNING, logger=logger.name):
        separator = separation._build_separator("out", "ogg", 1, use_cuda=True)

    assert events == ["construct", "load"]
    value = torch.zeros(1)
    assert torch.equal(separator.model_instance.model_run(value), value)
    assert "ExecutionProvider not available" not in caplog.text
    assert "an unrelated separator warning" in caplog.text

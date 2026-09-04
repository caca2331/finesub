"""Separator reporting, exercised without torch or audio-separator.

The pieces under test are the adapter's own: whether the third-party bar is
silenced, and whether one progress line is maintained from block completions.
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

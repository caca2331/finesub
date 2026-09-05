"""The shared weights-preparation layer: what the loaders are told, and what
happens when it turns out to be wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.speech.runtime import hf_weights


class TestPrepare:
    def test_it_reports_the_pin_and_the_offline_gate(self, monkeypatch) -> None:
        from finesub_bootstrap import model_ensure

        monkeypatch.setattr(model_ensure, "pinned_revision", lambda _id: "abc123")
        monkeypatch.setattr(model_ensure, "ensure_hf_model", lambda *_a, **_k: None)
        monkeypatch.setattr(
            model_ensure, "pinned_snapshot_loadable", lambda _id: True
        )

        assert hf_weights.prepare("whisper") == hf_weights.HfLoad("abc123", True)

    def test_a_failed_fetch_is_never_fatal_here(self, monkeypatch) -> None:
        """The loader runs next and produces the error that describes what it
        wanted; a prefetch that raised on its own would hide that."""

        from finesub_bootstrap import model_ensure

        def boom(*_args, **_kwargs):
            raise RuntimeError("mirror is having an afternoon")

        monkeypatch.setattr(model_ensure, "pinned_revision", lambda _id: "abc123")
        monkeypatch.setattr(model_ensure, "ensure_hf_model", boom)
        monkeypatch.setattr(
            model_ensure, "pinned_snapshot_loadable", lambda _id: False
        )

        assert hf_weights.prepare("whisper") == hf_weights.HfLoad("abc123", False)

    def test_weights_judged_wrong_are_not_handed_on_as_loadable(
        self, monkeypatch
    ) -> None:
        """The one failure this layer does not absorb.

        Everything else it catches means "we could not get the weights", and
        the loader reports that better. A `VerificationMismatch` means the
        mirror *and* the official source were tried, the bytes were hashed,
        and they disagree with the manifest -- there is no better report to
        wait for, and the loader's own answer would be either garbage output
        or CTranslate2's bare `RuntimeError`.
        """

        from finesub_bootstrap import model_ensure
        from finesub_bootstrap.hf_verify import VerificationMismatch

        def wrong_bytes(*_args, **_kwargs):
            raise VerificationMismatch("whisper 下载后校验失败：model.bin")

        monkeypatch.setattr(model_ensure, "pinned_revision", lambda _id: "abc123")
        monkeypatch.setattr(model_ensure, "ensure_hf_model", wrong_bytes)

        with pytest.raises(VerificationMismatch):
            hf_weights.prepare("whisper")

    def test_a_fetch_that_did_not_land_does_not_earn_the_offline_load(
        self, monkeypatch
    ) -> None:
        """`ensure_hf_model` returning is not the same question: it can write
        into a cache root the loader does not read."""

        from finesub_bootstrap import model_ensure

        monkeypatch.setattr(model_ensure, "pinned_revision", lambda _id: "abc123")
        monkeypatch.setattr(model_ensure, "ensure_hf_model", lambda *_a, **_k: None)
        monkeypatch.setattr(
            model_ensure, "pinned_snapshot_loadable", lambda _id: False
        )

        assert hf_weights.prepare("whisper").local_files_only is False


class TestOfflineFirst:
    """The ladder that lets the gate be optimistic.

    `_hf_repo_complete` accepts one window by design -- an interruption between
    two files -- so `local_files_only` can be wrong. It must cost an attempt,
    never a run.
    """

    def test_a_working_offline_load_is_the_only_attempt(self) -> None:
        seen: list[bool] = []

        def load(plan):
            seen.append(plan.local_files_only)
            return "model"

        plan = hf_weights.HfLoad("abc123", True)
        assert hf_weights.offline_first(load, plan, what="asr") == "model"
        assert seen == [True]

    def test_a_failed_offline_load_falls_back_to_the_hub(self) -> None:
        seen: list[bool] = []

        def load(plan):
            seen.append(plan.local_files_only)
            if plan.local_files_only:
                raise OSError("no local file named tokenizer.json")
            return "model"

        plan = hf_weights.HfLoad("abc123", True)
        assert hf_weights.offline_first(load, plan, what="asr") == "model"
        assert seen == [True, False]

    def test_the_retry_keeps_the_pinned_revision(self) -> None:
        """Falling back to the hub is not licence to re-resolve `main`."""

        seen: list[str | None] = []

        def load(plan):
            seen.append(plan.revision)
            if plan.local_files_only:
                raise OSError("no local file")
            return "model"

        hf_weights.offline_first(
            load, hf_weights.HfLoad("abc123", True), what="asr"
        )
        assert seen == ["abc123", "abc123"]

    def test_a_failure_that_is_not_about_the_cache_is_never_retried(self) -> None:
        """An OOM, a CUDA init failure, a model the library cannot read: each
        would be paid for twice and then reported as whatever the second
        attempt hit -- on an offline machine, a network timeout burying the
        real cause."""

        attempts: list[bool] = []

        def load(plan):
            attempts.append(plan.local_files_only)
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="out of memory"):
            hf_weights.offline_first(
                load, hf_weights.HfLoad("abc123", True), what="asr"
            )
        assert attempts == [True]

    def test_the_offline_error_survives_a_failed_retry(self) -> None:
        """Both attempts failing must not lose the first error: on a machine
        that cannot reach the hub the second one says only "timeout"."""

        def load(plan):
            if plan.local_files_only:
                raise FileNotFoundError("no local file named tokenizer.json")
            raise OSError("connect timed out")

        with pytest.raises(OSError, match="connect timed out") as caught:
            hf_weights.offline_first(
                load, hf_weights.HfLoad("abc123", True), what="asr"
            )
        assert "tokenizer.json" in str(caught.value.__context__)

    def test_an_online_load_that_fails_is_the_caller_s_error(self) -> None:
        """No second attempt, and no wrapping: there was nothing to retry."""

        attempts: list[bool] = []

        def load(plan):
            attempts.append(plan.local_files_only)
            raise OSError("connect timed out")

        with pytest.raises(OSError, match="connect timed out"):
            hf_weights.offline_first(
                load, hf_weights.HfLoad("abc123", False), what="asr"
            )
        assert attempts == [False]

    def test_the_second_failure_is_what_the_caller_sees(self) -> None:
        """The online attempt's error describes what the loader wanted; the
        offline one only ever says a file was missing."""

        def load(plan):
            raise OSError(
                "offline" if plan.local_files_only else "connection refused"
            )

        with pytest.raises(OSError, match="connection refused"):
            hf_weights.offline_first(
                load, hf_weights.HfLoad("abc123", True), what="asr"
            )


def _populate(hub: Path, entry, cache_dir: str) -> Path:
    """A snapshot that passes every cheap check: complete repo, pinned
    revision, every manifest file at exactly its recorded size."""

    snapshot = hub / cache_dir / "snapshots" / entry.revision
    snapshot.mkdir(parents=True)
    (hub / cache_dir / "blobs").mkdir(parents=True)
    for item in entry.files:
        (snapshot / item.name).write_bytes(b"\0" * item.size)
    return snapshot


class TestFailedMarkerIsNotLoadable:
    """A snapshot whose last verification failed must never be offered to a
    loader as offline-ready.

    `_discard_failed` is best effort by construction and says so: it removes
    what it can and leaves the failure marker to fence off the rest. On
    Windows "the rest" is routine -- a file another process still holds stays
    behind, at exactly the size the manifest records. Every check
    `pinned_snapshot_loadable` used to make then answers yes, because none of
    them hashes anything, and bytes we have already judged corrupt get loaded
    offline with no further questions asked.
    """

    def _entry(self):
        from finesub_bootstrap.model_caches import _ENSURABLE_HF_CACHE_DIRS
        from finesub_bootstrap.model_manifest import entry_for

        model_id = "qwen-referee"
        return model_id, entry_for(model_id), _ENSURABLE_HF_CACHE_DIRS[model_id]

    def test_a_verified_snapshot_is_loadable(self, tmp_path, monkeypatch) -> None:
        from finesub_bootstrap import hf_verify, model_ensure

        model_id, entry, cache_dir = self._entry()
        hub = tmp_path / "hub"
        _populate(hub, entry, cache_dir)
        hf_verify.write_marker(hub, cache_dir, entry)
        monkeypatch.setattr(model_ensure, "_loader_hub_dir", lambda: hub)

        assert model_ensure.pinned_snapshot_loadable(model_id) is True

    def test_a_failed_marker_refuses_even_when_the_files_look_right(
        self, tmp_path, monkeypatch
    ) -> None:
        from finesub_bootstrap import hf_verify, model_ensure

        model_id, entry, cache_dir = self._entry()
        hub = tmp_path / "hub"
        _populate(hub, entry, cache_dir)
        # The exact state `verify_and_mark` leaves when the bytes were wrong
        # and `_discard_failed` could not remove them.
        hf_verify.write_marker(
            hub, cache_dir, entry, failed=tuple(item.name for item in entry.files)
        )
        monkeypatch.setattr(model_ensure, "_loader_hub_dir", lambda: hub)

        assert hf_verify.marker_state(hub, cache_dir, entry) == "failed"
        assert model_ensure.pinned_snapshot_loadable(model_id) is False

    def test_the_marker_is_read_at_the_root_the_files_were_stat_ed_in(
        self, tmp_path, monkeypatch
    ) -> None:
        """The two roots are not the same directory on a managed install, and
        an implementation that read the download root would pass every test
        above while never actually reading a marker.

        Here the loader's root is the damaged one and the download root is
        clean, so reading the wrong root answers "loadable" -- which is the
        original defect wearing a different hat.
        """

        from finesub_bootstrap import hf_verify, model_ensure

        model_id, entry, cache_dir = self._entry()
        loader_hub = tmp_path / "loader"
        download_hub = tmp_path / "download"
        _populate(loader_hub, entry, cache_dir)
        _populate(download_hub, entry, cache_dir)
        hf_verify.write_marker(
            loader_hub,
            cache_dir,
            entry,
            failed=tuple(item.name for item in entry.files),
        )
        hf_verify.write_marker(download_hub, cache_dir, entry)
        monkeypatch.setattr(model_ensure, "_loader_hub_dir", lambda: loader_hub)
        monkeypatch.setattr(model_ensure, "_hub_dir", lambda _root=None: download_hub)

        # What makes this test discriminate rather than merely pass: the wrong
        # root says "current", so an implementation reading it would answer
        # loadable.
        assert hf_verify.marker_state(download_hub, cache_dir, entry) == "current"
        assert model_ensure.pinned_snapshot_loadable(model_id) is False

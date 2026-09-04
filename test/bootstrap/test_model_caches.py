from __future__ import annotations

from pathlib import Path

from finesub_bootstrap import model_caches


def test_the_managed_directory_wins_when_it_already_has_the_checkpoint(
    tmp_path: Path,
) -> None:
    managed = tmp_path / "models" / "audio-separator"
    managed.mkdir(parents=True)
    (managed / model_caches.SEPARATOR_CHECKPOINT).write_bytes(b"weights")

    assert (
        model_caches.existing_separator_dir(
            managed, model_caches.SEPARATOR_CHECKPOINT
        )
        == managed
    )


def test_a_checkpoint_the_machine_already_has_is_not_downloaded_again(
    tmp_path: Path, monkeypatch
) -> None:
    # 610MB is worth finding. The managed directory is where new downloads go,
    # not the only place worth looking.
    conventional = tmp_path / "home" / ".cache" / "audio-separator"
    conventional.mkdir(parents=True)
    (conventional / model_caches.SEPARATOR_CHECKPOINT).write_bytes(b"weights")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))

    managed = tmp_path / "models" / "audio-separator"

    assert (
        model_caches.existing_separator_dir(
            managed, model_caches.SEPARATOR_CHECKPOINT
        )
        == conventional
    )


def test_without_a_cached_checkpoint_the_managed_directory_is_used(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    managed = tmp_path / "models" / "audio-separator"

    assert (
        model_caches.existing_separator_dir(
            managed, model_caches.SEPARATOR_CHECKPOINT
        )
        == managed
    )


def test_an_hf_cache_holding_one_of_our_repos_is_reused(
    tmp_path: Path, monkeypatch
) -> None:
    # Hugging Face has one cache root and no way to search several, so the
    # decision is per-cache: if it already holds a repository this pipeline
    # uses, it is used for all of them.
    conventional = tmp_path / "home" / ".cache" / "huggingface"
    (conventional / "hub" / model_caches.HF_REPO_DIRS[0]).mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    managed = tmp_path / "models" / "huggingface"

    assert model_caches.existing_hf_home(managed) == conventional


def test_an_unrelated_hf_cache_is_left_alone(tmp_path: Path, monkeypatch) -> None:
    # Someone else's models are not ours; downloading into their cache would be
    # a surprise, and an uninstall could not clean it up.
    conventional = tmp_path / "home" / ".cache" / "huggingface"
    (conventional / "hub" / "models--someone--else").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    managed = tmp_path / "models" / "huggingface"

    assert model_caches.existing_hf_home(managed) == managed


def test_an_explicit_hf_home_is_never_second_guessed(
    tmp_path: Path, monkeypatch
) -> None:
    conventional = tmp_path / "home" / ".cache" / "huggingface"
    (conventional / "hub" / model_caches.HF_REPO_DIRS[0]).mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "chosen"))

    managed = tmp_path / "models" / "huggingface"

    assert model_caches.existing_hf_home(managed) == managed


def test_every_ensurable_model_matches_the_shipped_manifest() -> None:
    """Cache directory, manifest entry and repository id must agree.

    Three tables have to name the same repository -- the manifest (what to
    fetch and how to verify it), the cache-dir map (where it lands) and
    `HF_REPO_DIRS` (which conventional cache counts as ours). They are three
    tables because they answer three questions, and nothing but this check
    stops one of them from being edited alone.
    """

    from finesub_bootstrap.model_manifest import load_manifest

    manifest = load_manifest()
    for model_id, cache_dir in model_caches._ENSURABLE_HF_CACHE_DIRS.items():
        entry = manifest.get(model_id)
        assert entry is not None, model_id
        assert entry.repo, model_id
        assert cache_dir == f"models--{entry.repo.replace('/', '--')}"
        assert entry.revision, model_id
        assert cache_dir in model_caches.HF_REPO_DIRS

    # The other direction: a Hugging Face model in the manifest that no cache
    # dir names would be fetched but never recognised as already present.
    for model_id, entry in manifest.items():
        if entry.repo:
            assert model_id in model_caches._ENSURABLE_HF_CACHE_DIRS


def test_the_japanese_alternative_is_listed_but_never_prefetched() -> None:
    """Listed alternative, not part of the default roster.

    Being in the manifest buys it mirror routing and digest verification; being
    out of `PIPELINE_MODEL_IDS` is what keeps a default run from downloading
    3 GB nobody asked for.
    """

    assert "whisper-ja" not in model_caches.PIPELINE_MODEL_IDS
    assert model_caches.WHISPER_JA_CACHE_DIR in model_caches.HF_REPO_DIRS

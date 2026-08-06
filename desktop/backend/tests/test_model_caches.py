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

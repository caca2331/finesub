from __future__ import annotations

import os
from pathlib import Path

import pytest

from asr_playground import paths


def test_checkout_root_is_found_from_package_location() -> None:
    root = Path(__file__).resolve().parents[1]

    assert paths.resolve_checkout_root() == root


def test_explicit_root_must_be_a_checkout(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="must be a source checkout"):
        paths.resolve_checkout_root(tmp_path)


def _packaged_install(root: Path, monkeypatch, version: str = "0.3.2") -> Path:
    """A release package's app snapshot, imported from as the pipeline would."""

    source = root / "app" / "versions" / version
    (source / "src" / "asr_playground").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        "[project]\nname='finesub'\n", encoding="utf-8"
    )
    monkeypatch.chdir(source)
    monkeypatch.setattr(
        paths,
        "__file__",
        str(source / "src" / "asr_playground" / "paths.py"),
    )
    for name in (
        "FINESUB_ROOT",
        "FINESUB_KNOWLEDGE_ROOT",
        "FINESUB_STATE_DIR",
        "FINESUB_ENV_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    return source


def _managed_data_root(tmp_path, monkeypatch) -> Path:
    """Point the managed data root at a scratch directory, whatever the OS.

    The product is Windows-only, but the tests also run on Linux (CI), where
    ``default_data_root`` hangs the same layout off ``~/.finesub`` instead of
    ``%LOCALAPPDATA%\\FineSub``.
    """

    local_app_data = tmp_path / "LocalAppData"
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    if os.name == "nt":
        return local_app_data.resolve() / "FineSub"
    return home.resolve() / ".finesub"


def test_a_release_package_is_not_mistaken_for_a_checkout(
    tmp_path, monkeypatch
) -> None:
    # app/versions/<ver> ships pyproject.toml and src/asr_playground, so only
    # the surrounding layout separates it from a checkout. Getting this wrong
    # writes personal data into a directory the next update replaces wholesale.
    root = tmp_path / "FineSub"
    _packaged_install(root, monkeypatch)
    data_root = _managed_data_root(tmp_path, monkeypatch)

    assert paths.resolve_checkout_root() is None
    assert paths.resolve_knowledge_root() == data_root / "user-data" / "knowledge"
    # The launcher points the limiter at the same file, so a task started from
    # the app and one started against its interpreter share the budget.
    assert paths.resolve_state_file() == root.resolve() / "cache" / "state"


def test_a_package_reads_the_env_file_from_the_shared_user_data(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "FineSub"
    _packaged_install(root, monkeypatch)
    user_data = _managed_data_root(tmp_path, monkeypatch) / "user-data"
    user_data.mkdir(parents=True)
    env_file = user_data / ".env"
    env_file.write_text("GEMINI_FREE=test\n", encoding="utf-8")

    assert paths.resolve_env_file() == env_file


def test_env_file_prefers_explicit_configuration(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / "runtime.env"
    env_file.write_text("GEMINI_FREE=test\n", encoding="utf-8")
    monkeypatch.setenv("FINESUB_ENV_FILE", str(env_file))

    assert paths.resolve_env_file() == env_file.resolve()


def test_config_file_prefers_explicit_configuration(tmp_path, monkeypatch) -> None:
    config_file = tmp_path / "runtime.toml"
    config_file.write_text("[pools]\n", encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config_file))

    assert paths.resolve_config_file() == config_file.resolve()


def test_state_dir_falls_back_to_the_managed_layout(tmp_path, monkeypatch) -> None:
    # Not a checkout and not inside a package -- still the same machine, so the
    # limiter state belongs where the launcher would point it rather than in a
    # fifth invented location.
    monkeypatch.setattr(paths, "resolve_checkout_root", lambda *args, **kwargs: None)
    monkeypatch.delenv("FINESUB_STATE_DIR", raising=False)
    data_root = _managed_data_root(tmp_path, monkeypatch)
    # XDG_STATE_HOME is the last resort behind the managed layout, not ahead of
    # it: the limiter state belongs with the rest of this install's data.
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))

    assert paths.resolve_state_file() == data_root / "cache" / "state"


def test_token_counter_candidates_use_checkout(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_TOKEN_COUNTER_EXE", raising=False)
    monkeypatch.setattr(
        paths,
        "resolve_checkout_root",
        lambda *args, **kwargs: tmp_path,
    )

    assert paths.token_counter_candidates() == (
        tmp_path / "bin" / "windows-amd64" / "tokcount.exe",
        tmp_path / "bin" / "gemini-token-counter",
    )


def test_separator_model_dir_honors_finesub_model_dir(monkeypatch, tmp_path) -> None:
    # Nothing is cached anywhere, so new weights go to the managed directory.
    monkeypatch.setenv("FINESUB_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))

    assert paths.resolve_separator_model_dir() == (
        (tmp_path / "models").resolve() / "audio-separator"
    )


def test_a_checkpoint_already_in_the_shared_cache_is_reused(
    monkeypatch, tmp_path
) -> None:
    # The managed directory is where downloads go, not the only place to look:
    # re-fetching 610MB the machine already holds helps nobody.
    from finesub_bootstrap.model_caches import SEPARATOR_CHECKPOINT

    shared = tmp_path / "home" / ".cache" / "audio-separator"
    shared.mkdir(parents=True)
    (shared / SEPARATOR_CHECKPOINT).write_bytes(b"weights")
    monkeypatch.setenv("FINESUB_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))

    assert paths.resolve_separator_model_dir() == shared
    # The compiled accel cache still belongs to this install: it is keyed to one
    # torch build and one GPU, and a shared cache could not attribute it.
    assert paths.managed_separator_model_dir() == (
        (tmp_path / "models").resolve() / "audio-separator"
    )


def test_separator_model_dir_defaults_to_the_shared_user_cache(monkeypatch) -> None:
    monkeypatch.delenv("FINESUB_MODEL_DIR", raising=False)

    assert paths.resolve_separator_model_dir() == (
        Path.home() / ".cache" / "audio-separator"
    )


def test_knowledge_root_never_falls_back_to_cwd(tmp_path, monkeypatch) -> None:
    # Outside a checkout and outside a package it resolves to the one managed
    # location every front end shares -- never to whatever directory the
    # process happens to have been started in.
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.setattr(paths, "resolve_checkout_root", lambda *args, **kwargs: None)
    data_root = _managed_data_root(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)

    resolved = paths.resolve_knowledge_root()

    assert resolved == data_root / "user-data" / "knowledge"
    assert paths.resolve_knowledge_root(required=False) == resolved


def _checkout(root: Path) -> Path:
    (root / "src" / "asr_playground").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='finesub'\n", encoding="utf-8"
    )
    return root


def test_a_checkout_keeps_its_own_data_by_default(tmp_path, monkeypatch) -> None:
    # Almost every run from a checkout is development, and the repository's own
    # .env holds the real API keys; sending those runs at the shared knowledge
    # base would mix development noise into it irreversibly.
    checkout = _checkout(tmp_path / "repo")
    monkeypatch.chdir(checkout)
    monkeypatch.delenv("FINESUB_CHECKOUT_DATA", raising=False)
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.delenv("FINESUB_STATE_DIR", raising=False)

    assert paths.resolve_knowledge_root() == checkout.resolve() / "knowledge"
    assert paths.resolve_state_file() == checkout.resolve() / ".state"


def test_a_checkout_can_be_opted_out_of_its_own_data(tmp_path, monkeypatch) -> None:
    checkout = _checkout(tmp_path / "repo")
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("FINESUB_CHECKOUT_DATA", "0")
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    data_root = _managed_data_root(tmp_path, monkeypatch)

    assert paths.resolve_knowledge_root() == data_root / "user-data" / "knowledge"


def test_a_worktree_uses_the_main_checkouts_data(tmp_path, monkeypatch) -> None:
    # A linked worktree carries no untracked knowledge/ or .env of its own, so
    # resolving to itself would start an empty knowledge base for every
    # throwaway branch. Three segments up from the recorded gitdir: dropping
    # <name>, worktrees and .git.
    main = _checkout(tmp_path / "main")
    (main / ".git" / "worktrees" / "feature").mkdir(parents=True)
    worktree = _checkout(tmp_path / "feature")
    (worktree / ".git").write_text(
        f"gitdir: {main / '.git' / 'worktrees' / 'feature'}\n", encoding="utf-8"
    )
    monkeypatch.chdir(worktree)
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)

    assert paths.resolve_checkout_root() == main.resolve()
    assert paths.resolve_knowledge_root() == main.resolve() / "knowledge"
    assert paths.is_linked_worktree()


def test_a_relative_gitdir_pointer_is_resolved(tmp_path, monkeypatch) -> None:
    main = _checkout(tmp_path / "main")
    (main / ".git" / "worktrees" / "feature").mkdir(parents=True)
    worktree = _checkout(tmp_path / "feature")
    (worktree / ".git").write_text(
        "gitdir: ../main/.git/worktrees/feature\n", encoding="utf-8"
    )
    monkeypatch.chdir(worktree)

    assert paths.resolve_checkout_root() == main.resolve()


def test_an_ordinary_checkout_is_not_a_worktree(tmp_path, monkeypatch) -> None:
    checkout = _checkout(tmp_path / "repo")
    (checkout / ".git").mkdir()
    monkeypatch.chdir(checkout)

    assert paths.resolve_checkout_root() == checkout.resolve()
    assert not paths.is_linked_worktree()


def test_runtime_modules_do_not_infer_root_from_parent_depth() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    offenders = [
        source.relative_to(source_root)
        for source in source_root.rglob("*.py")
        if source != Path(paths.__file__).resolve()
        and ".parents[" in source.read_text(encoding="utf-8")
    ]

    assert offenders == []

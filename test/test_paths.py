from __future__ import annotations

import os
from pathlib import Path

import pytest

from asr_playground import paths


def test_checkout_root_is_found_from_package_location() -> None:
    root = Path(__file__).resolve().parents[1]

    assert paths.resolve_checkout_root() == root


def test_explicit_root_must_be_a_checkout(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing pyproject"):
        paths.resolve_checkout_root(tmp_path)


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


def test_state_dir_has_wheel_safe_user_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "resolve_checkout_root", lambda *args, **kwargs: None)
    local_app_data = tmp_path / "local-app-data"
    xdg_state_home = tmp_path / "xdg-state"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg_state_home))

    expected = (
        local_app_data / "FineSub" / "state"
        if os.name == "nt"
        else xdg_state_home / "finesub"
    )
    assert paths.resolve_state_file() == expected


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


def test_knowledge_root_never_falls_back_to_cwd(monkeypatch) -> None:
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    monkeypatch.setattr(paths, "resolve_checkout_root", lambda *args, **kwargs: None)

    assert paths.resolve_knowledge_root(required=False) is None
    with pytest.raises(RuntimeError, match="Knowledge root is unavailable"):
        paths.resolve_knowledge_root()


def test_runtime_modules_do_not_infer_root_from_parent_depth() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    offenders = [
        source.relative_to(source_root)
        for source in source_root.rglob("*.py")
        if source != Path(paths.__file__).resolve()
        and ".parents[" in source.read_text(encoding="utf-8")
    ]

    assert offenders == []

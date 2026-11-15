from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import sys

import pytest

from finesub import paths


@pytest.mark.requires_main_checkout
def test_checkout_root_is_found_from_package_location() -> None:
    root = Path(__file__).resolve().parents[1]

    assert paths.resolve_checkout_root() == root


def test_explicit_root_must_be_a_checkout(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="must be a source checkout"):
        paths.resolve_checkout_root(tmp_path)


def _packaged_install(root: Path, monkeypatch, version: str = "0.3.2") -> Path:
    """A release package's app snapshot, imported from as the pipeline would."""

    source = root / "app" / "versions" / version
    (source / "src" / "finesub").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        "[project]\nname='finesub'\n", encoding="utf-8"
    )
    monkeypatch.chdir(source)
    monkeypatch.setattr(
        paths,
        "__file__",
        str(source / "src" / "finesub" / "paths.py"),
    )
    for name in (
        "FINESUB_ROOT",
        "FINESUB_KNOWLEDGE_ROOT",
        "FINESUB_STATE_FILE",
        "FINESUB_ENV_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    return source


def _managed_data_root(tmp_path, monkeypatch) -> Path:
    """Point the managed data root at a scratch directory, whatever the OS.

    The tests run on Windows, Linux and macOS (CI), and ``default_data_root``
    hangs the same layout off each platform's own convention.
    """

    local_app_data = tmp_path / "LocalAppData"
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    if os.name == "nt":
        return local_app_data.resolve() / "FineSub"
    if sys.platform == "darwin":
        return home.resolve() / "Library" / "Application Support" / "FineSub"
    return home.resolve() / ".finesub"


def test_a_release_package_is_not_mistaken_for_a_checkout(
    tmp_path, monkeypatch
) -> None:
    # app/versions/<ver> ships pyproject.toml and src/finesub, so only
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


def test_state_file_falls_back_to_the_managed_layout(tmp_path, monkeypatch) -> None:
    # Not a checkout and not inside a package -- still the same machine, so the
    # limiter state belongs where the launcher would point it rather than in a
    # fifth invented location.
    monkeypatch.setattr(paths, "resolve_checkout_root", lambda *args, **kwargs: None)
    monkeypatch.delenv("FINESUB_STATE_FILE", raising=False)
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

    # Exactly one. The second candidate used to be `bin/gemini-token-counter`,
    # the Go module's old name, and no such file has ever been in this repo.
    assert paths.token_counter_candidates() == (
        tmp_path / "bin" / "windows-amd64" / "tokcount.exe",
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
    (root / "src" / "finesub").mkdir(parents=True)
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
    monkeypatch.delenv("FINESUB_STATE_FILE", raising=False)

    assert paths.resolve_knowledge_root() == checkout.resolve() / "knowledge"
    assert paths.resolve_state_file() == checkout.resolve() / ".state"


def test_a_checkout_can_be_opted_out_of_its_own_data(tmp_path, monkeypatch) -> None:
    checkout = _checkout(tmp_path / "repo")
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("FINESUB_CHECKOUT_DATA", "0")
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    data_root = _managed_data_root(tmp_path, monkeypatch)

    assert paths.resolve_knowledge_root() == data_root / "user-data" / "knowledge"


def test_an_empty_checkout_data_switch_counts_as_unset(tmp_path, monkeypatch) -> None:
    checkout = _checkout(tmp_path / "repo")
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("FINESUB_CHECKOUT_DATA", " ")
    monkeypatch.delenv("FINESUB_KNOWLEDGE_ROOT", raising=False)
    _managed_data_root(tmp_path, monkeypatch)

    assert paths.checkout_data_enabled()
    assert paths.resolve_knowledge_root() == checkout.resolve() / "knowledge"


def test_installed_code_started_in_a_checkout_uses_user_data(tmp_path, monkeypatch) -> None:
    # Audit X-3: the packaged launcher's runtime or a wheel, run with a checkout
    # (or anything laid out like one) as the working directory, used to take that
    # directory's .env, config.toml and model directory.
    checkout = _checkout(tmp_path / "repo")
    (checkout / ".env").write_text("FINESUB_KEY=planted\n", encoding="utf-8")
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(paths, "__file__", str(tmp_path / "site-packages" / "finesub" / "paths.py"))
    for name in ("FINESUB_CHECKOUT_DATA", "FINESUB_KNOWLEDGE_ROOT", "FINESUB_STATE_FILE", "FINESUB_ENV_FILE"):
        monkeypatch.delenv(name, raising=False)
    data_root = _managed_data_root(tmp_path, monkeypatch)

    assert paths.own_checkout_root() is None
    assert paths.resolve_knowledge_root() == data_root / "user-data" / "knowledge"
    assert paths.resolve_env_file() is None


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


def test_the_provisioning_layer_stays_reachable_without_pydantic() -> None:
    """Every lazy import in this module has to work on a `[harness]` install.

    `finesub_bootstrap` as a whole needs pydantic, `[harness]` does not install
    it, and this module reaches into that package four times. It works only
    because the modules it picks -- `paths`, and through it `fsops`/`locks` --
    happen to be stdlib-only, and nothing writes that down: adding
    `from finesub_bootstrap.models import ...` to `finesub_bootstrap/paths.py`
    is a perfectly reasonable local change whose damage lands here.

    Both failure modes are covered because they look nothing alike.
    `_packaged_root` has no guard, so it would raise -- on the hot path, since
    `_is_checkout_root` calls it for every candidate. `_managed_paths` catches
    ImportError and would quietly start answering None, sending a bare wheel
    install's personal data to an invented location instead of the shared one.

    The root suite runs without pydantic, which is what makes this a real test
    rather than a description of one; it is skipped anywhere that is untrue.
    """

    if importlib.util.find_spec("pydantic") is not None:
        pytest.skip("only meaningful in an environment without pydantic")

    # Raises if the un-guarded import site regressed.
    paths._packaged_root(Path(__file__).resolve())
    paths._packaged_paths()
    # None here means the ImportError branch fired: same regression, silent.
    assert paths._managed_paths() is not None


def test_the_launcher_and_the_pipeline_agree_on_variable_names() -> None:
    """The front ends configure the pipeline through names, not references.

    `shared_environment_overrides` writes `FINESUB_*` variables that
    `finesub` reads; nothing connects the two but the strings
    themselves, so renaming one side breaks nothing at import time and the
    pipeline simply resolves somewhere else -- a knowledge base that silently
    moves is the worst shape this can fail in.

    Read from source rather than imported: `finesub_bootstrap.environment`
    pulls in pydantic, which a `[harness]` install (this suite) does not have.
    The write side matches assignments only (`overrides["X"] =` and `"X":` in
    the worker's environment dict), so a variable the launcher merely *reads*
    for itself -- `FINESUB_HOME` -- is not mistaken for something it promises
    the pipeline. The read side stays deliberately loose: any mention counts.
    """

    source_root = Path(__file__).resolve().parents[1] / "src"
    variables = re.compile(r"FINESUB_[A-Z_]+")
    assigned = re.compile(r"\"(FINESUB_[A-Z_]+)\"\s*(?:\]\s*=|:)")

    written = set(
        assigned.findall(
            (source_root / "finesub_bootstrap" / "environment.py").read_text(
                encoding="utf-8"
            )
        )
    )
    read = set()
    # `finesub` alone, because the harness moved inside it. The second entry
    # used to be `llm`; left in place it would `rglob` a directory that no
    # longer exists and contribute nothing, without saying so.
    for source in (source_root / "finesub").rglob("*.py"):
        read |= set(variables.findall(source.read_text(encoding="utf-8")))

    assert written, "the launcher configures the pipeline through these"
    assert written <= read, sorted(written - read)


def test_cleanup_names_the_same_artifacts_the_pipeline_derives() -> None:
    """`finesub_bootstrap.artifacts` restates the pipeline's naming.

    It has to: the CLI shell decides what to clean up before it has an
    interpreter that could import the pipeline, and that layer must not depend
    on it anyway. A restatement is safe only while something compares the two,
    because the failure is silent in both directions -- a renamed artifact
    stops being cleaned (the tasks tree quietly grows) or, worse, a stale name
    starts matching something else.

    `-stable.json` and `-annotated.csv` are asserted absent rather than merely
    left out: keeping them is the entire point of the mode, and "we forgot to
    add it to the list" and "we deliberately never remove it" look identical
    in a list of things we do remove.
    """

    from finesub.llm.stages.correction.run import PASS2_SUFFIX
    from finesub.speech.diarization.evidence import evidence_path
    from finesub.stages import default_pipeline_paths
    from finesub_bootstrap import artifacts

    delivered = Path("C:/tasks/clip-260811-2205-abc123/clip.srt")
    paths = default_pipeline_paths("C:/media/clip.wav", delivered)
    derived = {
        Path(paths.vocal_audio),
        Path(paths.vocal_audio).with_suffix(".flac"),
        Path(paths.vad_json),
        Path(paths.vad_energy_npz),
        Path(paths.aligned_json),
        # The clustering's evidence sidecar: reproducible from the audio, unlike the timeline
        # it sits beside, so it is cleaned while -diar.json is kept.
        evidence_path(paths.diar_json),
        Path(paths.raw_srt),
        Path(paths.translated_srt),
        Path(paths.metadata_json),
        Path(paths.final_srt),
        # Named after the *source* media, so the pipeline derives it from a
        # different stem: this entry only catches the (common) case where the
        # two stems agree. The sidecar record is what finds the rest.
        Path(paths.final_srt).with_name(f"{Path(paths.final_srt).stem}-decoded.flac"),
        # Three passes: pass 2's products, named after the correction's own constant.
        *(Path(paths.final_srt).with_name(f"{Path(paths.final_srt).stem}{PASS2_SUFFIX}{name}")
          for name in ("-translated.srt", "-corrected.srt", "-annotated.csv", ".srt")),
    }

    removable = set(artifacts.removable_artifacts(delivered))

    assert removable == {path.resolve() for path in derived}
    assert artifacts.artifact_directory(delivered) == Path(
        paths.task_artifact_dir
    ).resolve()
    # The deliverable has to be the file the pipeline actually writes, for
    # every shape of `-o` -- it only supplies `.srt` when there is no suffix at
    # all, so an output named `.txt` stays `.txt` and forcing `.srt` here would
    # look for a file that was never produced.
    for requested in ("C:/out/subs.txt", "C:/out/subs", "C:/out/subs.srt"):
        expected = default_pipeline_paths("C:/media/clip.wav", requested)
        assert artifacts.deliverable(requested, "final-srt") == Path(
            expected.final_srt
        ).resolve()
        assert artifacts.deliverable(requested, "raw-srt") == Path(
            expected.raw_srt
        ).resolve()

    assert Path(paths.stable_json).resolve() not in removable
    assert (
        Path(paths.final_srt).with_name(
            f"{Path(paths.final_srt).stem}-annotated.csv"
        ).resolve()
        not in removable
    )


def test_every_pipeline_path_is_either_removable_or_a_record() -> None:
    """Enumerate the fields rather than list them, so a new one cannot hide.

    The test above builds its expected set by hand, which is exactly how
    `-diar.json` got past it: a new `PipelinePaths` field was in neither list and
    nothing noticed, because nothing asked about fields it had not been told of.
    """

    from finesub.stages import PipelinePaths, default_pipeline_paths
    from finesub_bootstrap import artifacts

    delivered = Path("C:/tasks/clip-260811-2205-abc123/clip.srt")
    paths = default_pipeline_paths("C:/media/clip.wav", delivered)
    removable = set(artifacts.removable_artifacts(delivered))
    stem = delivered.with_suffix("")
    records = {
        stem.with_name(f"{stem.name}{suffix}").resolve()
        for suffix in artifacts.RECORD_SUFFIXES
    }
    kept = {
        stem.with_name(f"{stem.name}{suffix}").resolve()
        for suffix in artifacts.KEPT_DIRECTORY_SUFFIXES
    }
    directories = {"task_artifact_dir"}

    unclassified = [
        field
        for field in PipelinePaths._fields
        if field not in directories
        and Path(getattr(paths, field)).resolve() not in removable | records | kept
    ]
    assert not unclassified, f"classify these in finesub_bootstrap.artifacts: {unclassified}"
    assert not removable & records, "a record must never be in the list of things we delete"
    # The bootstrap restates the judge's suffix (it may not import the pipeline);
    # the field and the reader must agree with it.
    from finesub.speech.diarization.judgments import SIDECAR_SUFFIX, sidecar_path
    from finesub.subtitles.stable_rows import OVERLAY_SUFFIX, overlay_path

    assert SIDECAR_SUFFIX in artifacts.RECORD_SUFFIXES
    assert sidecar_path(paths.stable_json) == paths.speaker_judge_json
    assert OVERLAY_SUFFIX in artifacts.RECORD_SUFFIXES
    assert overlay_path(paths.stable_json) == paths.diar_rows_json


def test_runtime_modules_do_not_infer_root_from_parent_depth() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    offenders = [
        source.relative_to(source_root)
        for source in source_root.rglob("*.py")
        if source != Path(paths.__file__).resolve()
        and ".parents[" in source.read_text(encoding="utf-8")
    ]

    assert offenders == []

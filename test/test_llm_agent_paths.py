from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub_bootstrap.locks import holding_activity
from finesub.llm.agent import agent_cleanup, agent_paths
from finesub.llm.agent.agent_paths import AgentEpisodeLocation
from finesub.llm.agent.local_agent import AgentDriverConfig, LocalAgentDriver


def _checkout(root: Path) -> Path:
    (root / "src" / "finesub").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='finesub'\n", encoding="utf-8")
    return root


def test_checkout_episode_parent_is_outside_the_repository(
    tmp_path: Path, monkeypatch
) -> None:
    checkout = _checkout(tmp_path / "repo")
    monkeypatch.setattr(agent_paths, "resolve_checkout_root", lambda: checkout)
    monkeypatch.setattr(agent_paths, "checkout_data_enabled", lambda: True)
    monkeypatch.setattr(agent_paths.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))

    location = agent_paths.resolve_agent_episode_location()

    assert location.domain_identity_anchor == checkout / ".state"
    assert location.activity_root == checkout
    assert location.parent.parent == (tmp_path / "temp" / "finesub-agent-runtime").resolve()
    assert checkout.resolve() not in location.parent.parents
    assert len(location.location_identity) == 64


def test_main_checkout_and_linked_worktree_share_one_domain(
    tmp_path: Path, monkeypatch
) -> None:
    main = _checkout(tmp_path / "main")
    (main / ".git" / "worktrees" / "feature").mkdir(parents=True)
    worktree = _checkout(tmp_path / "feature")
    (worktree / ".git").write_text(
        f"gitdir: {main / '.git' / 'worktrees' / 'feature'}\n", encoding="utf-8"
    )
    monkeypatch.setattr(agent_paths, "checkout_data_enabled", lambda: True)
    monkeypatch.setattr(agent_paths.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))

    monkeypatch.chdir(main)
    main_location = agent_paths.resolve_agent_episode_location()
    monkeypatch.chdir(worktree)
    worktree_location = agent_paths.resolve_agent_episode_location()

    assert worktree_location.location_identity == main_location.location_identity
    assert worktree_location.parent == main_location.parent
    assert worktree_location.activity_root == main.resolve()


def test_independent_clones_have_distinct_domains(tmp_path: Path, monkeypatch) -> None:
    first = _checkout(tmp_path / "first")
    second = _checkout(tmp_path / "second")
    monkeypatch.setattr(agent_paths, "checkout_data_enabled", lambda: True)
    monkeypatch.setattr(agent_paths.tempfile, "gettempdir", lambda: str(tmp_path / "temp"))

    monkeypatch.setattr(agent_paths, "resolve_checkout_root", lambda: first)
    first_location = agent_paths.resolve_agent_episode_location()
    monkeypatch.setattr(agent_paths, "resolve_checkout_root", lambda: second)
    second_location = agent_paths.resolve_agent_episode_location()

    assert first_location.location_identity != second_location.location_identity
    assert first_location.parent != second_location.parent


@pytest.mark.skipif(agent_paths.os.name != "nt", reason="Windows path identity")
def test_windows_path_case_does_not_split_a_domain(tmp_path: Path) -> None:
    mixed = Path(str(tmp_path).swapcase())

    assert agent_paths.location_identity("machine_temp", tmp_path) == (
        agent_paths.location_identity("machine_temp", mixed)
    )


def test_opted_out_checkout_uses_managed_big_data(
    tmp_path: Path, monkeypatch
) -> None:
    paths = SimpleNamespace(
        user_data=(tmp_path / "data" / "user-data").resolve(),
        agent_capsules=(tmp_path / "store" / "agent-capsules").resolve(),
    )
    monkeypatch.setattr(agent_paths, "checkout_data_enabled", lambda: False)
    monkeypatch.setattr(agent_paths, "resolve_managed_app_paths", lambda: paths)

    location = agent_paths.resolve_agent_episode_location()

    assert location.parent == paths.agent_capsules
    assert location.activity_root == paths.user_data
    assert location.domain_identity_anchor == paths.user_data
    assert location.locator_kind == agent_paths.LOCATOR_MANAGED


def test_driver_holds_activity_and_reresolves_parent(
    tmp_path: Path, monkeypatch
) -> None:
    activity_root = tmp_path / "coordination"
    first = AgentEpisodeLocation(
        parent=tmp_path / "old",
        domain_identity_anchor=tmp_path / "anchor",
        activity_root=activity_root,
        locator_kind="machine_temp",
        location_identity="a" * 64,
    )
    second = AgentEpisodeLocation(
        parent=tmp_path / "new",
        domain_identity_anchor=first.domain_identity_anchor,
        activity_root=activity_root,
        locator_kind=first.locator_kind,
        location_identity=first.location_identity,
    )
    driver = LocalAgentDriver(AgentDriverConfig(runtime_root=tmp_path / "unused"))
    locations = iter((first, second))
    monkeypatch.setattr(driver.capsules, "resolve_location", lambda: next(locations))
    observed: list[AgentEpisodeLocation] = []

    def fake_run(location, _messages, **_kwargs):
        observed.append(location)
        assert list((activity_root / ".task-activity").glob("*.lock"))
        return "done"

    monkeypatch.setattr(driver, "_run_episode", fake_run)

    assert driver.run([], task="test") == "done"
    assert observed == [second]
    assert not list((activity_root / ".task-activity").glob("*.lock"))


def test_default_cleanup_only_removes_the_current_domain(
    tmp_path: Path, monkeypatch
) -> None:
    own = tmp_path / "runtime" / "own"
    other = tmp_path / "runtime" / "other"
    (own / "failed-episode").mkdir(parents=True)
    (other / "failed-episode").mkdir(parents=True)
    location = AgentEpisodeLocation(
        parent=own,
        domain_identity_anchor=tmp_path / "anchor",
        activity_root=tmp_path / "coordination",
        locator_kind="machine_temp",
        location_identity="b" * 64,
    )
    monkeypatch.setattr(
        agent_cleanup, "resolve_agent_episode_location", lambda: location
    )

    assert agent_cleanup.main([]) == 0
    assert not own.exists()
    assert other.is_dir()


def test_evidence_locator_follows_the_current_managed_parent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    user_data = tmp_path / "user-data"
    parent = tmp_path / "relocated" / "agent-capsules"
    identity = agent_paths.location_identity(
        agent_paths.LOCATOR_MANAGED, user_data
    )
    episode = parent / "failed-call"
    episode.mkdir(parents=True)
    location = AgentEpisodeLocation(
        parent=parent,
        domain_identity_anchor=user_data,
        activity_root=user_data,
        locator_kind=agent_paths.LOCATOR_MANAGED,
        location_identity=identity,
    )
    monkeypatch.setattr(
        agent_paths, "resolve_agent_episode_location", lambda *_args: location
    )
    locator = {
        "locator_kind": agent_paths.LOCATOR_MANAGED,
        "location_identity": identity,
        "episode_id": episode.name,
        "absolute_at_write": str(tmp_path / "old" / episode.name),
    }

    assert agent_cleanup.main(["--locate", json.dumps(locator)]) == 0
    assert capsys.readouterr().out.strip() == str(episode.resolve())


def test_evidence_locator_rejects_episode_traversal(tmp_path: Path) -> None:
    locator = {
        "locator_kind": agent_paths.LOCATOR_MACHINE_TEMP,
        "location_identity": "d" * 64,
        "episode_id": "../outside",
        "absolute_at_write": str(tmp_path / "outside"),
    }

    with pytest.raises(ValueError, match="Invalid agent episode id"):
        agent_paths.resolve_evidence_locator(locator)


def test_cleanup_refuses_an_active_domain(tmp_path: Path, monkeypatch) -> None:
    parent = tmp_path / "runtime" / "own"
    (parent / "failed-episode").mkdir(parents=True)
    activity_root = tmp_path / "coordination"
    location = AgentEpisodeLocation(
        parent=parent,
        domain_identity_anchor=tmp_path / "anchor",
        activity_root=activity_root,
        locator_kind="machine_temp",
        location_identity="c" * 64,
    )
    monkeypatch.setattr(
        agent_cleanup, "resolve_agent_episode_location", lambda: location
    )

    with holding_activity(activity_root):
        assert agent_cleanup.main([]) == 1
    assert parent.is_dir()


def test_all_domains_cleanup_lists_and_removes_legacy_and_managed_roots(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    temp_root = tmp_path / "temp" / "finesub-agent-runtime"
    (temp_root / "domain" / "failed").mkdir(parents=True)
    legacy_lock = temp_root.with_name(temp_root.name + ".lock")
    legacy_lock.write_text("", encoding="utf-8")
    managed = tmp_path / "shared" / "agent-capsules"
    (managed / "failed").mkdir(parents=True)
    custom = tmp_path / "custom" / "agent-capsules"
    (custom / "failed").mkdir(parents=True)
    location = AgentEpisodeLocation(
        parent=custom,
        domain_identity_anchor=tmp_path / "user-data",
        activity_root=tmp_path / "coordination",
        locator_kind=agent_paths.LOCATOR_MANAGED,
        location_identity="e" * 64,
    )
    monkeypatch.setattr(
        agent_cleanup, "resolve_agent_episode_location", lambda: location
    )
    monkeypatch.setattr(agent_cleanup, "machine_temp_root", lambda: temp_root)
    monkeypatch.setattr(
        agent_cleanup, "managed_agent_capsule_parent", lambda: managed
    )

    assert agent_cleanup.main(["--all-domains", "--force"]) == 0

    output = capsys.readouterr().out
    for target in (temp_root, legacy_lock, managed, custom):
        assert str(target.resolve()) in output
        assert not target.exists()


def test_cleanup_package_imports_without_runtime_dependencies(tmp_path: Path) -> None:
    import subprocess
    import sys

    source = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import importlib.abc, sys\n"
        f"sys.path.insert(0, {str(source)!r})\n"
        "blocked={'httpx','pydantic','tomllib','torch'}\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.partition('.')[0] in blocked: raise ImportError(fullname)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import finesub.llm.agent.agent_cleanup\n"
        "assert not any(name.partition('.')[0] in blocked for name in sys.modules)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr


def test_an_explicit_runtime_root_locator_still_finds_its_evidence(tmp_path) -> None:
    """A named parent cannot be re-derived from a hash of itself.

    Machine-temp and managed domains both live at a path the resolver can
    rebuild, so their identity hash is enough to find them again. An explicit
    parent does not, and rebuilding it as `machine_temp_root()/identity`
    produced a path that never existed -- `--locate` printed it, exited 1, and
    the real evidence sat untouched somewhere else.
    """

    from finesub.llm.agent import agent_paths

    parent = tmp_path / "chosen-parent"
    location = agent_paths.resolve_agent_episode_location(parent)
    assert location.locator_kind == agent_paths.LOCATOR_EXPLICIT

    locator = agent_paths.evidence_locator(location, "episode-1").as_dict()
    assert agent_paths.resolve_evidence_locator(locator) == parent / "episode-1"


def test_an_explicit_locator_will_not_follow_a_rewritten_path(tmp_path) -> None:
    """The recorded path is the only route back, so it is verified, not trusted."""

    from finesub.llm.agent import agent_paths

    location = agent_paths.resolve_agent_episode_location(tmp_path / "chosen-parent")
    locator = agent_paths.evidence_locator(location, "episode-1").as_dict()
    locator["absolute_at_write"] = str(tmp_path / "somewhere-else" / "episode-1")

    with pytest.raises(ValueError, match="recorded parent"):
        agent_paths.resolve_evidence_locator(locator)


def test_the_managed_locator_kind_has_one_spelling() -> None:
    """Two launchers write it and `agent_paths` refuses anything else.

    A typo in either would only surface at run time, as a RuntimeError from a
    resolver that could not place the episode.
    """

    from finesub_bootstrap.locks import AGENT_LOCATOR_KIND_MANAGED
    from finesub.llm.agent.agent_paths import LOCATOR_MANAGED

    assert LOCATOR_MANAGED == AGENT_LOCATOR_KIND_MANAGED

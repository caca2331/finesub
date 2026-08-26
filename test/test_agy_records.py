"""agy project records: swept by directory ownership, tolerant of foreign ground."""

from __future__ import annotations

import json
from pathlib import Path

from finesub_bootstrap import agy_records


def _record(path: Path, folder: Path | str, **extra) -> None:
    # `as_posix()` already writes both of agy's spellings correctly: a Windows
    # path gives `C:/x`, so the drive lands where a host would (`file://C:/x`),
    # and a POSIX path keeps its root (`file:///tmp/x`). This used to
    # `lstrip("/")` first, which is a no-op on Windows and turns every POSIX
    # path into a *relative* URI -- the sweep then matched nothing, and three
    # tests failed on Linux only.
    uri = folder if isinstance(folder, str) else "file://" + folder.resolve().as_posix()
    path.write_text(
        json.dumps({"id": path.stem, "projectResources": {"resources": [{"folderUri": uri}]}, **extra}),
        encoding="utf-8",
    )


def test_records_under_a_removed_root_are_swept_and_others_kept(tmp_path) -> None:
    records = tmp_path / "projects"
    records.mkdir()
    domain = tmp_path / "agent-capsules"
    (domain / ".finesub-tool-0").mkdir(parents=True)
    keep_dir = tmp_path / "users-own-project"
    keep_dir.mkdir()
    _record(records / "a.json", domain)
    _record(records / "b.json", domain / ".finesub-tool-0")
    _record(records / "keep.json", keep_dir)
    (records / "broken.json").write_text("{not json", encoding="utf-8")
    _record(records / "remote.json", "https://example.com/x")
    (records / "nofolder.json").write_text(json.dumps({"id": "x"}), encoding="utf-8")

    removed = agy_records.remove_project_records_under([domain], records_dir=records)

    assert sorted(p.name for p in removed) == ["a.json", "b.json"]
    assert sorted(p.name for p in records.glob("*.json")) == [
        "broken.json", "keep.json", "nofolder.json", "remote.json",
    ]


def test_a_missing_records_directory_is_an_empty_sweep(tmp_path) -> None:
    assert agy_records.remove_project_records_under(
        [tmp_path], records_dir=tmp_path / "nowhere"
    ) == []
    assert agy_records.remove_project_records_under([], records_dir=tmp_path) == []


def test_windows_style_drive_host_uris_resolve() -> None:
    """Both spellings mean one path, and on every platform.

    Which one a record holds depends on the machine that wrote it, so reading
    it must not depend on the machine reading it -- the strip used to be
    guarded on `os.name` and this assertion only held on Windows.
    """

    record = {"projectResources": {"resources": [
        {"folderUri": "file://C:/Users/x/agent-capsules"},
        {"folderUri": "file:///C:/Users/x/other"},
    ]}}
    folders = [p.as_posix() for p in agy_records._folder_paths(record)]
    assert folders == ["C:/Users/x/agent-capsules", "C:/Users/x/other"]


def test_posix_uris_keep_their_root() -> None:
    """The other half of the same rule: `/` is a root, not a stray slash.

    A drive-letter strip that fired on any leading slash would turn these into
    relative paths, and the ownership check resolves against the cwd -- so the
    sweep would silently match nothing.
    """

    record = {"projectResources": {"resources": [
        {"folderUri": "file:///home/x/agent-capsules"},
        {"folderUri": "file:///a:b"},
    ]}}
    folders = [p.as_posix() for p in agy_records._folder_paths(record)]
    assert folders == ["/home/x/agent-capsules", "/a:b"]


def test_all_domains_cleanup_sweeps_the_records(tmp_path, monkeypatch, capsys) -> None:
    from finesub.llm.agent import agent_cleanup, agent_paths
    from finesub.llm.agent.agent_paths import AgentEpisodeLocation

    custom = tmp_path / "custom" / "agent-capsules"
    (custom / "failed").mkdir(parents=True)
    temp_root = tmp_path / "temp" / "finesub-agent-runtime"
    records = tmp_path / "projects"
    records.mkdir()
    _record(records / "ours.json", custom)
    _record(records / "theirs.json", tmp_path / "elsewhere")
    monkeypatch.setattr(agy_records, "agy_project_records_dir", lambda: records)
    location = AgentEpisodeLocation(
        parent=custom,
        domain_identity_anchor=tmp_path / "user-data",
        activity_root=tmp_path / "coordination",
        locator_kind=agent_paths.LOCATOR_MANAGED,
        location_identity="e" * 64,
    )
    monkeypatch.setattr(agent_cleanup, "resolve_agent_episode_location", lambda: location)
    monkeypatch.setattr(agent_cleanup, "machine_temp_root", lambda: temp_root)
    monkeypatch.setattr(agent_cleanup, "managed_agent_capsule_parent", lambda: None)

    assert agent_cleanup.main(["--all-domains", "--force"]) == 0

    assert "removed agy project record" in capsys.readouterr().out
    assert not (records / "ours.json").exists()
    assert (records / "theirs.json").exists()

"""The suite runs against tmp, not the checkout or the developer's home.

`test/conftest.py` moves every location a checkout run resolves to itself --
and the agy data FineSub keeps under the home directory -- into a per-test
tmp directory, and fails the
session if one leaked anyway. These pin both halves: each location really
resolves into tmp while a test runs, and the leak check recognises what a
leak looks like.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _in_tmp(path: Path, tmp_root: Path) -> bool:
    return Path(path).resolve().is_relative_to(tmp_root)


@pytest.fixture
def tmp_root(tmp_path_factory) -> Path:
    return tmp_path_factory.getbasetemp().resolve()


def test_machine_state_and_secrets_resolve_into_tmp(tmp_root: Path) -> None:
    from finesub import paths

    assert _in_tmp(paths.resolve_state_file(), tmp_root)
    assert paths.resolve_env_file() is None


def test_the_reference_data_root_resolves_into_tmp(tmp_root: Path) -> None:
    from finesub import paths
    from finesub.media import source

    assert _in_tmp(paths.resolve_reference_data_root(), tmp_root)
    assert _in_tmp(source.resolve_reference_data_root(), tmp_root)


def test_agy_data_under_the_home_directory_resolves_into_tmp(tmp_root: Path) -> None:
    """Windows keeps the real `Path.home()` (its model caches), so agy's two
    directories are moved one by one."""

    from finesub.llm.agent import agy_transcript
    from finesub_bootstrap import agy_records

    assert _in_tmp(agy_transcript.agy_data_dir(), tmp_root)
    assert _in_tmp(agy_records.agy_project_records_dir(), tmp_root)


def test_agent_activity_leases_land_in_tmp(tmp_root: Path) -> None:
    from finesub.llm.agent import agent_paths

    location = agent_paths.resolve_agent_episode_location()

    assert _in_tmp(location.activity_root, tmp_root)
    assert _in_tmp(location.parent, tmp_root)


def test_the_leak_check_sees_checkout_state_and_test_urls(tmp_path: Path) -> None:
    """A guard nobody has seen fail is a guard nobody should trust."""

    root_conftest = Path(__file__).with_name("conftest.py")
    conftest = next(
        module
        for module in list(sys.modules.values())
        if getattr(module, "__file__", None) and Path(module.__file__) == root_conftest
    )
    checkout = tmp_path / "checkout"
    reference = checkout / "data" / "reference"
    reference.mkdir(parents=True)
    batch_root = tmp_path / "out" / "batch"
    url_info = reference / "url-info.json"
    url_info.write_text(
        json.dumps({"https://www.bilibili.com/video/BV1": {"title": "real"}}), encoding="utf-8"
    )
    before = conftest._checkout_leaks(batch_root, None, checkout)

    (checkout / ".state").write_text("{}", encoding="utf-8")
    (checkout / ".task-activity.lock").write_text("", encoding="utf-8")
    url_info.write_text(
        json.dumps(
            {
                "https://www.bilibili.com/video/BV1": {"title": "real"},
                "https://example.test/v?a=1": {"title": ""},
                "https://example.com/watch?v=1": {"title": ""},
            }
        ),
        encoding="utf-8",
    )
    leaked = conftest._checkout_leaks(batch_root, None, checkout) - before

    assert leaked == {
        str(checkout / ".state"),
        str(checkout / ".task-activity.lock"),
        f"{url_info} entry https://example.test/v?a=1",
        f"{url_info} entry https://example.com/watch?v=1",
    }

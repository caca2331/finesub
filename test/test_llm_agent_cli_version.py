"""The agent-CLI version pin: warns below it, never gates on it.

The pin is advisory by design (a gate would turn "not upgraded yet" into "no
target on this machine"), so the thing worth nailing down is that a stale CLI
still reports ready -- and that the check cannot switch itself off silently
when a vendor changes its `--version` wording.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from finesub.llm.agent import local_agent
from finesub.llm.agent.local_agent import (
    AgentDriverConfig,
    AgyDriverConfig,
    ClaudeCodeDriverConfig,
    CodexDriverConfig,
    DriverProbe,
    DshDriverConfig,
    CodeBuddyDriverConfig,
    driver_readiness,
)


@pytest.fixture(autouse=True)
def _fresh_warning_state():
    """Warnings are once-per-process, so each test needs a clean slate."""

    local_agent._READINESS_REPORTED.clear()
    yield
    local_agent._READINESS_REPORTED.clear()


def _driver(version: str, *, min_version: str = "1.1.24", ready: bool = True):
    probe = DriverProbe(available=True, version=version)
    return SimpleNamespace(
        driver_id="fake",
        display_name="Fake CLI",
        config=AgentDriverConfig(min_version=min_version),
        probe=lambda: probe,
        meets_requirements=lambda _probe, native_search=False: ready,
    )


# What each vendor actually printed on the owner's machine (2026-09-02;
# CodeBuddy 2026-09-04; dsh re-read after its upgrade, 2026-11-15; codex
# 2026-11-19). These
# are the formats the shared parser has to keep reading: a vendor that changes
# its wording turns the pin off, so the format itself is under test.
VENDOR_VERSION_LINES = {
    CodexDriverConfig: "codex-cli 0.171.0",
    ClaudeCodeDriverConfig: "2.1.231 (Claude Code)",
    AgyDriverConfig: "1.5.7",
    DshDriverConfig: "0.2.2-rc.1",
    # The only one with no vendor name at all, which is why the shared
    # parser has to accept a bare dotted number rather than a prefix.
    CodeBuddyDriverConfig: "2.178.1",
}


@pytest.mark.parametrize("config_cls,line", sorted(
    VENDOR_VERSION_LINES.items(), key=lambda item: item[0].__name__
))
def test_each_shipped_pin_matches_the_version_line_it_came_from(config_cls, line) -> None:
    """The pins were taken from these lines, so they must parse to them."""

    config = config_cls()
    assert config.min_version, f"{config_cls.__name__} ships no version pin"
    pinned = local_agent._cli_version_key(config.min_version)
    reported = local_agent._cli_version_key(line)
    assert pinned is not None and reported is not None
    assert not local_agent._cli_version_is_older(reported, pinned)


def test_a_stale_cli_warns_but_stays_ready(reported) -> None:
    ready, detail = driver_readiness(_driver("1.1.23"))

    assert ready is True and detail == ""
    assert reported.codes() == ["agent-cli-stale"]
    assert "1.1.23" in reported.joined() and "1.1.24" in reported.joined()


@pytest.mark.parametrize(
    "version,min_version",
    [
        ("1.1.24", "1.1.24"),  # the pinned version itself
        ("2.0.0", "1.1.24"),
        # "1.2" and "1.2.0" are the same release; padding is what makes them equal.
        ("1.2.0", "1.2"),
        # dsh pins prereleases (first `0.1.1-rc.2`, now `0.2.2-rc.1`); the
        # release that follows one must not read as older than it.
        ("0.1.1", "0.1.1-rc.2"),
        ("whatever", ""),  # a driver without a pin is never judged
    ],
    ids=["pinned", "newer", "trailing-zeros", "release-over-prerelease", "no-pin"],
)
def test_a_cli_at_or_past_its_pin_says_nothing(reported, version, min_version) -> None:
    assert driver_readiness(_driver(version, min_version=min_version)) == (True, "")
    assert reported.codes() == []


def test_an_earlier_prerelease_is_still_stale(reported) -> None:
    ready, _ = driver_readiness(_driver("0.1.1-rc.1", min_version="0.1.1-rc.2"))

    assert ready is True
    assert reported.codes() == ["agent-cli-stale"]


def test_an_unreadable_version_warns_rather_than_going_quiet(reported) -> None:
    """A guard that stops looking must say so; silence would read as green."""

    ready, detail = driver_readiness(_driver("build main@deadbeef"))

    assert ready is True and detail == ""
    assert reported.codes() == ["agent-cli-version-unreadable"]


def test_the_warning_is_emitted_once_per_process(reported) -> None:
    """A run builds a driver per client; without dedup every one would warn."""

    for _ in range(3):
        driver_readiness(_driver("1.1.23"))

    assert reported.codes() == ["agent-cli-stale"]


def test_an_unavailable_driver_is_not_told_its_version_is_old(reported) -> None:
    """The stale check sits behind the requirement check, so it stays advisory."""

    driver = _driver("1.1.23", ready=False)
    driver.config = replace(driver.config, min_version="1.1.24")
    ready, detail = driver_readiness(driver)

    assert ready is False
    assert reported.codes() == ["agent-cli-unusable"]
    assert "stale" not in detail

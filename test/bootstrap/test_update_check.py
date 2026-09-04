"""The published CLI's "a newer finesub is out" notice.

Every test here is about a way the notice could misbehave rather than about the
happy path, because the happy path is one string and the failure modes are the
whole reason the module exists.
"""

from __future__ import annotations

import json

import pytest

from finesub_bootstrap import update_check
from finesub_bootstrap.update_check import (
    CHECK_INTERVAL_SECONDS,
    UpdateCheck,
    enabled,
    fetch_latest,
    is_newer,
    is_stale,
    read_state,
    state_path,
    write_state,
)


def _seeded(root, *, latest: str, now: float) -> None:
    write_state(root, latest=latest, now=now)


# ---- what counts as an upgrade -------------------------------------------


@pytest.mark.parametrize(
    "latest, current, expected",
    [
        ("0.5.0", "0.4.2", True),
        ("0.4.2", "0.4.2", False),
        ("0.4.1", "0.4.2", False),
        ("1.0", "0.9.9", True),
        ("0.10.0", "0.9.0", True),  # not a string comparison
        # Only formal releases are advertised. This is the whole "don't point
        # people at the 0.5.0pre checkpoint" rule, and it needs no special case.
        ("0.5.0rc1", "0.4.2", False),
        ("0.5.0b2", "0.4.2", False),
        ("0.5.0.dev3", "0.4.2", False),
        # ...but going from a pre-release to its own final release is exactly
        # the upgrade an rc is waiting for.
        ("0.5.0", "0.5.0rc1", True),
        # `.postN` is a FORMAL release of the same version and sorts after it.
        # Dropping the post number made a hotfix read as "no news" -- which is
        # the announcement a user most needs to hear.
        ("0.5.0.post1", "0.5.0", True),
        ("0.5.0.post2", "0.5.0.post1", True),
        ("0.5.0", "0.5.0.post1", False),
        # Trailing zeros are not a version difference (PEP 440): comparing the
        # raw tuples made the longer spelling look newer and invented a notice.
        ("1.0.0", "1.0", False),
        ("1.0", "1.0.0", False),
        ("1.0.0.0", "1.0", False),
        # Garbage is never an upgrade.
        ("", "0.4.2", False),
        ("nonsense", "0.4.2", False),
        ("0.5.0", "", False),
    ],
)
def test_is_newer_only_advertises_formal_releases(latest, current, expected) -> None:
    assert is_newer(latest, current) is expected


# ---- when the check may speak at all --------------------------------------


def test_disabled_by_env_and_by_ci(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.DISABLE_ENV, raising=False)
    assert enabled(command="run", user_data=tmp_path, isatty=True)

    monkeypatch.setenv(update_check.DISABLE_ENV, "1")
    assert not enabled(command="run", user_data=tmp_path, isatty=True)

    # An explicit falsey value is not an opt-out -- `FINESUB_NO_UPDATE_CHECK=0`
    # reads as "no, do not disable".
    monkeypatch.setenv(update_check.DISABLE_ENV, "0")
    assert enabled(command="run", user_data=tmp_path, isatty=True)

    monkeypatch.delenv(update_check.DISABLE_ENV, raising=False)
    monkeypatch.setenv("CI", "true")
    assert not enabled(command="run", user_data=tmp_path, isatty=True)


def test_disabled_without_a_tty_and_on_the_quiet_commands(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.DISABLE_ENV, raising=False)

    # Piped output is somebody's script, not somebody reading.
    assert not enabled(command="run", user_data=tmp_path, isatty=False)
    for command in ("setup", "uninstall", "--help"):
        assert not enabled(command=command, user_data=tmp_path, isatty=True), command


def test_config_toml_can_turn_it_off(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.DISABLE_ENV, raising=False)
    config = tmp_path / "config.toml"

    config.write_text("[cli]\nupdate_check = false\n", encoding="utf-8")
    assert not enabled(command="run", user_data=tmp_path, isatty=True)

    config.write_text("[cli]\nupdate_check = true\n", encoding="utf-8")
    assert enabled(command="run", user_data=tmp_path, isatty=True)

    # A config nobody can parse must not decide anything.
    config.write_text("[cli\nbroken", encoding="utf-8")
    assert enabled(command="run", user_data=tmp_path, isatty=True)


def test_an_unreadable_opt_out_is_treated_as_opted_out(monkeypatch, tmp_path) -> None:
    """No TOML parser must not mean "they did not opt out".

    On Python 3.10 the stdlib has none. Reading the absence as "not disabled"
    made the CLI contact PyPI for users who had written
    `[cli] update_check = false` exactly as `docs/manual/resources.md` tells
    them to -- a documented setting, silently ignored, on the one axis where
    silence is worst. The wheel ships `tomli` for 3.10 so this path is a
    backstop, not the normal case.
    """

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(update_check.DISABLE_ENV, raising=False)
    (tmp_path / "config.toml").write_text(
        '[cli]\nupdate_check = false\n', encoding="utf-8"
    )

    monkeypatch.setattr(update_check, "_toml_parser", lambda: None)
    assert not enabled(command="run", user_data=tmp_path, isatty=True)

    # With no config file at all there is nothing to honour, parser or not.
    empty = tmp_path / "elsewhere"
    empty.mkdir()
    assert enabled(command="run", user_data=empty, isatty=True)


# ---- throttling ------------------------------------------------------------


def test_a_fresh_state_is_not_refetched(tmp_path) -> None:
    _seeded(tmp_path, latest="0.4.2", now=1_000.0)
    assert not is_stale(read_state(tmp_path), now=1_000.0 + CHECK_INTERVAL_SECONDS - 1)
    assert is_stale(read_state(tmp_path), now=1_000.0 + CHECK_INTERVAL_SECONDS)


def test_a_clock_that_moved_backwards_does_not_freeze_the_check(tmp_path) -> None:
    """A VM restore or a container with no clock would otherwise stall it.

    `checked_at` in the future makes `now - checked_at` negative, and a plain
    `>= interval` test reads that as "checked recently" -- forever, until the
    clock catches up.
    """

    _seeded(tmp_path, latest="0.4.2", now=9_999_999.0)
    assert is_stale(read_state(tmp_path), now=1_000.0)


def test_unreadable_or_missing_state_is_simply_stale(tmp_path) -> None:
    assert is_stale(read_state(tmp_path), now=1_000.0)
    state_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert read_state(tmp_path) == {}
    assert is_stale(read_state(tmp_path), now=1_000.0)


# ---- the run-level behaviour ----------------------------------------------


def test_the_first_run_seeds_state_and_says_nothing(tmp_path) -> None:
    """Being told about an upgrade by the thing you just installed reads as a
    bug. No state file means "this install has never checked", i.e. it was just
    installed."""

    check = UpdateCheck(
        tmp_path, current="0.4.2", fetcher=lambda: "0.5.0", clock=lambda: 1_000.0
    )
    check.start()

    assert check.notice() == ""
    # ...but the answer is now on disk, so the next run can use it.
    assert read_state(tmp_path)["latest"] == "0.5.0"


def test_a_later_run_reports_the_fetched_version(tmp_path) -> None:
    _seeded(tmp_path, latest="0.4.2", now=0.0)
    check = UpdateCheck(
        tmp_path, current="0.4.2", fetcher=lambda: "0.5.0", clock=lambda: 1_000_000.0
    )
    check.start()

    notice = check.notice()
    assert "0.5.0" in notice and "0.4.2" in notice
    assert update_check.UPGRADE_COMMAND in notice


def test_a_failed_fetch_is_silent_and_keeps_the_previous_answer(tmp_path) -> None:
    """No network is not an error the user has to hear about."""

    _seeded(tmp_path, latest="0.4.2", now=0.0)

    def boom() -> str:
        raise OSError("no network")

    check = UpdateCheck(
        tmp_path, current="0.4.2", fetcher=boom, clock=lambda: 1_000_000.0
    )
    check.start()

    assert check.notice() == ""
    assert read_state(tmp_path)["latest"] == "0.4.2"


def test_a_fresh_state_is_reported_without_any_fetch(tmp_path) -> None:
    """Within the interval the run is fully offline: cached answer, no thread."""

    _seeded(tmp_path, latest="0.5.0", now=1_000.0)

    def boom() -> str:
        raise AssertionError("must not fetch inside the interval")

    check = UpdateCheck(
        tmp_path, current="0.4.2", fetcher=boom, clock=lambda: 1_000.0
    )
    check.start()

    assert "0.5.0" in check.notice()


def test_an_unwritable_state_directory_never_raises(tmp_path) -> None:
    blocked = tmp_path / "file-not-a-dir"
    blocked.write_text("x", encoding="utf-8")

    write_state(blocked / "nested", latest="0.5.0", now=1.0)  # must not raise
    assert read_state(blocked / "nested") == {}


# ---- the fetch itself ------------------------------------------------------


def test_fetch_reads_info_version_and_rejects_a_junk_body(monkeypatch) -> None:
    bodies: list[bytes] = []

    class _Response:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self, _limit: int) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> bool:
            return False

    monkeypatch.setattr(
        update_check.urllib.request,
        "urlopen",
        lambda request, timeout=None: _Response(bodies.pop(0)),
    )

    bodies.append(json.dumps({"info": {"version": "0.5.0"}}).encode("utf-8"))
    assert fetch_latest() == "0.5.0"

    # Every shape of "that is not an answer" comes back as "no answer".
    for junk in (b"not json", b"[]", b'{"info": null}', b'{"info": {"version": ""}}'):
        bodies.append(junk)
        assert fetch_latest() == "", junk

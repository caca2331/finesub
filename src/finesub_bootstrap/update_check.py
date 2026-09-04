""""A newer finesub is out" -- the published CLI's own notice.

Deliberately modelled on what pip, `gh` and npm's update-notifier settled on,
because the failure modes here are all in the plumbing rather than the idea:

* **Never blocks a command.** The fetch runs on a daemon thread started before
  the command and read back after it, and a run that ends before the thread
  does simply uses the previous answer. Nothing is ever waited on for longer
  than :data:`JOIN_TIMEOUT_SECONDS`.
* **Never speaks on failure.** No network, no PyPI, a malformed body, an
  unwritable state file -- all of it is silence. A version notice that can
  print an error is worse than no version notice.
* **Never on the first run.** Being told about an upgrade by the thing you just
  installed reads as a bug. No state file means "seed it and say nothing".
* **stderr, TTY only, and never in CI.** stdout carries pipeline output.
* **Notifies, never upgrades.** The command to run is printed; running it is
  the user's call.

Only **formal** releases are reported. PyPI's ``info.version`` already skips
pre-releases, and :func:`is_newer` refuses anything that is not a plain dotted
release on top of that -- so a checkpoint tag published as ``0.5.0rc1`` (or as a
GitHub prerelease) is invisible here by construction, with no special case.

stdlib-only, and importable on **Python 3.10**: the published CLI's shell runs
on whatever interpreter the user installed it with, and this module is imported
before every command it dispatches.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Optional
import urllib.error
import urllib.request

#: PyPI over GitHub Releases on purpose: one unauthenticated request with no
#: 60/hour/IP budget to share, and `info.version` is *defined* as the latest
#: non-prerelease -- the channel split we would otherwise have to implement.
PYPI_JSON_URL = "https://pypi.org/pypi/finesub/json"
STATE_FILENAME = "update-check.json"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
FETCH_TIMEOUT_SECONDS = 3.0
#: How long the caller may wait for an in-flight fetch once its command is
#: done. The thread has had the whole command to finish; this is the tail.
JOIN_TIMEOUT_SECONDS = 0.2
DISABLE_ENV = "FINESUB_NO_UPDATE_CHECK"
UPGRADE_COMMAND = "uv tool upgrade finesub"

#: Commands that must never print it: two are about leaving, and one is a help
#: screen whose whole value is being short.
QUIET_COMMANDS = frozenset({"setup", "uninstall", "help", "-h", "--help"})

_RELEASE_RE = re.compile(r"^(\d+(?:\.\d+)*)(.*)$")
_POST_RE = re.compile(r"^[._-]?post(\d+)$")


@dataclass(frozen=True)
class Release:
    release: tuple[int, ...]
    prerelease: bool
    #: `N` of a `.postN` suffix; 0 for a plain release. A post-release is a
    #: *formal* release of the same version, and it sorts after it -- dropping
    #: it made `0.5.0.post1` look like "no news" beside `0.5.0`, i.e. exactly
    #: the hotfix a user most needs to hear about.
    post: int = 0

    @property
    def key(self) -> tuple[tuple[int, ...], int]:
        """What to order by. Trailing zeros are stripped first: PEP 440 says
        `1.0` and `1.0.0` are the same version, and comparing the raw tuples
        made the longer spelling look newer."""

        release = list(self.release)
        while len(release) > 1 and release[-1] == 0:
            release.pop()
        return tuple(release), self.post


def parse_version(value: str) -> Optional[Release]:
    """PEP 440 as far as this needs it, without pulling in `packaging`.

    Everything that is not a plain dotted release counts as a pre-release, with
    one exception: ``.postN`` is a *formal* release of the same version, so it
    is kept and ordered after the release it patches. That exception is not
    cosmetic -- without it a `0.5.0.post1` hotfix reads as "no news" beside
    `0.5.0`, which is the announcement a user most needs.

    Everything else erring towards pre-release is deliberate: the only decision
    riding on it is "may we advertise this", and erring towards *no* costs a
    user one delayed notice, while erring the other way advertises a checkpoint.
    """

    match = _RELEASE_RE.match(str(value).strip())
    if match is None:
        return None
    try:
        release = tuple(int(part) for part in match.group(1).split("."))
    except ValueError:  # pragma: no cover - the regex already forbids this
        return None
    rest = match.group(2)
    post_match = _POST_RE.match(rest)
    if post_match is not None:
        return Release(release, False, int(post_match.group(1)))
    return Release(release, bool(rest))


def is_newer(latest: str, current: str) -> bool:
    """Is `latest` a formal release worth telling the user about?

    Equal releases still count when the *installed* one is a pre-release: going
    from ``0.5.0rc1`` to ``0.5.0`` is exactly the upgrade an rc is waiting for.
    """

    new = parse_version(latest)
    now = parse_version(current)
    if new is None or now is None or new.prerelease:
        return False
    if new.key != now.key:
        return new.key > now.key
    # Same version by every ordering rule: only a pre-release installed against
    # its own final release is still an upgrade.
    return now.prerelease


# ---- state ----------------------------------------------------------------


def state_path(data_root: Path) -> Path:
    return Path(data_root) / STATE_FILENAME


def read_state(data_root: Path) -> dict:
    try:
        loaded = json.loads(state_path(data_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_state(data_root: Path, *, latest: str, now: float) -> None:
    path = state_path(data_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"latest": latest, "checked_at": now}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass  # a cache nobody can write is a cache that misses, not an error


def is_stale(state: dict, *, now: float) -> bool:
    """Should this run refresh the cached answer?

    "No `checked_at`" is answered up front rather than by defaulting it to 0:
    with a real unix timestamp the arithmetic happens to say "stale", but with
    any small `now` it says the opposite -- an absent state would read as fresh
    and no refresh would ever start.
    """

    if "checked_at" not in state:
        return True
    try:
        checked_at = float(state["checked_at"])
    except (TypeError, ValueError):
        return True
    # A clock that moved backwards (a VM restore, a container with no clock)
    # would otherwise freeze the check until real time caught up.
    return not (0.0 <= now - checked_at < CHECK_INTERVAL_SECONDS)


# ---- policy ---------------------------------------------------------------


def _env_says_no() -> bool:
    value = os.environ.get(DISABLE_ENV, "").strip().lower()
    if value and value not in {"0", "false", "no", "off"}:
        return True
    # Every CI provider sets this, and no CI run wants an upgrade notice.
    return bool(os.environ.get("CI", "").strip())


def _toml_parser():
    """`tomllib.load`, or the `tomli` backport, or None.

    Imported here rather than at module scope: `tomllib` is 3.11+, and this
    module is imported before *every* command the published CLI runs -- whose
    shell supports Python 3.10 (`cli/pyproject.toml`). A top-level import turned
    "no update notice" into `ModuleNotFoundError` for `setup`, transcription and
    uninstall alike. The wheel declares `tomli` for 3.10 so the setting keeps
    working there; None is the belt-and-braces path if even that is absent.
    """

    try:
        from tomllib import load
    except ImportError:
        try:
            from tomli import load  # type: ignore[no-redef]
        except ImportError:
            return None
    return load


def _config_says_no(user_data: Path | None) -> bool:
    """`[cli] update_check = false` in the shared config.toml.

    Read with `tomllib` rather than the main package's resolver: this module is
    imported by the thin launcher, which has no `finesub` on its path yet.
    """

    if user_data is None:
        return False
    config = Path(user_data) / "config.toml"
    if not config.is_file():
        return False

    parser = _toml_parser()
    if parser is None:
        # No parser, so we cannot tell whether this user wrote
        # `[cli] update_check = false`. Treat that as "disabled": the setting is
        # a promise in `docs/manual/resources.md`, and the failure we refuse
        # here is contacting PyPI for somebody who asked us not to. The cost of
        # being wrong the other way is one missing notice.
        return True
    try:
        with config.open("rb") as handle:
            data = parser(handle)
    except (OSError, ValueError):
        # A config nobody can parse states nothing, including an opt-out.
        return False
    section = data.get("cli")
    if not isinstance(section, dict):
        return False
    return section.get("update_check") is False


def enabled(*, command: str, user_data: Path | None, isatty: bool) -> bool:
    if not isatty or command in QUIET_COMMANDS:
        return False
    return not _env_says_no() and not _config_says_no(user_data)


# ---- fetch ----------------------------------------------------------------


def fetch_latest(*, url: str = PYPI_JSON_URL, timeout: float = FETCH_TIMEOUT_SECONDS) -> str:
    """The newest formal version on PyPI, or `""` for any kind of no-answer."""

    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(1 << 20)
        payload = json.loads(body.decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return ""
    # `.get` straight off the decode would raise on a JSON list -- a body shape
    # nobody expects is still a body shape that must not crash a run.
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        return ""
    info = payload["info"]
    version = str(info.get("version") or "").strip()
    return version if parse_version(version) is not None else ""


class UpdateCheck:
    """One run's worth of "is there a newer finesub".

    Two calls, around the command: :meth:`start` before it and :meth:`notice`
    after. Both are total -- neither raises, whatever the network, the disk or
    the clock do.
    """

    def __init__(
        self,
        data_root: Path,
        *,
        current: str,
        fetcher=None,
        clock=time.time,
    ) -> None:
        self.data_root = Path(data_root)
        self.current = str(current)
        # Resolved at call time, not bound here as a default argument: a
        # default captures the function at class-definition time, so patching
        # `update_check.fetch_latest` in a test would silently keep hitting the
        # real PyPI -- which is exactly how this line got written twice.
        self._fetcher = fetcher
        self._clock = clock
        self._thread: threading.Thread | None = None
        # No state file means this install has never checked -- so it was just
        # installed, and the one thing not to do is greet it with an upgrade
        # notice. Recorded before the refresh, which is about to create one.
        self._had_state = bool(read_state(self.data_root))

    def start(self) -> None:
        state = read_state(self.data_root)
        if not is_stale(state, now=self._clock()):
            return
        self._thread = threading.Thread(target=self._refresh, daemon=True)
        self._thread.start()

    def _refresh(self) -> None:
        fetcher = self._fetcher if self._fetcher is not None else fetch_latest
        try:
            latest = fetcher()
        except Exception:  # noqa: BLE001 - a background nicety never raises
            return
        if latest:
            write_state(self.data_root, latest=latest, now=self._clock())

    def notice(self) -> str:
        """The line to show, or `""`. Waits at most `JOIN_TIMEOUT_SECONDS`."""

        if self._thread is not None:
            self._thread.join(JOIN_TIMEOUT_SECONDS)
        if not self._had_state:
            return ""
        latest = str(read_state(self.data_root).get("latest") or "")
        if not latest or not is_newer(latest, self.current):
            return ""
        return (
            f"finesub {latest} is available (you have {self.current}). "
            f"Upgrade with: {UPGRADE_COMMAND}"
        )

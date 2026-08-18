"""Repo-wide pytest setup, shared by every suite in this repository.

One thing lives here, and it is about pytest rather than about this project.

`tmp_path` and `tmp_path_factory.mktemp` both go through pytest's
`make_numbered_dir`, which hangs a `<prefix>current` symlink off the newest
temporary directory -- "best effort linking to the latest test run", in its own
words, for a human poking around the temp root. Nothing reads it.

On Windows without Developer Mode or an elevated shell, the `os.symlink` behind
it cannot succeed: it raises `WinError 1314` (no such privilege), pytest catches
it and carries on -- and the *failed* call costs about 0.2s, measured, against
0.3ms for the directory it is decorating. `test/conftest.py` has two autouse
fixtures that each take a temporary directory, so every test in the suite paid
that twice before its first line ran, and every test that also asks for
`tmp_path` paid it a third time. The bill grew with the number of tests rather
than with the work they do, which is why the suite kept getting slower while CI
-- Linux, where the symlink simply works -- noticed nothing.

Measured here: the full suite went from 38 minutes to 1m31s (`-n 2`, 1890
passing) with this file in place and nothing else changed.

Probed rather than assumed, so this stays a decision about the machine and not
about the platform: where symlinks work, pytest keeps the behaviour it
documents; where they cannot, we skip a system call whose only possible outcome
is a slow failure. Enabling Developer Mode on a Windows machine therefore turns
the convenience link back on by itself.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import _pytest.pathlib


def _symlinks_are_available() -> bool:
    """Whether this process may create a directory symlink at all."""

    with tempfile.TemporaryDirectory() as probe:
        root = Path(probe)
        target = root / "target"
        target.mkdir()
        try:
            (root / "link").symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            return False
    return True


# Guarded rather than asserted: `_force_symlink` is private pytest API, and a
# rename in a future release should cost the speed-up, not the ability to run
# the suite at all.
if hasattr(_pytest.pathlib, "_force_symlink") and not _symlinks_are_available():
    _pytest.pathlib._force_symlink = lambda root, target, link_to: None

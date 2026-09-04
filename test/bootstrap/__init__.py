"""Tests for `finesub_bootstrap`, the provisioning layer the CLI stands on.

Paths, the task index, downloads, migrations and the managed runtime are what
a CLI run stands on, so a change to `fsops.py` fails the pre-commit `pytest -q`
here rather than a suite nobody runs locally.

Two files are really executed only on Windows -- `test_fsops_links.py`
(junctions, robocopy) and `test_secrets.py`'s DPAPI cases in the root suite --
and skip elsewhere; the Windows job in `ci.yml` runs them by name so a skip on
the Linux runner never stands in for an execution.
"""

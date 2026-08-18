"""Tests for `finesub_bootstrap`, the provisioning layer both front ends share.

They live here rather than in the desktop suite because what they test is not
the desktop: paths, the task index, downloads, migrations and the managed
runtime are what a CLI run stands on too, and a change to `fsops.py` should
fail the pre-commit `pytest -q` rather than a suite nobody runs locally.

What stays in `desktop/backend/tests` is what only Windows can execute --
junctions, robocopy, DPAPI -- where moving here would turn a real execution
into a permanent skip on the Linux runner.
"""

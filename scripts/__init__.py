"""Repository-level build and release scripts.

A package only so `python -m scripts.make_cn_lock` works and its tests can
import it by name. The two `.ps1` entry points beside it are not part of it --
the naming convention is `.ps1` with hyphens for shell entry points, `.py` with
underscores for importable modules (see `README_DEV.md`).
"""

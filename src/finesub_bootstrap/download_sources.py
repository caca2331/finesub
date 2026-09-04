"""The public entry points a release is willing to download from.

Kept in a shipped JSON file rather than in Python so that adding a mirror is a
data change reviewed as data, and so the table someone audits is the table the
code reads.

Every list starts empty. That is deliberate rather than unfinished: the plan
requires each candidate to survive a Windows install drill -- a full runtime
install, a prefetch of all three models, a broken-hash recovery and a timing
comparison against the official source -- before it becomes a default. An
empty table means `auto` resolves to `global`, which is the official source,
so the mechanism can ship and be tested while no unverified host is ever
contacted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SOURCES_NAME = "download-sources.json"

#: Points at an alternative table, for the release drill itself: the operator
#: running it needs to exercise a candidate before it is anywhere near a
#: default.
SOURCES_ENVIRONMENT = "FINESUB_DOWNLOAD_SOURCES"

_EMPTY: dict[str, Any] = {
    "schemaVersion": 1,
    # [{"name": ..., "url": ...}] -- asked in order, briefly, for a country.
    "countryEndpoints": [],
    # PyPI index URL used when the region is cn.
    "pypiIndex": "",
    # Where the torch wheels come from in a generated cn lock. Read at build
    # time by scripts/make_cn_lock.py, not at runtime: the lock has the
    # URL baked in by then. Separate from `pypiIndex` because the two are not
    # the same host -- TUNA serves PyPI but not the pytorch-wheels layout.
    "torchMirror": "",
    # Hugging Face endpoint used when the region is cn.
    "hfEndpoint": "",
    # Prefix applied to a fixed GitHub release URL when the region is cn.
    "githubFileProxy": "",
}


def sources_path() -> Path:
    override = os.environ.get(SOURCES_ENVIRONMENT, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / SOURCES_NAME


def load_sources() -> dict[str, Any]:
    """The table, or an empty one -- never an exception.

    A malformed or missing file must degrade to "no mirrors" rather than stop
    a download that the official source can serve perfectly well.
    """

    try:
        body = json.loads(sources_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_EMPTY)
    if not isinstance(body, dict):
        return dict(_EMPTY)
    return {**_EMPTY, **body}


def mirror_for(resource_class: str, region: str) -> str:
    """The cn entry point for a resource class, or "" for the official source.

    The environment always wins, including when it is set to empty: that is
    how someone turns one class of acceleration off without touching the rest.
    """

    from finesub_bootstrap.download_routes import OVERRIDES

    variable = OVERRIDES.get(resource_class)
    if variable is not None and variable in os.environ:
        return os.environ[variable].strip()
    if region != "cn":
        return ""
    key = {
        "pypi": "pypiIndex",
        "huggingface": "hfEndpoint",
        "github": "githubFileProxy",
    }.get(resource_class)
    return str(load_sources().get(key, "")) if key else ""

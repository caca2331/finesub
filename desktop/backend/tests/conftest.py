from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def managed_data_root(tmp_path_factory, monkeypatch) -> None:
    """Keep personal data out of the developer's real `%LOCALAPPDATA%`.

    `AppPaths` now derives the data root from the environment, so without this
    a test that builds paths would resolve to -- and eventually write into --
    the machine's own FineSub installation.
    """

    monkeypatch.setenv(
        "LOCALAPPDATA", str(tmp_path_factory.mktemp("LocalAppData"))
    )
    # The developer machine may opt out of .env protection globally
    # (FINESUB_ENV_PROTECT=0, a transition hatch); tests need the default.
    monkeypatch.delenv("FINESUB_ENV_PROTECT", raising=False)
    # The shipped source table names real country endpoints, so anything that
    # resolves a download region would reach the network -- and write a cache
    # file doing it. Forcing the region keeps the suite offline; the tests that
    # are *about* resolution set this themselves.
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "global")

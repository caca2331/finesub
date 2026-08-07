import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def managed_data_root(tmp_path_factory, monkeypatch) -> None:
    """Keep personal data out of the developer's real `%LOCALAPPDATA%`.

    Personal-data paths derive from the environment now, and this suite drives
    `uninstall` -- without the redirect a test run would delete the machine's
    own FineSub data.
    """

    monkeypatch.setenv(
        "LOCALAPPDATA", str(tmp_path_factory.mktemp("LocalAppData"))
    )
    # The developer machine may opt out of .env protection globally
    # (FINESUB_ENV_PROTECT=0, a transition hatch); tests need the default.
    monkeypatch.delenv("FINESUB_ENV_PROTECT", raising=False)

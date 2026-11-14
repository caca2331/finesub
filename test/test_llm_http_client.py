"""Every remote API call -- the LLM layer's and the cloud ASR backends' -- goes out through one client factory.

The setting this protects is `[api] proxy` (old name `[llm] proxy`), and its whole value is being
answerable: "does this request use the proxy?" has to be readable off the code
rather than inferred from whatever `HTTPS_PROXY` happened to be set to.

The guard below is the reason that stays true. Writing it found a fifth call
site nobody had listed -- the Gemini resumable *upload* in `client.py` -- which
is precisely the failure mode an explicit factory has and an ambient
environment variable does not: forgetting one is silent either way, but only
one of them can be checked.
"""

from __future__ import annotations

import ast
from pathlib import Path

import httpx
import pytest

from finesub.api_http import DIRECT, client_kwargs
from finesub.llm.http import llm_http_client

SRC = Path(__file__).resolve().parents[1] / "src" / "finesub"
LLM_ROOT = SRC / "llm"
#: The packages whose every client must come from `finesub.api_http`, the only module allowed to construct one
#: (it is outside them). Not a list of call sites -- those move; this is the seam.
GUARDED = (LLM_ROOT, SRC / "speech" / "cloud")
#: The LLM layer's seam over it, which names the class in its type annotations.
SEAM = LLM_ROOT / "http.py"


def _httpx_client_mentions(path: Path) -> list[int]:
    """Lines naming `httpx.Client` / `httpx.AsyncClient` in `path`.

    Mentions, not calls. Four of the five call sites reach the network through
    an injectable `client_factory`, and a default of `client_factory=httpx.Client`
    bypasses the factory exactly as completely as a direct construction while
    reading as if it did not. `isinstance(exc, httpx.TimeoutException)` and
    `httpx.Timeout(...)` stay legal -- only the client class is forbidden.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"Client", "AsyncClient"}
        and isinstance(node.value, ast.Name)
        and node.value.id == "httpx"
    ]


def test_only_the_factory_names_an_http_client() -> None:
    """A sixth call site added later must fail here, not go quietly unproxied."""

    offenders: dict[str, list[int]] = {}
    for root in GUARDED:
        for path in sorted(root.rglob("*.py")):
            if path == SEAM:
                continue
            lines = _httpx_client_mentions(path)
            if lines:
                offenders[str(path.relative_to(SRC))] = lines

    assert offenders == {}, (
        "build API clients with `finesub.api_http.api_http_client` (the LLM layer through "
        f"`llm_http_client`) so the `[api] proxy` setting reaches them: {offenders}"
    )


def test_every_injectable_call_site_defaults_to_the_factory() -> None:
    """The three `client_factory` seams must default to `llm_http_client`.

    The AST guard forbids naming `httpx.Client`; it cannot check that what
    replaced it is the *right* thing. A factory that quietly became something
    else would leave those call sites unproxied with the guard still green.
    """

    import inspect

    from finesub.llm.media_upload import _upload_gemini_file_rest
    from finesub.llm.token_budget import GeminiCountTokensCounter
    from finesub.llm.web_search import WebSearchClient

    for owner in (
        WebSearchClient.__init__,
        GeminiCountTokensCounter,
        _upload_gemini_file_rest,
    ):
        factory = inspect.signature(owner).parameters["client_factory"].default
        assert factory is llm_http_client, owner


# ---- what the setting resolves to -----------------------------------------


@pytest.fixture(autouse=True)
def _forget_config():
    from finesub import config as config_module

    config_module.clear_config_cache()
    yield
    config_module.clear_config_cache()


def _config(tmp_path: Path, monkeypatch, body: str) -> None:
    from finesub import config as config_module

    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    config_module.clear_config_cache()


def test_no_setting_keeps_todays_behaviour(tmp_path, monkeypatch) -> None:
    """Absent means `HTTPS_PROXY` still decides, so adding the key broke nobody."""

    _config(tmp_path, monkeypatch, "[api]\n")
    assert client_kwargs() == {"trust_env": True}


def test_a_configured_proxy_wins_outright_over_the_environment(
    tmp_path, monkeypatch
) -> None:
    """The file is the more specific statement, so it is not merged with env."""

    _config(tmp_path, monkeypatch, '[api]\nproxy = "http://127.0.0.1:7890"\n')
    assert client_kwargs() == {
        "proxy": "http://127.0.0.1:7890",
        "trust_env": False,
    }


def test_direct_is_how_you_say_no_proxy_on_a_proxied_machine(
    tmp_path, monkeypatch
) -> None:
    """Without it, "unset the key" would still leave `HTTPS_PROXY` in charge."""

    _config(tmp_path, monkeypatch, f'[api]\nproxy = "{DIRECT}"\n')
    assert client_kwargs() == {"proxy": None, "trust_env": False}


def test_an_edited_proxy_reaches_the_next_client(tmp_path, monkeypatch) -> None:
    """No memo in between: config.toml edits take effect on the next read,
    here as everywhere else. It used to be pinned for the life of the process."""

    _config(tmp_path, monkeypatch, '[api]\nproxy = "http://127.0.0.1:7890"\n')
    assert client_kwargs()["proxy"] == "http://127.0.0.1:7890"

    (tmp_path / "config.toml").write_text(f'[api]\nproxy = "{DIRECT}"\n', encoding="utf-8")
    assert client_kwargs() == {"proxy": None, "trust_env": False}


def test_a_non_string_proxy_is_rejected_rather_than_coerced(
    tmp_path, monkeypatch
) -> None:
    _config(tmp_path, monkeypatch, "[api]\nproxy = 7890\n")
    with pytest.raises(ValueError, match="api.proxy must be a string"):
        client_kwargs()


def test_the_new_name_wins_and_the_old_one_is_still_read(tmp_path, monkeypatch) -> None:
    """Owner 2026-10-30: `[llm] proxy` became `[api] proxy` when the cloud ASR backends started reading it;
    a config that still says `[llm] proxy` keeps working."""

    _config(tmp_path, monkeypatch, '[llm]\nproxy = "http://127.0.0.1:7890"\n')
    assert client_kwargs()["proxy"] == "http://127.0.0.1:7890"
    (tmp_path / "config.toml").write_text(
        f'[api]\nproxy = "{DIRECT}"\n[llm]\nproxy = "http://127.0.0.1:7890"\n', encoding="utf-8")
    assert client_kwargs() == {"proxy": None, "trust_env": False}


def test_the_factory_builds_a_client_with_the_resolved_route(
    tmp_path, monkeypatch
) -> None:
    _config(tmp_path, monkeypatch, f'[api]\nproxy = "{DIRECT}"\n')
    with llm_http_client(timeout=httpx.Timeout(1.0)) as client:
        assert isinstance(client, httpx.Client)
        assert client.timeout.connect == 1.0

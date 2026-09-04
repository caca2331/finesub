"""Every LLM-layer API call goes out through one client factory.

The setting this protects is `[llm] proxy`, and its whole value is being
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

from finesub.llm import http as llm_http
from finesub.llm.http import DIRECT, client_kwargs, llm_http_client

LLM_ROOT = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm"

#: The only module allowed to construct one. Not a list of call sites -- those
#: move; this is the seam.
FACTORY_MODULE = "http.py"


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
    for path in sorted(LLM_ROOT.rglob("*.py")):
        if path.name == FACTORY_MODULE:
            continue
        lines = _httpx_client_mentions(path)
        if lines:
            offenders[str(path.relative_to(LLM_ROOT))] = lines

    assert offenders == {}, (
        "build LLM-layer clients with `finesub.llm.http.llm_http_client` so the "
        f"`[llm] proxy` setting reaches them: {offenders}"
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
    llm_http.reset_proxy_cache()
    yield
    llm_http.reset_proxy_cache()


def _config(tmp_path: Path, monkeypatch, body: str) -> None:
    from finesub import config as config_module

    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    config_module.clear_config_cache()


def test_no_setting_keeps_todays_behaviour(tmp_path, monkeypatch) -> None:
    """Absent means `HTTPS_PROXY` still decides, so adding the key broke nobody."""

    _config(tmp_path, monkeypatch, "[llm]\n")
    assert client_kwargs() == {"trust_env": True}


def test_a_configured_proxy_wins_outright_over_the_environment(
    tmp_path, monkeypatch
) -> None:
    """The file is the more specific statement, so it is not merged with env."""

    _config(tmp_path, monkeypatch, '[llm]\nproxy = "http://127.0.0.1:7890"\n')
    assert client_kwargs() == {
        "proxy": "http://127.0.0.1:7890",
        "trust_env": False,
    }


def test_direct_is_how_you_say_no_proxy_on_a_proxied_machine(
    tmp_path, monkeypatch
) -> None:
    """Without it, "unset the key" would still leave `HTTPS_PROXY` in charge."""

    _config(tmp_path, monkeypatch, f'[llm]\nproxy = "{DIRECT}"\n')
    assert client_kwargs() == {"proxy": None, "trust_env": False}


def test_a_non_string_proxy_is_rejected_rather_than_coerced(
    tmp_path, monkeypatch
) -> None:
    _config(tmp_path, monkeypatch, "[llm]\nproxy = 7890\n")
    with pytest.raises(ValueError, match="llm.proxy must be a string"):
        client_kwargs()


def test_the_factory_builds_a_client_with_the_resolved_route(
    tmp_path, monkeypatch
) -> None:
    _config(tmp_path, monkeypatch, f'[llm]\nproxy = "{DIRECT}"\n')
    with llm_http_client(timeout=httpx.Timeout(1.0)) as client:
        assert isinstance(client, httpx.Client)
        assert client.timeout.connect == 1.0

"""Region resolution: forced, cached, probed, and failing safe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub_bootstrap import download_routes, download_sources
from finesub_bootstrap.download_routes import RouteDecision


@pytest.fixture
def unforced_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the suite-wide forced region.

    conftest pins it so no test reaches a real country endpoint by accident.
    These tests are about the resolution itself, so they need it back -- and
    they supply their own probe rather than calling out.
    """

    monkeypatch.delenv("FINESUB_DOWNLOAD_REGION", raising=False)


def test_an_explicit_region_beats_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    download_routes.remember_region(
        tmp_path, RouteDecision(region="global", source="probe")
    )

    def probe():
        raise AssertionError("a forced region must not probe")

    decision = download_routes.resolve_region(tmp_path, probe=probe)

    assert decision.region == "cn"
    assert decision.source == "forced"


def test_nonsense_in_the_environment_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "mars")

    decision = download_routes.resolve_region(
        tmp_path, probe=lambda: RouteDecision(region="cn", source="probe")
    )

    assert decision.region == "cn"


def test_a_recent_answer_is_reused(tmp_path: Path, unforced_region) -> None:
    download_routes.remember_region(
        tmp_path, RouteDecision(region="cn", source="probe", endpoint="primary"), now=1000.0
    )
    probed = []

    decision = download_routes.resolve_region(
        tmp_path, probe=lambda: probed.append(1), now=1000.0 + 60
    )

    assert probed == []
    assert (decision.region, decision.source) == ("cn", "cached")


def test_a_stale_answer_is_probed_again(tmp_path: Path, unforced_region) -> None:
    download_routes.remember_region(
        tmp_path, RouteDecision(region="cn", source="probe"), now=1000.0
    )

    decision = download_routes.resolve_region(
        tmp_path,
        probe=lambda: RouteDecision(region="global", source="probe"),
        now=1000.0 + download_routes.CACHE_TTL_SEC + 1,
    )

    assert decision.region == "global"


def test_a_probe_that_fails_means_the_official_source(
    tmp_path: Path, unforced_region
) -> None:
    decision = download_routes.resolve_region(tmp_path, probe=lambda: None)

    assert (decision.region, decision.source) == ("global", "default")
    # Nothing is cached, so the next run gets to try again.
    assert download_routes.cached_region(tmp_path) is None


def test_no_address_is_ever_written_down(tmp_path: Path) -> None:
    download_routes.remember_region(
        tmp_path,
        RouteDecision(region="cn", source="probe", endpoint="primary"),
        now=1000.0,
    )

    body = json.loads(
        download_routes.state_path(tmp_path).read_text(encoding="utf-8")
    )
    assert set(body["region"]) == {"region", "decidedAt", "endpoint"}


def test_the_state_file_sits_with_the_small_personal_data(tmp_path: Path) -> None:
    """The big-data root may not have been chosen yet -- that is the download
    this decision exists for."""

    assert download_routes.state_path(tmp_path).parent == tmp_path


def test_an_unwritable_state_directory_does_not_stop_a_run(
    tmp_path: Path, unforced_region
) -> None:
    blocked = tmp_path / "file-not-a-dir"
    blocked.write_text("", encoding="utf-8")

    decision = download_routes.resolve_region(
        blocked, probe=lambda: RouteDecision(region="cn", source="probe")
    )

    assert decision.region == "cn"


def test_repeated_failures_turn_one_class_off_for_this_machine(
    tmp_path: Path,
) -> None:
    for _ in range(download_routes.FAILURE_LIMIT):
        download_routes.record_failure(tmp_path, "huggingface")

    assert download_routes.is_degraded(tmp_path, "huggingface")
    # Only that class: a broken HF mirror says nothing about a PyPI one.
    assert not download_routes.is_degraded(tmp_path, "pypi")


def test_one_place_weighs_every_reason_not_to_use_a_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each caller asking these questions itself is how a new resource class
    ends up honouring three of the four."""

    table = tmp_path / "sources.json"
    table.write_text(
        json.dumps({"hfEndpoint": "https://mirror.example"}), encoding="utf-8"
    )
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(table))

    assert (
        download_routes.active_mirror(tmp_path, "huggingface", "cn")
        == "https://mirror.example"
    )
    # Wrong region.
    assert download_routes.active_mirror(tmp_path, "huggingface", "global") == ""
    # Nothing configured for this class.
    assert download_routes.active_mirror(tmp_path, "pypi", "cn") == ""
    # Given up on by this machine.
    for _ in range(download_routes.FAILURE_LIMIT):
        download_routes.record_failure(tmp_path, "huggingface")
    assert download_routes.active_mirror(tmp_path, "huggingface", "cn") == ""


def test_a_success_clears_the_counter(tmp_path: Path) -> None:
    download_routes.record_failure(tmp_path, "pypi")
    download_routes.record_success(tmp_path, "pypi")

    assert download_routes.failures(tmp_path, "pypi") == 0


# -- the shipped source table ---------------------------------------------


def test_every_shipped_host_is_https() -> None:
    """A mirror is not a new root of trust, but it is still a host we chose.

    Plain HTTP would let anyone on the path decide what a machine downloads,
    and the digests only cover the files that carry one.
    """

    sources = download_sources.load_sources()
    urls = [entry["url"] for entry in sources["countryEndpoints"]]
    urls += [
        sources[key]
        for key in ("pypiIndex", "torchMirror", "hfEndpoint", "githubFileProxy")
        if sources[key]
    ]

    assert urls, "the table is expected to be populated"
    for url in urls:
        assert url.startswith("https://"), url


def test_the_github_proxy_can_be_concatenated_with_an_absolute_url() -> None:
    """It is applied as a prefix, so a missing slash silently mangles the URL."""

    proxy = download_sources.load_sources()["githubFileProxy"]

    assert not proxy or proxy.endswith("/")


def test_a_country_endpoint_is_asked_for_a_country_and_nothing_else() -> None:
    """These are the only hosts told anything about the user's connection."""

    for entry in download_sources.load_sources()["countryEndpoints"]:
        assert set(entry) == {"name", "url"}, entry


def test_a_missing_table_degrades_to_the_official_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(tmp_path / "absent.json"))

    assert download_sources.load_sources()["pypiIndex"] == ""
    assert download_sources.mirror_for("pypi", "cn") == ""


def test_a_corrupt_table_degrades_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(broken))

    assert download_sources.load_sources()["hfEndpoint"] == ""


def test_the_environment_can_point_a_class_somewhere_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "sources.json"
    table.write_text(
        json.dumps({"hfEndpoint": "https://mirror.example"}), encoding="utf-8"
    )
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(table))

    assert download_sources.mirror_for("huggingface", "cn") == "https://mirror.example"

    monkeypatch.setenv("FINESUB_HF_ENDPOINT", "https://chosen.example")
    assert download_sources.mirror_for("huggingface", "cn") == "https://chosen.example"


def test_an_empty_override_disables_that_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "sources.json"
    table.write_text(
        json.dumps({"hfEndpoint": "https://mirror.example"}), encoding="utf-8"
    )
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(table))
    monkeypatch.setenv("FINESUB_HF_ENDPOINT", "")

    assert download_sources.mirror_for("huggingface", "cn") == ""


def test_the_global_region_uses_no_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "sources.json"
    table.write_text(
        json.dumps({"pypiIndex": "https://mirror.example"}), encoding="utf-8"
    )
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(table))

    assert download_sources.mirror_for("pypi", "global") == ""


# -- probing --------------------------------------------------------------


def test_a_two_letter_body_is_read_as_a_country() -> None:
    class _Response:
        text = "CN\n"

    assert download_routes._country_from(_Response()) == "CN"


def test_a_json_body_is_read_as_a_country() -> None:
    class _Response:
        text = '{"country_code": "us"}'

        def json(self):
            return {"country_code": "us"}

    assert download_routes._country_from(_Response()) == "US"


def test_an_unreadable_body_yields_nothing() -> None:
    class _Response:
        text = "<html>go away</html>"

    assert download_routes._country_from(_Response()) is None


def test_no_endpoints_means_no_probe() -> None:
    assert download_routes.probe_region(endpoints=[]) is None


def test_the_diagnostic_never_prints_credentials(tmp_path: Path, monkeypatch) -> None:
    """`doctor` output gets pasted into issues and screen shares, and an index
    override is an ordinary place to carry a token."""

    from finesub_bootstrap.shell import _safe_host

    assert (
        _safe_host("https://alice:s3cret@pypi.example.com/simple?token=abc#x")
        == "https://pypi.example.com"
    )
    assert _safe_host("https://mirror.example:8443/pypi/web/simple") == (
        "https://mirror.example:8443"
    )
    assert _safe_host("") == ""
    assert _safe_host("not a url") == "(unparsable)"


def test_the_diagnostic_reports_without_reaching_out(tmp_path: Path) -> None:
    """`doctor` describes configuration and cache; it does not make a call."""

    from finesub_bootstrap.environment import RuntimeEnvironment
    from finesub_bootstrap.paths import AppPaths
    from finesub_bootstrap.resources import ResourceManager
    from finesub_bootstrap.shell import Shell

    probes: list[int] = []
    original = download_routes.probe_region

    def counting_probe(*args, **kwargs):
        probes.append(1)
        return original(*args, **kwargs)

    paths = AppPaths.for_root(tmp_path / "root", data_root=tmp_path / "data")
    shell = Shell(
        paths=paths,
        resources=ResourceManager(paths, []),
        runtime=RuntimeEnvironment(
            paths=paths,
            app_source=tmp_path,
            runtime_lock=tmp_path / "lock.toml",
            uv_executable=lambda: tmp_path / "uv.exe",
        ),
    )
    download_routes.probe_region = counting_probe
    try:
        report = shell._download_route_report()
    finally:
        download_routes.probe_region = original

    assert probes == []
    assert report.startswith("global")

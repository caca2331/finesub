"""Transport-level contracts shared by every bootstrap consumer."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetSource(StrictModel):
    """The one thing every asset has: somewhere to fetch it from."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("asset URL must use http or https")
        return value


class DownloadAsset(AssetSource):
    size: int
    sha256: str

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("asset size must not be negative")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        return normalized


class ResolvableAsset(AssetSource):
    """An asset whose bytes move, so its digest is asked for instead of pinned.

    Some upstreams publish under a stable URL and replace the bytes behind it on
    every build: BtbN's ffmpeg `latest` tag is rebuilt daily, release branches
    included. Against one of those, a pinned `size`/`sha256` fails within a day,
    and it fails expensively -- after the whole download, quarantining the
    result. Pinning nothing instead would mean installing a 90 MB executable
    with no integrity check.

    So the digest is fetched from the host at install time and the downloaded
    bytes are verified against it (`asset_resolve.resolve_asset`). What that
    keeps is integrity and resume safety; what it gives up is reproducibility --
    two machines installing a day apart get different builds, and neither is the
    one we tested. That trade is only acceptable for a tool with a tiny, stable
    interface. Anything whose behaviour we depend on in detail stays pinned;
    `test_runtime_manifest_pins_every_asset_it_can` is what keeps that honest.

    Full reasoning: `docs/cli-bootstrap-logging-download-plan.md` 5.6.
    """

    digest_from: Literal["github-release-api"]


class DownloadProgress(StrictModel):
    downloaded: int
    total: int
    bytes_per_second: float


class ResourceSpec(StrictModel):
    id: str
    version: str
    destination: Literal["runtime", "models"]
    directory: str
    archive_type: Literal["zip", "file"]
    required_files: list[str]
    # `extra="forbid"` is what discriminates the two: {url,size,sha256} can only
    # be a DownloadAsset and {url,digest_from} can only be a ResolvableAsset, so
    # a half-filled entry is a validation error rather than a silent choice.
    asset: DownloadAsset | ResolvableAsset

    @field_validator("id", "version", "directory")
    @classmethod
    def validate_path_component(cls, value: str) -> str:
        normalized = value.strip()
        if (
            not normalized
            or normalized in {".", ".."}
            or "/" in normalized
            or "\\" in normalized
            or ":" in normalized
        ):
            raise ValueError("resource path components must be simple names")
        return normalized

    @field_validator("required_files")
    @classmethod
    def validate_required_files(cls, values: list[str]) -> list[str]:
        for value in values:
            path = value.replace("\\", "/")
            if (
                not path
                or path.startswith("/")
                or ":" in path.split("/", 1)[0]
                or ".." in path.split("/")
            ):
                raise ValueError("required files must stay inside the resource")
        return values


class ResourceStatus(StrictModel):
    id: str
    version: str
    # "outdated" is installed and usable, just not the version the manifest now
    # names. It is deliberately not "missing": collapsing the two meant that
    # bumping a tool -- yt-dlp needs it regularly -- read as "you never
    # installed this" and, once these tools became required, stopped every
    # user's task until they fetched it. An upgrade should offer, not gate.
    state: Literal["missing", "downloading", "outdated", "ready", "failed"]
    #: What is on disk when `state` is "outdated"; `version` is what it should be.
    installed_version: str = ""
    detail: str = ""
    # On-demand tools (git, yt-dlp) are listed so the user can reach them, but
    # they must not read as "your install is incomplete": only a task that needs
    # one is blocked by it. Consumers exclude these from readiness counts and
    # from the "space required" total.
    optional: bool = False
    # A usable system Python was found, so provisioning only has to install
    # FineSub's AI dependencies. The UI keys an affordance off this; it used
    # to sniff `detail` for a Chinese sentence written in environment.py, so
    # rewording that message -- or translating it -- silently removed the
    # affordance with nothing to catch it.
    reuses_system_python: bool = False
    # Another resource has to be installed before this one can be. Set for the
    # model weights, which are fetched by the managed interpreter and so cannot
    # start before it exists. Structured for the same reason as the flag above:
    # the UI has to disable the button, and reading intent out of `detail` is
    # how that quietly stops working.
    blocked_by: str = ""

    @property
    def usable(self) -> bool:
        """Whether a task can run with what is on disk right now.

        The question every gate should ask. `state == "ready"` is the wrong
        one: it says "and it is the newest version", which is a different
        claim and not one a task cares about.
        """

        return self.state in {"ready", "outdated"}

"""Transport-level contracts shared by every bootstrap consumer."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DownloadAsset(StrictModel):
    url: str
    size: int
    sha256: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("asset URL must use http or https")
        return value

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
    asset: DownloadAsset

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
    state: Literal["missing", "downloading", "ready", "failed"]
    detail: str = ""
    # On-demand tools (git, yt-dlp) are listed so the user can reach them, but
    # they must not read as "your install is incomplete": only a task that needs
    # one is blocked by it. Consumers exclude these from readiness counts and
    # from the "space required" total.
    optional: bool = False

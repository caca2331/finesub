from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


PipelineStage = Literal[
    "vocal",
    "aligned",
    "stable",
    "raw-srt",
    "translated-srt",
    "final-srt",
]


class DesktopModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BridgeError(DesktopModel):
    code: str
    message: str
    action: str | None = None


class CapabilityState(DesktopModel):
    raw_srt: bool = True
    translation: bool = False
    web_search: bool = False


class PublicSettings(DesktopModel):
    api_keys: dict[str, Literal["configured", "missing"]]


class DownloadAsset(DesktopModel):
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


class DownloadProgress(DesktopModel):
    downloaded: int
    total: int
    bytes_per_second: float


class ResourceSpec(DesktopModel):
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


class ResourceStatus(DesktopModel):
    id: str
    version: str
    state: Literal["missing", "downloading", "ready", "failed"]
    detail: str = ""


class ResourceInstallSnapshot(DesktopModel):
    resource_id: str
    resource_version: str
    state: Literal["queued", "running", "paused", "ready", "failed"]
    phase: Literal[
        "waiting",
        "downloading",
        "verifying",
        "extracting",
        "installing_python",
        "creating_environment",
        "installing_dependencies",
        "activating",
        "complete",
    ] = "waiting"
    message: str = ""
    downloaded: int = 0
    total: int = 0
    bytes_per_second: float = 0
    cache_path: str
    install_path: str
    logs: list[str] = Field(default_factory=list)
    error: str = ""
    started_at: float
    updated_at: float


class TaskRequest(DesktopModel):
    input: str
    output: str | None = None
    stage: PipelineStage = "raw-srt"
    model_name: str = "large-v3-turbo"
    device: Literal["cuda", "cpu"] = "cuda"
    language: str | None = None
    gpu_budget_gb: Literal[4, 8, 12, 16] = 4
    word: bool = False
    asr_stabilize_profile: Literal[-1, 0, 1, 2] = 0
    llm_route: Literal["text", "mm"] = "mm"
    llm_level: Literal["low", "med", "high"] = "med"
    llm_fast: Literal["auto", "on", "off"] = "auto"
    llm_output_scale: float = 1.0
    extra_info: str = ""
    extra_style: str = ""
    enable_web_search: bool = True
    knowledge: Literal["none", "collect", "update"] = "none"
    postprocess_profile: Literal[-1, 0, 1, 2] = 0

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("input must not be blank")
        return normalized

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

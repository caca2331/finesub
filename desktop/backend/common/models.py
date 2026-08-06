from __future__ import annotations

from typing import Literal

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


class UpdateInstallSnapshot(DesktopModel):
    version: str
    kind: Literal["app", "full"]
    state: Literal["queued", "running", "ready", "failed"]
    phase: Literal["waiting", "downloading", "installing", "complete"] = "waiting"
    message: str = ""
    downloaded: int = 0
    total: int = 0
    bytes_per_second: float = 0
    # An "app" update swaps the version pointer, so the running launcher keeps
    # its process and only needs a restart. A "full" update hands control to an
    # external updater that replaces this install, so FineSub has to exit for it
    # to proceed -- a different ask of the user, hence two flags rather than one.
    restart_required: bool = False
    exit_required: bool = False
    error: str = ""
    started_at: float
    updated_at: float


class TaskRequest(DesktopModel):
    input: str
    output: str | None = None
    # The CLI's --name: a bare stem that becomes out/<name>/<name>.srt. It names
    # a directory, so a separator would escape the tree -- hence the validator.
    # Blank keeps the derived name (source filename or video id).
    name: str = ""
    # Off by default: the run directory is what makes a rerun cheap (the
    # pipeline skips stages whose outputs exist) and what a later LLM pass reads.
    cleanup_intermediate: bool = False
    stage: PipelineStage = "raw-srt"
    model_name: str = "large-v3-turbo"
    device: Literal["cuda", "cpu"] = "cuda"
    language: str | None = None
    gpu_budget_gb: Literal[4, 8, 12, 16] = 4
    word: bool = False
    asr_stabilize_profile: Literal[-1, 0, 1, 2] = 0
    llm_route: Literal["text", "mm"] = "mm"
    llm_level: Literal["low", "med", "high"] = "high"
    llm_fast: Literal["auto", "on", "off"] = "auto"
    llm_output_scale: float = 1.0
    extra_info: str = ""
    extra_style: str = ""
    enable_web_search: bool = True
    # Default on: the knowledge base is what makes later tasks better, and it
    # only runs when the LLM stage does -- a plain transcription ignores it.
    knowledge: Literal["none", "collect", "update"] = "update"
    postprocess_profile: Literal[-1, 0, 1, 2, 3, 4] = 0

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("input must not be blank")
        return normalized

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return ""
        if "/" in normalized or "\\" in normalized or normalized in {".", ".."}:
            raise ValueError("输出名称不能包含路径分隔符")
        return normalized

    @field_validator("language", mode="before")
    @classmethod
    def normalize_language(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

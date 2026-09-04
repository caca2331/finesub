from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
)


PipelineStage = Literal[
    "vocal",
    "aligned",
    "stable",
    "raw-srt",
    "translated-srt",
    "final-srt",
]

#: What class of GPU this machine has -- NOT a cap on what a run may use.
#: `auto` asks the driver, which is the default and almost always right; the
#: named tiers exist for "leave some of the card to something else". The names
#: mirror `finesub.speech.runtime.resources.GPU_TIERS`, which owns what each
#: one means.
GpuTier = Literal[
    "auto", "cpu", "entry", "standard", "standard_large_vram", "high"
]

#: Renamed 2026-08-12 (docs/llm_harness_behavior.md). Task history is shared
#: with the CLI, merged and never truncated, so every record written before
#: that rename still carries the old word -- and history is read back through
#: these models with unreadable entries *silently skipped* (jobs/history.py),
#: so rejecting them would empty a user's task list rather than fail loudly.
#: Accepted on input only; nothing writes the retired words again, so this
#: mapping can go once no such record can still be on disk.
_RETIRED_DIFFICULTY = {
    "high": "quality",
    "med": "intermediate",
    "minimum": "efficiency",
}


def _accept_retired_difficulty(value: object) -> object:
    return _RETIRED_DIFFICULTY.get(value, value) if isinstance(value, str) else value


#: The LLM layer's vocabulary (`finesub.llm.routing.profiles.DIFFICULTY`). The
#: worker passes this value straight through, so anything else raises there --
#: after ASR has already run, which is the most expensive place to find out.
#: Carrying the coercion on the alias means every model that uses it accepts
#: old history without each one remembering to add a validator.
LLMDifficulty = Annotated[
    Literal["quality", "intermediate", "efficiency"],
    BeforeValidator(_accept_retired_difficulty),
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


class TaskDefaults(BaseModel):
    """What the task form should start with, i.e. "what I picked last time".

    A subset of ``TaskRequest``, every field optional: absent means follow the
    code default, which is what lets a default still be improved later. Unknown
    fields are ignored rather than rejected -- this is program-written state,
    and a key left over from another version is not something a user can fix.

    ``split_length_scale`` is deliberately absent: it is a shared setting that
    the CLI reads too, so it lives in ``config.toml``. One key, one home.
    """

    model_config = ConfigDict(extra="ignore")

    model_name: str | None = None
    device: Literal["cuda", "cpu"] | None = None
    gpu_index: int | None = None
    gpu_name: str | None = None
    language: str | None = None
    gpu_tier: GpuTier | None = None
    word: bool | None = None
    asr_stabilize_profile: Literal[-1, 0, 1, 2] | None = None
    stage: PipelineStage | None = None
    llm_media: Literal["text", "audio", "video"] | None = None
    llm_retrieval: Literal["none", "local", "native"] | None = None
    llm_difficulty: LLMDifficulty | None = None
    llm_fast: Literal["auto", "on", "off"] | None = None
    llm_output_scale: float | None = None
    knowledge: Literal["none", "collect", "update"] | None = None
    postprocess_profile: Literal[-1, 0, 1, 2, 3, 4] | None = None
    cleanup_intermediate: bool | None = None

    @model_serializer
    def _sparse(self) -> dict[str, object]:
        """Serialize the chosen fields only -- never an unset one as null.

        The front end spreads this object straight over the task request, so a
        `model_name: null` on the wire does not mean "unset", it *overwrites* a
        real default and the request stops validating. Doing it here rather
        than at each dump site is the point: every path out of the process
        (bootstrap payload, get_preferences, the file itself) is sparse by
        construction and no caller can forget the flag.
        """

        return {key: value for key, value in self.__dict__.items() if value is not None}


class Preferences(DesktopModel):
    """The front end's own persisted state.

    ``ui`` is stored verbatim: it is the renderer's business (theme, language,
    dismissed dialogs), and validating every new toggle here would be a schema
    treadmill for values the backend never interprets. ``task_defaults`` is
    validated, because it ends up as pipeline arguments.
    """

    ui: dict[str, object] = Field(default_factory=dict)
    task_defaults: TaskDefaults = Field(default_factory=TaskDefaults)


class SharedSettings(DesktopModel):
    """The half of the settings panel that writes ``config.toml``.

    ``None`` means "not set" -- the key is removed from the file and the code
    default applies again, which is also what keeps the file readable by hand.
    """

    split_length_scale: float | None = None


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
    # The full transcript on disk; `logs` above is only its tail.
    log_path: str = ""
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
    # `None` = "not chosen", which is a different statement from "cuda" and has
    # to stay expressible: `--gpu-tier cpu` plus an explicit `--device cuda` is
    # a contradiction the CLI refuses, and a front end that always sends a
    # device would silently obey one half of it instead. The backend signature
    # supplies the default (README_DEV -> 开发原则: no second copy of it here).
    device: Literal["cuda", "cpu"] | None = None
    # Which CUDA card to use, when the machine has more than one. None lets
    # CUDA pick, which is what every single-GPU machine wants.
    gpu_index: int | None = None
    # The card that index meant when it was chosen. Indexes are positions, not
    # identities: swap the card in that slot and the index still resolves, so
    # without the name nothing would notice the task running on hardware the
    # user never picked.
    gpu_name: str = ""
    language: str | None = None
    # `auto` = ask the card. A tier names what CLASS of card this is, not a
    # cap on what a run may use, so the backend resolves it from the driver.
    gpu_tier: GpuTier = "auto"
    word: bool = False
    asr_stabilize_profile: Literal[-1, 0, 1, 2] = 0
# None = follow config.toml, then the calibrated default. Set per task only
    # to override the shared setting for this one run.
    split_length_scale: float | None = None
    # Orthogonal switch axes (docs/llm_harness_behavior.md); the old
    # llm_route/llm_level presets and enable_web_search retired with them.
    llm_media: Literal["text", "audio", "video"] = "audio"
    llm_retrieval: Literal["none", "local", "native"] = "local"
    llm_difficulty: LLMDifficulty = "quality"
    llm_fast: Literal["auto", "on", "off"] = "auto"
    llm_output_scale: float = 1.0
    extra_info: str = ""
    extra_style: str = ""
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

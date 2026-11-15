"""Every scalar setting in ``config.toml``: declared once, resolved one way.

A setting is one line here and one ``.resolve()`` where it is used::

    separate = settings.SEPARATE.resolve(separate)   # explicit -> config -> default

What that call answers is the same for every key, so no consumer writes it
again: an explicit value (a flag, a manifest row, a caller) wins over
``config.toml``, which wins over the default; the *chosen* value is checked
against the declaration, explicit ones included; an explicit value that wins
means the file is not read for that key at all; and ``False``, ``0`` and ``""``
stay distinct from "not written". Normalisation -- stripping, case, what an
empty string means, membership in a table only the domain knows -- is the
domain's, not this module's.

Declared here rather than beside each consumer so that the whole registry can
be imported cheaply (the documentation guard and the unknown-key warning need
all of it, and consumers pull in torch or the harness). Purely standard
library plus ``finesub.config`` / ``finesub.reporting``: ``finesub.speech``
reads this and may not grow an ``llm`` dependency.

Not here, on purpose: ``[providers]`` / ``[pools]`` (``finesub.api_keys``
validates their structure), the presets and model groups under ``[llm]``
(``llm.routing.model_routes``), and ``[cli] update_check`` (read by
``finesub_bootstrap``, which may not import this package).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from . import config as app_config
from .reporting import current_reporter, redact_credentials, reporter_delivers


class Loaded(NamedTuple):
    """One read of ``config.toml``: the parsed document and where it came from."""

    data: Mapping[str, Any]
    path: Path | None


def read(path: str | Path | None = None) -> Loaded:
    """Read the file once, for a caller that resolves several keys from it.

    Optional: ``.resolve()`` reads on its own. Passing one ``Loaded`` to several
    ``.resolve(config=...)`` calls just guarantees they saw the same version of
    the file.
    """

    data, config_path = app_config.read_config_with_path(path)
    _warn_unknown_keys(data, config_path)
    return Loaded(data, config_path)


_KIND_WORDING = {
    bool: "true or false",
    int: "an integer",
    float: "a number",
    str: "a string",
}


@dataclass(frozen=True)
class Setting:
    section: str
    key: str
    #: Given, not inferred: a default of ``None`` says nothing about the type.
    kind: type
    #: The one copy. Code that needs the default by name reads ``.default``.
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    #: The document that explains this key. Under ``docs/manual/`` means it is
    #: meant for users (the documentation guard then also wants it in
    #: ``config.example.toml``).
    doc: str = ""

    @property
    def name(self) -> str:
        return f"{self.section}.{self.key}"

    def check(self, value: Any, *, origin: str = "") -> Any:
        """``value`` if it is a valid value of this setting, else ``ValueError``.

        ``origin`` finishes the error message ("in <file>", "(explicit value)").
        Values are shown redacted: a URL here may carry ``user:token@``.
        """

        def fail(requirement: str) -> ValueError:
            shown = redact_credentials(repr(value))
            return ValueError(f"{self.name} must be {requirement} (got {shown}){origin}")

        kind = self.kind
        # `bool` is an int subclass, and `limit = true` is a typo, not a 1.
        if kind is bool:
            if not isinstance(value, bool):
                raise fail(_KIND_WORDING[bool])
            return value
        if kind is str:
            if not isinstance(value, str):
                raise fail(_KIND_WORDING[str])
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise fail(_KIND_WORDING[kind])
        if kind is int:
            if not isinstance(value, int):
                raise fail(_KIND_WORDING[int])
        else:
            value = float(value)
            if not math.isfinite(value):
                raise fail("a finite number")
        low, high = self.minimum, self.maximum
        if low is not None and high is not None and not low <= value <= high:
            raise fail(f"within [{low}, {high}]")
        if low is not None and value < low:
            raise fail(f"at least {low}")
        if high is not None and value > high:
            raise fail(f"at most {high}")
        return value

    def configured(
        self, *, config: Loaded | None = None, path: str | Path | None = None
    ) -> Any:
        """What ``config.toml`` says for this key, checked; ``None`` when absent.

        Absent is what keeps the file sparse and the defaults changeable for
        everyone who never wrote one down -- so ``""`` is *not* absent.
        """

        loaded = config if config is not None else read(path)
        table = loaded.data.get(self.section)
        if table is None:
            return None
        where = f" in {loaded.path}" if loaded.path else ""
        if not isinstance(table, Mapping):
            raise ValueError(f"[{self.section}] must be a TOML table{where}")
        if self.key not in table:
            return None
        return self.check(table[self.key], origin=where)

    def resolve(
        self,
        explicit: Any = None,
        *,
        config: Loaded | None = None,
        path: str | Path | None = None,
    ) -> Any:
        """Explicit value, else ``config.toml``, else the default."""

        if explicit is not None:
            return self.check(
                explicit, origin=" (explicit value: a command-line flag, a manifest row or a caller)"
            )
        value = self.configured(config=config, path=path)
        return self.default if value is None else value


# ------------------------------------------------------------------ recognition

#: `--asr`: `local`, or a cloud backend (`finesub.speech.joint.CHOICES`, aliases included).
ASR_BACKEND = Setting("asr", "backend", str, default="local", doc="docs/manual/joint-asr.md")
#: `--asr-aligner`: the forced aligner for MOSS Pro's long segments (heavy, optional).
ASR_ALIGNER = Setting("asr", "aligner", bool, default=False, doc="docs/manual/joint-asr.md")
#: `--asr-repair`: what a cloud transcript's repair may spend (`speech.joint.repair.MODES`).
ASR_REPAIR = Setting("asr", "repair", str, default="on", doc="docs/manual/joint-asr.md")
#: MOSS Pro: the pool's last key goes on past its free hours, billed to its prepaid balance (owner 2026-11-09).
ASR_MOSS_PRO_PAID = Setting("asr", "moss_pro_paid", bool, default=True, doc="docs/manual/joint-asr.md")
SPLIT_LENGTH_SCALE = Setting(
    "segmentation", "length_scale", float, default=1.0,
    minimum=0.6, maximum=1.6, doc="docs/manual/tuning.md",
)
VAD_SILERO_ASSIST = Setting(
    "vad", "silero_assist", bool, default=True, doc="docs/manual/tuning.md",
)
SEPARATE = Setting(
    "separator", "enabled", bool, default=True, doc="docs/manual/tuning.md",
)
#: Deliberately dev-only: no front end offers it (`test_option_defaults.py`).
VETO_LEVEL_FLOOR = Setting(
    "stabilize", "veto_level_floor", bool, default=False, doc="docs/asr-stabilize.md",
)

# ------------------------------------------------------------------ rendering

END_EXTEND = Setting(
    "postprocess", "end_extend_seconds", float, default=0.3,
    minimum=0.0, doc="docs/manual/tuning.md",
)
FLASH_GAP = Setting(
    "postprocess", "flash_gap_seconds", float, default=0.3,
    minimum=0.0, doc="docs/manual/tuning.md",
)
#: `--empty-cues`: a line with no text is a `""` cue in the rendered SRT instead of left out (owner 2026-11-07).
EMPTY_CUES = Setting("postprocess", "empty_cues", bool, default=False, doc="docs/manual/tuning.md")

# ------------------------------------------------------------------ LLM layer

#: Absent means "use the planning limits' own cap" -- the domain's fallback.
MAX_WINDOW_SUBTITLE_TOKENS = Setting(
    "chunking", "max_window_subtitle_tokens", int, default=None,
    minimum=0, doc="docs/llm_harness_behavior.md",
)
#: Absent means "use the planning limits' own cap", as above.
RESEARCH_TRANSCRIPT_MAX_TOKENS = Setting(
    "chunking", "research_transcript_max_tokens", int, default=None,
    minimum=0, doc="docs/llm_harness_research.md",
)
#: The proxy of every remote API call, the LLM layer's and the cloud ASR backends' (`finesub.api_http`).
PROXY = Setting("api", "proxy", str, default=None, doc="docs/manual/model-routing.md")
#: Its old name, from when only the LLM layer read it; still read when `[api] proxy` is absent (owner
#: 2026-10-30: the rename keeps the old spelling working).
LEGACY_LLM_PROXY = Setting("llm", "proxy", str, default=None, doc="docs/manual/model-routing.md")
STYLE = Setting("llm", "style", str, default=None, doc="docs/manual/knowledge.md")
STYLE_MODE = Setting("llm", "style_mode", str, default="read", doc="docs/manual/knowledge.md")
EXECUTION_POLICY = Setting(
    "llm", "execution_policy", str, default="agent-text-preferred",
    doc="docs/manual/model-routing.md",
)
#: 28 minutes: how long a local agent may spend on one call. No ceiling -- how
#: long is too long is exactly what only the person running it can know.
LOCAL_AGENT_TIMEOUT_SECONDS = Setting(
    "llm", "local_agent_timeout_seconds", int, default=1680,
    minimum=10, doc="docs/manual/agent.md",
)
#: This and the next are compatibility overrides for old agent CLIs, documented
#: for developers rather than in the manual.
LOCAL_AGENT_ALLOW_UNISOLATED_USER_CONFIG = Setting(
    "llm", "local_agent_allow_unisolated_user_config", bool, default=False,
    doc="docs/llm_harness_routing.md",
)
#: Debugging: keep every agent episode (capsule) and pseudo-session assignment
#: tree instead of deleting the successful ones and rolling the rest; Codex
#: then also keeps its own rollout. Nothing bounds the disk this takes.
LOCAL_AGENT_KEEP_EPISODES = Setting(
    "llm", "local_agent_keep_episodes", bool, default=False,
    doc="docs/llm_local_agent_runtime.md",
)
#: A single-task tool session ends its CLI the moment the runtime accepts the
#: submit, instead of letting it spend one more model request on "submitted".
#: `auto` | `on` | `off`. `auto` ends the CLI on accept only for drivers that
#: can rebuild the call's usage from per-request events (owner 2026-09-30:
#: Codex reports usage once at the end, so it stays off unless `on`).
LOCAL_AGENT_EXIT_ON_ACCEPT = Setting(
    "llm", "local_agent_exit_on_accept", str, default="auto",
    doc="docs/llm_local_agent_runtime.md",
)
LOCAL_AGENT_SERVICE_TIER = Setting(
    "llm", "local_agent_service_tier", str, default="", doc="docs/llm_harness_routing.md",
)
LOCAL_AGENT_REASONING_EFFORT = Setting(
    "llm", "local_agent_reasoning_effort", str, default="", doc="docs/manual/agent.md",
)
LOCAL_AGENT_MAX_PARALLEL = Setting(
    "llm", "local_agent_max_parallel", int, default=4,
    minimum=1, doc="docs/manual/agent.md",
)
#: The two tiers of the per-window retry budget (docs/llm_harness_behavior.md,
#: 「重试与拼接」): repairs inside one session chain, then fresh-session
#: replacements once a chain is spent.
MAX_RETRIES_PER_WINDOW = Setting(
    "llm", "max_retries_per_window", int, default=3,
    minimum=0, doc="docs/manual/tuning.md",
)
MAX_REPLACEMENTS_PER_WINDOW = Setting(
    "llm", "max_replacements_per_window", int, default=2,
    minimum=0, doc="docs/manual/tuning.md",
)
API_TIMEOUT_OVERHEAD_SECONDS = Setting(
    "llm", "api_timeout_overhead_seconds", float, default=180.0,
    minimum=0.0, doc="docs/llm_harness_behavior.md",
)
API_TIMEOUT_FLOOR_TOKENS_PER_SECOND = Setting(
    "llm", "api_timeout_floor_tokens_per_second", float, default=25.0,
    minimum=1.0, doc="docs/llm_harness_behavior.md",
)
API_SOFT_TIMEOUT_FRACTION = Setting(
    "llm", "api_soft_timeout_fraction", float, default=0.5,
    minimum=0.01, maximum=1.0, doc="docs/llm_harness_behavior.md",
)
SHARE_REMOTE = Setting("share", "remote", str, default=None, doc="docs/manual/knowledge.md")
SHARE_SIGNING_KEY = Setting("share", "signing_key", str, default=None, doc="docs/manual/share-server.md")

#: Every declaration above, in one place.
REGISTRY: tuple[Setting, ...] = tuple(
    value for value in list(globals().values()) if isinstance(value, Setting)
)

# ------------------------------------------------------------------ unknown keys

#: Keys of a registry table that another module reads, so they are not typos
#: there. `[llm]` also holds what `llm.routing.model_routes` reads (the active
#: preset and its tables; `providers` / `models` only to say they moved). Named
#: here rather than imported: this module may not depend on the harness.
#: `test_settings` checks the list against what model_routes reads.
_READ_ELSEWHERE: dict[str, frozenset[str]] = {
    "llm": frozenset(
        {"preset", "presets", "model_groups", "preferred_targets", "providers", "models"}
    ),
}
#: Read only to say they moved (model_routes raises on them): known, so not
#: warned about twice, but not offered in the "Known:" list either.
_MOVED: dict[str, frozenset[str]] = {"llm": frozenset({"providers", "models"})}
_DECLARED: dict[str, frozenset[str]] = {
    section: frozenset(s.key for s in REGISTRY if s.section == section)
    | _READ_ELSEWHERE.get(section, frozenset())
    for section in {s.section for s in REGISTRY}
}
_warned: set[tuple[str, str, str]] = set()


def _warn_unknown_keys(data: Mapping[str, Any], path: Path | None) -> None:
    """A key nobody reads, in a table the registry owns, is almost always a
    typo -- and a typo there silently does nothing. That includes a stray
    sub-table such as `[llm.bindings]` (bindings live under a preset).

    A warning, not an error: an older version reading a newer file must still
    run. Once per process and key, and only marked as said once it reached
    somebody (a worker thread without a reporter would swallow it).
    """

    for section, declared in _DECLARED.items():
        table = data.get(section)
        if not isinstance(table, Mapping):
            continue
        for key in table:
            if key in declared:
                continue
            marker = (str(path), section, key)
            if marker in _warned:
                continue
            reporter = current_reporter()
            reporter.warning(
                "config-unknown-key",
                f"config.toml [{section}] has no setting named {key!r}; it is "
                f"ignored. Known: {', '.join(sorted(declared - _MOVED.get(section, frozenset())))}"
                + (f" ({path})" if path else ""),
            )
            if reporter_delivers(reporter):
                _warned.add(marker)

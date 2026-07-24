"""Token counting and budget checks for LLM subtitle correction."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import httpx

from .config import DEFAULT_LIMITS, GEMINI_31_FLASH_LITE, ModelLimits
from .profiles import DEFAULT_PROFILE, TranslationProfile, expected_output_tokens


class TokenBudgetError(ValueError):
    """Raised when a request cannot fit within configured token limits."""


class TokenCounter(Protocol):
    source: str

    def count_text(self, text: str) -> int:
        ...

    def count_texts(self, texts: Iterable[str]) -> int:
        ...

    def count_audio_seconds(self, seconds: float) -> int:
        ...


@dataclass
class GeminiCountTokensCounter:
    """Token counter backed by the Gemini countTokens API.

    Text counting always calls the API (free, consumes no generation quota) with
    a per-text sha cache; failures raise instead of silently degrading — the old
    local heuristic underestimated CJK-heavy prompts by 25-40% and was removed.
    Audio is counted locally at the official 32 tok/s rate (the API cannot count
    audio without uploading it).
    """

    model: str = GEMINI_31_FLASH_LITE
    api_key: str | None = None
    api_version: str = "v1beta"
    client_factory: Callable[..., Any] = httpx.Client
    timeout_seconds: float = 60.0
    audio_tokens_per_second: int = DEFAULT_LIMITS.audio_tokens_per_second
    source: str = "gemini-countTokens"

    def __post_init__(self) -> None:
        self._cache: dict[str, int] = {}

    @property
    def model_name(self) -> str:
        return self.model.split("/", 1)[1] if self.model.startswith("gemini/") else self.model

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._cache:
            return self._cache[digest]
        count = self._count_parts_api([{"text": text}])
        self._cache[digest] = count
        return count

    def count_texts(self, texts: Iterable[str]) -> int:
        parts = [{"text": text} for text in texts if text]
        if not parts:
            return 0
        return self._count_parts_api(parts)

    def count_audio_seconds(self, seconds: float) -> int:
        if seconds <= 0:
            return 0
        return int(math.ceil(seconds * self.audio_tokens_per_second))

    def _api_key(self) -> str:
        if self.api_key:
            return self.api_key
        from .client import _first_gemini_api_key

        return _first_gemini_api_key()

    def _count_parts_api(self, parts: list[Mapping[str, str]]) -> int:
        api_key = self._api_key()
        url = (
            f"https://generativelanguage.googleapis.com/{self.api_version}/models/"
            f"{self.model_name}:countTokens"
        )
        payload = {"contents": [{"role": "user", "parts": list(parts)}]}
        with self.client_factory(timeout=self.timeout_seconds) as client:
            response = client.post(url, json=payload, headers={"x-goog-api-key": api_key})
            if response.status_code >= 400:
                body = response.text[:500].replace(api_key, "[redacted]")
                raise RuntimeError(f"Gemini countTokens failed with HTTP {response.status_code}: {body}")
            data = response.json()
        return int(data["totalTokens"])


# The gemini-token-counter Go binary runs the SDK's local tokenizer, which the
# API wraps in a `contents` envelope worth exactly one extra token. Verified
# constant across ASCII/CJK/emoji/empty-ish inputs and 1..1000-token lengths,
# so the local count is corrected by this offset to match countTokens exactly.
LOCAL_COUNTER_API_OFFSET = 1
# The local tokenizer ships vocabularies for a fixed model set; gemini-3.x is
# not among them, but the 2.5 vocabulary matches 3.1-flash-lite to within the
# constant offset above (that is what the verification measured).
LOCAL_TOKENIZER_MODEL = "gemini-2.5-flash"


def _local_counter_exe_is_runnable(path: Path) -> bool:
    """True when ``path`` is a binary we can execute on this host."""

    if not path.is_file():
        return False
    if path.suffix.lower() == ".exe":
        return os.name == "nt"
    return os.access(path, os.X_OK)


def _resolve_local_counter_exe() -> str | None:
    """Locate the bundled gemini-token-counter binary, if present."""

    override = os.environ.get("GEMINI_TOKEN_COUNTER_EXE")
    if override:
        candidate = Path(override)
        return str(candidate) if _local_counter_exe_is_runnable(candidate) else None
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "bin" / "windows-amd64" / "tokcount.exe",
        repo_root / "bin" / "gemini-token-counter.exe",
        repo_root / "bin" / "gemini-token-counter",
    ]
    for candidate in candidates:
        if _local_counter_exe_is_runnable(candidate):
            return str(candidate)
    for name in ("gemini-token-counter", "tokcount"):
        found = shutil.which(name)
        if found and _local_counter_exe_is_runnable(Path(found)):
            return found
    return None


@dataclass
class LocalGeminiTokenCounter:
    """Token counter backed by the bundled local gemini-token-counter binary.

    Offline and quota-free: shells out to the Go tokenizer CLI and applies the
    constant ``LOCAL_COUNTER_API_OFFSET`` so counts match the countTokens API.
    Raises (so callers can fall back) when the binary is missing or errors.
    """

    exe_path: str | None = None
    model: str = LOCAL_TOKENIZER_MODEL
    api_offset: int = LOCAL_COUNTER_API_OFFSET
    timeout_seconds: float = 30.0
    audio_tokens_per_second: int = DEFAULT_LIMITS.audio_tokens_per_second
    source: str = "gemini-token-counter-local"

    def __post_init__(self) -> None:
        if self.exe_path is None:
            self.exe_path = _resolve_local_counter_exe()
        self._cache: dict[str, int] = {}

    @property
    def available(self) -> bool:
        return bool(self.exe_path) and _local_counter_exe_is_runnable(Path(self.exe_path))

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in self._cache:
            return self._cache[digest]
        count = self._run(text)
        self._cache[digest] = count
        return count

    def count_texts(self, texts: Iterable[str]) -> int:
        # The local tokenizer takes a single text; the API merges parts inside
        # one content with no separator. Joining with newlines is within a
        # couple of tokens and errs high, which is safe for budget checks.
        joined = "\n".join(text for text in texts if text)
        return self.count_text(joined)

    def count_audio_seconds(self, seconds: float) -> int:
        if seconds <= 0:
            return 0
        return int(math.ceil(seconds * self.audio_tokens_per_second))

    def _run(self, text: str) -> int:
        if not self.available:
            raise RuntimeError(f"gemini-token-counter binary not found: {self.exe_path!r}")
        proc = subprocess.run(
            [str(self.exe_path), "-model", self.model],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        for line in reversed(proc.stdout.decode("utf-8", "replace").splitlines()):
            line = line.strip()
            if line.isdigit():
                return int(line) + self.api_offset
        stderr = proc.stderr.decode("utf-8", "replace")[:300]
        raise RuntimeError(
            f"gemini-token-counter returned no count (rc={proc.returncode}): {stderr}"
        )


# Per-character-class heuristic weights (tokens per char). Empirically fit (see
# repo history / scratchpad) so the estimate is an UPPER BOUND on the real
# countTokens count across every tested category — digits, Latin, Chinese,
# Japanese (kana+kanji), Korean, other scripts (Cyrillic/Arabic/Thai), CJK and
# ASCII punctuation, emoji, and the dense subtitle-CSV format the LLM layer
# actually feeds the model (margin ~+8% there). Weights are ceilings, not means:
# never under-counting matters more than tightness because the `lazy` truncation
# fast path trusts this as an upper bound (heuristic <= limit => real <= limit).
HEURISTIC_CHAR_WEIGHTS: dict[str, float] = {
    "digit": 1.1,
    "cjk": 0.82,          # CJK ideographs + Japanese kana
    "hangul": 0.90,       # Korean
    "wide_punct": 0.95,   # CJK symbols/punctuation + fullwidth forms
    "other_script": 0.48, # Cyrillic / Arabic / Thai / accented Latin / ...
    "latin": 0.32,
    "space": 0.35,
    "ascii_sym": 1.0,     # ASCII punctuation / symbols (non-space, non-alnum)
    "other": 1.15,        # emoji and rare symbols
}
# Small flat margin so tiny inputs never dip below real after rounding; this is
# the "末尾 constant". Added on top of api_offset (the +1 countTokens envelope).
HEURISTIC_STABILITY_CONSTANT = 2


def classify_char(char: str) -> str:
    """Bucket a character into a `HEURISTIC_CHAR_WEIGHTS` class."""

    code = ord(char)
    if char in " \t\n\r\f\v":
        return "space"
    if char.isdigit():
        return "digit"
    if 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return "cjk"
    if 0xAC00 <= code <= 0xD7A3:
        return "hangul"
    if 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF or 0x2E00 <= code <= 0x2FFF:
        return "wide_punct"
    if "a" <= char.lower() <= "z":
        return "latin"
    if char.isascii():
        return "ascii_sym"
    if char.isalpha():
        return "other_script"
    return "other"


@dataclass
class HeuristicTokenCounter:
    """Approximate, dependency-free token counter used only as a last resort.

    Reached when both the local binary and the countTokens API are unavailable,
    and also used as the cheap upper-bound pre-check for `lazy` truncation. Sums
    per-character-class weights (`HEURISTIC_CHAR_WEIGHTS`) plus a small constant.
    The weights are an **empirical upper bound over the tested categories**, not
    a proof for adversarial input — but they hold across digits / Latin / CJK /
    JP / KO / other scripts / punctuation / mixed subtitle-CSV. The old heuristic
    that under-counted CJK by 25-40% is deliberately not reused.
    """

    weights: Mapping[str, float] = field(
        default_factory=lambda: dict(HEURISTIC_CHAR_WEIGHTS)
    )
    stability_constant: int = HEURISTIC_STABILITY_CONSTANT
    api_offset: int = LOCAL_COUNTER_API_OFFSET
    audio_tokens_per_second: int = DEFAULT_LIMITS.audio_tokens_per_second
    source: str = "heuristic"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        default = self.weights.get("other", 1.15)
        estimate = 0.0
        for char in text:
            estimate += self.weights.get(classify_char(char), default)
        return max(1, math.ceil(estimate) + self.stability_constant + self.api_offset)

    def count_texts(self, texts: Iterable[str]) -> int:
        joined = "\n".join(text for text in texts if text)
        return self.count_text(joined)

    def count_audio_seconds(self, seconds: float) -> int:
        if seconds <= 0:
            return 0
        return int(math.ceil(seconds * self.audio_tokens_per_second))


@dataclass
class FallbackTokenCounter:
    """Try each backing counter in order, falling through on failure.

    Default order is local binary -> countTokens API -> heuristic. The result
    is cached per-text so repeated prefixes (e.g. the truncation search) reuse a
    single successful backend. ``last_source`` reflects the backend that most
    recently answered.
    """

    counters: Sequence[TokenCounter] = field(default_factory=tuple)
    source: str = "gemini-token-counter+api+heuristic"
    last_source: str = ""

    def __post_init__(self) -> None:
        if not self.counters:
            raise ValueError("FallbackTokenCounter requires at least one counter")
        self._text_cache: dict[str, int] = {}

    def _dispatch(self, method: str, key: str, *args: Any) -> int:
        if key in self._text_cache:
            return self._text_cache[key]
        errors: list[str] = []
        for counter in self.counters:
            try:
                value = getattr(counter, method)(*args)
            except Exception as exc:  # noqa: BLE001 - fall through to next backend
                errors.append(f"{getattr(counter, 'source', counter)}: {exc}")
                continue
            self.last_source = getattr(counter, "source", "")
            self._text_cache[key] = value
            return value
        raise RuntimeError("all token counters failed: " + "; ".join(errors))

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return self._dispatch("count_text", "t\x00" + text, text)

    def count_texts(self, texts: Iterable[str]) -> int:
        items = [text for text in texts if text]
        if not items:
            return 0
        return self._dispatch("count_texts", "m\x00" + "\x00".join(items), items)

    def count_audio_seconds(self, seconds: float) -> int:
        # Every backend computes this identically from the local audio rate.
        return self.counters[0].count_audio_seconds(seconds)


def default_token_counter(
    *,
    model: str = GEMINI_31_FLASH_LITE,
    api_version: str = "v1beta",
) -> FallbackTokenCounter:
    """Standard counter: local binary -> countTokens API -> heuristic fallback."""

    return FallbackTokenCounter(
        counters=(
            LocalGeminiTokenCounter(),
            GeminiCountTokensCounter(model=model, api_version=api_version),
            HeuristicTokenCounter(),
        )
    )


@dataclass(frozen=True)
class CorrectionBudget:
    input_tokens: int
    subtitle_input_tokens: int
    estimated_output_tokens: int
    total_with_margin: int
    token_counter_source: str


def requested_output_limit(limits: ModelLimits = DEFAULT_LIMITS) -> int:
    return limits.output_limit


def build_correction_budget(
    *,
    input_tokens: int,
    subtitle_input_tokens: int,
    token_counter_source: str,
    limits: ModelLimits = DEFAULT_LIMITS,
    profile: TranslationProfile = DEFAULT_PROFILE,
) -> CorrectionBudget:
    # Output expectation is k x c x csv_tokens per the route/level profile
    # (the old "csv x 5 + 10k" heuristic was replaced by the mm-med preset).
    output_tokens = expected_output_tokens(profile, subtitle_input_tokens)
    return CorrectionBudget(
        input_tokens=input_tokens,
        subtitle_input_tokens=subtitle_input_tokens,
        estimated_output_tokens=output_tokens,
        total_with_margin=input_tokens + output_tokens + limits.safety_margin,
        token_counter_source=token_counter_source,
    )


def validate_correction_budget(
    budget: CorrectionBudget,
    *,
    limits: ModelLimits = DEFAULT_LIMITS,
) -> None:
    if budget.input_tokens > limits.prompt_input_limit:
        raise TokenBudgetError(
            "Prompt input tokens exceed free-tier limit: "
            f"{budget.input_tokens} > {limits.prompt_input_limit}"
        )
    if budget.estimated_output_tokens > limits.output_limit:
        raise TokenBudgetError(
            "Estimated output tokens exceed output limit: "
            f"{budget.estimated_output_tokens} > {limits.output_limit}"
        )
    if budget.total_with_margin > limits.context_limit:
        raise TokenBudgetError(
            "Estimated request exceeds context limit: "
            f"{budget.total_with_margin} > {limits.context_limit}"
        )

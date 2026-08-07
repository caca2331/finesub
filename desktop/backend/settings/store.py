from __future__ import annotations

from pathlib import Path
from typing import Literal

from desktop.backend.common.models import (
    BridgeError,
    CapabilityState,
    PipelineStage,
    PublicSettings,
)
from finesub_bootstrap import secrets


Provider = Literal["gemini", "exa", "tavily"]

_ENV_NAMES: dict[Provider, str] = {
    "gemini": "GEMINI_FREE",
    "exa": "EXA_KEYS",
    "tavily": "TAVILY_KEYS",
}
_LEGACY_ENV_NAMES: dict[Provider, str] = {
    "gemini": "GEMINI_API_KEY",
    "exa": "EXA_API_KEY",
    "tavily": "TAVILY_API_KEY",
}


class SettingsStore:
    def __init__(self, user_data: Path) -> None:
        self.user_data = user_data.expanduser().resolve()
        self.env_path = self.user_data / ".env"

    def get_capabilities(self) -> CapabilityState:
        keys = self._read_keys()
        return CapabilityState(
            raw_srt=True,
            translation=bool(keys.get("GEMINI_FREE")),
            web_search=bool(keys.get("EXA_KEYS") or keys.get("TAVILY_KEYS")),
        )

    def validate_stage(self, stage: PipelineStage) -> BridgeError | None:
        if stage not in {"translated-srt", "final-srt"}:
            return None
        if self.get_capabilities().translation:
            return None
        return BridgeError(
            code="api_key_required",
            message="翻译功能需要填写 Gemini API Key。",
            action="open_settings",
        )

    def public_settings(self) -> PublicSettings:
        keys = self._read_keys()
        return PublicSettings(
            api_keys={
                provider: "configured" if keys.get(env_name) else "missing"
                for provider, env_name in _ENV_NAMES.items()
            }
        )

    def build_worker_env(self) -> dict[str, str]:
        keys = self._read_keys()
        return {
            env_name: keys[env_name]
            for env_name in _ENV_NAMES.values()
            if keys.get(env_name)
        }

    def save_api_keys(
        self,
        *,
        gemini: str | None = None,
        exa: str | None = None,
        tavily: str | None = None,
    ) -> None:
        requested: dict[Provider, str | None] = {
            "gemini": gemini,
            "exa": exa,
            "tavily": tavily,
        }
        # Only the providers the caller actually changed are written: a value
        # that is unreadable on this machine must keep its ciphertext line
        # (it may open fine on the machine the file came from).
        updates: dict[str, str | None] = {}
        for provider, value in requested.items():
            if value is None:
                continue
            normalized = self._normalize_secret(value)
            updates[_ENV_NAMES[provider]] = normalized or None
        if updates:
            self._write_keys(updates)

    def delete_api_key(self, provider: Provider) -> None:
        if provider not in _ENV_NAMES:
            raise ValueError(f"Unknown API provider: {provider}")
        self._write_keys({_ENV_NAMES[provider]: None})

    def reveal_api_keys(self) -> dict[str, list[dict[str, str]]]:
        """Plaintext entries per provider, for the settings panel.

        Exists because the protected values are bound to this Windows account:
        the user must be able to take their keys out *before* a machine switch
        or reinstall, and desktop users cannot be assumed to reach for the CLI.
        """

        keys = self._read_keys()
        return {
            provider: [
                {"name": label, "key": key, "masked": secrets.masked(key)}
                for label, key in secrets.iter_entries(keys.get(env_name, ""))
            ]
            for provider, env_name in _ENV_NAMES.items()
        }

    @staticmethod
    def _normalize_secret(value: str) -> str:
        normalized = value.strip()
        if "\r" in normalized or "\n" in normalized:
            raise ValueError("API keys must be a single line")
        return normalized

    def _read_keys(self) -> dict[str, str]:
        known_names = {
            *_ENV_NAMES.values(),
            *_LEGACY_ENV_NAMES.values(),
        }
        values = {
            name: value
            for name, value in secrets.read_env_file(self.env_path).items()
            if name in known_names and value
        }
        migration: dict[str, str | None] = {}
        for provider, legacy_name in _LEGACY_ENV_NAMES.items():
            current_name = _ENV_NAMES[provider]
            legacy_value = values.pop(legacy_name, "")
            if not legacy_value:
                continue
            migration[legacy_name] = None
            if not values.get(current_name):
                values[current_name] = legacy_value
                migration[current_name] = legacy_value
        if migration:
            self._write_keys(migration)
        return {
            name: values[name]
            for name in _ENV_NAMES.values()
            if values.get(name)
        }

    def _write_keys(self, updates: dict[str, str | None]) -> None:
        # Line-preserving by contract: comments, the FINESUB_KEYRING line and
        # variables not named here survive byte for byte, and new values are
        # born encrypted (plaintext with a warning when DPAPI is unavailable).
        secrets.update_env_file(self.env_path, updates)

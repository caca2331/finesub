from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from desktop.backend.common.models import (
    BridgeError,
    CapabilityState,
    PipelineStage,
    PublicSettings,
)


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
        updates: dict[Provider, str | None] = {
            "gemini": gemini,
            "exa": exa,
            "tavily": tavily,
        }
        keys = self._read_keys()
        for provider, value in updates.items():
            if value is None:
                continue
            normalized = self._normalize_secret(value)
            env_name = _ENV_NAMES[provider]
            if normalized:
                keys[env_name] = normalized
            else:
                keys.pop(env_name, None)
        self._write_keys(keys)

    def delete_api_key(self, provider: Provider) -> None:
        if provider not in _ENV_NAMES:
            raise ValueError(f"Unknown API provider: {provider}")
        keys = self._read_keys()
        keys.pop(_ENV_NAMES[provider], None)
        self._write_keys(keys)

    @staticmethod
    def _normalize_secret(value: str) -> str:
        normalized = value.strip()
        if "\r" in normalized or "\n" in normalized:
            raise ValueError("API keys must be a single line")
        return normalized

    def _read_keys(self) -> dict[str, str]:
        if not self.env_path.is_file():
            return {}
        values: dict[str, str] = {}
        known_names = {
            *_ENV_NAMES.values(),
            *_LEGACY_ENV_NAMES.values(),
        }
        for raw_line in self.env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if name in known_names:
                values[name] = value.strip()
        migrated = False
        for provider, legacy_name in _LEGACY_ENV_NAMES.items():
            current_name = _ENV_NAMES[provider]
            legacy_value = values.pop(legacy_name, "")
            if legacy_value and not values.get(current_name):
                values[current_name] = legacy_value
            if legacy_value:
                migrated = True
        current = {
            name: values[name]
            for name in _ENV_NAMES.values()
            if values.get(name)
        }
        if migrated:
            self._write_keys(current)
        return current

    def _write_keys(self, keys: dict[str, str]) -> None:
        self.user_data.mkdir(parents=True, exist_ok=True)
        temp_path = self.env_path.with_suffix(".tmp")
        lines = [
            f"{env_name}={keys[env_name]}"
            for env_name in _ENV_NAMES.values()
            if keys.get(env_name)
        ]
        payload = "\n".join(lines)
        if payload:
            payload += "\n"
        temp_path.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(temp_path, self.env_path)

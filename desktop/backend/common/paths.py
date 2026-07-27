from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path
    app: Path
    app_versions: Path
    app_current: Path
    runtime: Path
    models: Path
    user_data: Path
    cache: Path
    logs: Path

    @classmethod
    def for_root(cls, root: Path) -> "AppPaths":
        resolved = root.expanduser().resolve()
        app = resolved / "app"
        user_data = resolved / "user-data"
        return cls(
            root=resolved,
            app=app,
            app_versions=app / "versions",
            app_current=app / "current.json",
            runtime=resolved / "runtime",
            models=resolved / "models",
            user_data=user_data,
            cache=resolved / "cache",
            logs=user_data / "logs",
        )

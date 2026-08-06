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
    def for_root(
        cls,
        root: Path,
        *,
        user_data: Path | None = None,
    ) -> "AppPaths":
        """Lay out the application directories under ``root``.

        ``user_data`` relocates personal data (settings, API keys, task
        history, logs) away from the rebuildable root — installed desktop
        copies point it at ``%LOCALAPPDATA%\\FineSub\\user-data`` so the
        install directory stays disposable; portable copies leave it unset.
        """

        resolved = root.expanduser().resolve()
        app = resolved / "app"
        resolved_user_data = (
            user_data.expanduser().resolve()
            if user_data is not None
            else resolved / "user-data"
        )
        return cls(
            root=resolved,
            app=app,
            app_versions=app / "versions",
            app_current=app / "current.json",
            runtime=resolved / "runtime",
            models=resolved / "models",
            user_data=resolved_user_data,
            cache=resolved / "cache",
            logs=resolved_user_data / "logs",
        )

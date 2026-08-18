"""``user-data/settings.json``: the state only the desktop front end reads.

Kept apart from ``config.toml`` -- which the CLI, the worker and a human with
an editor all share -- along two lines that have nothing to do with which one
has a UI:

* **Blast radius.** This file is rewritten every time someone flips a theme;
  ``config.toml`` holds provider switches and is rewritten rarely and
  deliberately. A bug in the frequent writer must not be able to damage the
  careful one.
* **Unknown keys.** Here they are dropped in silence, because they are leftover
  program state from another version and a user cannot act on the error. In
  ``config.toml`` they are a typo someone made and must be reported.

Sparse throughout: a setting appears only once it has been chosen, so a code
default can still be improved for everyone who never chose. Nothing here
belongs in a produced artifact -- what a run actually used is recorded by the
pipeline itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from desktop.backend.common.models import Preferences, TaskDefaults
from finesub_bootstrap.fsops import write_atomic
from finesub_bootstrap.locks import holding_lock


SETTINGS_FILENAME = "settings.json"
SCHEMA_VERSION = 1
# A renderer preference blob is a handful of strings; anything approaching this
# is a bug or a misuse, and this file is on the app's startup path.
_MAX_BYTES = 64 * 1024
_LOCK_TIMEOUT_SECONDS = 10.0


class PreferencesStore:
    def __init__(self, user_data: Path) -> None:
        self.user_data = user_data.expanduser().resolve()
        self.path = self.user_data / SETTINGS_FILENAME

    def load(self) -> Preferences:
        """Never raises: an unreadable preference file is worth less than the
        app starting. A damaged one is replaced on the next save."""

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Preferences()
        if not isinstance(raw, dict):
            return Preferences()
        ui = raw.get("ui")
        try:
            defaults = TaskDefaults.model_validate(raw.get("task_defaults") or {})
        except ValidationError:
            # One bad field should not cost the user the rest; validate field
            # by field and keep what still makes sense.
            defaults = _salvage_defaults(raw.get("task_defaults"))
        return Preferences(
            ui=ui if isinstance(ui, dict) else {},
            task_defaults=defaults,
        )

    def save(
        self,
        *,
        ui: dict[str, Any] | None = None,
        task_defaults: dict[str, Any] | None = None,
    ) -> Preferences:
        """Merge the named settings in; ``None`` values reset a setting.

        Only what the caller passes is touched, so two panels saving different
        things cannot clobber each other's fields.

        Locked across the whole read-modify-write: bridge calls arrive on their
        own threads, and the front end fires its saves without awaiting, so two
        overlapping saves would otherwise both read the old file and the later
        writer would drop the earlier one's field. `write_atomic` alone only
        buys an untorn file, not a serialized update.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with holding_lock(
            self.path.with_name(f"{self.path.name}.lock"),
            timeout=_LOCK_TIMEOUT_SECONDS,
        ):
            return self._save_locked(ui=ui, task_defaults=task_defaults)

    def _save_locked(
        self,
        *,
        ui: dict[str, Any] | None,
        task_defaults: dict[str, Any] | None,
    ) -> Preferences:
        current = self.load()
        merged_ui = dict(current.ui)
        if ui is not None:
            for key, value in ui.items():
                if value is None:
                    merged_ui.pop(key, None)
                else:
                    merged_ui[key] = value
        merged_defaults = current.task_defaults.model_dump()
        if task_defaults is not None:
            for key, value in task_defaults.items():
                if value is None:
                    merged_defaults.pop(key, None)
                else:
                    merged_defaults[key] = value
        # Validate the merge, not the patch: a field that only makes sense
        # together with another must be judged in its final state.
        result = Preferences(
            ui=merged_ui,
            task_defaults=TaskDefaults.model_validate(merged_defaults),
        )
        payload = {
            "schema": SCHEMA_VERSION,
            "ui": result.ui,
            # Sparse by construction -- see TaskDefaults' serializer.
            "task_defaults": result.task_defaults.model_dump(),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if len(text.encode("utf-8")) > _MAX_BYTES:
            raise ValueError("settings payload is too large")
        write_atomic(self.path, text)
        return result


def _salvage_defaults(raw: object) -> TaskDefaults:
    if not isinstance(raw, dict):
        return TaskDefaults()
    kept: dict[str, Any] = {}
    for key, value in raw.items():
        try:
            TaskDefaults.model_validate({key: value})
        except ValidationError:
            continue
        kept[key] = value
    try:
        return TaskDefaults.model_validate(kept)
    except ValidationError:
        return TaskDefaults()


__all__ = ["PreferencesStore", "SETTINGS_FILENAME", "SCHEMA_VERSION"]

"""The two text tables in ``prompt_templates/``: ``mode_texts_v1.toml`` (the phrases that differ between a
translating task and a correct-only one) and ``phrases_v1.toml`` (the short phrases Python picks and joins by
condition, and the clauses of each correction prompt set).

They are read here, below every module that renders a prompt, because ``prompt_variants`` builds its sets from
them at import and ``prompt_compose`` imports ``prompt_variants``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib
from typing import Any, Dict, Mapping

PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"

MODE_TEXTS_FILE = "mode_texts_v1.toml"
PHRASES_FILE = "phrases_v1.toml"


@lru_cache(maxsize=None)
def _mode_table() -> Mapping[str, Mapping[str, str]]:
    return tomllib.loads((PROMPT_TEMPLATE_DIR / MODE_TEXTS_FILE).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _phrase_table() -> Mapping[str, Mapping[str, Any]]:
    return tomllib.loads((PROMPT_TEMPLATE_DIR / PHRASES_FILE).read_text(encoding="utf-8"))


def prompt_phrases(group: str) -> Mapping[str, Any]:
    """The short phrases a composition joins by condition (`prompt_templates/phrases_v1.toml`)."""

    return _phrase_table()[group]


def task_mode_texts(correct_only: bool) -> Dict[str, str]:
    """The phrases that differ between a translating task and a correct-only one, keyed by their template
    placeholders."""

    return dict(_mode_table()["correct_only" if correct_only else "translate"])

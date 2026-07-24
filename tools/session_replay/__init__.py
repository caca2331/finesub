"""Session replay: freeze prior session injections and re-run a later harness call.

Used to iterate prompts on correction R2 while reusing frozen R1 search/extract
text (and other injections). Additional session kinds register via ``registry``.
"""

from __future__ import annotations

from .registry import SESSIONS, get_session
from .run import run_session_replay

__all__ = ["SESSIONS", "get_session", "run_session_replay"]

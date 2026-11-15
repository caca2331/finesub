"""The languages the Qwen referee names, and their Whisper-style codes.

Qwen reports a language name ("Japanese"), Whisper and the rows record codes
("ja"). Light on purpose: the stable step's recall repair reads it too, and
stabilize must not import torch.
"""

from __future__ import annotations

from typing import Optional

QWEN_LANGUAGE_CODES = {
    "arabic": "ar",
    "cantonese": "yue",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "malay": "ms",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
}


def qwen_language_code(name: object) -> Optional[str]:
    """Whisper-style code for a Qwen language reply; None when unknown."""

    text = str(name or "").strip().lower().replace("-", "_")
    if not text:
        return None
    if len(text) <= 3:
        return text
    return QWEN_LANGUAGE_CODES.get(text)

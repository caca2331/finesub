from __future__ import annotations

from pathlib import Path
import re

import pytest

from desktop.backend.launcher.main import (
    DARK_WINDOW_COLORS,
    LIGHT_WINDOW_COLORS,
    TITLEBAR_HEIGHT_DP,
    WINDOW_CONTROLS_WIDTH_DP,
    _rgb,
)

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def _stylesheet() -> str:
    return (FRONTEND / "app" / "globals.css").read_text(encoding="utf-8")


def test_the_native_caption_band_matches_the_css_title_bar() -> None:
    # The frameless window has two owners of the same 40px strip: CSS draws it
    # and the Win32 hit test decides what the mouse does there. Drift makes the
    # window undraggable, or steals clicks from the buttons -- neither shows up
    # anywhere but in a user's hands.
    css = _stylesheet()
    height = re.search(r"--titlebar-height:\s*(\d+)px", css)
    buttons = re.search(
        r"\.window-actions\s*\{[^}]*?repeat\(\s*(\d+)\s*,\s*(\d+)px\s*\)", css, re.S
    )
    assert height and buttons

    assert TITLEBAR_HEIGHT_DP == int(height.group(1))
    assert WINDOW_CONTROLS_WIDTH_DP == int(buttons.group(1)) * int(buttons.group(2))


def test_the_startup_frame_colors_are_the_themes_own() -> None:
    # They decide what the frame looks like until the web layer reports the
    # active theme; taken from anywhere else they would be a visible flash.
    css = _stylesheet()
    backgrounds = re.findall(r"--app-bg:\s*(#[0-9a-fA-F]+)", css)
    foregrounds = re.findall(r"--text:\s*(#[0-9a-fA-F]+)", css)
    declared = {_rgb(color) for color in backgrounds + foregrounds}

    for color in LIGHT_WINDOW_COLORS + DARK_WINDOW_COLORS:
        assert _rgb(color) in declared


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#131316", (0x13, 0x13, 0x16)),
        ("  #E8E9EC  ", (0xE8, 0xE9, 0xEC)),
        ("#fff", (255, 255, 255)),
        ("rgb(19, 19, 22)", (19, 19, 22)),
        ("rgba(19 19 22 / 0.5)", (19, 19, 22)),
    ],
)
def test_window_colors_survive_the_forms_css_can_take(
    value: str, expected: tuple[int, int, int]
) -> None:
    # The frontend hands over whatever `getComputedStyle` returns for the
    # theme's custom properties, and a parse failure here is swallowed by the
    # bridge guard -- it would look like the feature simply stopped working.
    assert _rgb(value) == expected


@pytest.mark.parametrize("value", ["", "#12345", "teal", "rgb(1, 2)"])
def test_an_unreadable_window_color_is_rejected_rather_than_guessed(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        _rgb(value)

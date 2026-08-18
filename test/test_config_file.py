"""config.toml must survive being written by the UI and read by a human."""

from __future__ import annotations

from pathlib import Path

import pytest

from finesub_bootstrap.config_file import ConfigWriteError, update_config_file


HAND_WRITTEN = """\
# my notes -- do not lose these
[providers]
gemini_free = true   # the only one I use
tavily = false

[segmentation]
# shorter subtitles for the lecture recordings
length_scale = 0.8
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_editing_one_value_leaves_every_other_byte_alone(tmp_path) -> None:
    path = _write(tmp_path, HAND_WRITTEN)

    update_config_file(path, {"segmentation": {"length_scale": 0.9}})

    assert path.read_text(encoding="utf-8") == HAND_WRITTEN.replace(
        "length_scale = 0.8", "length_scale = 0.9"
    )


def test_a_trailing_comment_on_the_edited_line_survives(tmp_path) -> None:
    path = _write(tmp_path, HAND_WRITTEN)

    update_config_file(path, {"providers": {"gemini_free": False}})

    assert "gemini_free = false   # the only one I use" in path.read_text(
        encoding="utf-8"
    )


def test_removing_a_setting_removes_the_line(tmp_path) -> None:
    # The sparse policy: back at the default means gone from the file, not
    # written out explicitly, so the default can still be improved later.
    path = _write(tmp_path, HAND_WRITTEN)

    update_config_file(path, {"segmentation": {"length_scale": None}})

    text = path.read_text(encoding="utf-8")
    assert "length_scale" not in text
    assert "# shorter subtitles for the lecture recordings" in text
    assert "[segmentation]" in text


def test_a_new_key_lands_inside_its_table_not_after_the_blank_line(tmp_path) -> None:
    path = _write(tmp_path, "[providers]\ntavily = false\n\n[chunking]\nx = 1\n")

    update_config_file(path, {"providers": {"exa": True}})

    assert path.read_text(encoding="utf-8") == (
        "[providers]\ntavily = false\nexa = true\n\n[chunking]\nx = 1\n"
    )


def test_a_new_table_is_appended(tmp_path) -> None:
    path = _write(tmp_path, "[providers]\ntavily = false\n")

    update_config_file(path, {"segmentation": {"length_scale": 0.75}})

    assert path.read_text(encoding="utf-8") == (
        "[providers]\ntavily = false\n\n[segmentation]\nlength_scale = 0.75\n"
    )


def test_a_missing_file_is_created(tmp_path) -> None:
    path = tmp_path / "nested" / "config.toml"

    update_config_file(path, {"segmentation": {"length_scale": 1.2}})

    assert path.read_text(encoding="utf-8") == "[segmentation]\nlength_scale = 1.2\n"


def test_clearing_a_setting_that_was_never_written_creates_no_file(tmp_path) -> None:
    path = tmp_path / "config.toml"

    update_config_file(path, {"segmentation": {"length_scale": None}})

    assert not path.exists()


def test_removing_an_absent_key_is_a_no_op(tmp_path) -> None:
    path = _write(tmp_path, HAND_WRITTEN)

    update_config_file(path, {"chunking": {"max_window_subtitle_tokens": None}})

    assert path.read_text(encoding="utf-8") == HAND_WRITTEN


def test_crlf_endings_are_kept(tmp_path) -> None:
    path = _write(tmp_path, "[segmentation]\r\nlength_scale = 1.0\r\n")

    update_config_file(path, {"segmentation": {"length_scale": 0.7}})

    assert path.read_bytes() == b"[segmentation]\r\nlength_scale = 0.7\r\n"


def test_arrays_are_refused_rather_than_reformatted(tmp_path) -> None:
    path = _write(tmp_path, HAND_WRITTEN)

    with pytest.raises(ConfigWriteError, match="limited to scalars"):
        update_config_file(path, {"pools": {"gemini_free": ["main", "spare"]}})

    assert path.read_text(encoding="utf-8") == HAND_WRITTEN


def test_a_multiline_string_makes_the_writer_refuse(tmp_path) -> None:
    body = '[notes]\ntext = """\nline\n"""\n'
    path = _write(tmp_path, body)

    with pytest.raises(ConfigWriteError, match="multi-line string"):
        update_config_file(path, {"segmentation": {"length_scale": 0.7}})

    assert path.read_text(encoding="utf-8") == body


def test_an_array_of_tables_makes_the_writer_refuse(tmp_path) -> None:
    # `[[extra]]` is not a header the span scanner can see, so [providers] would
    # appear to run through it and disabling tavily would rewrite the wrong line.
    body = "[providers]\ntavily = false\n\n[[extra]]\ntavily = true\n"
    path = _write(tmp_path, body)

    with pytest.raises(ConfigWriteError, match="array of tables"):
        update_config_file(path, {"providers": {"tavily": True}})

    assert path.read_text(encoding="utf-8") == body


def test_several_settings_in_one_write(tmp_path) -> None:
    path = _write(tmp_path, HAND_WRITTEN)

    update_config_file(
        path,
        {
            "segmentation": {"length_scale": None},
            "providers": {"tavily": True, "exa": False},
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "tavily = true" in text
    assert "exa = false" in text
    assert "length_scale" not in text

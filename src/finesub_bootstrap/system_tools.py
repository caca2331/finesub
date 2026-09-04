"""Reuse an external tool already on PATH instead of downloading our own.

`RuntimeEnvironment` has always done this for Python: find a candidate with
`shutil.which`, then *run* it to confirm it is really usable before trusting
it. The same reasoning applies to the tools the pipeline shells out to -- a
machine that already has a working ffmpeg should not pay 140 MB for a second
copy -- but "on PATH" is not the same as "usable", so every candidate has to
answer a capability question before it is accepted.

What is deliberately *not* here is yt-dlp. The pipeline imports it rather than
executing it, and it imports it from the managed runtime's interpreter, which
cannot see the user's site-packages. A yt-dlp installed system-wide is
invisible there, so there is nothing to reuse and it is always managed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from finesub_bootstrap import token_counter


@dataclass(frozen=True, slots=True)
class SystemTool:
    """An external tool accepted for use, and where it came from."""

    path: Path
    version: str

    @property
    def directory(self) -> Path:
        return self.path.parent


def _version_token(banner: str, prefix: str) -> str:
    """Pull the version out of a `<tool> version X ...` banner.

    The full first line runs to ~90 characters for ffmpeg (build tag, copyright
    notice), and the UI renders this string next to the resource name.
    """

    first = banner.strip().splitlines()[0].strip() if banner.strip() else ""
    if not first:
        return "unknown"
    words = first.split()
    if len(words) >= 3 and words[0] == prefix and words[1] == "version":
        return words[2]
    return first


def no_window() -> int:
    """Creation flags that keep a subprocess from flashing a console window.

    Load-bearing for both front ends: they are packaged --windowed and own no
    console, so anything they spawn without this pops one up on screen.
    """

    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def probe(command: list[str], timeout: float = 10.0) -> str | None:
    """Run `command`, returning its combined output, or None if it is unusable."""

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=no_window(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


#: The encoders the pipeline actually asks ffmpeg for, and the reason this
#: check is not just "is ffmpeg on PATH". `aac` is every clip's audio track;
#: `libx264` is every clip's video track -- the video windows, agy's transcode,
#: and the single-black-frame MP4 that `containerize_audio_for_agy` wraps an
#: audio window in, because agy rejects a bare audio MIME type.
#:
#: `libx264` is the one that actually tells builds apart. It is GPL, so the
#: LGPL variants of the common Windows distributions are configured
#: `--disable-libx264` and answer `Unknown encoder 'libx264'` the moment a run
#: uses anything but text as its correction reference. `aac` and ffmpeg's
#: native `flac` (`transcode_to_lossless_audio`) are in every build; `flac` is
#: left out because a name that cannot fail teaches nothing about a build.
#:
#: `finesub_bootstrap` may not import the main package, so this is a second
#: copy of what `finesub.media.ffmpeg` requests. `test_system_tools.py` pins
#: the two together.
REQUIRED_FFMPEG_ENCODERS = ("aac", "libx264")


def find_system_ffmpeg(
    required_codecs: tuple[str, ...] = REQUIRED_FFMPEG_ENCODERS,
) -> SystemTool | None:
    """A system ffmpeg, if it can actually do what the pipeline needs.

    Presence is not enough: a build without the encoders the pipeline uses
    would fail mid-run, long after the user chose to skip the download. Both
    ffmpeg and ffprobe must be there, since the pipeline calls each.
    """

    executable = shutil.which("ffmpeg")
    if executable is None or shutil.which("ffprobe") is None:
        return None
    banner = probe([executable, "-version"])
    if not banner:
        return None
    encoders = probe([executable, "-hide_banner", "-encoders"])
    if not encoders:
        return None
    if any(codec not in encoders for codec in required_codecs):
        return None
    return SystemTool(
        path=Path(executable).resolve(),
        version=_version_token(banner, "ffmpeg"),
    )


def find_system_git() -> SystemTool | None:
    """A system git, if it runs.

    No capability check beyond that: the knowledge base uses init/add/commit/
    status/rev-parse, which every git in circulation has. What matters is that
    a broken shim on PATH does not read as success.
    """

    executable = shutil.which("git")
    if executable is None:
        return None
    banner = probe([executable, "--version"], timeout=5.0)
    if not banner:
        return None
    return SystemTool(
        path=Path(executable).resolve(),
        version=_version_token(banner, "git"),
    )


def find_system_token_counter() -> SystemTool | None:
    """A local tokenizer binary this machine already has, if it counts.

    Unlike the two finders above this one also honours the environment
    variable, because it answers a narrower question than "is it on PATH":
    *would the pipeline find a counter without us?* The pipeline reads
    `GEMINI_TOKEN_COUNTER_EXE` first and only then looks at PATH, so anything
    it would accept has to stop us from downloading a second copy.

    The capability question is the whole job: print an integer for a string.
    A shim that runs but answers something else is worse than nothing here,
    since we would then hand the pipeline a counter it trusts absolutely.
    """

    executable = token_counter.configured_path() or token_counter.find_on_path()
    if not executable or not Path(executable).is_file():
        return None
    # Loading the vocabulary is what makes this slow, and it happens before the
    # count -- a five-second budget would reject a working binary on a cold
    # disk. The tokenizer also prints an experimental-tokenizer warning to
    # stderr, which `probe` folds into stdout ahead of the number.
    output = probe([executable, "hello world"], timeout=30.0)
    if not output:
        return None
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines or not lines[-1].isdigit():
        return None
    # No version flag to ask: the binary reports only counts. "unknown" is what
    # `_version_token` falls back to for the same reason.
    return SystemTool(path=Path(executable).resolve(), version="unknown")

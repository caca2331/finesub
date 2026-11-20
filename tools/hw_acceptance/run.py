"""Community acceptance, step 1: run two arms of the pipeline on this machine.

The reference arm and the candidate arm run the same sources on the same
machine, so the comparison is not muddied by another machine or platform: the
candidate is the accelerated path under test, the reference is the CPU path
(`--device cpu`) unless another reference is given. What to send back and how
to read it: tools/hw_acceptance/README.md.

    python -m tools.hw_acceptance.run --out tmp/hw-acceptance \\
        a.mp4 b.mp4 long-talk.mp4

At least three sources, one of them at least 30 minutes long. Then score:

    python -m tools.hw_acceptance.score tmp/hw-acceptance

What to send back: `tmp/hw-acceptance/acceptance.json` (scores + environment).
The media and subtitles stay on your machine.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
import shutil
import subprocess
import sys
from pathlib import Path

REFERENCE = "reference"
CANDIDATE = "candidate"
MIN_SOURCES = 3
LONG_SOURCE_SEC = 30 * 60


def _duration_sec(source: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(source)],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def environment() -> dict:
    """The versions an acceptance record is bound to (it says nothing about others)."""

    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in ("mlx", "mlx-whisper", "faster-whisper"):
        try:
            info[package] = version(package)
        except PackageNotFoundError:
            info[package] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["torch_hip"] = getattr(torch.version, "hip", None)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        info["torch"] = f"unavailable: {exc}"
    try:
        import ctranslate2

        info["ctranslate2"] = ctranslate2.__version__
    except Exception as exc:  # noqa: BLE001
        info["ctranslate2"] = f"unavailable: {exc}"
    try:
        from finesub.speech.runtime import device, support_levels

        gpu = device.accelerator()
        info["accelerator"] = {"vendor": gpu.vendor, "name": gpu.name, "arch": gpu.arch}
        info["platform_family"] = support_levels.platform_family()
        if gpu.vendor:
            info["support_level"] = support_levels.support_level(gpu.vendor)
    except Exception as exc:  # noqa: BLE001
        info["accelerator"] = f"unavailable: {exc}"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference-args",
        default="--device cpu",
        help="pipeline options that make the reference arm (default: the CPU path)",
    )
    parser.add_argument(
        "--candidate-args",
        default="",
        help="pipeline options that make the candidate arm (default: none, i.e. auto)",
    )
    parser.add_argument(
        "--own-vocals",
        action="store_true",
        help=(
            "let the reference arm separate on its own instead of reusing the candidate's "
            "vocal track: compares separation too, and on the CPU that takes hours for long files"
        ),
    )
    args = parser.parse_args()

    stems = [source.stem for source in args.sources]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        # Both arms key their outputs by stem, so `talk.mp4` and `talk.mkv`
        # would run into the same directory and score against themselves.
        print(
            f"error: two sources share a name ({', '.join(duplicates)}); rename one",
            file=sys.stderr,
        )
        return 2

    durations = {str(source): _duration_sec(source) for source in args.sources}
    if len(args.sources) < MIN_SOURCES or not any(
        (seconds or 0.0) >= LONG_SOURCE_SEC for seconds in durations.values()
    ):
        print(
            f"warning: an acceptance record needs at least {MIN_SOURCES} sources, one of "
            f"them {LONG_SOURCE_SEC // 60} minutes or longer; this run can only be a "
            "characterisation",
            file=sys.stderr,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"environment": environment(), "arms": {}, "sources": []}
    # Candidate first: by default the reference arm reuses its vocal track (the
    # separation stage skips on an existing output), so the comparison isolates
    # recognition and the referee -- the section-10 triggers were calibrated on
    # one vocal track -- and the CPU arm does not spend hours separating.
    arms = {CANDIDATE: args.candidate_args.split(), REFERENCE: args.reference_args.split()}
    manifest["shared_vocals"] = not args.own_vocals
    manifest["arms"] = dict(arms)
    manifest_path = args.out / "run.json"

    def write_manifest() -> None:
        """After every arm: hours of finished runs must not need a clean exit."""

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    write_manifest()
    for source in args.sources:
        stem = source.stem
        for arm, options in arms.items():
            output = args.out / arm / stem / f"{stem}.srt"
            if arm == REFERENCE and not args.own_vocals:
                vocals = args.out / CANDIDATE / stem / f"{stem}-vocal.ogg"
                if vocals.is_file():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(vocals, output.parent / vocals.name)
            command = [sys.executable, "-m", "finesub.pipeline", str(source), "-o", str(output), *options]
            print(f"== {arm}: {stem}", flush=True)
            completed = subprocess.run(command)
            if completed.returncode != 0:
                print(f"{arm} failed on {stem} (exit {completed.returncode})", file=sys.stderr)
                print(
                    f"the files already in {manifest_path} can still be scored",
                    file=sys.stderr,
                )
                return completed.returncode
        # Listed once both arms are in: score reads pairs, never half of one.
        manifest["sources"].append({"stem": stem, "duration_sec": durations[str(source)]})
        write_manifest()
    print(f"done; now: python -m tools.hw_acceptance.score {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

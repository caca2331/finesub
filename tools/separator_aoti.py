"""CLI over the production AOTI builder, for building packages by hand.

The build itself lives in
``asr_playground.speech.preprocessing.separator_aoti`` because production calls
it too -- a machine with a C++ compiler builds its own packages on first run.
This wrapper exists to build one into a chosen directory and print the manifest,
which is how the benchmark variants in docs/separator-optimization.md are made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_playground.speech.preprocessing.separator_aoti import build_packages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--emulate-precision-casts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Round fused intermediates the way eager AMP does. The right value "
            "is torch-version dependent, so re-measure it on any bump: on 2.9 "
            "it was needed to keep a marginal VAD segment, on 2.11 it is the "
            "only thing costing quality (2.85dB, plus a 10ms boundary shift). "
            "Off by default for 2.11."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        choices=("axis", "auto"),
        default="axis",
        help=(
            "axis (default): bake the per-axis backend measured in E3 into each "
            "package. auto: let the dispatcher choose while tracing."
        ),
    )
    parser.add_argument(
        "--targets",
        choices=("all", "transformers"),
        default="all",
        help="all also builds the band-wise band_split and mask_estimator.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    manifest = build_packages(
        args.output_dir,
        emulate_precision_casts=args.emulate_precision_casts,
        attention_backend=args.attention_backend,
        targets=args.targets,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Is setting `sample_rate` after construction the same as passing it in?

The benchmark axis writes `separator.sample_rate` / `model_instance.sample_rate`
after the separator is built, because finesub's `_build_separator` owns the
constructor. audio-separator's own documented entry point is the constructor
argument. Run both on the same clip with no acceleration and compare bytes.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

from audio_separator.separator import Separator

from finesub.paths import resolve_separator_model_dir

MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
CLIP = Path("tmp/clip60.ogg").resolve()


def run(tag: str, *, via_constructor: bool) -> str:
    outdir = tempfile.mkdtemp(prefix=f"eq_{tag}_")
    kwargs = dict(
        output_dir=outdir,
        output_format="flac",
        output_single_stem="Vocals",
        model_file_dir=str(resolve_separator_model_dir()),
        mdxc_params={"batch_size": 1},
        log_level=logging.WARNING,
    )
    if via_constructor:
        kwargs["sample_rate"] = 16000
    separator = Separator(**kwargs)
    separator.load_model(MODEL)
    if not via_constructor:
        separator.sample_rate = 16000
        separator.model_instance.sample_rate = 16000
    files = separator.separate(str(CLIP), {"Vocals": f"eq_{tag}"})
    path = Path(outdir) / files[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{tag}: {path.name} sha256={digest}")
    return digest


ctor = run("ctor", via_constructor=True)
attr = run("attr", via_constructor=False)
print("identical" if ctor == attr else "DIFFERENT")

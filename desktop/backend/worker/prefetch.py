"""Download the pipeline's model weights ahead of the first task.

Runs under the *managed* interpreter, like the task worker: the launcher's
frozen environment has no torch, no faster-whisper and no huggingface_hub, and
the weights have to land wherever a real run would look for them. The launcher
passes that environment down (``HF_HOME`` and ``FINESUB_MODEL_DIR`` are already
resolved there, cache reuse included), so this module only has to trigger each
download through the library that owns it.

Triggering it through the owning library rather than a repository id of our own
is the point: `faster_whisper` decides which repository ``large-v3-turbo``
means, and the referee names its own model. A second list here would be a
second thing to keep in step.

Progress is deliberately coarse. Each model announces itself on stdout as
``STAGE <n>/<total> <message>`` and everything else is a log line; there is no
byte counter, because the three downloaders report progress three different
ways and a merged percentage would have been a number we maintain rather than
one we measure.
"""

from __future__ import annotations

import sys
import traceback

from finesub_bootstrap.model_caches import PIPELINE_MODEL_IDS
from finesub_bootstrap.model_ensure import verify_downloaded


def _announce(index: int, total: int, message: str) -> None:
    print(f"STAGE {index}/{total} {message}", flush=True)


def _fetch_separator() -> None:
    """Place the separator's files, then let audio-separator load them.

    Measured (docs §5.4): with the checkpoint, its config and the model index
    all present, `load_model` makes no network call at all -- so placing them
    ourselves is what keeps this off `raw.githubusercontent.com`, the hardest
    host to reach from the mainland in this whole path.
    """

    from finesub.paths import resolve_separator_model_dir
    from finesub.speech.preprocessing.separator.separation import place_separator_files
    from finesub_bootstrap.model_caches import SEPARATOR_CHECKPOINT

    from audio_separator.separator import Separator

    model_dir = resolve_separator_model_dir()
    # The separation stage does the same before it builds a separator, so the
    # CLI -- which has no prefetch -- is covered by the same routing and the
    # same digest checks.
    place_separator_files()
    separator = Separator(model_file_dir=str(model_dir))
    # Building the model is wasted work here, but this is the entry point that
    # owns the load, and paying it once beats copying its lookup table.
    separator.load_model(model_filename=SEPARATOR_CHECKPOINT)


def _pinned_revision(model_id: str) -> str | None:
    """The revision the manifest pins, or None to take whatever `main` is.

    Pinning is what stops a public mirror from resolving a moving branch to
    different content than the one this release was tested against. None is
    the honest answer for an unlisted model: pretending to pin would be worse
    than not pinning.
    """

    from finesub_bootstrap.model_manifest import entry_for

    entry = entry_for(model_id)
    return entry.revision if entry is not None and entry.revision else None


def _fetch_whisper() -> None:
    """Fetch the CT2 weights for the model every desktop task uses."""

    from faster_whisper.utils import download_model
    from finesub_bootstrap.model_caches import WHISPER_REPO_ID

    # Explicitly share the repository identity with the lightweight cache
    # status code. Passing the friendly model alias here lets faster-whisper's
    # private mapping drift away from what the resource row looks for.
    download_model(WHISPER_REPO_ID, revision=_pinned_revision("whisper"))


def _fetch_qwen_referee() -> None:
    """Fetch the second-model referee.

    `QwenReferee` loads lazily, so constructing one downloads nothing -- which
    is also why warming the models by running a throwaway task does not work:
    a synthetic clip yields no suspect segments, the referee is never asked
    anything, and its 1.5 GB is never fetched.
    """

    from huggingface_hub import snapshot_download

    from finesub.speech.verification.qwen_referee import DEFAULT_QWEN_MODEL

    snapshot_download(
        DEFAULT_QWEN_MODEL, revision=_pinned_revision("qwen-referee")
    )


_FETCHERS = {
    "separator": ("人声分离模型", _fetch_separator),
    "whisper": ("Whisper 识别模型", _fetch_whisper),
    "qwen-referee": ("Qwen 校验模型", _fetch_qwen_referee),
}


def main() -> int:
    requested = sys.argv[1:] or list(PIPELINE_MODEL_IDS)
    unknown = [model_id for model_id in requested if model_id not in _FETCHERS]
    if unknown:
        print(f"Unknown model ids: {', '.join(unknown)}", file=sys.stderr)
        return 2

    total = len(requested)
    for index, model_id in enumerate(requested, start=1):
        label, fetch = _FETCHERS[model_id]
        _announce(index, total, f"正在获取{label}")
        try:
            fetch()
            if model_id != "separator":
                # The separator's own fetcher verifies each fixed file and
                # stamps it as it goes; the Hugging Face half is verified here,
                # through the same helper the CLI's stage entry uses.
                verify_downloaded(model_id)
        except Exception as error:
            # Named, because "download failed" with three models in flight
            # tells the user nothing about what to retry.
            print(
                f"{label}（{model_id}）获取失败：{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            return 1
        print(f"{label}已就绪", flush=True)
    _announce(total, total, "模型已全部就绪")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

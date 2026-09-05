"""Put a Hugging Face model on disk, and say how it should then be loaded.

Two stages need this and used to answer it twice: `vad_asr_stage` for Whisper,
`QwenReferee` for the second model. The bodies had already drifted once -- the
referee grew the offline gate below while the ASR loader kept asking the hub --
which is the whole argument for one function: the two questions are the same
question, and a fix that lands in one copy is not a fix.

The answer is a `HfLoad`: which revision to load, and whether the loader may
skip the network entirely. Both are arguments to somebody's `from_pretrained`.

`finesub_bootstrap` owns the facts (the manifest, the caches, the mirror
routing) and may not import this package, so the direction is one way: this
module asks it, wraps its answer in the shape the loaders want, and adds the
one thing bootstrap cannot know -- that a wrong answer must not be fatal.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, TypeVar

from ...reporting import current_reporter


class HfLoad(NamedTuple):
    """How to load weights this process has just made sure of."""

    #: The revision the manifest pins, so the loader gets the snapshot that was
    #: verified rather than whatever `main` points at today. `None` for a model
    #: the manifest does not describe.
    revision: Optional[str]
    #: Whether the pinned snapshot is complete in the cache root the loader
    #: itself resolves -- the only condition under which asking the hub for
    #: nothing is safe. See `offline_first` for what happens when it is wrong.
    local_files_only: bool


#: For a model nothing was ensured for: a local path, a custom repository, an
#: unlisted size. It keeps the behaviour it has always had -- its own library
#: fetches it, from wherever it likes.
UNMANAGED = HfLoad(None, False)


def prepare(manifest_id: str) -> HfLoad:
    """Ensure `manifest_id`'s weights, then describe how to load them.

    A `stat` when they are already there, which is every run after the first;
    that is what lets this sit at the top of a stage instead of in a prefetch.
    Never fatal on its own -- when this cannot fetch the weights the loader
    tries next and produces the error that actually describes what it wanted.

    One exception, and it is the opposite kind of failure. Everything else
    caught here means "we could not get the weights", which the loader is
    better placed to report. `VerificationMismatch` means "we got them and
    they are wrong": `ensure_hf_model` has already tried the mirror *and* the
    official source, hashed what landed, and disagreed with the manifest. No
    loader improves on that. Handing it a snapshot we have positively judged
    corrupt gets one of two answers, and both are worse than this one -- a
    model that loads and produces garbage, or CTranslate2's bare
    `RuntimeError`, which is the same class a CUDA failure raises and cannot
    be told apart from one.

    Raising is affordable precisely because the caller decides what it costs:
    the referee's own call site contains it (`vad_asr_stage`), so a corrupt
    second model under `--qwen-verify auto` loses the evidence rather than the
    run, while the same failure on the ASR weights ends a run that had nothing
    it could honestly produce anyway.
    """

    revision = None
    try:
        from finesub_bootstrap.model_ensure import pinned_revision

        revision = pinned_revision(manifest_id)
    except Exception as error:  # noqa: BLE001 - the loader reports for real
        _debug(manifest_id, error)
        return UNMANAGED
    # Outside the `try` on purpose: `pinned_revision` above already imported
    # `model_ensure`, which imports this module, so a failure here would be a
    # broken install rather than the absent-bootstrap case handled above --
    # and binding the name inside the block would make the handler itself
    # raise `NameError` on the one path where it matters.
    from finesub_bootstrap.hf_verify import VerificationMismatch

    try:
        from finesub.paths import resolve_managed_app_paths
        from finesub_bootstrap.model_ensure import ensure_hf_model

        paths = resolve_managed_app_paths()
        if paths is not None:
            ensure_hf_model(
                manifest_id,
                data_root=paths.data_root,
                models_root=paths.models,
                log=lambda message: current_reporter().debug(message),
            )
    except VerificationMismatch:
        raise
    except Exception as error:  # noqa: BLE001 - the loader reports for real
        _debug(manifest_id, error)
    # Asked after the fetch, and about the loader's own cache root rather than
    # about whether the fetch raised: weights that predate this mechanism, or
    # that came from the machine's existing Hugging Face cache, deserve the
    # offline load too -- and a fetch that wrote somewhere the loader does not
    # read must not earn it.
    local_only = False
    try:
        from finesub_bootstrap.model_ensure import pinned_snapshot_loadable

        local_only = pinned_snapshot_loadable(manifest_id)
    except Exception as error:  # noqa: BLE001 - the loader reports for real
        _debug(manifest_id, error)
    return HfLoad(revision, local_only)


T = TypeVar("T")


def offline_first(load: Callable[[HfLoad], T], plan: HfLoad, *, what: str) -> T:
    """Load with `plan`, and once more over the network if the local copy is
    what turned out to be missing.

    `local_files_only` is worth having because the loaders reach for the
    network even when every byte is already on disk: transformers lists the
    repository's chat templates on each `AutoProcessor.from_pretrained`, so a
    machine that cannot reach huggingface.co dies at a step that had nothing
    left to download. (Its own offline fallback catches `httpx.NetworkError`,
    and a connect timeout is `httpx.TimeoutException` -- the sibling class, not
    a subclass.)

    But the gate reads a cache directory, so it can be wrong, and this is the
    cheap half of not caring: a wrong gate costs one wasted attempt rather than
    the run.

    Only the cheap half, though, and deliberately. Retrying *any* failure would
    be worse than never trying offline: an out-of-memory, a CUDA or CTranslate2
    init failure, a model the installed library cannot read -- each would be
    attempted twice and then reported as whatever the second attempt hit, which
    on the very machine this feature is for is a network timeout burying the
    real cause. So the retry is for `OSError`: a missing or unreadable file,
    which is how both loaders report a snapshot that is short of what they
    wanted (`LocalEntryNotFoundError` is a `FileNotFoundError`; transformers
    raises a plain `OSError`).

    That leaves one state this cannot rescue, and it is why the gate does not
    lean on the retry: a snapshot whose *weights* are gone but whose small
    files remain. CTranslate2 opens `model.bin` itself and raises a bare
    `RuntimeError` -- the same class as a CUDA failure, with nothing to tell
    them apart. `pinned_snapshot_loadable` therefore stats the manifest's files
    before ever setting `local_files_only`, which is where that state has to be
    caught.
    """

    try:
        return load(plan)
    except OSError as error:
        if not plan.local_files_only:
            raise
        current_reporter().debug(
            f"{what}: the cached copy was not enough, retrying with the hub",
            {"error": f"{type(error).__name__}: {error}"},
        )
        # Raised from inside the handler on purpose: if the hub fails too, the
        # offline error stays attached as its context instead of vanishing.
        return load(plan._replace(local_files_only=False))


def _debug(manifest_id: str, error: BaseException) -> None:
    current_reporter().debug(
        f"{manifest_id} weights prefetch skipped",
        {"error": f"{type(error).__name__}: {error}"},
    )

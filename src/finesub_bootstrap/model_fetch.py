"""Fetching model weights through whichever entry point is configured.

Shared by both front ends on purpose. The desktop prefetches before the first
task; the CLI has no prefetch at all and downloads lazily inside the run. If
the endpoint were only chosen in the desktop's prefetch, half the users would
never get it -- so the decision lives here, and `RuntimeEnvironment.
worker_context` is where it reaches a run.

Two shapes matter:

* **The endpoint is an environment variable, so it must be set before the
  process starts.** `huggingface_hub` reads it at import time, so a fallback
  that flips it mid-process changes nothing. Retrying means a new process.
* **One process per model.** Three models in one process means a failure in
  the third re-downloads the first two on retry -- 1.6 GB of already-finished
  work thrown away because something else failed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path

from finesub_bootstrap import download_routes

#: The resource class these downloads are accounted under, for the per-machine
#: degrade rule.
RESOURCE_CLASS = "huggingface"

HF_ENDPOINT = "HF_ENDPOINT"

#: Appended to a model file's name to record that it was verified in full.
#:
#: The separator checkpoint is 639 MB and is checked before *every* separation,
#: which meant hashing it from disk on every run. The stamp records what the
#: verification saw -- size, mtime and the digest it matched -- so a file that
#: has not been touched since is taken as verified without being read again.
#: Anything that does not line up (missing stamp, edited file, a manifest that
#: now pins a different digest) falls back to the full check, so the guarantee
#: is unchanged and only the repetition is gone.
VERIFIED_SUFFIX = ".verified"


def hf_endpoint_for(data_root: Path, region: str) -> str:
    """The Hugging Face endpoint for this region, or "" for the official one."""

    return download_routes.active_mirror(data_root, RESOURCE_CLASS, region)


def apply_hf_endpoint(
    environment: dict[str, str],
    *,
    data_root: Path,
    region: str,
) -> dict[str, str]:
    """Point Hugging Face traffic at the configured endpoint, if any.

    A user who set `HF_ENDPOINT` themselves is left alone -- they pointed it
    somewhere on purpose, and a region guess is not a reason to overrule them.
    """

    if os.environ.get(HF_ENDPOINT):
        return environment
    endpoint = hf_endpoint_for(data_root, region)
    if endpoint:
        environment[HF_ENDPOINT] = endpoint
    return environment


def fetch_with_fallback(
    fetch: Callable[[Mapping[str, str]], None],
    *,
    base_environment: Mapping[str, str],
    data_root: Path,
    region: str,
    is_retryable: Callable[[BaseException], bool] | None = None,
) -> None:
    """Fetch once through the configured endpoint, then once through the official.

    Only a retryable failure earns the second attempt: a full disk is not a
    reason to re-download several GB from somewhere else. A verification
    mismatch *is* retryable, but only because its verifier holds up both ends
    of a bargain -- the corrupt bytes are removed before the retry, and the
    official result is verified in its turn. A mismatch dressed up as
    retryable without those two properties is how a corrupt file gets a second
    chance to be accepted.
    """

    environment = dict(base_environment)
    endpoint = "" if os.environ.get(HF_ENDPOINT) else hf_endpoint_for(data_root, region)
    if not endpoint:
        fetch(environment)
        return

    environment[HF_ENDPOINT] = endpoint
    try:
        fetch(environment)
    except BaseException as error:
        if is_retryable is not None and not is_retryable(error):
            raise
        download_routes.record_failure(data_root, RESOURCE_CLASS)
        official = dict(base_environment)
        official.pop(HF_ENDPOINT, None)
        fetch(official)
        return
    download_routes.record_success(data_root, RESOURCE_CLASS)


#: What a failure says when a host, rather than this machine, is at fault.
#: Needed because some of these failures cross a process boundary -- the
#: desktop prefetches in a subprocess, so the httpx exception never reaches
#: us, only its message does.
NETWORK_FAILURE_MARKERS = (
    "connection",
    "timed out",
    "timeout",
    "temporary failure",
    "network",
    "reset by peer",
    "unreachable",
    "name resolution",
    "ssl",
    "502",
    "503",
    "504",
    "429",
    "404",
)

#: Local trouble, stated explicitly so an unrecognised message cannot be
#: quietly blamed on a mirror.
LOCAL_FAILURE_MARKERS = (
    "no space left",
    "disk full",
    "permission denied",
    "access is denied",
    "out of memory",
)


def is_mirror_failure(error: BaseException) -> bool:
    """Whether the mirror could plausibly be responsible for this.

    A wrong or short body and a refused connection are the mirror's doing, and
    the official source is the fix. A full disk is not: retrying downloads
    several GB a second time to fail the same way, and recording it would
    degrade a mirror that did nothing wrong.

    Unrecognised failures are *not* blamed on the mirror. That costs a fallback
    we might have wanted, which is recoverable; the opposite spends gigabytes
    and disables a working mirror on evidence that never pointed at it.
    """

    import httpx

    from finesub_bootstrap.downloader import DownloadError, DownloadPaused
    from finesub_bootstrap.hf_verify import MISMATCH_MARKER, VerificationMismatch

    if isinstance(error, DownloadPaused):
        # The user stopped it. Starting again elsewhere is not a recovery.
        return False
    if isinstance(error, DownloadError):
        # A short or wrong body -- the mirror served something else.
        return True
    if isinstance(error, VerificationMismatch):
        # Length right, bytes wrong: HTTP called it a success and only the
        # manifest knows better. Safe to retry elsewhere because the verifier
        # already removed the corrupt files -- the fallback re-fetches them
        # rather than re-accepting them, and verifies its own result too.
        return True
    # Every httpx failure: a refused connection, a timeout, and a status the
    # proxy answered with. A GitHub proxy that 404s or 502s is the case this
    # fallback exists for, and `raise_for_status` reports it as an
    # HTTPStatusError, which is not a transport error at all.
    if isinstance(error, httpx.HTTPError):
        return True
    text = f"{error} {getattr(error, 'output', '') or ''}".lower()
    if any(marker in text for marker in LOCAL_FAILURE_MARKERS):
        return False
    if MISMATCH_MARKER in text:
        # The desktop verifies inside its prefetch subprocess, so the
        # VerificationMismatch above never crosses back -- only its words do.
        return True
    return any(marker in text for marker in NETWORK_FAILURE_MARKERS)


def _fetch_unverified(
    url: str, destination: Path, proxy: str, *, data_root: Path | None = None
) -> None:
    """Fetch a small file that carries no digest, mirror first, official after.

    With no digest to check, the body is validated by *parsing* it: a proxy
    that answers 200 with an interception page or a truncated document is
    indistinguishable from success at the HTTP layer, and writing that would
    leave audio-separator to die inside `json.load` on a file it now believes
    it already has -- with the mirror never recorded as having failed.

    Written through a temporary name, so a half-written index is never visible.
    Failing outright is fine: the library fetches it itself the first time it
    is needed, which is the behaviour without any of this.
    """

    import json

    import httpx

    from finesub_bootstrap.http_client import create_client, network_routes

    candidates = [(f"{proxy}{url}", True), (url, False)] if proxy else [(url, False)]
    route = network_routes()[0]
    for candidate, through_proxy in candidates:
        try:
            with create_client(route, timeout=httpx.Timeout(30.0)) as client:
                response = client.get(candidate)
                response.raise_for_status()
                body = response.content
            if destination.suffix == ".json":
                json.loads(body)
        except Exception:
            if through_proxy and data_root is not None:
                download_routes.record_failure(data_root, "github")
            continue
        temporary = destination.with_name(f".{destination.name}.part")
        try:
            temporary.write_bytes(body)
            os.replace(temporary, destination)
        except OSError:
            temporary.unlink(missing_ok=True)
        return


def verified_stamp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{VERIFIED_SUFFIX}")


def _stamp_still_describes(path: Path, expected) -> bool:
    """Whether an earlier full verification still speaks for this file."""

    try:
        status = path.stat()
        record = json.loads(
            verified_stamp_path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    return (
        record.get("sha256") == expected.sha256
        and record.get("size") == expected.size == status.st_size
        and record.get("mtime_ns") == status.st_mtime_ns
    )


def _write_verified_stamp(path: Path, expected) -> None:
    """Record that `path` verified in full, so the next run need not redo it.

    Best effort: a model directory that cannot be written to (a read-only
    share, a shared cache someone else owns) costs the full check again, which
    is exactly today's behaviour.
    """

    try:
        status = path.stat()
        verified_stamp_path(path).write_text(
            json.dumps(
                {
                    "sha256": expected.sha256,
                    "size": status.st_size,
                    "mtime_ns": status.st_mtime_ns,
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def fetch_fixed_files(
    entry,
    directory: Path,
    *,
    data_root: Path,
    region: str,
    progress=None,
) -> None:
    """Place a model that is a fixed list of URLs, verified against the manifest.

    Used for the separator checkpoint, which audio-separator otherwise fetches
    from GitHub itself -- outside `HF_ENDPOINT`, and outside the region
    fallback every other download goes through. Separating "download" from
    "load" is what lets the same verification apply.

    A file that is already here and already matches is left alone. A mismatch
    is never patched over: the quarantined copy stays for inspection and the
    official source is asked for a whole new one, because splicing bytes from
    two sources produces something neither of them signed.
    """

    from finesub_bootstrap.downloader import DigestMismatch, download_asset
    from finesub_bootstrap.models import DownloadAsset
    from finesub_bootstrap.model_manifest import file_matches

    proxy = download_routes.active_mirror(data_root, "github", region)
    directory.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda _progress: None)
    for wanted in entry.files:
        destination = directory / wanted.name
        if destination.is_file():
            if wanted.is_verifiable and _stamp_still_describes(destination, wanted):
                continue
            if file_matches(destination, wanted):
                # Stamped here too, not only after a download: an install that
                # predates the stamp -- or one whose files came from a shared
                # cache -- gets it on its first full check rather than paying
                # for one forever.
                if wanted.is_verifiable:
                    _write_verified_stamp(destination, wanted)
                continue
        if not wanted.is_verifiable:
            # No digest, so the verified downloader cannot carry it -- that one
            # exists to refuse exactly what this file cannot promise. Fetched
            # plainly instead, and still worth routing: the separator's index
            # is 28 KB from the single hardest-to-reach host in this path, and
            # without it a present checkpoint separates nothing.
            _fetch_unverified(wanted.url, destination, proxy, data_root=data_root)
            continue
        asset = DownloadAsset(
            url=f"{proxy}{wanted.url}" if proxy else wanted.url,
            size=wanted.size,
            sha256=wanted.sha256,
        )
        try:
            download_asset(asset, destination, report)
        except Exception as error:
            if not proxy or not is_mirror_failure(error):
                raise
            download_routes.record_failure(data_root, "github")
            if isinstance(error, DigestMismatch):
                # The mirror served something else. Start over from the
                # official URL rather than resuming into a quarantined body.
                destination.unlink(missing_ok=True)
            download_asset(
                DownloadAsset(
                    url=wanted.url, size=wanted.size, sha256=wanted.sha256
                ),
                destination,
                report,
            )
        else:
            if proxy:
                download_routes.record_success(data_root, "github")
        # Reached only when the download returned without raising, which for a
        # verifiable file means the downloader matched the digest itself.
        _write_verified_stamp(destination, wanted)

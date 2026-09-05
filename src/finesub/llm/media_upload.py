"""The media reference a window clip is carried by, and how it gets there.

Two things live here because they are one decision: what an
:class:`UploadedFileRef` is, and the Gemini Files API upload that produces the
remote form of it. They came out of ``client.py`` (2026-09-03), which had grown
past 2400 lines while the resumable-upload protocol -- three network stages,
each with its own retry budget -- has nothing to do with calling a model.

``client.py`` imports this module, never the other way round. The boundary is
**model selection**, not routing as a whole: nothing here knows about role
clients, transports, or which candidate answers a call. It does read two
routing facts, deliberately and only these:

* which API key uploads (``routing.api_keys``). A Files object lives in the
  project of the key that created it and 403s for any other, so the uploader
  has to record its own key -- and the eager upload happens before any
  candidate is known, so there is nobody upstream to be told by.
* which carrier a policy implies (:func:`window_media_ref`): "keep it local for
  an agent" vs "upload it for an API candidate" is the same question as which
  of the two constructors below to call, so splitting it out would put half a
  decision in each module.

Anything beyond those two belongs to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import mimetypes
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable, Mapping

import httpx

from finesub.reporting import current_reporter
from .http import llm_http_client
from .routing import api_keys


@dataclass(frozen=True)
class UploadedFileRef:
    file_id: str
    filename: str
    mime_type: str
    local_path: str = ""
    # The local clip already owns agy's required 0.25 fps sampling, so the
    # driver may copy it into the capsule instead of encoding it again.
    agy_prepared: bool = False
    # Exact media duration when the caller cut this clip and therefore knows it.
    # Window clips are raw ADTS, which carries no container duration: ffprobe
    # has to guess a bitrate from the opening frames, and a window that starts
    # on quiet audio has been measured at 22x its real length. That guess feeds
    # the input-token estimate, so it can strike every candidate off the chain
    # as ``input_limit``. 0.0 means "unknown, fall back to probing".
    duration_seconds: float = 0.0
    # Which key uploaded this Files object -- a fact about the file, because
    # the object lives in that key's project and any other key 403s on it.
    #
    # Recording it is the fix for a bug that came back three times in one
    # afternoon (2026-08-19): the owning key was never written down, so the
    # upload side and the call side each *re-derived* it, and every time the
    # two derivations drifted apart -- key #1 vs first-unlocked, lock-aware vs
    # not, cooldown-aware vs not -- the call read a file it had no access to.
    # A derived fact can disagree with itself; a recorded one cannot. Empty
    # means a local-only ref (nothing uploaded yet).
    api_key_id: str = ""
    # The pool/tier that owns the key above. Key names are only unique inside
    # a pool (both FREE and PAID may legitimately call one ``main``), so the
    # id alone cannot prove that a candidate can read this Files object.
    api_provider_tier: str = ""

    @property
    def is_audio(self) -> bool:
        return self.mime_type.strip().lower().startswith("audio/")

    @property
    def is_video(self) -> bool:
        return self.mime_type.strip().lower().startswith("video/")


def local_media_file_ref(
    path: str | Path, *, agy_prepared_video: bool = False
) -> UploadedFileRef:
    """A media reference that needs no Gemini Files API upload.

    ``agent-only`` uses this form so clipping does not accidentally require a
    Gemini API key before the local agy candidate is even considered.
    """

    local = Path(path).expanduser().resolve()
    mime_type = mimetypes.guess_type(local.name)[0] or "application/octet-stream"
    return UploadedFileRef(
        file_id="",
        filename=local.name,
        mime_type=mime_type,
        local_path=str(local),
        agy_prepared=bool(agy_prepared_video and mime_type.startswith("video/")),
    )


def with_media_duration(ref: UploadedFileRef, seconds: float) -> UploadedFileRef:
    """Record the exact duration of a clip the caller cut itself."""

    seconds = float(seconds or 0.0)
    if seconds <= 0 or ref.duration_seconds == seconds:
        return ref
    return replace(ref, duration_seconds=seconds)


def mark_agy_prepared_video(ref: UploadedFileRef) -> UploadedFileRef:
    if not ref.is_video or ref.agy_prepared:
        return ref
    return replace(ref, agy_prepared=True)


# Policies that let a local agent answer at all. Necessary but no longer
# sufficient: a policy only subtracts backends, so the bound groups have to
# name an agent too -- see ``binds_local_agent``.
AGENT_CAPABLE_POLICY_IDS = frozenset({"agent-only", "agent-text-preferred"})


def window_media_ref(
    path: str | Path,
    *,
    execution_settings: Any | None = None,
    routes: Any | None = None,
    cancel: threading.Event | None = None,
) -> UploadedFileRef:
    """The reference one window clip should be carried by under this policy.

    Correction and fast mode need exactly this decision, and writing it out in
    both places meant the next policy id -- or the next change to the
    agy-prepared rule -- had two sites to get right, with "silently uploads to
    Gemini under an agent policy" as the failure mode.

    ``routes`` must be the same catalog the client's router will plan from --
    pass ``client.router.routes``. Falling back to the global one would let the
    table that decides *how the clip is carried* disagree with the table that
    decides *who answers*, and then a clip is either uploaded for an agent that
    never needed it or kept local for a chain that only has API candidates.
    """

    from .routing.model_routes import DEFAULT_EXECUTION_POLICY, default_model_routes

    local = Path(path)
    policy_id = str(
        getattr(execution_settings, "policy_id", "") or DEFAULT_EXECUTION_POLICY
    )
    agent_reachable = (
        policy_id in AGENT_CAPABLE_POLICY_IDS
        and (routes or default_model_routes()).binds_local_agent()
    )
    ref = (
        local_media_file_ref(local)
        if agent_reachable
        else upload_gemini_file(local, cancel=cancel)
    )
    if ref.is_video:
        # Clips are cut at agy's required sampling rate whoever answers, so
        # this is a fact about the file rather than a routing decision.
        ref = mark_agy_prepared_video(ref)
    return ref


def upload_gemini_file(
    path: str | Path,
    *,
    api_key: str | None = None,
    cancel: threading.Event | None = None,
) -> UploadedFileRef:
    file_path = Path(path).expanduser().resolve()
    if api_key is not None:
        return _upload_gemini_file_rest(file_path, api_key=api_key, cancel=cancel)

    # Eager uploads happen before routing knows a candidate. Preserve both
    # pieces of the chosen entry's canonical identity so a later candidate can
    # decide whether to reuse or re-upload the object safely.
    from . import llm_runtime

    entry, provider_tier = api_keys.first_enabled_gemini_entry(
        llm_runtime._read_dotenv()
    )
    remote = _upload_gemini_file_rest(file_path, api_key=entry.key, cancel=cancel)
    return replace(
        remote,
        api_key_id=entry.key_id,
        api_provider_tier=provider_tier,
    )

# The Files API can report ACTIVE while the media is still being prepared for
# sampling: generateContent issued seconds after a video upload has returned
# HTTP 200 with zero output and text-only billing (observed 2026-07-11).
# countTokens is free (auth-only), so it doubles as an exact readiness probe:
# once the file's media tokens are actually counted, generateContent sees the
# media too.
GEMINI_MEDIA_PROBE_MODEL = "gemini-3.1-flash-lite"


def _wait_for_media_tokens(
    client: Any,
    *,
    api_base: str,
    auth_header: Mapping[str, str],
    file_uri: str,
    mime_type: str,
    sleep_func: Callable[[float], None],
    poll_interval_seconds: float,
    max_poll_attempts: int,
    request: Callable[[Callable[[], Any]], Any] = lambda op: op(),
    wait: Callable[[float], None] | None = None,
) -> None:
    wait = wait or sleep_func
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"fileData": {"fileUri": file_uri, "mimeType": mime_type}}],
            }
        ]
    }
    url = f"{api_base}/models/{GEMINI_MEDIA_PROBE_MODEL}:countTokens"
    def probe_once():
        response = client.post(
            url,
            headers={**auth_header, "Content-Type": "application/json"},
            json=body,
        )
        # 400 is how countTokens says "not sampled yet" (observed 2026-07-11)
        # and is the readiness signal this loop waits on; every other error
        # is the request's own and classified like any upload request.
        if response.status_code != 400:
            response.raise_for_status()
        return response

    for _ in range(max_poll_attempts):
        response = request(probe_once)
        if response.status_code == 200:
            try:
                total = int(response.json().get("totalTokens") or 0)
            except (TypeError, ValueError, AttributeError):
                total = 0
            if total > 0:
                return
        wait(poll_interval_seconds)
    raise TimeoutError(
        f"Gemini media never became countable after upload: {file_uri}"
    )


class UploadCancelled(RuntimeError):
    """The owner of this upload stopped waiting for it (prefetch teardown)."""


# Upload retries are the Files API's own budget, separate from the model-call
# budget (`RoleClient(max_retries=...)`): that one only starts once a media ref
# exists, so a reset connection during the upload used to end the LLM stage on
# the first try. Classification is by exception *type*, not message text --
# `httpx.ReadError: [Errno 10054] 远程主机强迫关闭了一个现有的连接` carries
# none of the words the model-call classifier looks for.
GEMINI_UPLOAD_MAX_ATTEMPTS = 3
_UPLOAD_BACKOFF_SECONDS = (1.0, 3.0)
_UPLOAD_RETRY_AFTER_CAP_SECONDS = 30.0
_UPLOAD_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_UPLOAD_RETRYABLE_TRANSPORT = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)
# Per-operation limits, not a ceiling on the whole upload. Connect is short so
# an unreachable host is noticed quickly; write and read stay long because a
# window clip upload is one request and the finalize response waits on the
# server.
#
# ⚠ The connect budget is spent *per resolved address*, not per attempt, and
# `generativelanguage.googleapis.com` resolves to several -- httpx walks them
# in turn. So the wall-clock cost of a host that is filtered rather than down
# is this value times the address count before the first retry line is even
# printed: at the 45s this used to hold, a measured run sat silent for 169
# seconds, which reads as a hang rather than as a network fault. That is the
# argument for 15 and it points the opposite way from the one that used to be
# written here (that a slow proxy might want it raised): raising it multiplies
# the silence. A connect that needs more than 15s over a working link is not
# worth holding the whole stage open for.
GEMINI_UPLOAD_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=600.0, pool=45.0)


def _upload_error_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _UPLOAD_RETRYABLE_STATUS
    return isinstance(exc, _UPLOAD_RETRYABLE_TRANSPORT)


def _upload_failure_label(exc: BaseException) -> str:
    # Status errors quote the request URL, which for the finalize step is the
    # resumable session URL -- a capability token. Keep it out of the log.
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return f"{type(exc).__name__}: {str(exc)[:120]}"


def _upload_retry_delay(exc: BaseException, attempt: int) -> float:
    # Only the delta-seconds form of Retry-After is read; an HTTP-date (or
    # anything else) falls back to the fixed ladder.
    fallback = _UPLOAD_BACKOFF_SECONDS[min(attempt - 1, len(_UPLOAD_BACKOFF_SECONDS) - 1)]
    jitter = random.uniform(0.0, 0.5)
    if isinstance(exc, httpx.HTTPStatusError):
        header = exc.response.headers.get("retry-after")
        try:
            hinted = float(header) if header is not None else None
        except (TypeError, ValueError):
            hinted = None
        if hinted is not None and 0 <= hinted <= _UPLOAD_RETRY_AFTER_CAP_SECONDS:
            return hinted + jitter
    return fallback + jitter


def _upload_wait(
    seconds: float,
    *,
    cancel: threading.Event | None,
    sleep_func: Callable[[float], None],
    stage: str,
) -> None:
    """Sleep, unless an owner has cancelled -- then stop at once."""

    if cancel is None:
        sleep_func(seconds)
    elif cancel.wait(seconds):
        raise UploadCancelled(f"Gemini upload cancelled while waiting in {stage}")


def _with_upload_retries(
    operation: Callable[[], Any],
    *,
    stage: str,
    cancel: threading.Event | None,
    sleep_func: Callable[[float], None],
    max_attempts: int,
) -> Any:
    """Run one network operation with the upload budget; raise the last error.

    Each attempt re-runs ``operation`` from scratch, so for the upload stage
    that means a fresh resumable session (the old one is in an unknown state
    after a reset). The final exception is the last one seen, unwrapped, so
    callers keep classifying by type.
    """

    for attempt in range(1, max_attempts + 1):
        if cancel is not None and cancel.is_set():
            raise UploadCancelled(f"Gemini upload cancelled before {stage}")
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not _upload_error_retryable(exc):
                if attempt > 1:
                    current_reporter().warning(
                        "gemini-upload-failed",
                        f"Gemini upload {stage} failed after {attempt} attempts "
                        f"({_upload_failure_label(exc)}).",
                    )
                raise
            delay = _upload_retry_delay(exc, attempt)
            current_reporter().warning(
                "gemini-upload-retry",
                f"Gemini upload {stage} failed ({_upload_failure_label(exc)}), "
                f"attempt {attempt}/{max_attempts}; retrying in {delay:.1f}s.",
            )
            _upload_wait(delay, cancel=cancel, sleep_func=sleep_func, stage=stage)
    raise AssertionError("unreachable")


def _upload_gemini_file_rest(
    file_path: Path,
    *,
    api_key: str,
    client_factory: Callable[..., Any] = llm_http_client,
    sleep_func: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = 2.0,
    max_poll_attempts: int = 60,
    probe_poll_attempts: int = 150,
    cancel: threading.Event | None = None,
    max_attempts: int = GEMINI_UPLOAD_MAX_ATTEMPTS,
) -> UploadedFileRef:
    """Upload media to Gemini Files API using the resumable REST protocol.

    Three network stages, each with its own retry budget of ``max_attempts``:
    the upload (start + send + finalize, re-run whole on failure), then each
    state poll, then each countTokens probe. A normal PROCESSING wait is not a
    failure and spends none of it.
    """

    from .rate_limit import key_id_for_secret

    data = file_path.read_bytes()
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    upload_base = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    api_base = "https://generativelanguage.googleapis.com/v1beta"
    auth_header = {"x-goog-api-key": api_key}

    def retrying(operation: Callable[[], Any], stage: str) -> Any:
        return _with_upload_retries(
            operation,
            stage=stage,
            cancel=cancel,
            sleep_func=sleep_func,
            max_attempts=max_attempts,
        )

    with client_factory(timeout=GEMINI_UPLOAD_TIMEOUT) as client:

        def upload_once() -> dict:
            start = client.post(
                upload_base,
                headers={
                    **auth_header,
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(len(data)),
                    "X-Goog-Upload-Header-Content-Type": mime_type,
                    "Content-Type": "application/json",
                },
                json={"file": {"display_name": file_path.name}},
            )
            start.raise_for_status()
            upload_url = start.headers.get("x-goog-upload-url")
            if not upload_url:
                raise RuntimeError("Gemini Files API did not return x-goog-upload-url.")

            finalize = client.post(
                upload_url,
                headers={
                    "Content-Length": str(len(data)),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                content=data,
            )
            finalize.raise_for_status()
            return finalize.json().get("file", {})

        file_obj = retrying(upload_once, "upload")
        name = file_obj.get("name")
        if not name:
            raise RuntimeError("Gemini Files API response did not include file.name.")

        def poll_state() -> dict:
            status = client.get(f"{api_base}/{name}", headers=auth_header)
            status.raise_for_status()
            return status.json()

        for _ in range(max_poll_attempts):
            state = str(file_obj.get("state") or "").upper()
            if state in {"", "ACTIVE"}:
                break
            if state == "FAILED":
                raise RuntimeError(f"Gemini file processing failed: {file_obj}")
            _upload_wait(
                poll_interval_seconds,
                cancel=cancel,
                sleep_func=sleep_func,
                stage="state_poll",
            )
            file_obj = retrying(poll_state, "state_poll")
        else:
            raise TimeoutError(f"Gemini file did not become ACTIVE: {name}")

        file_id = file_obj.get("uri") or name
        final_mime = file_obj.get("mimeType") or mime_type
        if final_mime.split("/", 1)[0] in {"audio", "video"}:
            _wait_for_media_tokens(
                client,
                api_base=api_base,
                auth_header=auth_header,
                file_uri=file_id,
                mime_type=final_mime,
                sleep_func=sleep_func,
                poll_interval_seconds=poll_interval_seconds,
                max_poll_attempts=probe_poll_attempts,
                request=lambda op: retrying(op, "token_poll"),
                wait=lambda seconds: _upload_wait(
                    seconds, cancel=cancel, sleep_func=sleep_func, stage="token_poll"
                ),
            )

    return UploadedFileRef(
        file_id=file_id,
        filename=file_path.name,
        mime_type=final_mime,
        local_path=str(file_path),
        api_key_id=key_id_for_secret(api_key),
    )

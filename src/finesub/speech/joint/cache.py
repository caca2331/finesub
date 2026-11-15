"""Raw provider answers cached in the task directory, one file per call.

`<stem>-cloud/<provider>/<call>.raw.json`: a call whose file exists is never
made again, so deleting `-aligned.json` and rerunning spends no quota; deleting
the cache is what asks the provider again. The directory also holds
`identity.json` (provider, model, windowing parameters, the input audio's
fingerprint). The audio is the compatibility key: the calls are named by their
window's times, so another recording's answers would be pasted onto this one
by time. A different audio discards the directory's answers, which are then
asked again; the directory keeps one recording's, never piling up. Any other
difference only warns: the cached answers are neither corrupt nor unreadable,
and serving them is the task keeping its own earlier choice
(`README_DEV.md`「复用的依据是任务身份，不是质量判断」).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

from finesub.reporting import current_reporter

IDENTITY_FILE = "identity.json"
#: The identity field that invalidates rather than warns (see the module docstring).
AUDIO = "audio"
_EDGE_BYTES = 1 << 20
#: Repairs store from several threads at once (35t Live); the first answers would race on `identity.json`.
_IDENTITY_LOCK = threading.Lock()


class CacheMiss(RuntimeError):
    """A call the cache does not hold, asked for while calls are not allowed (replay)."""


def audio_fingerprint(path: str | Path) -> str:
    """Size plus a hash of the first and last MiB: cheap on hour-long files, and a
    different recording practically never shares both ends and the size."""

    p = Path(path)
    size = p.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with p.open("rb") as f:
        digest.update(f.read(_EDGE_BYTES))
        if size > 2 * _EDGE_BYTES:
            f.seek(size - _EDGE_BYTES)
            digest.update(f.read(_EDGE_BYTES))
    return f"{size}:{digest.hexdigest()[:16]}"


class CallCache:
    def __init__(self, root: str | Path, provider: str, *, identity: Mapping[str, Any], offline: bool = False):
        self.dir = Path(root) / provider
        self.identity = dict(identity)
        self.offline = offline
        self._checked = False
        self._check_lock = threading.Lock()
        self._other_audio = False  # offline, over another recording's answers: none of them is served
        self._names: dict[str, threading.Lock] = {}
        self._names_lock = threading.Lock()

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.raw.json"

    def load(self, name: str) -> dict[str, Any] | None:
        self._check_identity()
        path = self.path(name)
        if self._other_audio or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def fetch(self, name: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """The cached answer for `name`, or `call()`'s, stored before it is returned. Threads asking for one
        name take turns: the later ones read what the first stored (two repairs can share a Live window)."""

        with self._names_lock:
            lock = self._names.setdefault(name, threading.Lock())
        with lock:
            cached = self.load(name)
            if cached is not None:
                return cached
            if self.offline:
                held = " (the cache holds another recording's answers)" if self._other_audio else ""
                raise CacheMiss(f"{self.path(name)} is not cached{held} and calls are off")
            body = call()
            self._store(name, body)
            return body

    def _store(self, name: str, body: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        identity = self.dir / IDENTITY_FILE
        with _IDENTITY_LOCK:
            if not identity.exists():
                _write_atomic(identity, self.identity)
        self._check_identity()
        _write_atomic(self.path(name), body)

    def _check_identity(self) -> None:
        with self._check_lock:  # threads wait for the discard below before reading or storing
            if not self._checked:
                self._compare_identity()
                self._checked = True

    def _compare_identity(self) -> None:
        path = self.dir / IDENTITY_FILE
        if not path.exists():
            return
        recorded = json.loads(path.read_text(encoding="utf-8"))
        before, now = recorded.get(AUDIO), self.identity.get(AUDIO)
        if before is not None and now is not None and before != now:
            if self.offline:  # a replay stores nothing, so it deletes nothing either
                self._other_audio = True
                return
            answers = list(self.dir.glob("*.raw.json"))
            for stale in answers:
                stale.unlink()
            path.unlink()
            current_reporter().warning(
                "cloud-cache-audio",
                f"{self.dir} held answers for another audio ({before}, now {now}); its {len(answers)} cached "
                "answers are discarded and asked again",
                impact="源音频换了，缓存的回答作废、重新请求（花额度）",
            )
            return
        changed = sorted(k for k in set(recorded) | set(self.identity) if recorded.get(k) != self.identity.get(k))
        if changed:
            current_reporter().warning(
                "cloud-cache-identity",
                f"{self.dir} was made with different {', '.join(changed)}; its cached answers are used "
                "as they are -- delete the directory to ask the provider again",
                impact="识别结果来自先前的设置",
            )


def _write_atomic(path: Path, data: Any) -> None:
    temporary = path.with_name(f"{path.name}.{threading.get_ident()}.part")  # identity.json: one per cache
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temporary, path)

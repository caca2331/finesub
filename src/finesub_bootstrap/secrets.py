"""Machine-bound protection for API keys in ``.env``.

Envelope encryption: DPAPI (bound to the current Windows account, with app
entropy) wraps a random 32-byte master key stored as the ``FINESUB_KEYRING``
line of the ``.env`` file itself, and every API key value is sealed with a
stream cipher + MAC derived from that master. Only the key material is
replaced in place -- names, braces, quotes, commas, comments and line endings
are preserved byte for byte, so the file stays diffable and `config.toml`'s
``[pools]`` selectors can still be checked against it by eye.

What this protects against: accidental sharing of the file, generic API-key
regex scans, generic DPAPI-blob harvesting (the single blob is obfuscated and
uses entropy such tools do not know), and other accounts on the same machine.
What it cannot protect against: any code running as the current user -- the
program must decrypt without user input, so the material is reachable. Wording
elsewhere must say "绑定当前 Windows 账户的保护", never unqualified "加密".

Two permanent constraints keep this importable from a plain ``[harness]``
install (which lacks pydantic): this module is stdlib-only, and the package
``__init__`` stays free of imports. A guard test in ``test/test_secrets.py``
enforces both.

This module is also the only ``.env`` line-level parser/writer in the project
(read via `read_env_file`, write via `update_env_file`). Do not grow another:
a second parser that misses one decryption or splitting rule silently feeds
garbage keys to the APIs.
"""

from __future__ import annotations

import binascii
import ctypes
import hashlib
import hmac
import os
import re
import sys
from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from finesub_bootstrap.locks import LockUnavailable, holding_lock

KEYRING_NAME = "FINESUB_KEYRING"
KEK_PREFIX = "finesub$kek$v1$"
VALUE_PREFIX = "fs$"

# Two orthogonal uses of the same fixed string: DPAPI entropy (a generic
# harvester that finds the blob still cannot decrypt it) and the keystream
# for the keyring obfuscation layer (so the blob is not found at all).
_ENTROPY = b"finesub"

_MASTER_LEN = 32
_KEK_SALT_LEN = 16
_VALUE_SALT_LEN = 12
_TAG_LEN = 8

# The ciphertext alphabet is base64url without padding on purpose: it shares
# no character with the container syntax (: , " { }), so a value that failed
# to decrypt can never be silently re-split into garbage named keys.
_TOKEN_RE = re.compile(r"fs\$[A-Za-z0-9_-]+")

_LOCK_TIMEOUT_SECONDS = 10.0

_REVEAL_HINT = (
    "在原机器上用 `finesub keys --reveal` 导出明文后重填；"
    "重填或删除全部不可解密的值后，保护会自动恢复。"
)


class ProtectionUnavailable(RuntimeError):
    """Wrapping is impossible here: not Windows, or crypt32 is unusable."""


class SecretUnreadable(ValueError):
    """A ciphertext that cannot be opened: other machine/account, or corrupt."""


def _b64encode(data: bytes) -> str:
    return urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    return urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _xor(data: bytes, pad: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, pad))


# --- KEK backend ----------------------------------------------------------


class KekBackend(Protocol):
    def wrap(self, data: bytes) -> bytes: ...

    def unwrap(self, blob: bytes) -> bytes: ...


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.c_void_p)]


_CRYPTPROTECT_UI_FORBIDDEN = 0x01


class DpapiBackend:
    """CryptProtectData/CryptUnprotectData for the current Windows user.

    CryptProtectData is non-deterministic (internal random IV), which is why
    the design wraps a random master key instead of deriving one from a fixed
    string.
    """

    def __init__(self) -> None:
        if os.name != "nt":
            raise ProtectionUnavailable("DPAPI is only available on Windows")
        try:
            self._crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except OSError as error:
            raise ProtectionUnavailable(f"crypt32 unavailable: {error}") from error
        self._kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel32.LocalFree.restype = ctypes.c_void_p
        blob_pointer = ctypes.POINTER(_DataBlob)
        for name in ("CryptProtectData", "CryptUnprotectData"):
            function = getattr(self._crypt32, name)
            function.argtypes = [
                blob_pointer,
                ctypes.c_wchar_p,
                blob_pointer,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                blob_pointer,
            ]
            function.restype = ctypes.c_int

    def _call(self, function, data: bytes) -> bytes:
        buffer = ctypes.create_string_buffer(data, len(data))
        entropy_buffer = ctypes.create_string_buffer(_ENTROPY, len(_ENTROPY))
        blob_in = _DataBlob(
            len(data), ctypes.cast(buffer, ctypes.c_void_p)
        )
        blob_entropy = _DataBlob(
            len(_ENTROPY), ctypes.cast(entropy_buffer, ctypes.c_void_p)
        )
        blob_out = _DataBlob()
        succeeded = function(
            ctypes.byref(blob_in),
            None,
            ctypes.byref(blob_entropy),
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(blob_out),
        )
        if not succeeded:
            raise OSError(ctypes.get_last_error() or "DPAPI call failed")
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            self._kernel32.LocalFree(blob_out.pbData)

    def wrap(self, data: bytes) -> bytes:
        try:
            return self._call(self._crypt32.CryptProtectData, data)
        except OSError as error:
            raise ProtectionUnavailable(f"DPAPI wrap failed: {error}") from error

    def unwrap(self, blob: bytes) -> bytes:
        try:
            return self._call(self._crypt32.CryptUnprotectData, blob)
        except OSError as error:
            # Wrong machine, wrong account, or a corrupted blob -- for the
            # caller these are all "this ciphertext cannot be opened here".
            raise SecretUnreadable(f"DPAPI unwrap failed: {error}") from error


_INJECTED_BACKEND: KekBackend | None = None
_DEFAULT_BACKEND: DpapiBackend | None = None


def set_backend(backend: KekBackend | None) -> None:
    """Inject a fake backend for tests; ``None`` restores the default."""

    global _INJECTED_BACKEND, _DEFAULT_BACKEND
    _INJECTED_BACKEND = backend
    _DEFAULT_BACKEND = None
    _MASTER_CACHE.clear()
    _MASTER_FAILURES.clear()
    _ENSURED.clear()
    _WARNED.clear()


def _backend() -> KekBackend:
    global _DEFAULT_BACKEND
    if _INJECTED_BACKEND is not None:
        return _INJECTED_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = DpapiBackend()
    return _DEFAULT_BACKEND


def available() -> bool:
    if _INJECTED_BACKEND is not None:
        return True
    if os.name != "nt":
        return False
    try:
        _backend()
    except ProtectionUnavailable:
        return False
    return True


# --- Envelope ------------------------------------------------------------


def protection_enabled() -> bool:
    """Whether automatic ``.env`` protection is on (the default).

    ``FINESUB_ENV_PROTECT=0`` keeps the file plaintext: a transition hatch for
    setups where other readers still parse ``.env`` directly -- e.g. worktrees
    running code that predates this module, which would read ciphertext as
    garbage keys. Only the automatic encryption paths stand down; reading
    already-protected values still works. Migration 0004 keeps reporting
    "not done", so the conversion happens at the first start after the
    variable is removed.
    """

    configured = os.environ.get("FINESUB_ENV_PROTECT")
    if configured is None:
        return True
    return configured.strip().lower() not in {"0", "false", "no", "off", ""}


def new_master() -> bytes:
    return os.urandom(_MASTER_LEN)


def _obfuscate(blob: bytes, salt: bytes) -> bytes:
    """XOR the DPAPI blob with a keystream. Not a security boundary.

    Its only purpose is to erase the well-known DPAPI constant prefix
    (``AQAAANCMnd8...`` after base64) so disk-scanning tools do not find the
    blob in the first place. XOR is an involution: this is also the inverse.
    """

    pad = hashlib.shake_256(_ENTROPY + salt).digest(len(blob))
    return _xor(blob, pad)


_deobfuscate = _obfuscate


def wrap_master(master: bytes) -> str:
    salt = os.urandom(_KEK_SALT_LEN)
    blob = _backend().wrap(master)
    return KEK_PREFIX + _b64encode(salt + _obfuscate(blob, salt))


def unwrap_master(token: str) -> bytes:
    if not token.startswith(KEK_PREFIX):
        raise SecretUnreadable("keyring line has an unknown format")
    try:
        raw = _b64decode(token[len(KEK_PREFIX):])
    except (binascii.Error, ValueError) as error:
        raise SecretUnreadable(f"keyring line is corrupt: {error}") from error
    if len(raw) <= _KEK_SALT_LEN:
        raise SecretUnreadable("keyring line is truncated")
    salt, obfuscated = raw[:_KEK_SALT_LEN], raw[_KEK_SALT_LEN:]
    master = _backend().unwrap(_deobfuscate(obfuscated, salt))
    if len(master) != _MASTER_LEN:
        raise SecretUnreadable("keyring line does not hold a master key")
    return master


def _keystream(master: bytes, salt: bytes, length: int) -> bytes:
    return hashlib.shake_256(master + b"|v1|" + salt).digest(length)


def _tag(master: bytes, salt: bytes, ciphertext: bytes) -> bytes:
    # person= separates the MAC use of the master from the keystream use.
    return hashlib.blake2b(
        salt + ciphertext, key=master, person=b"fs-mac", digest_size=_TAG_LEN
    ).digest()


def seal(plaintext: str, master: bytes) -> str:
    salt = os.urandom(_VALUE_SALT_LEN)
    data = plaintext.encode("utf-8")
    ciphertext = _xor(data, _keystream(master, salt, len(data)))
    tag = _tag(master, salt, ciphertext)
    return VALUE_PREFIX + _b64encode(salt + tag + ciphertext)


def open_(token: str, master: bytes) -> str:
    if not token.startswith(VALUE_PREFIX):
        raise SecretUnreadable("not a sealed value")
    try:
        raw = _b64decode(token[len(VALUE_PREFIX):])
    except (binascii.Error, ValueError) as error:
        raise SecretUnreadable(f"sealed value is corrupt: {error}") from error
    if len(raw) < _VALUE_SALT_LEN + _TAG_LEN:
        raise SecretUnreadable("sealed value is truncated")
    salt = raw[:_VALUE_SALT_LEN]
    tag = raw[_VALUE_SALT_LEN:_VALUE_SALT_LEN + _TAG_LEN]
    ciphertext = raw[_VALUE_SALT_LEN + _TAG_LEN:]
    if not hmac.compare_digest(tag, _tag(master, salt, ciphertext)):
        # Without this check a corrupt ciphertext would silently decrypt to a
        # garbage key and fail much later, at the API, pointing nowhere.
        raise SecretUnreadable("sealed value failed authentication")
    return _xor(ciphertext, _keystream(master, salt, len(ciphertext))).decode(
        "utf-8"
    )


# --- .env file layer -------------------------------------------------------


def is_protected(value: str) -> bool:
    return value.startswith(VALUE_PREFIX)


@dataclass(slots=True)
class _EnvLine:
    body: str  # without the line terminator
    ending: str  # "\n", "\r\n" or "" for the last line
    name: str | None = None  # None for comments/blank/parse-skipped lines
    rhs_start: int = 0  # offset of the RHS within body (after '=')

    @property
    def rhs(self) -> str:
        return self.body[self.rhs_start:]


def _parse_lines(text: str) -> list[_EnvLine]:
    lines: list[_EnvLine] = []
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        ending = raw[len(body):]
        stripped = body.strip()
        if not stripped or stripped.startswith("#") or "=" not in body:
            lines.append(_EnvLine(body, ending))
            continue
        eq = body.index("=")
        name = body[:eq].strip()
        lines.append(_EnvLine(body, ending, name or None, eq + 1))
    return lines


def _trimmed(rhs: str, start: int, end: int) -> tuple[int, int]:
    while start < end and rhs[start].isspace():
        start += 1
    while end > start and rhs[end - 1].isspace():
        end -= 1
    return start, end


def _strip_one_quote_layer(rhs: str, start: int, end: int) -> tuple[int, int]:
    if end - start >= 2 and rhs[start] == rhs[end - 1] and rhs[start] in "\"'":
        return start + 1, end - 1
    return start, end


def _value_spans(rhs: str) -> list[tuple[int, int]]:
    """Offsets of the secret material inside a raw RHS string.

    Mirrors the splitting of ``api_keys.parse_key_map`` / ``parse_key_list``
    (split items on every comma, value is what follows the first colon, one
    layer of paired quotes comes off), recording offsets instead of building
    strings. The ``{}`` test is the whole dispatch: any future variable works
    without being registered anywhere, whatever shape its value takes.
    """

    start, end = _trimmed(rhs, 0, len(rhs))
    if start >= end:
        return []  # empty stays empty: encrypting it would fake "configured"
    if not (rhs[start] == "{" and rhs[end - 1] == "}"):
        span_start, span_end = _strip_one_quote_layer(rhs, start, end)
        return [(span_start, span_end)] if span_start < span_end else []
    inner_start, inner_end = start + 1, end - 1
    if inner_start == inner_end:
        # "{}" is deliberately encrypted (it also hides which pool is empty).
        return [(inner_start, inner_end)]
    spans: list[tuple[int, int]] = []
    item_start = inner_start
    while item_start <= inner_end:
        comma = rhs.find(",", item_start, inner_end)
        item_end = comma if comma != -1 else inner_end
        value_start = item_start
        colon = rhs.find(":", item_start, item_end)
        if colon != -1:
            value_start = colon + 1
        span_start, span_end = _trimmed(rhs, value_start, item_end)
        span_start, span_end = _strip_one_quote_layer(rhs, span_start, span_end)
        if span_start < span_end:
            spans.append((span_start, span_end))
        if comma == -1:
            break
        item_start = comma + 1
    return spans


def iter_entries(plaintext_rhs: str) -> list[tuple[str, str]]:
    """(label, key) pairs of a decrypted RHS, for masked display.

    Container items keep their user-chosen names; bare list items and scalar
    values get an empty label. Display-only -- production key selection stays
    in ``finesub.llm.routing.api_keys``.
    """

    entries: list[tuple[str, str]] = []
    start, end = _trimmed(plaintext_rhs, 0, len(plaintext_rhs))
    is_container = (
        start < end and plaintext_rhs[start] == "{" and plaintext_rhs[end - 1] == "}"
    )
    for span_start, span_end in _value_spans(plaintext_rhs):
        value = plaintext_rhs[span_start:span_end]
        if not value:
            continue
        label = ""
        if is_container:
            item_head = plaintext_rhs.rfind(",", start + 1, span_start)
            head = item_head + 1 if item_head != -1 else start + 1
            colon = plaintext_rhs.find(":", head, span_start)
            if colon != -1:
                label = plaintext_rhs[head:colon].strip().strip('"').strip("'")
        entries.append((label, value))
    return entries


def masked(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}…{key[-4:]}"


# Per-process caches. The master is unwrapped once per keyring token and never
# logged. (Python cannot guarantee memory wiping; no promise is made.)
_MASTER_CACHE: dict[str, bytes] = {}
_MASTER_FAILURES: set[str] = set()
_ENSURED: set[Path] = set()
_WARNED: set[tuple[str, str]] = set()


def _warn_once(path: Path, topic: str, message: str) -> None:
    if (str(path), topic) in _WARNED:
        return
    _WARNED.add((str(path), topic))
    print(f"Warning: {message}", file=sys.stderr)


def _resolve_master(keyring_token: str) -> bytes:
    if keyring_token in _MASTER_CACHE:
        return _MASTER_CACHE[keyring_token]
    if keyring_token in _MASTER_FAILURES:
        raise SecretUnreadable("keyring already failed to open in this process")
    try:
        master = unwrap_master(keyring_token)
    except SecretUnreadable:
        _MASTER_FAILURES.add(keyring_token)
        raise
    _MASTER_CACHE[keyring_token] = master
    return master


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``.env`` and return plaintext values. Pure: never writes.

    A value that cannot be decrypted here is skipped with a warning, so it
    surfaces as "not configured" and the caller's existing degradation paths
    apply. ``FINESUB_KEYRING`` is the protector, not a configuration item,
    and is never returned.
    """

    if not path.is_file():
        return {}
    keyring_token: str | None = None
    pending: list[tuple[str, str]] = []
    for line in _parse_lines(path.read_bytes().decode("utf-8")):
        if line.name is None:
            continue
        value = line.rhs.strip()
        if line.name == KEYRING_NAME:
            keyring_token = value
            continue
        pending.append((line.name, value))

    result: dict[str, str] = {}
    for name, value in pending:
        if not _TOKEN_RE.search(value):
            result[name] = value
            continue
        try:
            if keyring_token is None:
                raise SecretUnreadable("no FINESUB_KEYRING line")
            master = _resolve_master(keyring_token)
            result[name] = _TOKEN_RE.sub(
                lambda match: open_(match.group(0), master), value
            )
        except (SecretUnreadable, ProtectionUnavailable) as error:
            _warn_once(
                path,
                f"unreadable:{name}",
                f"{path.name} 中 {name} 的密钥无法在本机解密（{error}）。"
                f"{_REVEAL_HINT}",
            )
    return result


def export_env_file(path: Path) -> dict[str, str]:
    """Same as ``read_env_file``; the name states the CLI intent."""

    return read_env_file(path)


def env_status(path: Path) -> dict[str, str]:
    """Per-variable ``protected`` / ``plaintext`` / ``unreadable``.

    ``plaintext`` means "still needs protection", so a mixed value counts as
    plaintext. Variables with an empty RHS are not keys and are skipped.
    """

    if not path.is_file():
        return {}
    keyring_token: str | None = None
    lines = _parse_lines(path.read_bytes().decode("utf-8"))
    for line in lines:
        if line.name == KEYRING_NAME:
            keyring_token = line.rhs.strip()
    status: dict[str, str] = {}
    for line in lines:
        if line.name is None or line.name == KEYRING_NAME:
            continue
        spans = _value_spans(line.rhs)
        if not spans:
            continue
        values = [line.rhs[s:e] for s, e in spans]
        tokens = [value for value in values if is_protected(value)]
        state = "plaintext"
        if len(tokens) == len(values):
            state = "protected"
        for token in tokens:
            try:
                if keyring_token is None:
                    raise SecretUnreadable("no FINESUB_KEYRING line")
                open_(token, _resolve_master(keyring_token))
            except (SecretUnreadable, ProtectionUnavailable):
                state = "unreadable"
                break
        status[line.name] = state
    return status


@dataclass(slots=True)
class _ProtectResult:
    text: str
    ok: bool
    encrypted: list[str] = field(default_factory=list)
    reason: str = ""


def _protect_text(text: str) -> _ProtectResult:
    """Encrypt every plaintext span of ``text``, byte-preserving elsewhere.

    Raises nothing by design intent but may raise ProtectionUnavailable when
    no wrapping backend exists; callers own the degradation message.
    """

    lines = _parse_lines(text)
    ending = "\r\n" if "\r\n" in text else "\n"

    keyring_index: int | None = None
    for index, line in enumerate(lines):
        if line.name == KEYRING_NAME:
            keyring_index = index
            break

    has_ciphertext = any(
        line.name is not None
        and line.name != KEYRING_NAME
        and _TOKEN_RE.search(line.rhs)
        for line in lines
    )

    todo: list[tuple[int, list[tuple[int, int]]]] = []
    for index, line in enumerate(lines):
        if line.name is None or line.name == KEYRING_NAME:
            continue
        spans = [
            (start, end)
            for start, end in _value_spans(line.rhs)
            if not is_protected(line.rhs[start:end])
        ]
        if spans:
            todo.append((index, spans))

    master: bytes | None = None
    rebuild = False
    if keyring_index is not None:
        token = lines[keyring_index].rhs.strip()
        try:
            master = _resolve_master(token)
        except SecretUnreadable as error:
            if has_ciphertext:
                # "Unreadable" is an observation relative to THIS machine,
                # not proof the data is bad: the file may just be visiting
                # (USB stick, shared folder). Touch nothing.
                return _ProtectResult(
                    text,
                    False,
                    reason=(
                        f"{KEYRING_NAME} 无法在本机解密（{error}），"
                        f"已有密文保持原样。{_REVEAL_HINT}"
                    ),
                )
            # Every ciphertext is gone -- the user explicitly re-entered or
            # removed all values, a clear "use it on this machine" signal.
            rebuild = True

    if not todo:
        if rebuild:
            # Only a stale keyring line is left; drop it so status warnings
            # do not repeat forever.
            del lines[keyring_index]
            return _ProtectResult(_join(lines), True)
        return _ProtectResult(text, True)

    if master is None or rebuild:
        master = new_master()
        wrapped = wrap_master(master)  # ProtectionUnavailable propagates
        keyring_body = f"{KEYRING_NAME}={wrapped}"
        if rebuild:
            line = lines[keyring_index]
            lines[keyring_index] = _EnvLine(
                keyring_body, line.ending or ending, KEYRING_NAME,
                len(KEYRING_NAME) + 1,
            )
        else:
            header = [
                _EnvLine(
                    "# FineSub 主密钥：绑定当前 Windows 账户，勿手动修改或删除。",
                    ending,
                ),
                _EnvLine("# 导出明文密钥：finesub keys --reveal", ending),
                _EnvLine(
                    keyring_body, ending, KEYRING_NAME, len(KEYRING_NAME) + 1
                ),
            ]
            lines = header + lines
            todo = [(index + len(header), spans) for index, spans in todo]

    encrypted: list[str] = []
    for index, spans in todo:
        line = lines[index]
        body = line.body
        for start, end in reversed(spans):
            plaintext = line.rhs[start:end]
            token = seal(plaintext, master)
            if open_(token, master) != plaintext:
                # In-memory round-trip is the pre-write validation; giving up
                # here is why no plaintext .env.bak needs to be left behind.
                return _ProtectResult(
                    text, False, reason="加密自校验失败，.env 保持原样"
                )
            offset = line.rhs_start
            body = body[:offset + start] + token + body[offset + end:]
        lines[index] = _EnvLine(body, line.ending, line.name, line.rhs_start)
        encrypted.append(line.name or "")

    return _ProtectResult(_join(lines), True, encrypted=encrypted)


def _join(lines: list[_EnvLine]) -> str:
    return "".join(line.body + line.ending for line in lines)


def _write_env_text(path: Path, text: str) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(text, encoding="utf-8", newline="")
    os.replace(temp_path, path)


def _notice_converted(path: Path, names: list[str]) -> None:
    print(
        f"FineSub: {path} 中的 API key 已改为绑定当前 Windows 账户存储"
        f"（{', '.join(names)}）。导出明文：finesub keys --reveal；"
        "换机或重装系统前请先导出。",
        file=sys.stderr,
    )


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


def protect_env_file(path: Path) -> bool:
    """Encrypt all plaintext key material in place. Never raises.

    Returns True when the file needs no further work here (protected, empty,
    or absent), False when it gave up -- callers keep running on plaintext.
    """

    if not protection_enabled():
        # A deliberate opt-out is not a warning condition; stay silent so the
        # transition setup does not drown in noise. Returning False keeps the
        # migration unrecorded, ready for when the variable goes away.
        return False
    try:
        if not path.is_file():
            return True
        with holding_lock(_lock_path(path), timeout=_LOCK_TIMEOUT_SECONDS):
            text = path.read_bytes().decode("utf-8")
            result = _protect_text(text)
            if not result.ok:
                _warn_once(path, "protect", result.reason)
                return False
            if result.text != text:
                _write_env_text(path, result.text)
                if result.encrypted:
                    _notice_converted(path, result.encrypted)
            return True
    except LockUnavailable as error:
        _warn_once(path, "lock", f"未能加密 {path.name}（{error}），继续使用明文")
        return False
    except ProtectionUnavailable as error:
        _warn_once(
            path, "unavailable", f"本机不支持密钥保护（{error}），继续使用明文"
        )
        return False
    except (OSError, ValueError) as error:
        _warn_once(path, "error", f"未能加密 {path.name}（{error}），继续使用明文")
        return False


def ensure_protected(path: Path) -> None:
    """Once-per-process safety net for read paths without a migration."""

    resolved = Path(path).resolve()
    if resolved in _ENSURED:
        return
    _ENSURED.add(resolved)
    protect_env_file(path)


def update_env_file(path: Path, updates: Mapping[str, str | None]) -> None:
    """Set (or with ``None``: remove) variables, then protect, in one write.

    Everything not named in ``updates`` -- comments, the keyring line,
    unknown variables, line endings -- is preserved byte for byte. New values
    are born encrypted; if protection is unavailable they are written as
    plaintext with a warning, never dropped.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with holding_lock(_lock_path(path), timeout=_LOCK_TIMEOUT_SECONDS):
        text = path.read_bytes().decode("utf-8") if path.is_file() else ""
        lines = _parse_lines(text)
        ending = "\r\n" if "\r\n" in text else "\n"
        remaining = dict(updates)
        merged: list[_EnvLine] = []
        for line in lines:
            if line.name is None or line.name not in remaining:
                merged.append(line)
                continue
            value = remaining.pop(line.name)
            if value is None:
                continue
            body = line.body[:line.rhs_start] + value
            merged.append(
                _EnvLine(body, line.ending or ending, line.name, line.rhs_start)
            )
        appended = [
            (name, value) for name, value in remaining.items() if value is not None
        ]
        if appended and merged and merged[-1].ending == "":
            last = merged[-1]
            merged[-1] = _EnvLine(last.body, ending, last.name, last.rhs_start)
        for name, value in appended:
            merged.append(_EnvLine(f"{name}={value}", ending, name, len(name) + 1))
        plaintext = _join(merged)

        if not protection_enabled():
            _write_env_text(path, plaintext)
            return

        try:
            result = _protect_text(plaintext)
        except ProtectionUnavailable as error:
            result = _ProtectResult(
                plaintext, False, reason=f"本机不支持密钥保护（{error}）"
            )
        if not result.ok:
            _warn_once(path, "write", f"{result.reason}；新值以明文写入 {path.name}")
        _write_env_text(path, result.text if result.ok else plaintext)

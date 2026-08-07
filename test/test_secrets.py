"""The `.env` protection layer: envelope, span replacement, file semantics.

Everything runs against an injected fake KEK backend so the matrix works on
any platform; the real DPAPI backend gets one round-trip under skipif.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from finesub_bootstrap import secrets

MASTER = b"m" * 32

SAMPLE = (
    "# comment stays put\r\n"
    'GEMINI_FREE={"main":"AIzaMainKey1234567","spare":"AIzaSpareKey7654321"}\r\n'
    "GEMINI_PAID={}\n"
    "EXA_KEYS={exa-key-one-abcdef,exa-key-two-ghijkl}\n"
    "LEGACY_SCALAR=sk-plain-scalar-value\n"
    "EMPTY_VALUE=\n"
    "\n"
    "TRAILING=plain-trailing-key-000\n"
)

SAMPLE_VALUES = {
    "GEMINI_FREE": '{"main":"AIzaMainKey1234567","spare":"AIzaSpareKey7654321"}',
    "GEMINI_PAID": "{}",
    "EXA_KEYS": "{exa-key-one-abcdef,exa-key-two-ghijkl}",
    "LEGACY_SCALAR": "sk-plain-scalar-value",
    "EMPTY_VALUE": "",
    "TRAILING": "plain-trailing-key-000",
}

# What base64("AQAAANCMnd8...") decodes to: the constant header every DPAPI
# blob starts with, which disk scanners grep for.
DPAPI_HEADER = secrets._b64decode("AQAAANCMnd8BFdERjHoAwE/Cl+s")


class FakeBackend:
    """Deterministic stand-in for DPAPI, keyed by a fake machine identity."""

    def __init__(self, machine: bytes = b"machine-A") -> None:
        self.machine = machine

    def wrap(self, data: bytes) -> bytes:
        return DPAPI_HEADER + self.machine + b"|" + data

    def unwrap(self, blob: bytes) -> bytes:
        prefix = DPAPI_HEADER + self.machine + b"|"
        if not blob.startswith(prefix):
            raise secrets.SecretUnreadable("wrapped on another machine")
        return blob[len(prefix):]


class RefusingBackend:
    def wrap(self, data: bytes) -> bytes:
        raise secrets.ProtectionUnavailable("refused")

    def unwrap(self, blob: bytes) -> bytes:
        raise secrets.ProtectionUnavailable("refused")


@pytest.fixture()
def backend():
    fake = FakeBackend()
    secrets.set_backend(fake)
    yield fake
    secrets.set_backend(None)


@pytest.fixture()
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_bytes(SAMPLE.encode("utf-8"))
    return path


# --- envelope --------------------------------------------------------------


def test_seal_open_roundtrip_and_charset(backend) -> None:
    for plaintext in ("AIzaSyExample123", "", "含中文的值", "a:b,c\"d{e}f"):
        token = secrets.seal(plaintext, MASTER)
        assert secrets.open_(token, MASTER) == plaintext
        assert token.startswith("fs$")
        body = token[len("fs$"):]
        assert re.fullmatch(r"[A-Za-z0-9_-]+", body)
        for forbidden in ':,"{}':
            assert forbidden not in body


def test_seal_is_randomized(backend) -> None:
    assert secrets.seal("same", MASTER) != secrets.seal("same", MASTER)


def test_open_rejects_any_corruption(backend) -> None:
    token = secrets.seal("AIzaSyExample123", MASTER)
    flipped = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(secrets.SecretUnreadable):
        secrets.open_(flipped, MASTER)
    with pytest.raises(secrets.SecretUnreadable):
        secrets.open_(token, b"n" * 32)
    with pytest.raises(secrets.SecretUnreadable):
        secrets.open_("fs$abc", MASTER)  # truncated


def test_master_roundtrip_and_obfuscation(backend) -> None:
    master = secrets.new_master()
    token = secrets.wrap_master(master)
    assert secrets.unwrap_master(token) == master
    # The fake blob starts with the real DPAPI header; the obfuscation layer
    # must keep its well-known base64 form out of the stored line.
    assert "AQAAANCMnd8BFdERjHoAwE/Cl+s" not in token


def test_unwrap_master_wrong_machine(backend) -> None:
    token = secrets.wrap_master(secrets.new_master())
    secrets.set_backend(FakeBackend(b"machine-B"))
    with pytest.raises(secrets.SecretUnreadable):
        secrets.unwrap_master(token)


# --- span location -----------------------------------------------------------


@pytest.mark.parametrize(
    ("rhs", "expected"),
    [
        ('{"a":"k1","b":"k2"}', ["k1", "k2"]),
        ("{k1,k2}", ["k1", "k2"]),
        ("bare-scalar", ["bare-scalar"]),
        ("  spaced-out  ", ["spaced-out"]),
        ("'quoted-scalar'", ["quoted-scalar"]),
        ("{ 'a' : 'k1' , b: k2 }", ["k1", "k2"]),
        ("{}", [""]),
        ("", []),
        ("   ", []),
    ],
)
def test_value_spans(rhs: str, expected: list[str]) -> None:
    spans = secrets._value_spans(rhs)
    assert [rhs[start:end] for start, end in spans] == expected


def test_iter_entries_labels() -> None:
    assert secrets.iter_entries('{"a":"k1","b":"k2"}') == [("a", "k1"), ("b", "k2")]
    assert secrets.iter_entries("{k1,k2}") == [("", "k1"), ("", "k2")]
    assert secrets.iter_entries("scalar-key") == [("", "scalar-key")]
    assert secrets.iter_entries("{}") == []


# --- protect_env_file --------------------------------------------------------


def test_protect_preserves_everything_but_key_material(
    backend, env_file: Path, capsys
) -> None:
    assert secrets.protect_env_file(env_file) is True
    text = env_file.read_bytes().decode("utf-8")

    assert "# comment stays put\r\n" in text
    assert "\nEMPTY_VALUE=\n" in text  # empty RHS untouched
    assert "AIzaMainKey1234567" not in text
    assert "sk-plain-scalar-value" not in text
    # Names, braces, quotes, commas and line endings survive byte for byte.
    assert re.search(
        r'GEMINI_FREE=\{"main":"fs\$[A-Za-z0-9_-]+","spare":"fs\$[A-Za-z0-9_-]+"\}\r\n',
        text,
    )
    assert re.search(r"GEMINI_PAID=\{fs\$[A-Za-z0-9_-]+\}\n", text)
    assert re.search(
        r"EXA_KEYS=\{fs\$[A-Za-z0-9_-]+,fs\$[A-Za-z0-9_-]+\}\n", text
    )
    assert text.startswith("# FineSub 主密钥")
    assert secrets.read_env_file(env_file) == SAMPLE_VALUES
    notice = capsys.readouterr().err
    assert "GEMINI_FREE" in notice and "finesub keys --reveal" in notice


def test_protect_is_idempotent(backend, env_file: Path) -> None:
    secrets.protect_env_file(env_file)
    first = env_file.read_bytes()
    assert secrets.protect_env_file(env_file) is True
    assert env_file.read_bytes() == first


def test_existing_keyring_is_never_rewritten(backend, env_file: Path) -> None:
    secrets.protect_env_file(env_file)
    text = env_file.read_bytes().decode("utf-8")
    keyring_line = next(
        line for line in text.splitlines() if line.startswith("FINESUB_KEYRING=")
    )
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write("NEW_KEYS={fresh-plaintext-key}\n")

    secrets.protect_env_file(env_file)
    updated = env_file.read_bytes().decode("utf-8")
    assert keyring_line in updated.splitlines()
    assert "fresh-plaintext-key" not in updated
    assert secrets.read_env_file(env_file)["NEW_KEYS"] == "{fresh-plaintext-key}"


def test_missing_file_is_fine(backend, tmp_path: Path) -> None:
    assert secrets.protect_env_file(tmp_path / ".env") is True
    assert secrets.read_env_file(tmp_path / ".env") == {}


def test_unreadable_master_with_ciphertext_touches_nothing(
    backend, env_file: Path, capsys
) -> None:
    secrets.protect_env_file(env_file)
    before = env_file.read_bytes()

    secrets.set_backend(FakeBackend(b"machine-B"))  # the USB-stick visit
    values = secrets.read_env_file(env_file)
    assert "GEMINI_FREE" not in values
    assert values["EMPTY_VALUE"] == ""  # untouched plaintext still reads
    assert secrets.protect_env_file(env_file) is False
    assert env_file.read_bytes() == before  # not one byte moved
    assert "无法在本机解密" in capsys.readouterr().err


def test_self_heal_after_all_values_replaced(backend, env_file: Path) -> None:
    secrets.protect_env_file(env_file)
    old_keyring = next(
        line
        for line in env_file.read_bytes().decode("utf-8").splitlines()
        if line.startswith("FINESUB_KEYRING=")
    )

    secrets.set_backend(FakeBackend(b"machine-B"))
    # The user re-enters every value on the new machine: the explicit signal.
    secrets.update_env_file(
        env_file,
        {
            "GEMINI_FREE": '{"main":"NewKeyOnB1234567"}',
            "GEMINI_PAID": "{}",
            "EXA_KEYS": "{new-exa-key-abcdef}",
            "LEGACY_SCALAR": "new-scalar-key-value",
            "TRAILING": "new-trailing-key-111",
        },
    )
    text = env_file.read_bytes().decode("utf-8")
    assert old_keyring not in text  # rebuilt for this machine
    assert "FINESUB_KEYRING=finesub$kek$v1$" in text
    assert "NewKeyOnB1234567" not in text  # born encrypted again
    assert (
        secrets.read_env_file(env_file)["GEMINI_FREE"]
        == '{"main":"NewKeyOnB1234567"}'
    )


def test_corrupt_value_skips_only_that_variable(
    backend, env_file: Path, capsys
) -> None:
    secrets.protect_env_file(env_file)
    text = env_file.read_bytes().decode("utf-8")
    token = re.search(r"LEGACY_SCALAR=(fs\$[A-Za-z0-9_-]+)", text).group(1)
    mangled = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    env_file.write_text(text.replace(token, mangled), encoding="utf-8")

    values = secrets.read_env_file(env_file)
    assert "LEGACY_SCALAR" not in values
    assert values["GEMINI_FREE"] == SAMPLE_VALUES["GEMINI_FREE"]
    assert "LEGACY_SCALAR" in capsys.readouterr().err


def test_mixed_plaintext_and_ciphertext_reads(backend, tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("A_KEYS={plain-one}\n", encoding="utf-8")
    secrets.protect_env_file(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("B_KEYS={added-later}\n")

    assert secrets.read_env_file(path) == {
        "A_KEYS": "{plain-one}",
        "B_KEYS": "{added-later}",
    }


def test_concurrent_protect_loses_nothing(backend, env_file: Path) -> None:
    results: list[bool] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(secrets.protect_env_file(env_file))
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    text = env_file.read_bytes().decode("utf-8")
    assert text.count("FINESUB_KEYRING=") == 1
    assert secrets.read_env_file(env_file) == SAMPLE_VALUES


def test_refusing_backend_degrades_to_plaintext(env_file: Path, capsys) -> None:
    secrets.set_backend(RefusingBackend())
    try:
        before = env_file.read_bytes()
        assert secrets.protect_env_file(env_file) is False
        assert env_file.read_bytes() == before
        assert secrets.read_env_file(env_file) == SAMPLE_VALUES
        assert "明文" in capsys.readouterr().err
    finally:
        secrets.set_backend(None)


def test_ensure_protected_runs_once_per_process(
    backend, env_file: Path, monkeypatch
) -> None:
    calls: list[Path] = []
    real = secrets.protect_env_file
    monkeypatch.setattr(
        secrets, "protect_env_file", lambda p: calls.append(p) or real(p)
    )
    secrets.ensure_protected(env_file)
    secrets.ensure_protected(env_file)
    assert len(calls) == 1


# --- read_env_file / env_status ---------------------------------------------


def test_read_returns_no_keyring_and_matches_original(
    backend, env_file: Path
) -> None:
    assert secrets.read_env_file(env_file) == SAMPLE_VALUES
    secrets.protect_env_file(env_file)
    values = secrets.read_env_file(env_file)
    assert values == SAMPLE_VALUES
    assert secrets.KEYRING_NAME not in values


def test_env_status(backend, env_file: Path) -> None:
    assert set(secrets.env_status(env_file).values()) == {"plaintext"}
    secrets.protect_env_file(env_file)
    status = secrets.env_status(env_file)
    assert set(status.values()) == {"protected"}
    assert "EMPTY_VALUE" not in status  # not a key
    secrets.set_backend(FakeBackend(b"machine-B"))
    assert set(secrets.env_status(env_file).values()) == {"unreadable"}


# --- update_env_file ---------------------------------------------------------


def test_update_preserves_untouched_lines(backend, env_file: Path) -> None:
    secrets.protect_env_file(env_file)
    before = env_file.read_bytes().decode("utf-8")
    gemini_line = re.search(r"GEMINI_FREE=[^\r\n]+", before).group(0)
    keyring_line = re.search(r"FINESUB_KEYRING=[^\r\n]+", before).group(0)

    secrets.update_env_file(
        env_file,
        {
            "EXA_KEYS": "{brand-new-exa-key}",
            "LEGACY_SCALAR": None,
            "TAVILY_KEYS": '{"tvly1":"brand-new-tavily"}',
        },
    )
    text = env_file.read_bytes().decode("utf-8")
    assert gemini_line in text  # untouched ciphertext preserved verbatim
    assert keyring_line in text  # invariant 1
    assert "# comment stays put\r\n" in text
    assert "LEGACY_SCALAR" not in text  # None deletes the line
    assert "brand-new-exa-key" not in text  # born encrypted
    values = secrets.read_env_file(env_file)
    assert values["EXA_KEYS"] == "{brand-new-exa-key}"
    assert values["TAVILY_KEYS"] == '{"tvly1":"brand-new-tavily"}'
    assert values["GEMINI_FREE"] == SAMPLE_VALUES["GEMINI_FREE"]


def test_update_creates_file_and_encrypts(backend, tmp_path: Path) -> None:
    path = tmp_path / "sub" / ".env"
    secrets.update_env_file(path, {"GEMINI_FREE": '{"main":"BrandNewKey12345"}'})
    text = path.read_bytes().decode("utf-8")
    assert "BrandNewKey12345" not in text
    assert secrets.read_env_file(path) == {
        "GEMINI_FREE": '{"main":"BrandNewKey12345"}'
    }


def test_update_without_backend_writes_plaintext_and_warns(
    tmp_path: Path, capsys
) -> None:
    secrets.set_backend(RefusingBackend())
    try:
        path = tmp_path / ".env"
        secrets.update_env_file(path, {"GEMINI_FREE": '{"main":"PlainKey123"}'})
        assert '{"main":"PlainKey123"}' in path.read_bytes().decode("utf-8")
        assert "明文" in capsys.readouterr().err
    finally:
        secrets.set_backend(None)


def test_opt_out_variable_disables_auto_protection(
    backend, env_file: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("FINESUB_ENV_PROTECT", "0")
    before = env_file.read_bytes()

    assert secrets.protect_env_file(env_file) is False  # migration stays undone
    assert env_file.read_bytes() == before
    assert capsys.readouterr().err == ""  # a deliberate opt-out is silent

    secrets.update_env_file(env_file, {"NEW_KEYS": "{plain-new-key-123}"})
    text = env_file.read_bytes().decode("utf-8")
    assert "plain-new-key-123" in text
    assert secrets.KEYRING_NAME not in text


def test_opt_out_still_decrypts_existing_ciphertext(
    backend, env_file: Path, monkeypatch
) -> None:
    secrets.protect_env_file(env_file)  # protected before the opt-out

    monkeypatch.setenv("FINESUB_ENV_PROTECT", "0")
    assert secrets.read_env_file(env_file) == SAMPLE_VALUES

    # Updates keep existing ciphertext lines untouched beside plaintext ones.
    secrets.update_env_file(env_file, {"TAVILY_KEYS": "{tvly-plain-key-999}"})
    values = secrets.read_env_file(env_file)
    assert values["GEMINI_FREE"] == SAMPLE_VALUES["GEMINI_FREE"]
    assert values["TAVILY_KEYS"] == "{tvly-plain-key-999}"


def test_read_dotenv_decrypts_and_protects_on_first_read(
    backend, tmp_path: Path, monkeypatch
) -> None:
    from llm import llm_runtime

    env_path = tmp_path / ".env"
    env_path.write_bytes(b'GEMINI_FREE={"main":"AIzaIntegration1234"}\n')
    monkeypatch.setenv("FINESUB_ENV_FILE", str(env_path))

    values = llm_runtime._read_dotenv()

    assert values["GEMINI_FREE"] == '{"main":"AIzaIntegration1234"}'
    # The once-per-process safety net encrypted the checkout .env in place.
    assert "AIzaIntegration1234" not in env_path.read_bytes().decode("utf-8")


# --- contracts ---------------------------------------------------------------


def test_secrets_imports_no_third_party_packages() -> None:
    # A plain [harness] install has no pydantic; llm_runtime imports this
    # module, so it must stay stdlib-only and the package __init__ import-free.
    code = (
        "import sys, finesub_bootstrap.secrets; "
        "bad = sorted({m.split('.')[0] for m in sys.modules} "
        "& {'pydantic', 'httpx', 'desktop'}); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(os.name != "nt", reason="DPAPI is Windows-only")
def test_real_dpapi_roundtrip() -> None:
    backend = secrets.DpapiBackend()
    wrapped = backend.wrap(b"x" * 32)
    assert backend.unwrap(wrapped) == b"x" * 32
    assert wrapped != backend.wrap(b"x" * 32)  # non-deterministic by design
    with pytest.raises(secrets.SecretUnreadable):
        backend.unwrap(wrapped[:-4] + b"zzzz")
    assert secrets.available() is True

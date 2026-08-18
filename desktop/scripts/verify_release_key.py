"""Refuse to sign with a key the shipped clients do not trust.

The signing key lives in a secret now, and GitHub never reads a secret back. A
truncated paste, the wrong file, or a mangled line ending would all produce a
perfectly valid Ed25519 key that simply is not *the* key -- and the result would
be a release whose manifest every installed client rejects, discovered by users
rather than by us.

The public half is not a secret (it ships inside every installer as
`trusted-update-keys.json`), so comparing it costs nothing and can leak nothing.
Run this before signing: it is the only check that the secret is the key.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class KeyMismatch(SystemExit):
    pass


def public_key_of(private_key_path: Path) -> str:
    key = serialization.load_pem_private_key(
        private_key_path.read_bytes(), password=None
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise KeyMismatch(f"{private_key_path} is not an Ed25519 private key")
    raw = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return base64.b64encode(raw).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--trusted-keys", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    args = parser.parse_args()

    trusted = json.loads(args.trusted_keys.read_text(encoding="utf-8"))["keys"]
    expected = trusted.get(args.key_id)
    if expected is None:
        raise KeyMismatch(
            f"{args.trusted_keys} pins no key called {args.key_id}; "
            f"it knows {sorted(trusted)}"
        )

    actual = public_key_of(args.private_key)
    if actual != expected:
        raise KeyMismatch(
            "the signing key is not the one shipped clients trust.\n"
            f"  key id  : {args.key_id}\n"
            f"  expected: {expected}\n"
            f"  got     : {actual}\n"
            "Signing anyway would publish a release every installed client "
            "rejects. Check the secret was pasted whole, BEGIN/END lines "
            "included."
        )
    print(f"signing key matches {args.key_id} ({expected})")


if __name__ == "__main__":
    main()

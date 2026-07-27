from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a FineSub update key")
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--trusted-keys", type=Path, required=True)
    args = parser.parse_args()

    private_path = args.private_key.expanduser().resolve()
    public_path = args.trusted_keys.expanduser().resolve()
    if private_path.exists():
        raise FileExistsError(f"Refusing to overwrite private key: {private_path}")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trusted = {"schemaVersion": 1, "keys": {}}
    if public_path.is_file():
        loaded = json.loads(public_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            trusted.update(loaded)
            trusted.setdefault("keys", {})
    trusted["keys"][args.key_id] = base64.b64encode(public).decode("ascii")
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(
        json.dumps(trusted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Private key: {private_path}")
    print(f"Trusted public keys: {public_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

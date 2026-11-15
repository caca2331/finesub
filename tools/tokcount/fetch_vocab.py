"""Put the Gemma 3 vocabulary next to main.go, which embeds it at build time.

The file (4.7 MB, Apache-2.0) is not committed. This fetches the exact file
google.golang.org/genai v1.62.0's local tokenizer used, from the commit it
pinned, and refuses anything whose sha256 differs. Standard library only, so CI
and a bare checkout can run it before `go build`.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

NAME = "gemma3_cleaned_262144_v2.spiece.model"
URL = (
    "https://raw.githubusercontent.com/google/gemma_pytorch/"
    "014acb7ac4563a5f77c76d7ff98f31b568c16508/tokenizer/" + NAME
)
SHA256 = "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"
TARGET = Path(__file__).resolve().parent / NAME


def main() -> int:
    if TARGET.is_file() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == SHA256:
        print(f"{TARGET.name} already in place")
        return 0
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"{URL}: sha256 {digest}, expected {SHA256}", file=sys.stderr)
        return 1
    partial = TARGET.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(TARGET)
    print(f"fetched {TARGET.name} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

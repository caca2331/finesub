# tokcount

A small Go CLI that counts tokens for Gemini models **locally**, using the
`google.golang.org/genai/tokenizer` local tokenizer — no network, no API quota.
It supports both one-shot execution and a persistent stdio server that
initializes the tokenizer only once.

Used by the Python LLM layer as the first tier of `default_token_counter()`
(local binary → `countTokens` API → heuristic). The source lives here; the
pre-compiled binary is **not tracked** (published as the `tokcount-<version>`
GitHub Release since 2026-08-17, gitignored under `/bin/`) and is not a Python
dependency (nothing in `pyproject.toml` references it).

## Binary location

- `bin/windows-amd64/tokcount.exe` — what the Python layer runs from a source
  checkout. Untracked: a fresh clone lacks it and falls back to the free
  `countTokens` endpoint until you `go build` it here or unzip the published
  release asset into place.
- Anywhere else — the published CLI fetches it as a managed resource and
  names it through `GEMINI_TOKEN_COUNTER_EXE`; see
  [Publishing](#publishing-maintainers).

The Python resolver (`finesub.llm.token_budget._resolve_local_counter_exe`) honors
`GEMINI_TOKEN_COUNTER_EXE` first, then the checkout's `bin/`, then a `tokcount`
on `PATH`. All three layers that look for it now read one module,
`finesub_bootstrap.token_counter`, so the name is written down once — before
2026-08 each spelled it itself, two still searched for this module's old name
`gemini-token-counter`, and one offered a `bin/gemini-token-counter` candidate
that has never existed here.

## Usage

```bash
tokcount.exe "hello world"          # count a literal string
tokcount.exe -file input.txt        # count a file
type input.txt | tokcount.exe       # count stdin
tokcount.exe -model gemini-2.5-flash -file input.txt
```

Prints the token count as a single integer on stdout. An experimental-tokenizer
warning is printed to stderr and can be ignored.

## Persistent server

```bash
tokcount.exe --server
tokcount.exe --server --idle-timeout 5m
```

The server reads one JSON object per line from stdin and writes one JSON object
per line to stdout:

```json
{"text":"hello world"}
{"tokens":2}
```

Text may contain arbitrary newlines because it is encoded as a JSON string.
Requests are processed sequentially by the same tokenizer instance. A bad
request returns an `{"error":"..."}` response without stopping the server.
Closing stdin stops it immediately.

The initial and minimum idle timeout defaults to 300 seconds. A request may
optionally provide a positive timeout in milliseconds:

```json
{"text":"hello world","idle_timeout_ms":900000}
```

After each response:

- an explicit `idle_timeout_ms` replaces the current idle lease;
- when omitted, the next lease is the larger of the configured minimum
  (300 seconds by default) and the previous lease's remaining time;
- tokenization time does not consume the idle lease.

The process exits automatically when that lease expires. This lets a client
keep one process-wide worker and restart it transparently only after an idle
exit or process failure.

## Parity with the countTokens API (important)

The local tokenizer only ships vocabularies for a fixed model set (e.g.
`gemini-2.5-flash`); `gemini-3.1-flash-lite` is **not** supported locally. The
2.5 vocabulary nonetheless matches the 3.1-flash-lite `countTokens` API result
to within a **constant offset of +1 token** — the API wraps the text in a
`contents` envelope worth one extra structural token.

Verified constant across ASCII / CJK / emoji / whitespace-only / empty-ish
inputs and lengths from 1 to ~1000 tokens (local count was always exactly
`API − 1`). `LocalGeminiTokenCounter` adds this `+1` back, so its result equals
the API's exactly.

## Building

```bash
cd tools/tokcount
go build -o ../../bin/windows-amd64/tokcount.exe .
```

The tokenizer vocabulary is downloaded on first run, then cached locally.

## Publishing (maintainers)

A source checkout runs the committed binary directly. The published CLI
cannot: it ships no `bin/`, and the wheel vendors only the Python packages. It
gets it as a managed resource instead — one more row in
`src/finesub_bootstrap/runtime-manifest.json`, downloaded from a GitHub Release the
way ffmpeg and git are. It is the only row both front ends treat as optional:
without it token counting falls back to the free `countTokens` endpoint, so a
failed download costs a network round trip per count, never the run.

Publish it **zipped**, not as a bare `.exe`. `archive_type: "file"` keeps the
downloaded file's name (`tokcount-<version>.bin`), which nothing can then look
up as `tokcount.exe`; the zip path is also the one the other resources use.

```powershell
# 1. Build the archive and read off the two numbers the manifest needs.
@'
import hashlib, pathlib, zipfile
source = pathlib.Path("bin/windows-amd64/tokcount.exe")
target = pathlib.Path("dist/tokcount/tokcount-1.62.0-0-windows-amd64.zip")
target.parent.mkdir(parents=True, exist_ok=True)
entry = zipfile.ZipInfo("tokcount.exe", date_time=(1980, 1, 1, 0, 0, 0))
entry.compress_type = zipfile.ZIP_DEFLATED
entry.external_attr = 0o755 << 16
with zipfile.ZipFile(target, "w") as archive:
    archive.writestr(entry, source.read_bytes(), compresslevel=9)
print(len(target.read_bytes()), hashlib.sha256(target.read_bytes()).hexdigest())
'@ | python -

# 2. Upload it under its own tag, next to the patched CT2 wheel.
gh release create tokcount-1.62.0-0 dist/tokcount/tokcount-1.62.0-0-windows-amd64.zip
```

The fixed zip timestamp is what makes step 1 reproducible: rebuilding the
archive from the same `.exe` yields the same digest, so the manifest can be
re-derived instead of trusted.

**Versioning**: `<genai release>-<serial>`, e.g. `1.62.0-0`. The binary has no
version of its own to report, so the first half names the
`google.golang.org/genai` release it embeds (confirm with
`go version -m tokcount.exe`) — that is what decides the vocabulary and
therefore the counts. The second half is what makes the field do its other job:
`version` is both a directory name under `runtime/tokcount/` and the thing
`ResourceStatus` compares to decide `ready` vs `outdated`, so **every rebuild
needs a new string**, including one that changes only `main.go` or the Go
toolchain. Start at `-0` and bump the serial each time; reset it when genai
moves. Keep the whole thing to simple characters (no `/`, `\`, `:`).

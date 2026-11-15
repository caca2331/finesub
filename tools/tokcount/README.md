# tokcount

A small Go CLI that counts tokens for Gemini models **offline**: the Gemma 3
SentencePiece vocabulary is embedded in the binary and encoded with
`github.com/eliben/go-sentencepiece` — no network, not even on first run, and
no API quota. It supports both one-shot execution and a persistent stdio server
that loads the vocabulary only once.

Until `1.62.0-0` it used `google.golang.org/genai/tokenizer`, which downloads
the same vocabulary from `raw.githubusercontent.com` on first run and caches it
under `%TEMP%\vertexai_tokenizer_model\`. Where that host is blocked the first
count hung for tens of seconds and then failed, every time a cache was missing.
For one text part genai's `CountTokens` is exactly `len(Encode(text))` with this
vocabulary, so the counts did not change (the golden test in `main_test.go`
holds them).

Used by the Python LLM layer as the first tier of `default_token_counter()`
(local binary → `countTokens` API → heuristic). The source lives here; the
pre-compiled binary is **not tracked** (published as the `tokcount-<version>`
GitHub Release since 2026-08-17, gitignored under `/bin/`) and is not a Python
dependency (nothing in `pyproject.toml` references it).

## Binary location

- `bin/windows-amd64/tokcount.exe` — what the Python layer runs from a source
  checkout. Untracked: a fresh clone lacks it and falls back to the free
  `countTokens` endpoint until you unzip the published release asset into
  place or [build it](#building).
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
```

Prints the token count as a single integer on stdout.

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

The embedded vocabulary is the one genai's local tokenizer uses for
`gemini-2.5-flash`; genai has none for `gemini-3.1-flash-lite`. It nonetheless
matches the 3.1-flash-lite `countTokens` API result to within a **constant
offset of +1 token** — the API wraps the text in a `contents` envelope worth one
extra structural token.

Verified constant across ASCII / CJK / emoji / whitespace-only / empty-ish
inputs and lengths from 1 to ~1000 tokens (local count was always exactly
`API − 1`). `LocalGeminiTokenCounter` adds this `+1` back, so its result equals
the API's exactly.

## Building

Releases are built by CI (`.github/workflows/build-tokcount.yml`); a local Go
toolchain is only needed to try a change. The vocabulary (4.7 MB, Apache-2.0)
is not committed: `fetch_vocab.py` downloads it next to `main.go` from the
`google/gemma_pytorch` commit genai pinned and checks its sha256, and
`go build` fails with `no matching files found` until it is there.

```bash
cd tools/tokcount
python fetch_vocab.py
go test ./...
go build -trimpath -o ../../bin/windows-amd64/tokcount.exe .
```

## Publishing (maintainers)

A source checkout runs the binary in `bin/` directly. The published CLI
cannot: it ships no `bin/`, and the wheel vendors only the Python packages. It
gets it as a managed resource instead — one more row in
`src/finesub_bootstrap/runtime-manifest.json`, downloaded from a GitHub Release the
way ffmpeg and git are. It is the only row both front ends treat as optional:
without it token counting falls back to the free `countTokens` endpoint (and,
where Google is unreachable, to the heuristic with a warning), never failing
the run.

`build-tokcount.yml` does the whole build:

1. fetches the vocabulary, runs `go test` (golden counts included), builds
   with `-trimpath`;
2. counts `hello world` with an empty `TEMP` and a proxy that refuses every
   connection, so a binary that still downloads anything fails here;
3. zips `tokcount.exe` with the vocabulary's `LICENSE` at a fixed timestamp
   and writes the zip's size and sha256 into the job summary;
4. with `publish` checked, creates Release `tokcount-<version>` holding the
   zip, and fails rather than overwrite an existing one.

Runs on a push to `tokcount-build-test/**` (build and smoke only; the branch
must start from `main`, which the pre-push hook enforces) or by
`workflow_dispatch` (needs the workflow on `main`). Then copy the size and
sha256 from the job summary into the manifest row together with the new
`version` and URL.

While the workflow is not on `main` yet, publish from a test-branch run, as
`0.7.0-gemma3-0` was: download the run's artifact, check its sha256 against the
job summary, rename the zip to `tokcount-<version>-windows-amd64.zip` (the
zip's bytes do not depend on its name, so the sha256 holds), and
`gh release create tokcount-<version> <zip> --repo caca2331/finesub --target <the built commit> --latest=false`.
`--target` keeps the tag on the commit that was built; `--latest=false` keeps
the product release marked Latest. The workflow's publish job does the same.
Delete the test branch from `public` and locally once the Release exists; its
tag keeps the built commit.

Publish it **zipped**, not as a bare `.exe`. `archive_type: "file"` keeps the
downloaded file's name (`tokcount-<version>.bin`), which nothing can then look
up as `tokcount.exe`; the zip path is also the one the other resources use.

**Versioning**: `<go-sentencepiece version>-<vocabulary>-<serial>`, e.g.
`0.7.0-gemma3-0`. The binary has no version of its own to report; the first two
parts name what decides the counts. The serial is what makes the field do its
other job: `version` is both a directory name under `runtime/tokcount/` and the
thing `ResourceStatus` compares to decide `ready` vs `outdated`, so **every
rebuild needs a new string**, including one that changes only `main.go` or the
Go toolchain. Start at `-0` and bump the serial each time; reset it when either
of the first two parts moves. Keep the whole thing to simple characters (no
`/`, `\`, `:`). Releases up to `1.62.0-0` used `<genai release>-<serial>`.

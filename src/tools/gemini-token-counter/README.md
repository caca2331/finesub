# Gemini Token Counter

A small Go CLI that counts tokens for Gemini models **locally**, using the
`google.golang.org/genai/tokenizer` local tokenizer — no network, no API quota.
It supports both one-shot execution and a persistent stdio server that
initializes the tokenizer only once.

Used by the Python LLM layer as the first tier of `default_token_counter()`
(local binary → `countTokens` API → heuristic). The source lives here; the
pre-compiled binary is committed under `bin/` and is **not** a Python
dependency (nothing in `pyproject.toml` references it).

## Binary location

- `bin/windows-amd64/tokcount.exe` — pre-compiled, what the Python layer runs.

The Python resolver (`llm.token_budget._resolve_local_counter_exe`) also honors
`GEMINI_TOKEN_COUNTER_EXE`, and falls back to `bin/gemini-token-counter` or a
`tokcount` / `gemini-token-counter` on `PATH`.

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
cd src/tools/gemini-token-counter
go build -o ../../../bin/windows-amd64/tokcount.exe .
```

The tokenizer vocabulary is downloaded on first run, then cached locally.

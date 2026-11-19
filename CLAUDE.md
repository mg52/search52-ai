# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`search52-ai` (module `github.com/mg52/search52-ai`): a small in-memory HTTP search service where an LLM, not embeddings, does the matching. Documents and queries are both turned into short lists of text labels by an OpenAI-compatible chat endpoint; search returns documents that share labels with the query. Standard library only, no third-party dependencies. See `README.md` for the API and env var tables.

## Commands

```bash
go build ./...
go test ./...                                   # all tests; none need a real LLM
go test ./internal/index -run TestName          # single test
go vet ./...

# Run against a real LLM (LLM_BASE_URL and LLM_MODEL are required, see how-to-run.txt)
LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen3.5:2b go run ./cmd/server
curl -X POST localhost:8080/documents/batch --data-binary @testdata/sample_documents.json
```

## Architecture

Request path: `handler` (HTTP) → `index.Index` → `index.Chatter` interface → `ai.LLMClient` (HTTP to the LLM). `cmd/server/main.go` only wires `config.Load()` → `ai.NewLLMClient` → `index.New` → `handler.New(idx).Routes()`.

- **`internal/index`**: the core. Three maps under one `sync.RWMutex`: `docs` (id → `Document{ID, Content}`, returned as-is by search), `byID` (id → labels) and `byLabel` (label → set of ids). All are kept in sync by `Add`/`removeLocked`; `Add` with an existing id replaces it. Nothing is persisted.
- **Labels**: documents and queries go through the same single prompt (`ai.Prompt(maxLabels, maxWords)`: asks for 1 to `maxLabels` labels, broad categories first, then specific ones); by default the LLM is *not* shown the labels the index already has; with `SEND_EXISTING_LABELS=true` the index uses `ai.PromptWithLabels` instead, which is `ai.Prompt` plus all existing labels (most used first, ties by name, no cap) so the model reuses them. `parseLabels` normalizes output: lowercase, words are split on spaces/`_`/`-` (joined back with a single `-`), labels over `MaxLabelWords` (`MAX_LABEL_WORDS`, default 1) words are dropped, and each word loses a plural ending (`singular`). The prompt asks for single, simple English words.
- **Scoring**: each query label present in the index weighs `total/size` (`size` = docs with that label, `total` = sum of sizes over the query's labels present in the index, not the whole index), so rare labels count more (`size` counts as at least `MIN_LABEL_DOCS`, so a 1-doc label cannot dominate); a doc's score is the sum over its labels (rounded to 2 decimals), sort is score desc then id, only the first `MAX_RESULTS` are returned, and results under `MIN_RELATIVE_SCORE` of the best score are dropped (the best always stays). The LLM call happens *outside* the lock.
- **`parseLabels`** (`index.go`): deliberately lenient parser for model output (commas/newlines, code fences, bullets, quotes, case, dupes). The first `MaxLabels` labels are kept (the prompt asks for that many; multi-word labels are removed before counting); an empty result is an error.
- **`internal/ai`**: `LLMClient.Complete(ctx, system, user)` posts to `{base}/chat/completions` and retries transient failures (network, 5xx, 408, 429) with backoff (`LLM_MAX_ATTEMPTS`, default 3); per-attempt timeout `LLM_TIMEOUT` (default 10 min); `LLM_TEMPERATURE` (default 0) is always sent; `LLM_MAX_TOKENS` (default 300) is sent as `max_tokens` and a reply cut off by it loses its last label; `LLM_REASONING_EFFORT` is sent as `reasoning_effort` ("none" by default, empty omits it). `prompts.go` holds the one system prompt template (`{{max}}` and `{{words}}` placeholders; the word rule follows `MAX_LABEL_WORDS`); the document or query is the user message. Changing the prompts changes retrieval behavior, so check `prompts_test.go` and the e2e test.
- **`internal/handler`**: Go 1.22+ method-pattern `ServeMux`. `/documents/batch` indexes documents sequentially (one LLM call each) and reports per-document failures without aborting; it returns 400 only if nothing was indexed.
- **`internal/config`**: env-only config with validation and defaults (every tuning knob is an env var, documented in the `Config` struct and the README tables; positive ints, required `LLM_BASE_URL`/`LLM_MODEL`).

## Testing

`internal/mockllm` is a deterministic keyword-rule stand-in for the model. It satisfies `index.Chatter` directly (unit tests) and `mockllm.NewServer` wraps it in an `httptest` OpenAI-compatible server so `internal/e2e` can exercise the real `ai.LLMClient` + handler + index over HTTP. New retrieval behavior should be tested through the mock, not a live model.

## Gotchas

- `data/` and `not.txt` are gitignored scratch (`not.txt` is stale notes from an earlier embedding-based design); don't treat them as part of the project.
- Docker build (`Dockerfile`) pins the Go version via `GO_VERSION`; keep it in line with `go.mod` (`go 1.26.4`).

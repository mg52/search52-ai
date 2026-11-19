# search52-ai

A small in-memory search service that uses an LLM to label documents.

- **Index** (`POST /documents`) — a document is `{"id", "content"}`, both required.
  The `content` is sent to the LLM with one prompt (`internal/ai/prompts.go`) that
  asks for `MAX_LABELS` labels, broad categories first and then the specific and
  detailed ones, each a simple English word. With `SEND_EXISTING_LABELS=true` the prompt also lists the labels
  the index already has so the LLM reuses them. Labels are normalized: lowercase; spaces, `_` and `-` count as word
  separators, labels with more than `MAX_LABEL_WORDS` words are ignored (a
  multi-word label is stored with single hyphens, e.g. `noise-cancelling`), and
  plural endings are dropped (`waffles` = `waffle`, `batteries` = `battery`). The
  service keeps `id → labels`, `label → ids` and `id → document` in memory.
- **Search** (`POST /search`) — the query goes through the same prompt and
  normalization, and every document sharing a label with it is returned as the
  stored `id` + `content` + `score`, best first. Rare labels count more: each
  query label found in the index is worth `total / size`, where `size` is the
  number of documents with that label and `total` is the sum of those sizes over
  the query's labels that exist in the index (not the whole index). With labels of
  100, 200 and 150 documents `total` is 450 and the weights are 4.5, 2.25 and 3.
  A label with fewer than `MIN_LABEL_DOCS` documents counts as having that many,
  so a label that one or two documents happen to have cannot dominate. A
  document's score is the sum of its matching labels' weights; ties are ordered by
  id. Results scoring under `MIN_RELATIVE_SCORE` of the best one are dropped (the
  best result always stays) and at most `MAX_RESULTS` are returned.

The LLM output is parsed leniently: commas or newlines separate labels, and
case, quotes, bullets, code fences and duplicates are cleaned up.

Nothing is persisted: restarting the service empties the index.

## Run

Needs an OpenAI-compatible chat endpoint (OpenAI, Ollama, ...).

```bash
LLM_BASE_URL=http://localhost:11434/v1 \
LLM_API_KEY=ollama \
LLM_MODEL=qwen3 \
go run ./cmd/server
```

All settings are environment variables. Only the first two are required.

**Server and LLM endpoint**

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | required | Base URL of the OpenAI-compatible chat endpoint, e.g. `http://localhost:11434/v1` for Ollama. |
| `LLM_MODEL` | required | Chat model name, e.g. `qwen3.5:9b`. |
| `LLM_API_KEY` | none | Sent as a bearer token. Leave it out for unauthenticated endpoints (Ollama accepts any value). |
| `PORT` | `8080` | HTTP port the server listens on. |
| `LLM_TIMEOUT` | `10m` | Time limit for one LLM call attempt, as a Go duration (`90s`, `10m`). Local reasoning models can be slow, so keep it generous. |
| `LLM_MAX_ATTEMPTS` | `3` | How many times an LLM call is tried in total. Only transient failures (network errors, 5xx, 408, 429) are retried, with exponential backoff. `1` turns retries off. |
| `LLM_TEMPERATURE` | `0` | Sampling temperature, 0 to 2. `0` keeps the labels as repeatable as the model allows, so the same text gets the same labels. It is always sent, `0` included. |
| `LLM_MAX_TOKENS` | `300` | Most tokens one LLM reply may have, sent as `max_tokens`. A label list is short (about 3 tokens per label), so this is a safety net that stops a model that never ends the list from running for minutes. If a reply is cut off, its last, possibly half written label is dropped. Keep it well above `MAX_LABELS` x 3. |
| `LLM_REASONING_EFFORT` | `none` | Value of the `reasoning_effort` request field. `none` turns off chain-of-thought on reasoning models (Qwen3/3.5 via Ollama, OpenAI o-series), which only adds latency for labelling. Set it to an empty string to leave the field out if your endpoint rejects it. |

**Labelling** (what the LLM is asked for, and what is kept)

| Variable | Default | Description |
|---|---|---|
| `MAX_LABELS` | `50` | The LLM is asked for 1 to `MAX_LABELS` labels per document or query and may use fewer (broad categories first, then specific ones). The first `MAX_LABELS` labels it returns (after labels that break `MAX_LABEL_WORDS` are removed) go into the index, so extras are dropped. |
| `SEND_EXISTING_LABELS` | `false` | `true` or `false`. When `true`, every labelling call (documents and search queries) uses a second prompt: the normal one plus every label the index already has, most used first, with an instruction to reuse them instead of inventing synonyms. This makes the vocabulary more consistent, but the prompt carries every label, so it grows with the index and each call gets slower and costlier. `false` sends the normal prompt only. |
| `MAX_LABEL_WORDS` | `1` | Most words in one label. With `1` the LLM is asked for single words and any longer label is ignored. Words are split on spaces, `_` and `-`. The prompt follows this setting. |

**Search ranking** (applied at query time, so changing them needs no re-indexing)

| Variable | Default | Description |
|---|---|---|
| `MIN_LABEL_DOCS` | `5` | When weighing a search, a label counts as having at least this many documents, so a label that only one or two documents happen to have cannot dominate the ranking. Raise it for bigger indexes. |
| `MIN_RELATIVE_SCORE` | `0.2` | Results scoring less than this fraction of the best result are dropped (0 to 1; `0.2` means under 20% of the top score). The best result always stays. `0` keeps every match. |
| `MAX_RESULTS` | `100` | Most documents one search returns, best first. |

Docker: `docker build -t search52-ai .`, then `docker run -p 8080:8080 -e LLM_BASE_URL=... -e LLM_MODEL=... search52-ai`.

## API

| Method | Path                | Description                                              |
|--------|---------------------|----------------------------------------------------------|
| POST   | `/documents`        | Index one document (`id` and `content` required; same `id` replaces) |
| POST   | `/documents/batch`  | Index many documents (`{"documents": [...]}`)            |
| DELETE | `/documents/{id}`   | Remove a document                                        |
| POST   | `/search`           | Search by `{"q": "..."}`                                 |
| GET    | `/health`           | Liveness                                                 |

```bash
curl -X POST localhost:8080/documents -d '{"id": "audio-02", "content": "Bose QuietComfort earbuds with adjustable noise cancelling and a secure in-ear fit for workouts."}'
# {"id":"audio-02","content":"...","labels":["audio","sound","earphones","noise-cancelling"]}

curl -X POST localhost:8080/documents/batch --data-binary @testdata/sample_documents.json

curl -X POST localhost:8080/search -d '{"q": "noise cancelling earbuds"}'
# {"labels":["earphones","noise-cancelling",...],"documents":[{"id":"audio-02","content":"...","score":19.6}]}
```

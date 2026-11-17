# search52-ai

A small in-memory search service that uses an LLM to label documents.

- **Index** (`POST /documents`) — the document's `content` is sent to the LLM
  with the index prompt (`internal/ai/prompts.go`), together with the labels the
  index already has so the LLM reuses them (e.g. an existing `earphones`) and
  only creates a new label when none fits. The LLM answers with plain
  comma-separated labels, e.g. `audio, sound, noise cancelling earphones, earphones`.
  The service keeps `id → labels` and the reverse `label → document ids` in memory.
- **Search** (`POST /search`) — the query is labelled with a separate search
  prompt that also receives the existing labels, and every document sharing a
  label with it is returned (`id` + `content` + `score`). The score is the number
  of the query's labels the document has, so a document in all 3 query labels
  scores 3 and comes first (ties by id).

The LLM output is parsed leniently: commas or newlines separate labels, and
whitespace, case, quotes, bullets, code fences and duplicates are cleaned up.

Nothing is persisted: restarting the service empties the index.

## Run

Needs an OpenAI-compatible chat endpoint (OpenAI, Ollama, ...).

```bash
LLM_BASE_URL=http://localhost:11434/v1 \
LLM_API_KEY=ollama \
LLM_MODEL=qwen3 \
go run ./cmd/server
```

| Variable       | Default  | Description                                       |
|----------------|----------|---------------------------------------------------|
| `LLM_BASE_URL` | required | Base URL of the chat endpoint                     |
| `LLM_MODEL`    | required | Chat model name                                   |
| `LLM_API_KEY`  | optional | API key (`ollama` locally; omit if unauthenticated) |
| `MIN_LABELS`   | `5`      | Fewest labels the LLM is asked for per document/query |
| `MAX_LABELS`   | `10`     | Most labels asked for; extra labels are dropped   |
| `MAX_PROMPT_LABELS` | `1000` | Max existing labels shown to the LLM per call (most used first) |
| `PORT`         | `8080`   | HTTP port                                         |

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
# {"id":"audio-02","content":"...","labels":["audio","sound","noise cancelling earphones","earphones"]}

curl -X POST localhost:8080/documents/batch --data-binary @testdata/sample_documents.json

curl -X POST localhost:8080/search -d '{"q": "noise cancelling earbuds"}'
# {"labels":["earphones","noise cancelling earphones",...],"documents":[{"id":"audio-02","content":"...","labels":[...],"score":3}]}
```

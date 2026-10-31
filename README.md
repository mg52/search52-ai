# search52-ai

A small in-memory search service that uses an LLM to label documents.

- **Index** (`POST /documents`) — the document's `content` is sent to the LLM
  with a fixed prompt (`internal/ai/prompts.go`), which returns labels such as
  `audio, sound, noise cancelling earphones, earphones`. The service keeps
  `id → labels` and the reverse `label → document ids` in memory.
- **Search** (`POST /search`) — the query is labelled by the same prompt, and
  every document sharing a label with it is returned (`id` + `content`), most
  shared labels first.

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
# {"labels":["earphones","noise cancelling earphones",...],"documents":[{"id":"audio-02","content":"...","labels":[...]}]}
```

# ai-search-52

A semantic search service that automatically tags documents using an LLM and retrieves them via embedding-based similarity. Compatible with any OpenAI-compatible API (Ollama, OpenAI, etc.).

## How it works

Documents are indexed through a two-phase pipeline:

**Phase 1 — Tagging** (`POST /documents/llm`)
The document content is sent to an LLM alongside the current tag list. The LLM reuses existing tags where possible, and creates new ones only when needed. Tags are lowercase snake_case category names (e.g. `noise_cancelling_headphones`, `medical_equipment`).

**Phase 2 — Tag embedding** (background)
For each newly created tag, the LLM generates a keyword-rich description using the document that triggered the tag as an example. That description is then embedded into a vector and stored alongside the tag.

**Embedding-only path** (`POST /documents/embed`)
Once tags have vectors, new documents can be tagged without an LLM call. The document is embedded and matched against existing tag vectors via cosine similarity. No new tags are ever created this way.

**Search** (`POST /search`)
The query is embedded and compared against all tag vectors. Tags above the similarity threshold are matched; documents holding those tags are scored and ranked. The final score combines the top tag similarity and the proportion of matched tags:

```
score = (max_tag_similarity × vector_weight) + (matched_tags/total_tags × tag_weight)
```

The response is an object with two fields: `documents` (the ranked results, each with its own `matched_tags`) and a top-level `matched_tags` — the full list of tags the query matched, with per-tag similarity, sorted by score and independent of any document.

## Quickstart

Requires an OpenAI-compatible LLM and embedding endpoint (e.g. Ollama).

```bash
LLM_BASE_URL=http://localhost:11434/v1 \
LLM_API_KEY=ollama \
LLM_MODEL=qwen2.5:3b \
EMBEDDING_BASE_URL=http://localhost:11434/v1 \
EMBEDDING_API_KEY=ollama \
EMBEDDING_MODEL=nomic-embed-text \
go run ./cmd/server
```

## Docker

The image is a multi-stage build producing a static binary on `distroless/static:nonroot`
(~10 MB, no shell, runs as an unprivileged user, ships CA certs for outbound HTTPS).

```bash
docker build -t search52-ai .

docker run --rm -p 8080:8080 \
  -e LLM_BASE_URL=http://host.docker.internal:11434/v1 \
  -e LLM_API_KEY=ollama \
  -e LLM_MODEL=qwen2.5:3b \
  -e EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1 \
  -e EMBEDDING_API_KEY=ollama \
  -e EMBEDDING_MODEL=nomic-embed-text \
  search52-ai
```

Prompt templates are baked into the image at `/app/prompts` (`PROMPTS_DIR`). The container
exposes `8080`; point your orchestrator's liveness/readiness probe at `GET /health`.

## Configuration

| Variable              | Default      | Description                                  |
|-----------------------|--------------|----------------------------------------------|
| `LLM_BASE_URL`        | required     | Base URL for the LLM (OpenAI-compatible)     |
| `LLM_API_KEY`         | required     | API key (`ollama` for local Ollama)          |
| `LLM_MODEL`           | required     | Model name for tagging and description tasks |
| `EMBEDDING_BASE_URL`  | required     | Base URL for the embedding model             |
| `EMBEDDING_API_KEY`   | required     | API key for the embedding endpoint           |
| `EMBEDDING_MODEL`     | required     | Embedding model name                         |
| `PORT`                | `8080`       | HTTP port                                    |
| `PROMPTS_DIR`         | `./prompts`  | Directory containing prompt templates        |
| `MAX_TAGS_PER_DOC`    | `8`          | Maximum tags assigned per document           |
| `TAG_MATCH_THRESHOLD` | `0.60`       | Minimum cosine similarity to match a tag     |

## API

| Method | Path                  | Description                                          |
|--------|-----------------------|------------------------------------------------------|
| POST   | `/documents/llm`      | Index a document via LLM tagging                     |
| POST   | `/documents/embed`    | Index a document via embedding similarity (no LLM)   |
| GET    | `/documents`          | List documents (`?page=1&size=20`)                   |
| GET    | `/documents/{id}`     | Get a single document                                |
| PUT    | `/documents/{id}`     | Replace a document's content and re-tag it via LLM   |
| DELETE | `/documents/{id}`     | Delete a document and its tag associations           |
| GET    | `/tags`               | List all tags with status and description            |
| GET    | `/tags/{name}`        | Get tag detail including example documents           |
| POST   | `/search`             | Semantic search                                      |
| GET    | `/health`             | Service status                                       |

**Index a document**
```bash
curl -X POST http://localhost:8080/documents/llm \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "content": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones..."}'
```

**Search**
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"q": "wireless noise cancelling audio", "limit": 5, "vector_weight": 0.6, "tag_weight": 0.4}'
```

Response:
```json
{
  "documents": [
    {
      "document": { "id": "doc1", "content": "...", "tags": ["..."] },
      "score": 0.78,
      "matched_tags": ["noise_cancelling_headphones", "wireless_headphones"]
    }
  ],
  "matched_tags": [
    { "tag": "noise_cancelling_headphones", "score": 0.82 },
    { "tag": "wireless_headphones", "score": 0.74 }
  ]
}
```

## Architecture

```
POST /documents/llm
  └── LLM (existing tags + content) → assigned tags
        └── new tags → background: LLM description → embedding → stored

POST /documents/embed
  └── embed(content) → cosine similarity vs tag vectors → threshold filter → assigned tags

POST /search
  └── embed(query) → matching tags → documents → scored & ranked
        └── returns { documents, matched_tags } (matched_tags = query↔tag scores)
```

The store is in-memory; all data is lost on restart.

## TODO

- **Persistent storage** — swap the in-memory store for SQLite or PostgreSQL; add pgvector for tag vectors
- **Tag deduplication** — detect and merge near-duplicate tags (e.g. `headphones` / `headphone` / `noisecancelling_headphones`) using embedding similarity at creation time
- **Tag quality guard** — reject and retry description generation when the LLM echoes the tag name back as the description
- **Separate LLM and description models** — allow configuring a smaller/faster model for tagging and a larger model for description generation
- **Bulk ingestion** — `POST /documents/batch` with concurrent tagging and back-pressure
- **Incremental re-embedding** — re-embed tag descriptions when new examples accumulate, without a full restart
- **Search filters** — filter results by tag, date range, or minimum score

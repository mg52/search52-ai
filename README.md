# search52-ai

A semantic search service that **automatically clusters documents into categories**
by vector similarity and retrieves them via embedding search. No LLM is involved —
only an embedding model. Compatible with any OpenAI-compatible embeddings API
(Ollama, OpenAI, etc.).

Documents live in named **indexes**. Each index is an independent, self-contained
engine with its own documents, categories, and tuning; you can create many of
them under one server (e.g. `products`, `tickets`, `docs`). Every index is
snapshotted to disk under `DATA_DIR` and reloaded on startup, so data survives
restarts.

## How it works

Documents are indexed through a single embedding pipeline
(`POST /indexes/{index}/documents`):

**Embed** — the document content is embedded once into a dense vector.

**Incremental categorization** — the vector is compared against the centroids of
existing categories. It joins every category whose cosine similarity is at least
`CATEGORY_THRESHOLD` (up to `MAX_CATEGORIES_PER_DOC`, highest first). If none are
close enough, a new category is created (auto-named `category1`, `category2`, …),
unless the `MAX_CATEGORIES` cap is reached — in which case the document falls back
to its single nearest category so everything stays categorized. Each category's
centroid is the running mean of its members and is updated on every add/remove,
so categories drift to track their contents.

**Search** (`POST /indexes/{index}/search`) — the query is embedded, the
`TOP_N_CATEGORIES` nearest category centroids are selected, and their member
documents are ranked by cosine similarity of the query to each document's own
vector:

```
score = cosine(query_vector, document_vector)
```

The response has two fields: `documents` (ranked results, each with its
`categories`) and `matched_categories` (the nearest categories to the query, with
per-category similarity).

## Quickstart

Requires an OpenAI-compatible embedding endpoint (e.g. Ollama). No LLM needed.

```bash
EMBEDDING_BASE_URL=http://localhost:11434/v1 \
EMBEDDING_API_KEY=ollama \
EMBEDDING_MODEL=nomic-embed-text \
DATA_DIR=./data \
go run ./cmd/server
```

Indexes are persisted as one snapshot per index under `DATA_DIR` (default
`./data`) and reloaded on startup.

## Docker

The image is a multi-stage build producing a static binary on `distroless/static:nonroot`
(~10 MB, no shell, runs as an unprivileged user, ships CA certs for outbound HTTPS).

```bash
docker build -t search52-ai .

docker run --rm -p 8080:8080 \
  -e EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1 \
  -e EMBEDDING_API_KEY=ollama \
  -e EMBEDDING_MODEL=nomic-embed-text \
  -v search52-data:/data \
  search52-ai
```

The image declares `DATA_DIR=/data` and a `VOLUME` at `/data` (writable by the
non-root user); mount a named volume or host path there to persist indexes across
restarts. The container exposes `8080`; point your orchestrator's liveness/
readiness probe at `GET /health`.

### Docker Compose

`docker-compose.yml` runs the full stack — Ollama, a one-shot init that pulls
`nomic-embed-text` into it, and the service — with no local setup beyond Docker:

```bash
docker compose up --build
```

`ollama-pull` blocks `search52-ai` from starting until the model has finished
downloading (`service_completed_successfully`), so the first request never races
a missing model. Both Ollama's model cache and the service's index snapshots are
kept in named volumes (`ollama_data`, `search52_data`) and survive
`docker compose down` (add `-v` to wipe them).

## Configuration

| Variable                 | Default   | Description                                          |
|--------------------------|-----------|------------------------------------------------------|
| `EMBEDDING_BASE_URL`     | required  | Base URL for the embedding model (OpenAI-compatible) |
| `EMBEDDING_MODEL`        | required  | Embedding model name                                 |
| `EMBEDDING_API_KEY`      | optional  | API key (`ollama` locally; omit if unauthenticated)  |
| `PORT`                   | `8080`    | HTTP port                                            |
| `DATA_DIR`               | `./data`  | Directory holding one snapshot subdir per index      |
| `CATEGORY_THRESHOLD`     | `0.60`    | Min cosine similarity to join an existing category   |
| `MAX_CATEGORIES_PER_DOC` | `3`       | Max categories a single document may join            |
| `MAX_CATEGORIES`         | `100`     | Cap on the total number of categories                |
| `TOP_N_CATEGORIES`       | `3`       | Nearest categories scanned per query                 |

The four tuning variables set the **defaults** applied to new indexes. Any of
them can be overridden per index in the `POST /indexes` request body.

## API

All document, category, and search operations are scoped to an index at
`/indexes/{index}/…`.

| Method | Path                                   | Description                                       |
|--------|----------------------------------------|---------------------------------------------------|
| POST   | `/indexes`                             | Create an index (optional per-index tuning)       |
| GET    | `/indexes`                             | List indexes with document/category counts        |
| DELETE | `/indexes/{index}`                     | Delete an index and its on-disk snapshot          |
| POST   | `/indexes/{index}/documents`           | Embed and categorize a document                   |
| POST   | `/indexes/{index}/documents/batch`     | Bulk-index many documents, persisting once at end |
| GET    | `/indexes/{index}/documents/{id}`      | Get a single document                             |
| PUT    | `/indexes/{index}/documents/{id}`      | Replace a document's content and re-categorize it |
| DELETE | `/indexes/{index}/documents/{id}`      | Delete a document and its category memberships    |
| GET    | `/indexes/{index}/categories`          | List all categories with document counts          |
| GET    | `/indexes/{index}/categories/{name}`   | Get category detail                               |
| POST   | `/indexes/{index}/search`              | Semantic search                                   |
| GET    | `/health`                              | Service status (index count)                      |

Index names must match `^[A-Za-z0-9_.-]{1,128}$`.

**Create an index** (tuning fields are optional; omit to inherit server defaults)
```bash
curl -X POST http://localhost:8080/indexes \
  -H "Content-Type: application/json" \
  -d '{"name": "products", "category_threshold": 0.6, "max_categories_per_doc": 3, "max_categories": 100, "top_n_categories": 3}'
```

**Index a document** (`id` is optional; a timestamp id is generated when omitted)
```bash
curl -X POST http://localhost:8080/indexes/products/documents \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "content": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones..."}'
```

**Bulk-index documents** (embeds each in order, then writes a single snapshot)
```bash
curl -X POST http://localhost:8080/indexes/products/documents/batch \
  -H "Content-Type: application/json" \
  --data-binary @testdata/sample_documents.json
```
`testdata/sample_documents.json` ships 100 sample documents spanning five topics
(audio, medical, kitchen, gardening, developer tools). The response reports how
many were indexed and lists any per-document failures:
```json
{ "indexed": 100, "failed": 0, "errors": [] }
```

**Search**
```bash
curl -X POST http://localhost:8080/indexes/products/search \
  -H "Content-Type: application/json" \
  -d '{"q": "wireless noise cancelling audio", "limit": 5}'
```

Response:
```json
{
  "documents": [
    {
      "document": { "id": "doc1", "content": "...", "categories": ["category1"] },
      "score": 0.78,
      "categories": ["category1"]
    }
  ],
  "matched_categories": [
    { "name": "category1", "score": 0.82 }
  ]
}
```

## Architecture

```
Manager
  └── indexes[name] → SearchEngine (docs + categories, one RWMutex)
        └── persisted to DATA_DIR/<name>/index.gob (gzip'd), reloaded on startup

POST /indexes/{index}/documents
  └── embed(content) → cosine vs category centroids
        ├── ≥ threshold → join nearest categories (≤ MAX_CATEGORIES_PER_DOC)
        └── otherwise   → new category (or nearest, if MAX_CATEGORIES reached)
              └── update category centroid (running mean)

POST /indexes/{index}/search
  └── embed(query) → TOP_N_CATEGORIES nearest centroids
        └── member docs ranked by cosine(query, doc_vector)
              └── returns { documents, matched_categories }
```

Each index is held in memory for serving and snapshotted to disk after every
mutation (a gzip-compressed gob written via a temp file + atomic rename). On
startup every snapshot under `DATA_DIR` is loaded back into memory.

## TODO

- **Incremental persistence** — replace the whole-index snapshot with a write-ahead log + periodic compaction so writes don't rewrite the full index
- **Category merging** — detect and merge categories whose centroids drift close together over time
- **Recall tuning** — expose per-query `top_n` override; consider scanning more categories when the nearest are weak
- **Concurrent bulk ingestion** — `POST /indexes/{index}/documents/batch` exists but embeds sequentially; add concurrent embedding with back-pressure
- **Search filters** — filter results by category, date range, or minimum score
```

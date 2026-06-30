# search52-ai

A semantic search service that **automatically clusters documents into categories**
by vector similarity and retrieves them via embedding search. No LLM is involved —
only an embedding model. Compatible with any OpenAI-compatible embeddings API
(Ollama, OpenAI, etc.).

## How it works

Documents are indexed through a single embedding pipeline (`POST /documents`):

**Embed** — the document content is embedded once into a dense vector.

**Incremental categorization** — the vector is compared against the centroids of
existing categories. It joins every category whose cosine similarity is at least
`CATEGORY_THRESHOLD` (up to `MAX_CATEGORIES_PER_DOC`, highest first). If none are
close enough, a new category is created (auto-named `category1`, `category2`, …),
unless the `MAX_CATEGORIES` cap is reached — in which case the document falls back
to its single nearest category so everything stays categorized. Each category's
centroid is the running mean of its members and is updated on every add/remove,
so categories drift to track their contents.

**Search** (`POST /search`) — the query is embedded, the `TOP_N_CATEGORIES`
nearest category centroids are selected, and their member documents are ranked by
cosine similarity of the query to each document's own vector:

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
go run ./cmd/server
```

## Docker

The image is a multi-stage build producing a static binary on `distroless/static:nonroot`
(~10 MB, no shell, runs as an unprivileged user, ships CA certs for outbound HTTPS).

```bash
docker build -t search52-ai .

docker run --rm -p 8080:8080 \
  -e EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1 \
  -e EMBEDDING_API_KEY=ollama \
  -e EMBEDDING_MODEL=nomic-embed-text \
  search52-ai
```

The container exposes `8080`; point your orchestrator's liveness/readiness probe
at `GET /health`.

## Configuration

| Variable                 | Default   | Description                                          |
|--------------------------|-----------|------------------------------------------------------|
| `EMBEDDING_BASE_URL`     | required  | Base URL for the embedding model (OpenAI-compatible) |
| `EMBEDDING_MODEL`        | required  | Embedding model name                                 |
| `EMBEDDING_API_KEY`      | optional  | API key (`ollama` locally; omit if unauthenticated)  |
| `PORT`                   | `8080`    | HTTP port                                            |
| `CATEGORY_THRESHOLD`     | `0.60`    | Min cosine similarity to join an existing category   |
| `MAX_CATEGORIES_PER_DOC` | `3`       | Max categories a single document may join            |
| `MAX_CATEGORIES`         | `100`     | Cap on the total number of categories                |
| `TOP_N_CATEGORIES`       | `3`       | Nearest categories scanned per query                 |

## API

| Method | Path                  | Description                                          |
|--------|-----------------------|------------------------------------------------------|
| POST   | `/documents`          | Embed and categorize a document                      |
| GET    | `/documents`          | List documents (`?page=1&size=20`)                   |
| GET    | `/documents/{id}`     | Get a single document                                |
| PUT    | `/documents/{id}`     | Replace a document's content and re-categorize it    |
| DELETE | `/documents/{id}`     | Delete a document and its category memberships       |
| GET    | `/categories`         | List all categories with document counts             |
| GET    | `/categories/{name}`  | Get category detail                                  |
| POST   | `/search`             | Semantic search                                      |
| GET    | `/health`             | Service status                                       |

**Index a document**
```bash
curl -X POST http://localhost:8080/documents \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "content": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones..."}'
```

**Search**
```bash
curl -X POST http://localhost:8080/search \
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
POST /documents
  └── embed(content) → cosine vs category centroids
        ├── ≥ threshold → join nearest categories (≤ MAX_CATEGORIES_PER_DOC)
        └── otherwise   → new category (or nearest, if MAX_CATEGORIES reached)
              └── update category centroid (running mean)

POST /search
  └── embed(query) → TOP_N_CATEGORIES nearest centroids
        └── member docs ranked by cosine(query, doc_vector)
              └── returns { documents, matched_categories }
```

The store is in-memory; all data is lost on restart.

## TODO

- **Persistent storage** — swap the in-memory store for SQLite/PostgreSQL (+ pgvector for centroids and document vectors)
- **Category merging** — detect and merge categories whose centroids drift close together over time
- **Recall tuning** — expose per-query `top_n` override; consider scanning more categories when the nearest are weak
- **Bulk ingestion** — `POST /documents/batch` with concurrent embedding and back-pressure
- **Search filters** — filter results by category, date range, or minimum score
```

# search52-ai

A semantic search service that **automatically clusters documents into categories**
by vector similarity and retrieves them via embedding search. No LLM involved —
only an embedding model. Compatible with any OpenAI-compatible embeddings API
(Ollama, OpenAI, etc.).

Documents live in named **indexes**, each an independent engine with its own
documents, categories, and tuning. Create as many as you need under one server
(e.g. `products`, `tickets`, `docs`).

## How it works

**Embed** (`POST /indexes/{index}/documents`) — content is embedded once into a
dense vector.

**Categorize** — the vector joins every existing category whose centroid it's
at least `CATEGORY_THRESHOLD` cosine-similar to (up to `MAX_CATEGORIES_PER_DOC`,
closest first). If none qualify, it seeds a new category — unless `MAX_CATEGORIES`
is already reached, in which case it falls back to the single nearest category.
Each category's centroid is the running mean of its members, updated on every
add/remove.

**Search** (`POST /indexes/{index}/search`) — the query is embedded, the
`TOP_N_CATEGORIES` nearest centroids are selected, and their member documents
are ranked by `cosine(query_vector, document_vector)`. The response has
`documents` (ranked results) and `matched_categories` (nearest categories to
the query, with similarity).

**Split candidacy** — each category tracks the running variance of its
members' similarity to the centroid, using [Welford's online
algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).
Once that variance exceeds `VARIANCE_THRESHOLD` and the category has more than
`VARIANCE_MIN_COUNT` members, its `ShouldSplit` flag flips true. This is a
signal only — nothing currently splits a flagged category.

## Quickstart

Requires an OpenAI-compatible embedding endpoint (e.g. Ollama). No LLM needed.

```bash
EMBEDDING_BASE_URL=http://localhost:11434/v1 \
EMBEDDING_API_KEY=ollama \
EMBEDDING_MODEL=nomic-embed-text \
DATA_DIR=./data \
go run ./cmd/server
```

## Persistence

Each index is held in memory and snapshotted to `DATA_DIR/<name>/index.gob`
(gzip'd gob). Creating an index persists it immediately, and every snapshot
under `DATA_DIR` is reloaded on startup. Document writes (single, batch,
update, delete) are **not** auto-persisted — call `POST
/indexes/{index}/persist` to flush current state to disk.

## Docker

Multi-stage build producing a static binary on `distroless/static:nonroot`
(~10 MB, no shell, unprivileged user, ships CA certs).

```bash
docker build -t search52-ai .

docker run --rm -p 8080:8080 \
  -e EMBEDDING_BASE_URL=http://host.docker.internal:11434/v1 \
  -e EMBEDDING_API_KEY=ollama \
  -e EMBEDDING_MODEL=nomic-embed-text \
  -v search52-data:/data \
  search52-ai
```

`DATA_DIR=/data` is declared as a `VOLUME`; mount a named volume or host path
there to persist indexes across restarts. Liveness/readiness probe: `GET
/health`.

### Docker Compose

Runs the full stack — Ollama, a one-shot pull of `nomic-embed-text`, and the
service — with no local setup beyond Docker:

```bash
docker compose up --build
```

`ollama-pull` blocks `search52-ai` from starting until the model has finished
downloading, so the first request never races a missing model. Ollama's model
cache and the service's index snapshots are kept in named volumes
(`ollama_data`, `search52_data`) and survive `docker compose down` (add `-v`
to wipe them).

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
| `VARIANCE_THRESHOLD`     | `0.02`    | Variance above which a category's `ShouldSplit` flips true |
| `VARIANCE_MIN_COUNT`     | `100`     | Min category member count before `ShouldSplit` can fire |

These set the **defaults** for new indexes; any can be overridden per index in
the `POST /indexes` request body.

## API

All document, category, and search operations are scoped to an index at
`/indexes/{index}/…`. Index names must match `^[A-Za-z0-9_.-]{1,128}$`.

| Method | Path                                   | Description                                       |
|--------|----------------------------------------|---------------------------------------------------|
| POST   | `/indexes`                             | Create an index (optional per-index tuning)       |
| GET    | `/indexes`                             | List indexes with document/category counts        |
| DELETE | `/indexes/{index}`                     | Delete an index and its on-disk snapshot          |
| POST   | `/indexes/{index}/documents`           | Embed and categorize a document                   |
| POST   | `/indexes/{index}/documents/batch`     | Bulk-index many documents (upsert by id)          |
| GET    | `/indexes/{index}/documents/{id}`      | Get a single document                             |
| PUT    | `/indexes/{index}/documents/{id}`      | Replace a document's content and re-categorize it |
| DELETE | `/indexes/{index}/documents/{id}`      | Delete a document and its category memberships    |
| GET    | `/indexes/{index}/categories`          | List all categories with document counts          |
| GET    | `/indexes/{index}/categories/{name}`   | Get category detail                               |
| POST   | `/indexes/{index}/persist`             | Flush the index's current state to disk           |
| POST   | `/indexes/{index}/search`              | Semantic search                                   |
| GET    | `/health`                              | Service status (index count)                      |

`GET /indexes/{index}/categories[/{name}]` includes each category's `variance`
and `should_split`; the detail endpoint also reports `vector_dims`:
```json
{ "name": "category1", "doc_count": 42, "vector_dims": 768, "created_at": "...", "variance": 0.0113, "should_split": false }
```

**Create an index** (tuning fields are optional; omit to inherit server defaults)
```bash
curl -X POST http://localhost:8080/indexes \
  -H "Content-Type: application/json" \
  -d '{"name": "products", "category_threshold": 0.6, "max_categories_per_doc": 3, "max_categories": 100, "top_n_categories": 3, "variance_threshold": 0.02, "variance_min_count": 100}'
```

**Index a document** (`id` is optional; a timestamp id is generated when omitted)
```bash
curl -X POST http://localhost:8080/indexes/products/documents \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "content": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones..."}'
```

**Bulk-index documents**, then **persist** (batch ingestion doesn't auto-persist)
```bash
curl -X POST http://localhost:8080/indexes/products/documents/batch \
  -H "Content-Type: application/json" \
  --data-binary @testdata/sample_documents.json

curl -X POST http://localhost:8080/indexes/products/persist
```
`testdata/sample_documents.json` ships 100 sample documents spanning five topics
(audio, medical, kitchen, gardening, developer tools). The batch response:
```json
{ "indexed": 100, "failed": 0, "errors": [] }
```

**Search**
```bash
curl -X POST http://localhost:8080/indexes/products/search \
  -H "Content-Type: application/json" \
  -d '{"q": "wireless noise cancelling audio", "limit": 5}'
```
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

## TODO

- **Incremental persistence** — replace the whole-index snapshot with a write-ahead log + periodic compaction so writes don't rewrite the full index
- **Category merging** — detect and merge categories whose centroids drift close together over time
- **Category splitting** — `ShouldSplit` is currently a signal only; nothing consumes it yet. Implement an actual split (e.g. re-cluster a flagged category's members into two) and decide when it runs (inline on the triggering write vs. a background sweep)
- **Recall tuning** — expose per-query `top_n` override; consider scanning more categories when the nearest are weak
- **Concurrent bulk ingestion** — `POST /indexes/{index}/documents/batch` exists but embeds sequentially; add concurrent embedding with back-pressure
- **Search filters** — filter results by category, date range, or minimum score

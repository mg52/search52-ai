package common

import (
	"context"
	"time"
)

// Document is the API-facing shape of a stored item, shared by every engine
// implementation. Engine-specific runtime fields (embedding vectors, etc.)
// live on top of this in each engine's own type.
type Document struct {
	ID         string    `json:"id"`
	Content    string    `json:"content"`
	Categories []string  `json:"categories"`
	CreatedAt  time.Time `json:"created_at"`
}

// SearchResult is a ranked document with its similarity score.
type SearchResult struct {
	Document   Document `json:"document"`
	Score      float64  `json:"score"`
	Categories []string `json:"categories"`
}

// CategoryMatch is a category a query matched, with its similarity.
type CategoryMatch struct {
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

// Results is the search response: ranked documents plus the categories the
// query matched (independent of any single document).
type Results struct {
	Documents         []SearchResult  `json:"documents"`
	MatchedCategories []CategoryMatch `json:"matched_categories"`
}

// CategorySummary is the API-facing shape of a category. Fields meaningful to
// only one engine kind (Variance/ShouldSplit for the embedding engine,
// Description for the LLM engine) are omitted by the other.
type CategorySummary struct {
	Name        string    `json:"name"`
	DocCount    int       `json:"doc_count"`
	CreatedAt   time.Time `json:"created_at"`
	VectorDims  int       `json:"vector_dims,omitempty"`
	Variance    *float64  `json:"variance,omitempty"`
	ShouldSplit *bool     `json:"should_split,omitempty"`
	Description *string   `json:"description,omitempty"`
}

// Embedder turns text into an embedding vector. Implemented by
// ai.EmbeddingClient; both engine kinds depend on this interface (not the
// concrete client) so they can be tested with fakes.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// Chatter completes a system/user prompt pair with a raw text response.
// Implemented by ai.LLMClient; the LLM engine depends on this interface so
// it can be tested with fakes.
type Chatter interface {
	Complete(ctx context.Context, system, user string) (string, error)
}

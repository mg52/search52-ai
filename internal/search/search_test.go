package search

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/pipeline"
	"github.com/mg52/search52-ai/internal/store"
)

// vocabEmbed maps each known word to a one-hot dimension; a text embeds to its
// bag-of-words count, so cosine similarity reflects token overlap.
func vocabEmbed(vocab map[string]int, dim int) func(string) []float64 {
	return func(text string) []float64 {
		out := make([]float64, dim)
		for _, tok := range strings.Fields(strings.ToLower(text)) {
			if i, ok := vocab[tok]; ok {
				out[i]++
			}
		}
		return out
	}
}

func embedClient(t *testing.T, fn func(string) []float64) *ai.EmbeddingClient {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			Input string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&in)
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": fn(in.Input)}},
		})
	}))
	t.Cleanup(srv.Close)
	return ai.NewEmbeddingClient(srv.URL, "k", "m")
}

// addDoc seeds a document through the categorizer (the owner of clustering), so
// the store ends up with the categories and centroids search relies on.
func addDoc(t *testing.T, c *pipeline.Categorizer, id, content string) {
	t.Helper()
	if _, err := c.Process(context.Background(), store.Document{ID: id, Content: content}); err != nil {
		t.Fatalf("seed %s: %v", id, err)
	}
}

func TestSearchRanksByVector(t *testing.T) {
	vocab := map[string]int{"apple": 0, "pie": 1, "rocket": 2, "space": 3}
	fn := vocabEmbed(vocab, 4)
	st := store.NewMemoryStore()
	emb := embedClient(t, fn)
	cat := pipeline.NewCategorizer(emb, st, 0.6, 3, 10)
	addDoc(t, cat, "d1", "apple")
	addDoc(t, cat, "d2", "apple pie")
	addDoc(t, cat, "d3", "rocket space")

	eng := NewEngine(emb, st, 3)

	res, err := eng.Search(context.Background(), Query{Q: "apple"})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res.Documents) == 0 {
		t.Fatal("apple search returned no documents")
	}
	ids := map[string]bool{}
	for _, d := range res.Documents {
		ids[d.Document.ID] = true
	}
	if !ids["d1"] || !ids["d2"] {
		t.Fatalf("apple search = %v, want d1 and d2", ids)
	}
	if ids["d3"] {
		t.Fatalf("apple search unexpectedly returned d3")
	}
	// Exact match d1 should outrank partial d2.
	if res.Documents[0].Document.ID != "d1" {
		t.Fatalf("top result = %s, want d1", res.Documents[0].Document.ID)
	}
	if len(res.MatchedCategories) == 0 {
		t.Fatal("expected matched categories")
	}
}

func TestSearchLimit(t *testing.T) {
	vocab := map[string]int{"apple": 0, "pie": 1}
	fn := vocabEmbed(vocab, 2)
	st := store.NewMemoryStore()
	emb := embedClient(t, fn)
	cat := pipeline.NewCategorizer(emb, st, 0.6, 3, 10)
	addDoc(t, cat, "d1", "apple")
	addDoc(t, cat, "d2", "apple pie")

	eng := NewEngine(emb, st, 3)
	res, err := eng.Search(context.Background(), Query{Q: "apple", Limit: 1})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("limit not honored: %d results", len(res.Documents))
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	fn := vocabEmbed(map[string]int{"apple": 0}, 1)
	eng := NewEngine(embedClient(t, fn), store.NewMemoryStore(), 3)
	res, err := eng.Search(context.Background(), Query{Q: "apple"})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res.Documents) != 0 {
		t.Fatalf("empty index returned %d docs", len(res.Documents))
	}
}

package pipeline

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/store"
)

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

func mustProcess(t *testing.T, c *Categorizer, id, content string) store.Document {
	t.Helper()
	doc, err := c.Process(context.Background(), store.Document{ID: id, Content: content})
	if err != nil {
		t.Fatalf("process %s: %v", id, err)
	}
	return doc
}

func TestCategorizerIncremental(t *testing.T) {
	vocab := map[string]int{"apple": 0, "pie": 1, "rocket": 2, "space": 3}
	fn := vocabEmbed(vocab, 4)
	st := store.NewMemoryStore()
	c := NewCategorizer(embedClient(t, fn), st, 0.6, 3, 10)

	d1, err := c.Process(context.Background(), store.Document{ID: "d1", Content: "apple"})
	if err != nil {
		t.Fatalf("process d1: %v", err)
	}
	if len(d1.Categories) != 1 || st.CategoryCount() != 1 {
		t.Fatalf("d1 cats=%v count=%d, want 1/1", d1.Categories, st.CategoryCount())
	}

	d2, err := c.Process(context.Background(), store.Document{ID: "d2", Content: "apple pie"})
	if err != nil {
		t.Fatalf("process d2: %v", err)
	}
	if st.CategoryCount() != 1 || d2.Categories[0] != d1.Categories[0] {
		t.Fatalf("d2 should join d1: cats=%v count=%d", d2.Categories, st.CategoryCount())
	}

	d3, err := c.Process(context.Background(), store.Document{ID: "d3", Content: "rocket space"})
	if err != nil {
		t.Fatalf("process d3: %v", err)
	}
	if st.CategoryCount() != 2 {
		t.Fatalf("after d3 category count = %d, want 2", st.CategoryCount())
	}
	if d3.Categories[0] == d1.Categories[0] {
		t.Fatalf("d3 unexpectedly shares d1 category")
	}
}

func TestCategorizerCapFallback(t *testing.T) {
	vocab := map[string]int{"a": 0, "b": 1, "c": 2}
	fn := vocabEmbed(vocab, 3)
	st := store.NewMemoryStore()
	// strict threshold, one category per doc, cap at 2 total.
	c := NewCategorizer(embedClient(t, fn), st, 0.99, 1, 2)

	mustProcess(t, c, "d1", "a")
	mustProcess(t, c, "d2", "b")
	if st.CategoryCount() != 2 {
		t.Fatalf("category count = %d, want 2 at cap", st.CategoryCount())
	}
	// Distinct vector, but the cap is reached → falls back to the nearest one.
	d3 := mustProcess(t, c, "d3", "c")
	if st.CategoryCount() != 2 {
		t.Fatalf("category count = %d, want still 2", st.CategoryCount())
	}
	if len(d3.Categories) != 1 {
		t.Fatalf("d3 categories = %v, want exactly one (fallback)", d3.Categories)
	}
}

func TestCategorizerRemovePrunesEmptyCategory(t *testing.T) {
	fn := vocabEmbed(map[string]int{"apple": 0}, 1)
	st := store.NewMemoryStore()
	c := NewCategorizer(embedClient(t, fn), st, 0.6, 3, 10)

	mustProcess(t, c, "d1", "apple")
	if st.CategoryCount() != 1 {
		t.Fatalf("setup category count = %d, want 1", st.CategoryCount())
	}
	if _, ok := c.Remove("d1"); !ok {
		t.Fatal("remove returned false")
	}
	if _, ok := c.Remove("d1"); ok {
		t.Fatal("second remove should return false")
	}
	if st.CategoryCount() != 0 {
		t.Fatalf("category count after remove = %d, want 0 (pruned)", st.CategoryCount())
	}
	if st.DocCount() != 0 {
		t.Fatalf("doc count after remove = %d, want 0", st.DocCount())
	}
}

func TestCategorizerUpdateReassigns(t *testing.T) {
	vocab := map[string]int{"apple": 0, "pie": 1, "rocket": 2, "space": 3}
	fn := vocabEmbed(vocab, 4)
	st := store.NewMemoryStore()
	c := NewCategorizer(embedClient(t, fn), st, 0.6, 3, 10)

	a := mustProcess(t, c, "d1", "apple")        // category1
	r := mustProcess(t, c, "d2", "rocket space") // category2
	if st.CategoryCount() != 2 {
		t.Fatalf("setup category count = %d, want 2", st.CategoryCount())
	}

	// Re-process d1 with content matching d2's cluster.
	d1 := mustProcess(t, c, "d1", "rocket space")
	if d1.Categories[0] != r.Categories[0] {
		t.Fatalf("re-processed d1 category = %v, want %v", d1.Categories, r.Categories)
	}
	// category1 had only d1 → pruned on detach.
	if st.CategoryCount() != 1 {
		t.Fatalf("category count after reassign = %d, want 1", st.CategoryCount())
	}
	if got := st.DocCountByCategory(a.Categories[0]); got != 0 {
		t.Fatalf("old category still has %d docs, want 0", got)
	}
	if got := st.DocCountByCategory(r.Categories[0]); got != 2 {
		t.Fatalf("target category doc count = %d, want 2", got)
	}
}

func TestCategorizerVectorIsStored(t *testing.T) {
	fn := vocabEmbed(map[string]int{"apple": 0, "pie": 1}, 2)
	st := store.NewMemoryStore()
	c := NewCategorizer(embedClient(t, fn), st, 0.6, 3, 10)

	if _, err := c.Process(context.Background(), store.Document{ID: "d1", Content: "apple pie"}); err != nil {
		t.Fatalf("process: %v", err)
	}
	got, ok := st.GetDocument("d1")
	if !ok {
		t.Fatal("document not stored")
	}
	if len(got.Vector) != 2 || got.Norm == 0 {
		t.Fatalf("stored vector=%v norm=%v, want non-empty vector + norm", got.Vector, got.Norm)
	}
}

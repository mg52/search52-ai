package search

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/store"
	"github.com/mg52/search52-ai/internal/vec"
)

// embedConst returns an embedding client that always embeds to the given vector.
func embedConst(t *testing.T, v []float64) *ai.EmbeddingClient {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": v}},
		})
	}))
	t.Cleanup(srv.Close)
	return ai.NewEmbeddingClient(srv.URL, "k", "m")
}

func seedTag(st store.Store, name string, v []float64) {
	st.SaveTag(store.Tag{Name: name, Vector: v, Norm: vec.Norm(v)})
}

func TestSearchRanksByMatchedTags(t *testing.T) {
	st := store.NewMemoryStore()
	// Two tags aligned with the query, one orthogonal.
	seedTag(st, "audio", []float64{1, 0})
	seedTag(st, "music", []float64{1, 0})
	seedTag(st, "medical", []float64{0, 1})

	// d2 shares two matched tags; d1 shares one matched + one unmatched.
	st.SaveDocument(store.Document{ID: "d1", Tags: []string{"audio", "medical"}})
	st.SaveDocument(store.Document{ID: "d2", Tags: []string{"audio", "music"}})

	eng := NewEngine(embedConst(t, []float64{1, 0}), st, 0.5)
	res, err := eng.Search(context.Background(), Query{Q: "audio"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Documents) != 2 {
		t.Fatalf("results = %d, want 2", len(res.Documents))
	}
	if res.Documents[0].Document.ID != "d2" {
		t.Errorf("top result = %s, want d2 (more matched tags)", res.Documents[0].Document.ID)
	}
	// d2 matched both of its tags.
	if len(res.Documents[0].MatchedTags) != 2 {
		t.Errorf("d2 matched tags = %v, want 2", res.Documents[0].MatchedTags)
	}

	// Query-level matched tags sit alongside the documents, with per-tag scores,
	// sorted by similarity. "medical" is orthogonal to the query so it's excluded.
	if len(res.MatchedTags) != 2 {
		t.Fatalf("matched_tags = %v, want 2 entries", res.MatchedTags)
	}
	for _, m := range res.MatchedTags {
		if m.Score <= 0 {
			t.Errorf("matched tag %q has non-positive score %v", m.Tag, m.Score)
		}
		if m.Tag == "medical" {
			t.Error("orthogonal tag 'medical' should not be in matched_tags")
		}
	}
	// Sorted by score descending.
	if res.MatchedTags[0].Score < res.MatchedTags[1].Score {
		t.Errorf("matched_tags not sorted by score: %v", res.MatchedTags)
	}
}

func TestSearchNoMatch(t *testing.T) {
	st := store.NewMemoryStore()
	seedTag(st, "audio", []float64{1, 0})
	st.SaveDocument(store.Document{ID: "d1", Tags: []string{"audio"}})

	// Query orthogonal to every tag -> below threshold.
	eng := NewEngine(embedConst(t, []float64{0, 1}), st, 0.5)
	res, err := eng.Search(context.Background(), Query{Q: "x"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Documents) != 0 {
		t.Errorf("results = %d, want 0", len(res.Documents))
	}
	if len(res.MatchedTags) != 0 {
		t.Errorf("matched_tags = %v, want empty", res.MatchedTags)
	}
}

func TestSearchRespectsLimit(t *testing.T) {
	st := store.NewMemoryStore()
	seedTag(st, "audio", []float64{1, 0})
	for _, id := range []string{"d1", "d2", "d3"} {
		st.SaveDocument(store.Document{ID: id, Tags: []string{"audio"}})
	}
	eng := NewEngine(embedConst(t, []float64{1, 0}), st, 0.5)
	res, err := eng.Search(context.Background(), Query{Q: "audio", Limit: 2})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Documents) != 2 {
		t.Errorf("results = %d, want 2 (limit)", len(res.Documents))
	}
}

func TestSearchSkipsTagsWithoutVectors(t *testing.T) {
	st := store.NewMemoryStore()
	seedTag(st, "audio", []float64{1, 0})
	st.SaveTag(store.Tag{Name: "novec"}) // no vector
	st.SaveDocument(store.Document{ID: "d1", Tags: []string{"audio", "novec"}})

	eng := NewEngine(embedConst(t, []float64{1, 0}), st, 0.5)
	res, err := eng.Search(context.Background(), Query{Q: "audio"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Documents) != 1 || res.Documents[0].Document.ID != "d1" {
		t.Fatalf("results = %v, want [d1]", res.Documents)
	}
	// Only "audio" should be reported as matched (novec has no vector).
	if len(res.Documents[0].MatchedTags) != 1 || res.Documents[0].MatchedTags[0] != "audio" {
		t.Errorf("matched = %v, want [audio]", res.Documents[0].MatchedTags)
	}
	if len(res.MatchedTags) != 1 || res.MatchedTags[0].Tag != "audio" {
		t.Errorf("query matched_tags = %v, want [audio]", res.MatchedTags)
	}
}

func TestSearchZeroQueryVector(t *testing.T) {
	st := store.NewMemoryStore()
	seedTag(st, "audio", []float64{1, 0})
	st.SaveDocument(store.Document{ID: "d1", Tags: []string{"audio"}})

	eng := NewEngine(embedConst(t, []float64{0, 0}), st, 0.5)
	res, err := eng.Search(context.Background(), Query{Q: "x"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Documents) != 0 {
		t.Errorf("results = %d, want 0 for zero query vector", len(res.Documents))
	}
}

func TestSearchDefaultsWeightsAndLimit(t *testing.T) {
	st := store.NewMemoryStore()
	seedTag(st, "audio", []float64{1, 0})
	st.SaveDocument(store.Document{ID: "d1", Tags: []string{"audio"}})

	eng := NewEngine(embedConst(t, []float64{1, 0}), st, 0.5)
	// No weights or limit provided -> defaults applied, query still works.
	res, err := eng.Search(context.Background(), Query{Q: "audio"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res.Documents) != 1 || res.Documents[0].Score <= 0 {
		t.Errorf("expected one scored result, got %v", res.Documents)
	}
}

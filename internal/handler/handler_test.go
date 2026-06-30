package handler

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/pipeline"
	"github.com/mg52/search52-ai/internal/search"
	"github.com/mg52/search52-ai/internal/store"
)

// vocab one-hot embedding: each known word is a dimension, so cosine reflects
// token overlap. Unknown words are ignored.
var vocab = map[string]int{
	"sony": 0, "headphones": 1, "wireless": 2, "audio": 3, "device": 4,
	"gear": 5, "more": 6, "original": 7, "updated": 8, "content": 9,
	"a": 10, "b": 11, "x": 12,
}

func vocabEmbed(text string) []float64 {
	out := make([]float64, len(vocab))
	for _, tok := range strings.Fields(strings.ToLower(text)) {
		if i, ok := vocab[tok]; ok {
			out[i]++
		}
	}
	return out
}

// newStack wires the full handler against a fake OpenAI-compatible embedding
// server. No LLM is involved; categorization is synchronous.
func newStack(t *testing.T) *httptest.Server {
	t.Helper()
	aiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			http.NotFound(w, r)
			return
		}
		var in struct {
			Input string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&in)
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": vocabEmbed(in.Input)}},
		})
	}))
	t.Cleanup(aiSrv.Close)

	st := store.NewMemoryStore()
	emb := ai.NewEmbeddingClient(aiSrv.URL, "k", "m")
	categorizer := pipeline.NewCategorizer(emb, st, 0.6, 3, 10)
	searcher := search.NewEngine(emb, st, 3)
	h := New(st, categorizer, searcher)

	srv := httptest.NewServer(h.Routes())
	t.Cleanup(srv.Close)
	return srv
}

func do(t *testing.T, srv *httptest.Server, method, path string, body any) (int, []byte) {
	t.Helper()
	var r io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, srv.URL+path, r)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, data
}

func TestHealth(t *testing.T) {
	srv := newStack(t)
	status, body := do(t, srv, http.MethodGet, "/health", nil)
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200", status)
	}
	var resp map[string]any
	json.Unmarshal(body, &resp)
	if resp["status"] != "ok" {
		t.Errorf("status field = %v", resp["status"])
	}
	if _, ok := resp["category_count"]; !ok {
		t.Errorf("health missing category_count: %s", body)
	}
}

func TestCreateDocument(t *testing.T) {
	srv := newStack(t)
	status, body := do(t, srv, http.MethodPost, "/documents", map[string]string{
		"id": "d1", "content": "sony headphones",
	})
	if status != http.StatusCreated {
		t.Fatalf("status = %d, want 201 (body: %s)", status, body)
	}
	var doc store.Document
	json.Unmarshal(body, &doc)
	if len(doc.Categories) == 0 {
		t.Errorf("expected at least one category, got %v", doc.Categories)
	}

	if s, _ := do(t, srv, http.MethodGet, "/documents/d1", nil); s != http.StatusOK {
		t.Errorf("GET document status = %d, want 200", s)
	}
	s, listBody := do(t, srv, http.MethodGet, "/documents", nil)
	if s != http.StatusOK || !strings.Contains(string(listBody), `"total":1`) {
		t.Errorf("list = %d %s", s, listBody)
	}
}

func TestCreateValidation(t *testing.T) {
	srv := newStack(t)
	if s, _ := do(t, srv, http.MethodPost, "/documents", map[string]string{"id": "x"}); s != http.StatusBadRequest {
		t.Errorf("empty content status = %d, want 400", s)
	}
	do(t, srv, http.MethodPost, "/documents", map[string]string{"id": "dup", "content": "a"})
	if s, _ := do(t, srv, http.MethodPost, "/documents", map[string]string{"id": "dup", "content": "b"}); s != http.StatusConflict {
		t.Errorf("duplicate status = %d, want 409", s)
	}
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/documents", strings.NewReader("{bad"))
	resp, _ := http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("malformed body status = %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestCategoriesEndpoints(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents", map[string]string{"id": "d1", "content": "sony headphones"})

	s, body := do(t, srv, http.MethodGet, "/categories", nil)
	if s != http.StatusOK || !strings.Contains(string(body), "category1") {
		t.Fatalf("GET /categories = %d %s", s, body)
	}

	s, detail := do(t, srv, http.MethodGet, "/categories/category1", nil)
	if s != http.StatusOK {
		t.Fatalf("GET /categories/category1 = %d", s)
	}
	var d map[string]any
	json.Unmarshal(detail, &d)
	if d["doc_count"].(float64) != 1 {
		t.Errorf("doc_count = %v, want 1", d["doc_count"])
	}

	if s, _ := do(t, srv, http.MethodGet, "/categories/ghost", nil); s != http.StatusNotFound {
		t.Errorf("unknown category status = %d, want 404", s)
	}
}

func TestSearchEndpoint(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents", map[string]string{"id": "d1", "content": "wireless audio device"})

	s, body := do(t, srv, http.MethodPost, "/search", map[string]any{"q": "audio device", "limit": 5})
	if s != http.StatusOK {
		t.Fatalf("search status = %d (%s)", s, body)
	}
	var results search.Results
	json.Unmarshal(body, &results)
	if len(results.Documents) == 0 {
		t.Error("expected at least one search result")
	}
	if len(results.MatchedCategories) == 0 {
		t.Error("expected matched_categories at the top level")
	}

	if s, _ := do(t, srv, http.MethodPost, "/search", map[string]any{"q": ""}); s != http.StatusBadRequest {
		t.Errorf("empty query status = %d, want 400", s)
	}
}

func TestUpdateDocument(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents", map[string]string{"id": "d1", "content": "original"})

	s, body := do(t, srv, http.MethodPut, "/documents/d1", map[string]string{"content": "updated content"})
	if s != http.StatusOK {
		t.Fatalf("update status = %d (%s)", s, body)
	}
	_, getBody := do(t, srv, http.MethodGet, "/documents/d1", nil)
	if !strings.Contains(string(getBody), "updated content") {
		t.Errorf("content not updated: %s", getBody)
	}

	if s, _ := do(t, srv, http.MethodPut, "/documents/ghost", map[string]string{"content": "x"}); s != http.StatusNotFound {
		t.Errorf("update missing status = %d, want 404", s)
	}
	if s, _ := do(t, srv, http.MethodPut, "/documents/d1", map[string]string{"content": ""}); s != http.StatusBadRequest {
		t.Errorf("update empty content status = %d, want 400", s)
	}
}

func TestDeleteDocument(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents", map[string]string{"id": "d1", "content": "headphones"})

	if s, _ := do(t, srv, http.MethodDelete, "/documents/d1", nil); s != http.StatusNoContent {
		t.Fatalf("delete status = %d, want 204", s)
	}
	if s, _ := do(t, srv, http.MethodGet, "/documents/d1", nil); s != http.StatusNotFound {
		t.Errorf("get after delete = %d, want 404", s)
	}
	if s, _ := do(t, srv, http.MethodDelete, "/documents/missing", nil); s != http.StatusNotFound {
		t.Errorf("delete missing = %d, want 404", s)
	}
}

func TestGetMissingDocument(t *testing.T) {
	srv := newStack(t)
	if s, _ := do(t, srv, http.MethodGet, "/documents/nope", nil); s != http.StatusNotFound {
		t.Errorf("status = %d, want 404", s)
	}
}

func TestListDocumentsClampsParams(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents", map[string]string{"id": "d1", "content": "a"})

	for _, path := range []string{
		"/documents?page=0&size=999",
		"/documents?page=-5",
		"/documents?size=abc",
	} {
		s, body := do(t, srv, http.MethodGet, path, nil)
		if s != http.StatusOK {
			t.Errorf("GET %s = %d, want 200", path, s)
		}
		if !strings.Contains(string(body), `"page":1`) {
			t.Errorf("GET %s did not clamp page to 1: %s", path, body)
		}
	}
}

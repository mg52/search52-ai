package handler

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/pipeline"
	"github.com/mg52/search52-ai/internal/search"
	"github.com/mg52/search52-ai/internal/store"
)

func testPrompts(t *testing.T) *ai.Prompts {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"tagging_system.template":     "tagging system max {{.MaxTags}}",
		"tagging_user.template":       "existing {{.ExistingTags}} doc {{.Content}}",
		"description_system.template": "describe tag {{.TagName}}",
		"description_user.template":   "tag {{.TagName}} examples {{.Examples}}",
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	p, err := ai.LoadPrompts(dir)
	if err != nil {
		t.Fatalf("LoadPrompts: %v", err)
	}
	return p
}

func hashEmbed(s string) []float64 {
	v := make([]float64, 8)
	for i, ch := range s {
		v[i%8] += float64(ch)
	}
	allZero := true
	for _, x := range v {
		if x != 0 {
			allZero = false
		}
	}
	if allZero {
		v[0] = 1
	}
	return v
}

// newStack wires the full handler against a fake OpenAI-compatible server.
// Thresholds are 0 so any tag with a vector matches, keeping search/embed
// deterministic regardless of the toy embeddings.
func newStack(t *testing.T) *httptest.Server {
	t.Helper()
	aiSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/chat/completions":
			raw, _ := io.ReadAll(r.Body)
			content := `{"tags":["audio","wireless"]}`
			if strings.Contains(string(raw), "describe tag") {
				content = `{"description":"dense keywords"}`
			}
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"content": content}}},
			})
		case "/embeddings":
			var in struct {
				Input string `json:"input"`
			}
			json.NewDecoder(r.Body).Decode(&in)
			json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"embedding": hashEmbed(in.Input)}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(aiSrv.Close)

	st := store.NewMemoryStore()
	llm := ai.NewLLMClient(aiSrv.URL, "k", "m")
	emb := ai.NewEmbeddingClient(aiSrv.URL, "k", "m")
	prompts := testPrompts(t)
	embedder := pipeline.NewEmbedder(llm, emb, prompts, st)
	tagger := pipeline.NewTagger(llm, emb, prompts, st, 8, 0, embedder)
	searcher := search.NewEngine(emb, st, 0)
	h := New(st, tagger, searcher)

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

// waitTagsReady polls until the named tag reports a vector (background embedding).
func waitTagsReady(t *testing.T, srv *httptest.Server, tag string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		status, body := do(t, srv, http.MethodGet, "/tags", nil)
		if status == http.StatusOK {
			var tags []struct {
				Name      string `json:"name"`
				HasVector bool   `json:"has_vector"`
			}
			json.Unmarshal(body, &tags)
			for _, tg := range tags {
				if tg.Name == tag && tg.HasVector {
					return
				}
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("tag %q never got a vector", tag)
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
}

func TestTagByLLMEndpoint(t *testing.T) {
	srv := newStack(t)
	status, body := do(t, srv, http.MethodPost, "/documents/llm", map[string]string{
		"id": "d1", "content": "sony headphones",
	})
	if status != http.StatusCreated {
		t.Fatalf("status = %d, want 201 (body: %s)", status, body)
	}
	var doc store.Document
	json.Unmarshal(body, &doc)
	if len(doc.Tags) != 2 {
		t.Errorf("tags = %v, want 2", doc.Tags)
	}

	// Document is retrievable and listed.
	if s, _ := do(t, srv, http.MethodGet, "/documents/d1", nil); s != http.StatusOK {
		t.Errorf("GET document status = %d, want 200", s)
	}
	s, listBody := do(t, srv, http.MethodGet, "/documents", nil)
	if s != http.StatusOK || !strings.Contains(string(listBody), `"total":1`) {
		t.Errorf("list = %d %s", s, listBody)
	}
}

func TestTagByLLMValidation(t *testing.T) {
	srv := newStack(t)
	// Missing content.
	if s, _ := do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "x"}); s != http.StatusBadRequest {
		t.Errorf("empty content status = %d, want 400", s)
	}
	// Duplicate ID.
	do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "dup", "content": "a"})
	if s, _ := do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "dup", "content": "b"}); s != http.StatusConflict {
		t.Errorf("duplicate status = %d, want 409", s)
	}
	// Malformed JSON.
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/documents/llm", strings.NewReader("{bad"))
	resp, _ := http.DefaultClient.Do(req)
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("malformed body status = %d, want 400", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestTagsEndpoints(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "d1", "content": "headphones"})
	waitTagsReady(t, srv, "audio")

	s, body := do(t, srv, http.MethodGet, "/tags", nil)
	if s != http.StatusOK || !strings.Contains(string(body), "audio") {
		t.Fatalf("GET /tags = %d %s", s, body)
	}

	s, detail := do(t, srv, http.MethodGet, "/tags/audio", nil)
	if s != http.StatusOK {
		t.Fatalf("GET /tags/audio = %d", s)
	}
	var d map[string]any
	json.Unmarshal(detail, &d)
	if d["doc_count"].(float64) != 1 {
		t.Errorf("doc_count = %v, want 1", d["doc_count"])
	}

	if s, _ := do(t, srv, http.MethodGet, "/tags/ghost", nil); s != http.StatusNotFound {
		t.Errorf("unknown tag status = %d, want 404", s)
	}
}

func TestSearchEndpoint(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "d1", "content": "wireless headphones"})
	waitTagsReady(t, srv, "audio")

	s, body := do(t, srv, http.MethodPost, "/search", map[string]any{"q": "audio device", "limit": 5})
	if s != http.StatusOK {
		t.Fatalf("search status = %d (%s)", s, body)
	}
	var results search.Results
	json.Unmarshal(body, &results)
	if len(results.Documents) == 0 {
		t.Error("expected at least one search result")
	}
	// Query-level matched tags are returned alongside the documents.
	if len(results.MatchedTags) == 0 {
		t.Error("expected matched_tags at the top level")
	}

	if s, _ := do(t, srv, http.MethodPost, "/search", map[string]any{"q": ""}); s != http.StatusBadRequest {
		t.Errorf("empty query status = %d, want 400", s)
	}
}

func TestEmbedEndpoint(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "d1", "content": "headphones"})
	waitTagsReady(t, srv, "audio")

	s, body := do(t, srv, http.MethodPost, "/documents/embed", map[string]string{"id": "d2", "content": "more audio gear"})
	if s != http.StatusCreated {
		t.Fatalf("embed status = %d (%s)", s, body)
	}
	if !strings.Contains(string(body), "tag_scores") {
		t.Errorf("embed response missing tag_scores: %s", body)
	}
}

func TestUpdateDocument(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "d1", "content": "original"})

	s, body := do(t, srv, http.MethodPut, "/documents/d1", map[string]string{"content": "updated content"})
	if s != http.StatusOK {
		t.Fatalf("update status = %d (%s)", s, body)
	}
	_, getBody := do(t, srv, http.MethodGet, "/documents/d1", nil)
	if !strings.Contains(string(getBody), "updated content") {
		t.Errorf("content not updated: %s", getBody)
	}

	// Update missing doc -> 404.
	if s, _ := do(t, srv, http.MethodPut, "/documents/ghost", map[string]string{"content": "x"}); s != http.StatusNotFound {
		t.Errorf("update missing status = %d, want 404", s)
	}
	// Update with empty content -> 400.
	if s, _ := do(t, srv, http.MethodPut, "/documents/d1", map[string]string{"content": ""}); s != http.StatusBadRequest {
		t.Errorf("update empty content status = %d, want 400", s)
	}
}

func TestDeleteDocument(t *testing.T) {
	srv := newStack(t)
	do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "d1", "content": "headphones"})

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
	do(t, srv, http.MethodPost, "/documents/llm", map[string]string{"id": "d1", "content": "a"})

	// page<1 and size>100 and a non-numeric size all fall back to defaults.
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

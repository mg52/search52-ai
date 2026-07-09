package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mg52/search52-ai/internal/engine"
)

// vocabEmbed is a deterministic one-hot embedder: each known word is a
// dimension, so cosine similarity reflects token overlap. It implements
// engine.Embedder directly, no HTTP round-trip needed.
var vocab = map[string]int{
	"sony": 0, "headphones": 1, "wireless": 2, "audio": 3, "device": 4,
	"medical": 5, "stethoscope": 6, "blood": 7, "pressure": 8, "updated": 9,
}

type fakeEmbedder struct{}

func (fakeEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	out := make([]float32, len(vocab))
	for _, tok := range strings.Fields(strings.ToLower(text)) {
		if i, ok := vocab[tok]; ok {
			out[i]++
		}
	}
	return out, nil
}

func newStack(t *testing.T) *httptest.Server {
	t.Helper()
	mgr := NewManager(t.TempDir(), fakeEmbedder{}, engine.Config{
		CategoryThreshold: 0.6, MaxCategoriesPerDoc: 3, MaxCategories: 10, TopNCategories: 3,
	})
	srv := httptest.NewServer(mgr.Routes())
	t.Cleanup(srv.Close)
	return srv
}

func do(t *testing.T, srv *httptest.Server, method, path, body string) (int, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(method, srv.URL+path, bytes.NewBufferString(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if len(raw) > 0 {
		json.Unmarshal(raw, &out)
	}
	return resp.StatusCode, out
}

func TestCreateIndexAndDocument(t *testing.T) {
	srv := newStack(t)

	if code, _ := do(t, srv, "POST", "/indexes", `{"name":"products"}`); code != http.StatusCreated {
		t.Fatalf("create index: got %d", code)
	}
	// Duplicate index -> 409.
	if code, _ := do(t, srv, "POST", "/indexes", `{"name":"products"}`); code != http.StatusConflict {
		t.Fatalf("duplicate index: got %d", code)
	}
	// Document into a missing index -> 404.
	if code, _ := do(t, srv, "POST", "/indexes/nope/documents", `{"id":"d1","content":"sony headphones"}`); code != http.StatusNotFound {
		t.Fatalf("missing index: got %d", code)
	}

	code, body := do(t, srv, "POST", "/indexes/products/documents", `{"id":"d1","content":"sony wireless headphones audio device"}`)
	if code != http.StatusCreated {
		t.Fatalf("create doc: got %d (%v)", code, body)
	}
	cats, _ := body["categories"].([]any)
	if len(cats) == 0 {
		t.Fatalf("expected a category, got %v", body["categories"])
	}
}

func TestSearchRanksAndSeparatesDomains(t *testing.T) {
	srv := newStack(t)
	do(t, srv, "POST", "/indexes", `{"name":"products"}`)

	docs := map[string]string{
		"d1": "sony wireless headphones audio device",
		"d2": "wireless headphones audio",
		"d3": "medical stethoscope",
		"d4": "medical blood pressure",
	}
	for id, content := range docs {
		if code, b := do(t, srv, "POST", "/indexes/products/documents", `{"id":"`+id+`","content":"`+content+`"}`); code != http.StatusCreated {
			t.Fatalf("seed %s: %d (%v)", id, code, b)
		}
	}

	code, body := do(t, srv, "POST", "/indexes/products/search", `{"q":"wireless audio headphones","limit":5}`)
	if code != http.StatusOK {
		t.Fatalf("search: got %d", code)
	}
	results, _ := body["documents"].([]any)
	if len(results) == 0 {
		t.Fatalf("expected audio hits, got none: %v", body)
	}
	// Top hit must be an audio doc, not a medical one.
	top := results[0].(map[string]any)["document"].(map[string]any)["id"].(string)
	if top != "d1" && top != "d2" {
		t.Fatalf("expected audio doc on top, got %q", top)
	}
	for _, r := range results {
		id := r.(map[string]any)["document"].(map[string]any)["id"].(string)
		if id == "d3" || id == "d4" {
			t.Fatalf("medical doc %q leaked into audio search", id)
		}
	}
	// documents must always be a JSON array, never null.
	if body["documents"] == nil {
		t.Fatalf("documents should be [] not null")
	}
}

func TestPersistenceRoundTrip(t *testing.T) {
	dir := t.TempDir()
	mgr := NewManager(dir, fakeEmbedder{}, engine.Config{CategoryThreshold: 0.6, MaxCategoriesPerDoc: 3, MaxCategories: 10, TopNCategories: 3})
	srv := httptest.NewServer(mgr.Routes())
	do(t, srv, "POST", "/indexes", `{"name":"products"}`)
	do(t, srv, "POST", "/indexes/products/documents", `{"id":"d1","content":"sony wireless headphones"}`)
	srv.Close()

	// Fresh manager over the same dir should reload the index and its document.
	mgr2 := NewManager(dir, fakeEmbedder{}, engine.Config{})
	if err := mgr2.LoadExisting(); err != nil {
		t.Fatalf("load: %v", err)
	}
	srv2 := httptest.NewServer(mgr2.Routes())
	defer srv2.Close()

	code, body := do(t, srv2, "GET", "/indexes/products/documents/d1", "")
	if code != http.StatusOK {
		t.Fatalf("reloaded doc: got %d", code)
	}
	if body["id"] != "d1" {
		t.Fatalf("expected d1, got %v", body["id"])
	}
}

// doList is like do but for endpoints that return a top-level JSON array
// (e.g. GET .../categories), which does not unmarshal into a map.
func doList(t *testing.T, srv *httptest.Server, method, path, body string) (int, []any) {
	t.Helper()
	req, err := http.NewRequest(method, srv.URL+path, bytes.NewBufferString(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var out []any
	if len(raw) > 0 {
		json.Unmarshal(raw, &out)
	}
	return resp.StatusCode, out
}

func TestIndexLifecycle(t *testing.T) {
	srv := newStack(t)

	// No indexes yet.
	code, body := do(t, srv, "GET", "/indexes", "")
	if code != http.StatusOK {
		t.Fatalf("list indexes: %d", code)
	}
	if body["total"].(float64) != 0 {
		t.Fatalf("expected 0 indexes, got %v", body["total"])
	}

	do(t, srv, "POST", "/indexes", `{"name":"a"}`)
	do(t, srv, "POST", "/indexes", `{"name":"b"}`)

	code, body = do(t, srv, "GET", "/indexes", "")
	if body["total"].(float64) != 2 {
		t.Fatalf("expected 2 indexes, got %v", body["total"])
	}
	indexes := body["indexes"].([]any)
	// Sorted by name: "a" first.
	if indexes[0].(map[string]any)["name"] != "a" {
		t.Fatalf("indexes not sorted by name: %v", indexes)
	}

	// Delete "a", then it must be gone.
	if code, _ = do(t, srv, "DELETE", "/indexes/a", ""); code != http.StatusNoContent {
		t.Fatalf("delete index: %d", code)
	}
	if code, _ = do(t, srv, "DELETE", "/indexes/a", ""); code != http.StatusNotFound {
		t.Fatalf("delete missing index: %d", code)
	}
	if code, _ = do(t, srv, "POST", "/indexes/a/documents", `{"content":"x"}`); code != http.StatusNotFound {
		t.Fatalf("op on deleted index: %d", code)
	}
}

func TestCreateIndexValidation(t *testing.T) {
	srv := newStack(t)
	// Invalid name.
	if code, _ := do(t, srv, "POST", "/indexes", `{"name":"bad name!"}`); code != http.StatusBadRequest {
		t.Fatalf("invalid name: %d", code)
	}
	// Malformed JSON body.
	if code, _ := do(t, srv, "POST", "/indexes", `{`); code != http.StatusBadRequest {
		t.Fatalf("bad body: %d", code)
	}
}

func TestInvalidIndexNameInPath(t *testing.T) {
	srv := newStack(t)
	// A path segment that fails the name pattern is rejected before lookup.
	if code, _ := do(t, srv, "POST", "/indexes/bad!name/search", `{"q":"x"}`); code != http.StatusBadRequest {
		t.Fatalf("invalid index name in path: %d", code)
	}
}

func TestDocumentCRUD(t *testing.T) {
	srv := newStack(t)
	do(t, srv, "POST", "/indexes", `{"name":"i"}`)

	// Auto-generated id when omitted.
	code, body := do(t, srv, "POST", "/indexes/i/documents", `{"content":"sony wireless headphones"}`)
	if code != http.StatusCreated {
		t.Fatalf("create doc: %d", code)
	}
	id, _ := body["id"].(string)
	if id == "" {
		t.Fatalf("expected auto-generated id, got %v", body["id"])
	}

	// Duplicate id -> 409.
	if code, _ = do(t, srv, "POST", "/indexes/i/documents", `{"id":"dup","content":"sony audio"}`); code != http.StatusCreated {
		t.Fatalf("seed dup: %d", code)
	}
	if code, _ = do(t, srv, "POST", "/indexes/i/documents", `{"id":"dup","content":"wireless"}`); code != http.StatusConflict {
		t.Fatalf("duplicate doc: %d", code)
	}

	// Empty content -> 400; malformed body -> 400.
	if code, _ = do(t, srv, "POST", "/indexes/i/documents", `{"id":"z"}`); code != http.StatusBadRequest {
		t.Fatalf("empty content: %d", code)
	}
	if code, _ = do(t, srv, "POST", "/indexes/i/documents", `{`); code != http.StatusBadRequest {
		t.Fatalf("bad body: %d", code)
	}

	// Get present / missing.
	if code, _ = do(t, srv, "GET", "/indexes/i/documents/"+id, ""); code != http.StatusOK {
		t.Fatalf("get doc: %d", code)
	}
	if code, _ = do(t, srv, "GET", "/indexes/i/documents/nope", ""); code != http.StatusNotFound {
		t.Fatalf("get missing doc: %d", code)
	}

	// Update: present, missing, empty content, bad body.
	if code, _ = do(t, srv, "PUT", "/indexes/i/documents/"+id, `{"content":"medical stethoscope"}`); code != http.StatusOK {
		t.Fatalf("update doc: %d", code)
	}
	if code, _ = do(t, srv, "PUT", "/indexes/i/documents/nope", `{"content":"x"}`); code != http.StatusNotFound {
		t.Fatalf("update missing doc: %d", code)
	}
	if code, _ = do(t, srv, "PUT", "/indexes/i/documents/"+id, `{"content":""}`); code != http.StatusBadRequest {
		t.Fatalf("update empty content: %d", code)
	}
	if code, _ = do(t, srv, "PUT", "/indexes/i/documents/"+id, `{`); code != http.StatusBadRequest {
		t.Fatalf("update bad body: %d", code)
	}

	// Delete: present, missing.
	if code, _ = do(t, srv, "DELETE", "/indexes/i/documents/"+id, ""); code != http.StatusNoContent {
		t.Fatalf("delete doc: %d", code)
	}
	if code, _ = do(t, srv, "DELETE", "/indexes/i/documents/"+id, ""); code != http.StatusNotFound {
		t.Fatalf("delete missing doc: %d", code)
	}
}

func TestCategoriesEndpoints(t *testing.T) {
	srv := newStack(t)
	do(t, srv, "POST", "/indexes", `{"name":"i"}`)
	do(t, srv, "POST", "/indexes/i/documents", `{"id":"a","content":"sony wireless headphones audio"}`)
	do(t, srv, "POST", "/indexes/i/documents", `{"id":"m","content":"medical stethoscope"}`)

	code, cats := doList(t, srv, "GET", "/indexes/i/categories", "")
	if code != http.StatusOK {
		t.Fatalf("list categories: %d", code)
	}
	if len(cats) != 2 {
		t.Fatalf("expected 2 categories, got %d: %v", len(cats), cats)
	}
	first := cats[0].(map[string]any)
	name := first["name"].(string)
	if first["doc_count"].(float64) < 1 {
		t.Fatalf("category %q should have >=1 doc, got %v", name, first["doc_count"])
	}
	if _, ok := first["created_at"]; !ok {
		t.Fatalf("category summary missing created_at")
	}
	if _, ok := first["variance"]; !ok {
		t.Fatalf("category summary missing variance")
	}
	if _, ok := first["should_split"]; !ok {
		t.Fatalf("category summary missing should_split")
	}

	// Get a real category and a missing one.
	code, body := do(t, srv, "GET", "/indexes/i/categories/"+name, "")
	if code != http.StatusOK {
		t.Fatalf("get category: %d", code)
	}
	if body["vector_dims"].(float64) == 0 {
		t.Fatalf("expected non-zero vector_dims, got %v", body["vector_dims"])
	}
	if _, ok := body["variance"]; !ok {
		t.Fatalf("category detail missing variance")
	}
	if _, ok := body["should_split"]; !ok {
		t.Fatalf("category detail missing should_split")
	}
	if code, _ = do(t, srv, "GET", "/indexes/i/categories/nope", ""); code != http.StatusNotFound {
		t.Fatalf("get missing category: %d", code)
	}
}

func TestSearchValidation(t *testing.T) {
	srv := newStack(t)
	do(t, srv, "POST", "/indexes", `{"name":"i"}`)

	// Missing q.
	if code, _ := do(t, srv, "POST", "/indexes/i/search", `{"limit":5}`); code != http.StatusBadRequest {
		t.Fatalf("missing q: %d", code)
	}
	// Malformed body.
	if code, _ := do(t, srv, "POST", "/indexes/i/search", `{`); code != http.StatusBadRequest {
		t.Fatalf("bad body: %d", code)
	}
	// Missing index.
	if code, _ := do(t, srv, "POST", "/indexes/missing/search", `{"q":"x"}`); code != http.StatusNotFound {
		t.Fatalf("missing index: %d", code)
	}
}

func TestHealthReportsIndexCount(t *testing.T) {
	srv := newStack(t)
	code, body := do(t, srv, "GET", "/health", "")
	if code != http.StatusOK || body["status"] != "ok" {
		t.Fatalf("health: %d %v", code, body)
	}
	if body["index_count"].(float64) != 0 {
		t.Fatalf("expected 0 indexes, got %v", body["index_count"])
	}
	do(t, srv, "POST", "/indexes", `{"name":"i"}`)
	_, body = do(t, srv, "GET", "/health", "")
	if body["index_count"].(float64) != 1 {
		t.Fatalf("expected 1 index, got %v", body["index_count"])
	}
}

func TestBatchDocuments(t *testing.T) {
	dir := t.TempDir()
	mgr := NewManager(dir, fakeEmbedder{}, engine.Config{
		CategoryThreshold: 0.6, MaxCategoriesPerDoc: 3, MaxCategories: 10, TopNCategories: 3,
	})
	srv := httptest.NewServer(mgr.Routes())
	do(t, srv, "POST", "/indexes", `{"name":"bulk"}`)

	// Two valid docs + one with empty content (reported, not fatal).
	body := `{"documents":[
		{"id":"a","content":"sony wireless headphones audio"},
		{"id":"b","content":"medical stethoscope"},
		{"id":"c","content":""}
	]}`
	code, resp := do(t, srv, "POST", "/indexes/bulk/documents/batch", body)
	if code != http.StatusCreated {
		t.Fatalf("batch: got %d (%v)", code, resp)
	}
	if resp["indexed"].(float64) != 2 {
		t.Fatalf("indexed = %v, want 2", resp["indexed"])
	}
	if resp["failed"].(float64) != 1 {
		t.Fatalf("failed = %v, want 1", resp["failed"])
	}
	srv.Close()

	// The single end-of-batch persist must have written every indexed doc: a
	// fresh manager over the same dir reloads them.
	mgr2 := NewManager(dir, fakeEmbedder{}, engine.Config{})
	if err := mgr2.LoadExisting(); err != nil {
		t.Fatalf("reload: %v", err)
	}
	srv2 := httptest.NewServer(mgr2.Routes())
	defer srv2.Close()
	for _, id := range []string{"a", "b"} {
		if code, _ := do(t, srv2, "GET", "/indexes/bulk/documents/"+id, ""); code != http.StatusOK {
			t.Fatalf("reloaded doc %s: got %d", id, code)
		}
	}
}

func TestBatchEmptyAndBadBody(t *testing.T) {
	srv := newStack(t)
	do(t, srv, "POST", "/indexes", `{"name":"bulk"}`)

	// Empty documents list -> 400.
	if code, _ := do(t, srv, "POST", "/indexes/bulk/documents/batch", `{"documents":[]}`); code != http.StatusBadRequest {
		t.Fatalf("empty batch: got %d", code)
	}
	// Malformed body -> 400.
	if code, _ := do(t, srv, "POST", "/indexes/bulk/documents/batch", `{`); code != http.StatusBadRequest {
		t.Fatalf("bad body: got %d", code)
	}
	// All-invalid batch -> 400 (nothing indexed).
	if code, r := do(t, srv, "POST", "/indexes/bulk/documents/batch", `{"documents":[{"id":"x","content":""}]}`); code != http.StatusBadRequest {
		t.Fatalf("all-invalid batch: got %d (%v)", code, r)
	}
}

// TestPerIndexConfigOverride verifies that a per-index max_categories_per_doc of
// 1 is honored: a document that overlaps two categories joins only one, whereas
// the server default (3) would let it join both.
func TestPerIndexConfigOverride(t *testing.T) {
	srv := newStack(t)
	do(t, srv, "POST", "/indexes", `{"name":"capped","max_categories_per_doc":1}`)

	// Seed two orthogonal categories.
	do(t, srv, "POST", "/indexes/capped/documents", `{"id":"a","content":"audio"}`)
	do(t, srv, "POST", "/indexes/capped/documents", `{"id":"m","content":"medical"}`)

	// This doc is above threshold to both categories, but the cap is 1.
	_, body := do(t, srv, "POST", "/indexes/capped/documents", `{"id":"both","content":"audio medical"}`)
	cats, _ := body["categories"].([]any)
	if len(cats) != 1 {
		t.Fatalf("max_categories_per_doc=1 should assign exactly 1 category, got %d: %v", len(cats), cats)
	}
}

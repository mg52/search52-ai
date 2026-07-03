package handler

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

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

	// Get a real category and a missing one.
	code, body := do(t, srv, "GET", "/indexes/i/categories/"+name, "")
	if code != http.StatusOK {
		t.Fatalf("get category: %d", code)
	}
	if body["vector_dims"].(float64) == 0 {
		t.Fatalf("expected non-zero vector_dims, got %v", body["vector_dims"])
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

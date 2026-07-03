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

package pipeline

import (
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
)

// testPrompts writes minimal templates whose system prompts carry distinct
// markers ("tagging" vs "describe") so the fake AI server can tell which kind
// of completion is being requested.
func testPrompts(t *testing.T) *ai.Prompts {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"tagging_system.template":     "tagging system, max {{.MaxTags}}",
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

// responder lets each test decide what the fake LLM returns for tagging vs
// description completions, and how text is embedded.
type responder struct {
	tagging     string // assistant content for tagging completions
	description string // assistant content for description completions
	embed       func(string) []float64
}

func hashEmbed(s string) []float64 {
	v := make([]float64, 8)
	for i, ch := range s {
		v[i%8] += float64(ch)
	}
	if vecIsZero(v) {
		v[0] = 1
	}
	return v
}

func vecIsZero(v []float64) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

func newAI(t *testing.T, r responder) (*ai.LLMClient, *ai.EmbeddingClient) {
	t.Helper()
	if r.embed == nil {
		r.embed = hashEmbed
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch req.URL.Path {
		case "/chat/completions":
			raw, _ := io.ReadAll(req.Body)
			body := string(raw)
			content := r.tagging
			if strings.Contains(body, "describe tag") {
				content = r.description
			}
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"content": content}}},
			})
		case "/embeddings":
			var in struct {
				Input string `json:"input"`
			}
			json.NewDecoder(req.Body).Decode(&in)
			json.NewEncoder(w).Encode(map[string]any{
				"data": []map[string]any{{"embedding": r.embed(in.Input)}},
			})
		default:
			http.NotFound(w, req)
		}
	}))
	t.Cleanup(srv.Close)
	return ai.NewLLMClient(srv.URL, "k", "m"), ai.NewEmbeddingClient(srv.URL, "k", "m")
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("condition not met within timeout")
}

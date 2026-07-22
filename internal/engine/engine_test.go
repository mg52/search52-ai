package engine

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/mg52/search52-ai/internal/ai"
)

type fakeEmbedder struct{}

func (fakeEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	return []float32{1, 0, 0}, nil
}

type fakeChatter struct{}

func (fakeChatter) Complete(_ context.Context, _, _ string) (string, error) {
	return `{"categories":["c"]}`, nil
}

func testPrompts(t *testing.T) *ai.Prompts {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"tagging_system.template":     "sys",
		"tagging_user.template":       "user {{.Content}}",
		"description_system.template": "sys",
		"description_user.template":   "user {{.CategoryName}}",
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

func TestNewDefaultsToEmbedding(t *testing.T) {
	idx, err := New("", "t", Deps{Embedder: fakeEmbedder{}}, Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if idx.Kind() != "embedding" {
		t.Fatalf("Kind() = %q, want embedding", idx.Kind())
	}
}

func TestNewEmbeddingRequiresEmbedder(t *testing.T) {
	if _, err := New(KindEmbedding, "t", Deps{}, Config{}); err == nil {
		t.Fatal("expected error without an embedder")
	}
}

func TestNewLLMRequiresDeps(t *testing.T) {
	if _, err := New(KindLLM, "t", Deps{}, Config{}); err == nil {
		t.Fatal("expected error without llm deps")
	}
	idx, err := New(KindLLM, "t", Deps{Embedder: fakeEmbedder{}, LLMClient: fakeChatter{}, Prompts: testPrompts(t)}, Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if idx.Kind() != "llm" {
		t.Fatalf("Kind() = %q, want llm", idx.Kind())
	}
}

func TestNewUnknownKind(t *testing.T) {
	if _, err := New(Kind("bogus"), "t", Deps{}, Config{}); err == nil {
		t.Fatal("expected error for unknown kind")
	}
}

func TestLoadDispatchesByKind(t *testing.T) {
	ctx := context.Background()
	deps := Deps{Embedder: fakeEmbedder{}, LLMClient: fakeChatter{}, Prompts: testPrompts(t)}

	embDir := t.TempDir()
	embIdx, err := New(KindEmbedding, "e", deps, Config{})
	if err != nil {
		t.Fatalf("New embedding: %v", err)
	}
	if _, err := embIdx.Process(ctx, "d1", "hello"); err != nil {
		t.Fatalf("process: %v", err)
	}
	if err := embIdx.Save(embDir); err != nil {
		t.Fatalf("save: %v", err)
	}

	llmDir := t.TempDir()
	llmIdx, err := New(KindLLM, "l", deps, Config{})
	if err != nil {
		t.Fatalf("New llm: %v", err)
	}
	if _, err := llmIdx.Process(ctx, "d1", "hello"); err != nil {
		t.Fatalf("process: %v", err)
	}
	if err := llmIdx.Save(llmDir); err != nil {
		t.Fatalf("save: %v", err)
	}

	gotEmb, err := Load(embDir, deps)
	if err != nil {
		t.Fatalf("Load embedding dir: %v", err)
	}
	if gotEmb.Kind() != "embedding" {
		t.Fatalf("Load(embDir).Kind() = %q, want embedding", gotEmb.Kind())
	}

	gotLLM, err := Load(llmDir, deps)
	if err != nil {
		t.Fatalf("Load llm dir: %v", err)
	}
	if gotLLM.Kind() != "llm" {
		t.Fatalf("Load(llmDir).Kind() = %q, want llm", gotLLM.Kind())
	}

	if _, err := Load(t.TempDir(), deps); err == nil {
		t.Fatal("expected error loading an empty dir")
	}
}

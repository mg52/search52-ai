package pipeline

import (
	"context"
	"testing"

	"github.com/mg52/search52-ai/internal/store"
)

func newEmbedder(t *testing.T, r responder) (*Embedder, store.Store) {
	t.Helper()
	prompts := testPrompts(t)
	llm, emb := newAI(t, r)
	st := store.NewMemoryStore()
	return NewEmbedder(llm, emb, prompts, st), st
}

func seedTagWithDoc(st store.Store, tag string) {
	st.SaveTag(store.Tag{Name: tag})
	st.SaveDocument(store.Document{ID: "d1", Content: "example document", Tags: []string{tag}})
}

func TestProcessTagSuccess(t *testing.T) {
	e, st := newEmbedder(t, responder{description: `{"description":"dense keywords"}`})
	seedTagWithDoc(st, "audio")

	if err := e.processTag(context.Background(), "audio"); err != nil {
		t.Fatalf("processTag: %v", err)
	}
	tag, _ := st.GetTag("audio")
	if tag.Description != "dense keywords" {
		t.Errorf("description = %q, want 'dense keywords'", tag.Description)
	}
	if tag.Norm <= 0 || len(tag.Vector) == 0 {
		t.Errorf("vector/norm not set: norm=%v dims=%d", tag.Norm, len(tag.Vector))
	}
	if len(tag.Examples) == 0 {
		t.Error("examples not populated")
	}
}

func TestProcessTagCodeFence(t *testing.T) {
	e, st := newEmbedder(t, responder{description: "```json\n{\"description\":\"fenced\"}\n```"})
	seedTagWithDoc(st, "audio")

	if err := e.processTag(context.Background(), "audio"); err != nil {
		t.Fatalf("processTag: %v", err)
	}
	tag, _ := st.GetTag("audio")
	if tag.Description != "fenced" {
		t.Errorf("description = %q, want 'fenced'", tag.Description)
	}
}

func TestProcessTagPlainTextFallback(t *testing.T) {
	// Model never returns JSON; after retries the prose is used as-is.
	e, st := newEmbedder(t, responder{description: "just prose, no json here"})
	seedTagWithDoc(st, "audio")

	if err := e.processTag(context.Background(), "audio"); err != nil {
		t.Fatalf("processTag: %v", err)
	}
	tag, _ := st.GetTag("audio")
	if tag.Description != "just prose, no json here" {
		t.Errorf("description = %q, want plain prose fallback", tag.Description)
	}
	if tag.Norm <= 0 {
		t.Error("vector should still be generated for fallback description")
	}
}

func TestProcessTagMissing(t *testing.T) {
	e, _ := newEmbedder(t, responder{description: `{"description":"x"}`})
	if err := e.processTag(context.Background(), "ghost"); err == nil {
		t.Fatal("expected error for missing tag")
	}
}

func TestProcessTagsSwallowsErrors(t *testing.T) {
	// ProcessTags must not panic on a missing tag; it logs and continues.
	e, st := newEmbedder(t, responder{description: `{"description":"x"}`})
	seedTagWithDoc(st, "audio")
	e.ProcessTags(context.Background(), []string{"ghost", "audio"})
	tag, _ := st.GetTag("audio")
	if tag.Norm <= 0 {
		t.Error("valid tag should still be processed after a missing one")
	}
}

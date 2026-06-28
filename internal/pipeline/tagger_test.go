package pipeline

import (
	"context"
	"testing"

	"github.com/mg52/search52-ai/internal/store"
	"github.com/mg52/search52-ai/internal/vec"
)

func newTagger(t *testing.T, r responder) (*Tagger, store.Store) {
	t.Helper()
	prompts := testPrompts(t)
	llm, emb := newAI(t, r)
	st := store.NewMemoryStore()
	embedder := NewEmbedder(llm, emb, prompts, st)
	tagger := NewTagger(llm, emb, prompts, st, 8, 0.6, embedder)
	return tagger, st
}

func TestTagByLLMSuccess(t *testing.T) {
	tagger, st := newTagger(t, responder{
		tagging:     `{"tags":["audio","wireless"]}`,
		description: `{"description":"rich keyword description"}`,
	})

	doc, err := tagger.TagByLLM(context.Background(), store.Document{ID: "d1", Content: "sony headphones"})
	if err != nil {
		t.Fatalf("TagByLLM: %v", err)
	}
	if len(doc.Tags) != 2 || doc.Tags[0] != "audio" || doc.Tags[1] != "wireless" {
		t.Fatalf("tags = %v, want [audio wireless]", doc.Tags)
	}
	// Document persisted and indexed.
	if _, ok := st.GetDocument("d1"); !ok {
		t.Error("document not saved")
	}
	if st.DocCountByTag("audio") != 1 {
		t.Errorf("DocCountByTag(audio) = %d, want 1", st.DocCountByTag("audio"))
	}
	// New tags created.
	if _, ok := st.GetTag("audio"); !ok {
		t.Error("tag audio not created")
	}
	// Background embedding eventually populates vectors.
	waitFor(t, func() bool {
		a, _ := st.GetTag("audio")
		w, _ := st.GetTag("wireless")
		return a.Norm > 0 && len(a.Vector) > 0 && w.Norm > 0
	})
	a, _ := st.GetTag("audio")
	if a.Description != "rich keyword description" {
		t.Errorf("description = %q", a.Description)
	}
}

func TestTagByLLMReusesExistingTag(t *testing.T) {
	tagger, st := newTagger(t, responder{
		tagging:     `{"tags":["audio"]}`,
		description: `{"description":"d"}`,
	})
	// Pre-seed an existing tag; LLM returns it again -> not recreated, no new embed.
	st.SaveTag(store.Tag{Name: "audio"})

	if _, err := tagger.TagByLLM(context.Background(), store.Document{ID: "d1", Content: "x"}); err != nil {
		t.Fatalf("TagByLLM: %v", err)
	}
	if st.TagCount() != 1 {
		t.Errorf("TagCount = %d, want 1 (no duplicate tag)", st.TagCount())
	}
	if st.DocCountByTag("audio") != 1 {
		t.Errorf("DocCountByTag(audio) = %d, want 1", st.DocCountByTag("audio"))
	}
}

func TestTagByLLMInvalidJSON(t *testing.T) {
	tagger, _ := newTagger(t, responder{tagging: "this is not json at all"})
	if _, err := tagger.TagByLLM(context.Background(), store.Document{ID: "d1", Content: "x"}); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestTagByLLMEmptyTags(t *testing.T) {
	tagger, _ := newTagger(t, responder{tagging: `{"tags":[]}`})
	if _, err := tagger.TagByLLM(context.Background(), store.Document{ID: "d1", Content: "x"}); err == nil {
		t.Fatal("expected error for empty tag list")
	}
}

func TestTagByLLMBlankTagsError(t *testing.T) {
	// Model emits a blank tag every time -> no usable tags -> error (after retries).
	tagger, st := newTagger(t, responder{tagging: `{"tags":[""]}`})
	if _, err := tagger.TagByLLM(context.Background(), store.Document{ID: "d1", Content: "x"}); err == nil {
		t.Fatal("expected error when all tags are blank")
	}
	// No empty-named tag should leak into the store.
	if _, ok := st.GetTag(""); ok {
		t.Error("blank tag was saved to the store")
	}
	if st.TagCount() != 0 {
		t.Errorf("TagCount = %d, want 0", st.TagCount())
	}
}

func TestTagByLLMCleansTags(t *testing.T) {
	// Blanks, whitespace, and duplicates are stripped; valid tags survive.
	tagger, st := newTagger(t, responder{
		tagging:     `{"tags":["audio","",""," ","audio","wireless"]}`,
		description: `{"description":"d"}`,
	})
	doc, err := tagger.TagByLLM(context.Background(), store.Document{ID: "d1", Content: "x"})
	if err != nil {
		t.Fatalf("TagByLLM: %v", err)
	}
	if len(doc.Tags) != 2 || doc.Tags[0] != "audio" || doc.Tags[1] != "wireless" {
		t.Errorf("tags = %v, want [audio wireless]", doc.Tags)
	}
	if st.DocCountByTag("audio") != 1 {
		t.Errorf("DocCountByTag(audio) = %d, want 1", st.DocCountByTag("audio"))
	}
}

func TestTagByEmbedding(t *testing.T) {
	tagger, st := newTagger(t, responder{})
	// Seed two tags with vectors; identical vectors -> cosine 1.0 >= threshold.
	v := hashEmbed("doc content")
	st.SaveTag(store.Tag{Name: "audio", Vector: v, Norm: vec.Norm(v)})
	st.SaveTag(store.Tag{Name: "novec"}) // no vector -> skipped

	res, err := tagger.TagByEmbedding(context.Background(), store.Document{ID: "d1", Content: "doc content"})
	if err != nil {
		t.Fatalf("TagByEmbedding: %v", err)
	}
	if len(res.Document.Tags) != 1 || res.Document.Tags[0] != "audio" {
		t.Fatalf("tags = %v, want [audio]", res.Document.Tags)
	}
	if _, ok := res.TagScores["audio"]; !ok {
		t.Error("expected tag_scores to include audio")
	}
	if st.DocCountByTag("audio") != 1 {
		t.Errorf("DocCountByTag(audio) = %d, want 1", st.DocCountByTag("audio"))
	}
}

func TestTagByEmbeddingNoMatch(t *testing.T) {
	// Document always embeds to [1,0]; tag vector is orthogonal [0,1] -> cosine 0.
	tagger, st := newTagger(t, responder{
		embed: func(string) []float64 { return []float64{1, 0} },
	})
	st.SaveTag(store.Tag{Name: "audio", Vector: []float64{0, 1}, Norm: 1})

	if _, err := tagger.TagByEmbedding(context.Background(), store.Document{ID: "d1", Content: "x"}); err == nil {
		t.Fatal("expected error when no tag is above threshold")
	}
}

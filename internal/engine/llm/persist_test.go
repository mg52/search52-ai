package llm

import (
	"context"
	"testing"
	"time"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	chatter := &dispatchChatter{
		taggingResp:     `{"categories":["audio_devices"]}`,
		descriptionResp: `{"description":"wireless audio headphones"}`,
	}
	prompts := newTestPrompts(t)
	idx := New("t", chatter, fakeEmbedder{}, prompts, Config{MaxDocsPerCategory: 25})
	ctx := context.Background()

	if _, err := idx.Process(ctx, "a1", "sony wireless headphones audio"); err != nil {
		t.Fatalf("process: %v", err)
	}
	waitUntil(t, time.Second, func() bool {
		c, ok := idx.GetCategory("audio_devices")
		return ok && c.VectorDims > 0
	})

	dir := t.TempDir()
	if err := idx.Save(dir); err != nil {
		t.Fatalf("save: %v", err)
	}

	got, err := Load(dir, chatter, fakeEmbedder{}, prompts)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if got.maxDocsPerCategory != idx.maxDocsPerCategory {
		t.Fatalf("maxDocsPerCategory mismatch: got %d, want %d", got.maxDocsPerCategory, idx.maxDocsPerCategory)
	}
	if got.DocCount() != idx.DocCount() || got.CategoryCount() != idx.CategoryCount() {
		t.Fatalf("counts mismatch: docs %d/%d cats %d/%d", got.DocCount(), idx.DocCount(), got.CategoryCount(), idx.CategoryCount())
	}

	d, ok := got.GetDocument("a1")
	if !ok || len(d.Categories) != 1 || d.Categories[0] != "audio_devices" {
		t.Fatalf("doc a1 not round-tripped correctly: %+v", d)
	}

	c, ok := got.GetCategory("audio_devices")
	if !ok || c.VectorDims == 0 || c.Description == nil || *c.Description == "" {
		t.Fatalf("category audio_devices not round-tripped correctly: %+v", c)
	}

	res, err := got.Search(ctx, "wireless audio headphones", 5)
	if err != nil {
		t.Fatalf("search after load: %v", err)
	}
	if len(res.Documents) == 0 {
		t.Fatal("reloaded index returned no results")
	}
}

func TestLoadMissingSnapshot(t *testing.T) {
	if _, err := Load(t.TempDir(), &dispatchChatter{}, fakeEmbedder{}, newTestPrompts(t)); err == nil {
		t.Fatal("Load on a dir without a snapshot should error")
	}
}

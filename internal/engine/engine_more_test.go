package engine

import (
	"context"
	"testing"
)

func customEngine(threshold float64, maxPerDoc, maxCategories, topN int) *SearchEngine {
	return New("t", fakeEmbedder{}, Config{
		CategoryThreshold:   threshold,
		MaxCategoriesPerDoc: maxPerDoc,
		MaxCategories:       maxCategories,
		TopNCategories:      topN,
	})
}

func TestProcessValidation(t *testing.T) {
	se := newEngine()
	ctx := context.Background()

	if _, err := se.Process(ctx, "", "sony audio"); err == nil {
		t.Fatal("empty id should error")
	}
	if _, err := se.Process(ctx, "x", ""); err == nil {
		t.Fatal("empty content should error")
	}
	// Content with no known vocab embeds to a zero vector and must be rejected.
	if _, err := se.Process(ctx, "x", "qwerty zzz"); err == nil {
		t.Fatal("zero-vector embedding should error")
	}
	if se.DocCount() != 0 {
		t.Fatalf("no document should have been stored, got %d", se.DocCount())
	}
}

func TestMaxCategoriesCapFallback(t *testing.T) {
	se := customEngine(0.6, 3, 1, 3) // cap of a single category
	ctx := context.Background()

	se.Process(ctx, "a", "audio")
	// "medical" is orthogonal to the only category, so it would normally spawn a
	// new one — but the cap is reached, so it falls back to the nearest.
	d, _ := se.Process(ctx, "m", "medical")
	if len(d.Categories) != 1 || d.Categories[0] != "category1" {
		t.Fatalf("capped doc should fall back to category1, got %v", d.Categories)
	}
	if se.CategoryCount() != 1 {
		t.Fatalf("cap should hold category count at 1, got %d", se.CategoryCount())
	}
}

func TestMaxPerDocCap(t *testing.T) {
	se := customEngine(0.6, 1, 10, 3) // a doc may join at most one category
	ctx := context.Background()

	se.Process(ctx, "a", "audio")
	se.Process(ctx, "m", "medical")
	// "audio medical" is above threshold to both categories, but the per-doc cap
	// keeps it in only the single nearest.
	d, _ := se.Process(ctx, "both", "audio medical")
	if len(d.Categories) != 1 {
		t.Fatalf("per-doc cap should assign exactly 1 category, got %v", d.Categories)
	}
}

func TestRemoveNonExistent(t *testing.T) {
	se := newEngine()
	if _, ok := se.Remove("ghost"); ok {
		t.Fatal("removing a missing doc should report false")
	}
}

func TestListAndCounts(t *testing.T) {
	se := newEngine()
	ctx := context.Background()
	se.Process(ctx, "a", "sony wireless headphones audio")
	se.Process(ctx, "m", "medical stethoscope")

	cats := se.ListCategories()
	if len(cats) != 2 {
		t.Fatalf("expected 2 categories, got %d", len(cats))
	}
	total := 0
	for _, c := range cats {
		total += se.DocCountByCategory(c.Name)
	}
	if total != 2 {
		t.Fatalf("category doc counts should sum to 2, got %d", total)
	}
	if _, ok := se.GetCategory("does-not-exist"); ok {
		t.Fatal("GetCategory on a missing name should report false")
	}
}

func TestUpdatePreservesCreatedAt(t *testing.T) {
	se := newEngine()
	ctx := context.Background()
	before, _ := se.Process(ctx, "d", "medical stethoscope")
	after, _ := se.Process(ctx, "d", "sony wireless headphones audio")
	if !after.CreatedAt.Equal(before.CreatedAt) {
		t.Fatalf("CreatedAt should be preserved across update: before=%v after=%v", before.CreatedAt, after.CreatedAt)
	}
}

func TestSearchDeduplicatesMultiCategoryDoc(t *testing.T) {
	se := newEngine()
	ctx := context.Background()
	se.Process(ctx, "a", "audio")
	se.Process(ctx, "m", "medical")
	// This doc joins both categories; search must not return it twice even though
	// it is a member of two of the selected categories.
	se.Process(ctx, "both", "audio medical")

	res, err := se.Search(ctx, "audio medical", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	seen := map[string]int{}
	for _, r := range res.Documents {
		seen[r.Document.ID]++
	}
	if seen["both"] != 1 {
		t.Fatalf("doc 'both' should appear exactly once, got %d", seen["both"])
	}
}

package engine

import (
	"context"
	"strings"
	"testing"
)

var vocab = map[string]int{
	"sony": 0, "headphones": 1, "wireless": 2, "audio": 3,
	"medical": 4, "stethoscope": 5, "blood": 6, "pressure": 7,
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

func newEngine() *SearchEngine {
	return New("t", fakeEmbedder{}, Config{CategoryThreshold: 0.6, MaxCategoriesPerDoc: 3, MaxCategories: 10, TopNCategories: 3})
}

func TestClusteringSeparatesDomains(t *testing.T) {
	se := newEngine()
	ctx := context.Background()
	for id, c := range map[string]string{
		"a1": "sony wireless headphones audio",
		"a2": "wireless headphones audio",
		"m1": "medical stethoscope",
		"m2": "medical blood pressure",
	} {
		if _, err := se.Process(ctx, id, c); err != nil {
			t.Fatalf("process %s: %v", id, err)
		}
	}
	// Audio and medical docs are orthogonal (no shared vocab), so they must land
	// in different categories.
	a1, _ := se.GetDocument("a1")
	m1, _ := se.GetDocument("m1")
	for _, ac := range a1.Categories {
		for _, mc := range m1.Categories {
			if ac == mc {
				t.Fatalf("audio and medical share category %q", ac)
			}
		}
	}
}

func TestSearchTopKAndEmptyIsArray(t *testing.T) {
	se := newEngine()
	ctx := context.Background()
	se.Process(ctx, "a1", "sony wireless headphones audio")
	se.Process(ctx, "a2", "wireless headphones audio")
	se.Process(ctx, "m1", "medical stethoscope")

	res, err := se.Search(ctx, "wireless audio headphones", 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res.Documents) != 1 {
		t.Fatalf("limit=1 should cap results, got %d", len(res.Documents))
	}
	if id := res.Documents[0].Document.ID; id != "a1" && id != "a2" {
		t.Fatalf("expected audio doc, got %q", id)
	}

	// A query with no vocab overlap yields zero docs but a non-nil slice.
	res, _ = se.Search(ctx, "unrelated gibberish", 5)
	if res.Documents == nil {
		t.Fatalf("Documents must be non-nil (marshals to [])")
	}
}

func TestRemovePrunesEmptyCategory(t *testing.T) {
	se := newEngine()
	ctx := context.Background()
	doc, _ := se.Process(ctx, "only", "medical stethoscope")
	cat := doc.Categories[0]
	if _, ok := se.GetCategory(cat); !ok {
		t.Fatalf("category %q should exist", cat)
	}
	if _, ok := se.Remove("only"); !ok {
		t.Fatalf("remove failed")
	}
	if _, ok := se.GetCategory(cat); ok {
		t.Fatalf("category %q should be pruned after its last doc left", cat)
	}
}

func TestUpdateReclusters(t *testing.T) {
	se := newEngine()
	ctx := context.Background()
	se.Process(ctx, "d", "medical stethoscope")
	before, _ := se.GetDocument("d")

	// Re-process the same id with audio content; it must leave the medical
	// category and its old category should be pruned.
	after, _ := se.Process(ctx, "d", "sony wireless headphones audio")
	if strings.Join(before.Categories, ",") == strings.Join(after.Categories, ",") {
		t.Fatalf("categories should change after content update")
	}
	if se.DocCount() != 1 {
		t.Fatalf("update must not duplicate the doc, got %d", se.DocCount())
	}
}

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

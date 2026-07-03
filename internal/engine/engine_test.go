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

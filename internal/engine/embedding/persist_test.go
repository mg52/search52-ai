package embedding

import (
	"context"
	"testing"
)

func float32SliceEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestSaveLoadRoundTrip(t *testing.T) {
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

	dir := t.TempDir()
	if err := se.Save(dir); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := Load(dir, fakeEmbedder{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Config and counters survive the round trip.
	if got.threshold != se.threshold || got.maxPerDoc != se.maxPerDoc ||
		got.maxCategories != se.maxCategories || got.topN != se.topN {
		t.Fatalf("config mismatch: got %+v", got)
	}
	if got.nextCategoryID != se.nextCategoryID {
		t.Fatalf("nextCategoryID = %d, want %d", got.nextCategoryID, se.nextCategoryID)
	}
	if got.DocCount() != se.DocCount() || got.CategoryCount() != se.CategoryCount() {
		t.Fatalf("counts mismatch: docs %d/%d cats %d/%d",
			got.DocCount(), se.DocCount(), got.CategoryCount(), se.CategoryCount())
	}

	// Every document round-trips with its vector, norm, and memberships intact.
	for id, want := range se.docs {
		g, ok := got.docs[id]
		if !ok {
			t.Fatalf("doc %s missing after load", id)
		}
		if g.Content != want.Content || g.Norm != want.Norm || !float32SliceEqual(g.Vector, want.Vector) {
			t.Fatalf("doc %s content/vector mismatch", id)
		}
	}

	// Centroids and their cached norms round-trip exactly (not recomputed).
	for name, want := range se.categories {
		g, ok := got.categories[name]
		if !ok {
			t.Fatalf("category %s missing after load", name)
		}
		if g.Count != want.Count || g.Norm != want.Norm || !float32SliceEqual(g.Centroid, want.Centroid) {
			t.Fatalf("category %s centroid/norm/count mismatch", name)
		}
		if len(got.catDocs[name]) != len(se.catDocs[name]) {
			t.Fatalf("category %s membership size mismatch", name)
		}
	}

	// A reloaded index still searches correctly.
	res, err := got.Search(ctx, "wireless audio headphones", 2)
	if err != nil {
		t.Fatalf("search after load: %v", err)
	}
	if len(res.Documents) == 0 {
		t.Fatal("reloaded index returned no results")
	}
}

func TestLoadMissingSnapshot(t *testing.T) {
	if _, err := Load(t.TempDir(), fakeEmbedder{}); err == nil {
		t.Fatal("Load on a dir without a snapshot should error")
	}
}

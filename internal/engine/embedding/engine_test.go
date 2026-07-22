package embedding

import (
	"context"
	"math"
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

func TestCategoryVarianceFlagsShouldSplit(t *testing.T) {
	// threshold 0 + a single category cap forces every doc into one category
	// regardless of similarity, so its member similarities vary a lot.
	se := New("t", fakeEmbedder{}, Config{
		CategoryThreshold: 0, MaxCategoriesPerDoc: 1, MaxCategories: 1, TopNCategories: 3,
		VarianceThreshold: 0, VarianceMinCount: 2, // 2 lets this 3-member category clear the gate
	})
	ctx := context.Background()

	doc1, err := se.Process(ctx, "a1", "sony wireless headphones audio")
	if err != nil {
		t.Fatalf("process a1: %v", err)
	}
	catName := doc1.Categories[0]
	cat, ok := se.categories[catName]
	if !ok {
		t.Fatalf("category %q missing", catName)
	}
	if cat.ShouldSplit || cat.Variance != 0 {
		t.Fatalf("category founder must not seed the Welford stream (trivial self-similarity), got variance %v", cat.Variance)
	}

	// First real member: still only one Welford sample, so variance stays 0
	// (Variance needs >=2 samples) and ShouldSplit can't fire yet.
	if _, err := se.Process(ctx, "m1", "medical stethoscope"); err != nil {
		t.Fatalf("process m1: %v", err)
	}
	cat = se.categories[catName]
	if cat.ShouldSplit || cat.Variance != 0 {
		t.Fatalf("a single real sample has no variance yet, should not flag split, got %v", cat.Variance)
	}

	// Second real member shares vocab with both a1 and m1, so it lands at a
	// different, non-zero similarity to the now-updated centroid — joining
	// the same category (threshold 0) and widening the running variance
	// past VarianceThreshold=0.
	if _, err := se.Process(ctx, "m2", "audio medical"); err != nil {
		t.Fatalf("process m2: %v", err)
	}
	cat = se.categories[catName]
	if !cat.ShouldSplit {
		t.Fatalf("dissimilar second real member should push variance above threshold and flip ShouldSplit")
	}
	if cat.Variance <= 0 {
		t.Fatalf("expected positive variance, got %v", cat.Variance)
	}
}

// TestVarianceStabilizesForLongHomogeneousStream shows that a category kept
// fed with similar (not identical) similarity values does not shrink toward
// zero forever: the early monotonic decline seen with few samples is just a
// noisy small-n variance estimate settling toward the stream's true, fixed
// variance. Feeding 1000 more values from the same distribution does not
// keep driving it down further.
func TestVarianceStabilizesForLongHomogeneousStream(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{})
	se.categories["c"] = &Category{Name: "c"}

	// Alternating mean±delta has an exact, fixed population variance of
	// delta^2. Sample variance (n-1 denominator) converges to it from above
	// as n grows: n/(n-1) * delta^2.
	const mean, delta = 0.8, 0.05
	for i := 0; i < 1000; i++ {
		sim := mean - delta
		if i%2 == 1 {
			sim = mean + delta
		}
		se.updateVarianceLocked("c", float32(sim))
	}
	got := se.categories["c"].Variance
	want := delta * delta
	if diff := math.Abs(got - want); diff > 5e-4 {
		t.Fatalf("variance should stabilize near %v for a bounded homogeneous stream, got %v (diff %v)", want, got, diff)
	}
}

// TestVarianceRisesWhenDistantDocumentsJoin shows the other half of the
// story: once a category's variance has settled low on a tight cluster, a
// run of genuinely distant documents joining it (still passing the category
// threshold, just far from the established members) pushes the variance back
// up — it does not keep declining regardless of what joins.
func TestVarianceRisesWhenDistantDocumentsJoin(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{}) // VarianceThreshold defaults to 0.02
	se.categories["c"] = &Category{Name: "c"}

	// 200 members all at the exact same similarity: a perfectly tight
	// cluster, variance settles at (not just near) zero. Count is bumped
	// alongside each update, mirroring Process's real addToCentroidLocked +
	// updateVarianceLocked order, so this category also clears the default
	// VarianceMinCount (100) gate.
	for i := 0; i < 200; i++ {
		se.categories["c"].Count++
		se.updateVarianceLocked("c", 0.85)
	}
	stable := se.categories["c"].Variance
	if stable != 0 {
		t.Fatalf("expected exactly 0 variance for an identical-similarity stream, got %v", stable)
	}
	if se.categories["c"].ShouldSplit {
		t.Fatalf("a tight, stable cluster should not be flagged for split")
	}

	// 20 distant documents (sim 0.2 vs the established ~0.85) join the same
	// category. Variance must rise, and with enough of a gap it should clear
	// the default threshold and flag the category.
	for i := 0; i < 20; i++ {
		se.categories["c"].Count++
		se.updateVarianceLocked("c", 0.2)
	}
	got := se.categories["c"].Variance
	if got <= stable {
		t.Fatalf("variance should increase once distant documents join a stable category, got %v (was %v)", got, stable)
	}
	if !se.categories["c"].ShouldSplit {
		t.Fatalf("variance %v should have exceeded the threshold and flagged ShouldSplit", got)
	}
}

// TestShouldSplitRequiresMinimumCount verifies the VarianceMinCount gate: a
// young category must not be flagged for split just because a couple of
// wildly different samples spike its variance — only once it has accrued
// enough members for that variance estimate to be meaningful.
func TestShouldSplitRequiresMinimumCount(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{VarianceThreshold: 0}) // VarianceMinCount defaults to 100
	se.categories["c"] = &Category{Name: "c"}

	// Two far-apart samples give a large variance, comfortably above the
	// zero threshold, but Count is still 0 (< VarianceMinCount).
	se.updateVarianceLocked("c", 0.9)
	se.updateVarianceLocked("c", 0.1)
	if se.categories["c"].Variance <= 0 {
		t.Fatalf("expected positive variance, got %v", se.categories["c"].Variance)
	}
	if se.categories["c"].ShouldSplit {
		t.Fatalf("a young category (Count below VarianceMinCount) must not flag split despite high variance")
	}

	// Once Count clears the default VarianceMinCount (100), the same
	// high-variance category should flag on its next update.
	se.categories["c"].Count = 101
	se.updateVarianceLocked("c", 0.1)
	if !se.categories["c"].ShouldSplit {
		t.Fatalf("a mature category (Count above VarianceMinCount) with variance above threshold should flag ShouldSplit")
	}
}

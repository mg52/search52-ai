package engine

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/mg52/search52-ai/internal/vec"
)

// -------------------- kmeans2 --------------------

func TestKmeans2SeparatesTwoClusters(t *testing.T) {
	vecs := [][]float32{
		{1, 0, 0}, {0.9, 0.1, 0}, {1, 0.05, 0}, // cluster 0
		{0, 0, 1}, {0, 0.1, 0.9}, {0.05, 0, 1}, // cluster 1
	}
	norms := make([]float32, len(vecs))
	for i, v := range vecs {
		norms[i] = vec.Norm(v)
	}

	labels, ok := kmeans2(vecs, norms)
	if !ok {
		t.Fatal("expected a successful split")
	}
	first := labels[0]
	for i := 1; i < 3; i++ {
		if labels[i] != first {
			t.Fatalf("expected first 3 vectors in the same cluster, got labels %v", labels)
		}
	}
	second := labels[3]
	if second == first {
		t.Fatalf("expected the two groups in different clusters, got labels %v", labels)
	}
	for i := 4; i < 6; i++ {
		if labels[i] != second {
			t.Fatalf("expected last 3 vectors in the same cluster, got labels %v", labels)
		}
	}
}

func TestKmeans2TooFewVectors(t *testing.T) {
	if _, ok := kmeans2(nil, nil); ok {
		t.Fatal("kmeans2 on 0 vectors should report ok=false")
	}
	if _, ok := kmeans2([][]float32{{1, 0}}, []float32{1}); ok {
		t.Fatal("kmeans2 on 1 vector should report ok=false")
	}
}

// TestKmeans2IdenticalVectorsStillProducesTwoNonEmptyClusters exercises the
// empty-cluster reseed path: with no real separation to find, a naive
// nearest-centroid assignment could still collapse everything into one
// cluster, which would make a "split" pointless.
func TestKmeans2IdenticalVectorsStillProducesTwoNonEmptyClusters(t *testing.T) {
	vecs := make([][]float32, 5)
	norms := make([]float32, 5)
	for i := range vecs {
		vecs[i] = []float32{1, 2, 3}
		norms[i] = vec.Norm(vecs[i])
	}
	labels, ok := kmeans2(vecs, norms)
	if !ok {
		t.Fatal("expected kmeans2 to still report ok on identical vectors")
	}
	var count0, count1 int
	for _, l := range labels {
		if l == 0 {
			count0++
		} else {
			count1++
		}
	}
	if count0 == 0 || count1 == 0 {
		t.Fatalf("expected both clusters non-empty, got labels %v", labels)
	}
}

// -------------------- mergeSplitLocked --------------------

func addRawDoc(se *SearchEngine, id string, v []float32, categories []string, forcedFallback bool) {
	se.docs[id] = Document{
		ID:             id,
		Content:        id,
		Vector:         v,
		Norm:           vec.Norm(v),
		Categories:     categories,
		ForcedFallback: forcedFallback,
		CreatedAt:      time.Now(),
	}
}

func TestMergeSplitLockedReplacesCategoryAndMigratesMembers(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{MaxCategories: 10})
	se.categories["c"] = &Category{Name: "c", Count: 4}
	se.catDocs["c"] = map[string]struct{}{"a": {}, "b": {}, "x": {}, "y": {}}
	addRawDoc(se, "a", []float32{1, 0, 0}, []string{"c"}, false)
	addRawDoc(se, "b", []float32{0.9, 0.1, 0}, []string{"c"}, false)
	addRawDoc(se, "x", []float32{0, 0, 1}, []string{"c"}, false)
	addRawDoc(se, "y", []float32{0, 0.1, 0.9}, []string{"c"}, false)

	labelOf := map[string]int{"a": 0, "b": 0, "x": 1, "y": 1}
	seedCentroid := [2][]float32{{1.9, 0.1, 0}, {0, 0.1, 1.9}}
	seedNorm := [2]float32{vec.Norm(seedCentroid[0]), vec.Norm(seedCentroid[1])}

	se.mu.Lock()
	_, ok := se.mergeSplitLocked("c", labelOf, seedCentroid, seedNorm)
	se.mu.Unlock()
	if !ok {
		t.Fatal("expected merge to succeed")
	}

	if _, exists := se.GetCategory("c"); exists {
		t.Fatal("old category should be gone")
	}
	cats := se.ListCategories()
	if len(cats) != 2 {
		t.Fatalf("expected exactly 2 new categories, got %d", len(cats))
	}
	total := 0
	for _, cat := range cats {
		total += se.DocCountByCategory(cat.Name)
		if cat.Centroid == nil || cat.Norm == 0 {
			t.Fatalf("category %s missing a real centroid", cat.Name)
		}
	}
	if total != 4 {
		t.Fatalf("expected 4 total members across the new categories, got %d", total)
	}

	for _, id := range []string{"a", "b", "x", "y"} {
		d, ok := se.GetDocument(id)
		if !ok {
			t.Fatalf("doc %s missing after merge", id)
		}
		if len(d.Categories) != 1 || d.Categories[0] == "c" {
			t.Fatalf("doc %s categories not rewritten: %v", id, d.Categories)
		}
		if _, ok := se.GetCategory(d.Categories[0]); !ok {
			t.Fatalf("doc %s points at nonexistent category %s", id, d.Categories[0])
		}
	}

	da, _ := se.GetDocument("a")
	db, _ := se.GetDocument("b")
	dx, _ := se.GetDocument("x")
	dy, _ := se.GetDocument("y")
	if da.Categories[0] != db.Categories[0] {
		t.Fatalf("a and b should land in the same new category, got %s vs %s", da.Categories[0], db.Categories[0])
	}
	if dx.Categories[0] != dy.Categories[0] {
		t.Fatalf("x and y should land in the same new category, got %s vs %s", dx.Categories[0], dy.Categories[0])
	}
	if da.Categories[0] == dx.Categories[0] {
		t.Fatalf("a/b and x/y should land in different new categories")
	}
}

// TestMergeSplitLockedReconcilesConcurrentChanges is the core "engine keeps
// working during a split" guarantee, tested deterministically instead of via
// goroutine timing: labelOf/seedCentroid represent a stale snapshot, while
// se.catDocs/se.docs represent live state that has since diverged from it —
// "b" was removed after the snapshot was taken, and "e" was added after it.
// mergeSplitLocked must reconcile against the live state, not the snapshot.
func TestMergeSplitLockedReconcilesConcurrentChanges(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{MaxCategories: 10})
	se.categories["c"] = &Category{Name: "c", Count: 2}
	// Live membership no longer matches the snapshot: "b" and "x" (known to
	// labelOf) were concurrently removed; "e" (unknown to labelOf) was
	// concurrently added.
	se.catDocs["c"] = map[string]struct{}{"a": {}, "e": {}}
	addRawDoc(se, "a", []float32{1, 0, 0}, []string{"c"}, false)
	addRawDoc(se, "e", []float32{0, 0, 1}, []string{"c"}, false)

	labelOf := map[string]int{"a": 0, "b": 0, "x": 1}
	seedCentroid := [2][]float32{{1.9, 0.1, 0}, {0, 0, 1}} // side 1 seeded from the (now-removed) "x"
	seedNorm := [2]float32{vec.Norm(seedCentroid[0]), vec.Norm(seedCentroid[1])}

	se.mu.Lock()
	_, ok := se.mergeSplitLocked("c", labelOf, seedCentroid, seedNorm)
	se.mu.Unlock()
	if !ok {
		t.Fatal("expected merge to succeed")
	}

	// "b" must not appear anywhere: it no longer existed in the category by
	// the time the merge ran.
	cats := se.ListCategories()
	total := 0
	for _, cat := range cats {
		total += se.DocCountByCategory(cat.Name)
	}
	if total != 2 {
		t.Fatalf("expected exactly the 2 live members (a, e) migrated, got %d", total)
	}

	da, ok := se.GetDocument("a")
	if !ok || len(da.Categories) != 1 || da.Categories[0] == "c" {
		t.Fatalf("a not migrated properly: %+v", da)
	}
	de, ok := se.GetDocument("e")
	if !ok || len(de.Categories) != 1 || de.Categories[0] == "c" {
		t.Fatalf("e (added after the snapshot) not migrated properly: %+v", de)
	}
	if _, ok := se.GetCategory(de.Categories[0]); !ok {
		t.Fatalf("e points at a nonexistent category")
	}
	// "e" is identical in direction to the removed "x", so the nearest-side
	// tie-break must still route it to x's old side, not a's.
	if de.Categories[0] == da.Categories[0] {
		t.Fatalf("e should have landed on the opposite side from a (nearest to the removed x's direction)")
	}
}

// TestMergeSplitLockedPrunesEmptySide covers the case where reconciliation
// against live state leaves one of the two new categories with no members at
// all (every doc that would have gone there was concurrently removed) — it
// must not leave a permanent, empty category behind.
func TestMergeSplitLockedPrunesEmptySide(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{MaxCategories: 10})
	se.categories["c"] = &Category{Name: "c", Count: 1}
	se.catDocs["c"] = map[string]struct{}{"a": {}} // "b" was concurrently removed
	addRawDoc(se, "a", []float32{1, 0, 0}, []string{"c"}, false)

	labelOf := map[string]int{"a": 0, "b": 1}
	seedCentroid := [2][]float32{{1, 0, 0}, {0, 0, 1}}
	seedNorm := [2]float32{vec.Norm(seedCentroid[0]), vec.Norm(seedCentroid[1])}

	se.mu.Lock()
	_, ok := se.mergeSplitLocked("c", labelOf, seedCentroid, seedNorm)
	se.mu.Unlock()
	if !ok {
		t.Fatal("expected merge to succeed")
	}

	cats := se.ListCategories()
	if len(cats) != 1 {
		t.Fatalf("expected the empty side to be pruned, leaving exactly 1 category, got %d", len(cats))
	}
	if cats[0].Count != 1 {
		t.Fatalf("surviving category should have 1 member, got %d", cats[0].Count)
	}
}

func TestMergeSplitLockedRespectsMaxCategoriesCap(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{MaxCategories: 1})
	se.categories["c"] = &Category{Name: "c", Count: 2}
	se.catDocs["c"] = map[string]struct{}{"a": {}, "b": {}}
	addRawDoc(se, "a", []float32{1, 0, 0}, []string{"c"}, false)
	addRawDoc(se, "b", []float32{0, 0, 1}, []string{"c"}, false)

	labelOf := map[string]int{"a": 0, "b": 1}
	seedCentroid := [2][]float32{{1, 0, 0}, {0, 0, 1}}
	seedNorm := [2]float32{1, 1}

	se.mu.Lock()
	_, ok := se.mergeSplitLocked("c", labelOf, seedCentroid, seedNorm)
	se.mu.Unlock()
	if ok {
		t.Fatal("expected merge to be refused: no room under maxCategories for the net +1 category")
	}
	if _, exists := se.GetCategory("c"); !exists {
		t.Fatal("category c should be untouched when the cap blocks the split")
	}
}

// TestSplitCategoryBailsEarlyAtCap: at the maxCategories cap a split can
// never be merged (net +1 category), so splitCategory must bail out before
// clustering — leaving the category intact, ShouldSplit still set (so it can
// retry when the cap frees up), and its splitting entry cleared.
func TestSplitCategoryBailsEarlyAtCap(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{MaxCategories: 1})
	se.categories["c"] = &Category{Name: "c", Count: 2, ShouldSplit: true}
	se.catDocs["c"] = map[string]struct{}{"a": {}, "b": {}}
	addRawDoc(se, "a", []float32{1, 0, 0}, []string{"c"}, false)
	addRawDoc(se, "b", []float32{0, 0, 1}, []string{"c"}, false)
	se.splitting["c"] = true

	se.splitCategory("c")

	c, ok := se.GetCategory("c")
	if !ok {
		t.Fatal("category c should survive a cap-refused split")
	}
	if !c.ShouldSplit {
		t.Fatal("ShouldSplit must stay set so the split can retry when the cap frees up")
	}
	if se.CategoryCount() != 1 {
		t.Fatalf("expected category count unchanged at 1, got %d", se.CategoryCount())
	}
	se.mu.RLock()
	stillSplitting := se.splitting["c"]
	se.mu.RUnlock()
	if stillSplitting {
		t.Fatal("splitting entry for c must be cleared even on the early-out path")
	}
}

func TestMergeSplitLockedOnAlreadyPrunedCategory(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{MaxCategories: 10})
	// "c" doesn't exist at all — e.g. its last member was removed between
	// the split's snapshot and the merge.
	if _, ok := se.mergeSplitLocked("c", nil, [2][]float32{}, [2]float32{}); ok {
		t.Fatal("expected merge on a nonexistent category to report ok=false")
	}
}

// -------------------- end-to-end, via Process --------------------

func waitUntil(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	if !cond() {
		t.Fatalf("condition not met within %v", timeout)
	}
}

// TestProcessTriggersAsyncSplitEndToEnd feeds two completely disjoint
// vocabularies into what threshold=0 forces to be a single, ever-growing
// category, so ShouldSplit fires partway through — and the async split then
// runs concurrently with the rest of the ingestion loop, exactly the
// scenario the split machinery exists for. It asserts the system settles
// into a clean, fully reconciled state: no doc lost, no doc left pointing at
// a deleted category, and audio/medical docs cleanly separated.
func TestProcessTriggersAsyncSplitEndToEnd(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{
		CategoryThreshold:   0,
		MaxCategoriesPerDoc: 1,
		MaxCategories:       20,
		VarianceThreshold:   0,
		VarianceMinCount:    2,
	})
	ctx := context.Background()

	const n = 60
	for i := 0; i < n; i++ {
		content := "sony wireless headphones audio"
		if i%2 == 1 {
			content = "medical stethoscope blood pressure"
		}
		id := fmt.Sprintf("d%d", i)
		if _, err := se.Process(ctx, id, content); err != nil {
			t.Fatalf("process %s: %v", id, err)
		}
	}

	// The split(s) run asynchronously; wait for the system to settle into a
	// stable, fully-separated state.
	waitUntil(t, 3*time.Second, func() bool {
		if se.CategoryCount() != 2 {
			return false
		}
		for _, c := range se.ListCategories() {
			if c.ShouldSplit {
				return false // a cascading split is still pending/running
			}
		}
		return true
	})

	if se.DocCount() != n {
		t.Fatalf("expected all %d docs to survive the split(s), got %d", n, se.DocCount())
	}

	// Every doc must point at a category that actually exists, and every
	// audio doc must land in a different category than every medical doc.
	var audioCat, medicalCat string
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("d%d", i)
		d, ok := se.GetDocument(id)
		if !ok {
			t.Fatalf("doc %s missing", id)
		}
		if len(d.Categories) != 1 {
			t.Fatalf("doc %s should belong to exactly 1 category, got %v", id, d.Categories)
		}
		cat := d.Categories[0]
		if _, ok := se.GetCategory(cat); !ok {
			t.Fatalf("doc %s points at nonexistent category %q", id, cat)
		}
		if i%2 == 0 {
			if audioCat == "" {
				audioCat = cat
			} else if audioCat != cat {
				t.Fatalf("audio doc %s landed in %q, expected %q", id, cat, audioCat)
			}
		} else {
			if medicalCat == "" {
				medicalCat = cat
			} else if medicalCat != cat {
				t.Fatalf("medical doc %s landed in %q, expected %q", id, cat, medicalCat)
			}
		}
	}
	if audioCat == "" || medicalCat == "" || audioCat == medicalCat {
		t.Fatalf("expected audio and medical docs in two distinct categories, got %q and %q", audioCat, medicalCat)
	}

	// The engine must still search correctly after the split.
	res, err := se.Search(ctx, "sony wireless headphones audio", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	for _, r := range res.Documents {
		if len(r.Categories) != 1 || r.Categories[0] != audioCat {
			t.Fatalf("audio search returned doc %s from category %v, expected %q", r.Document.ID, r.Categories, audioCat)
		}
	}
}

// TestProcessDoesNotDuplicateInFlightSplit verifies the se.splitting gate:
// Process must not queue a second split for a category that is (as far as
// it's concerned) already being split.
func TestProcessDoesNotDuplicateInFlightSplit(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{
		CategoryThreshold: 0, MaxCategoriesPerDoc: 1, MaxCategories: 1,
		VarianceThreshold: 0, VarianceMinCount: 1,
	})
	ctx := context.Background()
	if _, err := se.Process(ctx, "a1", "sony wireless headphones audio"); err != nil {
		t.Fatalf("process a1: %v", err)
	}

	const catName = "category1"
	se.mu.Lock()
	se.splitting[catName] = true // pretend a split is already running
	se.mu.Unlock()

	// The first real member alone leaves variance at 0 (needs >=2 samples);
	// a second, differently-similar real member pushes variance above the
	// (zero) threshold, flipping ShouldSplit true — but Process must not
	// launch a second split while se.splitting still marks one in flight.
	if _, err := se.Process(ctx, "m1", "medical stethoscope"); err != nil {
		t.Fatalf("process m1: %v", err)
	}
	if _, err := se.Process(ctx, "m2", "audio medical"); err != nil {
		t.Fatalf("process m2: %v", err)
	}

	cat, ok := se.GetCategory(catName)
	if !ok {
		t.Fatalf("category %q should still exist (gated split must not have run)", catName)
	}
	if !cat.ShouldSplit {
		t.Fatalf("expected ShouldSplit to have flipped true")
	}

	// Give any (incorrectly) launched goroutine a moment to run — a real
	// split would delete the category almost immediately.
	time.Sleep(50 * time.Millisecond)
	if _, ok := se.GetCategory(catName); !ok {
		t.Fatalf("category %q was split even though a split was marked in flight", catName)
	}
	if se.CategoryCount() != 1 {
		t.Fatalf("expected category count unchanged at 1, got %d", se.CategoryCount())
	}
}

// TestSplitDoesNotBlockConcurrentOperations is the concurrency guarantee the
// whole feature exists for: while a real async split is in flight, unrelated
// search/ingest/removal calls must keep completing, not queue up behind it.
// It's bounded by a deadline instead of a fixed sleep, so it fails fast and
// deterministically on an actual deadlock/serialization bug rather than on
// timing noise.
func TestSplitDoesNotBlockConcurrentOperations(t *testing.T) {
	se := New("t", fakeEmbedder{}, Config{
		CategoryThreshold: 0, MaxCategoriesPerDoc: 1, MaxCategories: 20,
		VarianceThreshold: 0, VarianceMinCount: 2,
	})
	ctx := context.Background()

	// Force ~150 docs into a single category with real internal variance, so
	// a real split gets triggered and has non-trivial work to do.
	for i := 0; i < 150; i++ {
		content := "sony wireless headphones audio"
		if i%2 == 1 {
			content = "medical stethoscope blood pressure"
		}
		if _, err := se.Process(ctx, fmt.Sprintf("seed%d", i), content); err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
	}

	// Hammer the engine with unrelated search/ingest/removal from many
	// goroutines while any split(s) triggered above may still be running in
	// the background. If a split held the write lock for its full duration,
	// this would serialize behind it; done would still close, just slowly —
	// so a generous deadline is what actually catches a real regression
	// (e.g. clustering running under the lock) without being timing-flaky.
	done := make(chan struct{})
	go func() {
		var wg sync.WaitGroup
		for i := 0; i < 50; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				id := fmt.Sprintf("extra%d", i)
				if _, err := se.Process(ctx, id, "sony wireless headphones audio"); err != nil {
					t.Errorf("process %s: %v", id, err)
					return
				}
				if _, err := se.Search(ctx, "wireless audio", 5); err != nil {
					t.Errorf("search: %v", err)
					return
				}
				se.Remove(id)
			}(i)
		}
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("concurrent operations did not complete — the split appears to be blocking the engine")
	}
}

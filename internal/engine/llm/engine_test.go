package llm

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mg52/search52-ai/internal/ai"
)

func newTestPrompts(t *testing.T) *ai.Prompts {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"tagging_system.template":     "tagging system max={{.MaxCategories}} atcap={{.AtCategoryCap}}",
		"tagging_user.template":       "existing={{.ExistingCategories}} Document: {{.Content}}",
		"description_system.template": "describe {{.CategoryName}}",
		"description_user.template":   "Category: {{.CategoryName}} Example documents: {{.Examples}}",
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

// chatFunc adapts a plain func to common.Chatter.
type chatFunc func(ctx context.Context, system, user string) (string, error)

func (f chatFunc) Complete(ctx context.Context, system, user string) (string, error) {
	return f(ctx, system, user)
}

// dispatchChatter routes a Complete call to tagging or description behavior
// based on which fixed prompt rendered it (distinguishable by their fixed
// literal substrings), and counts calls of each kind.
type dispatchChatter struct {
	taggingResp      string
	descriptionResp  string
	taggingCalls     int32
	descriptionCalls int32
}

func (d *dispatchChatter) Complete(_ context.Context, _, user string) (string, error) {
	if strings.Contains(user, "Document:") {
		atomic.AddInt32(&d.taggingCalls, 1)
		return d.taggingResp, nil
	}
	atomic.AddInt32(&d.descriptionCalls, 1)
	return d.descriptionResp, nil
}

var vocab = map[string]int{
	"sony": 0, "wireless": 1, "headphones": 2, "audio": 3, "devices": 4,
	"medical": 5, "stethoscope": 6, "blood": 7, "pressure": 8, "equipment": 9,
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

func newTestIndex(t *testing.T, chatter *dispatchChatter, cfg Config) *Index {
	t.Helper()
	return New("t", chatter, fakeEmbedder{}, newTestPrompts(t), cfg)
}

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

func TestProcessCreatesCategoryAndGeneratesProfileAsync(t *testing.T) {
	chatter := &dispatchChatter{
		taggingResp:     `{"categories":["audio_devices"]}`,
		descriptionResp: `{"description":"wireless audio headphones and related gear"}`,
	}
	idx := newTestIndex(t, chatter, Config{})
	ctx := context.Background()

	doc, err := idx.Process(ctx, "a1", "sony wireless headphones audio")
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(doc.Categories) != 1 || doc.Categories[0] != "audio_devices" {
		t.Fatalf("Categories = %v, want [audio_devices]", doc.Categories)
	}

	// Profile generation is async; the category exists immediately but its
	// embedding (VectorDims > 0) lands shortly after.
	waitUntil(t, time.Second, func() bool {
		c, ok := idx.GetCategory("audio_devices")
		return ok && c.VectorDims > 0
	})
	c, _ := idx.GetCategory("audio_devices")
	if c.Description == nil || *c.Description == "" {
		t.Fatal("expected a non-empty category description after profile generation")
	}
	if atomic.LoadInt32(&chatter.descriptionCalls) != 1 {
		t.Fatalf("description calls = %d, want 1", chatter.descriptionCalls)
	}
}

func TestProcessReusesExistingCategoryWithoutRegeneratingProfile(t *testing.T) {
	chatter := &dispatchChatter{
		taggingResp:     `{"categories":["audio_devices"]}`,
		descriptionResp: `{"description":"audio gear"}`,
	}
	idx := newTestIndex(t, chatter, Config{})
	ctx := context.Background()

	if _, err := idx.Process(ctx, "a1", "sony wireless headphones"); err != nil {
		t.Fatalf("process a1: %v", err)
	}
	waitUntil(t, time.Second, func() bool {
		c, ok := idx.GetCategory("audio_devices")
		return ok && c.VectorDims > 0
	})

	if _, err := idx.Process(ctx, "a2", "wireless headphones audio"); err != nil {
		t.Fatalf("process a2: %v", err)
	}

	if idx.DocCountByCategory("audio_devices") != 2 {
		t.Fatalf("expected 2 docs in audio_devices, got %d", idx.DocCountByCategory("audio_devices"))
	}
	if idx.CategoryCount() != 1 {
		t.Fatalf("expected exactly 1 category, got %d", idx.CategoryCount())
	}
	// Description generation must only fire once — for the founding doc, not
	// for the doc that merely joined an already-existing category.
	if got := atomic.LoadInt32(&chatter.descriptionCalls); got != 1 {
		t.Fatalf("description calls = %d, want 1", got)
	}
}

func TestProcessUpdateKeepsSoleMemberCategoryProfile(t *testing.T) {
	chatter := &dispatchChatter{
		taggingResp:     `{"categories":["audio_devices"]}`,
		descriptionResp: `{"description":"audio gear"}`,
	}
	idx := newTestIndex(t, chatter, Config{})
	ctx := context.Background()

	if _, err := idx.Process(ctx, "a1", "sony wireless headphones"); err != nil {
		t.Fatalf("process a1: %v", err)
	}
	waitUntil(t, time.Second, func() bool {
		c, ok := idx.GetCategory("audio_devices")
		return ok && c.VectorDims > 0
	})

	// Re-process the sole member of the category with the same category
	// assignment. The category (and its generated profile) must survive
	// instead of being pruned and re-founded.
	if _, err := idx.Process(ctx, "a1", "sony wireless headphones audio"); err != nil {
		t.Fatalf("re-process a1: %v", err)
	}

	c, ok := idx.GetCategory("audio_devices")
	if !ok {
		t.Fatal("category vanished after updating its only member")
	}
	if c.VectorDims == 0 {
		t.Fatal("category profile (vector) was thrown away on update")
	}
	if got := idx.DocCountByCategory("audio_devices"); got != 1 {
		t.Fatalf("doc count = %d, want 1", got)
	}
	if got := atomic.LoadInt32(&chatter.descriptionCalls); got != 1 {
		t.Fatalf("description calls = %d, want 1 (profile must not regenerate on update)", got)
	}
}

func TestProcessUpdateDetachesFromLeftCategories(t *testing.T) {
	chatter := &dispatchChatter{
		taggingResp:     `{"categories":["audio_devices"]}`,
		descriptionResp: `{"description":"d"}`,
	}
	idx := newTestIndex(t, chatter, Config{})
	ctx := context.Background()

	if _, err := idx.Process(ctx, "a1", "sony wireless headphones"); err != nil {
		t.Fatalf("process a1: %v", err)
	}

	// Same doc re-categorized into a different category: it must leave the
	// old one (pruned as empty) and join the new one exactly once.
	chatter.taggingResp = `{"categories":["medical_equipment"]}`
	doc, err := idx.Process(ctx, "a1", "medical stethoscope")
	if err != nil {
		t.Fatalf("re-process a1: %v", err)
	}
	if len(doc.Categories) != 1 || doc.Categories[0] != "medical_equipment" {
		t.Fatalf("Categories = %v, want [medical_equipment]", doc.Categories)
	}
	if _, ok := idx.GetCategory("audio_devices"); ok {
		t.Fatal("old empty category should have been pruned")
	}
	if got := idx.DocCountByCategory("medical_equipment"); got != 1 {
		t.Fatalf("doc count = %d, want 1", got)
	}
	if idx.CategoryCount() != 1 {
		t.Fatalf("category count = %d, want 1", idx.CategoryCount())
	}
}

func TestProcessAtCapFallsBackToMostPopulousCategory(t *testing.T) {
	chatter := &dispatchChatter{
		taggingResp:     `{"categories":["a"]}`,
		descriptionResp: `{"description":"desc"}`,
	}
	idx := newTestIndex(t, chatter, Config{MaxCategories: 1})
	ctx := context.Background()

	if _, err := idx.Process(ctx, "d1", "sony wireless headphones"); err != nil {
		t.Fatalf("process d1: %v", err)
	}
	if idx.CategoryCount() != 1 {
		t.Fatalf("expected 1 category, got %d", idx.CategoryCount())
	}

	// The model disobeys the at-cap instruction and invents an unknown name;
	// resolveCategoriesLocked must drop it and fall back to the existing,
	// most populous category instead of creating a second one.
	chatter.taggingResp = `{"categories":["b"]}`
	doc, err := idx.Process(ctx, "d2", "medical stethoscope")
	if err != nil {
		t.Fatalf("process d2: %v", err)
	}
	if len(doc.Categories) != 1 || doc.Categories[0] != "a" {
		t.Fatalf("expected fallback to category 'a', got %v", doc.Categories)
	}
	if idx.CategoryCount() != 1 {
		t.Fatalf("cap must not be exceeded, got %d categories", idx.CategoryCount())
	}
}

func TestRemovePrunesEmptyCategory(t *testing.T) {
	chatter := &dispatchChatter{taggingResp: `{"categories":["a"]}`, descriptionResp: `{"description":"d"}`}
	idx := newTestIndex(t, chatter, Config{})
	ctx := context.Background()

	if _, err := idx.Process(ctx, "d1", "sony audio"); err != nil {
		t.Fatalf("process: %v", err)
	}
	if _, ok := idx.Remove("d1"); !ok {
		t.Fatal("expected Remove to report the doc existed")
	}
	if idx.CategoryCount() != 0 {
		t.Fatalf("expected the now-empty category to be pruned, got %d", idx.CategoryCount())
	}
	if _, ok := idx.GetDocument("d1"); ok {
		t.Fatal("doc should be gone after Remove")
	}
}

func TestSearchRanksByCategoryEmbeddingAndReturnsMembers(t *testing.T) {
	chatter := &dispatchChatter{descriptionResp: `{"description":"placeholder"}`}
	idx := newTestIndex(t, chatter, Config{MaxDocsPerCategory: 10, TopNCategories: 2})
	ctx := context.Background()

	chatter.taggingResp = `{"categories":["audio_devices"]}`
	if _, err := idx.Process(ctx, "a1", "sony wireless headphones audio devices"); err != nil {
		t.Fatalf("process a1: %v", err)
	}
	chatter.taggingResp = `{"categories":["medical_equipment"]}`
	if _, err := idx.Process(ctx, "m1", "medical stethoscope blood pressure equipment"); err != nil {
		t.Fatalf("process m1: %v", err)
	}

	waitUntil(t, time.Second, func() bool {
		a, ok1 := idx.GetCategory("audio_devices")
		m, ok2 := idx.GetCategory("medical_equipment")
		return ok1 && ok2 && a.VectorDims > 0 && m.VectorDims > 0
	})

	res, err := idx.Search(ctx, "wireless audio headphones", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res.MatchedCategories) == 0 {
		t.Fatal("expected at least one matched category")
	}
	if res.MatchedCategories[0].Name != "audio_devices" {
		t.Fatalf("top matched category = %q, want audio_devices", res.MatchedCategories[0].Name)
	}
	found := false
	for _, r := range res.Documents {
		if r.Document.ID == "a1" {
			found = true
		}
	}
	if !found {
		t.Fatal("expected a1 among the search results")
	}
}

func TestSearchDeduplicatesDocSharedAcrossMatchedCategories(t *testing.T) {
	chatter := &dispatchChatter{descriptionResp: `{"description":"placeholder"}`}
	idx := newTestIndex(t, chatter, Config{MaxCategoriesPerDoc: 2, MaxDocsPerCategory: 10, TopNCategories: 5})
	ctx := context.Background()

	chatter.taggingResp = `{"categories":["audio_devices","medical_equipment"]}`
	if _, err := idx.Process(ctx, "both", "sony wireless headphones audio medical stethoscope"); err != nil {
		t.Fatalf("process: %v", err)
	}
	waitUntil(t, time.Second, func() bool {
		a, ok1 := idx.GetCategory("audio_devices")
		m, ok2 := idx.GetCategory("medical_equipment")
		return ok1 && ok2 && a.VectorDims > 0 && m.VectorDims > 0
	})

	res, err := idx.Search(ctx, "wireless audio medical stethoscope", 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	count := 0
	for _, r := range res.Documents {
		if r.Document.ID == "both" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("doc 'both' should appear exactly once, got %d", count)
	}
}

func TestProcessConcurrentDoesNotRace(t *testing.T) {
	chatter := &dispatchChatter{taggingResp: `{"categories":["audio_devices"]}`, descriptionResp: `{"description":"d"}`}
	idx := newTestIndex(t, chatter, Config{})
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := fmt.Sprintf("d%d", i)
			if _, err := idx.Process(ctx, id, "sony wireless headphones audio"); err != nil {
				t.Errorf("process %s: %v", id, err)
			}
		}(i)
	}
	wg.Wait()

	if idx.DocCount() != 20 {
		t.Fatalf("expected 20 docs, got %d", idx.DocCount())
	}
}

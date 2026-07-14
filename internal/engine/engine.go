// Package engine is a self-contained, in-memory vector search index. A single
// SearchEngine owns everything: the documents, the incrementally-discovered
// categories (centroids), the category→documents index, the clustering math,
// and the search itself. There is no LLM and no separate persistence layer —
// one struct, one mutex.
//
// A document is embedded once, then assigned to every existing category whose
// cosine similarity to the document is >= threshold (highest first, capped at
// maxPerDoc), or seeds a new category when none are close enough (capped by
// maxCategories). Centroids hold the running SUM of member vectors; because
// cosine is scale-invariant the sum ranks identically to the mean, and member
// removal stays exact.
package engine

import (
	"context"
	"fmt"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/mg52/search52-ai/internal/vec"
)

// scoreScale converts a cosine similarity in [-1,1] to the integer score the
// min-heap orders on. 1e6 keeps ~6 significant digits, far more than embedding
// similarities can meaningfully distinguish.
const scoreScale = 1e6

// Embedder turns text into an embedding vector.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// Document is a stored item plus its embedding and the categories it clustered
// into. Vector/Norm are runtime-only (not serialized in API responses).
type Document struct {
	ID         string    `json:"id"`
	Content    string    `json:"content"`
	Categories []string  `json:"categories"`
	Vector     []float32 `json:"-"`
	Norm       float32   `json:"-"` // cached L2 norm of Vector
	CreatedAt  time.Time `json:"created_at"`

	// ForcedFallback is true when Categories was assigned via the
	// maxCategories cap-reached "nearest wins" fallback in assignLocked
	// rather than a genuine >=threshold match. Such a doc's vector is
	// excluded from its category's centroid (see addToCentroidLocked) so an
	// unrelated document dumped into the nearest category at cap can't drag
	// that centroid away from its real members.
	ForcedFallback bool `json:"-"`
}

// Category is an incrementally-discovered cluster. Centroid holds the running
// SUM of member document vectors; Norm caches its L2 norm; Count is the number
// of member documents. Categories are auto-named ("category1", …).
//
// welfordCount/welfordMean/welfordM2 implement Welford's online algorithm,
// tracking the running variance of each member's cosine similarity to the
// centroid at the moment it was assigned. Welford only accumulates — it has
// no exact inverse update — so these are never adjusted on document removal.
// Variance is the resulting sample variance; ShouldSplit flips true once
// Variance exceeds the engine's varianceThreshold AND Count exceeds
// varianceMinCount, so a category isn't flagged while it's still too young
// for the variance estimate to be meaningful. No split is performed yet,
// this only flags the category as a candidate.
type Category struct {
	Name      string    `json:"name"`
	Centroid  []float32 `json:"-"`
	Norm      float32   `json:"-"`
	Count     int       `json:"count"`
	CreatedAt time.Time `json:"created_at"`

	welfordCount int
	welfordMean  float64
	welfordM2    float64

	Variance    float64 `json:"variance"`
	ShouldSplit bool    `json:"should_split"`
}

// SearchResult is a ranked document with its similarity score.
type SearchResult struct {
	Document   Document `json:"document"`
	Score      float64  `json:"score"`
	Categories []string `json:"categories"`
}

// CategoryMatch is a category the query matched, with its similarity.
type CategoryMatch struct {
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

// Results is the search response: ranked documents plus the categories the
// query matched (independent of any single document).
type Results struct {
	Documents         []SearchResult  `json:"documents"`
	MatchedCategories []CategoryMatch `json:"matched_categories"`
}

// Config tunes clustering and search. Zero values fall back to defaults.
type Config struct {
	CategoryThreshold   float64 // min cosine to join an existing category
	MaxCategoriesPerDoc int     // cap on categories one document may join
	MaxCategories       int     // cap on total categories
	TopNCategories      int     // nearest categories scanned per query
	VarianceThreshold   float64 // Welford variance above which a category's ShouldSplit flips true
	VarianceMinCount    int     // min category member count before ShouldSplit can fire
}

func (c *Config) withDefaults() {
	if c.MaxCategoriesPerDoc <= 0 {
		c.MaxCategoriesPerDoc = 3
	}
	if c.MaxCategories <= 0 {
		c.MaxCategories = 100
	}
	if c.TopNCategories <= 0 {
		c.TopNCategories = 3
	}
	if c.VarianceThreshold <= 0 {
		c.VarianceThreshold = 0.02
	}
	if c.VarianceMinCount <= 0 {
		c.VarianceMinCount = 100
	}
}

// SearchEngine is one index: documents, categories, and the search over them,
// all guarded by a single RWMutex.
type SearchEngine struct {
	name     string
	embedder Embedder

	threshold         float32
	maxPerDoc         int
	maxCategories     int
	topN              int
	varianceThreshold float64
	varianceMinCount  int

	mu             sync.RWMutex
	docs           map[string]Document            // doc ID -> document
	categories     map[string]*Category           // name -> category
	catDocs        map[string]map[string]struct{} // category name -> set of doc IDs
	nextCategoryID int                            // monotonic; never reuses a pruned name

	// splitting holds the names of categories with a split goroutine
	// currently in flight, so a run of documents all flipping the same
	// category's ShouldSplit doesn't spawn one splitCategory per document.
	// Always accessed under mu (write-locked), never under RLock.
	splitting map[string]bool
}

// New constructs an empty index named name.
func New(name string, embedder Embedder, cfg Config) *SearchEngine {
	cfg.withDefaults()
	return &SearchEngine{
		name:              name,
		embedder:          embedder,
		threshold:         float32(cfg.CategoryThreshold),
		maxPerDoc:         cfg.MaxCategoriesPerDoc,
		maxCategories:     cfg.MaxCategories,
		topN:              cfg.TopNCategories,
		varianceThreshold: cfg.VarianceThreshold,
		varianceMinCount:  cfg.VarianceMinCount,
		docs:              make(map[string]Document),
		categories:        make(map[string]*Category),
		catDocs:           make(map[string]map[string]struct{}),
		splitting:         make(map[string]bool),
	}
}

func (se *SearchEngine) Name() string { return se.name }

// Process embeds content and clusters it under id, replacing any existing
// document with that id (its stale category memberships are detached first).
// If this assignment flips a category's ShouldSplit flag, Process launches a
// splitCategory goroutine for it (see split.go) and returns without waiting
// on it — the split runs asynchronously and folds its result back into live
// state on its own.
func (se *SearchEngine) Process(ctx context.Context, id, content string) (Document, error) {
	if id == "" {
		return Document{}, fmt.Errorf("document id is required")
	}
	if content == "" {
		return Document{}, fmt.Errorf("content is required")
	}
	v, err := se.embedder.Embed(ctx, content)
	if err != nil {
		return Document{}, fmt.Errorf("embedding document: %w", err)
	}
	norm := vec.Norm(v)
	if norm == 0 {
		return Document{}, fmt.Errorf("document embedding is a zero vector")
	}

	se.mu.Lock()

	doc := Document{ID: id, Content: content, Vector: v, Norm: norm, CreatedAt: time.Now()}
	if old, ok := se.docs[id]; ok {
		doc.CreatedAt = old.CreatedAt // preserve original creation time on update
		se.detachLocked(old)
	}

	assignments, forced := se.assignLocked(v, norm)
	doc.ForcedFallback = forced
	doc.Categories = make([]string, len(assignments))
	for i, a := range assignments {
		doc.Categories[i] = a.name
	}
	se.docs[id] = doc

	// Categories whose ShouldSplit just flipped true get a splitCategory
	// goroutine launched below, once mu is released — split.go's asynchronous
	// clustering must never run while the write lock is held. splitting
	// guards against launching a second one while an earlier split for the
	// same category is still in flight.
	var toSplit []string
	for _, a := range assignments {
		se.addToCentroidLocked(a.name, v, !forced)
		se.addDocToCategoryLocked(id, a.name)
		if !a.founding {
			se.updateVarianceLocked(a.name, a.sim)
		}
		if c, ok := se.categories[a.name]; ok && c.ShouldSplit && !se.splitting[a.name] {
			se.splitting[a.name] = true
			toSplit = append(toSplit, a.name)
		}
	}
	se.mu.Unlock()

	for _, name := range toSplit {
		go se.splitCategory(name)
	}

	return doc, nil
}

// GetDocument returns the document with id, if present.
func (se *SearchEngine) GetDocument(id string) (Document, bool) {
	se.mu.RLock()
	defer se.mu.RUnlock()
	d, ok := se.docs[id]
	return d, ok
}

// Remove deletes a document and detaches it from every category (pruning any
// left empty).
func (se *SearchEngine) Remove(id string) (Document, bool) {
	se.mu.Lock()
	defer se.mu.Unlock()
	doc, ok := se.docs[id]
	if !ok {
		return Document{}, false
	}
	se.detachLocked(doc)
	delete(se.docs, id)
	return doc, true
}

func (se *SearchEngine) DocCount() int {
	se.mu.RLock()
	defer se.mu.RUnlock()
	return len(se.docs)
}

func (se *SearchEngine) CategoryCount() int {
	se.mu.RLock()
	defer se.mu.RUnlock()
	return len(se.categories)
}

// ListCategories returns every category in unspecified order.
func (se *SearchEngine) ListCategories() []Category {
	se.mu.RLock()
	defer se.mu.RUnlock()
	out := make([]Category, 0, len(se.categories))
	for _, c := range se.categories {
		out = append(out, *c)
	}
	return out
}

// GetCategory returns the category named name, if present.
func (se *SearchEngine) GetCategory(name string) (Category, bool) {
	se.mu.RLock()
	defer se.mu.RUnlock()
	c, ok := se.categories[name]
	if !ok {
		return Category{}, false
	}
	return *c, true
}

// DocCountByCategory returns the number of documents in a category.
func (se *SearchEngine) DocCountByCategory(name string) int {
	se.mu.RLock()
	defer se.mu.RUnlock()
	return len(se.catDocs[name])
}

// Search embeds the query, selects the topN nearest categories, and ranks their
// member documents by cosine similarity to the query. Both selections use a
// bounded min-heap so the cost is O(n log k) instead of a full sort.
func (se *SearchEngine) Search(ctx context.Context, query string, limit int) (Results, error) {
	if limit <= 0 {
		limit = 10
	}
	empty := Results{Documents: []SearchResult{}, MatchedCategories: []CategoryMatch{}}

	qv, err := se.embedder.Embed(ctx, query)
	if err != nil {
		return Results{}, fmt.Errorf("embedding query: %w", err)
	}
	qn := vec.Norm(qv)
	if qn == 0 {
		return empty, nil
	}

	se.mu.RLock()
	defer se.mu.RUnlock()

	// --- Category loop: keep the topN nearest categories in a min-heap. ---
	// The heap stores indexes into catCands; when a full heap evicts its
	// minimum, the evicted entry's slot is reused, so catCands never grows
	// beyond topN.
	catCands := make([]CategoryMatch, 0, se.topN)
	ch := make([]internalHit, 0, se.topN)
	for name, c := range se.categories {
		if c.Norm == 0 {
			continue
		}
		sim := vec.Cosine(qv, c.Centroid, qn, c.Norm)
		score := int(sim * scoreScale)
		if len(ch) < se.topN {
			catCands = append(catCands, CategoryMatch{Name: name, Score: float64(sim)})
			ch = heapPushHit(ch, internalHit{id: uint32(len(catCands) - 1), score: score})
		} else if ch[0].score < score {
			slot := ch[0].id
			catCands[slot] = CategoryMatch{Name: name, Score: float64(sim)}
			heapReplaceTop(ch, internalHit{id: slot, score: score})
		}
	}
	if len(ch) == 0 {
		return empty, nil
	}

	// Drain the category heap into descending order.
	matched := make([]CategoryMatch, len(ch))
	for i := len(ch) - 1; i >= 0; i-- {
		hit := ch[0]
		if i > 0 {
			ch[0] = ch[i]
			siftDownHit(ch, 0, i)
		}
		matched[i] = catCands[hit.id]
	}

	// --- Document loop: keep the top `limit` documents in a min-heap, with the
	// same slot-reuse scheme (docCands never grows beyond limit). Candidates
	// carry the Document copy so the drain needs no second map lookup. ---
	type docCand struct {
		doc   Document
		score float32
	}
	docCands := make([]docCand, 0, limit)
	dh := make([]internalHit, 0, limit)

	for k, m := range matched {
		for id := range se.catDocs[m.Name] {
			doc, ok := se.docs[id]
			if !ok || doc.Norm == 0 {
				continue
			}
			// Dedup without a map: a doc shared with a higher-ranked selected
			// category was already scanned there. doc.Categories has at most
			// maxPerDoc entries, so this is a handful of string compares.
			if k > 0 && sharesEarlierCategory(doc.Categories, matched[:k]) {
				continue
			}
			sim := vec.Cosine(qv, doc.Vector, qn, doc.Norm)
			if sim <= 0 {
				continue // irrelevant to the query
			}
			score := int(sim * scoreScale)
			if len(dh) < limit {
				docCands = append(docCands, docCand{doc, sim})
				dh = heapPushHit(dh, internalHit{id: uint32(len(docCands) - 1), score: score})
			} else if dh[0].score < score {
				slot := dh[0].id
				docCands[slot] = docCand{doc, sim}
				heapReplaceTop(dh, internalHit{id: slot, score: score})
			}
		}
	}

	// Drain the document heap into descending order.
	results := make([]SearchResult, len(dh))
	for i := len(dh) - 1; i >= 0; i-- {
		hit := dh[0]
		if i > 0 {
			dh[0] = dh[i]
			siftDownHit(dh, 0, i)
		}
		dc := docCands[hit.id]
		results[i] = SearchResult{Document: dc.doc, Score: float64(dc.score), Categories: dc.doc.Categories}
	}

	return Results{Documents: results, MatchedCategories: matched}, nil
}

// sharesEarlierCategory reports whether any of docCats appears among the
// already-scanned matched categories.
func sharesEarlierCategory(docCats []string, earlier []CategoryMatch) bool {
	for _, m := range earlier {
		if slices.Contains(docCats, m.Name) {
			return true
		}
	}
	return false
}

// -------------------- clustering (caller holds the write lock) --------------------

// catAssignment pairs an assigned category name with the cosine similarity
// that earned it, so the caller can feed that value into the category's
// running Welford variance (see updateVarianceLocked). founding is true when
// this assignment just seeded a brand-new category: that doc trivially has
// similarity 1 to a centroid that is nothing but its own vector, which isn't
// a real similarity sample, so the caller must skip the Welford update for it.
type catAssignment struct {
	name     string
	sim      float32
	founding bool
}

// assignLocked picks the categories for a vector: every existing category whose
// cosine similarity is >= threshold (up to maxPerDoc, highest first); otherwise
// a new category, or — when the maxCategories cap is hit — the single nearest.
// The second return value reports whether that last case fired: a forced,
// sub-threshold assignment that the caller must exclude from the category's
// centroid math (see Document.ForcedFallback).
func (se *SearchEngine) assignLocked(v []float32, norm float32) ([]catAssignment, bool) {
	type catSim struct {
		name string
		sim  float32
	}
	sims := make([]catSim, 0, len(se.categories))
	for name, c := range se.categories {
		if c.Norm == 0 {
			continue
		}
		sims = append(sims, catSim{name, vec.Cosine(v, c.Centroid, norm, c.Norm)})
	}
	sort.Slice(sims, func(i, j int) bool { return sims[i].sim > sims[j].sim })

	var assigned []catAssignment
	for _, sc := range sims {
		if sc.sim < se.threshold || len(assigned) >= se.maxPerDoc {
			break
		}
		assigned = append(assigned, catAssignment{name: sc.name, sim: sc.sim})
	}
	if len(assigned) == 0 {
		switch {
		case len(se.categories) < se.maxCategories:
			assigned = append(assigned, catAssignment{name: se.newCategoryLocked(), founding: true})
		case len(sims) > 0:
			assigned = append(assigned, catAssignment{name: sims[0].name, sim: sims[0].sim}) // cap reached: nearest wins
			return assigned, true
		}
	}
	return assigned, false
}

func (se *SearchEngine) newCategoryLocked() string {
	se.nextCategoryID++
	name := fmt.Sprintf("category%d", se.nextCategoryID)
	se.categories[name] = &Category{Name: name, CreatedAt: time.Now()}
	return name
}

// addToCentroidLocked folds v into a category's running centroid sum, unless
// updateCentroid is false (a ForcedFallback assignment), in which case the doc
// is still counted as a member but its vector never touches the centroid.
// Mutation happens in place: search only reads centroids under RLock, so it
// never overlaps this write.
func (se *SearchEngine) addToCentroidLocked(name string, v []float32, updateCentroid bool) {
	c, ok := se.categories[name]
	if !ok {
		return
	}
	c.Count++
	if !updateCentroid {
		return
	}
	if c.Centroid == nil {
		c.Centroid = make([]float32, len(v))
	}
	for i := range v {
		c.Centroid[i] += v[i]
	}
	c.Norm = vec.Norm(c.Centroid)
}

// removeFromCentroidLocked subtracts v from a category's centroid sum, pruning
// the category when its last member leaves. updateCentroid must mirror the
// value passed to the matching addToCentroidLocked call, or the subtraction
// would corrupt a centroid the doc's vector was never folded into.
func (se *SearchEngine) removeFromCentroidLocked(name string, v []float32, updateCentroid bool) {
	c, ok := se.categories[name]
	if !ok {
		return
	}
	c.Count--
	if c.Count <= 0 {
		delete(se.categories, name)
		delete(se.catDocs, name)
		return
	}
	if !updateCentroid {
		return
	}
	for i := range v {
		if i < len(c.Centroid) {
			c.Centroid[i] -= v[i]
		}
	}
	c.Norm = vec.Norm(c.Centroid)
}

// updateVarianceLocked folds a member's assignment-time similarity to the
// centroid into that category's running Welford statistics, giving an
// O(1)-per-document, single-pass estimate of the category's spread without
// keeping every similarity value around. It runs once per (document,
// category) pair whenever a document joins an existing category — the
// caller skips it for the document that founds a brand-new category, since
// that doc's "similarity" to a centroid that is nothing but its own vector
// (1.0) isn't a real sample and would bias the running variance. There is
// also no corresponding update on removal (Welford has no exact inverse).
// ShouldSplit is a flag only — no split is performed here.
func (se *SearchEngine) updateVarianceLocked(name string, sim float32) {
	c, ok := se.categories[name]
	if !ok {
		return
	}
	c.welfordCount++
	x := float64(sim)
	delta := x - c.welfordMean
	c.welfordMean += delta / float64(c.welfordCount)
	c.welfordM2 += delta * (x - c.welfordMean)
	if c.welfordCount > 1 {
		c.Variance = c.welfordM2 / float64(c.welfordCount-1)
	}
	c.ShouldSplit = c.Variance > se.varianceThreshold && c.Count > se.varianceMinCount
}

func (se *SearchEngine) addDocToCategoryLocked(id, name string) {
	set := se.catDocs[name]
	if set == nil {
		set = make(map[string]struct{})
		se.catDocs[name] = set
	}
	set[id] = struct{}{}
}

// detachLocked removes a document's contribution to every category it belonged to.
func (se *SearchEngine) detachLocked(doc Document) {
	for _, name := range doc.Categories {
		se.removeFromCentroidLocked(name, doc.Vector, !doc.ForcedFallback)
		if set := se.catDocs[name]; set != nil {
			delete(set, doc.ID)
		}
	}
}

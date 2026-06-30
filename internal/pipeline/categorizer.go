// Package pipeline turns raw documents into categorized, embedded records and
// owns all clustering logic.
//
// There is no LLM in the loop: a document is embedded once, then assigned to
// categories purely by vector similarity. Categories are discovered
// incrementally — the first document seeds "category1"; each subsequent document
// joins the nearest existing categories above a cosine threshold, or spawns a new
// category when none are close enough (capped by maxCategories).
//
// The store is a dumb persistence layer; the Categorizer is the only thing that
// computes or maintains centroids. A single mutex serializes every cluster
// mutation (add/remove), so the read-modify-write of a centroid is atomic even
// though it spans several individually-locked store calls.
package pipeline

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/store"
	"github.com/mg52/search52-ai/internal/vec"
)

type Categorizer struct {
	embedding     *ai.EmbeddingClient
	store         store.Store
	threshold     float64
	maxPerDoc     int
	maxCategories int

	mu             sync.Mutex // serializes cluster mutations
	nextCategoryID int        // monotonic; never reuses a pruned category's name
}

func NewCategorizer(embedding *ai.EmbeddingClient, st store.Store, threshold float64, maxPerDoc, maxCategories int) *Categorizer {
	return &Categorizer{
		embedding:     embedding,
		store:         st,
		threshold:     threshold,
		maxPerDoc:     maxPerDoc,
		maxCategories: maxCategories,
	}
}

// Process embeds the document's content, clusters it into categories, and
// persists the result. Re-processing an existing ID first detaches its previous
// memberships so its categories stay consistent with the new content. The
// returned document carries its assigned Categories.
func (c *Categorizer) Process(ctx context.Context, doc store.Document) (store.Document, error) {
	if doc.ID == "" {
		return doc, fmt.Errorf("document id is required")
	}
	v, err := c.embedding.Embed(ctx, doc.Content)
	if err != nil {
		return doc, fmt.Errorf("embedding document: %w", err)
	}
	doc.Vector = v
	doc.Norm = vec.Norm(v)
	if doc.Norm == 0 {
		return doc, fmt.Errorf("document embedding is a zero vector")
	}
	if doc.CreatedAt.IsZero() {
		doc.CreatedAt = time.Now()
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if old, ok := c.store.GetDocument(doc.ID); ok {
		c.detach(old) // drop stale memberships + centroid contributions
	}

	doc.Categories = c.assign(doc.Vector, doc.Norm)
	c.store.PutDocument(doc)
	for _, name := range doc.Categories {
		c.addToCentroid(name, doc.Vector)
		c.store.AddDocToCategory(doc.ID, name)
	}
	return doc, nil
}

// Remove deletes a document and detaches it from every category it belonged to,
// updating centroids and pruning categories left empty.
func (c *Categorizer) Remove(id string) (store.Document, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	doc, ok := c.store.GetDocument(id)
	if !ok {
		return store.Document{}, false
	}
	c.detach(doc)
	c.store.DeleteDocument(id)
	return doc, true
}

// assign picks the categories for a vector: every existing category whose cosine
// similarity is >= threshold (up to maxPerDoc, highest first); otherwise a new
// category, or — when the maxCategories cap is hit — the single nearest.
func (c *Categorizer) assign(v []float64, norm float64) []string {
	type catSim struct {
		name string
		sim  float64
	}
	cats := c.store.ListCategories()
	sims := make([]catSim, 0, len(cats))
	for _, cat := range cats {
		if cat.Norm == 0 {
			continue
		}
		sims = append(sims, catSim{cat.Name, vec.Cosine(v, cat.Centroid, norm, cat.Norm)})
	}
	sort.Slice(sims, func(i, j int) bool { return sims[i].sim > sims[j].sim })

	var assigned []string
	for _, sc := range sims {
		if sc.sim < c.threshold || len(assigned) >= c.maxPerDoc {
			break
		}
		assigned = append(assigned, sc.name)
	}
	if len(assigned) == 0 {
		switch {
		case c.store.CategoryCount() < c.maxCategories:
			assigned = append(assigned, c.newCategory())
		case len(sims) > 0:
			assigned = append(assigned, sims[0].name) // cap reached: nearest wins
		}
	}
	return assigned
}

func (c *Categorizer) newCategory() string {
	c.nextCategoryID++
	name := fmt.Sprintf("category%d", c.nextCategoryID)
	c.store.PutCategory(store.Category{Name: name, CreatedAt: time.Now()})
	return name
}

// addToCentroid folds v into a category's running centroid sum. It always writes
// a freshly-allocated slice so a concurrent reader (search) never sees the
// centroid mid-mutation.
func (c *Categorizer) addToCentroid(name string, v []float64) {
	cat, ok := c.store.GetCategory(name)
	if !ok {
		return
	}
	centroid := make([]float64, len(v))
	copy(centroid, cat.Centroid)
	for i := range v {
		centroid[i] += v[i]
	}
	cat.Centroid = centroid
	cat.Count++
	cat.Norm = vec.Norm(centroid)
	c.store.PutCategory(cat)
}

// removeFromCentroid subtracts v from a category's centroid sum. When the last
// member is removed the category is pruned entirely.
func (c *Categorizer) removeFromCentroid(name string, v []float64) {
	cat, ok := c.store.GetCategory(name)
	if !ok {
		return
	}
	cat.Count--
	if cat.Count <= 0 {
		c.store.DeleteCategory(name)
		return
	}
	centroid := make([]float64, len(cat.Centroid))
	copy(centroid, cat.Centroid)
	for i := range v {
		if i < len(centroid) {
			centroid[i] -= v[i]
		}
	}
	cat.Centroid = centroid
	cat.Norm = vec.Norm(centroid)
	c.store.PutCategory(cat)
}

// detach removes a document's contribution to every category it belonged to.
func (c *Categorizer) detach(doc store.Document) {
	for _, name := range doc.Categories {
		c.removeFromCentroid(name, doc.Vector)
		c.store.RemoveDocFromCategory(doc.ID, name)
	}
}

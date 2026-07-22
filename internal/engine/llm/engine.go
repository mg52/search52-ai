// Package llm is an LLM-categorized search index: an alternative to the
// embedding package's cosine-clustering engine. A document's categories are
// decided by an LLM chat call against a fixed prompt, not by centroid
// similarity — the model reuses an existing category name or invents a new
// one. Individual documents are never embedded. Only categories are: the
// first time a category is founded, a background call asks the LLM for a
// dense description from the category's example documents, then embeds
// "name + description + examples" once, so Search can rank categories by
// cosine similarity to the query and return each matched category's member
// documents.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/engine/common"
	"github.com/mg52/search52-ai/internal/vec"
)

// scoreScale converts a cosine similarity in [-1,1] to the integer score the
// min-heap orders on.
const scoreScale = 1e6

// maxExamplesPerCategory bounds how many member documents feed the
// description prompt when a category is founded.
const maxExamplesPerCategory = 10

// Document is a stored item plus the categories the LLM assigned it. Unlike
// the embedding engine, documents here carry no vector of their own.
type Document struct {
	ID         string
	Content    string
	Categories []string
	CreatedAt  time.Time
}

func (d Document) toCommon() common.Document {
	return common.Document{ID: d.ID, Content: d.Content, Categories: d.Categories, CreatedAt: d.CreatedAt}
}

// Category is an LLM-named cluster. Vector/Norm are populated asynchronously,
// once, right after the category is founded (see generateProfile) — until
// then Norm is 0 and the category is invisible to Search.
type Category struct {
	Name        string
	Description string
	Vector      []float32
	Norm        float32
	Examples    []string
	Count       int
	CreatedAt   time.Time
}

func (c Category) summary(docCount int) common.CategorySummary {
	desc := c.Description
	return common.CategorySummary{
		Name:        c.Name,
		DocCount:    docCount,
		CreatedAt:   c.CreatedAt,
		VectorDims:  len(c.Vector),
		Description: &desc,
	}
}

// Config tunes categorization and search. Zero values fall back to defaults.
type Config struct {
	MaxCategoriesPerDoc int // cap on categories one document may join
	MaxCategories       int // cap on total categories
	TopNCategories      int // nearest categories (by embedding) scanned per query
	MaxDocsPerCategory  int // docs returned per matched category in Search
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
	if c.MaxDocsPerCategory <= 0 {
		c.MaxDocsPerCategory = 50
	}
}

// Index is one LLM-categorized index: documents, categories, and the search
// over them, all guarded by a single RWMutex.
type Index struct {
	name     string
	llm      common.Chatter
	embedder common.Embedder
	prompts  *ai.Prompts

	maxPerDoc          int
	maxCategories      int
	topN               int
	maxDocsPerCategory int

	mu         sync.RWMutex
	docs       map[string]Document
	categories map[string]*Category
	catDocs    map[string]map[string]struct{}
}

// New constructs an empty LLM-categorized index named name.
func New(name string, llmClient common.Chatter, embedder common.Embedder, prompts *ai.Prompts, cfg Config) *Index {
	cfg.withDefaults()
	return &Index{
		name:               name,
		llm:                llmClient,
		embedder:           embedder,
		prompts:            prompts,
		maxPerDoc:          cfg.MaxCategoriesPerDoc,
		maxCategories:      cfg.MaxCategories,
		topN:               cfg.TopNCategories,
		maxDocsPerCategory: cfg.MaxDocsPerCategory,
		docs:               make(map[string]Document),
		categories:         make(map[string]*Category),
		catDocs:            make(map[string]map[string]struct{}),
	}
}

func (idx *Index) Name() string { return idx.name }

func (idx *Index) Kind() string { return "llm" }

type taggingResponse struct {
	Categories []string `json:"categories"`
}

// Process asks the LLM to assign content to one or more categories (reusing
// existing ones where possible, creating new ones otherwise), replacing any
// existing document with that id. Brand-new categories get their
// description+embedding generated in a background goroutine launched after
// the write lock is released — that call must never run while it is held.
func (idx *Index) Process(ctx context.Context, id, content string) (common.Document, error) {
	if id == "" {
		return common.Document{}, fmt.Errorf("document id is required")
	}
	if content == "" {
		return common.Document{}, fmt.Errorf("content is required")
	}

	idx.mu.RLock()
	existing := make([]string, 0, len(idx.categories))
	for name := range idx.categories {
		existing = append(existing, name)
	}
	atCap := len(idx.categories) >= idx.maxCategories
	idx.mu.RUnlock()
	sort.Strings(existing)

	names, err := idx.assignCategories(ctx, content, existing, atCap)
	if err != nil {
		return common.Document{}, err
	}

	idx.mu.Lock()

	doc := Document{ID: id, Content: content, CreatedAt: time.Now()}
	old, hadOld := idx.docs[id]
	if hadOld {
		doc.CreatedAt = old.CreatedAt
	}

	finalCats, newlyCreated := idx.resolveCategoriesLocked(names)
	doc.Categories = finalCats

	// On replace, detach the old doc only from categories it is leaving.
	// Detaching from a kept category could prune it (sole member), throwing
	// away its generated profile and re-firing generateProfile for the
	// re-founded copy.
	if hadOld {
		leaving := old
		leaving.Categories = nil
		for _, name := range old.Categories {
			if !slices.Contains(finalCats, name) {
				leaving.Categories = append(leaving.Categories, name)
			}
		}
		idx.detachLocked(leaving)
	}

	idx.docs[id] = doc
	for _, name := range finalCats {
		if _, member := idx.catDocs[name][id]; !member {
			idx.addDocToCategoryLocked(id, name)
		}
	}

	idx.mu.Unlock()

	for _, name := range newlyCreated {
		go idx.generateProfile(name)
	}

	return doc.toCommon(), nil
}

// assignCategories renders the fixed tagging prompts and calls the LLM,
// retrying on a malformed or empty response.
func (idx *Index) assignCategories(ctx context.Context, content string, existing []string, atCap bool) ([]string, error) {
	data := ai.TaggingData{
		ExistingCategories: strings.Join(existing, ", "),
		MaxCategories:      idx.maxPerDoc,
		Content:            content,
		AtCategoryCap:      atCap,
	}
	systemPrompt, err := idx.prompts.RenderTaggingSystem(data)
	if err != nil {
		return nil, fmt.Errorf("rendering tagging system prompt: %w", err)
	}
	userPrompt, err := idx.prompts.RenderTaggingUser(data)
	if err != nil {
		return nil, fmt.Errorf("rendering tagging user prompt: %w", err)
	}

	const maxAttempts = 3
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		raw, err := idx.llm.Complete(ctx, systemPrompt, userPrompt)
		if err != nil {
			return nil, fmt.Errorf("LLM call: %w", err)
		}

		var resp taggingResponse
		if jsonErr := json.Unmarshal([]byte(extractJSON(raw)), &resp); jsonErr != nil {
			if attempt == maxAttempts {
				return nil, fmt.Errorf("parsing LLM tagging response %q: %w", raw, jsonErr)
			}
			log.Printf("llm: tagging JSON parse failed (attempt %d/%d), retrying: %v", attempt, maxAttempts, jsonErr)
			continue
		}

		names := cleanNames(resp.Categories)
		if len(names) > idx.maxPerDoc {
			names = names[:idx.maxPerDoc]
		}
		if len(names) > 0 {
			return names, nil
		}
		if attempt == maxAttempts {
			return nil, fmt.Errorf("LLM returned no usable categories after %d attempts", maxAttempts)
		}
		log.Printf("llm: no usable categories (attempt %d/%d), retrying", attempt, maxAttempts)
	}
	return nil, fmt.Errorf("unreachable")
}

// resolveCategoriesLocked turns the LLM's chosen names into the document's
// final category set against *live* state (not the snapshot assignCategories
// read before the network call): existing names are reused as-is; unknown
// names are founded as new categories unless the cap has been hit since the
// snapshot was taken, in which case they're dropped. If every name is
// dropped, the doc falls back to the index's most populous existing
// category rather than being left with none. Caller must hold the write
// lock.
func (idx *Index) resolveCategoriesLocked(names []string) (final []string, newlyCreated []string) {
	for _, name := range names {
		if _, ok := idx.categories[name]; ok {
			final = append(final, name)
			continue
		}
		if len(idx.categories) >= idx.maxCategories {
			continue // cap reached since the snapshot; the model must not have invented this
		}
		idx.categories[name] = &Category{Name: name, CreatedAt: time.Now()}
		final = append(final, name)
		newlyCreated = append(newlyCreated, name)
	}
	if len(final) == 0 {
		if best := idx.mostPopulousCategoryLocked(); best != "" {
			final = []string{best}
		}
	}
	if len(final) > idx.maxPerDoc {
		final = final[:idx.maxPerDoc]
	}
	return final, newlyCreated
}

func (idx *Index) mostPopulousCategoryLocked() string {
	best, bestCount := "", -1
	for name, c := range idx.categories {
		if c.Count > bestCount || (c.Count == bestCount && name < best) {
			best, bestCount = name, c.Count
		}
	}
	return best
}

// generateProfile is the async, once-per-category enrichment step: it asks
// the LLM for a description from up to maxExamplesPerCategory member
// documents, then embeds "name + description + examples" together so Search
// can rank the category by cosine similarity. Best-effort — a failure here
// just leaves the category with Norm 0 (invisible to Search) until the index
// is recreated; it does not fail the Process call that founded it.
func (idx *Index) generateProfile(name string) {
	ctx := context.Background()

	idx.mu.RLock()
	ids := make([]string, 0, len(idx.catDocs[name]))
	for id := range idx.catDocs[name] {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if len(ids) > maxExamplesPerCategory {
		ids = ids[:maxExamplesPerCategory]
	}
	examples := make([]string, 0, len(ids))
	lines := make([]string, 0, len(ids))
	for i, id := range ids {
		if d, ok := idx.docs[id]; ok {
			examples = append(examples, d.Content)
			lines = append(lines, fmt.Sprintf("%d. %s", i+1, d.Content))
		}
	}
	idx.mu.RUnlock()

	description, err := idx.generateDescription(ctx, name, lines)
	if err != nil {
		log.Printf("llm: generating description for category %q: %v", name, err)
		return
	}

	embedText := fmt.Sprintf("category: %s\ndescription: %s\nexamples:\n%s", name, description, strings.Join(lines, "\n"))
	vector, err := idx.embedder.Embed(ctx, embedText)
	if err != nil {
		log.Printf("llm: embedding category %q: %v", name, err)
		return
	}

	idx.mu.Lock()
	if c, ok := idx.categories[name]; ok {
		c.Description = description
		c.Vector = vector
		c.Norm = vec.Norm(vector)
		c.Examples = examples
	}
	idx.mu.Unlock()

	log.Printf("llm: category %q profile ready (dims=%d, examples=%d)", name, len(vector), len(examples))
}

type descriptionResponse struct {
	Description string `json:"description"`
}

func (idx *Index) generateDescription(ctx context.Context, name string, lines []string) (string, error) {
	data := ai.DescriptionData{CategoryName: name, Examples: strings.Join(lines, "\n")}
	systemPrompt, err := idx.prompts.RenderDescriptionSystem(data)
	if err != nil {
		return "", fmt.Errorf("rendering description system prompt: %w", err)
	}
	userPrompt, err := idx.prompts.RenderDescriptionUser(data)
	if err != nil {
		return "", fmt.Errorf("rendering description user prompt: %w", err)
	}

	const maxAttempts = 3
	var resp descriptionResponse
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		raw, err := idx.llm.Complete(ctx, systemPrompt, userPrompt)
		if err != nil {
			return "", fmt.Errorf("LLM call: %w", err)
		}
		if jsonErr := json.Unmarshal([]byte(extractJSON(raw)), &resp); jsonErr == nil {
			return resp.Description, nil
		} else if attempt == maxAttempts {
			// Last resort: the model returned plain prose — use it directly.
			if !strings.Contains(raw, "{") {
				return strings.TrimSpace(raw), nil
			}
			return "", fmt.Errorf("parsing description response %q: %w", raw, jsonErr)
		}
		log.Printf("llm: description JSON parse failed for category %q (attempt %d/%d), retrying", name, attempt, maxAttempts)
	}
	return "", fmt.Errorf("unreachable")
}

// GetDocument returns the document with id, if present.
func (idx *Index) GetDocument(id string) (common.Document, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	d, ok := idx.docs[id]
	if !ok {
		return common.Document{}, false
	}
	return d.toCommon(), true
}

// Remove deletes a document and detaches it from every category (pruning any
// left empty).
func (idx *Index) Remove(id string) (common.Document, bool) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	doc, ok := idx.docs[id]
	if !ok {
		return common.Document{}, false
	}
	idx.detachLocked(doc)
	delete(idx.docs, id)
	return doc.toCommon(), true
}

func (idx *Index) DocCount() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.docs)
}

func (idx *Index) CategoryCount() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.categories)
}

// ListCategories returns every category in unspecified order.
func (idx *Index) ListCategories() []common.CategorySummary {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	out := make([]common.CategorySummary, 0, len(idx.categories))
	for _, c := range idx.categories {
		out = append(out, c.summary(len(idx.catDocs[c.Name])))
	}
	return out
}

// GetCategory returns the category named name, if present.
func (idx *Index) GetCategory(name string) (common.CategorySummary, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	c, ok := idx.categories[name]
	if !ok {
		return common.CategorySummary{}, false
	}
	return c.summary(len(idx.catDocs[name])), true
}

// DocCountByCategory returns the number of documents in a category.
func (idx *Index) DocCountByCategory(name string) int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.catDocs[name])
}

// Search embeds the query, selects the topN nearest categories by embedding
// cosine similarity, and returns up to maxDocsPerCategory member documents
// from each — a document shared by two matched categories is only returned
// once, scored by the higher-ranked category's similarity. limit caps the
// total number of documents returned, same as the embedding engine.
func (idx *Index) Search(ctx context.Context, query string, limit int) (common.Results, error) {
	if limit <= 0 {
		limit = 10
	}
	empty := common.Results{Documents: []common.SearchResult{}, MatchedCategories: []common.CategoryMatch{}}

	qv, err := idx.embedder.Embed(ctx, query)
	if err != nil {
		return common.Results{}, fmt.Errorf("embedding query: %w", err)
	}
	qn := vec.Norm(qv)
	if qn == 0 {
		return empty, nil
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	catCands := make([]common.CategoryMatch, 0, idx.topN)
	ch := make([]common.InternalHit, 0, idx.topN)
	for name, c := range idx.categories {
		if c.Norm == 0 {
			continue // profile not generated yet
		}
		sim := vec.Cosine(qv, c.Vector, qn, c.Norm)
		score := int(sim * scoreScale)
		if len(ch) < idx.topN {
			catCands = append(catCands, common.CategoryMatch{Name: name, Score: float64(sim)})
			ch = common.HeapPushHit(ch, common.InternalHit{ID: uint32(len(catCands) - 1), Score: score})
		} else if ch[0].Score < score {
			slot := ch[0].ID
			catCands[slot] = common.CategoryMatch{Name: name, Score: float64(sim)}
			common.HeapReplaceTop(ch, common.InternalHit{ID: slot, Score: score})
		}
	}
	if len(ch) == 0 {
		return empty, nil
	}

	matched := make([]common.CategoryMatch, len(ch))
	for i := len(ch) - 1; i >= 0; i-- {
		hit := ch[0]
		if i > 0 {
			ch[0] = ch[i]
			common.SiftDownHit(ch, 0, i)
		}
		matched[i] = catCands[hit.ID]
	}

	seen := make(map[string]bool)
	results := make([]common.SearchResult, 0, limit)
	for _, m := range matched {
		taken := 0
		for id := range idx.catDocs[m.Name] {
			if taken >= idx.maxDocsPerCategory {
				break
			}
			if seen[id] {
				continue
			}
			d, ok := idx.docs[id]
			if !ok {
				continue
			}
			seen[id] = true
			taken++
			results = append(results, common.SearchResult{Document: d.toCommon(), Score: m.Score, Categories: d.Categories})
			if len(results) >= limit {
				return common.Results{Documents: results, MatchedCategories: matched}, nil
			}
		}
	}

	return common.Results{Documents: results, MatchedCategories: matched}, nil
}

func (idx *Index) addDocToCategoryLocked(id, name string) {
	c, ok := idx.categories[name]
	if !ok {
		return
	}
	c.Count++
	set := idx.catDocs[name]
	if set == nil {
		set = make(map[string]struct{})
		idx.catDocs[name] = set
	}
	set[id] = struct{}{}
}

// detachLocked removes a document from every category it belonged to,
// pruning any category left with no members.
func (idx *Index) detachLocked(doc Document) {
	for _, name := range doc.Categories {
		if set := idx.catDocs[name]; set != nil {
			delete(set, doc.ID)
		}
		if c, ok := idx.categories[name]; ok {
			c.Count--
			if c.Count <= 0 {
				delete(idx.categories, name)
				delete(idx.catDocs, name)
			}
		}
	}
}

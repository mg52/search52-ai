package search

import (
	"context"
	"fmt"
	"sort"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/store"
	"github.com/mg52/search52-ai/internal/vec"
)

type Engine struct {
	embedding *ai.EmbeddingClient
	store     store.Store
	topN      int // number of nearest categories to scan
}

func NewEngine(embedding *ai.EmbeddingClient, st store.Store, topN int) *Engine {
	if topN <= 0 {
		topN = 3
	}
	return &Engine{embedding: embedding, store: st, topN: topN}
}

type Query struct {
	Q     string `json:"q"`
	Limit int    `json:"limit"`
}

// Results is the search response: the ranked documents plus the categories the
// query matched (independent of any single document).
type Results struct {
	Documents         []store.SearchResult  `json:"documents"`
	MatchedCategories []store.CategoryMatch `json:"matched_categories"`
}

// Search embeds the query, finds the nearest categories, and ranks their member
// documents by cosine similarity of the query to each document's vector.
func (e *Engine) Search(ctx context.Context, q Query) (Results, error) {
	if q.Limit <= 0 {
		q.Limit = 10
	}
	empty := Results{Documents: []store.SearchResult{}, MatchedCategories: []store.CategoryMatch{}}

	queryVec, err := e.embedding.Embed(ctx, q.Q)
	if err != nil {
		return Results{}, fmt.Errorf("embedding query: %w", err)
	}
	queryNorm := vec.Norm(queryVec)
	if queryNorm == 0 {
		return empty, nil
	}

	candidates, matched := e.candidates(queryVec, queryNorm)
	if matched == nil {
		matched = []store.CategoryMatch{}
	}
	if len(candidates) == 0 {
		return Results{Documents: []store.SearchResult{}, MatchedCategories: matched}, nil
	}

	results := make([]store.SearchResult, 0, len(candidates))
	for _, doc := range candidates {
		if doc.Norm == 0 {
			continue
		}
		sim := vec.Cosine(queryVec, doc.Vector, queryNorm, doc.Norm)
		if sim <= 0 {
			continue // irrelevant to the query
		}
		results = append(results, store.SearchResult{
			Document:   doc,
			Score:      sim,
			Categories: doc.Categories,
		})
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > q.Limit {
		results = results[:q.Limit]
	}

	return Results{Documents: results, MatchedCategories: matched}, nil
}

// candidates ranks every category by cosine similarity of its centroid to the
// query, takes the topN nearest, and returns their deduped member documents
// along with the matched categories. The store only hands back stored data; the
// similarity ranking is the search layer's business.
func (e *Engine) candidates(queryVec []float64, queryNorm float64) ([]store.Document, []store.CategoryMatch) {
	cats := e.store.ListCategories()
	matches := make([]store.CategoryMatch, 0, len(cats))
	for _, c := range cats {
		if c.Norm == 0 {
			continue
		}
		matches = append(matches, store.CategoryMatch{Name: c.Name, Score: vec.Cosine(queryVec, c.Centroid, queryNorm, c.Norm)})
	}
	if len(matches) == 0 {
		return nil, nil
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
	if e.topN < len(matches) {
		matches = matches[:e.topN]
	}

	seen := make(map[string]struct{})
	var docs []store.Document
	for _, m := range matches {
		for _, doc := range e.store.DocsInCategory(m.Name) {
			if _, dup := seen[doc.ID]; dup {
				continue
			}
			seen[doc.ID] = struct{}{}
			docs = append(docs, doc)
		}
	}
	return docs, matches
}

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
	threshold float64
}

func NewEngine(embedding *ai.EmbeddingClient, st store.Store, threshold float64) *Engine {
	return &Engine{
		embedding: embedding,
		store:     st,
		threshold: threshold,
	}
}

type Query struct {
	Q            string  `json:"q"`
	Limit        int     `json:"limit"`
	VectorWeight float64 `json:"vector_weight"`
	TagWeight    float64 `json:"tag_weight"`
}

// TagMatch is a single tag the query matched, with its similarity to the query.
type TagMatch struct {
	Tag   string  `json:"tag"`
	Score float64 `json:"score"`
}

// Results is the search response: the ranked documents plus the query-level
// list of which tags the query matched (independent of any document).
type Results struct {
	Documents   []store.SearchResult `json:"documents"`
	MatchedTags []TagMatch           `json:"matched_tags"`
}

func (e *Engine) Search(ctx context.Context, q Query) (Results, error) {
	if q.Limit <= 0 {
		q.Limit = 10
	}
	if q.VectorWeight == 0 && q.TagWeight == 0 {
		q.VectorWeight = 0.6
		q.TagWeight = 0.4
	}

	empty := Results{Documents: []store.SearchResult{}, MatchedTags: []TagMatch{}}

	queryVec, err := e.embedding.Embed(ctx, q.Q)
	if err != nil {
		return Results{}, fmt.Errorf("embedding query: %w", err)
	}

	queryNorm := vec.Norm(queryVec)
	if queryNorm == 0 {
		return empty, nil
	}

	allTags := e.store.ListTags()

	// tag name to similarity float
	matchedTagSimilarity := make(map[string]float64)

	for _, tag := range allTags {
		if tag.Norm == 0 {
			continue // tag has no embedding yet
		}
		sim := vec.Cosine(queryVec, tag.Vector, queryNorm, tag.Norm)
		if sim >= e.threshold {
			matchedTagSimilarity[tag.Name] = sim
		}
	}

	if len(matchedTagSimilarity) == 0 {
		return empty, nil
	}

	// The query-level matched tags, sorted by similarity (most relevant first).
	// This sits alongside the documents, independent of which docs matched.
	matchedTags := make([]TagMatch, 0, len(matchedTagSimilarity))
	for name, sim := range matchedTagSimilarity {
		matchedTags = append(matchedTags, TagMatch{Tag: name, Score: sim})
	}
	sort.Slice(matchedTags, func(i, j int) bool {
		return matchedTags[i].Score > matchedTags[j].Score
	})

	// Accumulate one hit per document: which query tags matched it, and the
	// strongest of those tag similarities. maxSim is only final once every tag
	// is processed, so ranking happens in a second pass below.
	type hit struct {
		doc     store.Document
		matched []string
		maxSim  float64
		score   float64
	}
	acc := make(map[string]*hit)
	for tagName, sim := range matchedTagSimilarity {
		for _, id := range e.store.GetDocIDsByTag(tagName) {
			h := acc[id]
			if h == nil {
				doc, ok := e.store.GetDocument(id)
				if !ok {
					continue
				}
				h = &hit{doc: doc}
				acc[id] = h
			}
			h.matched = append(h.matched, tagName)
			if sim > h.maxSim {
				h.maxSim = sim
			}
		}
	}

	// Rank with a single equation that folds together how strongly the document
	// matched (maxSim) and how many query tags it shares (tagMatchScore), tuned by
	// q.VectorWeight / q.TagWeight. The score is quantized into an integer bucket
	// so that walking buckets top-down replaces the O(n log n) comparison sort with
	// an O(n + range) bucket sort, and still stops as soon as q.Limit is reached.
	const bucketScale = 100
	maxBucket := 0
	buckets := make(map[int][]*hit)
	for _, h := range acc {
		var tagMatchScore float64
		if len(h.doc.Tags) > 0 {
			tagMatchScore = float64(len(h.matched)) / float64(len(h.doc.Tags))
			// tagMatchScore = float64(len(h.matched))
		}
		h.score = (h.maxSim * q.VectorWeight) + (tagMatchScore * q.TagWeight)

		b := int(h.score * bucketScale)
		buckets[b] = append(buckets[b], h)
		if b > maxBucket {
			maxBucket = b
		}
	}

	results := make([]store.SearchResult, 0, min(q.Limit, len(acc)))
	for b := maxBucket; b >= 0; b-- {
		for _, h := range buckets[b] {
			results = append(results, store.SearchResult{
				Document:    h.doc,
				Score:       h.score,
				MatchedTags: h.matched,
			})
			if len(results) >= q.Limit {
				return Results{Documents: results, MatchedTags: matchedTags}, nil
			}
		}
	}

	return Results{Documents: results, MatchedTags: matchedTags}, nil
}

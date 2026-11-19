// Package index is an in-memory, LLM-labelled search index. Each document is
// turned into labels by the LLM; search labels the query the same way and
// returns the documents sharing labels with it.
package index

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"

	"github.com/mg52/search52-ai/internal/ai"
)

// Chatter completes a system/user prompt pair. Implemented by ai.LLMClient.
type Chatter interface {
	Complete(ctx context.Context, system, user string) (string, error)
}

// Document is what the caller indexes and gets back from search, unchanged.
type Document struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

// Result is a search hit: the stored document and its score.
type Result struct {
	Document
	Score float64 `json:"score"`
}

// Options tunes labelling and search. Zero values switch a limit off, except
// where noted; config.Load supplies the defaults.
type Options struct {
	// MaxLabels is how many labels the LLM is asked for, and the most that are
	// kept: the first ones it returns (broad categories come first) (<= 0: keep all).
	MaxLabels int
	// MaxLabelWords: labels with more words are ignored (<= 0: no limit).
	MaxLabelWords int
	// MinLabelDocs is the fewest documents a label counts as having when weighing
	// a search, so a label that only one or two documents happen to have cannot
	// dominate (<= 1: no floor).
	MinLabelDocs int
	// MaxResults is the most documents Search returns (<= 0: no limit).
	MaxResults int
	// MinRelativeScore drops results scoring less than this fraction of the best
	// result, which trims the long tail of documents that share only generic
	// labels (0: keep all).
	MinRelativeScore float64
	// SendExistingLabels adds the labels the index already has, most used first,
	// to the LLM prompt so it reuses them instead of inventing synonyms. Every
	// call carries the whole list, so the prompt grows with the index.
	SendExistingLabels bool
}

type Index struct {
	llm  Chatter
	opts Options

	mu      sync.RWMutex
	docs    map[string]Document            // id -> document metadata
	byID    map[string][]string            // id -> labels
	byLabel map[string]map[string]struct{} // label -> document ids
}

func New(llm Chatter, opts Options) *Index {
	return &Index{
		llm:     llm,
		opts:    opts,
		docs:    make(map[string]Document),
		byID:    make(map[string][]string),
		byLabel: make(map[string]map[string]struct{}),
	}
}

// Add labels content with the LLM and stores it under id, replacing any
// existing document with that id. It returns the labels the document got.
func (idx *Index) Add(ctx context.Context, id, content string) ([]string, error) {
	labels, err := idx.labels(ctx, content)
	if err != nil {
		return nil, err
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.removeLocked(id)
	idx.docs[id] = Document{ID: id, Content: content}
	idx.byID[id] = labels
	for _, l := range labels {
		if idx.byLabel[l] == nil {
			idx.byLabel[l] = make(map[string]struct{})
		}
		idx.byLabel[l][id] = struct{}{}
	}
	return labels, nil
}

// Remove deletes the document, reporting whether it existed.
func (idx *Index) Remove(id string) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	_, ok := idx.docs[id]
	idx.removeLocked(id)
	return ok
}

func (idx *Index) removeLocked(id string) {
	for _, l := range idx.byID[id] {
		delete(idx.byLabel[l], id)
		if len(idx.byLabel[l]) == 0 {
			delete(idx.byLabel, l)
		}
	}
	delete(idx.byID, id)
	delete(idx.docs, id)
}

// Search labels query with the LLM and returns the documents sharing at least
// one label with it, plus the query's labels.
//
// A rare label counts for more than a common one: each of the query's labels
// that is in the index is worth total/size, where size is the number of
// documents with that label and total is the sum of those sizes over the
// query's labels (not the whole index; labels missing from the index count 0).
// With labels of 100, 200 and 150 documents total is 450, so the weights are
// 4.5, 2.25 and 3. To stop a label that only one or two documents happen to
// have from dominating, size counts as at least Options.MinLabelDocs when weighing. A
// document's score is the sum of the weights of the labels it has. Results are
// ordered by score, then id, and cut to the first Options.MaxResults. Results
// scoring under Options.MinRelativeScore of the best one are dropped too (the best result always
// stays, so a query whose labels all match something never comes back empty).
func (idx *Index) Search(ctx context.Context, query string) ([]Result, []string, error) {
	labels, err := idx.labels(ctx, query)
	if err != nil {
		return nil, nil, err
	}

	idx.mu.RLock()
	total := 0.0
	for _, l := range labels {
		total += float64(len(idx.byLabel[l]))
	}
	scores := make(map[string]float64)
	for _, l := range labels {
		for id := range idx.byLabel[l] {
			scores[id] += total / float64(max(len(idx.byLabel[l]), idx.opts.MinLabelDocs))
		}
	}
	results := make([]Result, 0, len(scores))
	for id, score := range scores {
		results = append(results, Result{Document: idx.docs[id], Score: math.Round(score*100) / 100})
	}
	idx.mu.RUnlock()

	sort.Slice(results, func(i, j int) bool {
		a, b := results[i], results[j]
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		return a.ID < b.ID
	})
	if len(results) > 0 && idx.opts.MinRelativeScore > 0 {
		cut := results[0].Score * idx.opts.MinRelativeScore
		results = results[:sort.Search(len(results), func(i int) bool { return results[i].Score < cut })]
	}
	if max := idx.opts.MaxResults; max > 0 && len(results) > max {
		results = results[:max]
	}
	return results, labels, nil
}

// labels asks the LLM for the normalized labels of text.
func (idx *Index) labels(ctx context.Context, text string) ([]string, error) {
	prompt := ai.Prompt(idx.opts.MaxLabels, idx.opts.MaxLabelWords)
	if idx.opts.SendExistingLabels {
		prompt = ai.PromptWithLabels(idx.opts.MaxLabels, idx.opts.MaxLabelWords, idx.existingLabels())
	}
	raw, err := idx.llm.Complete(ctx, prompt, text)
	if err != nil {
		return nil, err
	}
	labels := parseLabels(raw, idx.opts.MaxLabelWords)
	if len(labels) == 0 {
		return nil, fmt.Errorf("LLM returned no labels: %q", raw)
	}
	if max := idx.opts.MaxLabels; max > 0 && len(labels) > max {
		labels = labels[:max]
	}
	return labels, nil
}

// existingLabels returns every label in the index, most used first (ties by
// name, so the prompt is the same for the same index).
func (idx *Index) existingLabels() []string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	out := make([]string, 0, len(idx.byLabel))
	for l := range idx.byLabel {
		out = append(out, l)
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if na, nb := len(idx.byLabel[a]), len(idx.byLabel[b]); na != nb {
			return na > nb
		}
		return a < b
	})
	return out
}

// parseLabels reads "label1, label2" (newlines also separate) into
// de-duplicated labels, tolerating the junk models add around them: code
// fences, quotes, bullets, trailing periods and blank entries. Each label is
// lowercased and its spaces, underscores and hyphens become single hyphens
// ("Noise Cancelling" and "noise_cancelling" both give "noise-cancelling").
// Labels of more than maxWords words are dropped, and each word loses a
// plural ending (see singular) so "waffles" and "waffle" match.
func parseLabels(raw string, maxWords int) []string {
	fields := strings.FieldsFunc(raw, func(r rune) bool { return r == ',' || r == '\n' })
	seen := make(map[string]struct{}, len(fields))
	out := make([]string, 0, len(fields))
	for _, l := range fields {
		l = strings.ToLower(strings.TrimSpace(l))
		if strings.HasPrefix(l, "```") {
			continue // a fence line such as ``` or ```text
		}
		l = strings.Trim(l, " \t\r-*•\"'`.") // bullets, quotes, trailing period
		words := strings.FieldsFunc(l, func(r rune) bool { return r == ' ' || r == '\t' || r == '_' || r == '-' })
		if maxWords > 0 && len(words) > maxWords {
			continue
		}
		for i, w := range words {
			words[i] = singular(w)
		}
		l = strings.Join(words, "-")
		if _, dup := seen[l]; l == "" || dup {
			continue
		}
		seen[l] = struct{}{}
		out = append(out, l)
	}
	return out
}

// singular drops a plural ending: "batteries" -> "battery", "waffles" -> "waffle".
// shortcut: English only and no -es or irregular plurals ("glasses" stays
// "glasse", "mice" stays), fine while both sides are stemmed the same way;
// swap in a real stemmer if odd mismatches show up.
func singular(w string) string {
	switch {
	case len(w) > 4 && strings.HasSuffix(w, "ies"):
		return w[:len(w)-3] + "y"
	case len(w) > 3 && strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss") && !strings.HasSuffix(w, "us") && !strings.HasSuffix(w, "is"):
		return w[:len(w)-1]
	}
	return w
}

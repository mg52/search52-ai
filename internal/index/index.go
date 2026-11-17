// Package index is an in-memory, LLM-labelled search index. Each document is
// turned into labels by the LLM; search labels the query the same way and
// returns the documents sharing labels with it.
package index

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/mg52/search52-ai/internal/ai"
)

// Chatter completes a system/user prompt pair. Implemented by ai.LLMClient.
type Chatter interface {
	Complete(ctx context.Context, system, user string) (string, error)
}

type Document struct {
	ID      string   `json:"id"`
	Content string   `json:"content"`
	Labels  []string `json:"labels,omitempty"`
}

// Result is a search hit: a document and its score, the number of the query's
// labels the document has.
type Result struct {
	Document
	Score int `json:"score"`
}

// Options tunes how the LLM is prompted.
type Options struct {
	MaxPromptLabels int // most existing labels shown to the LLM per call; <= 0 means all
	MinLabels       int // fewest labels the LLM is asked for
	MaxLabels       int // most labels the LLM is asked for; extras are dropped, <= 0 means no cap
}

type Index struct {
	llm  Chatter
	opts Options

	mu      sync.RWMutex
	docs    map[string]Document            // id -> document with its labels
	byLabel map[string]map[string]struct{} // label -> document ids
}

func New(llm Chatter, opts Options) *Index {
	return &Index{
		llm:     llm,
		opts:    opts,
		docs:    make(map[string]Document),
		byLabel: make(map[string]map[string]struct{}),
	}
}

// Add labels content with the LLM and stores it under id, replacing any
// existing document with that id.
func (idx *Index) Add(ctx context.Context, id, content string) (Document, error) {
	labels, err := idx.labels(ctx, ai.IndexPrompt, content)
	if err != nil {
		return Document{}, err
	}
	doc := Document{ID: id, Content: content, Labels: labels}

	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.removeLocked(id)
	idx.docs[id] = doc
	for _, l := range labels {
		if idx.byLabel[l] == nil {
			idx.byLabel[l] = make(map[string]struct{})
		}
		idx.byLabel[l][id] = struct{}{}
	}
	return doc, nil
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
	for _, l := range idx.docs[id].Labels {
		delete(idx.byLabel[l], id)
		if len(idx.byLabel[l]) == 0 {
			delete(idx.byLabel, l)
		}
	}
	delete(idx.docs, id)
}

// Search labels query with the LLM and returns the documents sharing at least
// one label with it, plus the query's labels. A document's score is how many
// of the query's labels it has; results are ordered by score, then id.
func (idx *Index) Search(ctx context.Context, query string) ([]Result, []string, error) {
	labels, err := idx.labels(ctx, ai.SearchPrompt, query)
	if err != nil {
		return nil, nil, err
	}

	idx.mu.RLock()
	shared := make(map[string]int)
	for _, l := range labels {
		for id := range idx.byLabel[l] {
			shared[id]++
		}
	}
	results := make([]Result, 0, len(shared))
	for id, score := range shared {
		results = append(results, Result{Document: idx.docs[id], Score: score})
	}
	idx.mu.RUnlock()

	sort.Slice(results, func(i, j int) bool {
		a, b := results[i], results[j]
		if a.Score != b.Score {
			return a.Score > b.Score
		}
		return a.ID < b.ID
	})
	return results, labels, nil
}

// labels asks the LLM for labels of text under the given prompt builder, which
// is shown the labels the index already has.
func (idx *Index) labels(ctx context.Context, prompt func([]string, int, int) string, text string) ([]string, error) {
	raw, err := idx.llm.Complete(ctx, prompt(idx.existingLabels(), idx.opts.MinLabels, idx.opts.MaxLabels), text)
	if err != nil {
		return nil, err
	}
	labels := parseLabels(raw)
	if len(labels) == 0 {
		return nil, fmt.Errorf("LLM returned no labels: %q", raw)
	}
	if max := idx.opts.MaxLabels; max > 0 && len(labels) > max {
		labels = labels[:max]
	}
	return labels, nil
}

// existingLabels returns the labels in the index for the prompt: most used
// first (ties by name), capped at Options.MaxPromptLabels.
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
	if idx.opts.MaxPromptLabels > 0 && len(out) > idx.opts.MaxPromptLabels {
		out = out[:idx.opts.MaxPromptLabels]
	}
	return out
}

// parseLabels reads "label1, label2" (newlines also separate) into trimmed,
// lowercased, de-duplicated labels, tolerating the junk models add around
// them: code fences, quotes, bullets, trailing periods and blank entries.
func parseLabels(raw string) []string {
	fields := strings.FieldsFunc(raw, func(r rune) bool { return r == ',' || r == '\n' })
	seen := make(map[string]struct{}, len(fields))
	out := make([]string, 0, len(fields))
	for _, l := range fields {
		l = strings.ToLower(strings.TrimSpace(l))
		if strings.HasPrefix(l, "```") {
			continue // a fence line such as ``` or ```text
		}
		l = strings.Trim(l, " \t\r-*•\"'`.") // bullets, quotes, trailing period
		if _, dup := seen[l]; l == "" || dup {
			continue
		}
		seen[l] = struct{}{}
		out = append(out, l)
	}
	return out
}

// Package index is an in-memory, LLM-labelled search index. Each document is
// turned into labels by the LLM; search labels the query the same way and
// returns the documents sharing labels with it.
package index

import (
	"context"
	"encoding/json"
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

type Index struct {
	llm Chatter

	mu      sync.RWMutex
	docs    map[string]Document            // id -> document with its labels
	byLabel map[string]map[string]struct{} // label -> document ids
}

func New(llm Chatter) *Index {
	return &Index{
		llm:     llm,
		docs:    make(map[string]Document),
		byLabel: make(map[string]map[string]struct{}),
	}
}

// Add labels content with the LLM and stores it under id, replacing any
// existing document with that id.
func (idx *Index) Add(ctx context.Context, id, content string) (Document, error) {
	labels, err := idx.labels(ctx, content)
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
// one label with it, most shared labels first, plus the query's labels.
func (idx *Index) Search(ctx context.Context, query string) ([]Document, []string, error) {
	labels, err := idx.labels(ctx, query)
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
	docs := make([]Document, 0, len(shared))
	for id := range shared {
		docs = append(docs, idx.docs[id])
	}
	idx.mu.RUnlock()

	sort.Slice(docs, func(i, j int) bool {
		a, b := docs[i], docs[j]
		if shared[a.ID] != shared[b.ID] {
			return shared[a.ID] > shared[b.ID]
		}
		return a.ID < b.ID
	})
	return docs, labels, nil
}

func (idx *Index) labels(ctx context.Context, text string) ([]string, error) {
	raw, err := idx.llm.Complete(ctx, ai.LabelsSystemPrompt, text)
	if err != nil {
		return nil, err
	}
	var resp struct {
		Labels []string `json:"labels"`
	}
	if err := json.Unmarshal([]byte(extractJSON(raw)), &resp); err != nil {
		return nil, fmt.Errorf("parsing LLM labels %q: %w", raw, err)
	}
	labels := cleanLabels(resp.Labels)
	if len(labels) == 0 {
		return nil, fmt.Errorf("LLM returned no labels")
	}
	return labels, nil
}

// extractJSON pulls the first {...} object out of s, stripping markdown code
// fences and any trailing model-specific tokens.
func extractJSON(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start != -1 && end != -1 && end >= start {
		return s[start : end+1]
	}
	return strings.TrimSpace(s)
}

// cleanLabels trims whitespace, lowercases, drops blank entries, and removes
// duplicates while preserving order — guarding against models that emit junk
// like [""] or repeats.
func cleanLabels(raw []string) []string {
	seen := make(map[string]struct{}, len(raw))
	out := make([]string, 0, len(raw))
	for _, l := range raw {
		l = strings.TrimSpace(strings.ToLower(l))
		if _, dup := seen[l]; l == "" || dup {
			continue
		}
		seen[l] = struct{}{}
		out = append(out, l)
	}
	return out
}

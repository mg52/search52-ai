// Package engine is the facade over the two index implementations: embedding
// (cosine-similarity clustering, package embedding) and llm (LLM-decided
// categorization, package llm). It picks one by Kind at creation/load time
// and hands back the common Index interface, so callers (internal/handler)
// never need to know which kind they're talking to.
package engine

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/engine/common"
	"github.com/mg52/search52-ai/internal/engine/embedding"
	"github.com/mg52/search52-ai/internal/engine/llm"
)

// Kind selects which index implementation backs a given index.
type Kind string

const (
	KindEmbedding Kind = "embedding"
	KindLLM       Kind = "llm"
)

// Index is the interface both engine implementations satisfy. Handler code
// depends only on this, never on the concrete embedding/llm types.
type Index interface {
	Name() string
	Kind() string
	Process(ctx context.Context, id, content string) (common.Document, error)
	GetDocument(id string) (common.Document, bool)
	Remove(id string) (common.Document, bool)
	DocCount() int
	CategoryCount() int
	ListCategories() []common.CategorySummary
	GetCategory(name string) (common.CategorySummary, bool)
	DocCountByCategory(name string) int
	Search(ctx context.Context, query string, limit int) (common.Results, error)
	Save(dir string) error
}

var (
	_ Index = (*embedding.SearchEngine)(nil)
	_ Index = (*llm.Index)(nil)
)

// Deps are the shared clients every index kind may need. Embedder is
// required by both kinds; LLMClient/Prompts are only required for KindLLM.
type Deps struct {
	Embedder  common.Embedder
	LLMClient common.Chatter
	Prompts   *ai.Prompts
}

// Config is the union of every tunable across both index kinds. Fields not
// relevant to the chosen Kind are ignored.
type Config struct {
	// shared
	MaxCategoriesPerDoc int
	MaxCategories       int
	TopNCategories      int

	// embedding-only
	CategoryThreshold float64
	VarianceThreshold float64
	VarianceMinCount  int
	DisableSplit      bool

	// llm-only
	MaxDocsPerCategory int
}

// New constructs an empty index of the given kind. An empty Kind defaults to
// KindEmbedding for backward compatibility with callers that predate the LLM
// index kind.
func New(kind Kind, name string, deps Deps, cfg Config) (Index, error) {
	switch kind {
	case "", KindEmbedding:
		if deps.Embedder == nil {
			return nil, fmt.Errorf("embedding index requires an embedder")
		}
		return embedding.New(name, deps.Embedder, embedding.Config{
			CategoryThreshold:   cfg.CategoryThreshold,
			MaxCategoriesPerDoc: cfg.MaxCategoriesPerDoc,
			MaxCategories:       cfg.MaxCategories,
			TopNCategories:      cfg.TopNCategories,
			VarianceThreshold:   cfg.VarianceThreshold,
			VarianceMinCount:    cfg.VarianceMinCount,
			DisableSplit:        cfg.DisableSplit,
		}), nil
	case KindLLM:
		if deps.Embedder == nil || deps.LLMClient == nil || deps.Prompts == nil {
			return nil, fmt.Errorf("llm index requires an embedder, an llm client, and prompts")
		}
		return llm.New(name, deps.LLMClient, deps.Embedder, deps.Prompts, llm.Config{
			MaxCategoriesPerDoc: cfg.MaxCategoriesPerDoc,
			MaxCategories:       cfg.MaxCategories,
			TopNCategories:      cfg.TopNCategories,
			MaxDocsPerCategory:  cfg.MaxDocsPerCategory,
		}), nil
	default:
		return nil, fmt.Errorf("unknown index kind %q", kind)
	}
}

// Load reconstructs whichever index kind is persisted under dir, identified
// by which of the two engines' snapshot files is present.
func Load(dir string, deps Deps) (Index, error) {
	if _, err := os.Stat(filepath.Join(dir, embedding.SnapshotFile)); err == nil {
		if deps.Embedder == nil {
			return nil, fmt.Errorf("embedding index requires an embedder")
		}
		return embedding.Load(dir, deps.Embedder)
	}
	if _, err := os.Stat(filepath.Join(dir, llm.SnapshotFile)); err == nil {
		if deps.Embedder == nil || deps.LLMClient == nil || deps.Prompts == nil {
			return nil, fmt.Errorf("llm index requires an embedder, an llm client, and prompts")
		}
		return llm.Load(dir, deps.LLMClient, deps.Embedder, deps.Prompts)
	}
	return nil, fmt.Errorf("no recognized index snapshot in %s", dir)
}

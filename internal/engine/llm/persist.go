package llm

import (
	"bufio"
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/engine/common"
)

// SnapshotFile is the single file each LLM index directory holds. It uses a
// different name than the embedding engine's index.gob so the facade's Load
// can tell the two kinds apart by probing the directory.
const SnapshotFile = "index_llm.gob"

type snapshot struct {
	Name               string
	MaxPerDoc          int
	MaxCategories      int
	TopN               int
	MaxDocsPerCategory int
	Documents          []docSnapshot
	Categories         []catSnapshot
	CatDocs            map[string][]string
}

type docSnapshot struct {
	ID         string
	Content    string
	Categories []string
	CreatedAt  time.Time
}

type catSnapshot struct {
	Name        string
	Description string
	Vector      []float32
	Norm        float32
	Examples    []string
	Count       int
	CreatedAt   time.Time
}

// Save writes a gzip-compressed gob snapshot to dir/index_llm.gob via a temp
// file + atomic rename. The RLock is held only while copying state into the
// snapshot, not during encoding or disk I/O.
func (idx *Index) Save(dir string) error {
	idx.mu.RLock()
	snap := snapshot{
		Name:               idx.name,
		MaxPerDoc:          idx.maxPerDoc,
		MaxCategories:      idx.maxCategories,
		TopN:               idx.topN,
		MaxDocsPerCategory: idx.maxDocsPerCategory,
		Documents:          make([]docSnapshot, 0, len(idx.docs)),
		Categories:         make([]catSnapshot, 0, len(idx.categories)),
		CatDocs:            make(map[string][]string, len(idx.catDocs)),
	}
	for _, d := range idx.docs {
		snap.Documents = append(snap.Documents, docSnapshot{
			ID:         d.ID,
			Content:    d.Content,
			Categories: d.Categories,
			CreatedAt:  d.CreatedAt,
		})
	}
	for _, c := range idx.categories {
		snap.Categories = append(snap.Categories, catSnapshot{
			Name:        c.Name,
			Description: c.Description,
			Vector:      c.Vector,
			Norm:        c.Norm,
			Examples:    c.Examples,
			Count:       c.Count,
			CreatedAt:   c.CreatedAt,
		})
	}
	for name, set := range idx.catDocs {
		ids := make([]string, 0, len(set))
		for id := range set {
			ids = append(ids, id)
		}
		snap.CatDocs[name] = ids
	}
	idx.mu.RUnlock()

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}
	file := filepath.Join(dir, SnapshotFile)
	tmp := file + ".tmp"

	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmp, err)
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	gz := gzip.NewWriter(bw)
	encErr := gob.NewEncoder(gz).Encode(snap)
	gzErr := gz.Close()
	flushErr := bw.Flush()
	closeErr := f.Close()
	if err := firstErr(encErr, gzErr, flushErr, closeErr); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("encode snapshot: %w", err)
	}
	if err := os.Rename(tmp, file); err != nil {
		return fmt.Errorf("rename %s: %w", tmp, err)
	}
	return nil
}

// Load reconstructs an Index from dir/index_llm.gob and attaches llmClient,
// embedder, and prompts (none of which are persisted). It returns an error
// if the snapshot is absent.
func Load(dir string, llmClient common.Chatter, embedder common.Embedder, prompts *ai.Prompts) (*Index, error) {
	file := filepath.Join(dir, SnapshotFile)
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(bufio.NewReaderSize(f, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("gzip reader %s: %w", file, err)
	}
	defer gz.Close()

	var snap snapshot
	if err := gob.NewDecoder(gz).Decode(&snap); err != nil {
		return nil, fmt.Errorf("decode %s: %w", file, err)
	}

	idx := New(snap.Name, llmClient, embedder, prompts, Config{
		MaxCategoriesPerDoc: snap.MaxPerDoc,
		MaxCategories:       snap.MaxCategories,
		TopNCategories:      snap.TopN,
		MaxDocsPerCategory:  snap.MaxDocsPerCategory,
	})
	for _, d := range snap.Documents {
		idx.docs[d.ID] = Document{
			ID:         d.ID,
			Content:    d.Content,
			Categories: d.Categories,
			CreatedAt:  d.CreatedAt,
		}
	}
	for _, c := range snap.Categories {
		idx.categories[c.Name] = &Category{
			Name:        c.Name,
			Description: c.Description,
			Vector:      c.Vector,
			Norm:        c.Norm,
			Examples:    c.Examples,
			Count:       c.Count,
			CreatedAt:   c.CreatedAt,
		}
	}
	for name, ids := range snap.CatDocs {
		set := make(map[string]struct{}, len(ids))
		for _, id := range ids {
			set[id] = struct{}{}
		}
		idx.catDocs[name] = set
	}
	return idx, nil
}

func firstErr(errs ...error) error {
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}

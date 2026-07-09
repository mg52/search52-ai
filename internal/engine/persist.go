package engine

import (
	"bufio"
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// snapshotFile is the single file each index directory holds.
const snapshotFile = "index.gob"

// snapshot is the gob-serializable form of a SearchEngine. Centroid norms are
// persisted alongside the centroids (not recomputed on load).
type snapshot struct {
	Name              string
	Threshold         float64
	MaxPerDoc         int
	MaxCategories     int
	TopN              int
	VarianceThreshold float64
	VarianceMinCount  int
	NextCategoryID    int
	Documents         []docSnapshot
	Categories        []catSnapshot
	CatDocs           map[string][]string
}

type docSnapshot struct {
	ID             string
	Content        string
	Categories     []string
	Vector         []float32
	Norm           float32
	CreatedAt      time.Time
	ForcedFallback bool
}

type catSnapshot struct {
	Name         string
	Centroid     []float32
	Norm         float32
	Count        int
	CreatedAt    time.Time
	WelfordCount int
	WelfordMean  float64
	WelfordM2    float64
	Variance     float64
	ShouldSplit  bool
}

// Save writes a gzip-compressed gob snapshot to dir/index.gob via a temp file +
// atomic rename. The RLock is held only while copying state into the snapshot,
// not during encoding or disk I/O.
func (se *SearchEngine) Save(dir string) error {
	se.mu.RLock()
	snap := snapshot{
		Name:              se.name,
		Threshold:         float64(se.threshold),
		MaxPerDoc:         se.maxPerDoc,
		MaxCategories:     se.maxCategories,
		TopN:              se.topN,
		VarianceThreshold: se.varianceThreshold,
		VarianceMinCount:  se.varianceMinCount,
		NextCategoryID:    se.nextCategoryID,
		Documents:         make([]docSnapshot, 0, len(se.docs)),
		Categories:        make([]catSnapshot, 0, len(se.categories)),
		CatDocs:           make(map[string][]string, len(se.catDocs)),
	}
	for _, d := range se.docs {
		snap.Documents = append(snap.Documents, docSnapshot{
			ID:             d.ID,
			Content:        d.Content,
			Categories:     d.Categories,
			Vector:         d.Vector,
			Norm:           d.Norm,
			CreatedAt:      d.CreatedAt,
			ForcedFallback: d.ForcedFallback,
		})
	}
	for _, c := range se.categories {
		snap.Categories = append(snap.Categories, catSnapshot{
			Name:         c.Name,
			Centroid:     c.Centroid,
			Norm:         c.Norm,
			Count:        c.Count,
			CreatedAt:    c.CreatedAt,
			WelfordCount: c.welfordCount,
			WelfordMean:  c.welfordMean,
			WelfordM2:    c.welfordM2,
			Variance:     c.Variance,
			ShouldSplit:  c.ShouldSplit,
		})
	}
	for name, set := range se.catDocs {
		ids := make([]string, 0, len(set))
		for id := range set {
			ids = append(ids, id)
		}
		snap.CatDocs[name] = ids
	}
	se.mu.RUnlock()

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}
	file := filepath.Join(dir, snapshotFile)
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

// Load reconstructs a SearchEngine from dir/index.gob and attaches embedder
// (embedders are not persisted). It returns an error if the snapshot is absent.
func Load(dir string, embedder Embedder) (*SearchEngine, error) {
	file := filepath.Join(dir, snapshotFile)
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

	se := New(snap.Name, embedder, Config{
		CategoryThreshold:   snap.Threshold,
		MaxCategoriesPerDoc: snap.MaxPerDoc,
		MaxCategories:       snap.MaxCategories,
		TopNCategories:      snap.TopN,
		VarianceThreshold:   snap.VarianceThreshold,
		VarianceMinCount:    snap.VarianceMinCount,
	})
	se.nextCategoryID = snap.NextCategoryID
	for _, d := range snap.Documents {
		se.docs[d.ID] = Document{
			ID:             d.ID,
			Content:        d.Content,
			Categories:     d.Categories,
			Vector:         d.Vector,
			Norm:           d.Norm,
			CreatedAt:      d.CreatedAt,
			ForcedFallback: d.ForcedFallback,
		}
	}
	for _, c := range snap.Categories {
		se.categories[c.Name] = &Category{
			Name:         c.Name,
			Centroid:     c.Centroid,
			Norm:         c.Norm,
			Count:        c.Count,
			CreatedAt:    c.CreatedAt,
			welfordCount: c.WelfordCount,
			welfordMean:  c.WelfordMean,
			welfordM2:    c.WelfordM2,
			Variance:     c.Variance,
			ShouldSplit:  c.ShouldSplit,
		}
	}
	for name, ids := range snap.CatDocs {
		set := make(map[string]struct{}, len(ids))
		for _, id := range ids {
			set[id] = struct{}{}
		}
		se.catDocs[name] = set
	}
	return se, nil
}

func firstErr(errs ...error) error {
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}

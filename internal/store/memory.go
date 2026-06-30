package store

import (
	"sort"
	"sync"
)

// MemoryStore is an in-memory implementation of Store. It owns three maps —
// documents, categories, and a category→doc-IDs index — guarded by a single
// RWMutex. It is deliberately dumb: every method is a straight read or write of
// stored data. All clustering and ranking decisions are made by callers.
type MemoryStore struct {
	mu         sync.RWMutex
	docs       map[string]Document
	docList    []string
	categories map[string]Category
	catDocs    map[string]map[string]struct{} // category name -> set of doc IDs
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		docs:       make(map[string]Document),
		categories: make(map[string]Category),
		catDocs:    make(map[string]map[string]struct{}),
	}
}

// PutDocument inserts or replaces a document by ID, tracking insertion order so
// ListDocuments is stable.
func (s *MemoryStore) PutDocument(doc Document) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.docs[doc.ID]; !exists {
		s.docList = append(s.docList, doc.ID)
	}
	s.docs[doc.ID] = doc
}

func (s *MemoryStore) GetDocument(id string) (Document, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	doc, ok := s.docs[id]
	return doc, ok
}

// DeleteDocument removes a document. It does NOT touch category memberships or
// centroids — the caller detaches those first (see pipeline.Categorizer.Remove).
func (s *MemoryStore) DeleteDocument(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.docs[id]; !ok {
		return false
	}
	delete(s.docs, id)
	for i, d := range s.docList {
		if d == id {
			s.docList = append(s.docList[:i], s.docList[i+1:]...)
			break
		}
	}
	return true
}

func (s *MemoryStore) ListDocuments(page, size int) ([]Document, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	total := len(s.docList)
	start := (page - 1) * size
	if start >= total {
		return nil, total
	}
	end := min(start+size, total)
	ids := s.docList[start:end]
	docs := make([]Document, 0, len(ids))
	for _, id := range ids {
		if doc, ok := s.docs[id]; ok {
			docs = append(docs, doc)
		}
	}
	return docs, total
}

func (s *MemoryStore) DocCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.docs)
}

// PutCategory inserts or replaces a category by Name. The store takes ownership
// of the supplied Centroid slice; callers must hand over a slice they no longer
// mutate (the pipeline always builds a fresh one), so concurrent readers never
// observe a half-updated centroid.
func (s *MemoryStore) PutCategory(cat Category) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.categories[cat.Name] = cat
}

func (s *MemoryStore) GetCategory(name string) (Category, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	c, ok := s.categories[name]
	return c, ok
}

// DeleteCategory drops a category and its index entry.
func (s *MemoryStore) DeleteCategory(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.categories, name)
	delete(s.catDocs, name)
}

func (s *MemoryStore) ListCategories() []Category {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cats := make([]Category, 0, len(s.categories))
	for _, c := range s.categories {
		cats = append(cats, c)
	}
	// Stable order so callers (and tests) see deterministic results.
	sort.Slice(cats, func(i, j int) bool { return cats[i].Name < cats[j].Name })
	return cats
}

func (s *MemoryStore) CategoryCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.categories)
}

func (s *MemoryStore) AddDocToCategory(id, name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	set := s.catDocs[name]
	if set == nil {
		set = make(map[string]struct{})
		s.catDocs[name] = set
	}
	set[id] = struct{}{}
}

func (s *MemoryStore) RemoveDocFromCategory(id, name string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if set := s.catDocs[name]; set != nil {
		delete(set, id)
	}
}

// DocsInCategory returns the member documents of a category (skipping any whose
// document record is gone). Order is unspecified.
func (s *MemoryStore) DocsInCategory(name string) []Document {
	s.mu.RLock()
	defer s.mu.RUnlock()
	set := s.catDocs[name]
	docs := make([]Document, 0, len(set))
	for id := range set {
		if doc, ok := s.docs[id]; ok {
			docs = append(docs, doc)
		}
	}
	return docs
}

func (s *MemoryStore) DocCountByCategory(name string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.catDocs[name])
}

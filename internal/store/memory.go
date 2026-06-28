package store

import (
	"fmt"
	"sync"
	"time"
)

type MemoryStore struct {
	mu      sync.RWMutex
	docs    map[string]Document
	docList []string
	tags    map[string]Tag
	tagDocs map[string]map[string]struct{} // inverted index: tag name -> set of doc IDs
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		docs:    make(map[string]Document),
		tags:    make(map[string]Tag),
		tagDocs: make(map[string]map[string]struct{}),
	}
}

func (s *MemoryStore) SaveDocument(doc Document) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if doc.CreatedAt.IsZero() {
		doc.CreatedAt = time.Now()
	}
	if old, exists := s.docs[doc.ID]; exists {
		s.unindexTags(doc.ID, old.Tags) // drop stale associations on re-save
	} else {
		s.docList = append(s.docList, doc.ID)
	}
	s.docs[doc.ID] = doc
	s.indexTags(doc.ID, doc.Tags)
	return nil
}

// indexTags adds doc ID to the inverted index for each tag. The set membership
// makes a (tag, doc) pair idempotent, so duplicate tag names are harmless.
func (s *MemoryStore) indexTags(id string, tags []string) {
	for _, t := range tags {
		set := s.tagDocs[t]
		if set == nil {
			set = make(map[string]struct{})
			s.tagDocs[t] = set
		}
		set[id] = struct{}{}
	}
}

// unindexTags removes doc ID from the inverted index for each tag in O(1).
func (s *MemoryStore) unindexTags(id string, tags []string) {
	for _, t := range tags {
		set := s.tagDocs[t]
		if set == nil {
			continue
		}
		delete(set, id)
		if len(set) == 0 {
			delete(s.tagDocs, t)
		}
	}
}

func (s *MemoryStore) GetDocument(id string) (Document, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	doc, ok := s.docs[id]
	return doc, ok
}

// DeleteDocument removes a document and all of its inverted-index associations,
// returning the deleted document. Tag definitions are left intact even if their
// posting list becomes empty.
func (s *MemoryStore) DeleteDocument(id string) (Document, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	doc, ok := s.docs[id]
	if !ok {
		return Document{}, false
	}
	s.unindexTags(id, doc.Tags)
	delete(s.docs, id)
	for i, d := range s.docList {
		if d == id {
			s.docList = append(s.docList[:i], s.docList[i+1:]...)
			break
		}
	}
	return doc, true
}

func (s *MemoryStore) ListDocuments(page, size int) ([]Document, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	total := len(s.docList)
	start := (page - 1) * size
	if start >= total {
		return nil, total
	}
	end := start + size
	if end > total {
		end = total
	}
	ids := s.docList[start:end]
	docs := make([]Document, 0, len(ids))
	for _, id := range ids {
		if doc, ok := s.docs[id]; ok {
			docs = append(docs, doc)
		}
	}
	return docs, total
}

func (s *MemoryStore) SaveTag(tag Tag) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if tag.CreatedAt.IsZero() {
		tag.CreatedAt = time.Now()
	}
	s.tags[tag.Name] = tag
	return nil
}

func (s *MemoryStore) GetTag(name string) (Tag, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	tag, ok := s.tags[name]
	return tag, ok
}

func (s *MemoryStore) UpdateTag(tag Tag) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.tags[tag.Name]; !ok {
		return fmt.Errorf("tag not found: %s", tag.Name)
	}
	s.tags[tag.Name] = tag
	return nil
}

func (s *MemoryStore) ListTags() []Tag {
	s.mu.RLock()
	defer s.mu.RUnlock()
	tags := make([]Tag, 0, len(s.tags))
	for _, t := range s.tags {
		tags = append(tags, t)
	}
	return tags
}

func (s *MemoryStore) GetTagNames() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	names := make([]string, 0, len(s.tags))
	for name := range s.tags {
		names = append(names, name)
	}
	return names
}

func (s *MemoryStore) GetDocumentsByTag(tag string) []Document {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := s.tagDocs[tag]
	result := make([]Document, 0, len(ids))
	for id := range ids {
		if doc, ok := s.docs[id]; ok {
			result = append(result, doc)
		}
	}
	return result
}

// GetDocIDsByTag returns the doc IDs associated with a tag from the inverted
// index. The result is a copy, safe to iterate without holding the store lock.
func (s *MemoryStore) GetDocIDsByTag(tag string) []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := s.tagDocs[tag]
	out := make([]string, 0, len(ids))
	for id := range ids {
		out = append(out, id)
	}
	return out
}

// DocCountByTag returns how many documents currently carry the tag, derived
// directly from the inverted index (the single source of truth).
func (s *MemoryStore) DocCountByTag(tag string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.tagDocs[tag])
}

func (s *MemoryStore) DocCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.docs)
}

func (s *MemoryStore) TagCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.tags)
}

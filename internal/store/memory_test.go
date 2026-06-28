package store

import (
	"testing"
	"time"
)

func doc(id string, tags ...string) Document {
	return Document{ID: id, Content: "content of " + id, Tags: tags}
}

func TestSaveAndGetDocument(t *testing.T) {
	s := NewMemoryStore()
	if err := s.SaveDocument(doc("d1", "a")); err != nil {
		t.Fatalf("SaveDocument: %v", err)
	}
	got, ok := s.GetDocument("d1")
	if !ok {
		t.Fatal("GetDocument: not found")
	}
	if got.ID != "d1" || got.CreatedAt.IsZero() {
		t.Errorf("unexpected doc: %+v", got)
	}
	if _, ok := s.GetDocument("missing"); ok {
		t.Error("expected missing doc to return false")
	}
}

func TestSaveDocumentPreservesCreatedAt(t *testing.T) {
	s := NewMemoryStore()
	ts := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	d := doc("d1", "a")
	d.CreatedAt = ts
	s.SaveDocument(d)
	got, _ := s.GetDocument("d1")
	if !got.CreatedAt.Equal(ts) {
		t.Errorf("CreatedAt overwritten: got %v", got.CreatedAt)
	}
}

func TestInvertedIndex(t *testing.T) {
	s := NewMemoryStore()
	s.SaveDocument(doc("d1", "a", "b"))
	s.SaveDocument(doc("d2", "a"))

	if got := s.DocCountByTag("a"); got != 2 {
		t.Errorf("DocCountByTag(a) = %d, want 2", got)
	}
	if got := s.DocCountByTag("b"); got != 1 {
		t.Errorf("DocCountByTag(b) = %d, want 1", got)
	}
	if got := s.DocCountByTag("missing"); got != 0 {
		t.Errorf("DocCountByTag(missing) = %d, want 0", got)
	}

	docs := s.GetDocumentsByTag("a")
	if len(docs) != 2 {
		t.Fatalf("GetDocumentsByTag(a) = %d docs, want 2", len(docs))
	}
	ids := s.GetDocIDsByTag("a")
	if len(ids) != 2 {
		t.Fatalf("GetDocIDsByTag(a) = %d ids, want 2", len(ids))
	}
	seen := map[string]bool{}
	for _, id := range ids {
		seen[id] = true
	}
	if !seen["d1"] || !seen["d2"] {
		t.Errorf("GetDocIDsByTag(a) = %v, want d1 and d2", ids)
	}
}

func TestDuplicateTagsDeduped(t *testing.T) {
	s := NewMemoryStore()
	s.SaveDocument(doc("d1", "a", "a", "a"))
	if got := s.DocCountByTag("a"); got != 1 {
		t.Errorf("DocCountByTag(a) = %d, want 1 (deduped)", got)
	}
	if got := s.GetDocIDsByTag("a"); len(got) != 1 {
		t.Errorf("GetDocIDsByTag(a) = %v, want single id", got)
	}
}

func TestResaveUpdatesIndex(t *testing.T) {
	s := NewMemoryStore()
	s.SaveDocument(doc("d1", "a", "b"))
	// Re-save the same ID with a different tag set.
	s.SaveDocument(doc("d1", "b", "c"))

	if got := s.DocCountByTag("a"); got != 0 {
		t.Errorf("stale tag a still present: count %d", got)
	}
	if got := s.GetDocIDsByTag("a"); len(got) != 0 {
		t.Errorf("stale tag a posting list: %v", got)
	}
	if got := s.DocCountByTag("b"); got != 1 {
		t.Errorf("DocCountByTag(b) = %d, want 1", got)
	}
	if got := s.DocCountByTag("c"); got != 1 {
		t.Errorf("DocCountByTag(c) = %d, want 1", got)
	}
	// Re-save must not create duplicates in the index.
	if got := s.GetDocIDsByTag("b"); len(got) != 1 {
		t.Errorf("GetDocIDsByTag(b) = %v, want single id", got)
	}
}

func TestDeleteDocument(t *testing.T) {
	s := NewMemoryStore()
	s.SaveDocument(doc("d1", "a", "b"))
	s.SaveDocument(doc("d2", "a"))

	deleted, ok := s.DeleteDocument("d1")
	if !ok {
		t.Fatal("DeleteDocument: expected ok")
	}
	if deleted.ID != "d1" {
		t.Errorf("returned doc = %s, want d1", deleted.ID)
	}
	if _, ok := s.GetDocument("d1"); ok {
		t.Error("d1 still retrievable after delete")
	}
	// Tag b had only d1 -> gone; tag a had d1,d2 -> now only d2.
	if got := s.DocCountByTag("b"); got != 0 {
		t.Errorf("DocCountByTag(b) = %d, want 0", got)
	}
	if got := s.DocCountByTag("a"); got != 1 {
		t.Errorf("DocCountByTag(a) = %d, want 1", got)
	}
	// docList kept consistent: total reflects remaining docs.
	_, total := s.ListDocuments(1, 10)
	if total != 1 {
		t.Errorf("total after delete = %d, want 1", total)
	}
	if s.DocCount() != 1 {
		t.Errorf("DocCount = %d, want 1", s.DocCount())
	}

	if _, ok := s.DeleteDocument("missing"); ok {
		t.Error("deleting missing doc should return false")
	}
}

func TestListDocumentsPagination(t *testing.T) {
	s := NewMemoryStore()
	for _, id := range []string{"d1", "d2", "d3", "d4", "d5"} {
		s.SaveDocument(doc(id, "a"))
	}
	page1, total := s.ListDocuments(1, 2)
	if total != 5 {
		t.Errorf("total = %d, want 5", total)
	}
	if len(page1) != 2 {
		t.Errorf("page1 len = %d, want 2", len(page1))
	}
	page3, _ := s.ListDocuments(3, 2)
	if len(page3) != 1 {
		t.Errorf("page3 len = %d, want 1", len(page3))
	}
	// Out-of-range page returns nil docs but correct total.
	pageX, total := s.ListDocuments(99, 2)
	if pageX != nil || total != 5 {
		t.Errorf("out-of-range page = %v total %d", pageX, total)
	}
}

func TestTagCRUD(t *testing.T) {
	s := NewMemoryStore()
	if err := s.SaveTag(Tag{Name: "a"}); err != nil {
		t.Fatalf("SaveTag: %v", err)
	}
	got, ok := s.GetTag("a")
	if !ok || got.CreatedAt.IsZero() {
		t.Errorf("GetTag(a) = %+v, ok=%v", got, ok)
	}
	if _, ok := s.GetTag("missing"); ok {
		t.Error("expected missing tag false")
	}

	got.Description = "updated"
	if err := s.UpdateTag(got); err != nil {
		t.Fatalf("UpdateTag: %v", err)
	}
	reread, _ := s.GetTag("a")
	if reread.Description != "updated" {
		t.Errorf("description = %q, want updated", reread.Description)
	}

	if err := s.UpdateTag(Tag{Name: "ghost"}); err == nil {
		t.Error("UpdateTag of missing tag should error")
	}
}

func TestListTagsAndNames(t *testing.T) {
	s := NewMemoryStore()
	s.SaveTag(Tag{Name: "a"})
	s.SaveTag(Tag{Name: "b"})
	if got := s.ListTags(); len(got) != 2 {
		t.Errorf("ListTags len = %d, want 2", len(got))
	}
	if got := s.GetTagNames(); len(got) != 2 {
		t.Errorf("GetTagNames len = %d, want 2", len(got))
	}
	if got := s.TagCount(); got != 2 {
		t.Errorf("TagCount = %d, want 2", got)
	}
}

package store

import "testing"

func doc(id string, cats ...string) Document {
	return Document{ID: id, Content: "content of " + id, Categories: cats}
}

func TestPutGetDeleteDocument(t *testing.T) {
	s := NewMemoryStore()
	s.PutDocument(doc("d1"))
	if _, ok := s.GetDocument("d1"); !ok {
		t.Fatal("d1 not found after put")
	}
	if s.DocCount() != 1 {
		t.Fatalf("doc count = %d, want 1", s.DocCount())
	}

	// Re-putting the same ID replaces, not duplicates.
	s.PutDocument(Document{ID: "d1", Content: "changed"})
	if got, _ := s.GetDocument("d1"); got.Content != "changed" {
		t.Fatalf("content = %q, want changed", got.Content)
	}
	if s.DocCount() != 1 {
		t.Fatalf("doc count after replace = %d, want 1", s.DocCount())
	}

	if !s.DeleteDocument("d1") {
		t.Fatal("delete returned false")
	}
	if s.DeleteDocument("d1") {
		t.Fatal("second delete should return false")
	}
	if s.DocCount() != 0 {
		t.Fatalf("doc count after delete = %d, want 0", s.DocCount())
	}
}

func TestListDocumentsPagination(t *testing.T) {
	s := NewMemoryStore()
	for _, id := range []string{"d1", "d2", "d3", "d4", "d5"} {
		s.PutDocument(doc(id))
	}
	page1, total := s.ListDocuments(1, 2)
	if total != 5 || len(page1) != 2 {
		t.Fatalf("page1 len=%d total=%d, want 2/5", len(page1), total)
	}
	// Insertion order is preserved.
	if page1[0].ID != "d1" || page1[1].ID != "d2" {
		t.Fatalf("page1 = %v, want d1,d2", []string{page1[0].ID, page1[1].ID})
	}
	page3, _ := s.ListDocuments(3, 2)
	if len(page3) != 1 {
		t.Fatalf("page3 len = %d, want 1", len(page3))
	}
	if pageX, total := s.ListDocuments(99, 2); pageX != nil || total != 5 {
		t.Fatalf("out-of-range page = %v total %d", pageX, total)
	}
}

func TestPutGetDeleteCategory(t *testing.T) {
	s := NewMemoryStore()
	s.PutCategory(Category{Name: "category1", Centroid: []float64{1, 0}, Norm: 1, Count: 1})
	c, ok := s.GetCategory("category1")
	if !ok || c.Count != 1 || len(c.Centroid) != 2 {
		t.Fatalf("category1 = %+v ok=%v", c, ok)
	}
	if s.CategoryCount() != 1 {
		t.Fatalf("category count = %d, want 1", s.CategoryCount())
	}

	s.AddDocToCategory("d1", "category1")
	s.DeleteCategory("category1")
	if _, ok := s.GetCategory("category1"); ok {
		t.Fatal("category1 still present after delete")
	}
	// DeleteCategory also clears the index.
	if s.DocCountByCategory("category1") != 0 {
		t.Fatalf("index not cleared on category delete: %d", s.DocCountByCategory("category1"))
	}
}

func TestListCategoriesSorted(t *testing.T) {
	s := NewMemoryStore()
	for _, n := range []string{"category3", "category1", "category2"} {
		s.PutCategory(Category{Name: n})
	}
	cats := s.ListCategories()
	if len(cats) != 3 || cats[0].Name != "category1" || cats[2].Name != "category3" {
		t.Fatalf("ListCategories not sorted: %v", cats)
	}
}

func TestCategoryIndex(t *testing.T) {
	s := NewMemoryStore()
	s.PutDocument(doc("d1"))
	s.PutDocument(doc("d2"))
	s.AddDocToCategory("d1", "category1")
	s.AddDocToCategory("d2", "category1")
	s.AddDocToCategory("d1", "category1") // idempotent

	if got := s.DocCountByCategory("category1"); got != 2 {
		t.Fatalf("doc count by category = %d, want 2", got)
	}
	docs := s.DocsInCategory("category1")
	if len(docs) != 2 {
		t.Fatalf("DocsInCategory len = %d, want 2", len(docs))
	}

	// DeleteDocument does NOT touch the index, so the membership lingers but
	// DocsInCategory skips the now-missing document.
	s.DeleteDocument("d1")
	if got := len(s.DocsInCategory("category1")); got != 1 {
		t.Fatalf("DocsInCategory after doc delete = %d, want 1", got)
	}
	if got := s.DocCountByCategory("category1"); got != 2 {
		t.Fatalf("index unchanged by DeleteDocument = %d, want 2", got)
	}

	// Explicit membership removal is what shrinks the index.
	s.RemoveDocFromCategory("d1", "category1")
	s.RemoveDocFromCategory("d2", "category1")
	if got := s.DocCountByCategory("category1"); got != 0 {
		t.Fatalf("doc count after remove = %d, want 0", got)
	}
}

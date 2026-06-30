package store

import "time"

// Document is a stored item plus its embedding and the categories it was
// clustered into. Vector/Norm are runtime-only (not serialized in API responses).
type Document struct {
	ID         string    `json:"id"`
	Content    string    `json:"content"`
	Categories []string  `json:"categories"`
	Vector     []float64 `json:"-"`
	Norm       float64   `json:"-"` // cached L2 norm of Vector
	CreatedAt  time.Time `json:"created_at"`
}

// Category is an incrementally-discovered cluster. Centroid holds the running
// SUM of member document vectors (cosine is scale-invariant, so the sum behaves
// like the mean and makes member removal exact); Norm caches its L2 norm; Count
// is the number of member documents. Categories are auto-named ("category1", …).
//
// The store treats Category as opaque data — it never computes or interprets the
// centroid. Maintaining it (the clustering math) lives in the pipeline layer.
type Category struct {
	Name      string    `json:"name"`
	Centroid  []float64 `json:"-"`
	Norm      float64   `json:"-"`
	Count     int       `json:"count"`
	CreatedAt time.Time `json:"created_at"`
}

// SearchResult is a ranked document with its similarity score.
type SearchResult struct {
	Document   Document `json:"document"`
	Score      float64  `json:"score"`
	Categories []string `json:"categories"`
}

// CategoryMatch is a category the query matched, with its similarity.
type CategoryMatch struct {
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

// Store is a pure persistence layer: it stores and retrieves documents,
// categories, and the category→documents index. It holds NO business logic —
// no embeddings, similarity, centroid math, thresholds, or category naming.
// Clustering lives in the pipeline layer, query ranking in the search layer;
// both drive the store through these primitives.
type Store interface {
	// Documents.
	PutDocument(doc Document) // insert or replace by ID
	GetDocument(id string) (Document, bool)
	DeleteDocument(id string) bool // removes the document only (no index cleanup)
	ListDocuments(page, size int) ([]Document, int)
	DocCount() int

	// Categories.
	PutCategory(cat Category) // insert or replace by Name
	GetCategory(name string) (Category, bool)
	DeleteCategory(name string) // drops the category and its index entry
	ListCategories() []Category
	CategoryCount() int

	// Category → documents index.
	AddDocToCategory(id, name string)
	RemoveDocFromCategory(id, name string)
	DocsInCategory(name string) []Document
	DocCountByCategory(name string) int
}

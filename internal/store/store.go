package store

import "time"

type Document struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Tags      []string  `json:"tags"`
	CreatedAt time.Time `json:"created_at"`
}

type Tag struct {
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Vector      []float64 `json:"-"`
	Norm        float64   `json:"-"` // cached L2 norm of Vector, set when Vector is assigned
	Examples    []string  `json:"-"`
	CreatedAt   time.Time `json:"created_at"`
}

type SearchResult struct {
	Document    Document `json:"document"`
	Score       float64  `json:"score"`
	MatchedTags []string `json:"matched_tags"`
}

type Store interface {
	SaveDocument(doc Document) error
	GetDocument(id string) (Document, bool)
	DeleteDocument(id string) (Document, bool)
	ListDocuments(page, size int) ([]Document, int)

	SaveTag(tag Tag) error
	GetTag(name string) (Tag, bool)
	UpdateTag(tag Tag) error
	ListTags() []Tag
	GetTagNames() []string

	GetDocumentsByTag(tag string) []Document
	GetDocIDsByTag(tag string) []string
	DocCountByTag(tag string) int

	DocCount() int
	TagCount() int
}

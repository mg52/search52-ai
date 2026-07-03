package engine

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
)

const benchDims = 768

// benchEmbedder returns a deterministic pseudo-random unit-scale vector per
// text, mimicking a real embedding model's output shape (768 dims).
type benchEmbedder struct{}

func (benchEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	h := int64(0)
	for _, r := range text {
		h = h*31 + int64(r)
	}
	rng := rand.New(rand.NewSource(h))
	v := make([]float32, benchDims)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v, nil
}

func benchEngine(b *testing.B, nDocs int) *SearchEngine {
	b.Helper()
	se := New("bench", benchEmbedder{}, Config{
		CategoryThreshold:   0.6,
		MaxCategoriesPerDoc: 3,
		MaxCategories:       20,
		TopNCategories:      5,
	})
	ctx := context.Background()
	for i := 0; i < nDocs; i++ {
		if _, err := se.Process(ctx, fmt.Sprintf("doc%d", i), fmt.Sprintf("content body %d", i)); err != nil {
			b.Fatalf("process: %v", err)
		}
	}
	return se
}

func BenchmarkSearch5000(b *testing.B) {
	se := benchEngine(b, 5000)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := se.Search(ctx, "some interesting query text", 10); err != nil {
			b.Fatalf("search: %v", err)
		}
	}
}

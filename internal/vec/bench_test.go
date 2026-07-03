package vec

import (
	"math/rand"
	"testing"
)

func randVec(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func BenchmarkDot768(b *testing.B) {
	a := randVec(768, 1)
	c := randVec(768, 2)
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		sink += Dot(a, c)
	}
	_ = sink
}

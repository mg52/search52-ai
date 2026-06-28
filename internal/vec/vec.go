package vec

import "math"

// Norm returns the Euclidean (L2) norm of v.
func Norm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// Dot returns the dot product of a and b. Returns 0 if their lengths differ.
func Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var d float64
	for i := range a {
		d += a[i] * b[i]
	}
	return d
}

// Cosine returns the cosine similarity between a and b using their precomputed
// norms. Callers cache normA/normB to avoid recomputing them per comparison.
func Cosine(a, b []float64, normA, normB float64) float64 {
	if normA == 0 || normB == 0 {
		return 0
	}
	return Dot(a, b) / (normA * normB)
}

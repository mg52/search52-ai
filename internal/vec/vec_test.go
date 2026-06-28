package vec

import (
	"math"
	"testing"
)

const eps = 1e-9

func approx(a, b float64) bool { return math.Abs(a-b) < eps }

func TestNorm(t *testing.T) {
	cases := []struct {
		in   []float64
		want float64
	}{
		{[]float64{3, 4}, 5},
		{[]float64{0, 0, 0}, 0},
		{nil, 0},
		{[]float64{}, 0},
		{[]float64{-3, -4}, 5},
		{[]float64{1}, 1},
	}
	for _, c := range cases {
		if got := Norm(c.in); !approx(got, c.want) {
			t.Errorf("Norm(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestDot(t *testing.T) {
	if got := Dot([]float64{1, 2, 3}, []float64{4, 5, 6}); !approx(got, 32) {
		t.Errorf("Dot = %v, want 32", got)
	}
	// Mismatched lengths return 0.
	if got := Dot([]float64{1, 2}, []float64{1, 2, 3}); got != 0 {
		t.Errorf("Dot mismatch = %v, want 0", got)
	}
	// Empty vectors return 0.
	if got := Dot(nil, nil); got != 0 {
		t.Errorf("Dot(nil,nil) = %v, want 0", got)
	}
}

func TestCosine(t *testing.T) {
	cases := []struct {
		name         string
		a, b         []float64
		normA, normB float64
		want         float64
	}{
		{"identical", []float64{1, 0}, []float64{1, 0}, 1, 1, 1},
		{"orthogonal", []float64{1, 0}, []float64{0, 1}, 1, 1, 0},
		{"opposite", []float64{1, 0}, []float64{-1, 0}, 1, 1, -1},
		{"zero normA", []float64{0, 0}, []float64{1, 0}, 0, 1, 0},
		{"zero normB", []float64{1, 0}, []float64{0, 0}, 1, 0, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Cosine(c.a, c.b, c.normA, c.normB); !approx(got, c.want) {
				t.Errorf("Cosine = %v, want %v", got, c.want)
			}
		})
	}
}

func TestCosineMatchesManualComputation(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{2, 1, 0}
	want := Dot(a, b) / (Norm(a) * Norm(b))
	if got := Cosine(a, b, Norm(a), Norm(b)); !approx(got, want) {
		t.Errorf("Cosine = %v, want %v", got, want)
	}
}

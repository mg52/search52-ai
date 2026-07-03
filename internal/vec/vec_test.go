package vec

import (
	"math"
	"testing"
)

// float32 carries ~7 significant digits.
const eps = 1e-6

func approx(a, b float32) bool { return math.Abs(float64(a-b)) < eps }

func TestNorm(t *testing.T) {
	cases := []struct {
		in   []float32
		want float32
	}{
		{[]float32{3, 4}, 5},
		{[]float32{0, 0, 0}, 0},
		{nil, 0},
		{[]float32{}, 0},
		{[]float32{-3, -4}, 5},
		{[]float32{1}, 1},
	}
	for _, c := range cases {
		if got := Norm(c.in); !approx(got, c.want) {
			t.Errorf("Norm(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestDot(t *testing.T) {
	if got := Dot([]float32{1, 2, 3}, []float32{4, 5, 6}); !approx(got, 32) {
		t.Errorf("Dot = %v, want 32", got)
	}
	// Mismatched lengths return 0.
	if got := Dot([]float32{1, 2}, []float32{1, 2, 3}); got != 0 {
		t.Errorf("Dot mismatch = %v, want 0", got)
	}
	// Empty vectors return 0.
	if got := Dot(nil, nil); got != 0 {
		t.Errorf("Dot(nil,nil) = %v, want 0", got)
	}
	// Tail elements past the unrolled block are included (len%4 != 0).
	if got := Dot([]float32{1, 1, 1, 1, 1, 2}, []float32{1, 1, 1, 1, 1, 3}); !approx(got, 11) {
		t.Errorf("Dot tail = %v, want 11", got)
	}
}

// TestDotUnrolledBlockAndRemainder exercises the 8-wide unrolled main loop plus
// the scalar remainder loop with a length that is not a multiple of 8 (19 = 2*8
// + 3). The dot of two all-ones vectors of length n is exactly n.
func TestDotUnrolledBlockAndRemainder(t *testing.T) {
	for _, n := range []int{8, 16, 19, 100} {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := range a {
			a[i], b[i] = 1, 1
		}
		if got := Dot(a, b); !approx(got, float32(n)) {
			t.Errorf("Dot(ones[%d]) = %v, want %d", n, got, n)
		}
	}
	// Norm of a length-16 all-twos vector is sqrt(16*4) = 8.
	sixteenTwos := make([]float32, 16)
	for i := range sixteenTwos {
		sixteenTwos[i] = 2
	}
	if got := Norm(sixteenTwos); !approx(got, 8) {
		t.Errorf("Norm(twos[16]) = %v, want 8", got)
	}
}

func TestCosine(t *testing.T) {
	cases := []struct {
		name         string
		a, b         []float32
		normA, normB float32
		want         float32
	}{
		{"identical", []float32{1, 0}, []float32{1, 0}, 1, 1, 1},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 1, 1, 0},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, 1, 1, -1},
		{"zero normA", []float32{0, 0}, []float32{1, 0}, 0, 1, 0},
		{"zero normB", []float32{1, 0}, []float32{0, 0}, 1, 0, 0},
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
	a := []float32{1, 2, 3}
	b := []float32{2, 1, 0}
	want := Dot(a, b) / (Norm(a) * Norm(b))
	if got := Cosine(a, b, Norm(a), Norm(b)); !approx(got, want) {
		t.Errorf("Cosine = %v, want %v", got, want)
	}
}

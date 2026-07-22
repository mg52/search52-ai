package common

import (
	"math/rand"
	"sort"
	"testing"
)

// verifyHeapInvariant checks the min-heap property: every parent's score is
// <= both of its children's scores.
func verifyHeapInvariant(t *testing.T, h []InternalHit) {
	t.Helper()
	for i := range h {
		left := 2*i + 1
		right := 2*i + 2
		if left < len(h) && h[i].Score > h[left].Score {
			t.Fatalf("heap invariant broken at %d: parent %d > left child %d", i, h[i].Score, h[left].Score)
		}
		if right < len(h) && h[i].Score > h[right].Score {
			t.Fatalf("heap invariant broken at %d: parent %d > right child %d", i, h[i].Score, h[right].Score)
		}
	}
}

func TestHeapPushHitMaintainsInvariant(t *testing.T) {
	var h []InternalHit
	scores := []int{5, 3, 8, 1, 9, 2, 7, 0, 6, 4}
	for i, s := range scores {
		h = HeapPushHit(h, InternalHit{ID: uint32(i), Score: s})
		verifyHeapInvariant(t, h)
	}
	if len(h) != len(scores) {
		t.Fatalf("len = %d, want %d", len(h), len(scores))
	}
	// The root must be the minimum of everything pushed.
	if h[0].Score != 0 {
		t.Fatalf("root score = %d, want 0 (the minimum)", h[0].Score)
	}
}

func TestHeapPushSingle(t *testing.T) {
	h := HeapPushHit(nil, InternalHit{ID: 42, Score: 7})
	if len(h) != 1 || h[0].ID != 42 || h[0].Score != 7 {
		t.Fatalf("single push = %+v", h)
	}
}

func TestHeapReplaceTopMaintainsInvariant(t *testing.T) {
	var h []InternalHit
	for i, s := range []int{5, 3, 8, 1, 9} {
		h = HeapPushHit(h, InternalHit{ID: uint32(i), Score: s})
	}
	verifyHeapInvariant(t, h)

	// Replacing the root (score 1) with something larger must re-establish the
	// invariant and the new root must be the new minimum.
	HeapReplaceTop(h, InternalHit{ID: 99, Score: 6})
	verifyHeapInvariant(t, h)
	want := 3 // remaining scores: 5,3,8,9,6 -> min is 3
	if h[0].Score != want {
		t.Fatalf("root after replace = %d, want %d", h[0].Score, want)
	}

	// Confirm the replaced value is actually present somewhere in the heap.
	found := false
	for _, hit := range h {
		if hit.ID == 99 && hit.Score == 6 {
			found = true
		}
	}
	if !found {
		t.Fatalf("replaced hit not found in heap: %+v", h)
	}
}

func TestHeapReplaceTopSingleElement(t *testing.T) {
	h := HeapPushHit(nil, InternalHit{ID: 1, Score: 5})
	HeapReplaceTop(h, InternalHit{ID: 2, Score: 9})
	if len(h) != 1 || h[0].ID != 2 || h[0].Score != 9 {
		t.Fatalf("single-element replace = %+v", h)
	}
}

// TestHeapDrainProducesSortedOrder replicates the exact drain pattern used by
// Search (pop root, move last element to root, sift down) and checks the
// result is fully descending — this is the property the search hot path
// actually depends on.
func TestHeapDrainProducesSortedOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const n = 200
	scores := make([]int, n)
	for i := range scores {
		scores[i] = rng.Intn(1000)
	}

	var h []InternalHit
	for i, s := range scores {
		h = HeapPushHit(h, InternalHit{ID: uint32(i), Score: s})
		verifyHeapInvariant(t, h)
	}

	drained := make([]int, len(h))
	for i := len(h) - 1; i >= 0; i-- {
		hit := h[0]
		if i > 0 {
			h[0] = h[i]
			SiftDownHit(h, 0, i)
		}
		drained[i] = hit.Score
	}

	want := append([]int(nil), scores...)
	sort.Sort(sort.Reverse(sort.IntSlice(want)))
	for i := range drained {
		if drained[i] != want[i] {
			t.Fatalf("drained[%d] = %d, want %d (full: got=%v want=%v)", i, drained[i], want[i], drained, want)
		}
	}
}

// TestHeapBoundedTopK replicates the fixed-capacity top-K pattern used by
// Search: push until the heap reaches capacity k, then only replace the root
// when a new score beats it. The final heap must hold exactly the k largest
// scores seen.
func TestHeapBoundedTopK(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const n = 500
	const k = 10
	scores := make([]int, n)
	for i := range scores {
		scores[i] = rng.Intn(10000)
	}

	var h []InternalHit
	for i, s := range scores {
		if len(h) < k {
			h = HeapPushHit(h, InternalHit{ID: uint32(i), Score: s})
		} else if h[0].Score < s {
			HeapReplaceTop(h, InternalHit{ID: uint32(i), Score: s})
		}
		verifyHeapInvariant(t, h)
	}

	got := make([]int, len(h))
	for i, hit := range h {
		got[i] = hit.Score
	}
	sort.Ints(got)

	want := append([]int(nil), scores...)
	sort.Sort(sort.Reverse(sort.IntSlice(want)))
	want = want[:k]
	sort.Ints(want)

	if len(got) != len(want) {
		t.Fatalf("heap size = %d, want %d", len(got), k)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("bounded top-%d mismatch: got=%v want=%v", k, got, want)
		}
	}
}

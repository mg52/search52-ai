// Package common holds the types and helpers shared by every engine
// implementation (embedding-based clustering, LLM-based categorization):
// the API-facing Document/Results shapes and a specialized top-N min-heap.
package common

type InternalHit struct {
	ID    uint32
	Score int
}

// Specialized min-heap over []InternalHit. Inlined sift operations avoid the
// interface dispatch and any-boxing overhead of container/heap, which matters
// in the per-document inner loops of search.

// HeapPushHit appends hit and sifts it up. Returns the new slice header.
func HeapPushHit(h []InternalHit, hit InternalHit) []InternalHit {
	h = append(h, hit)
	i := len(h) - 1
	for i > 0 {
		parent := (i - 1) >> 1
		if h[parent].Score <= h[i].Score {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
	return h
}

// HeapReplaceTop overwrites the root and sifts it down: one operation instead
// of pop+push when the heap is already at capacity. len(h) > 0.
func HeapReplaceTop(h []InternalHit, hit InternalHit) {
	h[0] = hit
	SiftDownHit(h, 0, len(h))
}

func SiftDownHit(h []InternalHit, start, n int) {
	i := start
	for {
		left := 2*i + 1
		if left >= n {
			return
		}
		smallest := left
		if right := left + 1; right < n && h[right].Score < h[left].Score {
			smallest = right
		}
		if h[i].Score <= h[smallest].Score {
			return
		}
		h[i], h[smallest] = h[smallest], h[i]
		i = smallest
	}
}

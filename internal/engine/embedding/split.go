// Category splitting. When a category's ShouldSplit flag fires, Process
// launches splitCategory in its own goroutine. The split touches se.mu only
// twice, briefly: once (RLock) to snapshot the category's current members,
// once (Lock) to fold the clustering result back into live state. The
// clustering itself — the only potentially slow part — runs with no lock
// held, so search, ingestion, and removal (including on the very category
// being split) keep working the whole time. Whatever changed on the category
// between the snapshot and the merge — docs added, removed, or reassigned
// elsewhere by a concurrent Process/Remove — is reconciled against live state
// at merge time, never against the stale snapshot.
package embedding

import "github.com/mg52/search52-ai/internal/vec"

// kmeans2 splits vecs (each with precomputed norm) into two spherical
// clusters by cosine similarity and returns each vector's cluster label (0 or
// 1). Seeds are chosen deterministically — the vector farthest from the
// overall mean, then the vector farthest from that seed — so results don't
// depend on map/slice iteration order or randomness. Empty clusters are
// reseeded every iteration from the farthest member of the surviving
// cluster, so a converged result always has two non-empty clusters. ok is
// false when there are fewer than two vectors to seed distinct clusters.
func kmeans2(vecs [][]float32, norms []float32) (labels []int, ok bool) {
	n := len(vecs)
	if n < 2 {
		return nil, false
	}
	dim := len(vecs[0])

	mean := make([]float32, dim)
	for _, v := range vecs {
		for i, x := range v {
			mean[i] += x
		}
	}
	meanNorm := vec.Norm(mean)

	seedA := 0
	if meanNorm > 0 {
		worst := float32(2) // cosine similarity is always <= 1
		for i, v := range vecs {
			if sim := vec.Cosine(v, mean, norms[i], meanNorm); sim < worst {
				worst, seedA = sim, i
			}
		}
	}
	seedB := -1
	{
		worst := float32(2)
		for i, v := range vecs {
			if i == seedA {
				continue
			}
			if sim := vec.Cosine(v, vecs[seedA], norms[i], norms[seedA]); sim < worst {
				worst, seedB = sim, i
			}
		}
	}
	if seedB < 0 {
		return nil, false // only one distinct vector to seed from
	}

	var centroids [2][]float32
	var centroidNorms [2]float32
	centroids[0] = append([]float32(nil), vecs[seedA]...)
	centroids[1] = append([]float32(nil), vecs[seedB]...)
	centroidNorms[0], centroidNorms[1] = norms[seedA], norms[seedB]

	labels = make([]int, n)
	const maxIters = 25
	for iter := 0; iter < maxIters; iter++ {
		changed := false
		var sums [2][]float32
		sums[0] = make([]float32, dim)
		sums[1] = make([]float32, dim)
		var counts [2]int

		for i, v := range vecs {
			lbl := 0
			if vec.Cosine(v, centroids[1], norms[i], centroidNorms[1]) > vec.Cosine(v, centroids[0], norms[i], centroidNorms[0]) {
				lbl = 1
			}
			if iter == 0 || labels[i] != lbl {
				changed = true
			}
			labels[i] = lbl
			counts[lbl]++
			for d, x := range v {
				sums[lbl][d] += x
			}
		}

		for _, lbl := range [2]int{0, 1} {
			if counts[lbl] > 0 {
				continue
			}
			other := 1 - lbl
			farthest, worst := -1, float32(2)
			for i, v := range vecs {
				if labels[i] != other {
					continue
				}
				if sim := vec.Cosine(v, centroids[other], norms[i], centroidNorms[other]); sim < worst {
					worst, farthest = sim, i
				}
			}
			if farthest < 0 {
				continue
			}
			labels[farthest] = lbl
			counts[other]--
			counts[lbl]++
			for d, x := range vecs[farthest] {
				sums[other][d] -= x
				sums[lbl][d] += x
			}
			changed = true
		}

		centroids[0], centroids[1] = sums[0], sums[1]
		centroidNorms[0], centroidNorms[1] = vec.Norm(sums[0]), vec.Norm(sums[1])

		if !changed {
			break
		}
	}
	return labels, true
}

// migration records that document id was folded into new category dest
// during a split, so its similarity to dest's final centroid can be folded
// into dest's Welford variance once every member has been migrated.
type migration struct {
	id   string
	dest string
}

// splitCategory reclusters category name into two fresh categories. It is
// meant to be run via `go se.splitCategory(name)`.
func (se *SearchEngine) splitCategory(name string) {
	defer func() {
		se.mu.Lock()
		delete(se.splitting, name)
		se.mu.Unlock()
	}()

	se.mu.RLock()
	if len(se.categories) >= se.maxCategories {
		// A split is a net +1 category, so mergeSplitLocked would refuse it
		// anyway; bail before burning a full k-means run on it. The category
		// keeps its ShouldSplit flag and is retried (cheaply, through here)
		// on future ingests, so it still splits if the cap frees up. The
		// merge-time check remains authoritative — this is only an early out.
		se.mu.RUnlock()
		return
	}
	memberSet := se.catDocs[name]
	ids := make([]string, 0, len(memberSet))
	vecs := make([][]float32, 0, len(memberSet))
	norms := make([]float32, 0, len(memberSet))
	for id := range memberSet {
		d, ok := se.docs[id]
		if !ok || d.ForcedFallback {
			// ForcedFallback docs never touched this category's centroid (see
			// addToCentroidLocked), so they carry no real signal about the
			// category's shape; they're placed post-hoc below, same as docs
			// that join mid-split.
			continue
		}
		ids = append(ids, id)
		vecs = append(vecs, d.Vector)
		norms = append(norms, d.Norm)
	}
	se.mu.RUnlock()

	labels, ok := kmeans2(vecs, norms)
	if !ok {
		return // too few/degenerate real members to split meaningfully
	}

	// Snapshot-derived centroids, used only to place documents that aren't in
	// the snapshot (ForcedFallback members, or docs that joined `name` after
	// the snapshot was taken) into whichever side they're nearest to.
	dim := len(vecs[0])
	seedCentroid := [2][]float32{make([]float32, dim), make([]float32, dim)}
	for i, lbl := range labels {
		for d, x := range vecs[i] {
			seedCentroid[lbl][d] += x
		}
	}
	seedNorm := [2]float32{vec.Norm(seedCentroid[0]), vec.Norm(seedCentroid[1])}
	labelOf := make(map[string]int, len(ids))
	for i, id := range ids {
		labelOf[id] = labels[i]
	}

	se.mu.Lock()
	retrigger, ok := se.mergeSplitLocked(name, labelOf, seedCentroid, seedNorm)
	se.mu.Unlock()
	if !ok {
		return
	}

	for _, n := range retrigger {
		go se.splitCategory(n)
	}
}

// mergeSplitLocked folds a k-means split of category name back into live
// state: name is replaced by two fresh categories. labelOf maps the ids that
// were present at snapshot time to their cluster (0 or 1); seedCentroid/
// seedNorm are the snapshot-derived centroids of those two clusters, used to
// place any *other* current member of name — one added after the snapshot
// was taken, or excluded from it as ForcedFallback — into whichever side
// it's nearest to. Crucially, every document migrated is read fresh from
// se.catDocs[name]/se.docs, not from labelOf's keys, so members added to or
// removed from name by a concurrent Process/Remove between the snapshot and
// this call are reconciled against live state rather than the stale
// snapshot. Returns the names of any freshly split-off category whose own
// ShouldSplit immediately fired, for the caller to relaunch a split on once
// it has released the lock; ok is false if there was nothing to merge (the
// category was pruned to empty in the meantime, or the maxCategories cap has
// no room for the net +1 category a split adds). Caller must hold the write
// lock, which is not released here.
func (se *SearchEngine) mergeSplitLocked(name string, labelOf map[string]int, seedCentroid [2][]float32, seedNorm [2]float32) (retrigger []string, ok bool) {
	curSet, exists := se.catDocs[name]
	if !exists || len(curSet) == 0 || len(se.categories) >= se.maxCategories {
		return nil, false
	}

	nameA := se.newCategoryLocked()
	nameB := se.newCategoryLocked()
	newNames := [2]string{nameA, nameB}

	memberIDs := make([]string, 0, len(curSet))
	for id := range curSet {
		memberIDs = append(memberIDs, id)
	}

	migrations := make([]migration, 0, len(memberIDs))
	for _, id := range memberIDs {
		doc, ok := se.docs[id]
		if !ok {
			continue
		}
		lbl, known := labelOf[id]
		if !known {
			simA := vec.Cosine(doc.Vector, seedCentroid[0], doc.Norm, seedNorm[0])
			simB := vec.Cosine(doc.Vector, seedCentroid[1], doc.Norm, seedNorm[1])
			lbl = 0
			if simB > simA {
				lbl = 1
			}
		}
		dest := newNames[lbl]

		se.addToCentroidLocked(dest, doc.Vector, !doc.ForcedFallback)
		se.addDocToCategoryLocked(id, dest)

		newCats := make([]string, len(doc.Categories))
		copy(newCats, doc.Categories)
		for i, cn := range newCats {
			if cn == name {
				newCats[i] = dest
				break
			}
		}
		doc.Categories = newCats
		se.docs[id] = doc
		migrations = append(migrations, migration{id: id, dest: dest})
	}

	delete(se.categories, name)
	delete(se.catDocs, name)

	// Reconciliation against live membership (rather than the snapshot) can
	// leave one side empty — e.g. every doc that would have landed there was
	// concurrently removed. Nothing will ever prune an empty category on its
	// own (that only happens on an explicit member removal), so do it here
	// instead of leaking a permanent, memberless category.
	for _, n := range newNames {
		if c, ok := se.categories[n]; ok && c.Count == 0 {
			delete(se.categories, n)
			delete(se.catDocs, n)
		}
	}

	// Re-seed each new category's Welford variance stats from its migrated
	// members' similarity to the category's *final* centroid. Without this,
	// both new categories would start at zero variance regardless of their
	// actual (inherited) spread, and ShouldSplit couldn't fire again until
	// VarianceMinCount more brand-new documents happened to join.
	for _, m := range migrations {
		doc := se.docs[m.id]
		if c, ok := se.categories[m.dest]; ok {
			sim := vec.Cosine(doc.Vector, c.Centroid, doc.Norm, c.Norm)
			se.updateVarianceLocked(m.dest, sim)
		}
	}

	// A migrated half can itself already be heterogeneous enough to need
	// another split; catch that immediately rather than waiting for
	// unrelated future traffic to land in it.
	for _, n := range newNames {
		if c, ok := se.categories[n]; ok && c.ShouldSplit && !se.splitting[n] {
			se.splitting[n] = true
			retrigger = append(retrigger, n)
		}
	}
	return retrigger, true
}

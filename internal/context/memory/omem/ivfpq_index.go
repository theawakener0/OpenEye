package omem

import (
	"context"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type pqCodebook struct {
	Centroids [][][]float32
}

type ivfListEntry struct {
	FactID int64
	Codes  []uint16
	Norm   float32
}

type ivfPQPersisted struct {
	Version        int
	Dimension      int
	Normalized     bool
	Centroids      [][]float32
	Lists          [][]ivfListEntry
	Vectors        map[int64][]float32
	Assignments    map[int64]int
	ResidualNorms  map[int64]float32
	PQ             pqCodebook
	PQSubvectors   int
	PQBits         int
	BuiltAt        time.Time
	EmbeddingModel string
}

type ivfPQIndex struct {
	mu         sync.RWMutex
	config     ANNConfig
	dimension  int
	normalized bool
	indexPath  string

	centroids     [][]float32
	lists         [][]ivfListEntry
	vectors       map[int64][]float32
	assignments   map[int64]int
	residualNorms map[int64]float32
	pq            pqCodebook
	builtAt       time.Time
	stats         ivfPQStats
}

type ivfPQStats struct {
	Searches      int64
	Fallbacks     int64
	Rebuilds      int64
	Upserts       int64
	Deletes       int64
	LastBuildTime time.Time
	FactCount     int
	CentroidCount int
	Dimension     int
}

func newIVFPQIndex(cfg ANNConfig, dimension int, normalized bool, embeddingModel string) (*ivfPQIndex, error) {
	idx := &ivfPQIndex{
		config:        cfg,
		dimension:     dimension,
		normalized:    normalized,
		indexPath:     cfg.IndexPath,
		vectors:       make(map[int64][]float32),
		assignments:   make(map[int64]int),
		residualNorms: make(map[int64]float32),
		stats: ivfPQStats{
			Dimension: dimension,
		},
	}
	if cfg.IndexPath != "" {
		if err := idx.load(embeddingModel); err != nil && !os.IsNotExist(err) {
			return nil, err
		}
	}
	return idx, nil
}

func (idx *ivfPQIndex) Upsert(ctx context.Context, factID int64, embedding []float32) error {
	_ = ctx
	if len(embedding) == 0 {
		return nil
	}
	vec := prepareANNVector(embedding, idx.normalized)

	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.vectors[factID] = vec
	idx.stats.Upserts++
	idx.stats.FactCount = len(idx.vectors)
	if len(idx.vectors) < idx.config.TrainMinFacts {
		return idx.persistLocked("")
	}
	idx.rebuildStructuresLocked()
	return idx.persistLocked("")
}

func (idx *ivfPQIndex) Delete(ctx context.Context, factID int64) error {
	_ = ctx

	idx.mu.Lock()
	defer idx.mu.Unlock()

	delete(idx.vectors, factID)
	delete(idx.assignments, factID)
	delete(idx.residualNorms, factID)
	idx.stats.Deletes++
	idx.stats.FactCount = len(idx.vectors)
	if len(idx.vectors) >= idx.config.TrainMinFacts {
		idx.rebuildStructuresLocked()
	}
	return idx.persistLocked("")
}

func (idx *ivfPQIndex) Search(ctx context.Context, query []float32, k int, oversample int, exactRerankLimit int) ([]ANNCandidate, error) {
	_ = ctx

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(query) == 0 || len(idx.centroids) == 0 || len(idx.lists) == 0 {
		idx.stats.Fallbacks++
		return nil, nil
	}
	if idx.config.MinFactsToEnable > 0 && len(idx.vectors) < idx.config.MinFactsToEnable {
		idx.stats.Fallbacks++
		return nil, nil
	}
	if k <= 0 {
		k = 10
	}
	if oversample < k {
		oversample = k
	}
	if exactRerankLimit > 0 && oversample > exactRerankLimit {
		oversample = exactRerankLimit
	}

	queryVec := prepareANNVector(query, idx.normalized)
	centroidOrder := idx.nearestCentroids(queryVec)
	nprobe := idx.config.NProbe
	if nprobe <= 0 || nprobe > len(centroidOrder) {
		nprobe = len(centroidOrder)
	}

	type scored struct {
		id    int64
		score float64
	}
	buffer := make([]scored, 0, oversample*2)
	for _, clusterID := range centroidOrder[:nprobe] {
		lookup := idx.buildADCLookup(queryVec, idx.centroids[clusterID])
		for _, entry := range idx.lists[clusterID] {
			score := idx.scorePQEntry(entry, lookup)
			buffer = append(buffer, scored{id: entry.FactID, score: score})
		}
	}
	if len(buffer) == 0 {
		idx.stats.Fallbacks++
		return nil, nil
	}

	sort.Slice(buffer, func(i, j int) bool {
		return buffer[i].score > buffer[j].score
	})
	if len(buffer) > oversample {
		buffer = buffer[:oversample]
	}

	results := make([]ANNCandidate, len(buffer))
	for i, item := range buffer {
		results[i] = ANNCandidate{FactID: item.id, Score: item.score}
	}
	idx.stats.Searches++
	return results, nil
}

func (idx *ivfPQIndex) Rebuild(ctx context.Context, facts []Fact) error {
	_ = ctx

	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.vectors = make(map[int64][]float32, len(facts))
	for _, fact := range facts {
		if fact.IsObsolete || len(fact.Embedding) == 0 {
			continue
		}
		vec := prepareANNVector(fact.Embedding, idx.normalized)
		if idx.dimension == 0 {
			idx.dimension = len(vec)
			idx.stats.Dimension = idx.dimension
		}
		if len(vec) != idx.dimension {
			continue
		}
		idx.vectors[fact.ID] = vec
	}
	if len(idx.vectors) < idx.config.TrainMinFacts {
		idx.stats.FactCount = len(idx.vectors)
		return idx.persistLocked("")
	}
	idx.rebuildStructuresLocked()
	return idx.persistLocked("")
}

func (idx *ivfPQIndex) Stats(ctx context.Context) (map[string]interface{}, error) {
	_ = ctx

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	return map[string]interface{}{
		"backend":             idx.config.Backend,
		"fact_count":          len(idx.vectors),
		"centroid_count":      len(idx.centroids),
		"dimension":           idx.dimension,
		"normalized":          idx.normalized,
		"searches":            idx.stats.Searches,
		"fallbacks":           idx.stats.Fallbacks,
		"rebuilds":            idx.stats.Rebuilds,
		"upserts":             idx.stats.Upserts,
		"deletes":             idx.stats.Deletes,
		"last_build_time":     idx.stats.LastBuildTime,
		"min_facts_to_enable": idx.config.MinFactsToEnable,
		"nlist":               idx.config.NList,
		"nprobe":              idx.config.NProbe,
		"pq_subvectors":       idx.config.PQSubvectors,
		"pq_bits":             idx.config.PQBits,
	}, nil
}

func (idx *ivfPQIndex) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	return idx.persistLocked("")
}

func (idx *ivfPQIndex) rebuildStructuresLocked() {
	vectors := make([][]float32, 0, len(idx.vectors))
	ids := make([]int64, 0, len(idx.vectors))
	for id, vec := range idx.vectors {
		vectors = append(vectors, vec)
		ids = append(ids, id)
	}
	idx.centroids = idx.buildCentroids(vectors)
	idx.pq = idx.trainPQCodebook(vectors)
	idx.assignments = make(map[int64]int, len(vectors))
	idx.residualNorms = make(map[int64]float32, len(vectors))
	idx.lists = make([][]ivfListEntry, len(idx.centroids))

	for i, vec := range vectors {
		clusterID := nearestCentroid(vec, idx.centroids)
		residual := subtractVec(vec, idx.centroids[clusterID])
		codes, norm := idx.encodeResidual(residual)
		factID := ids[i]
		idx.assignments[factID] = clusterID
		idx.residualNorms[factID] = norm
		idx.lists[clusterID] = append(idx.lists[clusterID], ivfListEntry{FactID: factID, Codes: codes, Norm: norm})
	}

	idx.builtAt = time.Now()
	idx.stats.Rebuilds++
	idx.stats.LastBuildTime = idx.builtAt
	idx.stats.FactCount = len(idx.vectors)
	idx.stats.CentroidCount = len(idx.centroids)
}

func (idx *ivfPQIndex) buildCentroids(vectors [][]float32) [][]float32 {
	if len(vectors) == 0 {
		return nil
	}
	wanted := idx.config.NList
	if wanted <= 0 {
		wanted = 1
	}
	if wanted > len(vectors) {
		wanted = len(vectors)
	}
	centroids := make([][]float32, wanted)
	for i := 0; i < wanted; i++ {
		seed := vectors[(i*len(vectors))/wanted]
		copySeed := make([]float32, len(seed))
		copy(copySeed, seed)
		centroids[i] = copySeed
	}
	assignments := make([]int, len(vectors))
	for iter := 0; iter < 4; iter++ {
		for i, vec := range vectors {
			assignments[i] = nearestCentroid(vec, centroids)
		}
		sums := make([][]float64, wanted)
		counts := make([]int, wanted)
		for i := range sums {
			sums[i] = make([]float64, idx.dimension)
		}
		for i, vec := range vectors {
			cid := assignments[i]
			counts[cid]++
			for d, value := range vec {
				sums[cid][d] += float64(value)
			}
		}
		for cid := range centroids {
			if counts[cid] == 0 {
				continue
			}
			for d := range centroids[cid] {
				centroids[cid][d] = float32(sums[cid][d] / float64(counts[cid]))
			}
			if idx.normalized {
				centroids[cid] = normalizeVectorF32(centroids[cid])
			}
		}
	}
	return centroids
}

func (idx *ivfPQIndex) trainPQCodebook(vectors [][]float32) pqCodebook {
	m := idx.config.PQSubvectors
	if m <= 0 {
		m = 1
	}
	if idx.dimension < m {
		m = idx.dimension
	}
	subDim := idx.dimension / m
	if subDim == 0 {
		subDim = 1
		m = idx.dimension
	}
	k := 1 << idx.config.PQBits
	if k <= 0 {
		k = 16
	}
	codebook := pqCodebook{Centroids: make([][][]float32, m)}
	for sub := 0; sub < m; sub++ {
		start := sub * subDim
		end := start + subDim
		if sub == m-1 {
			end = idx.dimension
		}
		if end <= start {
			end = start + 1
		}
		segments := make([][]float32, 0, len(vectors))
		for _, vec := range vectors {
			segments = append(segments, append([]float32(nil), vec[start:end]...))
		}
		codebook.Centroids[sub] = trainSubspaceKMeans(segments, k, 4)
	}
	return codebook
}

func trainSubspaceKMeans(vectors [][]float32, k int, iterations int) [][]float32 {
	if len(vectors) == 0 {
		return nil
	}
	if k > len(vectors) {
		k = len(vectors)
	}
	centroids := make([][]float32, k)
	for i := 0; i < k; i++ {
		seed := vectors[(i*len(vectors))/k]
		centroid := make([]float32, len(seed))
		copy(centroid, seed)
		centroids[i] = centroid
	}
	assignments := make([]int, len(vectors))
	for iter := 0; iter < iterations; iter++ {
		for i, vec := range vectors {
			assignments[i] = nearestByL2(vec, centroids)
		}
		sums := make([][]float64, len(centroids))
		counts := make([]int, len(centroids))
		for i := range sums {
			sums[i] = make([]float64, len(centroids[i]))
		}
		for i, vec := range vectors {
			cid := assignments[i]
			counts[cid]++
			for d, value := range vec {
				sums[cid][d] += float64(value)
			}
		}
		for cid := range centroids {
			if counts[cid] == 0 {
				continue
			}
			for d := range centroids[cid] {
				centroids[cid][d] = float32(sums[cid][d] / float64(counts[cid]))
			}
		}
	}
	return centroids
}

func nearestByL2(vec []float32, centroids [][]float32) int {
	bestID := 0
	bestDist := math.Inf(1)
	for i, centroid := range centroids {
		dist := l2Distance(vec, centroid)
		if dist < bestDist {
			bestDist = dist
			bestID = i
		}
	}
	return bestID
}

func l2Distance(a, b []float32) float64 {
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	var sum float64
	for i := 0; i < limit; i++ {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum
}

func subtractVec(a, b []float32) []float32 {
	result := make([]float32, len(a))
	for i := range a {
		if i < len(b) {
			result[i] = a[i] - b[i]
		} else {
			result[i] = a[i]
		}
	}
	return result
}

func (idx *ivfPQIndex) encodeResidual(residual []float32) ([]uint16, float32) {
	m := len(idx.pq.Centroids)
	if m == 0 {
		return nil, 0
	}
	codes := make([]uint16, m)
	var norm float64
	for _, v := range residual {
		norm += float64(v * v)
	}
	norm = math.Sqrt(norm)
	start := 0
	for sub := 0; sub < m; sub++ {
		codebook := idx.pq.Centroids[sub]
		width := idx.dimension / m
		if width == 0 {
			width = 1
		}
		end := start + width
		if sub == m-1 || end > len(residual) {
			end = len(residual)
		}
		segment := residual[start:end]
		codes[sub] = uint16(nearestByL2(segment, codebook))
		start = end
	}
	return codes, float32(norm)
}

func (idx *ivfPQIndex) buildADCLookup(query []float32, centroid []float32) [][][]float32 {
	residualQuery := subtractVec(query, centroid)
	m := len(idx.pq.Centroids)
	lookup := make([][][]float32, m)
	start := 0
	for sub := 0; sub < m; sub++ {
		codebook := idx.pq.Centroids[sub]
		width := idx.dimension / m
		if width == 0 {
			width = 1
		}
		end := start + width
		if sub == m-1 || end > len(residualQuery) {
			end = len(residualQuery)
		}
		segment := residualQuery[start:end]
		lookup[sub] = make([][]float32, len(codebook))
		for code := range codebook {
			lookup[sub][code] = []float32{float32(l2Distance(segment, codebook[code]))}
		}
		start = end
	}
	return lookup
}

func (idx *ivfPQIndex) scorePQEntry(entry ivfListEntry, lookup [][][]float32) float64 {
	var dist float64
	for sub, code := range entry.Codes {
		if sub >= len(lookup) || int(code) >= len(lookup[sub]) || len(lookup[sub][code]) == 0 {
			continue
		}
		dist += float64(lookup[sub][code][0])
	}
	return 1.0 / (1.0 + dist)
}

func (idx *ivfPQIndex) nearestCentroids(query []float32) []int {
	type centroidScore struct {
		id    int
		score float64
	}
	buffer := make([]centroidScore, 0, len(idx.centroids))
	for i, centroid := range idx.centroids {
		buffer = append(buffer, centroidScore{id: i, score: cosineSimilarityOptimized(query, centroid)})
	}
	sort.Slice(buffer, func(i, j int) bool {
		return buffer[i].score > buffer[j].score
	})
	order := make([]int, len(buffer))
	for i, item := range buffer {
		order[i] = item.id
	}
	return order
}

func prepareANNVector(vec []float32, normalized bool) []float32 {
	copyVec := make([]float32, len(vec))
	copy(copyVec, vec)
	if normalized {
		return normalizeVectorF32(copyVec)
	}
	return copyVec
}

func nearestCentroid(vec []float32, centroids [][]float32) int {
	bestID := 0
	bestScore := math.Inf(-1)
	for i, centroid := range centroids {
		score := cosineSimilarityOptimized(vec, centroid)
		if score > bestScore {
			bestScore = score
			bestID = i
		}
	}
	return bestID
}

func (idx *ivfPQIndex) persistLocked(embeddingModel string) error {
	if idx.indexPath == "" {
		return nil
	}
	if dir := filepath.Dir(filepath.Clean(idx.indexPath)); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("create ANN index directory: %w", err)
		}
	}
	file, err := os.Create(idx.indexPath)
	if err != nil {
		return fmt.Errorf("create ANN index file: %w", err)
	}
	defer file.Close()
	payload := ivfPQPersisted{
		Version:        1,
		Dimension:      idx.dimension,
		Normalized:     idx.normalized,
		Centroids:      idx.centroids,
		Lists:          idx.lists,
		Vectors:        idx.vectors,
		Assignments:    idx.assignments,
		ResidualNorms:  idx.residualNorms,
		PQ:             idx.pq,
		PQSubvectors:   idx.config.PQSubvectors,
		PQBits:         idx.config.PQBits,
		BuiltAt:        idx.builtAt,
		EmbeddingModel: embeddingModel,
	}
	enc := gob.NewEncoder(file)
	if err := enc.Encode(&payload); err != nil {
		return fmt.Errorf("encode ANN index: %w", err)
	}
	return nil
}

func (idx *ivfPQIndex) load(expectedEmbeddingModel string) error {
	file, err := os.Open(idx.indexPath)
	if err != nil {
		return err
	}
	defer file.Close()
	var payload ivfPQPersisted
	dec := gob.NewDecoder(file)
	if err := dec.Decode(&payload); err != nil {
		return fmt.Errorf("decode ANN index: %w", err)
	}
	if payload.Version != 1 {
		return fmt.Errorf("unsupported ANN index version: %d", payload.Version)
	}
	if idx.dimension != 0 && payload.Dimension != idx.dimension {
		return fmt.Errorf("ANN index dimension mismatch: got %d want %d", payload.Dimension, idx.dimension)
	}
	if expectedEmbeddingModel != "" && payload.EmbeddingModel != "" && payload.EmbeddingModel != expectedEmbeddingModel {
		return fmt.Errorf("ANN index embedding model mismatch")
	}
	idx.dimension = payload.Dimension
	idx.normalized = payload.Normalized
	idx.centroids = payload.Centroids
	idx.lists = payload.Lists
	idx.vectors = payload.Vectors
	if idx.vectors == nil {
		idx.vectors = make(map[int64][]float32)
	}
	idx.residualNorms = payload.ResidualNorms
	if idx.residualNorms == nil {
		idx.residualNorms = make(map[int64]float32)
	}
	idx.pq = payload.PQ
	idx.builtAt = payload.BuiltAt
	idx.stats.LastBuildTime = payload.BuiltAt
	idx.stats.CentroidCount = len(payload.Centroids)
	idx.stats.Dimension = payload.Dimension
	idx.stats.FactCount = len(idx.vectors)
	idx.assignments = payload.Assignments
	if idx.assignments == nil {
		idx.assignments = make(map[int64]int)
		for listID, list := range payload.Lists {
			for _, entry := range list {
				idx.assignments[entry.FactID] = listID
			}
		}
	}
	return nil
}

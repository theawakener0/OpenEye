package omem

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

type HotCacheConfig struct {
	MaxSize              int
	TTL                  time.Duration
	EnableAutoInvalidate bool
}

func DefaultHotCacheConfig() HotCacheConfig {
	return HotCacheConfig{
		MaxSize:              500,
		TTL:                  10 * time.Minute,
		EnableAutoInvalidate: true,
	}
}

type CachedFact struct {
	Fact       Fact
	Embedding  []float32
	Score      float64
	LastHit    time.Time
	HitCount   int
	InsertedAt time.Time
	Version    int64
}

type HotCache struct {
	config HotCacheConfig
	mu     sync.RWMutex

	factsByID   map[int64]*list.Element
	factsByHash map[string]*list.Element
	lru         *list.List

	version int64
	stats   HotCacheStats
}

type HotCacheStats struct {
	Hits          int64
	Misses        int64
	Evictions     int64
	Invalidations int64
	CurrentSize   int
}

func NewHotCache(cfg HotCacheConfig) *HotCache {
	if cfg.MaxSize <= 0 {
		cfg.MaxSize = 500
	}
	if cfg.TTL <= 0 {
		cfg.TTL = 10 * time.Minute
	}

	return &HotCache{
		config:      cfg,
		factsByID:   make(map[int64]*list.Element),
		factsByHash: make(map[string]*list.Element),
		lru:         list.New(),
		version:     1,
		stats:       HotCacheStats{},
	}
}

func (hc *HotCache) Get(ctx context.Context, factID int64) (*CachedFact, bool) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	elem, ok := hc.factsByID[factID]
	if !ok {
		hc.stats.Misses++
		return nil, false
	}

	cached := elem.Value.(*CachedFact)

	if time.Since(cached.InsertedAt) > hc.config.TTL {
		hc.mu.RUnlock()
		hc.mu.Lock()
		defer hc.mu.Unlock()
		hc.removeElement(elem)
		hc.stats.Misses++
		hc.stats.Invalidations++
		return nil, false
	}

	cached.LastHit = time.Now()
	cached.HitCount++
	hc.stats.Hits++

	hc.lru.MoveToFront(elem)

	return cached, true
}

func (hc *HotCache) GetByQueryHash(ctx context.Context, queryHash string) (*CachedFact, bool) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	elem, ok := hc.factsByHash[queryHash]
	if !ok {
		return nil, false
	}

	cached := elem.Value.(*CachedFact)
	cached.LastHit = time.Now()
	cached.HitCount++
	hc.stats.Hits++

	hc.lru.MoveToFront(elem)

	return cached, true
}

func (hc *HotCache) Put(ctx context.Context, fact Fact, embedding []float32, score float64) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	if elem, exists := hc.factsByID[fact.ID]; exists {
		cached := elem.Value.(*CachedFact)
		cached.Embedding = embedding
		cached.Score = score
		cached.Version = hc.version
		hc.lru.MoveToFront(elem)
		return
	}

	for hc.lru.Len() >= hc.config.MaxSize {
		hc.evictOldest()
	}

	cached := &CachedFact{
		Fact:       fact,
		Embedding:  embedding,
		Score:      score,
		LastHit:    time.Now(),
		HitCount:   1,
		InsertedAt: time.Now(),
		Version:    hc.version,
	}

	elem := hc.lru.PushFront(cached)
	hc.factsByID[fact.ID] = elem

	queryHash := generateQueryHash(fact.AtomicText)
	hc.factsByHash[queryHash] = elem

	hc.stats.CurrentSize = hc.lru.Len()
}

func (hc *HotCache) PutBatch(ctx context.Context, facts []Fact, embeddings [][]float32, scores []float64) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	for i, fact := range facts {
		if elem, exists := hc.factsByID[fact.ID]; exists {
			cached := elem.Value.(*CachedFact)
			if i < len(embeddings) {
				cached.Embedding = embeddings[i]
			}
			if i < len(scores) {
				cached.Score = scores[i]
			}
			cached.Version = hc.version
			hc.lru.MoveToFront(elem)
			continue
		}

		for hc.lru.Len() >= hc.config.MaxSize {
			hc.evictOldest()
		}

		var emb []float32
		if i < len(embeddings) {
			emb = embeddings[i]
		}
		var scr float64
		if i < len(scores) {
			scr = scores[i]
		}

		cached := &CachedFact{
			Fact:       fact,
			Embedding:  emb,
			Score:      scr,
			LastHit:    time.Now(),
			HitCount:   1,
			InsertedAt: time.Now(),
			Version:    hc.version,
		}

		elem := hc.lru.PushFront(cached)
		hc.factsByID[fact.ID] = elem

		queryHash := generateQueryHash(fact.AtomicText)
		hc.factsByHash[queryHash] = elem
	}

	hc.stats.CurrentSize = hc.lru.Len()
}

func (hc *HotCache) Invalidate(factID int64) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	if elem, ok := hc.factsByID[factID]; ok {
		hc.removeElement(elem)
		hc.stats.Invalidations++
	}
}

func (hc *HotCache) InvalidateBatch(factIDs []int64) {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	for _, factID := range factIDs {
		if elem, ok := hc.factsByID[factID]; ok {
			hc.removeElement(elem)
			hc.stats.Invalidations++
		}
	}
}

func (hc *HotCache) InvalidateAll() {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	hc.factsByID = make(map[int64]*list.Element)
	hc.factsByHash = make(map[string]*list.Element)
	hc.lru = list.New()
	hc.version++
	hc.stats.CurrentSize = 0
	hc.stats.Invalidations++
}

func (hc *HotCache) GetStats() HotCacheStats {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	stats := hc.stats
	stats.CurrentSize = hc.lru.Len()
	return stats
}

func (hc *HotCache) Size() int {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	return hc.lru.Len()
}

func (hc *HotCache) HitRate() float64 {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	total := hc.stats.Hits + hc.stats.Misses
	if total == 0 {
		return 0.0
	}
	return float64(hc.stats.Hits) / float64(total)
}

func (hc *HotCache) evictOldest() {
	if hc.lru.Len() == 0 {
		return
	}

	elem := hc.lru.Back()
	if elem != nil {
		hc.removeElement(elem)
		hc.stats.Evictions++
	}
}

func (hc *HotCache) removeElement(elem *list.Element) {
	cached := elem.Value.(*CachedFact)
	delete(hc.factsByID, cached.Fact.ID)

	queryHash := generateQueryHash(cached.Fact.AtomicText)
	delete(hc.factsByHash, queryHash)

	hc.lru.Remove(elem)
	hc.stats.CurrentSize = hc.lru.Len()
}

func generateQueryHash(text string) string {
	normalized := normalizeForHash(text)
	hash := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(hash[:])
}

func normalizeForHash(text string) string {
	lower := toLowerASCII(text)
	trimmed := trimSpaces(lower)
	return trimmed
}

func toLowerASCII(s string) string {
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		result[i] = c
	}
	return string(result)
}

func trimSpaces(s string) string {
	start := 0
	end := len(s)
	for start < end && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r') {
		end--
	}
	if start >= end {
		return ""
	}
	return s[start:end]
}

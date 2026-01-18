package embedding

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

// CachedProvider wraps an embedding provider with caching capabilities.
// It provides both request-scoped caching and persistent LRU caching.
type CachedProvider struct {
	provider Provider
	cache    *lruCache
	mu       sync.RWMutex
}

// lruCache is a simple LRU cache for embeddings.
type lruCache struct {
	maxSize   int
	items     map[string]*cacheEntry
	evictList []string
	mu        sync.RWMutex
}

type cacheEntry struct {
	embedding []float32
	createdAt time.Time
	hits      int
}

// NewCachedProvider creates a new cached embedding provider.
func NewCachedProvider(provider Provider, cacheSize int) *CachedProvider {
	if cacheSize <= 0 {
		cacheSize = 1000
	}
	return &CachedProvider{
		provider: provider,
		cache: &lruCache{
			maxSize:   cacheSize,
			items:     make(map[string]*cacheEntry),
			evictList: make([]string, 0, cacheSize),
		},
	}
}

// Embed generates an embedding for the given text, using cache if available.
func (c *CachedProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	if c == nil || c.provider == nil {
		return nil, nil
	}

	key := hashText(text)

	// Check cache first
	c.cache.mu.RLock()
	if entry, ok := c.cache.items[key]; ok {
		entry.hits++
		result := make([]float32, len(entry.embedding))
		copy(result, entry.embedding)
		c.cache.mu.RUnlock()
		return result, nil
	}
	c.cache.mu.RUnlock()

	// Generate embedding
	embedding, err := c.provider.Embed(ctx, text)
	if err != nil {
		return nil, err
	}

	// Store in cache
	c.cache.mu.Lock()
	c.cache.set(key, embedding)
	c.cache.mu.Unlock()

	return embedding, nil
}

// EmbedBatch generates embeddings for multiple texts efficiently.
// It checks cache for each text and only computes missing embeddings.
func (c *CachedProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if c == nil || c.provider == nil {
		return nil, nil
	}

	results := make([][]float32, len(texts))
	toCompute := make([]int, 0)

	// Check cache for each text
	c.cache.mu.RLock()
	for i, text := range texts {
		key := hashText(text)
		if entry, ok := c.cache.items[key]; ok {
			entry.hits++
			results[i] = make([]float32, len(entry.embedding))
			copy(results[i], entry.embedding)
		} else {
			toCompute = append(toCompute, i)
		}
	}
	c.cache.mu.RUnlock()

	// Compute missing embeddings
	for _, i := range toCompute {
		embedding, err := c.provider.Embed(ctx, texts[i])
		if err != nil {
			return nil, err
		}
		results[i] = embedding

		// Store in cache
		key := hashText(texts[i])
		c.cache.mu.Lock()
		c.cache.set(key, embedding)
		c.cache.mu.Unlock()
	}

	return results, nil
}

// Close releases resources.
func (c *CachedProvider) Close() error {
	if c == nil || c.provider == nil {
		return nil
	}
	return c.provider.Close()
}

// CacheStats returns cache statistics.
func (c *CachedProvider) CacheStats() map[string]interface{} {
	c.cache.mu.RLock()
	defer c.cache.mu.RUnlock()

	totalHits := 0
	for _, entry := range c.cache.items {
		totalHits += entry.hits
	}

	return map[string]interface{}{
		"size":       len(c.cache.items),
		"max_size":   c.cache.maxSize,
		"total_hits": totalHits,
	}
}

func (l *lruCache) set(key string, embedding []float32) {
	// Check if we need to evict
	if len(l.items) >= l.maxSize && len(l.evictList) > 0 {
		// Evict oldest entry
		oldestKey := l.evictList[0]
		l.evictList = l.evictList[1:]
		delete(l.items, oldestKey)
	}

	// Store new entry
	stored := make([]float32, len(embedding))
	copy(stored, embedding)
	l.items[key] = &cacheEntry{
		embedding: stored,
		createdAt: time.Now(),
		hits:      0,
	}
	l.evictList = append(l.evictList, key)
}

func hashText(text string) string {
	h := sha256.Sum256([]byte(text))
	return hex.EncodeToString(h[:16]) // Use first 16 bytes for shorter key
}

// RequestScopedCache provides embedding caching within a single request.
// This eliminates redundant embedding calls when the same query is used
// for multiple operations (vector search, RAG, summarization).
type RequestScopedCache struct {
	provider Provider
	cache    map[string][]float32
	mu       sync.RWMutex
}

// NewRequestScopedCache creates a new request-scoped cache.
func NewRequestScopedCache(provider Provider) *RequestScopedCache {
	return &RequestScopedCache{
		provider: provider,
		cache:    make(map[string][]float32),
	}
}

// Embed generates or retrieves a cached embedding.
func (r *RequestScopedCache) Embed(ctx context.Context, text string) ([]float32, error) {
	if r == nil || r.provider == nil {
		return nil, nil
	}

	key := hashText(text)

	// Check request-local cache
	r.mu.RLock()
	if cached, ok := r.cache[key]; ok {
		result := make([]float32, len(cached))
		copy(result, cached)
		r.mu.RUnlock()
		return result, nil
	}
	r.mu.RUnlock()

	// Generate embedding
	embedding, err := r.provider.Embed(ctx, text)
	if err != nil {
		return nil, err
	}

	// Store in request cache
	r.mu.Lock()
	stored := make([]float32, len(embedding))
	copy(stored, embedding)
	r.cache[key] = stored
	r.mu.Unlock()

	return embedding, nil
}

// Close is a no-op for request-scoped cache (doesn't own the provider).
func (r *RequestScopedCache) Close() error {
	return nil
}

// GetCached returns a cached embedding without computing if not found.
func (r *RequestScopedCache) GetCached(text string) ([]float32, bool) {
	if r == nil {
		return nil, false
	}

	key := hashText(text)

	r.mu.RLock()
	defer r.mu.RUnlock()

	if cached, ok := r.cache[key]; ok {
		result := make([]float32, len(cached))
		copy(result, cached)
		return result, true
	}
	return nil, false
}

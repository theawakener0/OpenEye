package memory

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// SlidingContextConfig configures the sliding context window manager.
type SlidingContextConfig struct {
	MaxTokens          int
	ReservedForPrompt  int
	ReservedForSummary int
	RecencyWeight      float64
	RelevanceWeight    float64
}

// DefaultSlidingContextConfig returns sensible defaults.
func DefaultSlidingContextConfig() SlidingContextConfig {
	return SlidingContextConfig{
		MaxTokens:          2048,
		ReservedForPrompt:  512,
		ReservedForSummary: 256,
		RecencyWeight:      0.3,
		RelevanceWeight:    0.7,
	}
}

// SlidingContextWindow manages context fitting for limited context windows.
type SlidingContextWindow struct {
	config SlidingContextConfig
	mu     sync.RWMutex
}

// NewSlidingContextWindow creates a new sliding context window manager.
func NewSlidingContextWindow(cfg SlidingContextConfig) *SlidingContextWindow {
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 2048
	}
	if cfg.ReservedForPrompt <= 0 {
		cfg.ReservedForPrompt = 512
	}
	if cfg.ReservedForSummary <= 0 {
		cfg.ReservedForSummary = 256
	}
	if cfg.RecencyWeight <= 0 {
		cfg.RecencyWeight = 0.3
	}
	if cfg.RelevanceWeight <= 0 {
		cfg.RelevanceWeight = 0.7
	}

	return &SlidingContextWindow{config: cfg}
}

// ContextItem represents a piece of context with metadata for ranking.
type ContextItem struct {
	ID           int64
	Text         string
	Role         string
	TokenCount   int
	Timestamp    time.Time
	Relevance    float64
	CombinedRank float64
}

// FitContext selects and orders context items to fit within the token budget.
func (s *SlidingContextWindow) FitContext(items []ContextItem, summary string, promptTokens int) ([]ContextItem, string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	availableTokens := s.config.MaxTokens - promptTokens - s.config.ReservedForPrompt

	// If we have a summary, reserve space for it
	summaryTokens := 0
	if summary != "" {
		summaryTokens = estimateTokens(summary)
		if summaryTokens > s.config.ReservedForSummary {
			// Truncate summary if too long
			summary = truncateToTokens(summary, s.config.ReservedForSummary)
			summaryTokens = s.config.ReservedForSummary
		}
		availableTokens -= summaryTokens
	}

	if availableTokens <= 0 {
		return nil, summary, nil
	}

	// Score and rank items
	rankedItems := s.rankItems(items)

	// Select items that fit
	var selected []ContextItem
	usedTokens := 0

	for _, item := range rankedItems {
		if usedTokens+item.TokenCount > availableTokens {
			continue
		}
		selected = append(selected, item)
		usedTokens += item.TokenCount
	}

	// Sort selected by timestamp for coherent context
	sort.Slice(selected, func(i, j int) bool {
		return selected[i].Timestamp.Before(selected[j].Timestamp)
	})

	return selected, summary, nil
}

// rankItems scores items based on recency and relevance.
func (s *SlidingContextWindow) rankItems(items []ContextItem) []ContextItem {
	if len(items) == 0 {
		return items
	}

	// Find time range for normalization
	now := time.Now()
	var maxAge float64
	for _, item := range items {
		age := now.Sub(item.Timestamp).Seconds()
		if age > maxAge {
			maxAge = age
		}
	}
	if maxAge == 0 {
		maxAge = 1
	}

	// Calculate combined rank
	result := make([]ContextItem, len(items))
	copy(result, items)

	for i := range result {
		age := now.Sub(result[i].Timestamp).Seconds()
		recencyScore := 1 - (age / maxAge) // Newer = higher score

		// Combine recency and relevance
		result[i].CombinedRank = s.config.RecencyWeight*recencyScore + s.config.RelevanceWeight*result[i].Relevance
	}

	// Sort by combined rank descending
	sort.Slice(result, func(i, j int) bool {
		return result[i].CombinedRank > result[j].CombinedRank
	})

	return result
}

// truncateToTokens truncates text to approximately fit within token limit.
func truncateToTokens(text string, maxTokens int) string {
	maxChars := maxTokens * 4 // Approximate
	if len(text) <= maxChars {
		return text
	}
	return text[:maxChars] + "..."
}

// ContextBuilder helps construct context from various sources.
type ContextBuilder struct {
	systemMessage   string
	summary         string
	recentMemories  []ContextItem
	retrievedItems  []ContextItem
	knowledge       []string
	maxTokens       int
	mu              sync.Mutex
}

// NewContextBuilder creates a new context builder.
func NewContextBuilder(maxTokens int) *ContextBuilder {
	if maxTokens <= 0 {
		maxTokens = 2048
	}
	return &ContextBuilder{maxTokens: maxTokens}
}

// SetSystemMessage sets the system message.
func (b *ContextBuilder) SetSystemMessage(msg string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.systemMessage = strings.TrimSpace(msg)
}

// SetSummary sets the memory summary.
func (b *ContextBuilder) SetSummary(summary string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.summary = strings.TrimSpace(summary)
}

// AddRecentMemory adds a recent memory item.
func (b *ContextBuilder) AddRecentMemory(item ContextItem) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.recentMemories = append(b.recentMemories, item)
}

// AddRetrievedItem adds a retrieved (semantic search) item.
func (b *ContextBuilder) AddRetrievedItem(item ContextItem) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.retrievedItems = append(b.retrievedItems, item)
}

// AddKnowledge adds a knowledge snippet (from RAG).
func (b *ContextBuilder) AddKnowledge(text string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.knowledge = append(b.knowledge, strings.TrimSpace(text))
}

// Build constructs the final context string within token limits.
func (b *ContextBuilder) Build(prompt string) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	var parts []string
	usedTokens := 0
	promptTokens := estimateTokens(prompt)

	// Reserve space for prompt
	availableTokens := b.maxTokens - promptTokens - 100 // Buffer

	// System message (required)
	if b.systemMessage != "" {
		sysTokens := estimateTokens(b.systemMessage)
		if sysTokens < availableTokens {
			parts = append(parts, "## System\n"+b.systemMessage)
			usedTokens += sysTokens
		}
	}

	// Summary (if available)
	if b.summary != "" && usedTokens < availableTokens {
		summaryTokens := estimateTokens(b.summary)
		remaining := availableTokens - usedTokens
		if summaryTokens < remaining {
			parts = append(parts, "\n## Memory Summary\n"+b.summary)
			usedTokens += summaryTokens
		} else if remaining > 100 {
			truncated := truncateToTokens(b.summary, remaining-10)
			parts = append(parts, "\n## Memory Summary\n"+truncated)
			usedTokens += estimateTokens(truncated)
		}
	}

	// Merge and deduplicate memories
	allMemories := b.mergeMemories()
	if len(allMemories) > 0 && usedTokens < availableTokens {
		var memoryParts []string
		for _, item := range allMemories {
			remaining := availableTokens - usedTokens
			if remaining < 50 {
				break
			}
			if item.TokenCount < remaining {
				memoryParts = append(memoryParts, fmt.Sprintf("%s: %s", item.Role, item.Text))
				usedTokens += item.TokenCount
			}
		}
		if len(memoryParts) > 0 {
			parts = append(parts, "\n## Recent Context\n"+strings.Join(memoryParts, "\n"))
		}
	}

	// Knowledge (RAG results)
	if len(b.knowledge) > 0 && usedTokens < availableTokens {
		var knowledgeParts []string
		for i, k := range b.knowledge {
			kTokens := estimateTokens(k)
			remaining := availableTokens - usedTokens
			if remaining < 50 {
				break
			}
			if kTokens < remaining {
				knowledgeParts = append(knowledgeParts, fmt.Sprintf("[%d] %s", i+1, k))
				usedTokens += kTokens
			}
		}
		if len(knowledgeParts) > 0 {
			parts = append(parts, "\n## Retrieved Knowledge\n"+strings.Join(knowledgeParts, "\n"))
		}
	}

	// Prompt
	parts = append(parts, "\n## User Query\n"+prompt)

	return strings.Join(parts, "\n"), nil
}

// mergeMemories deduplicates and sorts memories by relevance and recency.
func (b *ContextBuilder) mergeMemories() []ContextItem {
	seen := make(map[int64]bool)
	var merged []ContextItem

	// Prioritize retrieved items (higher relevance)
	for _, item := range b.retrievedItems {
		if !seen[item.ID] {
			seen[item.ID] = true
			merged = append(merged, item)
		}
	}

	// Add recent memories not already included
	for _, item := range b.recentMemories {
		if !seen[item.ID] {
			seen[item.ID] = true
			merged = append(merged, item)
		}
	}

	// Sort by combined score (relevance + recency)
	now := time.Now()
	sort.Slice(merged, func(i, j int) bool {
		// Combine relevance with recency
		recencyI := 1.0 / (1.0 + now.Sub(merged[i].Timestamp).Hours())
		recencyJ := 1.0 / (1.0 + now.Sub(merged[j].Timestamp).Hours())
		scoreI := merged[i].Relevance*0.7 + recencyI*0.3
		scoreJ := merged[j].Relevance*0.7 + recencyJ*0.3
		return scoreI > scoreJ
	})

	return merged
}

// MemoryCompressor handles memory compression operations.
type MemoryCompressor struct {
	vectorStore   *VectorStore
	summarizeFn   func(ctx context.Context, texts []string) (string, error)
	embedFn       func(ctx context.Context, text string) ([]float32, error)
	batchSize     int
	compressionAge time.Duration
	mu            sync.Mutex
}

// MemoryCompressorConfig configures the memory compressor.
type MemoryCompressorConfig struct {
	BatchSize      int
	CompressionAge time.Duration
}

// NewMemoryCompressor creates a new memory compressor.
func NewMemoryCompressor(
	store *VectorStore,
	summarize func(ctx context.Context, texts []string) (string, error),
	embed func(ctx context.Context, text string) ([]float32, error),
	cfg MemoryCompressorConfig,
) *MemoryCompressor {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 10
	}
	if cfg.CompressionAge <= 0 {
		cfg.CompressionAge = 24 * time.Hour
	}

	return &MemoryCompressor{
		vectorStore:    store,
		summarizeFn:    summarize,
		embedFn:        embed,
		batchSize:      cfg.BatchSize,
		compressionAge: cfg.CompressionAge,
	}
}

// CompressOld compresses memories older than the configured age.
func (c *MemoryCompressor) CompressOld(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.vectorStore == nil || c.summarizeFn == nil {
		return nil
	}

	summaryFn := func(texts []string) (string, error) {
		return c.summarizeFn(ctx, texts)
	}

	var embedFn func(string) ([]float32, error)
	if c.embedFn != nil {
		embedFn = func(text string) ([]float32, error) {
			return c.embedFn(ctx, text)
		}
	}

	return c.vectorStore.CompressOldMemories(ctx, c.compressionAge, summaryFn, embedFn)
}

// ComputeCompressionRatio calculates how much memory has been compressed.
func (c *MemoryCompressor) ComputeCompressionRatio(ctx context.Context) (float64, error) {
	if c.vectorStore == nil {
		return 0, nil
	}

	stats, err := c.vectorStore.GetMemoryStats(ctx)
	if err != nil {
		return 0, err
	}

	total, _ := stats["total_memories"].(int)
	compressed, _ := stats["compressed_memories"].(int)

	if total == 0 {
		return 0, nil
	}

	return float64(compressed) / float64(total), nil
}

// HybridMemoryEngine combines vector search with compression and context fitting.
type HybridMemoryEngine struct {
	vectorStore     *VectorStore
	contextWindow   *SlidingContextWindow
	compressor      *MemoryCompressor
	asyncWriter     *AsyncWriter
	embedFn         func(ctx context.Context, text string) ([]float32, error)
	summarizeFn     func(ctx context.Context, texts []string) (string, error)
	autoCompress    bool
	compressCounter int
	compressEvery   int
	mu              sync.RWMutex

	// Cache for last query embedding to avoid redundant calls
	lastQuery     string
	lastEmbedding []float32
}

type embeddingProviderFunc struct {
	fn func(context.Context, string) ([]float32, error)
}

func (f *embeddingProviderFunc) Embed(ctx context.Context, text string) ([]float32, error) {
	if f.fn == nil {
		return nil, nil
	}
	return f.fn(ctx, text)
}

type summarizerProviderFunc struct {
	fn func(context.Context, []string) (string, error)
}

func (f *summarizerProviderFunc) Summarize(ctx context.Context, texts []string) (string, error) {
	if f.fn == nil {
		return "", nil
	}
	return f.fn(ctx, texts)
}

// HybridMemoryConfig configures the hybrid memory engine.
type HybridMemoryConfig struct {
	VectorConfig   VectorStoreConfig
	ContextConfig  SlidingContextConfig
	CompressConfig MemoryCompressorConfig
	AutoCompress   bool
	CompressEvery  int
}

// DefaultHybridMemoryConfig returns sensible defaults.
func DefaultHybridMemoryConfig() HybridMemoryConfig {
	return HybridMemoryConfig{
		VectorConfig:   DefaultVectorStoreConfig(),
		ContextConfig:  DefaultSlidingContextConfig(),
		CompressConfig: MemoryCompressorConfig{BatchSize: 10, CompressionAge: 24 * time.Hour},
		AutoCompress:   true,
		CompressEvery:  50, // Every 50 inserts
	}
}

// NewHybridMemoryEngine creates a complete hybrid memory engine.
func NewHybridMemoryEngine(
	cfg HybridMemoryConfig,
	embed func(ctx context.Context, text string) ([]float32, error),
	summarize func(ctx context.Context, texts []string) (string, error),
) (*HybridMemoryEngine, error) {
	vectorStore, err := NewVectorStore(cfg.VectorConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}

	contextWindow := NewSlidingContextWindow(cfg.ContextConfig)

	compressor := NewMemoryCompressor(vectorStore, summarize, embed, cfg.CompressConfig)

	// Initialize async writer for non-blocking storage
	asyncWriter := NewAsyncWriter(
		vectorStore,
		&embeddingProviderFunc{fn: embed},
		&summarizerProviderFunc{fn: summarize},
		DefaultAsyncWriterConfig(),
	)
	asyncWriter.Start()

	engine := &HybridMemoryEngine{
		vectorStore:   vectorStore,
		contextWindow: contextWindow,
		compressor:    compressor,
		asyncWriter:   asyncWriter,
		embedFn:       embed,
		summarizeFn:   summarize,
		autoCompress:  cfg.AutoCompress,
		compressEvery: cfg.CompressEvery,
	}

	if engine.compressEvery <= 0 {
		engine.compressEvery = 50
	}

	return engine, nil
}

// Store saves a new memory entry. It uses an async writer to avoid blocking
// the critical path with embedding and summarization.
func (e *HybridMemoryEngine) Store(ctx context.Context, text, role string) (int64, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Check if we have a cached embedding for this text (likely from a recent Retrieve call)
	var embedding []float32
	if text == e.lastQuery {
		embedding = e.lastEmbedding
	}

	// Queue for async processing
	if e.asyncWriter != nil {
		if embedding != nil {
			err := e.asyncWriter.WriteWithEmbedding(text, role, embedding)
			if err == nil {
				return 0, nil // ID not available yet for async writes
			}
		} else {
			err := e.asyncWriter.Write(ctx, text, role)
			if err == nil {
				return 0, nil
			}
		}
	}

	// Fallback to synchronous if async writer is missing or fails
	var summary string
	if len(text) > 500 && e.summarizeFn != nil {
		summaryResult, _ := e.summarizeFn(ctx, []string{text})
		summary = summaryResult
	}

	if embedding == nil && e.embedFn != nil {
		embedding, _ = e.embedFn(ctx, text)
	}

	id, err := e.vectorStore.InsertMemory(ctx, text, summary, role, embedding)
	if err != nil {
		return 0, err
	}

	// Check if we should auto-compress
	e.compressCounter++
	if e.autoCompress && e.compressCounter >= e.compressEvery {
		e.compressCounter = 0
		go func() {
			compressCtx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			e.compressor.CompressOld(compressCtx)
		}()
	}

	return id, nil
}

// Retrieve gets relevant memories for a query.
func (e *HybridMemoryEngine) Retrieve(ctx context.Context, query string, limit int) ([]VectorEntry, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.embedFn == nil {
		// Fallback to recent memories if no embedding
		return e.vectorStore.GetRecentMemories(ctx, limit)
	}

	var queryEmbedding []float32
	if query == e.lastQuery && e.lastEmbedding != nil {
		queryEmbedding = e.lastEmbedding
	} else {
		var err error
		queryEmbedding, err = e.embedFn(ctx, query)
		if err != nil {
			// Fallback to recent
			return e.vectorStore.GetRecentMemories(ctx, limit)
		}
		// Cache for reuse in Store call
		e.lastQuery = query
		e.lastEmbedding = queryEmbedding
	}

	return e.vectorStore.SearchMemory(ctx, queryEmbedding, limit)
}

// BuildContext constructs optimized context for the SLM.
func (e *HybridMemoryEngine) BuildContext(ctx context.Context, query string, maxTokens int) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	builder := NewContextBuilder(maxTokens)

	// Get recent memories
	recent, err := e.vectorStore.GetRecentMemories(ctx, 10)
	if err == nil {
		for _, entry := range recent {
			builder.AddRecentMemory(ContextItem{
				ID:         entry.ID,
				Text:       entry.Text,
				Role:       entry.Role,
				TokenCount: entry.TokenCount,
				Timestamp:  entry.CreatedAt,
				Relevance:  0.5, // Default relevance for recent items
			})
		}
	}

	// Get semantically relevant memories
	if e.embedFn != nil {
		var queryEmbedding []float32
		if query == e.lastQuery && e.lastEmbedding != nil {
			queryEmbedding = e.lastEmbedding
		} else {
			var err error
			queryEmbedding, err = e.embedFn(ctx, query)
			if err == nil {
				e.lastQuery = query
				e.lastEmbedding = queryEmbedding
			}
		}

		if queryEmbedding != nil {
			relevant, err := e.vectorStore.SearchMemory(ctx, queryEmbedding, 10)
			if err == nil {
				for i, entry := range relevant {
					// Higher relevance for better matches (assumes sorted by score)
					relevance := 1.0 - float64(i)*0.1
					if relevance < 0.3 {
						relevance = 0.3
					}
					builder.AddRetrievedItem(ContextItem{
						ID:         entry.ID,
						Text:       entry.Text,
						Role:       entry.Role,
						TokenCount: entry.TokenCount,
						Timestamp:  entry.CreatedAt,
						Relevance:  relevance,
					})
				}
			}
		}
	}

	// Generate summary of old memories if needed
	if e.summarizeFn != nil {
		oldMemories, _ := e.vectorStore.GetRecentMemories(ctx, 50)
		if len(oldMemories) > 20 {
			var texts []string
			for _, m := range oldMemories[:20] {
				texts = append(texts, fmt.Sprintf("%s: %s", m.Role, m.Text))
			}
			summary, err := e.summarizeFn(ctx, texts)
			if err == nil {
				builder.SetSummary(summary)
			}
		}
	}

	return builder.Build(query)
}

// GetStats returns memory statistics.
func (e *HybridMemoryEngine) GetStats(ctx context.Context) (map[string]interface{}, error) {
	return e.vectorStore.GetMemoryStats(ctx)
}

// Compress manually triggers memory compression.
func (e *HybridMemoryEngine) Compress(ctx context.Context) error {
	return e.compressor.CompressOld(ctx)
}

// Close releases all resources, including the async writer.
func (e *HybridMemoryEngine) Close() error {
	if e.asyncWriter != nil {
		e.asyncWriter.Stop()
	}
	return e.vectorStore.Close()
}

// Utility functions for context management

// EstimateContextSize estimates how many tokens a context will use.
func EstimateContextSize(systemMsg, summary string, memories []VectorEntry, knowledge []string) int {
	total := estimateTokens(systemMsg) + estimateTokens(summary)
	for _, m := range memories {
		total += m.TokenCount
	}
	for _, k := range knowledge {
		total += estimateTokens(k)
	}
	return total
}

// TruncateToFit truncates content to fit within a token budget.
func TruncateToFit(content string, maxTokens int) string {
	return truncateToTokens(content, maxTokens)
}

// CalculateRecencyScore returns a score based on how recent something is.
func CalculateRecencyScore(timestamp time.Time, halfLifeHours float64) float64 {
	if halfLifeHours <= 0 {
		halfLifeHours = 24
	}
	hours := time.Since(timestamp).Hours()
	return math.Exp(-0.693 * hours / halfLifeHours) // Exponential decay
}

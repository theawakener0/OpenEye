package memory

import (
	"context"
	"time"
)

// MemoryEngine defines the interface for memory storage and retrieval.
type MemoryEngine interface {
	// Store saves a new memory entry.
	Store(ctx context.Context, text, role string) (int64, error)

	// Retrieve gets relevant memories for a query.
	Retrieve(ctx context.Context, query string, limit int) ([]VectorEntry, error)

	// BuildContext constructs optimized context for the SLM.
	BuildContext(ctx context.Context, query string, maxTokens int) (string, error)

	// GetStats returns memory statistics.
	GetStats(ctx context.Context) (map[string]interface{}, error)

	// Compress triggers memory compression.
	Compress(ctx context.Context) error

	// Close releases resources.
	Close() error
}

// EmbeddingProvider generates embeddings for text.
type EmbeddingProvider interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// SummaryProvider generates summaries from text.
type SummaryProvider interface {
	Summarize(ctx context.Context, texts []string) (string, error)
}

// EngineConfig holds configuration for the memory engine.
type EngineConfig struct {
	// Database settings
	DBPath       string
	EmbeddingDim int

	// Context window settings
	MaxContextTokens   int
	ReservedForPrompt  int
	ReservedForSummary int

	// Retrieval settings
	MinSimilarity     float64
	SlidingWindowSize int
	RecencyWeight     float64
	RelevanceWeight   float64

	// Compression settings
	CompressionEnabled bool
	CompressionAge     time.Duration
	CompressBatchSize  int
	AutoCompress       bool
	CompressEveryN     int
}

// DefaultEngineConfig returns sensible defaults.
func DefaultEngineConfig() EngineConfig {
	return EngineConfig{
		DBPath:             "openeye_memory.duckdb",
		EmbeddingDim:       384,
		MaxContextTokens:   2048,
		ReservedForPrompt:  512,
		ReservedForSummary: 256,
		MinSimilarity:      0.3,
		SlidingWindowSize:  50,
		RecencyWeight:      0.3,
		RelevanceWeight:    0.7,
		CompressionEnabled: true,
		CompressionAge:     24 * time.Hour,
		CompressBatchSize:  10,
		AutoCompress:       true,
		CompressEveryN:     50,
	}
}

// NewEngine creates a new memory engine with the given configuration.
func NewEngine(
	cfg EngineConfig,
	embedder EmbeddingProvider,
	summarizer SummaryProvider,
) (MemoryEngine, error) {
	// Convert to internal config types
	hybridCfg := HybridMemoryConfig{
		VectorConfig: VectorStoreConfig{
			DBPath:             cfg.DBPath,
			EmbeddingDim:       cfg.EmbeddingDim,
			MaxContextTokens:   cfg.MaxContextTokens,
			CompressionEnabled: cfg.CompressionEnabled,
			SlidingWindowSize:  cfg.SlidingWindowSize,
			MinSimilarity:      cfg.MinSimilarity,
		},
		ContextConfig: SlidingContextConfig{
			MaxTokens:          cfg.MaxContextTokens,
			ReservedForPrompt:  cfg.ReservedForPrompt,
			ReservedForSummary: cfg.ReservedForSummary,
			RecencyWeight:      cfg.RecencyWeight,
			RelevanceWeight:    cfg.RelevanceWeight,
		},
		CompressConfig: MemoryCompressorConfig{
			BatchSize:      cfg.CompressBatchSize,
			CompressionAge: cfg.CompressionAge,
		},
		AutoCompress:  cfg.AutoCompress,
		CompressEvery: cfg.CompressEveryN,
	}

	// Create wrapper functions for embedder and summarizer
	var embedFn func(ctx context.Context, text string) ([]float32, error)
	if embedder != nil {
		embedFn = embedder.Embed
	}

	var summarizeFn func(ctx context.Context, texts []string) (string, error)
	if summarizer != nil {
		summarizeFn = summarizer.Summarize
	}

	return NewHybridMemoryEngine(hybridCfg, embedFn, summarizeFn)
}

// MemoryItem represents a memory item for external use.
type MemoryItem struct {
	ID        int64
	Text      string
	Summary   string
	Role      string
	Score     float64
	Timestamp time.Time
}

// ToMemoryItems converts VectorEntries to MemoryItems.
func ToMemoryItems(entries []VectorEntry) []MemoryItem {
	items := make([]MemoryItem, len(entries))
	for i, e := range entries {
		items[i] = MemoryItem{
			ID:        e.ID,
			Text:      e.Text,
			Summary:   e.Summary,
			Role:      e.Role,
			Timestamp: e.CreatedAt,
		}
	}
	return items
}

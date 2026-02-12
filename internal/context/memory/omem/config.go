package omem

import (
	"fmt"
	"runtime"
	"time"
)

// FactCategory represents the type of fact being stored.
type FactCategory string

const (
	CategoryPreference   FactCategory = "preference"
	CategoryBelief       FactCategory = "belief"
	CategoryBiographical FactCategory = "biographical"
	CategoryEvent        FactCategory = "event"
	CategoryRelationship FactCategory = "relationship"
	CategoryTask         FactCategory = "task"
	CategoryKnowledge    FactCategory = "knowledge"
	CategoryOther        FactCategory = "other"
)

// ValidCategories returns all valid fact categories.
func ValidCategories() []FactCategory {
	return []FactCategory{
		CategoryPreference,
		CategoryBelief,
		CategoryBiographical,
		CategoryEvent,
		CategoryRelationship,
		CategoryTask,
		CategoryKnowledge,
		CategoryOther,
	}
}

// EntityType represents the type of entity in the knowledge graph.
type EntityType string

const (
	EntityPerson       EntityType = "person"
	EntityPlace        EntityType = "place"
	EntityOrganization EntityType = "organization"
	EntityConcept      EntityType = "concept"
	EntityThing        EntityType = "thing"
	EntityTime         EntityType = "time"
	EntityOther        EntityType = "other"
)

// Config holds all configuration for the Omem memory system.
type Config struct {
	// Enabled toggles the entire Omem system
	Enabled bool `yaml:"enabled"`

	// Storage configuration
	Storage StorageConfig `yaml:"storage"`

	// Embedding configuration (fully configurable)
	Embedding EmbeddingConfig `yaml:"embedding"`

	// AtomicEncoder configuration (SimpleMem-inspired)
	AtomicEncoder AtomicEncoderConfig `yaml:"atomic_encoder"`

	// MultiViewIndex configuration (hybrid indexing)
	MultiViewIndex MultiViewConfig `yaml:"multi_view_index"`

	// EntityGraph configuration (lightweight HippoRAG-inspired)
	EntityGraph EntityGraphConfig `yaml:"entity_graph"`

	// Retrieval configuration (complexity-aware adaptive)
	Retrieval RetrievalConfig `yaml:"retrieval"`

	// Episodes configuration (Zep-inspired session tracking)
	Episodes EpisodeConfig `yaml:"episodes"`

	// Summary configuration (rolling summary)
	Summary SummaryConfig `yaml:"summary"`

	// Parallel processing configuration
	Parallel ParallelConfig `yaml:"parallel"`
}

// StorageConfig configures the underlying DuckDB storage engine.
type StorageConfig struct {
	// DBPath is the path to the DuckDB database file
	DBPath string `yaml:"db_path"`

	// MaxFacts is the maximum number of facts to retain (0 = unlimited)
	MaxFacts int `yaml:"max_facts"`

	// PruneThreshold triggers pruning when facts exceed this count
	PruneThreshold int `yaml:"prune_threshold"`

	// PruneKeepRecent keeps this many recent facts when pruning
	PruneKeepRecent int `yaml:"prune_keep_recent"`

	// EnableFTS enables DuckDB Full-Text Search extension for BM25
	EnableFTS bool `yaml:"enable_fts"`
}

// EmbeddingConfig configures the embedding model (fully configurable).
type EmbeddingConfig struct {
	// Provider specifies the embedding provider ("local", "openai", etc.)
	Provider string `yaml:"provider"`

	// Model is the model name/path (user configurable)
	// Examples: "all-MiniLM-L6-v2", "gte-small", "bge-small-en-v1.5"
	Model string `yaml:"model"`

	// Dimension is the embedding vector dimension (auto-detected if 0)
	Dimension int `yaml:"dimension"`

	// BatchSize for batch embedding operations
	BatchSize int `yaml:"batch_size"`

	// CacheSize for LRU embedding cache (0 = disabled)
	CacheSize int `yaml:"cache_size"`

	// Normalize vectors to unit length
	Normalize bool `yaml:"normalize"`
}

// AtomicEncoderConfig configures the SimpleMem-inspired atomic encoder.
type AtomicEncoderConfig struct {
	// Enabled toggles atomic encoding
	Enabled bool `yaml:"enabled"`

	// EnableCoreference resolves pronouns to actual entity names
	// "he/she/it/they" -> "John Smith"
	EnableCoreference bool `yaml:"enable_coreference"`

	// EnableTemporal anchors relative times to absolute dates
	// "yesterday" -> "2026-01-15"
	EnableTemporal bool `yaml:"enable_temporal"`

	// MaxFactsPerTurn limits facts extracted per conversation turn
	MaxFactsPerTurn int `yaml:"max_facts_per_turn"`

	// MinFactImportance threshold for storage (0.0-1.0)
	MinFactImportance float64 `yaml:"min_fact_importance"`

	// MinTextLength is minimum text length to process
	MinTextLength int `yaml:"min_text_length"`

	// UseLLMForComplex uses LLM for complex disambiguation (vs pure rules)
	UseLLMForComplex bool `yaml:"use_llm_for_complex"`
}

// MultiViewConfig configures the hybrid multi-view indexing.
type MultiViewConfig struct {
	// Enabled toggles multi-view indexing
	Enabled bool `yaml:"enabled"`

	// SemanticWeight for dense vector similarity (0.0-1.0)
	SemanticWeight float64 `yaml:"semantic_weight"`

	// LexicalWeight for BM25 keyword matching (0.0-1.0)
	LexicalWeight float64 `yaml:"lexical_weight"`

	// SymbolicWeight for metadata filters (0.0-1.0)
	SymbolicWeight float64 `yaml:"symbolic_weight"`

	// BM25_K1 term frequency saturation parameter
	BM25_K1 float64 `yaml:"bm25_k1"`

	// BM25_B length normalization parameter
	BM25_B float64 `yaml:"bm25_b"`

	// ExtractKeywords enables keyword extraction for BM25
	ExtractKeywords bool `yaml:"extract_keywords"`

	// MaxKeywordsPerFact limits keywords per fact
	MaxKeywordsPerFact int `yaml:"max_keywords_per_fact"`
}

// EntityGraphConfig configures the lightweight entity-relationship graph.
type EntityGraphConfig struct {
	// Enabled toggles entity graph
	Enabled bool `yaml:"enabled"`

	// MaxHops for graph traversal (1 recommended for SLMs)
	MaxHops int `yaml:"max_hops"`

	// EntityResolution merges similar entities
	EntityResolution bool `yaml:"entity_resolution"`

	// SimilarityThreshold for entity resolution
	SimilarityThreshold float64 `yaml:"similarity_threshold"`

	// UseRegexExtraction uses regex+heuristics vs LLM for entity extraction
	UseRegexExtraction bool `yaml:"use_regex_extraction"`

	// GraphBoostWeight for retrieval scoring
	GraphBoostWeight float64 `yaml:"graph_boost_weight"`
}

// RetrievalConfig configures the complexity-aware adaptive retriever.
type RetrievalConfig struct {
	// DefaultTopK is the base number of results to retrieve
	DefaultTopK int `yaml:"default_top_k"`

	// ComplexityDelta scales retrieval depth with query complexity
	// k_dyn = k_base * (1 + delta * complexity)
	ComplexityDelta float64 `yaml:"complexity_delta"`

	// MaxTopK caps the maximum retrieval depth
	MaxTopK int `yaml:"max_top_k"`

	// MinScore is the minimum score threshold for results
	MinScore float64 `yaml:"min_score"`

	// MaxContextTokens limits total tokens in retrieved context
	MaxContextTokens int `yaml:"max_context_tokens"`

	// RecencyHalfLifeHours for recency decay calculation
	RecencyHalfLifeHours float64 `yaml:"recency_half_life_hours"`

	// ImportanceWeight in final scoring
	ImportanceWeight float64 `yaml:"importance_weight"`

	// RecencyWeight in final scoring
	RecencyWeight float64 `yaml:"recency_weight"`

	// AccessFrequencyWeight in final scoring
	AccessFrequencyWeight float64 `yaml:"access_frequency_weight"`

	// EnableComplexityEstimation enables adaptive depth
	EnableComplexityEstimation bool `yaml:"enable_complexity_estimation"`
}

// EpisodeConfig configures Zep-inspired session/episode tracking.
type EpisodeConfig struct {
	// Enabled toggles episode management
	Enabled bool `yaml:"enabled"`

	// SessionTimeout starts new episode after this duration of inactivity
	SessionTimeout time.Duration `yaml:"session_timeout"`

	// SummaryOnClose generates summary when session ends
	SummaryOnClose bool `yaml:"summary_on_close"`

	// MaxEpisodesInCache keeps recent episodes in memory
	MaxEpisodesInCache int `yaml:"max_episodes_in_cache"`

	// TrackEntityMentions records entities per episode
	TrackEntityMentions bool `yaml:"track_entity_mentions"`
}

// SummaryConfig configures the rolling summary system.
type SummaryConfig struct {
	// Enabled toggles rolling summary
	Enabled bool `yaml:"enabled"`

	// RefreshInterval for automatic summary updates
	RefreshInterval time.Duration `yaml:"refresh_interval"`

	// MaxFacts to include in summary generation
	MaxFacts int `yaml:"max_facts"`

	// MaxTokens for the generated summary
	MaxTokens int `yaml:"max_tokens"`

	// Async refreshes summary in background
	Async bool `yaml:"async"`

	// IncrementalUpdate uses delta-based updates vs full regeneration
	IncrementalUpdate bool `yaml:"incremental_update"`

	// MinNewFactsForUpdate threshold for incremental updates
	MinNewFactsForUpdate int `yaml:"min_new_facts_for_update"`
}

// ParallelConfig configures parallel processing.
type ParallelConfig struct {
	// MaxWorkers for goroutine pool (0 = NumCPU)
	MaxWorkers int `yaml:"max_workers"`

	// BatchSize for batch operations
	BatchSize int `yaml:"batch_size"`

	// QueueSize for async processing queue
	QueueSize int `yaml:"queue_size"`

	// EnableAsync enables background processing
	EnableAsync bool `yaml:"enable_async"`
}

// DefaultConfig returns a Config with sensible defaults optimized for SLMs.
func DefaultConfig() Config {
	return Config{
		Enabled: true,

		Storage: StorageConfig{
			DBPath:          "openeye_omem.duckdb",
			MaxFacts:        10000,
			PruneThreshold:  12000,
			PruneKeepRecent: 5000,
			EnableFTS:       true,
		},

		Embedding: EmbeddingConfig{
			Provider:  "local",
			Model:     "all-MiniLM-L6-v2", // 22MB, good default
			Dimension: 384,
			BatchSize: 32,
			CacheSize: 1000,
			Normalize: true,
		},

		AtomicEncoder: AtomicEncoderConfig{
			Enabled:           true,
			EnableCoreference: true,
			EnableTemporal:    true,
			MaxFactsPerTurn:   10,
			MinFactImportance: 0.3,
			MinTextLength:     10,
			UseLLMForComplex:  true,
		},

		MultiViewIndex: MultiViewConfig{
			Enabled:            true,
			SemanticWeight:     0.5,
			LexicalWeight:      0.3,
			SymbolicWeight:     0.2,
			BM25_K1:            1.2,
			BM25_B:             0.75,
			ExtractKeywords:    true,
			MaxKeywordsPerFact: 20,
		},

		EntityGraph: EntityGraphConfig{
			Enabled:             true,
			MaxHops:             1, // Keep it light for SLMs
			EntityResolution:    true,
			SimilarityThreshold: 0.85,
			UseRegexExtraction:  true, // No LLM for entity extraction
			GraphBoostWeight:    0.2,
		},

		Retrieval: RetrievalConfig{
			DefaultTopK:                5,
			ComplexityDelta:            2.0,
			MaxTopK:                    20,
			MinScore:                   -1.0, // -1 means "unset", will default to 0.0 (no filtering)
			MaxContextTokens:           1000,
			RecencyHalfLifeHours:       168, // 1 week
			ImportanceWeight:           0.15,
			RecencyWeight:              0.1,
			AccessFrequencyWeight:      0.05,
			EnableComplexityEstimation: true,
		},

		Episodes: EpisodeConfig{
			Enabled:             true,
			SessionTimeout:      30 * time.Minute,
			SummaryOnClose:      true,
			MaxEpisodesInCache:  10,
			TrackEntityMentions: true,
		},

		Summary: SummaryConfig{
			Enabled:              true,
			RefreshInterval:      5 * time.Minute,
			MaxFacts:             50,
			MaxTokens:            512,
			Async:                true,
			IncrementalUpdate:    true,
			MinNewFactsForUpdate: 5,
		},

		Parallel: ParallelConfig{
			MaxWorkers:  0, // Will be set to NumCPU
			BatchSize:   10,
			QueueSize:   100,
			EnableAsync: true,
		},
	}
}

// Validate checks the configuration for errors and applies fixes.
func (c *Config) Validate() error {
	// Normalize multi-view weights to sum to 1.0
	c.normalizeMultiViewWeights()

	// Apply sensible minimums
	if c.Embedding.Dimension <= 0 {
		c.Embedding.Dimension = 384
	}
	if c.Embedding.BatchSize <= 0 {
		c.Embedding.BatchSize = 32
	}
	if c.AtomicEncoder.MaxFactsPerTurn <= 0 {
		c.AtomicEncoder.MaxFactsPerTurn = 10
	}
	if c.EntityGraph.MaxHops <= 0 {
		c.EntityGraph.MaxHops = 1
	}
	if c.EntityGraph.MaxHops > 3 {
		c.EntityGraph.MaxHops = 3 // Cap for SLMs
	}
	if c.Retrieval.DefaultTopK <= 0 {
		c.Retrieval.DefaultTopK = 5
	}
	if c.Retrieval.MaxTopK <= 0 {
		c.Retrieval.MaxTopK = 20
	}
	if c.Retrieval.MaxContextTokens <= 0 {
		c.Retrieval.MaxContextTokens = 1000
	}
	if c.Summary.MaxFacts <= 0 {
		c.Summary.MaxFacts = 50
	}
	if c.Summary.MaxTokens <= 0 {
		c.Summary.MaxTokens = 512
	}
	if c.Parallel.MaxWorkers <= 0 {
		c.Parallel.MaxWorkers = runtime.NumCPU()
	}
	if c.Parallel.BatchSize <= 0 {
		c.Parallel.BatchSize = 10
	}

	// Validate BM25 parameters
	if c.MultiViewIndex.BM25_K1 <= 0 {
		c.MultiViewIndex.BM25_K1 = 1.2
	}
	if c.MultiViewIndex.BM25_B <= 0 || c.MultiViewIndex.BM25_B > 1 {
		c.MultiViewIndex.BM25_B = 0.75
	}

	return nil
}

// normalizeMultiViewWeights ensures multi-view weights sum to 1.0.
func (c *Config) normalizeMultiViewWeights() {
	sum := c.MultiViewIndex.SemanticWeight +
		c.MultiViewIndex.LexicalWeight +
		c.MultiViewIndex.SymbolicWeight

	if sum <= 0 {
		// Reset to defaults
		c.MultiViewIndex.SemanticWeight = 0.5
		c.MultiViewIndex.LexicalWeight = 0.3
		c.MultiViewIndex.SymbolicWeight = 0.2
		return
	}

	// Normalize
	c.MultiViewIndex.SemanticWeight /= sum
	c.MultiViewIndex.LexicalWeight /= sum
	c.MultiViewIndex.SymbolicWeight /= sum
}

// WithDefaults fills in zero values with defaults.
func (c Config) WithDefaults() Config {
	defaults := DefaultConfig()

	// Storage
	if c.Storage.DBPath == "" {
		c.Storage.DBPath = defaults.Storage.DBPath
	}
	if c.Storage.MaxFacts == 0 {
		c.Storage.MaxFacts = defaults.Storage.MaxFacts
	}
	if c.Storage.PruneThreshold == 0 {
		c.Storage.PruneThreshold = defaults.Storage.PruneThreshold
	}

	// Embedding
	if c.Embedding.Model == "" {
		c.Embedding.Model = defaults.Embedding.Model
	}
	if c.Embedding.Provider == "" {
		c.Embedding.Provider = defaults.Embedding.Provider
	}
	if c.Embedding.Dimension == 0 {
		c.Embedding.Dimension = defaults.Embedding.Dimension
	}
	if c.Embedding.BatchSize == 0 {
		c.Embedding.BatchSize = defaults.Embedding.BatchSize
	}

	// AtomicEncoder
	if c.AtomicEncoder.MaxFactsPerTurn == 0 {
		c.AtomicEncoder.MaxFactsPerTurn = defaults.AtomicEncoder.MaxFactsPerTurn
	}

	// MultiViewIndex
	if c.MultiViewIndex.SemanticWeight == 0 && c.MultiViewIndex.LexicalWeight == 0 {
		c.MultiViewIndex.SemanticWeight = defaults.MultiViewIndex.SemanticWeight
		c.MultiViewIndex.LexicalWeight = defaults.MultiViewIndex.LexicalWeight
		c.MultiViewIndex.SymbolicWeight = defaults.MultiViewIndex.SymbolicWeight
	}
	if c.MultiViewIndex.BM25_K1 == 0 {
		c.MultiViewIndex.BM25_K1 = defaults.MultiViewIndex.BM25_K1
	}
	if c.MultiViewIndex.BM25_B == 0 {
		c.MultiViewIndex.BM25_B = defaults.MultiViewIndex.BM25_B
	}
	if c.MultiViewIndex.MaxKeywordsPerFact == 0 {
		c.MultiViewIndex.MaxKeywordsPerFact = defaults.MultiViewIndex.MaxKeywordsPerFact
	}

	// EntityGraph
	if c.EntityGraph.MaxHops == 0 {
		c.EntityGraph.MaxHops = defaults.EntityGraph.MaxHops
	}
	if c.EntityGraph.SimilarityThreshold == 0 {
		c.EntityGraph.SimilarityThreshold = defaults.EntityGraph.SimilarityThreshold
	}
	if c.EntityGraph.GraphBoostWeight == 0 {
		c.EntityGraph.GraphBoostWeight = defaults.EntityGraph.GraphBoostWeight
	}

	// Retrieval
	if c.Retrieval.DefaultTopK == 0 {
		c.Retrieval.DefaultTopK = defaults.Retrieval.DefaultTopK
	}
	if c.Retrieval.ComplexityDelta == 0 {
		c.Retrieval.ComplexityDelta = defaults.Retrieval.ComplexityDelta
	}
	if c.Retrieval.MaxTopK == 0 {
		c.Retrieval.MaxTopK = defaults.Retrieval.MaxTopK
	}
	if c.Retrieval.MaxContextTokens == 0 {
		c.Retrieval.MaxContextTokens = defaults.Retrieval.MaxContextTokens
	}
	if c.Retrieval.RecencyHalfLifeHours == 0 {
		c.Retrieval.RecencyHalfLifeHours = defaults.Retrieval.RecencyHalfLifeHours
	}

	// Episodes
	if c.Episodes.SessionTimeout == 0 {
		c.Episodes.SessionTimeout = defaults.Episodes.SessionTimeout
	}
	if c.Episodes.MaxEpisodesInCache == 0 {
		c.Episodes.MaxEpisodesInCache = defaults.Episodes.MaxEpisodesInCache
	}

	// Summary
	if c.Summary.RefreshInterval == 0 {
		c.Summary.RefreshInterval = defaults.Summary.RefreshInterval
	}
	if c.Summary.MaxFacts == 0 {
		c.Summary.MaxFacts = defaults.Summary.MaxFacts
	}
	if c.Summary.MaxTokens == 0 {
		c.Summary.MaxTokens = defaults.Summary.MaxTokens
	}
	if c.Summary.MinNewFactsForUpdate == 0 {
		c.Summary.MinNewFactsForUpdate = defaults.Summary.MinNewFactsForUpdate
	}

	// Parallel
	if c.Parallel.MaxWorkers == 0 {
		c.Parallel.MaxWorkers = defaults.Parallel.MaxWorkers
	}
	if c.Parallel.BatchSize == 0 {
		c.Parallel.BatchSize = defaults.Parallel.BatchSize
	}
	if c.Parallel.QueueSize == 0 {
		c.Parallel.QueueSize = defaults.Parallel.QueueSize
	}

	return c
}

// String returns a human-readable configuration summary.
func (c *Config) String() string {
	return fmt.Sprintf(
		"Omem Config: enabled=%t, storage=%s, embedding=%s/%s (dim=%d), "+
			"atomic_encoder=%t, multi_view=%t (sem=%.2f/lex=%.2f/sym=%.2f), "+
			"entity_graph=%t (hops=%d), episodes=%t, summary=%t",
		c.Enabled,
		c.Storage.DBPath,
		c.Embedding.Provider, c.Embedding.Model, c.Embedding.Dimension,
		c.AtomicEncoder.Enabled,
		c.MultiViewIndex.Enabled,
		c.MultiViewIndex.SemanticWeight,
		c.MultiViewIndex.LexicalWeight,
		c.MultiViewIndex.SymbolicWeight,
		c.EntityGraph.Enabled, c.EntityGraph.MaxHops,
		c.Episodes.Enabled,
		c.Summary.Enabled,
	)
}

// ============================================================================
// ABLATION STUDY PRESETS
// These configurations are used for research benchmarking to isolate the
// contribution of each Omem component.
// ============================================================================

// AblationPreset identifies a specific ablation configuration.
type AblationPreset string

const (
	// AblationFull enables all Omem features (baseline)
	AblationFull AblationPreset = "full"

	// AblationNoAtomicEncoder disables atomic encoding (raw text storage)
	AblationNoAtomicEncoder AblationPreset = "no_atomic_encoder"

	// AblationSemanticOnly uses only vector similarity (no BM25, no graph)
	AblationSemanticOnly AblationPreset = "semantic_only"

	// AblationNoGraph disables entity graph
	AblationNoGraph AblationPreset = "no_graph"

	// AblationNoSummary disables rolling summary
	AblationNoSummary AblationPreset = "no_summary"

	// AblationFixedK disables complexity estimation (fixed top-K)
	AblationFixedK AblationPreset = "fixed_k"

	// AblationNoEpisodes disables session/episode tracking
	AblationNoEpisodes AblationPreset = "no_episodes"

	// AblationMinimal disables all advanced features (vector-only baseline)
	AblationMinimal AblationPreset = "minimal"
)

// AblationConfig returns a Config modified for the specified ablation preset.
// This is used for research benchmarking to measure the contribution of each component.
func AblationConfig(preset AblationPreset) Config {
	base := DefaultConfig()

	switch preset {
	case AblationFull:
		// All features enabled (default)
		return base

	case AblationNoAtomicEncoder:
		// Disable atomic encoding - store raw text without denoising
		base.AtomicEncoder.Enabled = false
		base.AtomicEncoder.EnableCoreference = false
		base.AtomicEncoder.EnableTemporal = false
		return base

	case AblationSemanticOnly:
		// Vector similarity only - no BM25, no graph
		base.MultiViewIndex.SemanticWeight = 1.0
		base.MultiViewIndex.LexicalWeight = 0.0
		base.MultiViewIndex.SymbolicWeight = 0.0
		base.EntityGraph.Enabled = false
		base.Storage.EnableFTS = false
		return base

	case AblationNoGraph:
		// Disable entity graph
		base.EntityGraph.Enabled = false
		base.MultiViewIndex.SymbolicWeight = 0.0
		// Re-normalize weights
		base.MultiViewIndex.SemanticWeight = 0.625 // 0.5 / (0.5+0.3)
		base.MultiViewIndex.LexicalWeight = 0.375  // 0.3 / (0.5+0.3)
		return base

	case AblationNoSummary:
		// Disable rolling summary
		base.Summary.Enabled = false
		return base

	case AblationFixedK:
		// Disable complexity estimation - fixed top-K
		base.Retrieval.EnableComplexityEstimation = false
		base.Retrieval.DefaultTopK = 10 // Fixed at 10
		base.Retrieval.ComplexityDelta = 0.0
		return base

	case AblationNoEpisodes:
		// Disable episode/session tracking
		base.Episodes.Enabled = false
		return base

	case AblationMinimal:
		// Minimal configuration - vector-only baseline
		base.AtomicEncoder.Enabled = false
		base.MultiViewIndex.SemanticWeight = 1.0
		base.MultiViewIndex.LexicalWeight = 0.0
		base.MultiViewIndex.SymbolicWeight = 0.0
		base.EntityGraph.Enabled = false
		base.Summary.Enabled = false
		base.Episodes.Enabled = false
		base.Retrieval.EnableComplexityEstimation = false
		base.Storage.EnableFTS = false
		return base

	default:
		return base
	}
}

// GetAblationPresets returns all available ablation presets.
func GetAblationPresets() []AblationPreset {
	return []AblationPreset{
		AblationFull,
		AblationNoAtomicEncoder,
		AblationSemanticOnly,
		AblationNoGraph,
		AblationNoSummary,
		AblationFixedK,
		AblationNoEpisodes,
		AblationMinimal,
	}
}

// AblationDescription returns a human-readable description of an ablation preset.
func AblationDescription(preset AblationPreset) string {
	descriptions := map[AblationPreset]string{
		AblationFull:            "Full Omem: All features enabled (baseline)",
		AblationNoAtomicEncoder: "No Atomic Encoder: Raw text storage without coreference/temporal resolution",
		AblationSemanticOnly:    "Semantic Only: Vector similarity only (no BM25, no entity graph)",
		AblationNoGraph:         "No Graph: Disable entity relationship graph",
		AblationNoSummary:       "No Summary: Disable rolling user biography summary",
		AblationFixedK:          "Fixed K: Disable complexity estimation, use fixed top-K=10",
		AblationNoEpisodes:      "No Episodes: Disable session/episode tracking",
		AblationMinimal:         "Minimal: Vector-only baseline (all advanced features disabled)",
	}
	if desc, ok := descriptions[preset]; ok {
		return desc
	}
	return string(preset)
}

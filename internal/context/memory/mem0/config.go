// Package mem0 implements a mem0-style long-term memory system for OpenEye.
// It provides intelligent fact extraction, memory updates (ADD/UPDATE/DELETE/NOOP),
// entity-relationship graph storage, and hybrid retrieval optimized for SLMs.
package mem0

import (
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
		CategoryOther,
	}
}

// EntityType represents the type of entity in the knowledge graph.
type EntityType string

const (
	EntityPerson       EntityType = "person"
	EntityPlace        EntityType = "place"
	EntityThing        EntityType = "thing"
	EntityConcept      EntityType = "concept"
	EntityOrganization EntityType = "organization"
	EntityTime         EntityType = "time"
	EntityOther        EntityType = "other"
)

// UpdateOperation represents the type of memory update operation.
type UpdateOperation string

const (
	OpAdd    UpdateOperation = "ADD"
	OpUpdate UpdateOperation = "UPDATE"
	OpDelete UpdateOperation = "DELETE"
	OpNoop   UpdateOperation = "NOOP"
)

// Config holds all configuration for the mem0 memory system.
type Config struct {
	// Storage configuration
	Storage StorageConfig `yaml:"storage"`

	// Fact extraction configuration
	Extraction ExtractionConfig `yaml:"extraction"`

	// Memory update configuration
	Updates UpdateConfig `yaml:"updates"`

	// Graph memory configuration
	Graph GraphConfig `yaml:"graph"`

	// Retrieval configuration
	Retrieval RetrievalConfig `yaml:"retrieval"`

	// Rolling summary configuration
	Summary SummaryConfig `yaml:"summary"`

	// Enabled toggles the entire mem0 system
	Enabled bool `yaml:"enabled"`
}

// StorageConfig configures the underlying storage engine.
type StorageConfig struct {
	// DBPath is the path to the DuckDB database file
	DBPath string `yaml:"db_path"`

	// EmbeddingDim is the dimension of embedding vectors
	EmbeddingDim int `yaml:"embedding_dim"`

	// MaxFacts is the maximum number of facts to retain (0 = unlimited)
	MaxFacts int `yaml:"max_facts"`

	// PruneThreshold triggers pruning when facts exceed this count
	PruneThreshold int `yaml:"prune_threshold"`

	// PruneKeepRecent keeps this many recent facts when pruning
	PruneKeepRecent int `yaml:"prune_keep_recent"`
}

// ExtractionConfig configures fact extraction from conversations.
type ExtractionConfig struct {
	// Enabled toggles fact extraction
	Enabled bool `yaml:"enabled"`

	// BatchSize is the number of conversation turns to process together
	BatchSize int `yaml:"batch_size"`

	// Async runs extraction in background goroutine
	Async bool `yaml:"async"`

	// MinTextLength is the minimum text length to process
	MinTextLength int `yaml:"min_text_length"`

	// MaxFactsPerExtraction limits facts extracted per batch
	MaxFactsPerExtraction int `yaml:"max_facts_per_extraction"`

	// ExtractEntities enables entity extraction
	ExtractEntities bool `yaml:"extract_entities"`

	// ExtractRelationships enables relationship extraction
	ExtractRelationships bool `yaml:"extract_relationships"`
}

// UpdateConfig configures memory update operations.
type UpdateConfig struct {
	// Enabled toggles intelligent updates (ADD/UPDATE/DELETE/NOOP)
	Enabled bool `yaml:"enabled"`

	// ConflictThreshold is the similarity threshold for conflict detection
	ConflictThreshold float64 `yaml:"conflict_threshold"`

	// TopSimilarCount is the number of similar facts to compare
	TopSimilarCount int `yaml:"top_similar_count"`

	// AutoResolveConflicts automatically resolves contradictions
	AutoResolveConflicts bool `yaml:"auto_resolve_conflicts"`

	// TrackSupersession tracks which facts replace others
	TrackSupersession bool `yaml:"track_supersession"`
}

// GraphConfig configures entity-relationship graph storage.
type GraphConfig struct {
	// Enabled toggles graph memory
	Enabled bool `yaml:"enabled"`

	// EntityResolution enables merging similar entities
	EntityResolution bool `yaml:"entity_resolution"`

	// EntitySimilarityThreshold for entity resolution
	EntitySimilarityThreshold float64 `yaml:"entity_similarity_threshold"`

	// MaxHops is the maximum graph traversal depth
	MaxHops int `yaml:"max_hops"`

	// TrackRelationshipHistory keeps history of relationship changes
	TrackRelationshipHistory bool `yaml:"track_relationship_history"`
}

// RetrievalConfig configures the hybrid retrieval engine.
type RetrievalConfig struct {
	// SemanticWeight is the weight for semantic similarity (0-1)
	SemanticWeight float64 `yaml:"semantic_weight"`

	// ImportanceWeight is the weight for fact importance (0-1)
	ImportanceWeight float64 `yaml:"importance_weight"`

	// RecencyWeight is the weight for recency (0-1)
	RecencyWeight float64 `yaml:"recency_weight"`

	// AccessFrequencyWeight is the weight for access frequency (0-1)
	AccessFrequencyWeight float64 `yaml:"access_frequency_weight"`

	// MinScore is the minimum score threshold for results
	MinScore float64 `yaml:"min_score"`

	// MaxResults is the maximum number of results to return
	MaxResults int `yaml:"max_results"`

	// IncludeGraphResults includes graph traversal results
	IncludeGraphResults bool `yaml:"include_graph_results"`

	// GraphResultsWeight is the weight for graph-based results
	GraphResultsWeight float64 `yaml:"graph_results_weight"`

	// RecencyHalfLifeHours is the half-life for recency decay
	RecencyHalfLifeHours float64 `yaml:"recency_half_life_hours"`
}

// SummaryConfig configures the rolling summary system.
type SummaryConfig struct {
	// Enabled toggles rolling summary
	Enabled bool `yaml:"enabled"`

	// RefreshInterval is how often to refresh the summary
	RefreshInterval time.Duration `yaml:"refresh_interval"`

	// MaxFacts is the maximum facts to include in summary
	MaxFacts int `yaml:"max_facts"`

	// MaxTokens is the maximum tokens for the summary
	MaxTokens int `yaml:"max_tokens"`

	// Async refreshes summary in background
	Async bool `yaml:"async"`
}

// DefaultConfig returns a Config with sensible defaults optimized for SLMs.
func DefaultConfig() Config {
	return Config{
		Enabled: true,
		Storage: StorageConfig{
			DBPath:          "openeye_mem0.duckdb",
			EmbeddingDim:    384,
			MaxFacts:        10000,
			PruneThreshold:  12000,
			PruneKeepRecent: 5000,
		},
		Extraction: ExtractionConfig{
			Enabled:               true,
			BatchSize:             3,
			Async:                 true,
			MinTextLength:         10,
			MaxFactsPerExtraction: 10,
			ExtractEntities:       true,
			ExtractRelationships:  true,
		},
		Updates: UpdateConfig{
			Enabled:              true,
			ConflictThreshold:    0.85,
			TopSimilarCount:      5,
			AutoResolveConflicts: true,
			TrackSupersession:    true,
		},
		Graph: GraphConfig{
			Enabled:                   true,
			EntityResolution:          true,
			EntitySimilarityThreshold: 0.9,
			MaxHops:                   2,
			TrackRelationshipHistory:  false,
		},
		Retrieval: RetrievalConfig{
			SemanticWeight:        0.50,
			ImportanceWeight:      0.25,
			RecencyWeight:         0.15,
			AccessFrequencyWeight: 0.10,
			MinScore:              0.3,
			MaxResults:            20,
			IncludeGraphResults:   true,
			GraphResultsWeight:    0.3,
			RecencyHalfLifeHours:  168, // 1 week
		},
		Summary: SummaryConfig{
			Enabled:         true,
			RefreshInterval: 5 * time.Minute,
			MaxFacts:        50,
			MaxTokens:       512,
			Async:           true,
		},
	}
}

// Validate checks the configuration for errors.
func (c *Config) Validate() error {
	// Normalize weights to sum to 1.0
	c.normalizeRetrievalWeights()

	// Apply sensible minimums
	if c.Storage.EmbeddingDim <= 0 {
		c.Storage.EmbeddingDim = 384
	}
	if c.Extraction.BatchSize <= 0 {
		c.Extraction.BatchSize = 3
	}
	if c.Updates.TopSimilarCount <= 0 {
		c.Updates.TopSimilarCount = 5
	}
	if c.Graph.MaxHops <= 0 {
		c.Graph.MaxHops = 2
	}
	if c.Retrieval.MaxResults <= 0 {
		c.Retrieval.MaxResults = 20
	}
	if c.Summary.MaxFacts <= 0 {
		c.Summary.MaxFacts = 50
	}

	return nil
}

// normalizeRetrievalWeights ensures retrieval weights sum to 1.0.
func (c *Config) normalizeRetrievalWeights() {
	sum := c.Retrieval.SemanticWeight +
		c.Retrieval.ImportanceWeight +
		c.Retrieval.RecencyWeight +
		c.Retrieval.AccessFrequencyWeight

	if sum <= 0 {
		// Reset to defaults
		c.Retrieval.SemanticWeight = 0.50
		c.Retrieval.ImportanceWeight = 0.25
		c.Retrieval.RecencyWeight = 0.15
		c.Retrieval.AccessFrequencyWeight = 0.10
		return
	}

	// Normalize
	c.Retrieval.SemanticWeight /= sum
	c.Retrieval.ImportanceWeight /= sum
	c.Retrieval.RecencyWeight /= sum
	c.Retrieval.AccessFrequencyWeight /= sum
}

// WithDefaults fills in zero values with defaults.
func (c Config) WithDefaults() Config {
	defaults := DefaultConfig()

	if c.Storage.DBPath == "" {
		c.Storage.DBPath = defaults.Storage.DBPath
	}
	if c.Storage.EmbeddingDim == 0 {
		c.Storage.EmbeddingDim = defaults.Storage.EmbeddingDim
	}
	if c.Extraction.BatchSize == 0 {
		c.Extraction.BatchSize = defaults.Extraction.BatchSize
	}
	if c.Updates.ConflictThreshold == 0 {
		c.Updates.ConflictThreshold = defaults.Updates.ConflictThreshold
	}
	if c.Updates.TopSimilarCount == 0 {
		c.Updates.TopSimilarCount = defaults.Updates.TopSimilarCount
	}
	if c.Graph.MaxHops == 0 {
		c.Graph.MaxHops = defaults.Graph.MaxHops
	}
	if c.Retrieval.MaxResults == 0 {
		c.Retrieval.MaxResults = defaults.Retrieval.MaxResults
	}
	if c.Retrieval.RecencyHalfLifeHours == 0 {
		c.Retrieval.RecencyHalfLifeHours = defaults.Retrieval.RecencyHalfLifeHours
	}
	if c.Summary.RefreshInterval == 0 {
		c.Summary.RefreshInterval = defaults.Summary.RefreshInterval
	}
	if c.Summary.MaxFacts == 0 {
		c.Summary.MaxFacts = defaults.Summary.MaxFacts
	}
	if c.Summary.MaxTokens == 0 {
		c.Summary.MaxTokens = defaults.Summary.MaxTokens
	}

	return c
}

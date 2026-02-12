package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config captures runtime, memory, server and conversation settings for OpenEye.
type Config struct {
	Runtime      RuntimeConfig      `yaml:"runtime"`
	Memory       MemoryConfig       `yaml:"memory"`
	Server       ServerConfig       `yaml:"server"`
	Conversation ConversationConfig `yaml:"conversation"`
	RAG          RAGConfig          `yaml:"rag"`
	Assistants   AssistantsConfig   `yaml:"assistants"`
	Embedding    EmbeddingConfig    `yaml:"embedding"`
	Image        ImageConfig        `yaml:"image"`
}

// RuntimeConfig selects which backend implementation to use and its settings.
type RuntimeConfig struct {
	Backend  string              `yaml:"backend"`
	HTTP     HTTPBackendConfig   `yaml:"http"`
	Native   NativeBackendConfig `yaml:"native"`
	Defaults GenerationDefaults  `yaml:"defaults"`
}

// GenerationDefaults allows overriding common inference parameters globally.
type GenerationDefaults struct {
	MaxTokens     int      `yaml:"max_tokens"`
	Temperature   float64  `yaml:"temperature"`
	TopK          int      `yaml:"top_k"`
	TopP          float64  `yaml:"top_p"`
	MinP          float64  `yaml:"min_p"`
	RepeatPenalty float64  `yaml:"repeat_penalty"`
	RepeatLastN   int      `yaml:"repeat_last_n"`
	Stop          []string `yaml:"stop"`
}

// NativeBackendConfig configures the native llama.cpp backend used when
// backend is set to "native" and the binary is compiled with -tags native.
type NativeBackendConfig struct {
	ModelPath      string `yaml:"model_path"`
	MmprojPath     string `yaml:"mmproj_path"` // Path to mmproj GGUF for vision/multimodal
	ContextSize    int    `yaml:"context_size"`
	BatchSize      int    `yaml:"batch_size"`
	Threads        int    `yaml:"threads"`
	ThreadsBatch   int    `yaml:"threads_batch"`
	GPULayers      int    `yaml:"gpu_layers"`
	Mmap           *bool  `yaml:"mmap"`
	Mlock          *bool  `yaml:"mlock"`
	FlashAttention *bool  `yaml:"flash_attention"`
	Warmup         bool   `yaml:"warmup"`
	WarmupTokens   int    `yaml:"warmup_tokens"` // 0=single BOS token, >0=multi-token warmup

	// --- Inference optimizations ---

	// DraftModelPath is the path to a smaller GGUF model for speculative
	// decoding. When set, the draft model generates candidate tokens that
	// the target model verifies in batch, yielding 1.5-2.5x generation
	// speedup on CPU. Use a model from the same family but much smaller
	// (e.g., SmolLM2-135M-Q4_K_M.gguf as draft for a 1.2B target).
	DraftModelPath string `yaml:"draft_model_path"`

	// SpeculativeN is the number of draft tokens to generate before
	// verification. Higher values amortize more overhead but risk more
	// rejections. Default: 5. Range: 2-8.
	SpeculativeN int `yaml:"speculative_n"`

	// KVCacheType controls quantization of the KV cache. Lower precision
	// reduces memory usage (allowing larger context or more headroom)
	// with minimal quality impact. Options: "f16" (default), "q8_0", "q4_0".
	KVCacheType string `yaml:"kv_cache_type"`

	// StreamChunkSize controls how many tokens are buffered before
	// emitting to the stream callback. Higher values produce smoother
	// word-level streaming. 0 or 1 = emit every token. Default: 3.
	StreamChunkSize int `yaml:"stream_chunk_size"`

	// ContextShift enables automatic sliding window when the KV cache
	// fills up. Instead of failing, the oldest portion of the context
	// is discarded and generation continues. Default: true.
	ContextShift *bool `yaml:"context_shift"`
}

// HTTPBackendConfig configures the default HTTP completion backend.
type HTTPBackendConfig struct {
	BaseURL string `yaml:"base_url"`
	Timeout string `yaml:"timeout"`
}

// MemoryConfig configures persistent conversation history storage and vector memory.
type MemoryConfig struct {
	Path       string `yaml:"path"`
	TurnsToUse int    `yaml:"turns_to_use"`

	// Vector memory settings (legacy)
	VectorEnabled      bool    `yaml:"vector_enabled"`
	VectorDBPath       string  `yaml:"vector_db_path"`
	EmbeddingDim       int     `yaml:"embedding_dim"`
	MaxContextTokens   int     `yaml:"max_context_tokens"`
	ReservedForPrompt  int     `yaml:"reserved_for_prompt"`
	ReservedForSummary int     `yaml:"reserved_for_summary"`
	MinSimilarity      float64 `yaml:"min_similarity"`
	SlidingWindowSize  int     `yaml:"sliding_window_size"`
	RecencyWeight      float64 `yaml:"recency_weight"`
	RelevanceWeight    float64 `yaml:"relevance_weight"`

	// Memory compression settings
	CompressionEnabled bool   `yaml:"compression_enabled"`
	CompressionAge     string `yaml:"compression_age"`
	CompressBatchSize  int    `yaml:"compress_batch_size"`
	AutoCompress       bool   `yaml:"auto_compress"`
	CompressEveryN     int    `yaml:"compress_every_n"`

	// Mem0-style long-term memory (legacy)
	Mem0 Mem0Config `yaml:"mem0"`

	// Omem advanced long-term memory (next-gen)
	Omem OmemConfig `yaml:"omem"`
}

// OmemConfig configures the Omem (Optimal Memory) advanced long-term memory system.
// Omem combines techniques from SimpleMem, HippoRAG, and Zep for efficient SLM memory.
type OmemConfig struct {
	Enabled *bool `yaml:"enabled"`

	// Storage configuration
	Storage OmemStorageConfig `yaml:"storage"`

	// Atomic encoding (SimpleMem-inspired)
	AtomicEncoder OmemEncoderConfig `yaml:"atomic_encoder"`

	// Multi-view indexing (semantic + lexical + symbolic)
	MultiView OmemMultiViewConfig `yaml:"multi_view"`

	// Entity graph (lightweight HippoRAG-inspired)
	EntityGraph OmemEntityGraphConfig `yaml:"entity_graph"`

	// Adaptive retrieval (complexity-aware)
	Retrieval OmemRetrievalConfig `yaml:"retrieval"`

	// Episode/session tracking (Zep-inspired)
	Episodes OmemEpisodesConfig `yaml:"episodes"`

	// Rolling summary
	Summary OmemSummaryConfig `yaml:"summary"`

	// Parallel processing
	Parallel OmemParallelConfig `yaml:"parallel"`
}

// OmemStorageConfig configures Omem DuckDB storage.
type OmemStorageConfig struct {
	DBPath          string `yaml:"db_path"`
	MaxFacts        int    `yaml:"max_facts"`
	PruneThreshold  int    `yaml:"prune_threshold"`
	PruneKeepRecent int    `yaml:"prune_keep_recent"`
	EnableFTS       bool   `yaml:"enable_fts"`
}

// OmemEncoderConfig configures atomic fact encoding.
type OmemEncoderConfig struct {
	Enabled           bool    `yaml:"enabled"`
	EnableCoreference bool    `yaml:"enable_coreference"`
	EnableTemporal    bool    `yaml:"enable_temporal"`
	MaxFactsPerTurn   int     `yaml:"max_facts_per_turn"`
	MinFactImportance float64 `yaml:"min_fact_importance"`
	MinTextLength     int     `yaml:"min_text_length"`
	UseLLMForComplex  bool    `yaml:"use_llm_for_complex"`
}

// OmemMultiViewConfig configures hybrid multi-view indexing.
type OmemMultiViewConfig struct {
	Enabled            bool    `yaml:"enabled"`
	SemanticWeight     float64 `yaml:"semantic_weight"`
	LexicalWeight      float64 `yaml:"lexical_weight"`
	SymbolicWeight     float64 `yaml:"symbolic_weight"`
	BM25_K1            float64 `yaml:"bm25_k1"`
	BM25_B             float64 `yaml:"bm25_b"`
	ExtractKeywords    bool    `yaml:"extract_keywords"`
	MaxKeywordsPerFact int     `yaml:"max_keywords_per_fact"`
}

// OmemEntityGraphConfig configures the lightweight entity-relationship graph.
type OmemEntityGraphConfig struct {
	Enabled             bool    `yaml:"enabled"`
	MaxHops             int     `yaml:"max_hops"`
	EntityResolution    bool    `yaml:"entity_resolution"`
	SimilarityThreshold float64 `yaml:"similarity_threshold"`
	UseRegexExtraction  bool    `yaml:"use_regex_extraction"`
	GraphBoostWeight    float64 `yaml:"graph_boost_weight"`
}

// OmemRetrievalConfig configures complexity-aware adaptive retrieval.
type OmemRetrievalConfig struct {
	DefaultTopK                int     `yaml:"default_top_k"`
	ComplexityDelta            float64 `yaml:"complexity_delta"`
	MaxTopK                    int     `yaml:"max_top_k"`
	MinScore                   float64 `yaml:"min_score"`
	MaxContextTokens           int     `yaml:"max_context_tokens"`
	RecencyHalfLifeHours       float64 `yaml:"recency_half_life_hours"`
	ImportanceWeight           float64 `yaml:"importance_weight"`
	RecencyWeight              float64 `yaml:"recency_weight"`
	AccessFrequencyWeight      float64 `yaml:"access_frequency_weight"`
	EnableComplexityEstimation bool    `yaml:"enable_complexity_estimation"`
}

// OmemEpisodesConfig configures Zep-inspired session/episode tracking.
type OmemEpisodesConfig struct {
	Enabled             bool   `yaml:"enabled"`
	SessionTimeout      string `yaml:"session_timeout"`
	SummaryOnClose      bool   `yaml:"summary_on_close"`
	MaxEpisodesInCache  int    `yaml:"max_episodes_in_cache"`
	TrackEntityMentions bool   `yaml:"track_entity_mentions"`
}

// OmemSummaryConfig configures rolling summary generation.
type OmemSummaryConfig struct {
	Enabled              bool   `yaml:"enabled"`
	RefreshInterval      string `yaml:"refresh_interval"`
	MaxFacts             int    `yaml:"max_facts"`
	MaxTokens            int    `yaml:"max_tokens"`
	Async                bool   `yaml:"async"`
	IncrementalUpdate    bool   `yaml:"incremental_update"`
	MinNewFactsForUpdate int    `yaml:"min_new_facts_for_update"`
}

// OmemParallelConfig configures parallel processing.
type OmemParallelConfig struct {
	MaxWorkers  int  `yaml:"max_workers"`
	BatchSize   int  `yaml:"batch_size"`
	QueueSize   int  `yaml:"queue_size"`
	EnableAsync bool `yaml:"enable_async"`
}

// Mem0Config configures the mem0-style intelligent memory system.
type Mem0Config struct {
	Enabled    *bool                `yaml:"enabled"`
	Storage    Mem0StorageConfig    `yaml:"storage"`
	Extraction Mem0ExtractionConfig `yaml:"extraction"`
	Updates    Mem0UpdatesConfig    `yaml:"updates"`
	Graph      Mem0GraphConfig      `yaml:"graph"`
	Retrieval  Mem0RetrievalConfig  `yaml:"retrieval"`
	Summary    Mem0SummaryConfig    `yaml:"summary"`
}

// Mem0StorageConfig configures mem0 storage.
type Mem0StorageConfig struct {
	DBPath          string `yaml:"db_path"`
	EmbeddingDim    int    `yaml:"embedding_dim"`
	MaxFacts        int    `yaml:"max_facts"`
	PruneThreshold  int    `yaml:"prune_threshold"`
	PruneKeepRecent int    `yaml:"prune_keep_recent"`
}

// Mem0ExtractionConfig configures fact extraction.
type Mem0ExtractionConfig struct {
	Enabled               bool `yaml:"enabled"`
	BatchSize             int  `yaml:"batch_size"`
	Async                 bool `yaml:"async"`
	MinTextLength         int  `yaml:"min_text_length"`
	MaxFactsPerExtraction int  `yaml:"max_facts_per_extraction"`
	ExtractEntities       bool `yaml:"extract_entities"`
	ExtractRelationships  bool `yaml:"extract_relationships"`
}

// Mem0UpdatesConfig configures memory update operations.
type Mem0UpdatesConfig struct {
	Enabled              bool    `yaml:"enabled"`
	ConflictThreshold    float64 `yaml:"conflict_threshold"`
	TopSimilarCount      int     `yaml:"top_similar_count"`
	AutoResolveConflicts bool    `yaml:"auto_resolve_conflicts"`
	TrackSupersession    bool    `yaml:"track_supersession"`
}

// Mem0GraphConfig configures entity-relationship graph.
type Mem0GraphConfig struct {
	Enabled                   bool    `yaml:"enabled"`
	EntityResolution          bool    `yaml:"entity_resolution"`
	EntitySimilarityThreshold float64 `yaml:"entity_similarity_threshold"`
	MaxHops                   int     `yaml:"max_hops"`
	TrackRelationshipHistory  bool    `yaml:"track_relationship_history"`
}

// Mem0RetrievalConfig configures hybrid retrieval.
type Mem0RetrievalConfig struct {
	SemanticWeight        float64 `yaml:"semantic_weight"`
	ImportanceWeight      float64 `yaml:"importance_weight"`
	RecencyWeight         float64 `yaml:"recency_weight"`
	AccessFrequencyWeight float64 `yaml:"access_frequency_weight"`
	MinScore              float64 `yaml:"min_score"`
	MaxResults            int     `yaml:"max_results"`
	IncludeGraphResults   bool    `yaml:"include_graph_results"`
	GraphResultsWeight    float64 `yaml:"graph_results_weight"`
	RecencyHalfLifeHours  float64 `yaml:"recency_half_life_hours"`
}

// Mem0SummaryConfig configures rolling summary.
type Mem0SummaryConfig struct {
	Enabled         bool   `yaml:"enabled"`
	RefreshInterval string `yaml:"refresh_interval"`
	MaxFacts        int    `yaml:"max_facts"`
	MaxTokens       int    `yaml:"max_tokens"`
	Async           bool   `yaml:"async"`
}

// ServerConfig defines TCP server settings for the message transport.
type ServerConfig struct {
	Host    string `yaml:"host"`
	Port    int    `yaml:"port"`
	Enabled *bool  `yaml:"enabled"`
}

// ConversationConfig governs how the prompt context is assembled.
type ConversationConfig struct {
	SystemMessage string `yaml:"system_message"`
	TemplatePath  string `yaml:"template_path"`
}

// RAGConfig governs retrieval augmented generation helpers.
type RAGConfig struct {
	Enabled      bool     `yaml:"enabled"`
	CorpusPath   string   `yaml:"corpus_path"`
	MaxChunks    int      `yaml:"max_chunks"`
	ChunkSize    int      `yaml:"chunk_size"`
	ChunkOverlap int      `yaml:"chunk_overlap"`
	MinScore     float64  `yaml:"min_score"`
	IndexPath    string   `yaml:"index_path"`
	Extensions   []string `yaml:"extensions"`

	// Hybrid retrieval settings
	HybridEnabled        bool    `yaml:"hybrid_enabled"`
	MaxCandidates        int     `yaml:"max_candidates"`
	DiversityThreshold   float64 `yaml:"diversity_threshold"`
	SemanticWeight       float64 `yaml:"semantic_weight"`
	KeywordWeight        float64 `yaml:"keyword_weight"`
	RAGRecencyWeight     float64 `yaml:"rag_recency_weight"`
	EnableQueryExpansion bool    `yaml:"enable_query_expansion"`
	DedupeThreshold      float64 `yaml:"dedupe_threshold"`
	MergeAdjacentChunks  bool    `yaml:"merge_adjacent_chunks"`
	MaxMergedTokens      int     `yaml:"max_merged_tokens"`
}

// AssistantsConfig groups secondary helper models invoked by the pipeline.
type AssistantsConfig struct {
	Summarizer SummarizerConfig `yaml:"summarizer"`
}

// SummarizerConfig configures the helper summarisation stage.
type SummarizerConfig struct {
	Enabled             bool    `yaml:"enabled"`
	Prompt              string  `yaml:"prompt"`
	MaxTokens           int     `yaml:"max_tokens"`
	MinTurns            int     `yaml:"min_turns"`
	MaxReferences       int     `yaml:"max_references"`
	SimilarityThreshold float64 `yaml:"similarity_threshold"`
	MaxTranscriptTokens int     `yaml:"max_transcript_tokens"`
}

// EmbeddingConfig captures settings for semantic embedding providers.
type EmbeddingConfig struct {
	Enabled  *bool                   `yaml:"enabled"`
	Backend  string                  `yaml:"backend"`
	LlamaCpp LlamaCppEmbeddingConfig `yaml:"llamacpp"`
	Native   NativeEmbeddingConfig   `yaml:"native"`
}

// NativeEmbeddingConfig configures the native in-process embedding provider.
// Used when embedding.backend is "native" and the binary is compiled with -tags native.
type NativeEmbeddingConfig struct {
	ModelPath   string `yaml:"model_path"`
	ContextSize int    `yaml:"context_size"`
	BatchSize   int    `yaml:"batch_size"`
	Threads     int    `yaml:"threads"`
	GPULayers   int    `yaml:"gpu_layers"`
	Mmap        *bool  `yaml:"mmap"`
	Mlock       *bool  `yaml:"mlock"`
}

// LlamaCppEmbeddingConfig configures llama.cpp embedding server usage.
type LlamaCppEmbeddingConfig struct {
	BaseURL string `yaml:"base_url"`
	Model   string `yaml:"model"`
	Timeout string `yaml:"timeout"`
}

// ImageConfig configures image processing for vision models.
type ImageConfig struct {
	// Enabled enables image processing before sending to the model.
	Enabled bool `yaml:"enabled"`
	// MaxWidth is the maximum width; images wider will be resized.
	MaxWidth int `yaml:"max_width"`
	// MaxHeight is the maximum height; images taller will be resized.
	MaxHeight int `yaml:"max_height"`
	// OutputFormat is the target format (jpeg, png, bmp).
	OutputFormat string `yaml:"output_format"`
	// Quality is the JPEG quality (1-100), only used for JPEG output.
	Quality int `yaml:"quality"`
	// PreserveAspectRatio maintains aspect ratio when resizing.
	PreserveAspectRatio bool `yaml:"preserve_aspect_ratio"`
	// AutoDetectInput automatically detects if input is base64 or file path.
	AutoDetectInput bool `yaml:"auto_detect_input"`
	// OutputAsBase64 returns processed images as base64 strings.
	OutputAsBase64 bool `yaml:"output_as_base64"`
}

const defaultConfigFile = "openeye.yaml"

// boolPtr returns a pointer to the given bool value.
// Used for *bool config fields that need to distinguish "not set" from "false".
func boolPtr(b bool) *bool { return &b }

// Default returns a Config pre-populated with opinionated defaults for local SLMs.
func Default() Config {
	return Config{
		Runtime: RuntimeConfig{
			Backend: "http",
			HTTP: HTTPBackendConfig{
				BaseURL: "http://127.0.0.1:42069",
				Timeout: "60s",
			},
			Defaults: GenerationDefaults{
				MaxTokens:     512,
				Temperature:   0.7,
				TopK:          40,
				TopP:          0.9,
				MinP:          0.05,
				RepeatPenalty: 1.1,
				RepeatLastN:   64,
				Stop:          nil,
			},
		},
		Memory: MemoryConfig{
			Path:               "openeye_memory.db",
			TurnsToUse:         6,
			VectorEnabled:      true,
			VectorDBPath:       "openeye_vector.duckdb",
			EmbeddingDim:       384,
			MaxContextTokens:   2048,
			ReservedForPrompt:  512,
			ReservedForSummary: 256,
			MinSimilarity:      0.3,
			SlidingWindowSize:  50,
			RecencyWeight:      0.3,
			RelevanceWeight:    0.7,
			CompressionEnabled: true,
			CompressionAge:     "24h",
			CompressBatchSize:  10,
			AutoCompress:       true,
			CompressEveryN:     50,
			Mem0: Mem0Config{
				Enabled: boolPtr(true),
				Storage: Mem0StorageConfig{
					DBPath:          "openeye_mem0.duckdb",
					EmbeddingDim:    384,
					MaxFacts:        10000,
					PruneThreshold:  12000,
					PruneKeepRecent: 5000,
				},
				Extraction: Mem0ExtractionConfig{
					Enabled:               true,
					BatchSize:             3,
					Async:                 true,
					MinTextLength:         10,
					MaxFactsPerExtraction: 10,
					ExtractEntities:       true,
					ExtractRelationships:  true,
				},
				Updates: Mem0UpdatesConfig{
					Enabled:              true,
					ConflictThreshold:    0.85,
					TopSimilarCount:      5,
					AutoResolveConflicts: true,
					TrackSupersession:    true,
				},
				Graph: Mem0GraphConfig{
					Enabled:                   true,
					EntityResolution:          true,
					EntitySimilarityThreshold: 0.9,
					MaxHops:                   2,
					TrackRelationshipHistory:  false,
				},
				Retrieval: Mem0RetrievalConfig{
					SemanticWeight:        0.50,
					ImportanceWeight:      0.25,
					RecencyWeight:         0.15,
					AccessFrequencyWeight: 0.10,
					MinScore:              0.3,
					MaxResults:            20,
					IncludeGraphResults:   true,
					GraphResultsWeight:    0.3,
					RecencyHalfLifeHours:  168,
				},
				Summary: Mem0SummaryConfig{
					Enabled:         true,
					RefreshInterval: "5m",
					MaxFacts:        50,
					MaxTokens:       512,
					Async:           true,
				},
			},
			Omem: OmemConfig{
				Enabled: boolPtr(true),
				Storage: OmemStorageConfig{
					DBPath:          "openeye_omem.duckdb",
					MaxFacts:        10000,
					PruneThreshold:  12000,
					PruneKeepRecent: 5000,
					EnableFTS:       true,
				},
				AtomicEncoder: OmemEncoderConfig{
					Enabled:           true,
					EnableCoreference: true,
					EnableTemporal:    true,
					MaxFactsPerTurn:   10,
					MinFactImportance: 0.3,
					MinTextLength:     10,
					UseLLMForComplex:  true,
				},
				MultiView: OmemMultiViewConfig{
					Enabled:            true,
					SemanticWeight:     0.5,
					LexicalWeight:      0.3,
					SymbolicWeight:     0.2,
					BM25_K1:            1.2,
					BM25_B:             0.75,
					ExtractKeywords:    true,
					MaxKeywordsPerFact: 20,
				},
				EntityGraph: OmemEntityGraphConfig{
					Enabled:             true,
					MaxHops:             1,
					EntityResolution:    true,
					SimilarityThreshold: 0.85,
					UseRegexExtraction:  true,
					GraphBoostWeight:    0.2,
				},
				Retrieval: OmemRetrievalConfig{
					DefaultTopK:                5,
					ComplexityDelta:            2.0,
					MaxTopK:                    20,
					MinScore:                   0.3,
					MaxContextTokens:           1000,
					RecencyHalfLifeHours:       168,
					ImportanceWeight:           0.15,
					RecencyWeight:              0.1,
					AccessFrequencyWeight:      0.05,
					EnableComplexityEstimation: true,
				},
				Episodes: OmemEpisodesConfig{
					Enabled:             true,
					SessionTimeout:      "30m",
					SummaryOnClose:      true,
					MaxEpisodesInCache:  10,
					TrackEntityMentions: true,
				},
				Summary: OmemSummaryConfig{
					Enabled:              true,
					RefreshInterval:      "5m",
					MaxFacts:             50,
					MaxTokens:            512,
					Async:                true,
					IncrementalUpdate:    true,
					MinNewFactsForUpdate: 5,
				},
				Parallel: OmemParallelConfig{
					MaxWorkers:  0, // Auto-detect from CPU
					BatchSize:   10,
					QueueSize:   100,
					EnableAsync: true,
				},
			},
		},
		Server: ServerConfig{
			Host:    "127.0.0.1",
			Port:    42067,
			Enabled: boolPtr(true),
		},
		Conversation: ConversationConfig{},
		RAG: RAGConfig{
			Enabled:              false,
			CorpusPath:           "",
			MaxChunks:            4,
			ChunkSize:            512,
			ChunkOverlap:         64,
			MinScore:             0.2,
			IndexPath:            "openeye_rag.index",
			Extensions:           []string{".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".pdf"},
			HybridEnabled:        true,
			MaxCandidates:        50,
			DiversityThreshold:   0.3,
			SemanticWeight:       0.7,
			KeywordWeight:        0.2,
			RAGRecencyWeight:     0.1,
			EnableQueryExpansion: true,
			DedupeThreshold:      0.85,
			MergeAdjacentChunks:  true,
			MaxMergedTokens:      1000,
		},
		Assistants: AssistantsConfig{
			Summarizer: SummarizerConfig{
				Enabled:             false,
				Prompt:              "Summarize the following conversation history in concise bullet points highlighting commitments, open questions, and facts. Respond with at most five bullets.",
				MaxTokens:           128,
				MinTurns:            3,
				MaxReferences:       8,
				SimilarityThreshold: 0.1,
				MaxTranscriptTokens: 0,
			},
		},
		Embedding: EmbeddingConfig{
			Enabled: boolPtr(false),
			Backend: "llamacpp",
			LlamaCpp: LlamaCppEmbeddingConfig{
				BaseURL: "http://127.0.0.1:8080",
				Model:   "",
				Timeout: "30s",
			},
		},
		Image: ImageConfig{
			Enabled:             true,
			MaxWidth:            1024,
			MaxHeight:           1024,
			OutputFormat:        "jpeg",
			Quality:             85,
			PreserveAspectRatio: true,
			AutoDetectInput:     true,
			OutputAsBase64:      true,
		},
	}
}

// Resolve loads configuration from file and environment variables.
func Resolve() (Config, error) {
	cfg := Default()

	path := strings.TrimSpace(os.Getenv("APP_CONFIG"))
	if path == "" {
		if _, err := os.Stat(defaultConfigFile); err == nil {
			path = defaultConfigFile
		}
	} else if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return cfg, fmt.Errorf("provided APP_CONFIG file %q not found", path)
	}

	if path != "" {
		loaded, err := loadFile(path)
		if err != nil {
			return cfg, err
		}
		cfg = merge(cfg, loaded)
	}

	applyEnvOverrides(&cfg)

	return cfg, nil
}

func loadFile(path string) (Config, error) {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return Config{}, fmt.Errorf("failed to read config %q: %w", path, err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("failed to parse config %q: %w", path, err)
	}

	return cfg, nil
}

// merge overlays non-zero override values onto the base config.
//
// KNOWN LIMITATION: Most boolean fields (VectorEnabled, CompressionEnabled,
// AutoCompress, RAG.Enabled, etc.) can only be toggled ON via YAML override,
// never OFF. This is because Go's zero value for bool is false, making it
// impossible to distinguish "not set" from "explicitly set to false" without
// using *bool pointer fields. The following critical boolean fields use *bool
// and CAN be explicitly set to false: Server.Enabled, Embedding.Enabled,
// Mem0.Enabled, Omem.Enabled, plus NativeConfig's Mmap/Mlock/FlashAttention.
// Extending this pattern to all boolean config fields is a future refactor.
func merge(base, override Config) Config {
	result := base

	if override.Runtime.Backend != "" {
		result.Runtime.Backend = override.Runtime.Backend
	}
	if override.Runtime.HTTP.BaseURL != "" {
		result.Runtime.HTTP.BaseURL = override.Runtime.HTTP.BaseURL
	}
	if override.Runtime.HTTP.Timeout != "" {
		result.Runtime.HTTP.Timeout = override.Runtime.HTTP.Timeout
	}

	// Merge Native backend config.
	if override.Runtime.Native.ModelPath != "" {
		result.Runtime.Native.ModelPath = override.Runtime.Native.ModelPath
	}
	if override.Runtime.Native.MmprojPath != "" {
		result.Runtime.Native.MmprojPath = override.Runtime.Native.MmprojPath
	}
	if override.Runtime.Native.ContextSize != 0 {
		result.Runtime.Native.ContextSize = override.Runtime.Native.ContextSize
	}
	if override.Runtime.Native.BatchSize != 0 {
		result.Runtime.Native.BatchSize = override.Runtime.Native.BatchSize
	}
	if override.Runtime.Native.Threads != 0 {
		result.Runtime.Native.Threads = override.Runtime.Native.Threads
	}
	if override.Runtime.Native.ThreadsBatch != 0 {
		result.Runtime.Native.ThreadsBatch = override.Runtime.Native.ThreadsBatch
	}
	if override.Runtime.Native.GPULayers != 0 {
		result.Runtime.Native.GPULayers = override.Runtime.Native.GPULayers
	}
	if override.Runtime.Native.Mmap != nil {
		result.Runtime.Native.Mmap = override.Runtime.Native.Mmap
	}
	if override.Runtime.Native.Mlock != nil {
		result.Runtime.Native.Mlock = override.Runtime.Native.Mlock
	}
	if override.Runtime.Native.FlashAttention != nil {
		result.Runtime.Native.FlashAttention = override.Runtime.Native.FlashAttention
	}
	if override.Runtime.Native.Warmup {
		result.Runtime.Native.Warmup = true
	}
	if override.Runtime.Native.WarmupTokens != 0 {
		result.Runtime.Native.WarmupTokens = override.Runtime.Native.WarmupTokens
	}
	if override.Runtime.Native.DraftModelPath != "" {
		result.Runtime.Native.DraftModelPath = override.Runtime.Native.DraftModelPath
	}
	if override.Runtime.Native.SpeculativeN != 0 {
		result.Runtime.Native.SpeculativeN = override.Runtime.Native.SpeculativeN
	}
	if override.Runtime.Native.KVCacheType != "" {
		result.Runtime.Native.KVCacheType = override.Runtime.Native.KVCacheType
	}
	if override.Runtime.Native.StreamChunkSize != 0 {
		result.Runtime.Native.StreamChunkSize = override.Runtime.Native.StreamChunkSize
	}
	if override.Runtime.Native.ContextShift != nil {
		result.Runtime.Native.ContextShift = override.Runtime.Native.ContextShift
	}

	d := override.Runtime.Defaults
	if d.MaxTokens != 0 {
		result.Runtime.Defaults.MaxTokens = d.MaxTokens
	}
	if d.Temperature != 0 {
		result.Runtime.Defaults.Temperature = d.Temperature
	}
	if d.TopK != 0 {
		result.Runtime.Defaults.TopK = d.TopK
	}
	if d.TopP != 0 {
		result.Runtime.Defaults.TopP = d.TopP
	}
	if d.MinP != 0 {
		result.Runtime.Defaults.MinP = d.MinP
	}
	if d.RepeatPenalty != 0 {
		result.Runtime.Defaults.RepeatPenalty = d.RepeatPenalty
	}
	if d.RepeatLastN != 0 {
		result.Runtime.Defaults.RepeatLastN = d.RepeatLastN
	}
	if len(d.Stop) != 0 {
		result.Runtime.Defaults.Stop = append([]string(nil), d.Stop...)
	}

	if override.Memory.Path != "" {
		result.Memory.Path = override.Memory.Path
	}
	if override.Memory.TurnsToUse != 0 {
		result.Memory.TurnsToUse = override.Memory.TurnsToUse
	}
	if override.Memory.VectorEnabled {
		result.Memory.VectorEnabled = true
	}
	if override.Memory.VectorDBPath != "" {
		result.Memory.VectorDBPath = override.Memory.VectorDBPath
	}
	if override.Memory.EmbeddingDim != 0 {
		result.Memory.EmbeddingDim = override.Memory.EmbeddingDim
	}
	if override.Memory.MaxContextTokens != 0 {
		result.Memory.MaxContextTokens = override.Memory.MaxContextTokens
	}
	if override.Memory.ReservedForPrompt != 0 {
		result.Memory.ReservedForPrompt = override.Memory.ReservedForPrompt
	}
	if override.Memory.ReservedForSummary != 0 {
		result.Memory.ReservedForSummary = override.Memory.ReservedForSummary
	}
	if override.Memory.MinSimilarity != 0 {
		result.Memory.MinSimilarity = override.Memory.MinSimilarity
	}
	if override.Memory.SlidingWindowSize != 0 {
		result.Memory.SlidingWindowSize = override.Memory.SlidingWindowSize
	}
	if override.Memory.RecencyWeight != 0 {
		result.Memory.RecencyWeight = override.Memory.RecencyWeight
	}
	if override.Memory.RelevanceWeight != 0 {
		result.Memory.RelevanceWeight = override.Memory.RelevanceWeight
	}
	if override.Memory.CompressionEnabled {
		result.Memory.CompressionEnabled = true
	}
	if override.Memory.CompressionAge != "" {
		result.Memory.CompressionAge = override.Memory.CompressionAge
	}
	if override.Memory.CompressBatchSize != 0 {
		result.Memory.CompressBatchSize = override.Memory.CompressBatchSize
	}
	if override.Memory.AutoCompress {
		result.Memory.AutoCompress = true
	}
	if override.Memory.CompressEveryN != 0 {
		result.Memory.CompressEveryN = override.Memory.CompressEveryN
	}

	// Merge Mem0 configuration
	result.Memory.Mem0 = mergeMem0Config(result.Memory.Mem0, override.Memory.Mem0)

	// Merge Omem configuration
	result.Memory.Omem = mergeOmemConfig(result.Memory.Omem, override.Memory.Omem)

	if override.Server.Host != "" {
		result.Server.Host = override.Server.Host
	}
	if override.Server.Port != 0 {
		result.Server.Port = override.Server.Port
	}
	if override.Server.Enabled != nil {
		result.Server.Enabled = override.Server.Enabled
	}

	if override.Conversation.SystemMessage != "" {
		result.Conversation.SystemMessage = override.Conversation.SystemMessage
	}
	if override.Conversation.TemplatePath != "" {
		result.Conversation.TemplatePath = override.Conversation.TemplatePath
	}

	if override.RAG.Enabled {
		result.RAG.Enabled = true
	}
	if override.RAG.CorpusPath != "" {
		result.RAG.CorpusPath = override.RAG.CorpusPath
	}
	if override.RAG.MaxChunks != 0 {
		result.RAG.MaxChunks = override.RAG.MaxChunks
	}
	if override.RAG.ChunkSize != 0 {
		result.RAG.ChunkSize = override.RAG.ChunkSize
	}
	if override.RAG.ChunkOverlap != 0 {
		result.RAG.ChunkOverlap = override.RAG.ChunkOverlap
	}
	if override.RAG.MinScore != 0 {
		result.RAG.MinScore = override.RAG.MinScore
	}
	if override.RAG.IndexPath != "" {
		result.RAG.IndexPath = override.RAG.IndexPath
	}
	if len(override.RAG.Extensions) != 0 {
		result.RAG.Extensions = append([]string(nil), override.RAG.Extensions...)
	}
	if override.RAG.HybridEnabled {
		result.RAG.HybridEnabled = true
	}
	if override.RAG.MaxCandidates != 0 {
		result.RAG.MaxCandidates = override.RAG.MaxCandidates
	}
	if override.RAG.DiversityThreshold != 0 {
		result.RAG.DiversityThreshold = override.RAG.DiversityThreshold
	}
	if override.RAG.SemanticWeight != 0 {
		result.RAG.SemanticWeight = override.RAG.SemanticWeight
	}
	if override.RAG.KeywordWeight != 0 {
		result.RAG.KeywordWeight = override.RAG.KeywordWeight
	}
	if override.RAG.RAGRecencyWeight != 0 {
		result.RAG.RAGRecencyWeight = override.RAG.RAGRecencyWeight
	}
	if override.RAG.EnableQueryExpansion {
		result.RAG.EnableQueryExpansion = true
	}
	if override.RAG.DedupeThreshold != 0 {
		result.RAG.DedupeThreshold = override.RAG.DedupeThreshold
	}
	if override.RAG.MergeAdjacentChunks {
		result.RAG.MergeAdjacentChunks = true
	}
	if override.RAG.MaxMergedTokens != 0 {
		result.RAG.MaxMergedTokens = override.RAG.MaxMergedTokens
	}

	if override.Assistants.Summarizer.Enabled {
		result.Assistants.Summarizer.Enabled = true
	}
	if override.Assistants.Summarizer.Prompt != "" {
		result.Assistants.Summarizer.Prompt = override.Assistants.Summarizer.Prompt
	}
	if override.Assistants.Summarizer.MaxTokens != 0 {
		result.Assistants.Summarizer.MaxTokens = override.Assistants.Summarizer.MaxTokens
	}
	if override.Assistants.Summarizer.MinTurns != 0 {
		result.Assistants.Summarizer.MinTurns = override.Assistants.Summarizer.MinTurns
	}
	if override.Assistants.Summarizer.MaxReferences != 0 {
		result.Assistants.Summarizer.MaxReferences = override.Assistants.Summarizer.MaxReferences
	}
	if override.Assistants.Summarizer.SimilarityThreshold != 0 {
		result.Assistants.Summarizer.SimilarityThreshold = override.Assistants.Summarizer.SimilarityThreshold
	}
	if override.Assistants.Summarizer.MaxTranscriptTokens != 0 {
		result.Assistants.Summarizer.MaxTranscriptTokens = override.Assistants.Summarizer.MaxTranscriptTokens
	}

	if override.Embedding.Enabled != nil {
		result.Embedding.Enabled = override.Embedding.Enabled
	}
	if override.Embedding.Backend != "" {
		result.Embedding.Backend = override.Embedding.Backend
	}
	if override.Embedding.LlamaCpp.BaseURL != "" {
		result.Embedding.LlamaCpp.BaseURL = override.Embedding.LlamaCpp.BaseURL
	}
	if override.Embedding.LlamaCpp.Model != "" {
		result.Embedding.LlamaCpp.Model = override.Embedding.LlamaCpp.Model
	}
	if override.Embedding.LlamaCpp.Timeout != "" {
		result.Embedding.LlamaCpp.Timeout = override.Embedding.LlamaCpp.Timeout
	}
	if override.Embedding.Native.ModelPath != "" {
		result.Embedding.Native.ModelPath = override.Embedding.Native.ModelPath
	}
	if override.Embedding.Native.ContextSize != 0 {
		result.Embedding.Native.ContextSize = override.Embedding.Native.ContextSize
	}
	if override.Embedding.Native.BatchSize != 0 {
		result.Embedding.Native.BatchSize = override.Embedding.Native.BatchSize
	}
	if override.Embedding.Native.Threads != 0 {
		result.Embedding.Native.Threads = override.Embedding.Native.Threads
	}
	if override.Embedding.Native.GPULayers != 0 {
		result.Embedding.Native.GPULayers = override.Embedding.Native.GPULayers
	}
	if override.Embedding.Native.Mmap != nil {
		result.Embedding.Native.Mmap = override.Embedding.Native.Mmap
	}
	if override.Embedding.Native.Mlock != nil {
		result.Embedding.Native.Mlock = override.Embedding.Native.Mlock
	}

	return result
}

func applyEnvOverrides(cfg *Config) {
	if v := strings.TrimSpace(os.Getenv("APP_LLM_BASEURL")); v != "" {
		cfg.Runtime.HTTP.BaseURL = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_NATIVE_MODEL")); v != "" {
		cfg.Runtime.Native.ModelPath = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_NATIVE_MMPROJ")); v != "" {
		cfg.Runtime.Native.MmprojPath = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_NATIVE_CTX_SIZE")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Runtime.Native.ContextSize = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_NATIVE_THREADS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Runtime.Native.Threads = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_NATIVE_GPU_LAYERS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.Runtime.Native.GPULayers = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_MEMORY_PATH")); v != "" {
		cfg.Memory.Path = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_MEMORY_TURNS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Memory.TurnsToUse = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_SYSMSG")); v != "" {
		cfg.Conversation.SystemMessage = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_CONTEXT_PATH")); v != "" {
		cfg.Conversation.TemplatePath = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_SERVER_HOST")); v != "" {
		cfg.Server.Host = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_SERVER_PORT")); v != "" {
		if port, err := strconv.Atoi(v); err == nil && port > 0 {
			cfg.Server.Port = port
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_SERVER_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Server.Enabled = boolPtr(enabled)
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_RAG_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.RAG.Enabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_RAG_CORPUS")); v != "" {
		cfg.RAG.CorpusPath = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_RAG_MAXCHUNKS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.RAG.MaxChunks = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_RAG_CHUNKSIZE")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.RAG.ChunkSize = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_RAG_CHUNKOVERLAP")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.RAG.ChunkOverlap = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_RAG_MINSCORE")); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			cfg.RAG.MinScore = f
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_RAG_INDEX")); v != "" {
		cfg.RAG.IndexPath = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_RAG_EXTENSIONS")); v != "" {
		parts := strings.Split(v, ",")
		cfg.RAG.Extensions = make([]string, 0, len(parts))
		for _, part := range parts {
			trimmed := strings.TrimSpace(part)
			if trimmed == "" {
				continue
			}
			if !strings.HasPrefix(trimmed, ".") {
				trimmed = "." + trimmed
			}
			cfg.RAG.Extensions = append(cfg.RAG.Extensions, strings.ToLower(trimmed))
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_SUMMARY_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Assistants.Summarizer.Enabled = enabled
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_SUMMARY_PROMPT")); v != "" {
		cfg.Assistants.Summarizer.Prompt = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_SUMMARY_MAXTOKENS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Assistants.Summarizer.MaxTokens = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_SUMMARY_MIN_TURNS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.Assistants.Summarizer.MinTurns = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_SUMMARY_MAX_REFERENCES")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.Assistants.Summarizer.MaxReferences = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_SUMMARY_SIMILARITY")); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 {
			cfg.Assistants.Summarizer.SimilarityThreshold = f
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_SUMMARY_MAX_TRANSCRIPT_TOKENS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			cfg.Assistants.Summarizer.MaxTranscriptTokens = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_EMBEDDING_ENABLED")); v != "" {
		if enabled, err := strconv.ParseBool(v); err == nil {
			cfg.Embedding.Enabled = boolPtr(enabled)
		}
	}
	if v := strings.TrimSpace(os.Getenv("APP_EMBEDDING_BACKEND")); v != "" {
		cfg.Embedding.Backend = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_EMBEDDING_BASEURL")); v != "" {
		cfg.Embedding.LlamaCpp.BaseURL = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_EMBEDDING_MODEL")); v != "" {
		cfg.Embedding.LlamaCpp.Model = v
	}
	if v := strings.TrimSpace(os.Getenv("APP_EMBEDDING_TIMEOUT")); v != "" {
		cfg.Embedding.LlamaCpp.Timeout = v
	}
}

// ServerEnabled reports if the TCP server should be started.
func (c Config) ServerEnabled() bool {
	return c.Server.Enabled != nil && *c.Server.Enabled
}

// mergeMem0Config merges mem0 configuration with overrides.
func mergeMem0Config(base, override Mem0Config) Mem0Config {
	result := base

	if override.Enabled != nil {
		result.Enabled = override.Enabled
	}

	// Storage
	if override.Storage.DBPath != "" {
		result.Storage.DBPath = override.Storage.DBPath
	}
	if override.Storage.EmbeddingDim != 0 {
		result.Storage.EmbeddingDim = override.Storage.EmbeddingDim
	}
	if override.Storage.MaxFacts != 0 {
		result.Storage.MaxFacts = override.Storage.MaxFacts
	}
	if override.Storage.PruneThreshold != 0 {
		result.Storage.PruneThreshold = override.Storage.PruneThreshold
	}
	if override.Storage.PruneKeepRecent != 0 {
		result.Storage.PruneKeepRecent = override.Storage.PruneKeepRecent
	}

	// Extraction
	if override.Extraction.Enabled {
		result.Extraction.Enabled = true
	}
	if override.Extraction.BatchSize != 0 {
		result.Extraction.BatchSize = override.Extraction.BatchSize
	}
	if override.Extraction.Async {
		result.Extraction.Async = true
	}
	if override.Extraction.MinTextLength != 0 {
		result.Extraction.MinTextLength = override.Extraction.MinTextLength
	}
	if override.Extraction.MaxFactsPerExtraction != 0 {
		result.Extraction.MaxFactsPerExtraction = override.Extraction.MaxFactsPerExtraction
	}
	if override.Extraction.ExtractEntities {
		result.Extraction.ExtractEntities = true
	}
	if override.Extraction.ExtractRelationships {
		result.Extraction.ExtractRelationships = true
	}

	// Updates
	if override.Updates.Enabled {
		result.Updates.Enabled = true
	}
	if override.Updates.ConflictThreshold != 0 {
		result.Updates.ConflictThreshold = override.Updates.ConflictThreshold
	}
	if override.Updates.TopSimilarCount != 0 {
		result.Updates.TopSimilarCount = override.Updates.TopSimilarCount
	}
	if override.Updates.AutoResolveConflicts {
		result.Updates.AutoResolveConflicts = true
	}
	if override.Updates.TrackSupersession {
		result.Updates.TrackSupersession = true
	}

	// Graph
	if override.Graph.Enabled {
		result.Graph.Enabled = true
	}
	if override.Graph.EntityResolution {
		result.Graph.EntityResolution = true
	}
	if override.Graph.EntitySimilarityThreshold != 0 {
		result.Graph.EntitySimilarityThreshold = override.Graph.EntitySimilarityThreshold
	}
	if override.Graph.MaxHops != 0 {
		result.Graph.MaxHops = override.Graph.MaxHops
	}
	if override.Graph.TrackRelationshipHistory {
		result.Graph.TrackRelationshipHistory = true
	}

	// Retrieval
	if override.Retrieval.SemanticWeight != 0 {
		result.Retrieval.SemanticWeight = override.Retrieval.SemanticWeight
	}
	if override.Retrieval.ImportanceWeight != 0 {
		result.Retrieval.ImportanceWeight = override.Retrieval.ImportanceWeight
	}
	if override.Retrieval.RecencyWeight != 0 {
		result.Retrieval.RecencyWeight = override.Retrieval.RecencyWeight
	}
	if override.Retrieval.AccessFrequencyWeight != 0 {
		result.Retrieval.AccessFrequencyWeight = override.Retrieval.AccessFrequencyWeight
	}
	if override.Retrieval.MinScore != 0 {
		result.Retrieval.MinScore = override.Retrieval.MinScore
	}
	if override.Retrieval.MaxResults != 0 {
		result.Retrieval.MaxResults = override.Retrieval.MaxResults
	}
	if override.Retrieval.IncludeGraphResults {
		result.Retrieval.IncludeGraphResults = true
	}
	if override.Retrieval.GraphResultsWeight != 0 {
		result.Retrieval.GraphResultsWeight = override.Retrieval.GraphResultsWeight
	}
	if override.Retrieval.RecencyHalfLifeHours != 0 {
		result.Retrieval.RecencyHalfLifeHours = override.Retrieval.RecencyHalfLifeHours
	}

	// Summary
	if override.Summary.Enabled {
		result.Summary.Enabled = true
	}
	if override.Summary.RefreshInterval != "" {
		result.Summary.RefreshInterval = override.Summary.RefreshInterval
	}
	if override.Summary.MaxFacts != 0 {
		result.Summary.MaxFacts = override.Summary.MaxFacts
	}
	if override.Summary.MaxTokens != 0 {
		result.Summary.MaxTokens = override.Summary.MaxTokens
	}
	if override.Summary.Async {
		result.Summary.Async = true
	}

	return result
}

// mergeOmemConfig merges Omem configuration with overrides.
func mergeOmemConfig(base, override OmemConfig) OmemConfig {
	result := base

	if override.Enabled != nil {
		result.Enabled = override.Enabled
	}

	// Storage
	if override.Storage.DBPath != "" {
		result.Storage.DBPath = override.Storage.DBPath
	}
	if override.Storage.MaxFacts != 0 {
		result.Storage.MaxFacts = override.Storage.MaxFacts
	}
	if override.Storage.PruneThreshold != 0 {
		result.Storage.PruneThreshold = override.Storage.PruneThreshold
	}
	if override.Storage.PruneKeepRecent != 0 {
		result.Storage.PruneKeepRecent = override.Storage.PruneKeepRecent
	}
	if override.Storage.EnableFTS {
		result.Storage.EnableFTS = true
	}

	// AtomicEncoder
	if override.AtomicEncoder.Enabled {
		result.AtomicEncoder.Enabled = true
	}
	if override.AtomicEncoder.EnableCoreference {
		result.AtomicEncoder.EnableCoreference = true
	}
	if override.AtomicEncoder.EnableTemporal {
		result.AtomicEncoder.EnableTemporal = true
	}
	if override.AtomicEncoder.MaxFactsPerTurn != 0 {
		result.AtomicEncoder.MaxFactsPerTurn = override.AtomicEncoder.MaxFactsPerTurn
	}
	if override.AtomicEncoder.MinFactImportance != 0 {
		result.AtomicEncoder.MinFactImportance = override.AtomicEncoder.MinFactImportance
	}
	if override.AtomicEncoder.MinTextLength != 0 {
		result.AtomicEncoder.MinTextLength = override.AtomicEncoder.MinTextLength
	}
	if override.AtomicEncoder.UseLLMForComplex {
		result.AtomicEncoder.UseLLMForComplex = true
	}

	// MultiView
	if override.MultiView.Enabled {
		result.MultiView.Enabled = true
	}
	if override.MultiView.SemanticWeight != 0 {
		result.MultiView.SemanticWeight = override.MultiView.SemanticWeight
	}
	if override.MultiView.LexicalWeight != 0 {
		result.MultiView.LexicalWeight = override.MultiView.LexicalWeight
	}
	if override.MultiView.SymbolicWeight != 0 {
		result.MultiView.SymbolicWeight = override.MultiView.SymbolicWeight
	}
	if override.MultiView.BM25_K1 != 0 {
		result.MultiView.BM25_K1 = override.MultiView.BM25_K1
	}
	if override.MultiView.BM25_B != 0 {
		result.MultiView.BM25_B = override.MultiView.BM25_B
	}
	if override.MultiView.ExtractKeywords {
		result.MultiView.ExtractKeywords = true
	}
	if override.MultiView.MaxKeywordsPerFact != 0 {
		result.MultiView.MaxKeywordsPerFact = override.MultiView.MaxKeywordsPerFact
	}

	// EntityGraph
	if override.EntityGraph.Enabled {
		result.EntityGraph.Enabled = true
	}
	if override.EntityGraph.MaxHops != 0 {
		result.EntityGraph.MaxHops = override.EntityGraph.MaxHops
	}
	if override.EntityGraph.EntityResolution {
		result.EntityGraph.EntityResolution = true
	}
	if override.EntityGraph.SimilarityThreshold != 0 {
		result.EntityGraph.SimilarityThreshold = override.EntityGraph.SimilarityThreshold
	}
	if override.EntityGraph.UseRegexExtraction {
		result.EntityGraph.UseRegexExtraction = true
	}
	if override.EntityGraph.GraphBoostWeight != 0 {
		result.EntityGraph.GraphBoostWeight = override.EntityGraph.GraphBoostWeight
	}

	// Retrieval
	if override.Retrieval.DefaultTopK != 0 {
		result.Retrieval.DefaultTopK = override.Retrieval.DefaultTopK
	}
	if override.Retrieval.ComplexityDelta != 0 {
		result.Retrieval.ComplexityDelta = override.Retrieval.ComplexityDelta
	}
	if override.Retrieval.MaxTopK != 0 {
		result.Retrieval.MaxTopK = override.Retrieval.MaxTopK
	}
	if override.Retrieval.MinScore != 0 {
		result.Retrieval.MinScore = override.Retrieval.MinScore
	}
	if override.Retrieval.MaxContextTokens != 0 {
		result.Retrieval.MaxContextTokens = override.Retrieval.MaxContextTokens
	}
	if override.Retrieval.RecencyHalfLifeHours != 0 {
		result.Retrieval.RecencyHalfLifeHours = override.Retrieval.RecencyHalfLifeHours
	}
	if override.Retrieval.ImportanceWeight != 0 {
		result.Retrieval.ImportanceWeight = override.Retrieval.ImportanceWeight
	}
	if override.Retrieval.RecencyWeight != 0 {
		result.Retrieval.RecencyWeight = override.Retrieval.RecencyWeight
	}
	if override.Retrieval.AccessFrequencyWeight != 0 {
		result.Retrieval.AccessFrequencyWeight = override.Retrieval.AccessFrequencyWeight
	}
	if override.Retrieval.EnableComplexityEstimation {
		result.Retrieval.EnableComplexityEstimation = true
	}

	// Episodes
	if override.Episodes.Enabled {
		result.Episodes.Enabled = true
	}
	if override.Episodes.SessionTimeout != "" {
		result.Episodes.SessionTimeout = override.Episodes.SessionTimeout
	}
	if override.Episodes.SummaryOnClose {
		result.Episodes.SummaryOnClose = true
	}
	if override.Episodes.MaxEpisodesInCache != 0 {
		result.Episodes.MaxEpisodesInCache = override.Episodes.MaxEpisodesInCache
	}
	if override.Episodes.TrackEntityMentions {
		result.Episodes.TrackEntityMentions = true
	}

	// Summary
	if override.Summary.Enabled {
		result.Summary.Enabled = true
	}
	if override.Summary.RefreshInterval != "" {
		result.Summary.RefreshInterval = override.Summary.RefreshInterval
	}
	if override.Summary.MaxFacts != 0 {
		result.Summary.MaxFacts = override.Summary.MaxFacts
	}
	if override.Summary.MaxTokens != 0 {
		result.Summary.MaxTokens = override.Summary.MaxTokens
	}
	if override.Summary.Async {
		result.Summary.Async = true
	}
	if override.Summary.IncrementalUpdate {
		result.Summary.IncrementalUpdate = true
	}
	if override.Summary.MinNewFactsForUpdate != 0 {
		result.Summary.MinNewFactsForUpdate = override.Summary.MinNewFactsForUpdate
	}

	// Parallel
	if override.Parallel.MaxWorkers != 0 {
		result.Parallel.MaxWorkers = override.Parallel.MaxWorkers
	}
	if override.Parallel.BatchSize != 0 {
		result.Parallel.BatchSize = override.Parallel.BatchSize
	}
	if override.Parallel.QueueSize != 0 {
		result.Parallel.QueueSize = override.Parallel.QueueSize
	}
	if override.Parallel.EnableAsync {
		result.Parallel.EnableAsync = true
	}

	return result
}

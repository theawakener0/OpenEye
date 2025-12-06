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
	Backend  string             `yaml:"backend"`
	HTTP     HTTPBackendConfig  `yaml:"http"`
	Defaults GenerationDefaults `yaml:"defaults"`
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

// HTTPBackendConfig configures the default HTTP completion backend.
type HTTPBackendConfig struct {
	BaseURL string `yaml:"base_url"`
	Timeout string `yaml:"timeout"`
}

// MemoryConfig configures persistent conversation history storage and vector memory.
type MemoryConfig struct {
	Path       string `yaml:"path"`
	TurnsToUse int    `yaml:"turns_to_use"`

	// Vector memory settings
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
}

// ServerConfig defines TCP server settings for the message transport.
type ServerConfig struct {
	Host    string `yaml:"host"`
	Port    int    `yaml:"port"`
	Enabled bool   `yaml:"enabled"`
}

// ConversationConfig governs how the prompt context is assembled.
type ConversationConfig struct {
	SystemMessage string `yaml:"system_message"`
	TemplatePath  string `yaml:"template_path"`
}

// RAGConfig governs retrieval augmented generation helpers.
type RAGConfig struct {
	Enabled      bool    `yaml:"enabled"`
	CorpusPath   string  `yaml:"corpus_path"`
	MaxChunks    int     `yaml:"max_chunks"`
	ChunkSize    int     `yaml:"chunk_size"`
	ChunkOverlap int     `yaml:"chunk_overlap"`
	MinScore     float64 `yaml:"min_score"`
	IndexPath    string  `yaml:"index_path"`
	Extensions   []string `yaml:"extensions"`

	// Hybrid retrieval settings
	HybridEnabled      bool    `yaml:"hybrid_enabled"`
	MaxCandidates      int     `yaml:"max_candidates"`
	DiversityThreshold float64 `yaml:"diversity_threshold"`
	SemanticWeight     float64 `yaml:"semantic_weight"`
	KeywordWeight      float64 `yaml:"keyword_weight"`
	RAGRecencyWeight   float64 `yaml:"rag_recency_weight"`
	EnableQueryExpansion bool  `yaml:"enable_query_expansion"`
	DedupeThreshold    float64 `yaml:"dedupe_threshold"`
	MergeAdjacentChunks bool   `yaml:"merge_adjacent_chunks"`
	MaxMergedTokens    int     `yaml:"max_merged_tokens"`
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
	Enabled  bool                    `yaml:"enabled"`
	Backend  string                  `yaml:"backend"`
	LlamaCpp LlamaCppEmbeddingConfig `yaml:"llamacpp"`
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
		},
		Server: ServerConfig{
			Host:    "127.0.0.1",
			Port:    42067,
			Enabled: true,
		},
		Conversation: ConversationConfig{},
		RAG: RAGConfig{
			Enabled:             false,
			CorpusPath:          "",
			MaxChunks:           4,
			ChunkSize:           512,
			ChunkOverlap:        64,
			MinScore:            0.2,
			IndexPath:           "openeye_rag.index",
			Extensions:          []string{".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".pdf"},
			HybridEnabled:       true,
			MaxCandidates:       50,
			DiversityThreshold:  0.3,
			SemanticWeight:      0.7,
			KeywordWeight:       0.2,
			RAGRecencyWeight:    0.1,
			EnableQueryExpansion: true,
			DedupeThreshold:     0.85,
			MergeAdjacentChunks: true,
			MaxMergedTokens:     1000,
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
			Enabled: false,
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

	if override.Server.Host != "" {
		result.Server.Host = override.Server.Host
	}
	if override.Server.Port != 0 {
		result.Server.Port = override.Server.Port
	}
	if override.Server.Enabled {
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

	if override.Embedding.Enabled {
		result.Embedding.Enabled = true
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

	return result
}

func applyEnvOverrides(cfg *Config) {
	if v := strings.TrimSpace(os.Getenv("APP_LLM_BASEURL")); v != "" {
		cfg.Runtime.HTTP.BaseURL = v
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
			cfg.Server.Enabled = enabled
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
			cfg.Embedding.Enabled = enabled
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
	return c.Server.Enabled
}

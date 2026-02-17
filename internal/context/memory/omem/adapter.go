package omem

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/embedding"
	"OpenEye/internal/rag"
	"OpenEye/internal/runtime"
)

// Adapter wraps the omem Engine to integrate with the OpenEye pipeline.
// It provides a simpler interface for common operations and handles
// the conversion between pipeline types and omem internal types.
type Adapter struct {
	engine      *Engine
	externalRAG rag.Retriever
}

// NewAdapterFromConfig creates a new omem adapter from the external config.OmemConfig.
// This is the primary constructor for pipeline integration.
func NewAdapterFromConfig(cfg config.OmemConfig, manager *runtime.Manager, embedder embedding.Provider) (*Adapter, error) {
	internalCfg := configFromOmemConfig(cfg)
	return NewAdapter(internalCfg, manager, embedder)
}

// configFromOmemConfig converts config.OmemConfig to internal omem.Config.
func configFromOmemConfig(cfg config.OmemConfig) Config {
	// Parse session timeout
	sessionTimeout := 30 * time.Minute
	if cfg.Episodes.SessionTimeout != "" {
		if parsed, err := time.ParseDuration(cfg.Episodes.SessionTimeout); err == nil {
			sessionTimeout = parsed
		}
	}

	// Parse refresh interval
	refreshInterval := 5 * time.Minute
	if cfg.Summary.RefreshInterval != "" {
		if parsed, err := time.ParseDuration(cfg.Summary.RefreshInterval); err == nil {
			refreshInterval = parsed
		}
	}

	return Config{
		Enabled: cfg.Enabled != nil && *cfg.Enabled,

		Storage: StorageConfig{
			DBPath:          cfg.Storage.DBPath,
			MaxFacts:        cfg.Storage.MaxFacts,
			PruneThreshold:  cfg.Storage.PruneThreshold,
			PruneKeepRecent: cfg.Storage.PruneKeepRecent,
			EnableFTS:       cfg.Storage.EnableFTS,
		},

		AtomicEncoder: AtomicEncoderConfig{
			Enabled:           cfg.AtomicEncoder.Enabled,
			EnableCoreference: cfg.AtomicEncoder.EnableCoreference,
			EnableTemporal:    cfg.AtomicEncoder.EnableTemporal,
			MaxFactsPerTurn:   cfg.AtomicEncoder.MaxFactsPerTurn,
			MinFactImportance: cfg.AtomicEncoder.MinFactImportance,
			MinTextLength:     cfg.AtomicEncoder.MinTextLength,
			UseLLMForComplex:  cfg.AtomicEncoder.UseLLMForComplex,
		},

		MultiViewIndex: MultiViewConfig{
			Enabled:            cfg.MultiView.Enabled,
			SemanticWeight:     cfg.MultiView.SemanticWeight,
			LexicalWeight:      cfg.MultiView.LexicalWeight,
			SymbolicWeight:     cfg.MultiView.SymbolicWeight,
			BM25_K1:            cfg.MultiView.BM25_K1,
			BM25_B:             cfg.MultiView.BM25_B,
			ExtractKeywords:    cfg.MultiView.ExtractKeywords,
			MaxKeywordsPerFact: cfg.MultiView.MaxKeywordsPerFact,
		},

		EntityGraph: EntityGraphConfig{
			Enabled:             cfg.EntityGraph.Enabled,
			MaxHops:             cfg.EntityGraph.MaxHops,
			EntityResolution:    cfg.EntityGraph.EntityResolution,
			SimilarityThreshold: cfg.EntityGraph.SimilarityThreshold,
			UseRegexExtraction:  cfg.EntityGraph.UseRegexExtraction,
			GraphBoostWeight:    cfg.EntityGraph.GraphBoostWeight,
		},

		Retrieval: RetrievalConfig{
			DefaultTopK:                cfg.Retrieval.DefaultTopK,
			ComplexityDelta:            cfg.Retrieval.ComplexityDelta,
			MaxTopK:                    cfg.Retrieval.MaxTopK,
			MinScore:                   cfg.Retrieval.MinScore,
			MaxContextTokens:           cfg.Retrieval.MaxContextTokens,
			RecencyHalfLifeHours:       cfg.Retrieval.RecencyHalfLifeHours,
			ImportanceWeight:           cfg.Retrieval.ImportanceWeight,
			RecencyWeight:              cfg.Retrieval.RecencyWeight,
			AccessFrequencyWeight:      cfg.Retrieval.AccessFrequencyWeight,
			EnableComplexityEstimation: cfg.Retrieval.EnableComplexityEstimation,
		},

		Episodes: EpisodeConfig{
			Enabled:             cfg.Episodes.Enabled,
			SessionTimeout:      sessionTimeout,
			SummaryOnClose:      cfg.Episodes.SummaryOnClose,
			MaxEpisodesInCache:  cfg.Episodes.MaxEpisodesInCache,
			TrackEntityMentions: cfg.Episodes.TrackEntityMentions,
		},

		Summary: SummaryConfig{
			Enabled:              cfg.Summary.Enabled,
			RefreshInterval:      refreshInterval,
			MaxFacts:             cfg.Summary.MaxFacts,
			MaxTokens:            cfg.Summary.MaxTokens,
			Async:                cfg.Summary.Async,
			IncrementalUpdate:    cfg.Summary.IncrementalUpdate,
			MinNewFactsForUpdate: cfg.Summary.MinNewFactsForUpdate,
		},

		Parallel: ParallelConfig{
			MaxWorkers:  cfg.Parallel.MaxWorkers,
			BatchSize:   cfg.Parallel.BatchSize,
			QueueSize:   cfg.Parallel.QueueSize,
			EnableAsync: cfg.Parallel.EnableAsync,
		},
	}
}

// NewAdapter creates a new omem adapter with the given configuration.
// The adapter handles initialization of the engine with the provided
// runtime manager and embedding provider.
func NewAdapter(cfg Config, manager *runtime.Manager, embedder embedding.Provider) (*Adapter, error) {
	if !cfg.Enabled {
		return &Adapter{engine: nil}, nil
	}

	engine, err := NewEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create omem engine: %w", err)
	}

	// Create LLM generate function from runtime manager
	var llmGenerate func(ctx context.Context, prompt string) (string, error)
	if manager != nil {
		llmGenerate = func(ctx context.Context, prompt string) (string, error) {
			resp, err := manager.Generate(ctx, runtime.Request{
				Prompt: prompt,
				Options: runtime.GenerationOptions{
					MaxTokens:   512,
					Temperature: 0.3, // Lower temp for factual extraction
				},
			})
			if err != nil {
				return "", err
			}
			return resp.Text, nil
		}
	}

	// Create embedding function from provider
	var embeddingFunc func(ctx context.Context, text string) ([]float32, error)
	if embedder != nil {
		embeddingFunc = func(ctx context.Context, text string) ([]float32, error) {
			return embedder.Embed(ctx, text)
		}
	}

	// Initialize the engine
	if err := engine.Initialize(llmGenerate, embeddingFunc); err != nil {
		return nil, fmt.Errorf("failed to initialize omem engine: %w", err)
	}

	return &Adapter{engine: engine}, nil
}

// NewAdapterWithFunctions creates an adapter with custom LLM and embedding functions.
// This is useful for testing or when using custom implementations.
func NewAdapterWithFunctions(
	cfg Config,
	llmGenerate func(ctx context.Context, prompt string) (string, error),
	embeddingFunc func(ctx context.Context, text string) ([]float32, error),
) (*Adapter, error) {
	if !cfg.Enabled {
		return &Adapter{engine: nil}, nil
	}

	engine, err := NewEngine(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create omem engine: %w", err)
	}

	if err := engine.Initialize(llmGenerate, embeddingFunc); err != nil {
		return nil, fmt.Errorf("failed to initialize omem engine: %w", err)
	}

	return &Adapter{engine: engine}, nil
}

// IsEnabled returns whether omem is enabled and initialized.
func (a *Adapter) IsEnabled() bool {
	return a != nil && a.engine != nil && a.engine.isReady()
}

// ProcessTurn processes a single conversation turn (user message + assistant response).
// This should be called after each exchange in the pipeline.
func (a *Adapter) ProcessTurn(ctx context.Context, userMessage, assistantResponse string, turnID string) error {
	if !a.IsEnabled() {
		return nil
	}

	turns := []ConversationTurn{
		{Role: "user", Content: userMessage, TurnID: turnID + "_user"},
		{Role: "assistant", Content: assistantResponse, TurnID: turnID + "_assistant"},
	}

	_, err := a.engine.ProcessConversation(ctx, turns)
	return err
}

// ProcessTurnAsync processes a turn asynchronously.
// The callback is invoked when processing completes (may be nil).
func (a *Adapter) ProcessTurnAsync(userMessage, assistantResponse string, turnID string, callback func(error)) error {
	if !a.IsEnabled() {
		if callback != nil {
			callback(nil)
		}
		return nil
	}

	// Process in a goroutine with timeout to prevent indefinite blocking
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		turns := []ConversationTurn{
			{Role: "user", Content: userMessage, TurnID: turnID + "_user"},
			{Role: "assistant", Content: assistantResponse, TurnID: turnID + "_assistant"},
		}

		_, err := a.engine.ProcessConversation(ctx, turns)
		if err != nil {
			log.Printf("omem: ProcessTurnAsync failed: %v", err)
		}
		if callback != nil {
			callback(err)
		}
	}()

	return nil
}

// GetContext returns memory context to be injected into prompts.
// This is the primary method for enriching prompts with user memory.
func (a *Adapter) GetContext(ctx context.Context, query string, maxTokens int) string {
	if !a.IsEnabled() {
		return ""
	}

	result, err := a.engine.GetContextForPrompt(ctx, query, maxTokens)
	if err != nil {
		log.Printf("warning: omem failed to get context: %v", err)
		return ""
	}

	return result.FormattedContext
}

// GetSummary returns the user profile summary.
func (a *Adapter) GetSummary(ctx context.Context) string {
	if !a.IsEnabled() {
		return ""
	}

	return a.engine.GetSummary(ctx)
}

// GetRelevantFacts retrieves facts relevant to the query.
func (a *Adapter) GetRelevantFacts(ctx context.Context, query string, limit int) []string {
	if !a.IsEnabled() {
		return nil
	}

	facts, err := a.engine.GetFacts(ctx, query)
	if err != nil {
		log.Printf("warning: omem failed to retrieve facts: %v", err)
		return nil
	}

	// Apply limit
	if limit > 0 && len(facts) > limit {
		facts = facts[:limit]
	}

	result := make([]string, len(facts))
	for i, f := range facts {
		result[i] = f.Fact.AtomicText
	}

	return result
}

// AddManualFact allows explicit addition of facts.
func (a *Adapter) AddManualFact(ctx context.Context, text string, category string, importance float64) error {
	if !a.IsEnabled() {
		return nil
	}

	_, err := a.engine.ProcessText(ctx, text, "system")
	return err
}

// StartSession starts a new session with the given ID.
func (a *Adapter) StartSession(ctx context.Context, sessionID string) error {
	if !a.IsEnabled() {
		return nil
	}

	return a.engine.StartSession(ctx, sessionID)
}

// EndSession ends the current session.
func (a *Adapter) EndSession(ctx context.Context) error {
	if !a.IsEnabled() {
		return nil
	}

	return a.engine.EndSession(ctx)
}

// RefreshSummary triggers a summary refresh.
func (a *Adapter) RefreshSummary(ctx context.Context) error {
	if !a.IsEnabled() {
		return nil
	}

	return a.engine.RefreshSummary(ctx)
}

// GetStats returns memory statistics.
func (a *Adapter) GetStats(ctx context.Context) map[string]interface{} {
	if !a.IsEnabled() {
		return map[string]interface{}{"enabled": false}
	}

	return a.engine.GetStats(ctx)
}

// Close releases resources.
func (a *Adapter) Close() error {
	if a == nil || a.engine == nil {
		return nil
	}
	return a.engine.Close()
}

// SetExternalRAG sets the external RAG retriever for knowledge retrieval.
// When set, Omem will use the external RAG system as a knowledge source
// instead of or in addition to its internal storage.
func (a *Adapter) SetExternalRAG(retriever rag.Retriever) {
	a.externalRAG = retriever
	if a.engine != nil {
		a.engine.SetExternalRAG(retriever)
	}
}

// IsExternalRAGEnabled returns whether external RAG is configured.
func (a *Adapter) IsExternalRAGEnabled() bool {
	return a.externalRAG != nil
}

// FormatMemoryForPrompt formats memory context with a header for prompt inclusion.
func (a *Adapter) FormatMemoryForPrompt(ctx context.Context, query string, maxTokens int) string {
	memCtx := a.GetContext(ctx, query, maxTokens)
	if memCtx == "" {
		return ""
	}

	// Add a clear header for the memory section
	return fmt.Sprintf("<|Long-term Memory|>\n%s\n<|/Long-term Memory|>", memCtx)
}

// ============================================================================
// Pipeline Hook Integration
// ============================================================================

// PipelineHook provides hooks for integrating with the OpenEye pipeline.
// It implements the standard pipeline hook interface for memory systems.
type PipelineHook struct {
	adapter *Adapter
}

// NewPipelineHook creates pipeline integration hooks.
func NewPipelineHook(adapter *Adapter) *PipelineHook {
	return &PipelineHook{adapter: adapter}
}

// OnBeforeGenerate is called before LLM generation to enrich the context.
// Returns additional context to append to the prompt.
func (h *PipelineHook) OnBeforeGenerate(ctx context.Context, userMessage string, maxTokens int) string {
	if h == nil || h.adapter == nil || !h.adapter.IsEnabled() {
		return ""
	}

	return h.adapter.GetContext(ctx, userMessage, maxTokens)
}

// OnAfterGenerate is called after LLM generation to learn from the exchange.
// This processes the conversation turn asynchronously to avoid blocking responses.
func (h *PipelineHook) OnAfterGenerate(ctx context.Context, userMessage, assistantResponse, turnID string) {
	if h == nil || h.adapter == nil || !h.adapter.IsEnabled() {
		return
	}

	// Process asynchronously to not block the response
	log.Printf("omem: starting async processing for turn %s", turnID)
	_ = h.adapter.ProcessTurnAsync(userMessage, assistantResponse, turnID, func(err error) {
		if err != nil {
			log.Printf("omem: async processing failed for turn %s: %v", turnID, err)
		} else {
			log.Printf("omem: async processing completed for turn %s", turnID)
		}
	})
}

// OnSessionStart is called when a new session begins.
func (h *PipelineHook) OnSessionStart(ctx context.Context, sessionID string) {
	if h == nil || h.adapter == nil || !h.adapter.IsEnabled() {
		return
	}

	if err := h.adapter.StartSession(ctx, sessionID); err != nil {
		log.Printf("warning: omem session start failed: %v", err)
	}
}

// OnSessionEnd is called when a session ends.
func (h *PipelineHook) OnSessionEnd(ctx context.Context) {
	if h == nil || h.adapter == nil || !h.adapter.IsEnabled() {
		return
	}

	if err := h.adapter.EndSession(ctx); err != nil {
		log.Printf("warning: omem session end failed: %v", err)
	}
}

// ============================================================================
// Helper Functions
// ============================================================================

// BuildEnrichedContext combines omem context with existing context sources.
// This allows layering omem memory on top of other context providers.
func BuildEnrichedContext(omemContext, existingSummary, vectorContext string) string {
	var parts []string

	// Add omem context first (user profile and relevant facts)
	if omemContext != "" {
		parts = append(parts, omemContext)
	}

	// Add existing summary if not already included
	if existingSummary != "" && !strings.Contains(omemContext, existingSummary) {
		parts = append(parts, existingSummary)
	}

	// Add vector context
	if vectorContext != "" {
		parts = append(parts, "Relevant Recent Context:\n"+vectorContext)
	}

	return strings.Join(parts, "\n\n")
}

// NormalizeCategory converts a string to a valid FactCategory.
func NormalizeCategory(category string) FactCategory {
	switch strings.ToLower(strings.TrimSpace(category)) {
	case "preference", "preferences", "pref":
		return CategoryPreference
	case "belief", "beliefs":
		return CategoryBelief
	case "biographical", "biography", "bio":
		return CategoryBiographical
	case "event", "events":
		return CategoryEvent
	case "relationship", "relationships", "rel":
		return CategoryRelationship
	case "task", "tasks", "todo":
		return CategoryTask
	case "knowledge", "fact", "facts":
		return CategoryKnowledge
	default:
		return CategoryOther
	}
}

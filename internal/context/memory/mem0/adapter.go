package mem0

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
)

// Adapter wraps the mem0 Engine to integrate with the OpenEye pipeline.
// It provides a simpler interface for common operations.
type Adapter struct {
	engine *Engine
}

// NewAdapter creates a new mem0 adapter from configuration.
func NewAdapter(cfg config.Mem0Config, manager *runtime.Manager, embedder embedding.Provider) (*Adapter, error) {
	// Convert config types
	engineCfg := configFromMem0Config(cfg)

	if !engineCfg.Enabled {
		return &Adapter{engine: nil}, nil
	}

	engine, err := NewEngine(engineCfg, manager, embedder)
	if err != nil {
		return nil, fmt.Errorf("failed to create mem0 engine: %w", err)
	}

	return &Adapter{engine: engine}, nil
}

// configFromMem0Config converts the config package type to mem0 Config.
func configFromMem0Config(cfg config.Mem0Config) Config {
	return Config{
		Enabled: cfg.Enabled,
		Storage: StorageConfig{
			DBPath:          cfg.Storage.DBPath,
			EmbeddingDim:    cfg.Storage.EmbeddingDim,
			MaxFacts:        cfg.Storage.MaxFacts,
			PruneThreshold:  cfg.Storage.PruneThreshold,
			PruneKeepRecent: cfg.Storage.PruneKeepRecent,
		},
		Extraction: ExtractionConfig{
			Enabled:               cfg.Extraction.Enabled,
			BatchSize:             cfg.Extraction.BatchSize,
			Async:                 cfg.Extraction.Async,
			MinTextLength:         cfg.Extraction.MinTextLength,
			MaxFactsPerExtraction: cfg.Extraction.MaxFactsPerExtraction,
			ExtractEntities:       cfg.Extraction.ExtractEntities,
			ExtractRelationships:  cfg.Extraction.ExtractRelationships,
		},
		Updates: UpdateConfig{
			Enabled:              cfg.Updates.Enabled,
			ConflictThreshold:    cfg.Updates.ConflictThreshold,
			TopSimilarCount:      cfg.Updates.TopSimilarCount,
			AutoResolveConflicts: cfg.Updates.AutoResolveConflicts,
			TrackSupersession:    cfg.Updates.TrackSupersession,
		},
		Graph: GraphConfig{
			Enabled:                   cfg.Graph.Enabled,
			EntityResolution:          cfg.Graph.EntityResolution,
			EntitySimilarityThreshold: cfg.Graph.EntitySimilarityThreshold,
			MaxHops:                   cfg.Graph.MaxHops,
			TrackRelationshipHistory:  cfg.Graph.TrackRelationshipHistory,
		},
		Retrieval: RetrievalConfig{
			SemanticWeight:        cfg.Retrieval.SemanticWeight,
			ImportanceWeight:      cfg.Retrieval.ImportanceWeight,
			RecencyWeight:         cfg.Retrieval.RecencyWeight,
			AccessFrequencyWeight: cfg.Retrieval.AccessFrequencyWeight,
			MinScore:              cfg.Retrieval.MinScore,
			MaxResults:            cfg.Retrieval.MaxResults,
			IncludeGraphResults:   cfg.Retrieval.IncludeGraphResults,
			GraphResultsWeight:    cfg.Retrieval.GraphResultsWeight,
			RecencyHalfLifeHours:  cfg.Retrieval.RecencyHalfLifeHours,
		},
		Summary: SummaryConfig{
			Enabled:         cfg.Summary.Enabled,
			RefreshInterval: parseRefreshInterval(cfg.Summary.RefreshInterval),
			MaxFacts:        cfg.Summary.MaxFacts,
			MaxTokens:       cfg.Summary.MaxTokens,
			Async:           cfg.Summary.Async,
		},
	}
}

// parseRefreshInterval parses a duration string like "5m" or "1h".
func parseRefreshInterval(s string) time.Duration {
	if s == "" {
		return 5 * time.Minute
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return 5 * time.Minute
	}
	return d
}

// IsEnabled returns whether mem0 is enabled.
func (a *Adapter) IsEnabled() bool {
	return a != nil && a.engine != nil && a.engine.IsEnabled()
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

	return a.engine.ProcessConversation(ctx, turns)
}

// ProcessTurnAsync processes a turn asynchronously.
func (a *Adapter) ProcessTurnAsync(userMessage, assistantResponse string, turnID string, callback func(error)) error {
	if !a.IsEnabled() {
		if callback != nil {
			callback(nil)
		}
		return nil
	}

	turns := []ConversationTurn{
		{Role: "user", Content: userMessage, TurnID: turnID + "_user"},
		{Role: "assistant", Content: assistantResponse, TurnID: turnID + "_assistant"},
	}

	return a.engine.ProcessConversationAsync(turns, callback)
}

// GetContext returns memory context to be injected into prompts.
// This is the primary method for enriching prompts with user memory.
func (a *Adapter) GetContext(ctx context.Context, query string, maxTokens int) string {
	if !a.IsEnabled() {
		return ""
	}

	memoryContext, err := a.engine.GetContextForPrompt(ctx, query, maxTokens)
	if err != nil {
		log.Printf("warning: mem0 failed to get context: %v", err)
		return ""
	}

	return memoryContext
}

// GetSummary returns the user profile summary.
func (a *Adapter) GetSummary(ctx context.Context) string {
	if !a.IsEnabled() {
		return ""
	}

	summary, err := a.engine.GetSummary(ctx)
	if err != nil {
		log.Printf("warning: mem0 failed to get summary: %v", err)
		return ""
	}

	return summary
}

// GetRelevantFacts retrieves facts relevant to the query.
func (a *Adapter) GetRelevantFacts(ctx context.Context, query string, limit int) []string {
	if !a.IsEnabled() {
		return nil
	}

	facts, err := a.engine.Retrieve(ctx, query, limit)
	if err != nil {
		log.Printf("warning: mem0 failed to retrieve facts: %v", err)
		return nil
	}

	result := make([]string, len(facts))
	for i, f := range facts {
		result[i] = f.Text
	}

	return result
}

// AddManualFact allows explicit addition of facts.
func (a *Adapter) AddManualFact(ctx context.Context, text string, category string, importance float64) (int64, error) {
	if !a.IsEnabled() {
		return 0, nil
	}

	cat := normalizeCategory(category)
	return a.engine.AddFact(ctx, text, cat, importance)
}

// GetStats returns memory statistics.
func (a *Adapter) GetStats(ctx context.Context) map[string]interface{} {
	if !a.IsEnabled() {
		return map[string]interface{}{"enabled": false}
	}

	stats, err := a.engine.GetStats(ctx)
	if err != nil {
		return map[string]interface{}{"enabled": true, "error": err.Error()}
	}

	return stats
}

// Close releases resources.
func (a *Adapter) Close() error {
	if a == nil || a.engine == nil {
		return nil
	}
	return a.engine.Close()
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

// PipelineHook provides hooks for integrating with the OpenEye pipeline.
type PipelineHook struct {
	adapter *Adapter
}

// NewPipelineHook creates pipeline integration hooks.
func NewPipelineHook(adapter *Adapter) *PipelineHook {
	return &PipelineHook{adapter: adapter}
}

// OnBeforeGenerate is called before LLM generation to enrich the context.
// Returns additional context to append to the summary/context section.
func (h *PipelineHook) OnBeforeGenerate(ctx context.Context, userMessage string, maxTokens int) string {
	if h == nil || h.adapter == nil || !h.adapter.IsEnabled() {
		return ""
	}

	return h.adapter.GetContext(ctx, userMessage, maxTokens)
}

// OnAfterGenerate is called after LLM generation to learn from the exchange.
func (h *PipelineHook) OnAfterGenerate(ctx context.Context, userMessage, assistantResponse, turnID string) {
	if h == nil || h.adapter == nil || !h.adapter.IsEnabled() {
		return
	}

	// Process asynchronously to not block the response
	_ = h.adapter.ProcessTurnAsync(userMessage, assistantResponse, turnID, func(err error) {
		if err != nil {
			log.Printf("warning: mem0 async processing failed: %v", err)
		}
	})
}

// BuildEnrichedContext combines mem0 context with existing context sources.
func BuildEnrichedContext(mem0Context, existingSummary, vectorContext string) string {
	var parts []string

	// Add mem0 context first (user profile and relevant facts)
	if mem0Context != "" {
		parts = append(parts, mem0Context)
	}

	// Add existing summary
	if existingSummary != "" && !strings.Contains(mem0Context, existingSummary) {
		parts = append(parts, existingSummary)
	}

	// Add vector context
	if vectorContext != "" {
		parts = append(parts, "Relevant Recent Context:\n"+vectorContext)
	}

	return strings.Join(parts, "\n\n")
}

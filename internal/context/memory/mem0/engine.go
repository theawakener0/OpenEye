package mem0

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
)

// Engine is the main orchestrator for the mem0 memory system.
// It coordinates fact extraction, storage, updates, retrieval, and summarization.
type Engine struct {
	config Config

	// Core components
	factStore      *FactStore
	entityGraph    *EntityGraph
	extractor      *FactExtractor
	updater        *MemoryUpdater
	retriever      *HybridRetriever
	summaryManager *RollingSummaryManager

	// External dependencies
	manager  *runtime.Manager
	embedder embedding.Provider

	mu     sync.RWMutex
	closed bool
}

// NewEngine creates a new mem0 engine with all components initialized.
func NewEngine(cfg Config, manager *runtime.Manager, embedder embedding.Provider) (*Engine, error) {
	if manager == nil {
		return nil, errors.New("runtime manager required")
	}

	// Apply defaults and validate
	cfg = cfg.WithDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	if !cfg.Enabled {
		return &Engine{config: cfg, closed: true}, nil
	}

	engine := &Engine{
		config:   cfg,
		manager:  manager,
		embedder: embedder,
	}

	// Initialize fact store
	factStore, err := NewFactStore(cfg.Storage)
	if err != nil {
		return nil, fmt.Errorf("failed to create fact store: %w", err)
	}
	engine.factStore = factStore

	// Initialize entity graph (shares DB connection)
	if cfg.Graph.Enabled {
		entityGraph, err := NewEntityGraph(factStore.db, cfg.Graph)
		if err != nil {
			factStore.Close()
			return nil, fmt.Errorf("failed to create entity graph: %w", err)
		}
		engine.entityGraph = entityGraph
	}

	// Initialize fact extractor
	if cfg.Extraction.Enabled {
		extractor, err := NewFactExtractor(cfg.Extraction, manager, embedder)
		if err != nil {
			engine.Close()
			return nil, fmt.Errorf("failed to create fact extractor: %w", err)
		}
		engine.extractor = extractor
	}

	// Initialize memory updater
	if cfg.Updates.Enabled {
		updater, err := NewMemoryUpdater(cfg.Updates, factStore, engine.entityGraph, manager, embedder)
		if err != nil {
			engine.Close()
			return nil, fmt.Errorf("failed to create memory updater: %w", err)
		}
		engine.updater = updater
	}

	// Initialize hybrid retriever
	retriever, err := NewHybridRetriever(cfg.Retrieval, factStore, engine.entityGraph, embedder)
	if err != nil {
		engine.Close()
		return nil, fmt.Errorf("failed to create retriever: %w", err)
	}
	engine.retriever = retriever

	// Initialize rolling summary manager
	if cfg.Summary.Enabled {
		summaryManager, err := NewRollingSummaryManager(cfg.Summary, factStore, manager, embedder)
		if err != nil {
			engine.Close()
			return nil, fmt.Errorf("failed to create summary manager: %w", err)
		}
		engine.summaryManager = summaryManager
	}

	return engine, nil
}

// ProcessConversation extracts and stores facts from a conversation.
// This is the main entry point for learning from user interactions.
func (e *Engine) ProcessConversation(ctx context.Context, turns []ConversationTurn) error {
	if e == nil || e.closed {
		return errors.New("engine not initialized")
	}

	if !e.config.Enabled || e.extractor == nil {
		return nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	// Extract facts with embeddings
	factsWithEmb, entities, relationships, err := e.extractor.ExtractWithEmbeddings(ctx, turns)
	if err != nil {
		return fmt.Errorf("extraction failed: %w", err)
	}

	if len(factsWithEmb) == 0 {
		return nil
	}

	// Get turn IDs for provenance
	var turnIDs []string
	for _, turn := range turns {
		if turn.TurnID != "" {
			turnIDs = append(turnIDs, turn.TurnID)
		}
	}

	// Process facts through updater
	var processedFactIDs []int64
	if e.updater != nil {
		results, err := e.updater.ProcessFactBatch(ctx, factsWithEmb, turnIDs)
		if err != nil {
			log.Printf("warning: some facts failed to process: %v", err)
		}
		for _, result := range results {
			if result.NewFactID > 0 {
				processedFactIDs = append(processedFactIDs, result.NewFactID)
			}
			if result.UpdatedFactID > 0 {
				processedFactIDs = append(processedFactIDs, result.UpdatedFactID)
			}
		}
	} else {
		// Direct storage without update logic
		for _, fact := range factsWithEmb {
			newFact := Fact{
				Text:          fact.Text,
				Category:      fact.Category,
				Importance:    fact.Importance,
				Embedding:     fact.Embedding,
				CreatedAt:     time.Now(),
				SourceTurnIDs: strings.Join(turnIDs, ","),
			}
			id, err := e.factStore.InsertFact(ctx, newFact)
			if err == nil {
				processedFactIDs = append(processedFactIDs, id)
			}
		}
	}

	// Store entities and relationships in graph
	if e.entityGraph != nil && e.updater != nil {
		// Link entities to the first processed fact
		factID := int64(0)
		if len(processedFactIDs) > 0 {
			factID = processedFactIDs[0]
		}

		if err := e.updater.ProcessEntities(ctx, entities, factID); err != nil {
			log.Printf("warning: entity processing failed: %v", err)
		}
		if err := e.updater.ProcessRelationships(ctx, relationships, factID); err != nil {
			log.Printf("warning: relationship processing failed: %v", err)
		}
	}

	// Mark summary as dirty
	if e.summaryManager != nil {
		e.summaryManager.MarkDirty()
	}

	return nil
}

// ProcessConversationAsync processes a conversation asynchronously.
func (e *Engine) ProcessConversationAsync(turns []ConversationTurn, callback func(error)) error {
	if e == nil || e.closed {
		return errors.New("engine not initialized")
	}

	if !e.config.Enabled || e.extractor == nil {
		if callback != nil {
			callback(nil)
		}
		return nil
	}

	if !e.config.Extraction.Async {
		// Fall back to sync processing
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			err := e.ProcessConversation(ctx, turns)
			if callback != nil {
				callback(err)
			}
		}()
		return nil
	}

	return e.extractor.ExtractAsync(turns, func(facts []ExtractedFact, entities []ExtractedEntity, relationships []ExtractedRelationship, err error) {
		if err != nil {
			if callback != nil {
				callback(err)
			}
			return
		}

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		// Generate embeddings and process
		var turnIDs []string
		for _, turn := range turns {
			if turn.TurnID != "" {
				turnIDs = append(turnIDs, turn.TurnID)
			}
		}

		for _, fact := range facts {
			var emb []float32
			if e.embedder != nil {
				emb, _ = e.embedder.Embed(ctx, fact.Text)
			}

			if e.updater != nil {
				_, _ = e.updater.ProcessFact(ctx, ExtractedFactWithEmbedding{
					ExtractedFact: fact,
					Embedding:     emb,
				}, turnIDs)
			}
		}

		if e.summaryManager != nil {
			e.summaryManager.MarkDirty()
		}

		if callback != nil {
			callback(nil)
		}
	})
}

// Retrieve finds relevant facts for a given query.
func (e *Engine) Retrieve(ctx context.Context, query string, maxResults int) ([]Fact, error) {
	if e == nil || e.closed || e.retriever == nil {
		return nil, errors.New("engine not initialized")
	}

	if !e.config.Enabled {
		return nil, nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	return e.retriever.RetrieveForContext(ctx, query, maxResults)
}

// RetrieveWithScores returns facts along with their relevance scores.
func (e *Engine) RetrieveWithScores(ctx context.Context, query RetrievalQuery) ([]RetrievalResult, error) {
	if e == nil || e.closed || e.retriever == nil {
		return nil, errors.New("engine not initialized")
	}

	if !e.config.Enabled {
		return nil, nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	return e.retriever.Retrieve(ctx, query)
}

// GetSummary returns the current user summary.
func (e *Engine) GetSummary(ctx context.Context) (string, error) {
	if e == nil || e.closed || e.summaryManager == nil {
		return "", nil
	}

	if !e.config.Enabled || !e.config.Summary.Enabled {
		return "", nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	return e.summaryManager.GetSummary(ctx)
}

// GetContextForPrompt returns formatted memory context for inclusion in prompts.
// This is the primary method for enriching prompts with user memory.
func (e *Engine) GetContextForPrompt(ctx context.Context, query string, maxTokens int) (string, error) {
	if e == nil || e.closed {
		return "", nil
	}

	if !e.config.Enabled {
		return "", nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	var parts []string

	// Get summary if available
	if e.summaryManager != nil {
		summary := e.summaryManager.GetFormattedSummary(ctx)
		if summary != "" {
			parts = append(parts, summary)
		}
	}

	// Get relevant facts
	if e.retriever != nil && query != "" {
		facts, err := e.retriever.RetrieveForContext(ctx, query, e.config.Retrieval.MaxResults)
		if err == nil && len(facts) > 0 {
			factsFormatted := FormatFactsForPrompt(facts, maxTokens/2)
			if factsFormatted != "" {
				parts = append(parts, factsFormatted)
			}
		}
	}

	if len(parts) == 0 {
		return "", nil
	}

	return strings.Join(parts, "\n\n"), nil
}

// AddFact manually adds a fact to memory (for bootstrapping or explicit storage).
func (e *Engine) AddFact(ctx context.Context, text string, category FactCategory, importance float64) (int64, error) {
	if e == nil || e.closed || e.factStore == nil {
		return 0, errors.New("engine not initialized")
	}

	if !e.config.Enabled {
		return 0, nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	fact := Fact{
		Text:       text,
		Category:   category,
		Importance: clampFloat(importance, 0.0, 1.0),
		CreatedAt:  time.Now(),
	}

	// Generate embedding
	if e.embedder != nil {
		emb, err := e.embedder.Embed(ctx, text)
		if err == nil {
			fact.Embedding = emb
		}
	}

	id, err := e.factStore.InsertFact(ctx, fact)
	if err != nil {
		return 0, err
	}

	if e.summaryManager != nil {
		e.summaryManager.MarkDirty()
	}

	return id, nil
}

// DeleteFact marks a fact as obsolete.
func (e *Engine) DeleteFact(ctx context.Context, factID int64) error {
	if e == nil || e.closed || e.factStore == nil {
		return errors.New("engine not initialized")
	}

	if !e.config.Enabled {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.factStore.MarkObsolete(ctx, factID, nil); err != nil {
		return err
	}

	if e.summaryManager != nil {
		e.summaryManager.MarkDirty()
	}

	return nil
}

// GetStats returns statistics about the memory system.
func (e *Engine) GetStats(ctx context.Context) (map[string]interface{}, error) {
	if e == nil || e.closed {
		return nil, errors.New("engine not initialized")
	}

	stats := make(map[string]interface{})
	stats["enabled"] = e.config.Enabled

	if !e.config.Enabled {
		return stats, nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	// Fact store stats
	if e.factStore != nil {
		factStats, err := e.factStore.GetStats(ctx)
		if err == nil {
			stats["facts"] = factStats
		}
	}

	// Graph stats
	if e.entityGraph != nil {
		graphStats, err := e.entityGraph.GetStats(ctx)
		if err == nil {
			stats["graph"] = graphStats
		}
	}

	// Summary stats
	if e.summaryManager != nil {
		stats["summary"] = e.summaryManager.GetStats(ctx)
	}

	return stats, nil
}

// RefreshSummary forces a summary refresh.
func (e *Engine) RefreshSummary(ctx context.Context) error {
	if e == nil || e.closed || e.summaryManager == nil {
		return nil
	}

	if !e.config.Enabled || !e.config.Summary.Enabled {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	_, err := e.summaryManager.ForceRefresh(ctx)
	return err
}

// PruneFacts removes old, low-importance facts to stay within limits.
func (e *Engine) PruneFacts(ctx context.Context) (int, error) {
	if e == nil || e.closed || e.factStore == nil {
		return 0, errors.New("engine not initialized")
	}

	if !e.config.Enabled {
		return 0, nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	return e.factStore.PruneOldFacts(ctx)
}

// Close releases all resources.
func (e *Engine) Close() error {
	if e == nil {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}

	var firstErr error

	if e.summaryManager != nil {
		if err := e.summaryManager.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.extractor != nil {
		if err := e.extractor.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.entityGraph != nil {
		if err := e.entityGraph.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.factStore != nil {
		if err := e.factStore.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	e.closed = true
	return firstErr
}

// IsEnabled returns whether the mem0 system is enabled.
func (e *Engine) IsEnabled() bool {
	return e != nil && e.config.Enabled && !e.closed
}

// GetConfig returns the current configuration.
func (e *Engine) GetConfig() Config {
	if e == nil {
		return Config{}
	}
	return e.config
}

package omem

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"OpenEye/internal/rag"
)

// Engine is the main orchestrator for the Omem memory system.
// It coordinates all components:
// - AtomicEncoder: preprocesses text with coreference resolution and temporal anchoring
// - FactStore: persists facts with multi-view indexing (DuckDB)
// - EntityGraphLite: lightweight entity-relationship graph
// - AdaptiveRetriever: complexity-aware hybrid retrieval
// - RollingSummaryManager: incremental summary updates
// - EpisodeManager: session/episode tracking
// - ParallelProcessor: efficient batch processing
// - ExternalRAG: optional external knowledge base integration
type Engine struct {
	config Config
	mu     sync.RWMutex

	// Core components
	store     *FactStore
	encoder   *AtomicEncoder
	indexer   *MultiViewIndexer
	graph     *EntityGraphLite
	retriever *AdaptiveRetriever
	summary   *RollingSummaryManager
	episodes  *EpisodeManager
	processor *ParallelProcessor

	// External RAG for knowledge retrieval
	externalRAG rag.Retriever

	// LLM functions (configurable)
	llmGenerate   func(ctx context.Context, prompt string) (string, error)
	embeddingFunc func(ctx context.Context, text string) ([]float32, error)

	// State
	initialized bool
}

// NewEngine creates a new Omem engine with the given configuration.
func NewEngine(cfg Config) (*Engine, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	engine := &Engine{
		config: cfg,
	}

	return engine, nil
}

// Initialize initializes all engine components.
// This must be called before using the engine.
func (e *Engine) Initialize(
	llmGenerate func(ctx context.Context, prompt string) (string, error),
	embeddingFunc func(ctx context.Context, text string) ([]float32, error),
) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.initialized {
		return nil
	}

	e.llmGenerate = llmGenerate
	e.embeddingFunc = embeddingFunc

	var err error

	// Initialize fact store (DuckDB)
	e.store, err = NewFactStore(e.config.Storage)
	if err != nil {
		return err
	}

	// Initialize atomic encoder
	e.encoder = NewAtomicEncoder(e.config.AtomicEncoder, llmGenerate)

	// Initialize multi-view indexer
	e.indexer = NewMultiViewIndexer(e.config.MultiViewIndex, embeddingFunc)

	// Initialize entity graph (uses store's DB)
	if e.config.EntityGraph.Enabled {
		e.graph, err = NewEntityGraphLite(e.store.GetDB(), e.config.EntityGraph)
		if err != nil {
			// Non-fatal, continue without graph
			e.graph = nil
		}
	}

	// Initialize adaptive retriever
	e.retriever = NewAdaptiveRetriever(
		e.config.Retrieval,
		e.config.MultiViewIndex,
		e.store,
		e.indexer,
		e.graph,
		embeddingFunc,
	)

	// Initialize rolling summary manager
	if e.config.Summary.Enabled {
		e.summary = NewRollingSummaryManager(e.config.Summary, e.store, llmGenerate)
	}

	// Initialize episode manager
	if e.config.Episodes.Enabled {
		e.episodes = NewEpisodeManager(e.config.Episodes, e.store, llmGenerate)
	}

	// Initialize parallel processor
	if e.config.Parallel.EnableAsync {
		e.processor = NewParallelProcessor(e.config.Parallel)
		e.processor.Start()
	}

	e.initialized = true
	return nil
}

// ============================================================================
// Write Path: Processing Conversations
// ============================================================================

// ProcessConversation processes a conversation turn and extracts memories.
// This is the main write path for the memory system.
func (e *Engine) ProcessConversation(ctx context.Context, turns []ConversationTurn) (*ProcessingResult, error) {
	if !e.isReady() {
		return nil, ErrNotInitialized
	}

	result := &ProcessingResult{
		ProcessedAt: time.Now(),
	}

	// Combine turns into a single text for processing
	var textBuilder strings.Builder
	for _, turn := range turns {
		textBuilder.WriteString(turn.Role)
		textBuilder.WriteString(": ")
		textBuilder.WriteString(turn.Content)
		textBuilder.WriteString("\n")
	}
	text := textBuilder.String()

	// Step 1: Atomic encoding (coreference resolution, temporal anchoring, fact extraction)
	encoded, err := e.encoder.Encode(ctx, text, time.Now())
	if err != nil {
		return nil, err
	}
	result.EncodedText = encoded.AtomicText

	// Get current episode ID
	var episodeID int64
	if e.episodes != nil {
		episodeID = e.episodes.GetCurrentEpisodeID()
	}

	// Step 2: Process extracted facts
	var storedFacts []Fact
	var entityNames []string

	// Fallback: If no facts were extracted but we should store the text
	effectiveFacts := encoded.ExtractedFacts
	if len(effectiveFacts) == 0 {
		effectiveFacts = append(effectiveFacts, ExtractedFact{
			Text:       encoded.AtomicText,
			Category:   CategoryOther,
			Importance: 0.5,
		})
	}

	for _, ef := range effectiveFacts {
		// Create fact
		fact := Fact{
			Text:       ef.Text,
			AtomicText: ef.Text, // Already atomic from encoding
			Category:   ef.Category,
			Importance: ef.Importance,
			EpisodeID:  episodeID,
			CreatedAt:  time.Now(),
		}

		// Generate embedding from CLEAN text (without role prefix)
		// Use original turn content if available, otherwise use atomic text
		embeddingText := fact.AtomicText
		if len(turns) == 1 && turns[0].Content != "" {
			// Single turn - use clean content for better query matching
			embeddingText = strings.TrimSpace(turns[0].Content)
		}

		if e.embeddingFunc != nil {
			emb, err := e.embeddingFunc(ctx, embeddingText)
			if err == nil {
				fact.Embedding = emb
			}
		}

		// Index with multi-view indexer
		if e.indexer != nil {
			indexed, err := e.indexer.Index(ctx, ef.Text, fact.AtomicText, fact.Category, fact.Importance)
			if err == nil && indexed != nil {
				fact.Keywords = indexed.Keywords
				if fact.Embedding == nil {
					fact.Embedding = indexed.Embedding
				}
				// Metadata is a value type, always valid
				fact.TimestampAnchor = indexed.Metadata.TimestampAnchor
				fact.Location = indexed.Metadata.Location
				// Entities are on IndexedFact, not Metadata
				for _, ent := range indexed.Entities {
					entityNames = append(entityNames, ent.Name)
				}
			}
		}

		// Store fact
		factID, err := e.store.InsertFact(ctx, fact)
		if err != nil {
			continue
		}
		fact.ID = factID

		// Add entities to graph
		if e.graph != nil {
			for _, entity := range encoded.DiscoveredEntities {
				_, _ = e.graph.UpsertEntity(ctx, entity, factID)
			}

			// Extract and add relations
			relations := e.graph.ExtractRelationsFromFact(fact.AtomicText)
			for _, rel := range relations {
				_, _ = e.graph.AddRelation(ctx, rel, factID)
			}
		}

		storedFacts = append(storedFacts, fact)
	}

	result.ExtractedFacts = storedFacts
	result.FactCount = len(storedFacts)

	// Step 3: Update episode
	if e.episodes != nil {
		_ = e.episodes.OnTurnProcessed(ctx, storedFacts, entityNames)
	}

	// Step 4: Mark summary as dirty
	if e.summary != nil && len(storedFacts) > 0 {
		factIDs := make([]int64, len(storedFacts))
		for i, f := range storedFacts {
			factIDs[i] = f.ID
		}
		e.summary.MarkDirty(factIDs...)
	}

	return result, nil
}

// ProcessText processes a single text string (convenience method).
func (e *Engine) ProcessText(ctx context.Context, text string, role string) (*ProcessingResult, error) {
	if role == "" {
		role = "user"
	}
	return e.ProcessConversation(ctx, []ConversationTurn{{
		Role:    role,
		Content: text,
	}})
}

// ============================================================================
// Read Path: Retrieving Context
// ============================================================================

// GetContextForPrompt retrieves relevant memory context for a prompt.
// This is the main read path for the memory system.
// When external RAG is enabled, it will also retrieve knowledge from the external corpus.
func (e *Engine) GetContextForPrompt(ctx context.Context, prompt string, maxTokens int) (*ContextResult, error) {
	if !e.isReady() {
		return nil, ErrNotInitialized
	}

	result := &ContextResult{
		Query:       prompt,
		RetrievedAt: time.Now(),
	}

	// Check if external RAG is available
	useExternalRAG := e.externalRAG != nil

	// Step 1: Retrieve relevant facts from internal storage
	if e.retriever != nil {
		retrieval, err := e.retriever.Retrieve(ctx, RetrievalRequest{
			Query:       prompt,
			CurrentTime: time.Now(),
			MaxTokens:   maxTokens,
		})
		if err == nil && retrieval != nil {
			result.Facts = retrieval.Facts
			result.Complexity = retrieval.Complexity
			result.QueryType = retrieval.QueryType
		}
	}

	// Step 1b: Retrieve from external RAG if enabled
	if useExternalRAG {
		externalDocs, err := e.externalRAG.Retrieve(ctx, prompt, 5)
		if err == nil && len(externalDocs) > 0 {
			result.ExternalKnowledge = make([]string, 0, len(externalDocs))
			for _, doc := range externalDocs {
				result.ExternalKnowledge = append(result.ExternalKnowledge,
					fmt.Sprintf("[%s] %s", doc.Source, doc.Text))
			}
		}
	}

	// Step 2: Get rolling summary
	if e.summary != nil {
		summaryText := e.summary.GetSummaryText(ctx)
		result.Summary = summaryText
	}

	// Step 3: Format context
	result.FormattedContext = e.formatContextWithExternal(result)
	result.TokenEstimate = e.estimateTokens(result.FormattedContext)

	return result, nil
}

// GetFacts retrieves facts for a query (convenience method).
func (e *Engine) GetFacts(ctx context.Context, query string) ([]ScoredFact, error) {
	if !e.isReady() {
		return nil, ErrNotInitialized
	}

	if e.retriever == nil {
		return nil, ErrNotInitialized
	}

	return e.retriever.RetrieveForQuery(ctx, query)
}

// formatContext formats the retrieval results into a context string.
func (e *Engine) formatContext(result *ContextResult) string {
	var sb strings.Builder

	// Add summary if available
	if result.Summary != "" {
		sb.WriteString("User Summary:\n")
		sb.WriteString(result.Summary)
		sb.WriteString("\n\n")
	}

	// Add relevant facts
	if len(result.Facts) > 0 {
		sb.WriteString("Relevant Memories:\n")
		for _, sf := range result.Facts {
			sb.WriteString("- ")
			sb.WriteString(sf.Fact.AtomicText)
			sb.WriteString("\n")
		}
	}

	return strings.TrimSpace(sb.String())
}

// formatContextWithExternal formats the retrieval results into a context string,
// including external RAG knowledge if available.
func (e *Engine) formatContextWithExternal(result *ContextResult) string {
	var sb strings.Builder

	// Add summary if available
	if result.Summary != "" {
		sb.WriteString("User Summary:\n")
		sb.WriteString(result.Summary)
		sb.WriteString("\n\n")
	}

	// Add external knowledge first (higher priority for factual information)
	if len(result.ExternalKnowledge) > 0 {
		sb.WriteString("Knowledge Base:\n")
		for _, knowledge := range result.ExternalKnowledge {
			sb.WriteString("- ")
			sb.WriteString(knowledge)
			sb.WriteString("\n")
		}
		sb.WriteString("\n")
	}

	// Add relevant facts
	if len(result.Facts) > 0 {
		sb.WriteString("Relevant Memories:\n")
		for _, sf := range result.Facts {
			sb.WriteString("- ")
			sb.WriteString(sf.Fact.AtomicText)
			sb.WriteString("\n")
		}
	}

	return strings.TrimSpace(sb.String())
}

// estimateTokens estimates the token count for a text.
func (e *Engine) estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	// Rough estimate: ~4 characters per token
	return (len(text) + 3) / 4
}

// ============================================================================
// Session Management
// ============================================================================

// StartSession starts a new session with the given ID.
func (e *Engine) StartSession(ctx context.Context, sessionID string) error {
	if !e.isReady() {
		return ErrNotInitialized
	}

	if e.episodes != nil {
		_, err := e.episodes.StartSession(ctx, sessionID)
		return err
	}

	return nil
}

// EndSession ends the current session.
func (e *Engine) EndSession(ctx context.Context) error {
	if !e.isReady() {
		return nil
	}

	if e.episodes != nil {
		return e.episodes.EndSession(ctx)
	}

	return nil
}

// ============================================================================
// Summary Management
// ============================================================================

// RefreshSummary triggers a summary refresh.
func (e *Engine) RefreshSummary(ctx context.Context) error {
	if !e.isReady() || e.summary == nil {
		return nil
	}

	return e.summary.Refresh(ctx)
}

// GetSummary returns the current rolling summary.
func (e *Engine) GetSummary(ctx context.Context) string {
	if !e.isReady() || e.summary == nil {
		return ""
	}

	return e.summary.GetSummaryText(ctx)
}

// ============================================================================
// Statistics and Management
// ============================================================================

// GetStats returns comprehensive statistics about the memory system.
func (e *Engine) GetStats(ctx context.Context) map[string]interface{} {
	stats := make(map[string]interface{})

	stats["initialized"] = e.initialized

	if e.store != nil {
		storeStats, err := e.store.GetStats(ctx)
		if err == nil {
			stats["store"] = storeStats
		}
	}

	if e.graph != nil {
		graphStats, err := e.graph.GetStats(ctx)
		if err == nil {
			stats["graph"] = graphStats
		}
	}

	if e.retriever != nil {
		stats["retriever"] = e.retriever.GetRetrievalStats(ctx)
	}

	if e.summary != nil {
		stats["summary"] = map[string]interface{}{
			"is_dirty":           e.summary.IsDirty(),
			"pending_fact_count": e.summary.PendingFactCount(),
		}
	}

	if e.episodes != nil {
		stats["episodes"] = e.episodes.GetStats()
	}

	if e.processor != nil {
		stats["processor"] = e.processor.GetStats()
	}

	return stats
}

// Close gracefully shuts down the engine.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.initialized {
		return nil
	}

	var firstErr error

	// Stop background workers
	if e.summary != nil {
		e.summary.Stop()
	}

	if e.processor != nil {
		if err := e.processor.Stop(5 * time.Second); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	// Close components
	if e.graph != nil {
		if err := e.graph.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if e.store != nil {
		if err := e.store.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	e.initialized = false
	return firstErr
}

// isReady checks if the engine is initialized and ready.
func (e *Engine) isReady() bool {
	if e == nil {
		return false
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.initialized
}

// SetExternalRAG sets the external RAG retriever for knowledge retrieval.
// When set, Omem can use the external RAG system as a knowledge source.
func (e *Engine) SetExternalRAG(retriever rag.Retriever) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.externalRAG = retriever
}

// IsExternalRAGEnabled returns whether external RAG is configured.
func (e *Engine) IsExternalRAGEnabled() bool {
	if e == nil {
		return false
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.externalRAG != nil
}

// ============================================================================
// Result Types
// ============================================================================

// ProcessingResult contains the results of conversation processing.
type ProcessingResult struct {
	EncodedText    string    // Text after atomic encoding
	ExtractedFacts []Fact    // Facts extracted and stored
	FactCount      int       // Number of facts stored
	ProcessedAt    time.Time // Processing timestamp
}

// ContextResult contains retrieved memory context.
type ContextResult struct {
	Query             string           // Original query
	Facts             []ScoredFact     // Retrieved facts with scores
	Summary           string           // Rolling summary
	Complexity        ComplexityResult // Query complexity analysis
	QueryType         QueryType        // Classified query type
	ExternalKnowledge []string         // Knowledge from external RAG
	FormattedContext  string           // Formatted context for prompt
	TokenEstimate     int              // Estimated tokens
	RetrievedAt       time.Time        // Retrieval timestamp
}

// ============================================================================
// Errors
// ============================================================================

// Common engine errors.
var (
	ErrNotInitialized = engineError("omem engine not initialized")
)

type engineError string

func (e engineError) Error() string { return string(e) }

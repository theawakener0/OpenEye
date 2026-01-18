package omem

import (
	"context"
	"errors"
	"math"
	"sort"
	"strings"
	"time"
)

// AdaptiveRetriever provides complexity-aware hybrid retrieval with dynamic depth.
// Key features:
// - Multi-view retrieval: semantic + lexical + symbolic (from fact_store/multi_view_index)
// - Graph boost: entity/relationship-based relevance (from entity_graph_lite)
// - Adaptive depth: complexity-based k_dyn (from complexity_estimator)
// - Token-aware truncation: fits results to context budget
type AdaptiveRetriever struct {
	config        RetrievalConfig
	multiViewCfg  MultiViewConfig
	store         *FactStore
	indexer       *MultiViewIndexer
	graph         *EntityGraphLite
	complexity    *ComplexityEstimator
	embeddingFunc func(ctx context.Context, text string) ([]float32, error)
}

// RetrievalRequest contains parameters for a retrieval operation.
type RetrievalRequest struct {
	Query       string    // User query
	SessionID   string    // Current session ID (optional, for recency)
	CurrentTime time.Time // For recency calculations
	MaxTokens   int       // Maximum tokens to return (0 = use config)
	MinScore    float64   // Minimum score threshold (0 = use config)
	TopK        int       // Override for retrieval depth (0 = adaptive)
}

// RetrievalResult contains the results of a retrieval operation.
type RetrievalResult struct {
	Facts           []ScoredFact     // Retrieved facts with scores
	Complexity      ComplexityResult // Complexity analysis of query
	QueryType       QueryType        // Classified query type
	TotalCandidates int              // Total facts considered
	TruncatedCount  int              // Facts removed due to token budget
	TotalTokens     int              // Estimated tokens in result
	StrategyUsed    string           // Description of retrieval strategy
}

// NewAdaptiveRetriever creates a new adaptive retriever.
func NewAdaptiveRetriever(
	cfg RetrievalConfig,
	multiViewCfg MultiViewConfig,
	store *FactStore,
	indexer *MultiViewIndexer,
	graph *EntityGraphLite,
	embeddingFunc func(ctx context.Context, text string) ([]float32, error),
) *AdaptiveRetriever {
	cfg = applyRetrievalDefaults(cfg)

	return &AdaptiveRetriever{
		config:        cfg,
		multiViewCfg:  multiViewCfg,
		store:         store,
		indexer:       indexer,
		graph:         graph,
		complexity:    NewComplexityEstimator(cfg),
		embeddingFunc: embeddingFunc,
	}
}

// applyRetrievalDefaults fills in missing configuration values.
func applyRetrievalDefaults(cfg RetrievalConfig) RetrievalConfig {
	if cfg.DefaultTopK <= 0 {
		cfg.DefaultTopK = 5
	}
	if cfg.MaxTopK <= 0 {
		cfg.MaxTopK = 20
	}
	if cfg.ComplexityDelta <= 0 {
		cfg.ComplexityDelta = 2.0
	}
	if cfg.MinScore <= 0 {
		cfg.MinScore = 0.3
	}
	if cfg.MaxContextTokens <= 0 {
		cfg.MaxContextTokens = 1000
	}
	if cfg.RecencyHalfLifeHours <= 0 {
		cfg.RecencyHalfLifeHours = 168 // 1 week
	}
	if cfg.ImportanceWeight <= 0 {
		cfg.ImportanceWeight = 0.15
	}
	if cfg.RecencyWeight <= 0 {
		cfg.RecencyWeight = 0.1
	}
	if cfg.AccessFrequencyWeight <= 0 {
		cfg.AccessFrequencyWeight = 0.05
	}
	return cfg
}

// Retrieve performs adaptive retrieval for a query.
func (ar *AdaptiveRetriever) Retrieve(ctx context.Context, req RetrievalRequest) (*RetrievalResult, error) {
	if ar == nil || ar.store == nil {
		return nil, errors.New("adaptive retriever not initialized")
	}

	req.Query = strings.TrimSpace(req.Query)
	if req.Query == "" {
		return &RetrievalResult{}, nil
	}

	if req.CurrentTime.IsZero() {
		req.CurrentTime = time.Now()
	}

	// Step 1: Analyze query complexity
	complexity := ar.complexity.EstimateComplexity(req.Query)
	queryType := ar.complexity.ClassifyQuery(req.Query)

	// Step 2: Determine retrieval depth
	topK := req.TopK
	if topK <= 0 {
		topK = complexity.DynamicK
	}

	// Step 3: Extract keywords for lexical search
	keywords := ar.complexity.ExtractQueryKeywords(req.Query)

	// Step 4: Generate query embedding for semantic search
	var queryEmbedding []float32
	if ar.embeddingFunc != nil {
		emb, err := ar.embeddingFunc(ctx, req.Query)
		if err == nil {
			queryEmbedding = emb
		}
	}

	// Step 5: Perform hybrid search
	candidates, err := ar.hybridSearch(ctx, queryEmbedding, keywords, complexity.Entities, topK*3) // Over-retrieve for filtering
	if err != nil {
		return nil, err
	}

	totalCandidates := len(candidates)

	// Step 6: Apply graph boost
	candidates = ar.applyGraphBoost(ctx, candidates, complexity.Entities)

	// Step 7: Calculate final scores
	candidates = ar.calculateFinalScores(candidates, req.CurrentTime)

	// Step 8: Sort and filter
	minScore := req.MinScore
	if minScore <= 0 {
		minScore = ar.config.MinScore
	}

	candidates = ar.filterByScore(candidates, minScore)
	candidates = ar.sortByScore(candidates)

	// Step 9: Truncate to topK
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	// Step 10: Fit to token budget
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = ar.config.MaxContextTokens
	}

	truncatedBefore := len(candidates)
	candidates, totalTokens := ar.fitToTokenBudget(candidates, maxTokens)
	truncatedCount := truncatedBefore - len(candidates)

	// Build result
	result := &RetrievalResult{
		Facts:           candidates,
		Complexity:      complexity,
		QueryType:       queryType,
		TotalCandidates: totalCandidates,
		TruncatedCount:  truncatedCount,
		TotalTokens:     totalTokens,
		StrategyUsed:    ar.describeStrategy(queryType, complexity),
	}

	// Step 11: Update access counts for retrieved facts
	go ar.updateAccessCounts(ctx, candidates)

	return result, nil
}

// hybridSearch performs combined semantic + lexical search.
func (ar *AdaptiveRetriever) hybridSearch(
	ctx context.Context,
	queryEmbedding []float32,
	keywords []string,
	entities []string,
	limit int,
) ([]ScoredFact, error) {
	var results []ScoredFact

	// Semantic search (if embedding available)
	if len(queryEmbedding) > 0 && ar.store != nil {
		semanticResults, err := ar.store.SemanticSearch(ctx, queryEmbedding, limit)
		if err == nil {
			for _, sf := range semanticResults {
				sf.SemanticScore = sf.Score
				results = append(results, sf)
			}
		}
	}

	// Lexical search (if keywords available)
	if len(keywords) > 0 && ar.store != nil {
		lexicalResults, err := ar.store.FTSSearch(ctx, strings.Join(keywords, " "), limit)
		if err == nil {
			for _, sf := range lexicalResults {
				sf.LexicalScore = sf.Score
				results = append(results, sf)
			}
		}
	}

	// Entity-based retrieval (if entities detected and graph available)
	if len(entities) > 0 && ar.graph != nil {
		factIDs, err := ar.graph.GetFactsForEntities(ctx, entities)
		if err == nil && len(factIDs) > 0 {
			entityFacts, err := ar.store.GetFactsByIDs(ctx, factIDs)
			if err == nil {
				for _, fact := range entityFacts {
					results = append(results, ScoredFact{
						Fact:          fact,
						Score:         0.5, // Base score for entity-matched facts
						SymbolicScore: 0.5,
					})
				}
			}
		}
	}

	// Deduplicate by fact ID and merge scores
	return ar.deduplicateAndMerge(results), nil
}

// deduplicateAndMerge combines scores for duplicate facts.
func (ar *AdaptiveRetriever) deduplicateAndMerge(results []ScoredFact) []ScoredFact {
	factMap := make(map[int64]*ScoredFact)

	for _, sf := range results {
		if existing, ok := factMap[sf.Fact.ID]; ok {
			// Merge scores (take max of each type)
			if sf.SemanticScore > existing.SemanticScore {
				existing.SemanticScore = sf.SemanticScore
			}
			if sf.LexicalScore > existing.LexicalScore {
				existing.LexicalScore = sf.LexicalScore
			}
			if sf.SymbolicScore > existing.SymbolicScore {
				existing.SymbolicScore = sf.SymbolicScore
			}
		} else {
			copy := sf
			factMap[sf.Fact.ID] = &copy
		}
	}

	merged := make([]ScoredFact, 0, len(factMap))
	for _, sf := range factMap {
		merged = append(merged, *sf)
	}

	return merged
}

// applyGraphBoost adds graph-based relevance boost.
func (ar *AdaptiveRetriever) applyGraphBoost(ctx context.Context, results []ScoredFact, entities []string) []ScoredFact {
	if ar.graph == nil || len(entities) == 0 {
		return results
	}

	// Get fact IDs
	factIDs := make([]int64, len(results))
	for i, sf := range results {
		factIDs[i] = sf.Fact.ID
	}

	// Score by graph
	graphScores, err := ar.graph.ScoreByGraph(ctx, entities, factIDs)
	if err != nil {
		return results
	}

	// Apply boost
	for i := range results {
		if gs, ok := graphScores[results[i].Fact.ID]; ok {
			results[i].GraphBoost = gs.GraphScore
		}
	}

	return results
}

// calculateFinalScores computes the final weighted score for each fact.
func (ar *AdaptiveRetriever) calculateFinalScores(results []ScoredFact, currentTime time.Time) []ScoredFact {
	for i := range results {
		sf := &results[i]

		// Multi-view score (weighted combination)
		mvScore := ar.multiViewCfg.SemanticWeight*sf.SemanticScore +
			ar.multiViewCfg.LexicalWeight*sf.LexicalScore +
			ar.multiViewCfg.SymbolicWeight*sf.SymbolicScore

		// Add graph boost
		mvScore += sf.GraphBoost

		// Importance boost
		importanceBoost := sf.Fact.Importance * ar.config.ImportanceWeight

		// Recency boost (exponential decay)
		recencyBoost := ar.calculateRecencyBoost(sf.Fact.LastAccessed, currentTime)

		// Access frequency boost (log scale to prevent outliers)
		accessBoost := ar.calculateAccessBoost(sf.Fact.AccessCount)

		// Final score
		sf.RecencyScore = recencyBoost
		sf.Score = mvScore + importanceBoost + recencyBoost*ar.config.RecencyWeight + accessBoost*ar.config.AccessFrequencyWeight

		// Clamp to [0, 1] for consistency
		if sf.Score > 1.0 {
			sf.Score = 1.0
		}
		if sf.Score < 0.0 {
			sf.Score = 0.0
		}
	}

	return results
}

// calculateRecencyBoost calculates recency-based boost using exponential decay.
func (ar *AdaptiveRetriever) calculateRecencyBoost(lastAccessed, currentTime time.Time) float64 {
	if lastAccessed.IsZero() {
		return 0.0
	}

	hoursAgo := currentTime.Sub(lastAccessed).Hours()
	if hoursAgo < 0 {
		hoursAgo = 0
	}

	// Exponential decay: boost = exp(-ln(2) * hours / half_life)
	halfLife := ar.config.RecencyHalfLifeHours
	if halfLife <= 0 {
		halfLife = 168 // 1 week
	}

	decay := math.Exp(-math.Ln2 * hoursAgo / halfLife)
	return decay
}

// calculateAccessBoost calculates access frequency-based boost.
func (ar *AdaptiveRetriever) calculateAccessBoost(accessCount int) float64 {
	if accessCount <= 0 {
		return 0.0
	}

	// Log scale to prevent heavily accessed facts from dominating
	// Normalize to [0, 1] assuming max useful access count ~100
	boost := math.Log1p(float64(accessCount)) / math.Log1p(100)
	if boost > 1.0 {
		boost = 1.0
	}

	return boost
}

// filterByScore removes facts below the minimum score threshold.
func (ar *AdaptiveRetriever) filterByScore(results []ScoredFact, minScore float64) []ScoredFact {
	filtered := make([]ScoredFact, 0, len(results))
	for _, sf := range results {
		if sf.Score >= minScore {
			filtered = append(filtered, sf)
		}
	}
	return filtered
}

// sortByScore sorts facts by score in descending order.
func (ar *AdaptiveRetriever) sortByScore(results []ScoredFact) []ScoredFact {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	return results
}

// fitToTokenBudget truncates results to fit within the token budget.
func (ar *AdaptiveRetriever) fitToTokenBudget(results []ScoredFact, maxTokens int) ([]ScoredFact, int) {
	if maxTokens <= 0 {
		return results, ar.estimateTotalTokens(results)
	}

	totalTokens := 0
	truncated := make([]ScoredFact, 0, len(results))

	for _, sf := range results {
		tokens := ar.estimateTokens(sf.Fact.AtomicText)
		if totalTokens+tokens > maxTokens {
			break
		}
		totalTokens += tokens
		truncated = append(truncated, sf)
	}

	return truncated, totalTokens
}

// estimateTokens estimates the token count for a text.
// Uses a simple heuristic: ~4 characters per token for English.
func (ar *AdaptiveRetriever) estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	// Rough estimate: 4 chars per token, minimum 1 token
	tokens := (len(text) + 3) / 4
	if tokens < 1 {
		tokens = 1
	}
	return tokens
}

// estimateTotalTokens estimates total tokens across all facts.
func (ar *AdaptiveRetriever) estimateTotalTokens(results []ScoredFact) int {
	total := 0
	for _, sf := range results {
		total += ar.estimateTokens(sf.Fact.AtomicText)
	}
	return total
}

// updateAccessCounts updates access timestamps and counts for retrieved facts.
func (ar *AdaptiveRetriever) updateAccessCounts(ctx context.Context, results []ScoredFact) {
	if ar.store == nil {
		return
	}

	for _, sf := range results {
		_ = ar.store.UpdateAccess(ctx, sf.Fact.ID)
	}
}

// describeStrategy returns a description of the retrieval strategy used.
func (ar *AdaptiveRetriever) describeStrategy(queryType QueryType, complexity ComplexityResult) string {
	var parts []string

	// Query type
	parts = append(parts, "type:"+string(queryType))

	// Complexity level
	level := "low"
	if complexity.TotalScore > 0.6 {
		level = "high"
	} else if complexity.TotalScore > 0.3 {
		level = "medium"
	}
	parts = append(parts, "complexity:"+level)

	// Dynamic K
	parts = append(parts, "k:"+itoa(complexity.DynamicK))

	return strings.Join(parts, ",")
}

// ============================================================================
// Query-Specific Retrieval Strategies
// ============================================================================

// RetrieveForQuery is a simplified interface for common retrieval.
func (ar *AdaptiveRetriever) RetrieveForQuery(ctx context.Context, query string) ([]ScoredFact, error) {
	result, err := ar.Retrieve(ctx, RetrievalRequest{
		Query:       query,
		CurrentTime: time.Now(),
	})
	if err != nil {
		return nil, err
	}
	return result.Facts, nil
}

// RetrieveWithContext retrieves facts and returns them formatted for context.
func (ar *AdaptiveRetriever) RetrieveWithContext(ctx context.Context, query string, maxTokens int) (string, error) {
	result, err := ar.Retrieve(ctx, RetrievalRequest{
		Query:       query,
		CurrentTime: time.Now(),
		MaxTokens:   maxTokens,
	})
	if err != nil {
		return "", err
	}

	return ar.FormatFactsForContext(result.Facts), nil
}

// FormatFactsForContext formats retrieved facts for inclusion in prompts.
func (ar *AdaptiveRetriever) FormatFactsForContext(facts []ScoredFact) string {
	if len(facts) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Relevant memories:\n")

	for i, sf := range facts {
		sb.WriteString("- ")
		sb.WriteString(sf.Fact.AtomicText)
		if i < len(facts)-1 {
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

// ============================================================================
// Batch Operations
// ============================================================================

// BatchRetrieve performs retrieval for multiple queries.
func (ar *AdaptiveRetriever) BatchRetrieve(ctx context.Context, queries []string, maxTokensPerQuery int) ([]RetrievalResult, error) {
	results := make([]RetrievalResult, len(queries))

	for i, query := range queries {
		result, err := ar.Retrieve(ctx, RetrievalRequest{
			Query:       query,
			CurrentTime: time.Now(),
			MaxTokens:   maxTokensPerQuery,
		})
		if err != nil {
			continue
		}
		results[i] = *result
	}

	return results, nil
}

// ============================================================================
// Statistics
// ============================================================================

// GetRetrievalStats returns statistics about retrieval operations.
func (ar *AdaptiveRetriever) GetRetrievalStats(ctx context.Context) map[string]interface{} {
	stats := make(map[string]interface{})

	if ar.store != nil {
		storeStats, err := ar.store.GetStats(ctx)
		if err == nil {
			stats["store"] = storeStats
		}
	}

	if ar.graph != nil {
		graphStats, err := ar.graph.GetStats(ctx)
		if err == nil {
			stats["graph"] = graphStats
		}
	}

	stats["config"] = map[string]interface{}{
		"default_top_k":      ar.config.DefaultTopK,
		"max_top_k":          ar.config.MaxTopK,
		"complexity_delta":   ar.config.ComplexityDelta,
		"min_score":          ar.config.MinScore,
		"max_context_tokens": ar.config.MaxContextTokens,
		"complexity_enabled": ar.config.EnableComplexityEstimation,
	}

	return stats
}

// ============================================================================
// Helper Functions
// ============================================================================

// itoa converts an int to string (simple implementation).
func itoa(n int) string {
	if n == 0 {
		return "0"
	}

	var digits []byte
	negative := n < 0
	if negative {
		n = -n
	}

	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}

	if negative {
		digits = append([]byte{'-'}, digits...)
	}

	return string(digits)
}

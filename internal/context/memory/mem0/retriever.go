package mem0

import (
	"context"
	"errors"
	"math"
	"sort"
	"strings"
	"time"

	"OpenEye/internal/embedding"
)

// HybridRetriever implements multi-signal ranking for memory retrieval.
// It combines semantic similarity, importance, recency, and access frequency.
type HybridRetriever struct {
	config   RetrievalConfig
	store    *FactStore
	graph    *EntityGraph
	embedder embedding.Provider
}

// RetrievalResult represents a scored fact from retrieval.
type RetrievalResult struct {
	Fact            Fact
	Score           float64
	SemanticScore   float64
	ImportanceScore float64
	RecencyScore    float64
	FrequencyScore  float64
	GraphBoost      float64
	Source          string // "semantic", "graph", "recent", "category"
}

// RetrievalQuery specifies what to retrieve.
type RetrievalQuery struct {
	Text         string         // Query text for embedding
	Embedding    []float32      // Pre-computed embedding (optional)
	Categories   []FactCategory // Filter by categories (optional)
	MaxResults   int            // Maximum results to return
	MinScore     float64        // Minimum score threshold
	IncludeGraph bool           // Include graph-based results
}

// NewHybridRetriever creates a new hybrid retriever.
func NewHybridRetriever(cfg RetrievalConfig, store *FactStore, graph *EntityGraph, embedder embedding.Provider) (*HybridRetriever, error) {
	if store == nil {
		return nil, errors.New("fact store required")
	}

	cfg = applyRetrievalDefaults(cfg)

	return &HybridRetriever{
		config:   cfg,
		store:    store,
		graph:    graph,
		embedder: embedder,
	}, nil
}

// applyRetrievalDefaults fills in missing configuration values.
func applyRetrievalDefaults(cfg RetrievalConfig) RetrievalConfig {
	// Ensure weights sum to 1.0
	sum := cfg.SemanticWeight + cfg.ImportanceWeight + cfg.RecencyWeight + cfg.AccessFrequencyWeight
	if sum <= 0 {
		cfg.SemanticWeight = 0.50
		cfg.ImportanceWeight = 0.25
		cfg.RecencyWeight = 0.15
		cfg.AccessFrequencyWeight = 0.10
	} else if sum != 1.0 {
		cfg.SemanticWeight /= sum
		cfg.ImportanceWeight /= sum
		cfg.RecencyWeight /= sum
		cfg.AccessFrequencyWeight /= sum
	}

	if cfg.MinScore <= 0 {
		cfg.MinScore = 0.3
	}
	if cfg.MaxResults <= 0 {
		cfg.MaxResults = 20
	}
	if cfg.GraphResultsWeight <= 0 {
		cfg.GraphResultsWeight = 0.3
	}
	if cfg.RecencyHalfLifeHours <= 0 {
		cfg.RecencyHalfLifeHours = 168 // 1 week
	}

	return cfg
}

// Retrieve performs hybrid retrieval based on the query.
func (r *HybridRetriever) Retrieve(ctx context.Context, query RetrievalQuery) ([]RetrievalResult, error) {
	if r == nil || r.store == nil {
		return nil, errors.New("hybrid retriever not initialized")
	}

	if query.MaxResults <= 0 {
		query.MaxResults = r.config.MaxResults
	}
	if query.MinScore <= 0 {
		query.MinScore = r.config.MinScore
	}

	// Get query embedding
	queryEmbedding := query.Embedding
	if len(queryEmbedding) == 0 && query.Text != "" && r.embedder != nil {
		emb, err := r.embedder.Embed(ctx, query.Text)
		if err == nil {
			queryEmbedding = emb
		}
	}

	var results []RetrievalResult

	// Semantic search if we have an embedding
	if len(queryEmbedding) > 0 {
		semanticResults, err := r.semanticSearch(ctx, queryEmbedding, query.MaxResults*2)
		if err == nil {
			results = append(results, semanticResults...)
		}
	}

	// Graph-based retrieval if enabled
	if (query.IncludeGraph || r.config.IncludeGraphResults) && r.graph != nil && query.Text != "" {
		graphResults, err := r.graphSearch(ctx, query.Text, query.MaxResults)
		if err == nil {
			results = mergeResults(results, graphResults)
		}
	}

	// If no semantic results, fall back to recent facts
	if len(results) == 0 {
		recentResults, err := r.recentSearch(ctx, query.MaxResults)
		if err == nil {
			results = recentResults
		}
	}

	// Filter by categories if specified
	if len(query.Categories) > 0 {
		results = filterByCategories(results, query.Categories)
	}

	// Calculate final hybrid scores
	r.calculateHybridScores(results)

	// Sort by final score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Apply minimum score threshold
	var filtered []RetrievalResult
	for _, result := range results {
		if result.Score >= query.MinScore {
			filtered = append(filtered, result)
		}
	}

	// Limit results
	if len(filtered) > query.MaxResults {
		filtered = filtered[:query.MaxResults]
	}

	// Update access counts for returned facts
	for _, result := range filtered {
		_ = r.store.UpdateAccess(ctx, result.Fact.ID)
	}

	return filtered, nil
}

// semanticSearch performs embedding-based similarity search.
func (r *HybridRetriever) semanticSearch(ctx context.Context, queryEmbedding []float32, limit int) ([]RetrievalResult, error) {
	facts, similarities, err := r.store.SearchSimilarFacts(ctx, queryEmbedding, limit, false)
	if err != nil {
		return nil, err
	}

	results := make([]RetrievalResult, len(facts))
	for i, fact := range facts {
		results[i] = RetrievalResult{
			Fact:          fact,
			SemanticScore: similarities[i],
			Source:        "semantic",
		}
	}

	return results, nil
}

// graphSearch finds facts connected to entities mentioned in the query.
func (r *HybridRetriever) graphSearch(ctx context.Context, queryText string, limit int) ([]RetrievalResult, error) {
	if r.graph == nil {
		return nil, nil
	}

	// Search for entities mentioned in query
	entities, err := r.graph.SearchEntitiesByName(ctx, queryText, 5)
	if err != nil || len(entities) == 0 {
		return nil, nil
	}

	// Collect facts from entity traversal
	factIDSet := make(map[int64]bool)
	var graphResults []RetrievalResult

	for _, entity := range entities {
		// Get facts directly linked to entity
		factIDs, err := r.graph.GetFactsForEntity(ctx, entity.ID)
		if err != nil {
			continue
		}

		for _, factID := range factIDs {
			if factIDSet[factID] {
				continue
			}
			factIDSet[factID] = true

			fact, err := r.store.GetFactByID(ctx, factID)
			if err != nil || fact == nil || fact.IsObsolete {
				continue
			}

			graphResults = append(graphResults, RetrievalResult{
				Fact:       *fact,
				GraphBoost: r.config.GraphResultsWeight,
				Source:     "graph",
			})
		}

		// Traverse relationships for connected facts
		paths, err := r.graph.TraverseFrom(ctx, entity.ID, r.config.MaxResults)
		if err != nil {
			continue
		}

		for _, path := range paths {
			for _, rel := range path.Relationships {
				if rel.FactID > 0 && !factIDSet[rel.FactID] {
					factIDSet[rel.FactID] = true

					fact, err := r.store.GetFactByID(ctx, rel.FactID)
					if err != nil || fact == nil || fact.IsObsolete {
						continue
					}

					// Reduce boost for facts found through traversal
					boost := r.config.GraphResultsWeight * (1.0 / float64(path.TotalHops+1))
					graphResults = append(graphResults, RetrievalResult{
						Fact:       *fact,
						GraphBoost: boost,
						Source:     "graph",
					})
				}
			}
		}

		if len(graphResults) >= limit {
			break
		}
	}

	return graphResults, nil
}

// recentSearch retrieves the most recent facts as a fallback.
func (r *HybridRetriever) recentSearch(ctx context.Context, limit int) ([]RetrievalResult, error) {
	facts, err := r.store.GetRecentFacts(ctx, limit, false)
	if err != nil {
		return nil, err
	}

	results := make([]RetrievalResult, len(facts))
	for i, fact := range facts {
		results[i] = RetrievalResult{
			Fact:   fact,
			Source: "recent",
		}
	}

	return results, nil
}

// calculateHybridScores computes the final weighted score for each result.
func (r *HybridRetriever) calculateHybridScores(results []RetrievalResult) {
	now := time.Now()
	halfLifeHours := r.config.RecencyHalfLifeHours

	// Find max access count for normalization
	maxAccessCount := 1
	for _, result := range results {
		if result.Fact.AccessCount > maxAccessCount {
			maxAccessCount = result.Fact.AccessCount
		}
	}

	for i := range results {
		result := &results[i]

		// Semantic score (already calculated for semantic results)
		semanticScore := result.SemanticScore

		// Importance score (direct from fact)
		importanceScore := result.Fact.Importance

		// Recency score (exponential decay)
		ageHours := now.Sub(result.Fact.CreatedAt).Hours()
		recencyScore := math.Pow(0.5, ageHours/halfLifeHours)

		// Access frequency score (normalized)
		frequencyScore := float64(result.Fact.AccessCount) / float64(maxAccessCount)

		// Store component scores
		result.ImportanceScore = importanceScore
		result.RecencyScore = recencyScore
		result.FrequencyScore = frequencyScore

		// Calculate weighted hybrid score
		hybridScore := semanticScore*r.config.SemanticWeight +
			importanceScore*r.config.ImportanceWeight +
			recencyScore*r.config.RecencyWeight +
			frequencyScore*r.config.AccessFrequencyWeight

		// Apply graph boost if present
		if result.GraphBoost > 0 {
			hybridScore = hybridScore*(1.0-result.GraphBoost) + result.GraphBoost
		}

		result.Score = hybridScore
	}
}

// mergeResults combines results from different sources, deduplicating by fact ID.
func mergeResults(existing, new []RetrievalResult) []RetrievalResult {
	seen := make(map[int64]int) // factID -> index in result

	for i, r := range existing {
		seen[r.Fact.ID] = i
	}

	for _, r := range new {
		if existingIdx, ok := seen[r.Fact.ID]; ok {
			// Merge scores - take the max of each component
			existing[existingIdx].SemanticScore = math.Max(existing[existingIdx].SemanticScore, r.SemanticScore)
			existing[existingIdx].GraphBoost = math.Max(existing[existingIdx].GraphBoost, r.GraphBoost)
		} else {
			existing = append(existing, r)
			seen[r.Fact.ID] = len(existing) - 1
		}
	}

	return existing
}

// filterByCategories filters results to only include specified categories.
func filterByCategories(results []RetrievalResult, categories []FactCategory) []RetrievalResult {
	categorySet := make(map[FactCategory]bool)
	for _, cat := range categories {
		categorySet[cat] = true
	}

	var filtered []RetrievalResult
	for _, r := range results {
		if categorySet[r.Fact.Category] {
			filtered = append(filtered, r)
		}
	}

	return filtered
}

// RetrieveByCategory retrieves facts from a specific category.
func (r *HybridRetriever) RetrieveByCategory(ctx context.Context, category FactCategory, limit int) ([]RetrievalResult, error) {
	if r == nil || r.store == nil {
		return nil, errors.New("hybrid retriever not initialized")
	}

	if limit <= 0 {
		limit = r.config.MaxResults
	}

	facts, err := r.store.GetFactsByCategory(ctx, category, limit)
	if err != nil {
		return nil, err
	}

	results := make([]RetrievalResult, len(facts))
	for i, fact := range facts {
		results[i] = RetrievalResult{
			Fact:   fact,
			Source: "category",
		}
	}

	r.calculateHybridScores(results)

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// RetrieveForContext retrieves facts most relevant for a given context string.
// This is a convenience method for common use cases.
func (r *HybridRetriever) RetrieveForContext(ctx context.Context, contextText string, maxFacts int) ([]Fact, error) {
	results, err := r.Retrieve(ctx, RetrievalQuery{
		Text:       contextText,
		MaxResults: maxFacts,
	})
	if err != nil {
		return nil, err
	}

	facts := make([]Fact, len(results))
	for i, result := range results {
		facts[i] = result.Fact
	}

	return facts, nil
}

// GetTopFacts retrieves the top facts by a combination of importance and recency.
// Useful for getting a snapshot of the most relevant memories.
func (r *HybridRetriever) GetTopFacts(ctx context.Context, limit int) ([]Fact, error) {
	if r == nil || r.store == nil {
		return nil, errors.New("hybrid retriever not initialized")
	}

	if limit <= 0 {
		limit = r.config.MaxResults
	}

	// Get recent facts
	recentResults, err := r.recentSearch(ctx, limit*2)
	if err != nil {
		return nil, err
	}

	// Calculate scores
	r.calculateHybridScores(recentResults)

	// Sort by score
	sort.Slice(recentResults, func(i, j int) bool {
		return recentResults[i].Score > recentResults[j].Score
	})

	// Limit and extract facts
	if len(recentResults) > limit {
		recentResults = recentResults[:limit]
	}

	facts := make([]Fact, len(recentResults))
	for i, result := range recentResults {
		facts[i] = result.Fact
	}

	return facts, nil
}

// FormatFactsForPrompt formats retrieved facts for inclusion in a prompt.
func FormatFactsForPrompt(facts []Fact, maxTokens int) string {
	if len(facts) == 0 {
		return ""
	}

	var result strings.Builder
	result.WriteString("Known facts about the user:\n")

	estimatedTokens := 10 // Header
	for _, fact := range facts {
		factLine := "- " + fact.Text + "\n"
		lineTokens := len(factLine) / 4 // Rough estimate

		if estimatedTokens+lineTokens > maxTokens {
			break
		}

		result.WriteString(factLine)
		estimatedTokens += lineTokens
	}

	return result.String()
}

package rag

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
	"fmt"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/embedding"
)

type HybridRetrieverConfig struct {
	RAGConfig config.RAGConfig

	MaxCandidates      int     // Number of candidates to consider before reranking
	DiversityThreshold float64 // Minimum distance between selected documents (0-1)
	
	// Scoring weights
	SemanticWeight   float64 // Weight for semantic similarity (0-1)
	KeywordWeight    float64 // Weight for keyword matching (0-1)
	RecencyWeight    float64 // Weight for document recency (0-1)
	
	// Query expansion
	EnableQueryExpansion bool
	MaxQueryTerms        int
	
	// Result filtering
	MinTokens       int // Minimum tokens per chunk
	MaxTokens       int // Maximum tokens per chunk
	DedupeThreshold float64 // Similarity threshold for deduplication
}

func DefaultHybridRetrieverConfig(ragCfg config.RAGConfig) HybridRetrieverConfig {
	return HybridRetrieverConfig{
		RAGConfig:          ragCfg,
		MaxCandidates:      50,
		DiversityThreshold: 0.3,
		SemanticWeight:     0.7,
		KeywordWeight:      0.2,
		RecencyWeight:      0.1,
		EnableQueryExpansion: true,
		MaxQueryTerms:      10,
		MinTokens:          10,
		MaxTokens:          500,
		DedupeThreshold:    0.85,
	}
}

type HybridRetriever struct {
	*FilesystemRetriever
	config   HybridRetrieverConfig
	embedder embedding.Provider
	
	queryCache     sync.Map
	queryCacheTTL  time.Duration
}

func NewHybridRetriever(cfg HybridRetrieverConfig, embedder embedding.Provider) (*HybridRetriever, error) {
	base, err := NewFilesystemRetriever(cfg.RAGConfig, embedder)
	if err != nil {
		return nil, err
	}
	if base == nil {
		return nil, nil
	}

	return &HybridRetriever{
		FilesystemRetriever: base,
		config:              cfg,
		embedder:            embedder,
		queryCacheTTL:       5 * time.Minute,
	}, nil
}

// Retrieve performs hybrid retrieval with reranking and diversity selection.
func (r *HybridRetriever) Retrieve(ctx context.Context, query string, limit int) ([]Document, error) {
	if r == nil || r.FilesystemRetriever == nil {
		return nil, nil
	}

	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}

	if limit <= 0 {
		limit = 4
	}

	// Get more candidates for reranking
	candidateLimit := r.config.MaxCandidates
	if candidateLimit < limit*3 {
		candidateLimit = limit * 3
	}

	// Get base semantic results
	candidates, err := r.FilesystemRetriever.Retrieve(ctx, query, candidateLimit)
	if err != nil {
		return nil, err
	}

	if len(candidates) == 0 {
		return nil, nil
	}

	// Apply hybrid scoring
	scored := r.hybridScore(query, candidates)

	// Apply diversity selection using MMR (Maximal Marginal Relevance)
	selected := r.mmrSelect(scored, limit)

	// Deduplicate similar results
	deduplicated := r.deduplicate(selected)

	if len(deduplicated) > limit {
		deduplicated = deduplicated[:limit]
	}

	return deduplicated, nil
}

type scoredDocument struct {
	doc            Document
	semanticScore  float64
	keywordScore   float64
	recencyScore   float64
	combinedScore  float64
	embedding      []float32
}

func (r *HybridRetriever) hybridScore(query string, candidates []Document) []scoredDocument {
	queryTerms := extractKeywords(query)
	
	scored := make([]scoredDocument, 0, len(candidates))
	
	for _, doc := range candidates {
		sd := scoredDocument{
			doc:           doc,
			semanticScore: doc.Score,
		}

		// Compute keyword score (BM25-like)
		sd.keywordScore = computeKeywordScore(queryTerms, doc.Text)

		// Compute recency score (assumes more recent = higher score, based on ID)
		sd.recencyScore = computeRecencyScore(doc.ID)

		// Combined weighted score
		sd.combinedScore = r.config.SemanticWeight*sd.semanticScore +
			r.config.KeywordWeight*sd.keywordScore +
			r.config.RecencyWeight*sd.recencyScore

		scored = append(scored, sd)
	}

	// Sort by combined score
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].combinedScore > scored[j].combinedScore
	})

	return scored
}

// mmrSelect uses Maximal Marginal Relevance to select diverse results.
func (r *HybridRetriever) mmrSelect(candidates []scoredDocument, limit int) []Document {
	if len(candidates) == 0 {
		return nil
	}

	lambda := 0.7 // Balance between relevance and diversity
	
	selected := make([]Document, 0, limit)
	selectedTexts := make([]string, 0, limit)
	remaining := make([]scoredDocument, len(candidates))
	copy(remaining, candidates)

	// Always select the top result first
	if len(remaining) > 0 {
		selected = append(selected, remaining[0].doc)
		selectedTexts = append(selectedTexts, remaining[0].doc.Text)
		remaining = remaining[1:]
	}

	// Iteratively select based on MMR
	for len(selected) < limit && len(remaining) > 0 {
		bestIdx := -1
		bestMMR := math.Inf(-1)

		for i, cand := range remaining {
			// Relevance term
			relevance := cand.combinedScore

			// Diversity term: max similarity to already selected
			maxSim := 0.0
			for _, selText := range selectedTexts {
				sim := textSimilarity(cand.doc.Text, selText)
				if sim > maxSim {
					maxSim = sim
				}
			}

			// MMR score
			mmr := lambda*relevance - (1-lambda)*maxSim

			if mmr > bestMMR {
				bestMMR = mmr
				bestIdx = i
			}
		}

		if bestIdx >= 0 {
			selected = append(selected, remaining[bestIdx].doc)
			selectedTexts = append(selectedTexts, remaining[bestIdx].doc.Text)
			// Remove selected from remaining
			remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
		} else {
			break
		}
	}

	return selected
}

// deduplicate removes near-duplicate documents.
func (r *HybridRetriever) deduplicate(docs []Document) []Document {
	if len(docs) <= 1 {
		return docs
	}

	result := make([]Document, 0, len(docs))
	
	for _, doc := range docs {
		isDupe := false
		for _, existing := range result {
			sim := textSimilarity(doc.Text, existing.Text)
			if sim > r.config.DedupeThreshold {
				isDupe = true
				break
			}
		}
		if !isDupe {
			result = append(result, doc)
		}
	}

	return result
}

func (r *HybridRetriever) RetrieveWithContext(ctx context.Context, query string, contextHints []string, limit int) ([]Document, error) {
	// Expand query with context hints
	expandedQuery := query
	if r.config.EnableQueryExpansion && len(contextHints) > 0 {
		expandedQuery = expandQueryWithContext(query, contextHints, r.config.MaxQueryTerms)
	}

	return r.Retrieve(ctx, expandedQuery, limit)
}

// RetrieveChunked retrieves documents and groups them by source.
func (r *HybridRetriever) RetrieveChunked(ctx context.Context, query string, limit int, maxChunksPerSource int) (map[string][]Document, error) {
	docs, err := r.Retrieve(ctx, query, limit*2) // Get more to allow grouping
	if err != nil {
		return nil, err
	}

	grouped := make(map[string][]Document)
	for _, doc := range docs {
		source := doc.Source
		if len(grouped[source]) < maxChunksPerSource {
			grouped[source] = append(grouped[source], doc)
		}
	}

	return grouped, nil
}


// extractKeywords extracts important terms from text.
func extractKeywords(text string) []string {
	words := strings.Fields(strings.ToLower(text))
	
	stopwords := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "must": true, "shall": true,
		"i": true, "you": true, "he": true, "she": true, "it": true,
		"we": true, "they": true, "what": true, "which": true, "who": true,
		"this": true, "that": true, "these": true, "those": true,
		"am": true, "of": true, "in": true, "to": true, "for": true,
		"on": true, "with": true, "at": true, "by": true, "from": true,
		"as": true, "into": true, "through": true, "during": true,
		"and": true, "or": true, "but": true, "if": true, "then": true,
		"so": true, "than": true, "too": true, "very": true, "just": true,
	}

	keywords := make([]string, 0)
	seen := make(map[string]bool)
	
	for _, word := range words {
		word = strings.Trim(word, ".,!?;:\"'()[]{}")
		if len(word) < 2 {
			continue
		}
		if stopwords[word] {
			continue
		}
		if seen[word] {
			continue
		}
		seen[word] = true
		keywords = append(keywords, word)
	}

	return keywords
}

// computeKeywordScore computes a BM25-like keyword matching score.
func computeKeywordScore(queryTerms []string, docText string) float64 {
	if len(queryTerms) == 0 {
		return 0
	}

	docLower := strings.ToLower(docText)
	docWords := strings.Fields(docLower)
	docLen := float64(len(docWords))
	avgDocLen := 100.0 // Assumed average
	
	k1 := 1.5
	b := 0.75
	
	score := 0.0
	for _, term := range queryTerms {
		// Count term frequency
		tf := 0.0
		for _, word := range docWords {
			if strings.Contains(word, term) {
				tf++
			}
		}
		
		if tf > 0 {
			// BM25 formula (simplified, assuming IDF = 1)
			numerator := tf * (k1 + 1)
			denominator := tf + k1*(1-b+b*(docLen/avgDocLen))
			score += numerator / denominator
		}
	}

	// Normalize by query length
	return score / float64(len(queryTerms))
}

// computeRecencyScore computes a recency score based on document ID.
func computeRecencyScore(docID string) float64 {
	// Extract chunk number from ID (format: "path#N")
	parts := strings.Split(docID, "#")
	if len(parts) < 2 {
		return 0.5
	}
	// Higher chunk numbers are assumed to be more recent (or just return base score)
	return 0.5
}

// textSimilarity computes a simple text similarity using Jaccard index.
func textSimilarity(text1, text2 string) float64 {
	words1 := make(map[string]bool)
	for _, w := range strings.Fields(strings.ToLower(text1)) {
		words1[w] = true
	}
	
	words2 := make(map[string]bool)
	for _, w := range strings.Fields(strings.ToLower(text2)) {
		words2[w] = true
	}

	intersection := 0
	for w := range words1 {
		if words2[w] {
			intersection++
		}
	}

	union := len(words1) + len(words2) - intersection
	if union == 0 {
		return 0
	}

	return float64(intersection) / float64(union)
}

// expandQueryWithContext expands the query with relevant terms from context.
func expandQueryWithContext(query string, contextHints []string, maxTerms int) string {
	queryTerms := extractKeywords(query)
	
	// Extract additional terms from context
	contextTerms := make(map[string]int)
	for _, hint := range contextHints {
		for _, term := range extractKeywords(hint) {
			contextTerms[term]++
		}
	}

	// Sort context terms by frequency
	type termFreq struct {
		term string
		freq int
	}
	sorted := make([]termFreq, 0, len(contextTerms))
	for t, f := range contextTerms {
		sorted = append(sorted, termFreq{term: t, freq: f})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].freq > sorted[j].freq
	})

	// Add top context terms not already in query
	existing := make(map[string]bool)
	for _, t := range queryTerms {
		existing[t] = true
	}

	expansion := make([]string, 0)
	for _, tf := range sorted {
		if len(expansion) >= maxTerms-len(queryTerms) {
			break
		}
		if !existing[tf.term] {
			expansion = append(expansion, tf.term)
		}
	}

	if len(expansion) > 0 {
		return query + " " + strings.Join(expansion, " ")
	}
	return query
}

// CrossEncoderReranker provides neural reranking (placeholder for future implementation).
type CrossEncoderReranker struct {
	// Would hold a cross-encoder model for reranking
	enabled bool
}

// Rerank reranks documents using a cross-encoder (placeholder).
func (r *CrossEncoderReranker) Rerank(ctx context.Context, query string, docs []Document) ([]Document, error) {
	// Placeholder - would use a cross-encoder model
	// For now, just return as-is
	return docs, nil
}

// ChunkMerger merges adjacent chunks from the same source.
type ChunkMerger struct {
	maxMergedTokens int
}

// NewChunkMerger creates a new chunk merger.
func NewChunkMerger(maxTokens int) *ChunkMerger {
	if maxTokens <= 0 {
		maxTokens = 1000
	}
	return &ChunkMerger{maxMergedTokens: maxTokens}
}

// MergeAdjacent merges adjacent chunks from the same source file.
func (m *ChunkMerger) MergeAdjacent(docs []Document) []Document {
	if len(docs) <= 1 {
		return docs
	}

	// Group by source
	bySource := make(map[string][]Document)
	for _, doc := range docs {
		bySource[doc.Source] = append(bySource[doc.Source], doc)
	}

	merged := make([]Document, 0, len(docs))
	for source, chunks := range bySource {
		// Sort chunks by ID (which contains chunk index)
		sort.Slice(chunks, func(i, j int) bool {
			return chunks[i].ID < chunks[j].ID
		})

		// Merge adjacent chunks
		current := chunks[0]
		currentTokens := estimateTokenCount(current.Text)

		for i := 1; i < len(chunks); i++ {
			nextTokens := estimateTokenCount(chunks[i].Text)
			
			// Check if chunks are adjacent (consecutive IDs) and fit in limit
			if isAdjacent(current.ID, chunks[i].ID) && currentTokens+nextTokens <= m.maxMergedTokens {
				// Merge
				current.Text = current.Text + "\n\n" + chunks[i].Text
				current.Score = math.Max(current.Score, chunks[i].Score)
				currentTokens += nextTokens
			} else {
				merged = append(merged, current)
				current = chunks[i]
				currentTokens = nextTokens
			}
		}
		merged = append(merged, current)
		_ = source // Avoid unused variable warning
	}

	// Re-sort by score
	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	return merged
}

// isAdjacent checks if two chunk IDs are adjacent.
func isAdjacent(id1, id2 string) bool {
	// Format: "path#N"
	parts1 := strings.Split(id1, "#")
	parts2 := strings.Split(id2, "#")
	
	if len(parts1) != 2 || len(parts2) != 2 {
		return false
	}
	
	if parts1[0] != parts2[0] { // Different files
		return false
	}

	// Parse chunk indices
	var idx1, idx2 int
	if _, err := fmt.Sscanf(parts1[1], "%d", &idx1); err != nil {
		return false
	}
	if _, err := fmt.Sscanf(parts2[1], "%d", &idx2); err != nil {
		return false
	}

	return idx2 == idx1+1
}

// estimateTokenCount estimates token count for text.
func estimateTokenCount(text string) int {
	return len(text) / 4 // Rough estimate
}

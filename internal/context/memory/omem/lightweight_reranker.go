package omem

import (
	"context"
	"sort"
	"strings"
	"unicode"
)

type RerankerConfig struct {
	EnableReranking     bool
	MaxCandidates       int
	MinScoreImprovement float64
	UseCrossEncoder     bool
}

func DefaultRerankerConfig() RerankerConfig {
	return RerankerConfig{
		EnableReranking:     false,
		MaxCandidates:       50,
		MinScoreImprovement: 0.01,
		UseCrossEncoder:     false,
	}
}

type RerankedFact struct {
	ScoredFact
	OriginalIndex int
	RerankScore   float64
}

type LightweightReranker struct {
	config RerankerConfig
}

func NewLightweightReranker(cfg RerankerConfig) *LightweightReranker {
	if cfg.MaxCandidates <= 0 {
		cfg.MaxCandidates = 50
	}

	return &LightweightReranker{
		config: cfg,
	}
}

func (lr *LightweightReranker) Rerank(ctx context.Context, query string, facts []ScoredFact) []ScoredFact {
	if len(facts) <= 1 || !lr.config.EnableReranking {
		return facts
	}

	if len(facts) > lr.config.MaxCandidates {
		facts = facts[:lr.config.MaxCandidates]
	}

	queryTerms := lr.extractQueryTerms(query)

	reranked := make([]RerankedFact, 0, len(facts))

	for i, sf := range facts {
		rerankScore := lr.calculateRerankScore(query, queryTerms, sf)
		reranked = append(reranked, RerankedFact{
			ScoredFact:    sf,
			OriginalIndex: i,
			RerankScore:   rerankScore,
		})
	}

	sort.Slice(reranked, func(i, j int) bool {
		return reranked[i].RerankScore > reranked[j].RerankScore
	})

	result := make([]ScoredFact, 0, len(reranked))
	for _, rf := range reranked {
		rf.Score = rf.RerankScore
		result = append(result, rf.ScoredFact)
	}

	return result
}

func (lr *LightweightReranker) extractQueryTerms(query string) []string {
	query = strings.ToLower(query)
	words := strings.Fields(query)

	terms := make([]string, 0, len(words))
	for _, word := range words {
		word = strings.TrimFunc(word, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsNumber(r)
		})

		if len(word) < 2 {
			continue
		}

		if isLexicalStopWord(word) {
			continue
		}

		terms = append(terms, word)
	}

	return terms
}

func (lr *LightweightReranker) calculateRerankScore(query string, queryTerms []string, sf ScoredFact) float64 {
	termScore := lr.calculateTermMatchScore(queryTerms, sf.Fact.AtomicText)

	positionScore := lr.calculatePositionScore(sf.Fact.AtomicText, query)

	lengthScore := lr.calculateLengthScore(sf.Fact.AtomicText)

	exactMatchScore := lr.calculateExactMatchScore(query, sf.Fact.AtomicText)

	fuzzyMatchScore := lr.calculateFuzzyMatchScore(queryTerms, sf.Fact.AtomicText)

	combinedScore := 0.4*termScore +
		0.2*positionScore +
		0.15*lengthScore +
		0.15*exactMatchScore +
		0.1*fuzzyMatchScore

	return combinedScore
}

func (lr *LightweightReranker) calculateTermMatchScore(queryTerms []string, text string) float64 {
	if len(queryTerms) == 0 {
		return 0.5
	}

	text = strings.ToLower(text)
	matches := 0

	for _, term := range queryTerms {
		if strings.Contains(text, term) {
			matches++
		}
	}

	return float64(matches) / float64(len(queryTerms))
}

func (lr *LightweightReranker) calculatePositionScore(text string, query string) float64 {
	text = strings.ToLower(text)
	query = strings.ToLower(query)

	pos := strings.Index(text, query)
	if pos == -1 {
		for _, term := range strings.Fields(query) {
			p := strings.Index(text, term)
			if p != -1 {
				pos = p
				break
			}
		}
	}

	if pos == -1 {
		return 0.5
	}

	length := float64(len(text))
	positionRatio := float64(pos) / length

	return 1.0 - positionRatio
}

func (lr *LightweightReranker) calculateLengthScore(text string) float64 {
	wordCount := len(strings.Fields(text))

	switch {
	case wordCount < 3:
		return 0.3
	case wordCount < 10:
		return 0.7
	case wordCount < 30:
		return 1.0
	case wordCount < 50:
		return 0.8
	default:
		return 0.6
	}
}

func (lr *LightweightReranker) calculateExactMatchScore(query string, text string) float64 {
	query = strings.ToLower(strings.TrimSpace(query))
	text = strings.ToLower(text)

	if query == text {
		return 1.0
	}

	if strings.HasPrefix(text, query) || strings.HasSuffix(text, query) {
		return 0.8
	}

	if strings.Contains(text, query) {
		return 0.6
	}

	return 0.0
}

func (lr *LightweightReranker) calculateFuzzyMatchScore(queryTerms []string, text string) float64 {
	if len(queryTerms) == 0 {
		return 0.5
	}

	text = strings.ToLower(text)
	words := strings.Fields(text)

	if len(words) == 0 {
		return 0.0
	}

	matches := 0
	for _, qt := range queryTerms {
		for _, tw := range words {
			if lr.levenshteinDistance(qt, tw) <= 2 {
				matches++
				break
			}
		}
	}

	return float64(matches) / float64(len(queryTerms))
}

func (lr *LightweightReranker) levenshteinDistance(s1, s2 string) int {
	if len(s1) == 0 {
		return len(s2)
	}
	if len(s2) == 0 {
		return len(s1)
	}

	matrix := make([][]int, len(s1)+1)
	for i := range matrix {
		matrix[i] = make([]int, len(s2)+1)
	}

	for i := 0; i <= len(s1); i++ {
		matrix[i][0] = i
	}
	for j := 0; j <= len(s2); j++ {
		matrix[0][j] = j
	}

	for i := 1; i <= len(s1); i++ {
		for j := 1; j <= len(s2); j++ {
			cost := 1
			if s1[i-1] == s2[j-1] {
				cost = 0
			}
			matrix[i][j] = min(
				matrix[i-1][j]+1,
				min(matrix[i][j-1]+1, matrix[i-1][j-1]+cost),
			)
		}
	}

	return matrix[len(s1)][len(s2)]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (lr *LightweightReranker) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"enable_reranking":      lr.config.EnableReranking,
		"max_candidates":        lr.config.MaxCandidates,
		"min_score_improvement": lr.config.MinScoreImprovement,
		"use_cross_encoder":     lr.config.UseCrossEncoder,
	}
}

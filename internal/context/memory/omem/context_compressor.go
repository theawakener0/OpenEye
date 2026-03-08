package omem

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

type ContextCompressorConfig struct {
	MaxTokens               int
	ImportanceWeight        float64
	RecencyWeight           float64
	PositionWeight          float64
	EnableImportancePruning bool
	MinImportanceThreshold  float64
}

func DefaultContextCompressorConfig() ContextCompressorConfig {
	return ContextCompressorConfig{
		MaxTokens:               1000,
		ImportanceWeight:        0.5,
		RecencyWeight:           0.3,
		PositionWeight:          0.2,
		EnableImportancePruning: true,
		MinImportanceThreshold:  0.1,
	}
}

type WeightedFact struct {
	Fact       Fact
	Importance float64
	Recency    float64
	Position   float64
	TotalScore float64
	TokenCount int
}

type ContextCompressor struct {
	config ContextCompressorConfig
	mu     sync.RWMutex
}

func NewContextCompressor(cfg ContextCompressorConfig) *ContextCompressor {
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 1000
	}
	if cfg.ImportanceWeight <= 0 {
		cfg.ImportanceWeight = 0.5
	}
	if cfg.RecencyWeight <= 0 {
		cfg.RecencyWeight = 0.3
	}
	if cfg.PositionWeight <= 0 {
		cfg.PositionWeight = 0.2
	}

	return &ContextCompressor{
		config: cfg,
	}
}

func (cc *ContextCompressor) Compress(ctx context.Context, facts []ScoredFact, maxTokens int) ([]ScoredFact, int) {
	if len(facts) == 0 {
		return facts, 0
	}

	if maxTokens <= 0 {
		maxTokens = cc.config.MaxTokens
	}

	totalTokens := cc.estimateTotalTokens(facts)
	if totalTokens <= maxTokens {
		return facts, totalTokens
	}

	weightedFacts := cc.calculateImportanceScores(facts, maxTokens)

	cc.sortByWeightedScore(weightedFacts)

	result := cc.selectTopByTokens(weightedFacts, maxTokens)

	outputTokens := cc.estimateTotalTokens(result)

	return result, outputTokens
}

func (cc *ContextCompressor) calculateImportanceScores(facts []ScoredFact, maxTokens int) []WeightedFact {
	now := time.Now()
	weightedFacts := make([]WeightedFact, 0, len(facts))

	for i, sf := range facts {
		importance := sf.Fact.Importance
		if cc.config.EnableImportancePruning && importance < cc.config.MinImportanceThreshold {
			continue
		}

		hoursSince := now.Sub(sf.Fact.CreatedAt).Hours()
		recency := math.Exp(-math.Ln2 * hoursSince / (24 * 7))

		positionScore := 1.0 - (float64(i) / float64(len(facts)))

		totalScore := cc.config.ImportanceWeight*importance +
			cc.config.RecencyWeight*recency +
			cc.config.PositionWeight*positionScore

		tokenCount := cc.estimateTokens(sf.Fact.AtomicText)

		weightedFacts = append(weightedFacts, WeightedFact{
			Fact:       sf.Fact,
			Importance: importance,
			Recency:    recency,
			Position:   positionScore,
			TotalScore: totalScore,
			TokenCount: tokenCount,
		})
	}

	return weightedFacts
}

func (cc *ContextCompressor) sortByWeightedScore(facts []WeightedFact) {
	sort.Slice(facts, func(i, j int) bool {
		return facts[i].TotalScore > facts[j].TotalScore
	})
}

func (cc *ContextCompressor) selectTopByTokens(facts []WeightedFact, maxTokens int) []ScoredFact {
	result := make([]ScoredFact, 0, len(facts))
	totalTokens := 0

	for _, wf := range facts {
		if totalTokens+wf.TokenCount > maxTokens {
			continue
		}
		totalTokens += wf.TokenCount
		result = append(result, ScoredFact{
			Fact:  wf.Fact,
			Score: wf.TotalScore,
		})
	}

	return result
}

func (cc *ContextCompressor) CompressWithOrdering(ctx context.Context, facts []ScoredFact, maxTokens int) ([]ScoredFact, string) {
	if len(facts) == 0 {
		return facts, "empty"
	}

	if maxTokens <= 0 {
		maxTokens = cc.config.MaxTokens
	}

	totalTokens := cc.estimateTotalTokens(facts)
	if totalTokens <= maxTokens {
		return facts, "no_compression"
	}

	weightedFacts := cc.calculateImportanceScores(facts, maxTokens)

	cc.sortByWeightedScore(weightedFacts)

	headSize := len(weightedFacts) / 3
	if headSize < 2 {
		headSize = 2
	}

	head := weightedFacts[:headSize]
	tail := weightedFacts[headSize:]
	cc.shuffleMiddle(tail)

	ordered := append(head, tail...)

	result := make([]ScoredFact, 0, len(ordered))
	totalTokens = 0
	for _, wf := range ordered {
		if totalTokens+wf.TokenCount > maxTokens {
			break
		}
		totalTokens += wf.TokenCount
		result = append(result, ScoredFact{
			Fact:  wf.Fact,
			Score: wf.TotalScore,
		})
	}

	orderDesc := "importance_head_tail"
	return result, orderDesc
}

func (cc *ContextCompressor) shuffleMiddle(facts []WeightedFact) {
	if len(facts) <= 1 {
		return
	}

	shuffled := make([]WeightedFact, len(facts))
	copy(shuffled, facts)

	for i := len(shuffled) - 1; i > 0; i-- {
		j := int(uint64(i) * 247 % uint64(len(facts)))
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	}

	copy(facts, shuffled)
}

func (cc *ContextCompressor) estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return (len(text) + 3) / 4
}

func (cc *ContextCompressor) estimateTotalTokens(facts []ScoredFact) int {
	total := 0
	for _, sf := range facts {
		total += cc.estimateTokens(sf.Fact.AtomicText)
	}
	return total
}

func (cc *ContextCompressor) FormatForContext(facts []ScoredFact) string {
	if len(facts) == 0 {
		return ""
	}

	var sb strings.Builder

	for i, sf := range facts {
		sb.WriteString("- ")
		sb.WriteString(sf.Fact.AtomicText)
		if i < len(facts)-1 {
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

func (cc *ContextCompressor) GetStats() map[string]interface{} {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	return map[string]interface{}{
		"max_tokens":               cc.config.MaxTokens,
		"importance_weight":        cc.config.ImportanceWeight,
		"recency_weight":           cc.config.RecencyWeight,
		"position_weight":          cc.config.PositionWeight,
		"importance_pruning":       cc.config.EnableImportancePruning,
		"min_importance_threshold": cc.config.MinImportanceThreshold,
	}
}

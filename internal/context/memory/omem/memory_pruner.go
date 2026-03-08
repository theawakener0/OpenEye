package omem

import (
	"context"
	"math"
	"sort"
	"sync"
	"time"
)

type MemoryPrunerConfig struct {
	PruneThreshold      int
	PruneKeepRecent     int
	MinImportanceToKeep float64
	PruneInterval       time.Duration
	EnableAutoPrune     bool
	ImportanceWeight    float64
	RecencyWeight       float64
	AccessWeight        float64
}

func DefaultMemoryPrunerConfig() MemoryPrunerConfig {
	return MemoryPrunerConfig{
		PruneThreshold:      15000,
		PruneKeepRecent:     8000,
		MinImportanceToKeep: 0.2,
		PruneInterval:       5 * time.Minute,
		EnableAutoPrune:     true,
		ImportanceWeight:    0.5,
		RecencyWeight:       0.3,
		AccessWeight:        0.2,
	}
}

type PruneCandidate struct {
	FactID      int64
	Importance  float64
	Recency     float64
	AccessCount int
	TotalScore  float64
	CreatedAt   time.Time
}

type MemoryPruner struct {
	config    MemoryPrunerConfig
	store     *FactStore
	mu        sync.RWMutex
	lastPrune time.Time
	stats     PrunerStats
}

type PrunerStats struct {
	PrunesPerformed   int64
	FactsPruned       int64
	LastPruneTime     time.Time
	LastPruneDuration time.Duration
}

func NewMemoryPruner(cfg MemoryPrunerConfig, store *FactStore) *MemoryPruner {
	if cfg.PruneThreshold <= 0 {
		cfg.PruneThreshold = 15000
	}
	if cfg.PruneKeepRecent <= 0 {
		cfg.PruneKeepRecent = 8000
	}
	if cfg.MinImportanceToKeep <= 0 {
		cfg.MinImportanceToKeep = 0.2
	}

	return &MemoryPruner{
		config:    cfg,
		store:     store,
		lastPrune: time.Now(),
		stats:     PrunerStats{},
	}
}

func (mp *MemoryPruner) ShouldPrune(ctx context.Context) (bool, int, error) {
	if mp.store == nil {
		return false, 0, nil
	}

	stats, err := mp.store.GetStats(ctx)
	if err != nil {
		return false, 0, err
	}

	totalFacts := 0
	if v, ok := stats["total_facts"].(int); ok {
		totalFacts = v
	}

	shouldPrune := totalFacts >= mp.config.PruneThreshold
	excess := totalFacts - mp.config.PruneKeepRecent

	return shouldPrune, excess, nil
}

func (mp *MemoryPruner) Prune(ctx context.Context) (int, error) {
	mp.mu.Lock()
	defer mp.mu.Unlock()

	if mp.store == nil {
		return 0, nil
	}

	startTime := time.Now()

	shouldPrune, excess, err := mp.ShouldPrune(ctx)
	if err != nil {
		return 0, err
	}

	if !shouldPrune && excess <= 0 {
		return 0, nil
	}

	if excess <= 0 {
		excess = mp.config.PruneKeepRecent / 4
	}

	candidates, err := mp.getPruneCandidates(ctx, excess*2)
	if err != nil {
		return 0, err
	}

	if len(candidates) == 0 {
		return 0, nil
	}

	mp.scoreCandidates(candidates)

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].TotalScore < candidates[j].TotalScore
	})

	pruneCount := excess
	if pruneCount > len(candidates) {
		pruneCount = len(candidates)
	}

	factsToPrune := candidates[:pruneCount]
	pruned, err := mp.executePrune(ctx, factsToPrune)
	if err != nil {
		return 0, err
	}

	mp.stats.PrunesPerformed++
	mp.stats.FactsPruned += int64(pruned)
	mp.stats.LastPruneTime = time.Now()
	mp.stats.LastPruneDuration = time.Since(startTime)
	mp.lastPrune = time.Now()

	return pruned, nil
}

func (mp *MemoryPruner) getPruneCandidates(ctx context.Context, limit int) ([]PruneCandidate, error) {
	if mp.store == nil {
		return []PruneCandidate{}, nil
	}

	facts, err := mp.store.GetRecentFacts(ctx, limit*2)
	if err != nil {
		return nil, err
	}

	candidates := make([]PruneCandidate, 0, len(facts))
	for _, fact := range facts {
		candidates = append(candidates, PruneCandidate{
			FactID:      fact.ID,
			Importance:  fact.Importance,
			CreatedAt:   fact.CreatedAt,
			AccessCount: fact.AccessCount,
		})
	}

	return candidates, nil
}

func (mp *MemoryPruner) scoreCandidates(candidates []PruneCandidate) {
	now := time.Now()

	for i := range candidates {
		c := &candidates[i]

		hoursSince := now.Sub(c.CreatedAt).Hours()
		recency := math.Exp(-math.Ln2 * hoursSince / (24 * 7))

		accessScore := 0.0
		if c.AccessCount > 0 {
			accessScore = math.Log1p(float64(c.AccessCount)) / math.Log1p(100)
			if accessScore > 1.0 {
				accessScore = 1.0
			}
		}

		c.Recency = recency
		c.TotalScore = mp.config.ImportanceWeight*c.Importance +
			mp.config.RecencyWeight*recency +
			mp.config.AccessWeight*accessScore
	}
}

func (mp *MemoryPruner) executePrune(ctx context.Context, candidates []PruneCandidate) (int, error) {
	if mp.store == nil {
		return 0, nil
	}

	pruned := 0
	for _, candidate := range candidates {
		if candidate.Importance < mp.config.MinImportanceToKeep {
			err := mp.store.MarkObsolete(ctx, candidate.FactID, nil)
			if err != nil {
				continue
			}
			pruned++
		}
	}

	return pruned, nil
}

func (mp *MemoryPruner) GetStats() map[string]interface{} {
	mp.mu.RLock()
	defer mp.mu.RUnlock()

	return map[string]interface{}{
		"prunes_performed":    mp.stats.PrunesPerformed,
		"facts_pruned":        mp.stats.FactsPruned,
		"last_prune_time":     mp.stats.LastPruneTime,
		"last_prune_duration": mp.stats.LastPruneDuration,
		"prune_threshold":     mp.config.PruneThreshold,
		"prune_keep_recent":   mp.config.PruneKeepRecent,
		"min_importance":      mp.config.MinImportanceToKeep,
		"auto_prune_enabled":  mp.config.EnableAutoPrune,
	}
}

func (mp *MemoryPruner) ForcePruneIfNeeded(ctx context.Context) (bool, error) {
	shouldPrune, _, err := mp.ShouldPrune(ctx)
	if err != nil {
		return false, err
	}

	if !shouldPrune {
		return false, nil
	}

	_, err = mp.Prune(ctx)
	if err != nil {
		return true, err
	}

	return true, nil
}

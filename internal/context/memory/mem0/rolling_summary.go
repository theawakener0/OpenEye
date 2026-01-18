package mem0

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
)

// RollingSummaryManager maintains an up-to-date summary of the user's facts.
// It runs asynchronously to minimize impact on response latency.
type RollingSummaryManager struct {
	config   SummaryConfig
	store    *FactStore
	manager  *runtime.Manager
	embedder embedding.Provider
	prompts  *PromptTemplates

	mu            sync.RWMutex
	cachedSummary *RollingSummary
	lastFactCount int
	dirty         bool

	// Async refresh
	refreshChan chan struct{}
	stopChan    chan struct{}
	wg          sync.WaitGroup
}

// NewRollingSummaryManager creates a new rolling summary manager.
func NewRollingSummaryManager(cfg SummaryConfig, store *FactStore, manager *runtime.Manager, embedder embedding.Provider) (*RollingSummaryManager, error) {
	if store == nil {
		return nil, errors.New("fact store required")
	}
	if manager == nil {
		return nil, errors.New("runtime manager required")
	}

	cfg = applySummaryDefaults(cfg)

	rsm := &RollingSummaryManager{
		config:   cfg,
		store:    store,
		manager:  manager,
		embedder: embedder,
		prompts:  NewPromptTemplates(),
	}

	if cfg.Async {
		rsm.refreshChan = make(chan struct{}, 1)
		rsm.stopChan = make(chan struct{})
		rsm.wg.Add(1)
		go rsm.asyncRefreshWorker()
	}

	return rsm, nil
}

// applySummaryDefaults fills in missing configuration values.
func applySummaryDefaults(cfg SummaryConfig) SummaryConfig {
	if cfg.RefreshInterval <= 0 {
		cfg.RefreshInterval = 5 * time.Minute
	}
	if cfg.MaxFacts <= 0 {
		cfg.MaxFacts = 50
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 512
	}
	return cfg
}

// GetSummary returns the current rolling summary.
func (rsm *RollingSummaryManager) GetSummary(ctx context.Context) (string, error) {
	if rsm == nil || rsm.store == nil {
		return "", errors.New("rolling summary manager not initialized")
	}

	if !rsm.config.Enabled {
		return "", nil
	}

	rsm.mu.RLock()
	cached := rsm.cachedSummary
	rsm.mu.RUnlock()

	// Return cached summary if available and not too stale
	if cached != nil && cached.Summary != "" {
		age := time.Since(cached.UpdatedAt)
		if age < rsm.config.RefreshInterval*2 {
			return cached.Summary, nil
		}
	}

	// Try to load from database
	summary, err := rsm.store.GetRollingSummary(ctx)
	if err != nil {
		return "", err
	}

	if summary != nil && summary.Summary != "" {
		rsm.mu.Lock()
		rsm.cachedSummary = summary
		rsm.mu.Unlock()
		return summary.Summary, nil
	}

	// No summary exists, trigger refresh
	if rsm.config.Async {
		rsm.TriggerRefresh()
	} else {
		if _, err := rsm.Refresh(ctx); err != nil {
			return "", err
		}
	}

	return "", nil
}

// GetSummaryWithEmbedding returns the summary along with its embedding.
func (rsm *RollingSummaryManager) GetSummaryWithEmbedding(ctx context.Context) (*RollingSummary, error) {
	if rsm == nil || rsm.store == nil {
		return nil, errors.New("rolling summary manager not initialized")
	}

	if !rsm.config.Enabled {
		return nil, nil
	}

	rsm.mu.RLock()
	cached := rsm.cachedSummary
	rsm.mu.RUnlock()

	if cached != nil && cached.Summary != "" {
		return cached, nil
	}

	summary, err := rsm.store.GetRollingSummary(ctx)
	if err != nil {
		return nil, err
	}

	if summary != nil {
		rsm.mu.Lock()
		rsm.cachedSummary = summary
		rsm.mu.Unlock()
	}

	return summary, nil
}

// TriggerRefresh signals that the summary should be refreshed.
func (rsm *RollingSummaryManager) TriggerRefresh() {
	if rsm == nil || !rsm.config.Async || rsm.refreshChan == nil {
		return
	}

	select {
	case rsm.refreshChan <- struct{}{}:
	default:
		// Channel full, refresh already pending
	}
}

// MarkDirty marks the summary as needing refresh.
func (rsm *RollingSummaryManager) MarkDirty() {
	if rsm == nil {
		return
	}

	rsm.mu.Lock()
	rsm.dirty = true
	rsm.mu.Unlock()

	if rsm.config.Async {
		rsm.TriggerRefresh()
	}
}

// Refresh regenerates the summary synchronously.
func (rsm *RollingSummaryManager) Refresh(ctx context.Context) (*RollingSummary, error) {
	if rsm == nil || rsm.store == nil || rsm.manager == nil {
		return nil, errors.New("rolling summary manager not initialized")
	}

	if !rsm.config.Enabled {
		return nil, nil
	}

	// Get top facts for summary
	facts, err := rsm.store.GetRecentFacts(ctx, rsm.config.MaxFacts, false)
	if err != nil {
		return nil, fmt.Errorf("failed to get facts for summary: %w", err)
	}

	if len(facts) == 0 {
		return &RollingSummary{
			ID:        1,
			Summary:   "",
			UpdatedAt: time.Now(),
		}, nil
	}

	// Check if summary needs updating
	rsm.mu.RLock()
	currentFactCount := rsm.lastFactCount
	isDirty := rsm.dirty
	rsm.mu.RUnlock()

	if !isDirty && len(facts) == currentFactCount && rsm.cachedSummary != nil {
		return rsm.cachedSummary, nil
	}

	// Generate new summary
	summary, err := rsm.generateSummary(ctx, facts)
	if err != nil {
		return nil, err
	}

	// Update cache
	rsm.mu.Lock()
	rsm.cachedSummary = summary
	rsm.lastFactCount = len(facts)
	rsm.dirty = false
	rsm.mu.Unlock()

	// Persist to database
	if err := rsm.store.UpdateRollingSummary(ctx, *summary); err != nil {
		log.Printf("warning: failed to persist rolling summary: %v", err)
	}

	return summary, nil
}

// generateSummary uses the LLM to create a summary from facts.
func (rsm *RollingSummaryManager) generateSummary(ctx context.Context, facts []Fact) (*RollingSummary, error) {
	// Extract fact texts
	factTexts := make([]string, len(facts))
	factIDs := make([]string, len(facts))
	for i, f := range facts {
		factTexts[i] = f.Text
		factIDs[i] = strconv.FormatInt(f.ID, 10)
	}

	// Check if we should do incremental update
	rsm.mu.RLock()
	existingSummary := rsm.cachedSummary
	rsm.mu.RUnlock()

	var summaryText string
	var err error

	if existingSummary != nil && existingSummary.Summary != "" && existingSummary.FactCount > 0 {
		// Find new facts since last summary
		newFactTexts := rsm.findNewFacts(factTexts, existingSummary.SourceFacts)
		if len(newFactTexts) > 0 && len(newFactTexts) < len(factTexts)/2 {
			// Incremental update if less than half are new
			summaryText, err = rsm.incrementalSummary(ctx, existingSummary.Summary, newFactTexts)
		} else {
			// Full regeneration
			summaryText, err = rsm.fullSummary(ctx, factTexts)
		}
	} else {
		// First time generation
		summaryText, err = rsm.fullSummary(ctx, factTexts)
	}

	if err != nil {
		return nil, err
	}

	summary := &RollingSummary{
		ID:          1,
		Summary:     summaryText,
		UpdatedAt:   time.Now(),
		SourceFacts: strings.Join(factIDs, ","),
		FactCount:   len(facts),
	}

	// Generate embedding for the summary
	if rsm.embedder != nil && summaryText != "" {
		emb, err := rsm.embedder.Embed(ctx, summaryText)
		if err == nil {
			summary.Embedding = emb
		}
	}

	return summary, nil
}

// fullSummary generates a complete summary from all facts.
func (rsm *RollingSummaryManager) fullSummary(ctx context.Context, factTexts []string) (string, error) {
	prompt := rsm.prompts.RollingSummaryPrompt(factTexts, rsm.config.MaxTokens)

	resp, err := rsm.manager.Generate(ctx, runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:     rsm.config.MaxTokens,
			Temperature:   0.5,
			TopK:          40,
			TopP:          0.9,
			MinP:          0.05,
			RepeatPenalty: 1.1,
			RepeatLastN:   64,
		},
	})
	if err != nil {
		return "", fmt.Errorf("failed to generate summary: %w", err)
	}

	return strings.TrimSpace(resp.Text), nil
}

// incrementalSummary updates an existing summary with new facts.
func (rsm *RollingSummaryManager) incrementalSummary(ctx context.Context, existingSummary string, newFactTexts []string) (string, error) {
	prompt := rsm.prompts.IncrementalSummaryPrompt(existingSummary, newFactTexts)

	resp, err := rsm.manager.Generate(ctx, runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:     rsm.config.MaxTokens,
			Temperature:   0.5,
			TopK:          40,
			TopP:          0.9,
			MinP:          0.05,
			RepeatPenalty: 1.1,
			RepeatLastN:   64,
		},
	})
	if err != nil {
		return "", fmt.Errorf("failed to update summary: %w", err)
	}

	return strings.TrimSpace(resp.Text), nil
}

// findNewFacts identifies facts not in the previous summary.
func (rsm *RollingSummaryManager) findNewFacts(currentFacts []string, previousFactIDs string) []string {
	if previousFactIDs == "" {
		return currentFacts
	}

	// Parse previous fact count (rough heuristic)
	previousCount := len(strings.Split(previousFactIDs, ","))

	if len(currentFacts) <= previousCount {
		return nil
	}

	// Return facts beyond the previous count (assuming ordered by recency)
	return currentFacts[previousCount:]
}

// asyncRefreshWorker runs periodic summary refreshes.
func (rsm *RollingSummaryManager) asyncRefreshWorker() {
	defer rsm.wg.Done()

	ticker := time.NewTicker(rsm.config.RefreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rsm.stopChan:
			return
		case <-ticker.C:
			rsm.doRefresh()
		case <-rsm.refreshChan:
			rsm.doRefresh()
		}
	}
}

// doRefresh performs the actual refresh with error handling.
func (rsm *RollingSummaryManager) doRefresh() {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	if _, err := rsm.Refresh(ctx); err != nil {
		log.Printf("warning: async summary refresh failed: %v", err)
	}
}

// GetFormattedSummary returns a formatted summary for inclusion in prompts.
func (rsm *RollingSummaryManager) GetFormattedSummary(ctx context.Context) string {
	summary, err := rsm.GetSummary(ctx)
	if err != nil || summary == "" {
		return ""
	}

	return fmt.Sprintf("User Profile:\n%s", summary)
}

// NeedsSummary checks if the user has enough facts to warrant a summary.
func (rsm *RollingSummaryManager) NeedsSummary(ctx context.Context) bool {
	stats, err := rsm.store.GetStats(ctx)
	if err != nil {
		return false
	}

	activeFacts, ok := stats["active_facts"].(int)
	if !ok {
		return false
	}

	return activeFacts >= 3 // Need at least 3 facts for a meaningful summary
}

// Close stops the async worker and cleans up resources.
func (rsm *RollingSummaryManager) Close() error {
	if rsm == nil {
		return nil
	}

	if rsm.config.Async && rsm.stopChan != nil {
		close(rsm.stopChan)
		rsm.wg.Wait()
	}

	return nil
}

// ForceRefresh forces an immediate summary regeneration.
func (rsm *RollingSummaryManager) ForceRefresh(ctx context.Context) (*RollingSummary, error) {
	rsm.mu.Lock()
	rsm.dirty = true
	rsm.cachedSummary = nil
	rsm.mu.Unlock()

	return rsm.Refresh(ctx)
}

// GetStats returns statistics about the rolling summary.
func (rsm *RollingSummaryManager) GetStats(ctx context.Context) map[string]interface{} {
	stats := make(map[string]interface{})

	rsm.mu.RLock()
	if rsm.cachedSummary != nil {
		stats["summary_length"] = len(rsm.cachedSummary.Summary)
		stats["fact_count"] = rsm.cachedSummary.FactCount
		stats["last_updated"] = rsm.cachedSummary.UpdatedAt
		stats["has_embedding"] = len(rsm.cachedSummary.Embedding) > 0
	}
	stats["is_dirty"] = rsm.dirty
	rsm.mu.RUnlock()

	return stats
}

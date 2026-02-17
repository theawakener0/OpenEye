package omem

import (
	"context"
	"errors"
	"log"
	"strings"
	"sync"
	"time"
)

// RollingSummaryManager provides incremental summary updates for the memory system.
// Key features:
// - Delta-based updates: only summarizes new facts since last update
// - Async refresh: background worker for non-blocking updates
// - Dirty tracking: marks summary as stale when new facts are added
// - Token-aware: respects maximum summary length
type RollingSummaryManager struct {
	config  SummaryConfig
	store   *FactStore
	prompts *PromptTemplates

	// LLM for summary generation
	llmGenerate func(ctx context.Context, prompt string) (string, error)

	// State
	mu             sync.RWMutex
	currentSummary *RollingSummary
	pendingFacts   []int64 // Fact IDs added since last summary update
	isDirty        bool
	isRefreshing   bool // Prevents concurrent refresh operations
	lastRefresh    time.Time

	// Background worker
	stopCh    chan struct{}
	stoppedCh chan struct{}
}

// NewRollingSummaryManager creates a new rolling summary manager.
func NewRollingSummaryManager(
	cfg SummaryConfig,
	store *FactStore,
	llmGenerate func(ctx context.Context, prompt string) (string, error),
) *RollingSummaryManager {
	cfg = applySummaryDefaults(cfg)

	rsm := &RollingSummaryManager{
		config:       cfg,
		store:        store,
		prompts:      NewPromptTemplates(),
		llmGenerate:  llmGenerate,
		pendingFacts: make([]int64, 0),
		stopCh:       make(chan struct{}),
		stoppedCh:    make(chan struct{}),
	}

	// Load existing summary from store
	if store != nil {
		summary, err := store.GetRollingSummary(context.Background())
		if err == nil && summary != nil {
			rsm.currentSummary = summary
			rsm.pendingFacts = summary.PendingFactIDs
		}
	}

	// Start background worker if async enabled
	if cfg.Async && cfg.Enabled {
		go rsm.backgroundWorker()
	}

	return rsm
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
	if cfg.MinNewFactsForUpdate <= 0 {
		cfg.MinNewFactsForUpdate = 5
	}
	return cfg
}

// GetSummary returns the current rolling summary.
func (rsm *RollingSummaryManager) GetSummary(ctx context.Context) (*RollingSummary, error) {
	if rsm == nil {
		return nil, errors.New("rolling summary manager not initialized")
	}

	rsm.mu.RLock()
	defer rsm.mu.RUnlock()

	if rsm.currentSummary == nil {
		return &RollingSummary{Summary: ""}, nil
	}

	return rsm.currentSummary, nil
}

// GetSummaryText returns just the summary text (convenience method).
func (rsm *RollingSummaryManager) GetSummaryText(ctx context.Context) string {
	summary, err := rsm.GetSummary(ctx)
	if err != nil || summary == nil {
		return ""
	}
	return summary.Summary
}

// MarkDirty marks the summary as needing an update.
func (rsm *RollingSummaryManager) MarkDirty(factIDs ...int64) {
	if rsm == nil {
		return
	}

	rsm.mu.Lock()
	defer rsm.mu.Unlock()

	rsm.isDirty = true
	rsm.pendingFacts = append(rsm.pendingFacts, factIDs...)
}

// IsDirty returns whether the summary needs an update.
func (rsm *RollingSummaryManager) IsDirty() bool {
	if rsm == nil {
		return false
	}

	rsm.mu.RLock()
	defer rsm.mu.RUnlock()

	return rsm.isDirty
}

// PendingFactCount returns the number of facts pending summarization.
func (rsm *RollingSummaryManager) PendingFactCount() int {
	if rsm == nil {
		return 0
	}

	rsm.mu.RLock()
	defer rsm.mu.RUnlock()

	return len(rsm.pendingFacts)
}

// Refresh updates the summary if dirty and enough new facts are pending.
func (rsm *RollingSummaryManager) Refresh(ctx context.Context) error {
	if rsm == nil || !rsm.config.Enabled {
		return nil
	}

	rsm.mu.Lock()
	// Check if update is needed
	if !rsm.isDirty || len(rsm.pendingFacts) < rsm.config.MinNewFactsForUpdate {
		rsm.mu.Unlock()
		return nil
	}

	// Get pending facts
	pendingIDs := make([]int64, len(rsm.pendingFacts))
	copy(pendingIDs, rsm.pendingFacts)
	currentSummary := ""
	if rsm.currentSummary != nil {
		currentSummary = rsm.currentSummary.Summary
	}
	rsm.mu.Unlock()

	// Perform the update
	var newSummary string
	var err error

	if rsm.config.IncrementalUpdate && currentSummary != "" {
		newSummary, err = rsm.incrementalUpdate(ctx, currentSummary, pendingIDs)
	} else {
		newSummary, err = rsm.fullUpdate(ctx)
	}

	if err != nil {
		return err
	}

	// Update state
	rsm.mu.Lock()
	defer rsm.mu.Unlock()

	// Collect all source fact IDs
	var allFactIDs []int64
	if rsm.currentSummary != nil {
		allFactIDs = rsm.currentSummary.SourceFactIDs
	}
	allFactIDs = append(allFactIDs, pendingIDs...)

	rsm.currentSummary = &RollingSummary{
		ID:             1,
		Summary:        newSummary,
		UpdatedAt:      time.Now(),
		SourceFactIDs:  allFactIDs,
		FactCount:      len(allFactIDs),
		PendingFactIDs: nil, // Cleared
	}
	rsm.pendingFacts = nil
	rsm.isDirty = false
	rsm.lastRefresh = time.Now()

	// Persist to store
	if rsm.store != nil {
		_ = rsm.store.UpdateRollingSummary(ctx, *rsm.currentSummary)
	}

	return nil
}

// ForceRefresh forces a full summary regeneration.
func (rsm *RollingSummaryManager) ForceRefresh(ctx context.Context) error {
	if rsm == nil || !rsm.config.Enabled {
		return nil
	}

	newSummary, err := rsm.fullUpdate(ctx)
	if err != nil {
		return err
	}

	rsm.mu.Lock()
	defer rsm.mu.Unlock()

	// Get all fact IDs from store
	var allFactIDs []int64
	if rsm.store != nil {
		facts, err := rsm.store.GetRecentFacts(ctx, rsm.config.MaxFacts)
		if err == nil {
			for _, f := range facts {
				allFactIDs = append(allFactIDs, f.ID)
			}
		}
	}

	rsm.currentSummary = &RollingSummary{
		ID:             1,
		Summary:        newSummary,
		UpdatedAt:      time.Now(),
		SourceFactIDs:  allFactIDs,
		FactCount:      len(allFactIDs),
		PendingFactIDs: nil,
	}
	rsm.pendingFacts = nil
	rsm.isDirty = false
	rsm.lastRefresh = time.Now()

	// Persist to store
	if rsm.store != nil {
		_ = rsm.store.UpdateRollingSummary(ctx, *rsm.currentSummary)
	}

	return nil
}

// incrementalUpdate updates the summary by incorporating new facts.
func (rsm *RollingSummaryManager) incrementalUpdate(ctx context.Context, currentSummary string, newFactIDs []int64) (string, error) {
	if rsm.llmGenerate == nil {
		return currentSummary, nil
	}

	// Get new facts
	if rsm.store == nil || len(newFactIDs) == 0 {
		return currentSummary, nil
	}

	newFacts, err := rsm.store.GetFactsByIDs(ctx, newFactIDs)
	if err != nil {
		return currentSummary, err
	}

	// Build new facts text
	var newFactsText strings.Builder
	for _, f := range newFacts {
		newFactsText.WriteString("- ")
		newFactsText.WriteString(f.AtomicText)
		newFactsText.WriteString("\n")
	}

	// Generate prompt for incremental update
	prompt := rsm.prompts.IncrementalSummaryUpdatePrompt(currentSummary, newFactsText.String(), rsm.config.MaxTokens)

	response, err := rsm.llmGenerate(ctx, prompt)
	if err != nil {
		return currentSummary, err
	}

	return strings.TrimSpace(response), nil
}

// fullUpdate regenerates the summary from scratch using recent facts.
func (rsm *RollingSummaryManager) fullUpdate(ctx context.Context) (string, error) {
	if rsm.llmGenerate == nil || rsm.store == nil {
		return "", nil
	}

	// Get recent facts
	facts, err := rsm.store.GetRecentFacts(ctx, rsm.config.MaxFacts)
	if err != nil {
		return "", err
	}

	if len(facts) == 0 {
		return "", nil
	}

	// Build facts text
	var factsText strings.Builder
	for _, f := range facts {
		factsText.WriteString("- ")
		factsText.WriteString(f.AtomicText)
		factsText.WriteString("\n")
	}

	// Generate prompt for full summary
	prompt := rsm.prompts.FullSummaryPrompt(factsText.String(), rsm.config.MaxTokens)

	response, err := rsm.llmGenerate(ctx, prompt)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(response), nil
}

// backgroundWorker periodically refreshes the summary.
func (rsm *RollingSummaryManager) backgroundWorker() {
	defer close(rsm.stoppedCh)

	ticker := time.NewTicker(rsm.config.RefreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rsm.stopCh:
			return
		case <-ticker.C:
			// Skip if already processing to prevent resource contention
			rsm.mu.Lock()
			if rsm.isRefreshing {
				rsm.mu.Unlock()
				continue
			}
			rsm.isRefreshing = true
			rsm.mu.Unlock()

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			err := rsm.Refresh(ctx)
			if err != nil {
				log.Printf("omem: background summary refresh failed: %v", err)
			}
			cancel()

			rsm.mu.Lock()
			rsm.isRefreshing = false
			rsm.mu.Unlock()
		}
	}
}

// Stop stops the background worker.
func (rsm *RollingSummaryManager) Stop() {
	if rsm == nil {
		return
	}

	close(rsm.stopCh)
	<-rsm.stoppedCh
}

// ============================================================================
// Prompt Templates for Summary
// ============================================================================

// IncrementalSummaryUpdatePrompt generates a prompt for incremental summary updates.
func (p *PromptTemplates) IncrementalSummaryUpdatePrompt(currentSummary, newFacts string, maxTokens int) string {
	return strings.TrimSpace(`Update this summary by incorporating new facts.

CURRENT SUMMARY:
` + currentSummary + `

NEW FACTS TO INCORPORATE:
` + newFacts + `

RULES:
- Keep the summary concise (maximum ~` + itoa(maxTokens) + ` tokens)
- Integrate new information smoothly
- Update or correct conflicting information
- Maintain most important facts
- Use third person ("User prefers..." not "I prefer...")

UPDATED SUMMARY:`)
}

// FullSummaryPrompt generates a prompt for full summary generation.
func (p *PromptTemplates) FullSummaryPrompt(facts string, maxTokens int) string {
	return strings.TrimSpace(`Create a concise summary of the user based on these facts.

FACTS:
` + facts + `

RULES:
- Maximum ~` + itoa(maxTokens) + ` tokens
- Focus on most important information (preferences, biography, relationships)
- Use third person ("User prefers..." not "I prefer...")
- Group related information together
- Omit temporary or trivial details

SUMMARY:`)
}

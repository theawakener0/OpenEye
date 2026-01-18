package omem

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"
)

// EpisodeManager provides Zep-inspired session/episode tracking.
// Key features:
// - Automatic session detection based on inactivity timeout
// - Episode-level summarization on session close
// - Entity mention tracking per episode
// - LRU cache for recent episodes
type EpisodeManager struct {
	config  EpisodeConfig
	store   *FactStore
	prompts *PromptTemplates

	// LLM for episode summarization
	llmGenerate func(ctx context.Context, prompt string) (string, error)

	// State
	mu             sync.RWMutex
	currentEpisode *Episode
	lastActivity   time.Time
	entityMentions map[string]int // Entity name -> mention count
	turnCount      int

	// LRU cache for recent episodes
	recentEpisodes []Episode
}

// NewEpisodeManager creates a new episode manager.
func NewEpisodeManager(
	cfg EpisodeConfig,
	store *FactStore,
	llmGenerate func(ctx context.Context, prompt string) (string, error),
) *EpisodeManager {
	cfg = applyEpisodeDefaults(cfg)

	em := &EpisodeManager{
		config:         cfg,
		store:          store,
		prompts:        NewPromptTemplates(),
		llmGenerate:    llmGenerate,
		entityMentions: make(map[string]int),
		recentEpisodes: make([]Episode, 0, cfg.MaxEpisodesInCache),
	}

	// Load recent episodes from store
	if store != nil {
		episodes, err := store.GetRecentEpisodes(context.Background(), cfg.MaxEpisodesInCache)
		if err == nil {
			em.recentEpisodes = episodes
		}
	}

	return em
}

// applyEpisodeDefaults fills in missing configuration values.
func applyEpisodeDefaults(cfg EpisodeConfig) EpisodeConfig {
	if cfg.SessionTimeout <= 0 {
		cfg.SessionTimeout = 30 * time.Minute
	}
	if cfg.MaxEpisodesInCache <= 0 {
		cfg.MaxEpisodesInCache = 10
	}
	return cfg
}

// StartSession starts or resumes a session with the given ID.
func (em *EpisodeManager) StartSession(ctx context.Context, sessionID string) (*Episode, error) {
	if em == nil {
		return nil, errors.New("episode manager not initialized")
	}

	em.mu.Lock()
	defer em.mu.Unlock()

	now := time.Now()

	// Check if we should continue existing episode or start new one
	if em.currentEpisode != nil {
		if em.currentEpisode.SessionID == sessionID && !em.shouldStartNewEpisode(now) {
			// Continue existing episode
			em.lastActivity = now
			return em.currentEpisode, nil
		}

		// Close existing episode
		if err := em.closeCurrentEpisodeLocked(ctx); err != nil {
			// Log error but don't fail
			_ = err
		}
	}

	// Start new episode
	episode := &Episode{
		SessionID: sessionID,
		StartedAt: now,
	}

	if em.store != nil {
		id, err := em.store.InsertEpisode(ctx, sessionID)
		if err == nil {
			episode.ID = id
		}
	}

	em.currentEpisode = episode
	em.lastActivity = now
	em.entityMentions = make(map[string]int)
	em.turnCount = 0

	return episode, nil
}

// shouldStartNewEpisode checks if a new episode should be started.
func (em *EpisodeManager) shouldStartNewEpisode(now time.Time) bool {
	if em.currentEpisode == nil {
		return true
	}

	// Check inactivity timeout
	if now.Sub(em.lastActivity) > em.config.SessionTimeout {
		return true
	}

	return false
}

// OnTurnProcessed is called after each conversation turn is processed.
func (em *EpisodeManager) OnTurnProcessed(ctx context.Context, facts []Fact, entities []string) error {
	if em == nil || !em.config.Enabled {
		return nil
	}

	em.mu.Lock()
	defer em.mu.Unlock()

	now := time.Now()

	// Check if we need to start a new episode
	if em.currentEpisode == nil || em.shouldStartNewEpisode(now) {
		if em.currentEpisode != nil {
			_ = em.closeCurrentEpisodeLocked(ctx)
		}
		// Start new episode with generated session ID
		sessionID := generateSessionID()
		em.currentEpisode = &Episode{
			SessionID: sessionID,
			StartedAt: now,
		}
		if em.store != nil {
			id, err := em.store.InsertEpisode(ctx, sessionID)
			if err == nil {
				em.currentEpisode.ID = id
			}
		}
		em.entityMentions = make(map[string]int)
		em.turnCount = 0
	}

	em.lastActivity = now
	em.turnCount++
	em.currentEpisode.FactCount += len(facts)

	// Track entity mentions
	if em.config.TrackEntityMentions {
		for _, e := range entities {
			em.entityMentions[e]++
		}
	}

	return nil
}

// closeCurrentEpisodeLocked closes the current episode (caller must hold lock).
func (em *EpisodeManager) closeCurrentEpisodeLocked(ctx context.Context) error {
	if em.currentEpisode == nil {
		return nil
	}

	now := time.Now()
	em.currentEpisode.EndedAt = &now

	// Collect entity mentions
	if em.config.TrackEntityMentions && len(em.entityMentions) > 0 {
		mentions := make([]string, 0, len(em.entityMentions))
		for e := range em.entityMentions {
			mentions = append(mentions, e)
		}
		em.currentEpisode.EntityMentions = mentions
	}

	// Generate summary if configured
	if em.config.SummaryOnClose && em.llmGenerate != nil && em.currentEpisode.FactCount > 0 {
		summary, err := em.generateEpisodeSummary(ctx, em.currentEpisode.ID)
		if err == nil {
			em.currentEpisode.Summary = summary
		}
	}

	// Update in store
	if em.store != nil {
		_ = em.store.UpdateEpisode(ctx, *em.currentEpisode)
	}

	// Add to LRU cache
	em.addToRecentEpisodes(*em.currentEpisode)

	em.currentEpisode = nil
	return nil
}

// generateEpisodeSummary generates a summary for the episode's facts.
func (em *EpisodeManager) generateEpisodeSummary(ctx context.Context, episodeID int64) (string, error) {
	if em.llmGenerate == nil || em.store == nil {
		return "", nil
	}

	// Get facts for this episode
	facts, err := em.store.GetFactsByEpisode(ctx, episodeID)
	if err != nil || len(facts) == 0 {
		return "", err
	}

	// Build facts text
	var factsText strings.Builder
	for _, f := range facts {
		factsText.WriteString("- ")
		factsText.WriteString(f.AtomicText)
		factsText.WriteString("\n")
	}

	// Generate summary
	prompt := em.prompts.EpisodeSummaryFromFactsPrompt(factsText.String())

	response, err := em.llmGenerate(ctx, prompt)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(response), nil
}

// addToRecentEpisodes adds an episode to the LRU cache.
func (em *EpisodeManager) addToRecentEpisodes(episode Episode) {
	// Check if already in cache
	for i, e := range em.recentEpisodes {
		if e.ID == episode.ID {
			// Update existing
			em.recentEpisodes[i] = episode
			return
		}
	}

	// Add to front
	em.recentEpisodes = append([]Episode{episode}, em.recentEpisodes...)

	// Trim to max size
	if len(em.recentEpisodes) > em.config.MaxEpisodesInCache {
		em.recentEpisodes = em.recentEpisodes[:em.config.MaxEpisodesInCache]
	}
}

// EndSession explicitly ends the current session.
func (em *EpisodeManager) EndSession(ctx context.Context) error {
	if em == nil {
		return nil
	}

	em.mu.Lock()
	defer em.mu.Unlock()

	return em.closeCurrentEpisodeLocked(ctx)
}

// GetCurrentEpisode returns the current episode if any.
func (em *EpisodeManager) GetCurrentEpisode() *Episode {
	if em == nil {
		return nil
	}

	em.mu.RLock()
	defer em.mu.RUnlock()

	return em.currentEpisode
}

// GetCurrentEpisodeID returns the current episode ID.
func (em *EpisodeManager) GetCurrentEpisodeID() int64 {
	if em == nil {
		return 0
	}

	em.mu.RLock()
	defer em.mu.RUnlock()

	if em.currentEpisode == nil {
		return 0
	}
	return em.currentEpisode.ID
}

// GetRecentEpisodes returns recently closed episodes from cache.
func (em *EpisodeManager) GetRecentEpisodes() []Episode {
	if em == nil {
		return nil
	}

	em.mu.RLock()
	defer em.mu.RUnlock()

	result := make([]Episode, len(em.recentEpisodes))
	copy(result, em.recentEpisodes)
	return result
}

// GetEpisode retrieves an episode by ID.
func (em *EpisodeManager) GetEpisode(ctx context.Context, episodeID int64) (*Episode, error) {
	if em == nil {
		return nil, errors.New("episode manager not initialized")
	}

	// Check current episode
	em.mu.RLock()
	if em.currentEpisode != nil && em.currentEpisode.ID == episodeID {
		ep := *em.currentEpisode
		em.mu.RUnlock()
		return &ep, nil
	}

	// Check cache
	for _, e := range em.recentEpisodes {
		if e.ID == episodeID {
			ep := e
			em.mu.RUnlock()
			return &ep, nil
		}
	}
	em.mu.RUnlock()

	// Check store
	if em.store != nil {
		return em.store.GetEpisode(ctx, episodeID)
	}

	return nil, errors.New("episode not found")
}

// GetEntityMentions returns entity mentions for the current episode.
func (em *EpisodeManager) GetEntityMentions() map[string]int {
	if em == nil {
		return nil
	}

	em.mu.RLock()
	defer em.mu.RUnlock()

	result := make(map[string]int, len(em.entityMentions))
	for k, v := range em.entityMentions {
		result[k] = v
	}
	return result
}

// GetStats returns episode manager statistics.
func (em *EpisodeManager) GetStats() map[string]interface{} {
	if em == nil {
		return nil
	}

	em.mu.RLock()
	defer em.mu.RUnlock()

	stats := map[string]interface{}{
		"has_current_episode": em.currentEpisode != nil,
		"cached_episodes":     len(em.recentEpisodes),
		"config": map[string]interface{}{
			"session_timeout":       em.config.SessionTimeout.String(),
			"summary_on_close":      em.config.SummaryOnClose,
			"track_entity_mentions": em.config.TrackEntityMentions,
			"max_episodes_in_cache": em.config.MaxEpisodesInCache,
		},
	}

	if em.currentEpisode != nil {
		stats["current_episode"] = map[string]interface{}{
			"id":              em.currentEpisode.ID,
			"session_id":      em.currentEpisode.SessionID,
			"fact_count":      em.currentEpisode.FactCount,
			"turn_count":      em.turnCount,
			"entity_mentions": len(em.entityMentions),
		}
	}

	return stats
}

// ============================================================================
// Prompt Templates for Episode Summary
// ============================================================================

// EpisodeSummaryFromFactsPrompt generates a prompt for episode summarization from facts.
func (p *PromptTemplates) EpisodeSummaryFromFactsPrompt(facts string) string {
	return strings.TrimSpace(`Summarize this conversation session in 1-2 sentences.

FACTS FROM SESSION:
` + facts + `

Focus on:
- Main topics discussed
- Key decisions or preferences expressed
- Important new information learned

SESSION SUMMARY:`)
}

// ============================================================================
// Helper Functions
// ============================================================================

// generateSessionID generates a unique session ID.
func generateSessionID() string {
	return time.Now().Format("20060102-150405-") + randomHex(4)
}

// randomHex generates a random hex string of specified length.
func randomHex(n int) string {
	const hexChars = "0123456789abcdef"
	result := make([]byte, n)
	now := time.Now().UnixNano()
	for i := 0; i < n; i++ {
		result[i] = hexChars[(now>>(i*4))&0xf]
	}
	return string(result)
}

package omem

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"sync"
	"time"
)

// AtomicEncoder performs SimpleMem-inspired semantic lossless compression.
// It transforms conversational text into atomic, self-contained facts by:
// - Resolving coreferences (pronouns -> entity names)
// - Anchoring temporal references (relative -> absolute dates)
// - Extracting atomic facts (each understandable in isolation)
type AtomicEncoder struct {
	config  AtomicEncoderConfig
	prompts *PromptTemplates

	// LLM for complex cases (optional)
	llmGenerate func(ctx context.Context, prompt string) (string, error)

	// Entity cache for coreference resolution within a session
	mu          sync.RWMutex
	entityCache *EntityCache
}

// EntityCache tracks recently mentioned entities for coreference resolution.
type EntityCache struct {
	entries    []CacheEntry
	maxEntries int
}

// CacheEntry represents a cached entity for coreference.
type CacheEntry struct {
	Name       string
	Type       EntityType
	Gender     string // "male", "female", "neutral", "unknown"
	LastSeen   time.Time
	Salience   float64 // Higher = more likely to be referenced
	IsSingular bool
}

// NewAtomicEncoder creates a new atomic encoder.
func NewAtomicEncoder(cfg AtomicEncoderConfig, llmGenerate func(ctx context.Context, prompt string) (string, error)) *AtomicEncoder {
	cfg = applyAtomicEncoderDefaults(cfg)

	return &AtomicEncoder{
		config:      cfg,
		prompts:     NewPromptTemplates(),
		llmGenerate: llmGenerate,
		entityCache: &EntityCache{
			entries:    make([]CacheEntry, 0, 50),
			maxEntries: 50,
		},
	}
}

func applyAtomicEncoderDefaults(cfg AtomicEncoderConfig) AtomicEncoderConfig {
	if cfg.MaxFactsPerTurn <= 0 {
		cfg.MaxFactsPerTurn = 10
	}
	if cfg.MinFactImportance <= 0 {
		cfg.MinFactImportance = 0.3
	}
	if cfg.MinTextLength <= 0 {
		cfg.MinTextLength = 10
	}
	return cfg
}

// EncodedResult contains the output of atomic encoding.
type EncodedResult struct {
	AtomicText         string            // Text with coreferences and times resolved
	ExtractedFacts     []ExtractedFact   // Atomic facts extracted
	DiscoveredEntities []ExtractedEntity // Entities found during processing
}

// Encode performs full atomic encoding on input text.
func (ae *AtomicEncoder) Encode(ctx context.Context, text string, currentTime time.Time) (*EncodedResult, error) {
	if ae == nil {
		return nil, errors.New("atomic encoder not initialized")
	}

	if !ae.config.Enabled {
		return &EncodedResult{AtomicText: text}, nil
	}

	text = strings.TrimSpace(text)
	if len(text) < ae.config.MinTextLength {
		return &EncodedResult{AtomicText: text}, nil
	}

	result := &EncodedResult{AtomicText: text}

	// Step 1: Extract entities from text (updates cache)
	result.DiscoveredEntities = ae.extractEntitiesFromText(text)

	// Step 2: Resolve coreferences
	if ae.config.EnableCoreference {
		resolved := ae.resolveCoreferences(ctx, text)
		result.AtomicText = resolved
	}

	// Step 3: Anchor temporal references
	if ae.config.EnableTemporal {
		result.AtomicText = ae.anchorTemporalReferences(ctx, result.AtomicText, currentTime)
	}

	// Step 4: Extract atomic facts (if LLM available)
	if ae.llmGenerate != nil {
		facts, err := ae.extractAtomicFacts(ctx, result.AtomicText)
		if err == nil {
			result.ExtractedFacts = facts
		}
	}

	return result, nil
}

// ============================================================================
// Coreference Resolution (Rule-Based)
// ============================================================================

// Pronoun patterns for coreference resolution
var (
	// Third person singular male
	pronounHe = regexp.MustCompile(`(?i)\b(he|him|his|himself)\b`)
	// Third person singular female
	pronounShe = regexp.MustCompile(`(?i)\b(she|her|hers|herself)\b`)
	// Third person singular neutral/object
	pronounIt = regexp.MustCompile(`(?i)\b(it|its|itself)\b`)
	// Third person plural
	pronounThey = regexp.MustCompile(`(?i)\b(they|them|their|theirs|themselves)\b`)
)

// resolveCoreferences replaces pronouns with entity names using rule-based approach.
func (ae *AtomicEncoder) resolveCoreferences(ctx context.Context, text string) string {
	ae.mu.RLock()
	cache := ae.entityCache
	ae.mu.RUnlock()

	if cache == nil || len(cache.entries) == 0 {
		return text
	}

	resolved := text

	// Find most salient entities by gender/type
	maleName := ae.findMostSalientEntity(cache, "male")
	femaleName := ae.findMostSalientEntity(cache, "female")
	neutralName := ae.findMostSalientEntity(cache, "neutral")
	pluralName := ae.findMostSalientEntityPlural(cache)

	// Replace pronouns with most salient candidates
	// Note: This is a simplified approach; complex cases may need LLM fallback

	if maleName != "" {
		resolved = pronounHe.ReplaceAllStringFunc(resolved, func(match string) string {
			lower := strings.ToLower(match)
			switch lower {
			case "he":
				return maleName
			case "him":
				return maleName
			case "his":
				return maleName + "'s"
			case "himself":
				return maleName
			}
			return match
		})
	}

	if femaleName != "" {
		resolved = pronounShe.ReplaceAllStringFunc(resolved, func(match string) string {
			lower := strings.ToLower(match)
			switch lower {
			case "she":
				return femaleName
			case "her":
				return femaleName
			case "hers":
				return femaleName + "'s"
			case "herself":
				return femaleName
			}
			return match
		})
	}

	if neutralName != "" {
		resolved = pronounIt.ReplaceAllStringFunc(resolved, func(match string) string {
			lower := strings.ToLower(match)
			switch lower {
			case "it":
				return neutralName
			case "its":
				return neutralName + "'s"
			case "itself":
				return neutralName
			}
			return match
		})
	}

	if pluralName != "" {
		resolved = pronounThey.ReplaceAllStringFunc(resolved, func(match string) string {
			lower := strings.ToLower(match)
			switch lower {
			case "they", "them":
				return pluralName
			case "their", "theirs":
				return pluralName + "'s"
			case "themselves":
				return pluralName
			}
			return match
		})
	}

	// If we made changes but result looks wrong, use LLM fallback
	if ae.config.UseLLMForComplex && ae.llmGenerate != nil && resolved != text {
		if ae.detectBadResolution(resolved) {
			llmResolved, err := ae.llmCoreference(ctx, text)
			if err == nil && llmResolved != "" {
				return llmResolved
			}
		}
	}

	return resolved
}

// findMostSalientEntity finds the most salient entity matching gender.
func (ae *AtomicEncoder) findMostSalientEntity(cache *EntityCache, gender string) string {
	var best *CacheEntry
	for i := range cache.entries {
		entry := &cache.entries[i]
		if entry.Gender == gender || (gender == "neutral" && entry.Type == EntityThing) {
			if best == nil || entry.Salience > best.Salience {
				best = entry
			}
		}
	}
	if best != nil {
		return best.Name
	}
	return ""
}

// findMostSalientEntityPlural finds most salient plural/group entity.
func (ae *AtomicEncoder) findMostSalientEntityPlural(cache *EntityCache) string {
	var best *CacheEntry
	for i := range cache.entries {
		entry := &cache.entries[i]
		if !entry.IsSingular || entry.Type == EntityOrganization {
			if best == nil || entry.Salience > best.Salience {
				best = entry
			}
		}
	}
	if best != nil {
		return best.Name
	}
	return ""
}

// detectBadResolution checks if resolution looks wrong (heuristic).
func (ae *AtomicEncoder) detectBadResolution(text string) bool {
	// Check for repeated names too close together
	words := strings.Fields(text)
	if len(words) < 3 {
		return false
	}

	// Simple heuristic: same word appearing 3+ times in 10 words = suspicious
	counts := make(map[string]int)
	for _, w := range words {
		w = strings.ToLower(w)
		counts[w]++
		if counts[w] >= 3 {
			return true
		}
	}
	return false
}

// llmCoreference uses LLM for complex coreference cases.
func (ae *AtomicEncoder) llmCoreference(ctx context.Context, text string) (string, error) {
	if ae.llmGenerate == nil {
		return text, nil
	}

	knownEntities := ae.getKnownEntityNames()
	prompt := ae.prompts.CoreferenceResolutionPrompt(text, knownEntities)

	response, err := ae.llmGenerate(ctx, prompt)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(response), nil
}

// getKnownEntityNames returns names of cached entities.
func (ae *AtomicEncoder) getKnownEntityNames() []string {
	ae.mu.RLock()
	defer ae.mu.RUnlock()

	if ae.entityCache == nil {
		return nil
	}

	names := make([]string, 0, len(ae.entityCache.entries))
	for _, e := range ae.entityCache.entries {
		names = append(names, e.Name)
	}
	return names
}

// ============================================================================
// Temporal Anchoring (Rule-Based)
// ============================================================================

// Temporal patterns for anchoring
var (
	patternToday        = regexp.MustCompile(`(?i)\b(today)\b`)
	patternYesterday    = regexp.MustCompile(`(?i)\b(yesterday)\b`)
	patternTomorrow     = regexp.MustCompile(`(?i)\b(tomorrow)\b`)
	patternLastWeek     = regexp.MustCompile(`(?i)\b(last week)\b`)
	patternThisWeek     = regexp.MustCompile(`(?i)\b(this week)\b`)
	patternNextWeek     = regexp.MustCompile(`(?i)\b(next week)\b`)
	patternLastMonth    = regexp.MustCompile(`(?i)\b(last month)\b`)
	patternThisMonth    = regexp.MustCompile(`(?i)\b(this month)\b`)
	patternNextMonth    = regexp.MustCompile(`(?i)\b(next month)\b`)
	patternLastYear     = regexp.MustCompile(`(?i)\b(last year)\b`)
	patternThisYear     = regexp.MustCompile(`(?i)\b(this year)\b`)
	patternNextYear     = regexp.MustCompile(`(?i)\b(next year)\b`)
	patternNDaysAgo     = regexp.MustCompile(`(?i)\b(\d+) days? ago\b`)
	patternNWeeksAgo    = regexp.MustCompile(`(?i)\b(\d+) weeks? ago\b`)
	patternNMonthsAgo   = regexp.MustCompile(`(?i)\b(\d+) months? ago\b`)
	patternInNDays      = regexp.MustCompile(`(?i)\bin (\d+) days?\b`)
	patternInNWeeks     = regexp.MustCompile(`(?i)\bin (\d+) weeks?\b`)
	patternThisMorning  = regexp.MustCompile(`(?i)\b(this morning)\b`)
	patternTonight      = regexp.MustCompile(`(?i)\b(tonight)\b`)
	patternLastNight    = regexp.MustCompile(`(?i)\b(last night)\b`)
	patternRecentlyJust = regexp.MustCompile(`(?i)\b(recently|just now|earlier)\b`)
)

// anchorTemporalReferences converts relative time references to absolute dates.
func (ae *AtomicEncoder) anchorTemporalReferences(ctx context.Context, text string, now time.Time) string {
	result := text

	// Simple replacements
	dateFormat := "2006-01-02"

	result = patternToday.ReplaceAllString(result, now.Format(dateFormat))
	result = patternYesterday.ReplaceAllString(result, now.AddDate(0, 0, -1).Format(dateFormat))
	result = patternTomorrow.ReplaceAllString(result, now.AddDate(0, 0, 1).Format(dateFormat))

	result = patternLastWeek.ReplaceAllString(result, "the week of "+now.AddDate(0, 0, -7).Format(dateFormat))
	result = patternThisWeek.ReplaceAllString(result, "the week of "+now.Format(dateFormat))
	result = patternNextWeek.ReplaceAllString(result, "the week of "+now.AddDate(0, 0, 7).Format(dateFormat))

	result = patternLastMonth.ReplaceAllString(result, now.AddDate(0, -1, 0).Format("January 2006"))
	result = patternThisMonth.ReplaceAllString(result, now.Format("January 2006"))
	result = patternNextMonth.ReplaceAllString(result, now.AddDate(0, 1, 0).Format("January 2006"))

	result = patternLastYear.ReplaceAllString(result, now.AddDate(-1, 0, 0).Format("2006"))
	result = patternThisYear.ReplaceAllString(result, now.Format("2006"))
	result = patternNextYear.ReplaceAllString(result, now.AddDate(1, 0, 0).Format("2006"))

	// "N days/weeks/months ago" patterns
	result = patternNDaysAgo.ReplaceAllStringFunc(result, func(match string) string {
		matches := patternNDaysAgo.FindStringSubmatch(match)
		if len(matches) >= 2 {
			n := parseInt(matches[1], 1)
			return now.AddDate(0, 0, -n).Format(dateFormat)
		}
		return match
	})

	result = patternNWeeksAgo.ReplaceAllStringFunc(result, func(match string) string {
		matches := patternNWeeksAgo.FindStringSubmatch(match)
		if len(matches) >= 2 {
			n := parseInt(matches[1], 1)
			return now.AddDate(0, 0, -n*7).Format(dateFormat)
		}
		return match
	})

	result = patternNMonthsAgo.ReplaceAllStringFunc(result, func(match string) string {
		matches := patternNMonthsAgo.FindStringSubmatch(match)
		if len(matches) >= 2 {
			n := parseInt(matches[1], 1)
			return now.AddDate(0, -n, 0).Format("January 2006")
		}
		return match
	})

	// "in N days/weeks" patterns
	result = patternInNDays.ReplaceAllStringFunc(result, func(match string) string {
		matches := patternInNDays.FindStringSubmatch(match)
		if len(matches) >= 2 {
			n := parseInt(matches[1], 1)
			return "by " + now.AddDate(0, 0, n).Format(dateFormat)
		}
		return match
	})

	result = patternInNWeeks.ReplaceAllStringFunc(result, func(match string) string {
		matches := patternInNWeeks.FindStringSubmatch(match)
		if len(matches) >= 2 {
			n := parseInt(matches[1], 1)
			return "by " + now.AddDate(0, 0, n*7).Format(dateFormat)
		}
		return match
	})

	// Time of day patterns
	result = patternThisMorning.ReplaceAllString(result, "on "+now.Format(dateFormat)+" morning")
	result = patternTonight.ReplaceAllString(result, "on "+now.Format(dateFormat)+" evening")
	result = patternLastNight.ReplaceAllString(result, "on "+now.AddDate(0, 0, -1).Format(dateFormat)+" evening")
	result = patternRecentlyJust.ReplaceAllString(result, "around "+now.Format(dateFormat))

	return result
}

// ============================================================================
// Atomic Fact Extraction
// ============================================================================

// extractAtomicFacts uses LLM to extract self-contained facts from processed text.
func (ae *AtomicEncoder) extractAtomicFacts(ctx context.Context, text string) ([]ExtractedFact, error) {
	if ae.llmGenerate == nil {
		return nil, errors.New("LLM not available for fact extraction")
	}

	prompt := ae.prompts.AtomicFactExtractionPrompt(text, ae.config.MaxFactsPerTurn)

	response, err := ae.llmGenerate(ctx, prompt)
	if err != nil {
		return nil, err
	}

	facts := ParseFactExtractionResponse(response)

	// Filter by minimum importance
	filtered := make([]ExtractedFact, 0, len(facts))
	for _, f := range facts {
		if f.Importance >= ae.config.MinFactImportance {
			filtered = append(filtered, f)
		}
	}

	return filtered, nil
}

// ============================================================================
// Entity Extraction (Rule-Based)
// ============================================================================

// Entity extraction patterns
var (
	// Capitalized words (potential proper nouns) - simple heuristic
	patternCapitalizedWord = regexp.MustCompile(`\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b`)
	// Common name patterns
	patternPersonName = regexp.MustCompile(`(?i)\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b`)
	// Email patterns (indicate person)
	patternEmail = regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`)
	// Company patterns
	patternCompany = regexp.MustCompile(`(?i)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Inc\.?|Corp\.?|LLC|Ltd\.?|Company|Co\.)\b`)
	// Location indicators
	patternLocation = regexp.MustCompile(`(?i)\b(in|at|from|to)\s+([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)\b`)
	// Common person indicators
	patternMyRelative = regexp.MustCompile(`(?i)\bmy\s+(wife|husband|brother|sister|mother|father|mom|dad|son|daughter|friend|colleague|boss|manager)\s+([A-Z][a-z]+)\b`)
)

// extractEntitiesFromText extracts entities using regex and heuristics.
func (ae *AtomicEncoder) extractEntitiesFromText(text string) []ExtractedEntity {
	entities := make(map[string]ExtractedEntity) // Use map to dedupe

	// Extract person names with titles
	personMatches := patternPersonName.FindAllStringSubmatch(text, -1)
	for _, match := range personMatches {
		if len(match) >= 3 {
			name := match[2]
			entities[strings.ToLower(name)] = ExtractedEntity{
				Name:       name,
				EntityType: EntityPerson,
			}
		}
	}

	// Extract relatives/relations with names
	relativeMatches := patternMyRelative.FindAllStringSubmatch(text, -1)
	for _, match := range relativeMatches {
		if len(match) >= 3 {
			name := match[2]
			entities[strings.ToLower(name)] = ExtractedEntity{
				Name:       name,
				EntityType: EntityPerson,
			}
		}
	}

	// Extract company names
	companyMatches := patternCompany.FindAllStringSubmatch(text, -1)
	for _, match := range companyMatches {
		if len(match) >= 2 {
			name := strings.TrimSpace(match[0])
			entities[strings.ToLower(name)] = ExtractedEntity{
				Name:       name,
				EntityType: EntityOrganization,
			}
		}
	}

	// Extract locations
	locationMatches := patternLocation.FindAllStringSubmatch(text, -1)
	for _, match := range locationMatches {
		if len(match) >= 3 {
			name := match[2]
			// Filter out common false positives
			if !isCommonWord(name) {
				entities[strings.ToLower(name)] = ExtractedEntity{
					Name:       name,
					EntityType: EntityPlace,
				}
			}
		}
	}

	// Update entity cache
	ae.updateEntityCache(entities)

	// Convert map to slice
	result := make([]ExtractedEntity, 0, len(entities))
	for _, e := range entities {
		result = append(result, e)
	}

	return result
}

// isCommonWord checks if a word is a common word (false positive filter).
func isCommonWord(word string) bool {
	commonWords := map[string]bool{
		"the": true, "a": true, "an": true, "is": true, "was": true,
		"are": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "must": true, "shall": true,
		"i": true, "you": true, "we": true, "they": true, "it": true,
		"this": true, "that": true, "these": true, "those": true,
		"monday": true, "tuesday": true, "wednesday": true, "thursday": true,
		"friday": true, "saturday": true, "sunday": true,
		"january": true, "february": true, "march": true, "april": true,
		"june": true, "july": true, "august": true,
		"september": true, "october": true, "november": true, "december": true,
	}
	return commonWords[strings.ToLower(word)]
}

// updateEntityCache updates the cache with newly discovered entities.
func (ae *AtomicEncoder) updateEntityCache(entities map[string]ExtractedEntity) {
	ae.mu.Lock()
	defer ae.mu.Unlock()

	now := time.Now()

	for _, entity := range entities {
		// Check if already in cache
		found := false
		for i := range ae.entityCache.entries {
			if strings.EqualFold(ae.entityCache.entries[i].Name, entity.Name) {
				// Update salience and last seen
				ae.entityCache.entries[i].LastSeen = now
				ae.entityCache.entries[i].Salience += 0.1
				found = true
				break
			}
		}

		if !found && len(ae.entityCache.entries) < ae.entityCache.maxEntries {
			// Add new entry
			gender := ae.inferGender(entity.Name, entity.EntityType)
			ae.entityCache.entries = append(ae.entityCache.entries, CacheEntry{
				Name:       entity.Name,
				Type:       entity.EntityType,
				Gender:     gender,
				LastSeen:   now,
				Salience:   1.0,
				IsSingular: true,
			})
		}
	}

	// Decay salience of old entries
	for i := range ae.entityCache.entries {
		age := now.Sub(ae.entityCache.entries[i].LastSeen)
		if age > time.Minute {
			ae.entityCache.entries[i].Salience *= 0.95
		}
	}
}

// inferGender attempts to infer gender from entity name/type (simple heuristic).
func (ae *AtomicEncoder) inferGender(name string, entType EntityType) string {
	if entType != EntityPerson {
		return "neutral"
	}

	// Common English first name heuristics
	lower := strings.ToLower(strings.Fields(name)[0])

	maleNames := map[string]bool{
		"john": true, "james": true, "michael": true, "david": true, "robert": true,
		"william": true, "richard": true, "joseph": true, "thomas": true, "charles": true,
		"daniel": true, "matthew": true, "anthony": true, "mark": true, "paul": true,
		"steven": true, "andrew": true, "joshua": true, "kevin": true, "brian": true,
	}

	femaleNames := map[string]bool{
		"mary": true, "patricia": true, "jennifer": true, "linda": true, "elizabeth": true,
		"barbara": true, "susan": true, "jessica": true, "sarah": true, "karen": true,
		"nancy": true, "lisa": true, "betty": true, "margaret": true, "sandra": true,
		"ashley": true, "kimberly": true, "emily": true, "donna": true, "michelle": true,
	}

	if maleNames[lower] {
		return "male"
	}
	if femaleNames[lower] {
		return "female"
	}

	return "unknown"
}

// ============================================================================
// Session Management
// ============================================================================

// ClearCache clears the entity cache (call at session start).
func (ae *AtomicEncoder) ClearCache() {
	ae.mu.Lock()
	defer ae.mu.Unlock()

	ae.entityCache.entries = ae.entityCache.entries[:0]
}

// AddEntityToCache manually adds an entity to the cache.
func (ae *AtomicEncoder) AddEntityToCache(name string, entType EntityType, gender string) {
	ae.mu.Lock()
	defer ae.mu.Unlock()

	// Check if exists
	for i := range ae.entityCache.entries {
		if strings.EqualFold(ae.entityCache.entries[i].Name, name) {
			ae.entityCache.entries[i].Salience += 0.5
			ae.entityCache.entries[i].LastSeen = time.Now()
			return
		}
	}

	if len(ae.entityCache.entries) < ae.entityCache.maxEntries {
		ae.entityCache.entries = append(ae.entityCache.entries, CacheEntry{
			Name:       name,
			Type:       entType,
			Gender:     gender,
			LastSeen:   time.Now(),
			Salience:   2.0, // Higher initial salience for manually added
			IsSingular: true,
		})
	}
}

// SetUserName adds the user's name to the cache with high salience.
func (ae *AtomicEncoder) SetUserName(name string) {
	ae.AddEntityToCache(name, EntityPerson, "unknown")
	// Boost salience
	ae.mu.Lock()
	for i := range ae.entityCache.entries {
		if strings.EqualFold(ae.entityCache.entries[i].Name, name) {
			ae.entityCache.entries[i].Salience = 10.0
			break
		}
	}
	ae.mu.Unlock()
}

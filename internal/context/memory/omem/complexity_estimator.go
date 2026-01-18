package omem

import (
	"regexp"
	"strings"
	"unicode"
)

// ComplexityEstimator provides rule-based query complexity estimation.
// This is a zero-LLM approach that estimates complexity using pure rules,
// enabling adaptive retrieval depth without LLM overhead at query time.
//
// The complexity score (0.0-1.0) is used to calculate dynamic retrieval depth:
//
//	k_dyn = k_base * (1 + delta * complexity)
//
// Complexity factors:
// - Query length (longer = more complex)
// - Entity count (more entities = more complex)
// - Temporal references (time-based queries need broader search)
// - Question type (why/how questions are harder than what/who)
// - Negation presence (negative constraints add complexity)
// - Comparison presence (comparisons need multiple facts)
// - Conditional/hypothetical language
type ComplexityEstimator struct {
	config RetrievalConfig
}

// ComplexityResult contains the breakdown of complexity factors.
type ComplexityResult struct {
	TotalScore       float64 // Combined complexity score (0.0-1.0)
	LengthScore      float64 // Based on query length
	EntityScore      float64 // Based on entity count
	TemporalScore    float64 // Based on temporal references
	QuestionScore    float64 // Based on question type
	NegationScore    float64 // Based on negation presence
	ComparisonScore  float64 // Based on comparison presence
	ConditionalScore float64 // Based on conditional language

	// Derived values
	DynamicK int      // Calculated retrieval depth
	Entities []string // Extracted entities for graph lookup
}

// NewComplexityEstimator creates a new complexity estimator.
func NewComplexityEstimator(cfg RetrievalConfig) *ComplexityEstimator {
	return &ComplexityEstimator{config: cfg}
}

// Question type patterns
var (
	// Hard questions (why, how, explain) need more context
	patternHardQuestion = regexp.MustCompile(`(?i)^\s*(why|how|explain|describe|what.*reason|what.*cause)`)
	// Medium questions (when, where, which) need specific context
	patternMediumQuestion = regexp.MustCompile(`(?i)^\s*(when|where|which|whose)`)
	// Easy questions (what, who, is/are/does) need less context
	patternEasyQuestion = regexp.MustCompile(`(?i)^\s*(what|who|is|are|does|do|did|can|will|has|have)`)

	// Temporal indicators
	patternTemporalRef = regexp.MustCompile(`(?i)\b(today|yesterday|tomorrow|last\s+(?:week|month|year)|this\s+(?:week|month|year)|next\s+(?:week|month|year)|ago|\d{4}|january|february|march|april|may|june|july|august|september|october|november|december)\b`)

	// Negation indicators (for complexity estimation)
	patternNegationQuery = regexp.MustCompile(`(?i)\b(not|never|no|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't|haven't|hasn't|isn't|aren't|wasn't|weren't)\b`)

	// Comparison indicators
	patternComparison = regexp.MustCompile(`(?i)\b(compare|difference|similar|between|versus|vs\.?|more|less|better|worse|same|different|like|unlike)\b`)

	// Conditional/hypothetical indicators
	patternConditional = regexp.MustCompile(`(?i)\b(if|would|could|might|should|suppose|assuming|hypothetically|imagine)\b`)

	// Proper noun detection (for complexity estimation)
	patternProperNounQuery = regexp.MustCompile(`\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b`)

	// Quoted strings (specific references)
	patternQuoted = regexp.MustCompile(`"[^"]+"|'[^']+'`)
)

// EstimateComplexity calculates the complexity of a query using rule-based heuristics.
func (ce *ComplexityEstimator) EstimateComplexity(query string) ComplexityResult {
	result := ComplexityResult{}

	query = strings.TrimSpace(query)
	if query == "" {
		result.DynamicK = ce.config.DefaultTopK
		return result
	}

	// 1. Length-based complexity (normalized to typical query length)
	wordCount := len(strings.Fields(query))
	result.LengthScore = ce.calculateLengthScore(wordCount)

	// 2. Entity-based complexity
	result.Entities = ce.extractQueryEntities(query)
	result.EntityScore = ce.calculateEntityScore(len(result.Entities))

	// 3. Temporal reference complexity
	temporalMatches := patternTemporalRef.FindAllString(query, -1)
	result.TemporalScore = ce.calculateTemporalScore(len(temporalMatches))

	// 4. Question type complexity
	result.QuestionScore = ce.calculateQuestionScore(query)

	// 5. Negation complexity
	negationMatches := patternNegationQuery.FindAllString(query, -1)
	result.NegationScore = ce.calculateNegationScore(len(negationMatches))

	// 6. Comparison complexity
	comparisonMatches := patternComparison.FindAllString(query, -1)
	result.ComparisonScore = ce.calculateComparisonScore(len(comparisonMatches))

	// 7. Conditional complexity
	conditionalMatches := patternConditional.FindAllString(query, -1)
	result.ConditionalScore = ce.calculateConditionalScore(len(conditionalMatches))

	// Calculate total complexity (weighted average)
	result.TotalScore = ce.combineScores(result)

	// Calculate dynamic K
	result.DynamicK = ce.calculateDynamicK(result.TotalScore)

	return result
}

// calculateLengthScore converts word count to complexity score.
func (ce *ComplexityEstimator) calculateLengthScore(wordCount int) float64 {
	// Short queries (< 5 words): low complexity
	// Medium queries (5-15 words): moderate complexity
	// Long queries (> 15 words): high complexity
	switch {
	case wordCount < 5:
		return 0.1
	case wordCount < 10:
		return 0.3
	case wordCount < 15:
		return 0.5
	case wordCount < 25:
		return 0.7
	default:
		return 0.9
	}
}

// calculateEntityScore converts entity count to complexity score.
func (ce *ComplexityEstimator) calculateEntityScore(entityCount int) float64 {
	// More entities = need to consider multiple contexts
	switch entityCount {
	case 0:
		return 0.1
	case 1:
		return 0.2
	case 2:
		return 0.4
	case 3:
		return 0.6
	default:
		return 0.8
	}
}

// calculateTemporalScore converts temporal reference count to complexity score.
func (ce *ComplexityEstimator) calculateTemporalScore(count int) float64 {
	// Temporal queries often need broader time-based filtering
	switch count {
	case 0:
		return 0.0
	case 1:
		return 0.3
	case 2:
		return 0.5
	default:
		return 0.7
	}
}

// calculateQuestionScore determines complexity based on question type.
func (ce *ComplexityEstimator) calculateQuestionScore(query string) float64 {
	query = strings.TrimSpace(query)

	// Hard questions (why, how, explain)
	if patternHardQuestion.MatchString(query) {
		return 0.8
	}

	// Medium questions (when, where, which)
	if patternMediumQuestion.MatchString(query) {
		return 0.5
	}

	// Easy questions (what, who, is/are)
	if patternEasyQuestion.MatchString(query) {
		return 0.3
	}

	// Statement or command (not a question)
	if !strings.HasSuffix(query, "?") {
		return 0.2
	}

	// Unknown question type
	return 0.4
}

// calculateNegationScore converts negation count to complexity score.
func (ce *ComplexityEstimator) calculateNegationScore(count int) float64 {
	// Negations require understanding of what's NOT true
	switch count {
	case 0:
		return 0.0
	case 1:
		return 0.2
	case 2:
		return 0.4
	default:
		return 0.6
	}
}

// calculateComparisonScore converts comparison indicator count to complexity score.
func (ce *ComplexityEstimator) calculateComparisonScore(count int) float64 {
	// Comparisons require multiple related facts
	switch count {
	case 0:
		return 0.0
	case 1:
		return 0.4
	case 2:
		return 0.6
	default:
		return 0.8
	}
}

// calculateConditionalScore converts conditional indicator count to complexity score.
func (ce *ComplexityEstimator) calculateConditionalScore(count int) float64 {
	// Conditionals/hypotheticals require reasoning over multiple scenarios
	switch count {
	case 0:
		return 0.0
	case 1:
		return 0.3
	default:
		return 0.5
	}
}

// combineScores calculates the weighted total complexity score.
func (ce *ComplexityEstimator) combineScores(result ComplexityResult) float64 {
	// Weights for each factor (sum to 1.0)
	weights := struct {
		Length      float64
		Entity      float64
		Temporal    float64
		Question    float64
		Negation    float64
		Comparison  float64
		Conditional float64
	}{
		Length:      0.15,
		Entity:      0.25,
		Temporal:    0.15,
		Question:    0.20,
		Negation:    0.05,
		Comparison:  0.10,
		Conditional: 0.10,
	}

	total := weights.Length*result.LengthScore +
		weights.Entity*result.EntityScore +
		weights.Temporal*result.TemporalScore +
		weights.Question*result.QuestionScore +
		weights.Negation*result.NegationScore +
		weights.Comparison*result.ComparisonScore +
		weights.Conditional*result.ConditionalScore

	// Clamp to [0.0, 1.0]
	if total < 0.0 {
		total = 0.0
	}
	if total > 1.0 {
		total = 1.0
	}

	return total
}

// calculateDynamicK calculates the adaptive retrieval depth.
// Formula: k_dyn = k_base * (1 + delta * complexity)
func (ce *ComplexityEstimator) calculateDynamicK(complexity float64) int {
	if !ce.config.EnableComplexityEstimation {
		return ce.config.DefaultTopK
	}

	baseK := ce.config.DefaultTopK
	if baseK <= 0 {
		baseK = 5
	}

	delta := ce.config.ComplexityDelta
	if delta <= 0 {
		delta = 2.0
	}

	// k_dyn = k_base * (1 + delta * complexity)
	dynamicK := float64(baseK) * (1 + delta*complexity)
	k := int(dynamicK + 0.5) // Round to nearest integer

	// Clamp to MaxTopK
	maxK := ce.config.MaxTopK
	if maxK <= 0 {
		maxK = 20
	}
	if k > maxK {
		k = maxK
	}

	return k
}

// extractQueryEntities extracts potential entity names from a query.
func (ce *ComplexityEstimator) extractQueryEntities(query string) []string {
	entities := make(map[string]bool)

	// Extract proper nouns (capitalized words)
	matches := patternProperNounQuery.FindAllString(query, -1)
	for _, match := range matches {
		// Filter out common false positives
		if !isQueryStopWord(match) {
			entities[match] = true
		}
	}

	// Extract quoted strings (specific references)
	quotedMatches := patternQuoted.FindAllString(query, -1)
	for _, match := range quotedMatches {
		// Remove quotes
		cleaned := strings.Trim(match, `"'`)
		if len(cleaned) > 0 {
			entities[cleaned] = true
		}
	}

	result := make([]string, 0, len(entities))
	for e := range entities {
		result = append(result, e)
	}

	return result
}

// isQueryStopWord checks if a word is a common query stop word.
func isQueryStopWord(word string) bool {
	stopWords := map[string]bool{
		// Common sentence starters
		"The": true, "A": true, "An": true, "This": true, "That": true,
		"These": true, "Those": true, "What": true, "Who": true,
		"When": true, "Where": true, "Why": true, "How": true,
		"Which": true, "Whose": true,
		// Pronouns
		"I": true, "You": true, "We": true, "They": true, "It": true,
		"He": true, "She": true,
		// Days/months (handled separately in temporal)
		"Monday": true, "Tuesday": true, "Wednesday": true, "Thursday": true,
		"Friday": true, "Saturday": true, "Sunday": true,
		"January": true, "February": true, "March": true, "April": true,
		"May": true, "June": true, "July": true, "August": true,
		"September": true, "October": true, "November": true, "December": true,
		// Common verbs
		"Is": true, "Are": true, "Was": true, "Were": true,
		"Do": true, "Does": true, "Did": true, "Has": true, "Have": true,
		"Can": true, "Could": true, "Will": true, "Would": true,
		"Should": true, "Might": true, // "May" is already listed as month
		// Other
		"Please": true, "Tell": true, "Show": true, "Give": true,
		"Find": true, "Get": true, "Let": true, "Make": true,
	}
	return stopWords[word]
}

// ============================================================================
// Query Classification
// ============================================================================

// QueryType represents the type of query for retrieval strategy selection.
type QueryType string

const (
	QueryTypeFactual    QueryType = "factual"    // What, who questions
	QueryTypeTemporal   QueryType = "temporal"   // When-based questions
	QueryTypeSpatial    QueryType = "spatial"    // Where-based questions
	QueryTypeCausal     QueryType = "causal"     // Why, how questions
	QueryTypeComparison QueryType = "comparison" // Comparison queries
	QueryTypeOpen       QueryType = "open"       // Open-ended queries
)

// ClassifyQuery determines the type of query for strategy selection.
func (ce *ComplexityEstimator) ClassifyQuery(query string) QueryType {
	query = strings.TrimSpace(strings.ToLower(query))

	// Check for comparison keywords first (highest priority)
	if patternComparison.MatchString(query) {
		return QueryTypeComparison
	}

	// Check for causal questions
	if patternHardQuestion.MatchString(query) {
		return QueryTypeCausal
	}

	// Check for temporal questions
	if strings.HasPrefix(query, "when") || patternTemporalRef.MatchString(query) {
		return QueryTypeTemporal
	}

	// Check for spatial questions
	if strings.HasPrefix(query, "where") {
		return QueryTypeSpatial
	}

	// Check for factual questions
	if patternEasyQuestion.MatchString(query) {
		return QueryTypeFactual
	}

	// Default to open-ended
	return QueryTypeOpen
}

// ============================================================================
// Keyword Extraction for Retrieval
// ============================================================================

// ExtractQueryKeywords extracts important keywords from a query for lexical search.
func (ce *ComplexityEstimator) ExtractQueryKeywords(query string) []string {
	query = strings.ToLower(query)
	words := strings.Fields(query)

	keywords := make([]string, 0, len(words))
	for _, word := range words {
		// Clean punctuation
		word = strings.TrimFunc(word, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsNumber(r)
		})

		if len(word) < 2 {
			continue
		}

		// Skip common stop words
		if isLexicalStopWord(word) {
			continue
		}

		keywords = append(keywords, word)
	}

	return keywords
}

// isLexicalStopWord checks if a word is a stop word for lexical search.
func isLexicalStopWord(word string) bool {
	stopWords := map[string]bool{
		// Common English stop words
		"a": true, "an": true, "the": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "must": true, "shall": true,
		"i": true, "you": true, "we": true, "they": true, "it": true,
		"he": true, "she": true, "me": true, "him": true, "her": true,
		"us": true, "them": true, "my": true, "your": true, "his": true,
		"its": true, "our": true, "their": true,
		"this": true, "that": true, "these": true, "those": true,
		"what": true, "who": true, "when": true, "where": true,
		"why": true, "how": true, "which": true, "whose": true,
		"and": true, "or": true, "but": true, "if": true, "then": true,
		"else": true, "so": true, "as": true, "than": true,
		"of": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "with": true, "by": true, "from": true, "about": true,
		"into": true, "through": true, "during": true, "before": true,
		"after": true, "above": true, "below": true, "between": true,
		"can": true, "tell": true, "please": true, "know": true,
	}
	return stopWords[word]
}

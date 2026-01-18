package omem

import (
	"context"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
)

// MultiViewIndexer provides hybrid indexing for facts.
// It generates multiple views of each fact:
// - Semantic: Dense vector embeddings for similarity search
// - Lexical: Keywords for BM25/full-text search
// - Symbolic: Structured metadata for filtering
type MultiViewIndexer struct {
	config MultiViewConfig

	// Embedding provider
	embedder func(ctx context.Context, text string) ([]float32, error)

	// Stopwords for keyword extraction
	stopwords map[string]bool

	// Porter stemmer cache (simplified)
	stemCache map[string]string
}

// IndexedFact represents a fact with all its index views.
type IndexedFact struct {
	// Original fact content
	Text       string
	AtomicText string

	// Classification
	Category   FactCategory
	Importance float64

	// Semantic view
	Embedding []float32

	// Lexical view
	Keywords []string

	// Symbolic view (metadata)
	Metadata FactMetadata

	// Entity references
	Entities []EntityRef
}

// FactMetadata contains structured metadata extracted from a fact.
type FactMetadata struct {
	// Temporal
	TimestampAnchor *time.Time
	TimeExpressions []string

	// Spatial
	Location string
	Places   []string

	// Entities
	PersonNames  []string
	OrgNames     []string
	ConceptNames []string

	// Categorical
	Topics []string

	// Quality signals
	IsQuestion  bool
	IsNegation  bool
	HasNumbers  bool
	Specificity float64 // Higher = more specific
}

// NewMultiViewIndexer creates a new multi-view indexer.
func NewMultiViewIndexer(cfg MultiViewConfig, embedder func(ctx context.Context, text string) ([]float32, error)) *MultiViewIndexer {
	cfg = applyMultiViewDefaults(cfg)

	return &MultiViewIndexer{
		config:    cfg,
		embedder:  embedder,
		stopwords: buildStopwordSet(),
		stemCache: make(map[string]string),
	}
}

func applyMultiViewDefaults(cfg MultiViewConfig) MultiViewConfig {
	if cfg.MaxKeywordsPerFact <= 0 {
		cfg.MaxKeywordsPerFact = 20
	}
	if cfg.BM25_K1 <= 0 {
		cfg.BM25_K1 = 1.2
	}
	if cfg.BM25_B <= 0 || cfg.BM25_B > 1 {
		cfg.BM25_B = 0.75
	}
	return cfg
}

// Index generates all views for a fact.
func (mvi *MultiViewIndexer) Index(ctx context.Context, text, atomicText string, category FactCategory, importance float64) (*IndexedFact, error) {
	if mvi == nil {
		return nil, nil
	}

	indexed := &IndexedFact{
		Text:       text,
		AtomicText: atomicText,
		Category:   category,
		Importance: importance,
	}

	// Use atomic text for indexing (more precise)
	indexText := atomicText
	if indexText == "" {
		indexText = text
	}

	// Generate embedding (semantic view)
	if mvi.embedder != nil {
		emb, err := mvi.embedder(ctx, indexText)
		if err == nil {
			indexed.Embedding = emb
		}
	}

	// Extract keywords (lexical view)
	if mvi.config.ExtractKeywords {
		indexed.Keywords = mvi.extractKeywords(indexText)
	}

	// Extract metadata (symbolic view)
	indexed.Metadata = mvi.extractMetadata(indexText)

	// Extract entities
	indexed.Entities = mvi.extractEntityRefs(indexText)

	return indexed, nil
}

// IndexBatch indexes multiple facts in parallel.
func (mvi *MultiViewIndexer) IndexBatch(ctx context.Context, facts []struct {
	Text       string
	AtomicText string
	Category   FactCategory
	Importance float64
}) ([]*IndexedFact, error) {
	results := make([]*IndexedFact, len(facts))

	for i, f := range facts {
		indexed, err := mvi.Index(ctx, f.Text, f.AtomicText, f.Category, f.Importance)
		if err != nil {
			continue
		}
		results[i] = indexed
	}

	return results, nil
}

// ============================================================================
// Keyword Extraction
// ============================================================================

// extractKeywords extracts keywords from text for BM25 search.
func (mvi *MultiViewIndexer) extractKeywords(text string) []string {
	// Tokenize
	tokens := tokenize(text)

	// Process each token
	keywords := make(map[string]int) // keyword -> frequency
	for _, token := range tokens {
		// Skip stopwords
		if mvi.stopwords[token] {
			continue
		}

		// Skip short tokens
		if len(token) < 2 {
			continue
		}

		// Stem the token
		stemmed := mvi.stem(token)

		keywords[stemmed]++
	}

	// Sort by frequency and take top N
	type kv struct {
		Key   string
		Value int
	}
	var sorted []kv
	for k, v := range keywords {
		sorted = append(sorted, kv{k, v})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Value > sorted[j].Value
	})

	result := make([]string, 0, mvi.config.MaxKeywordsPerFact)
	for i := 0; i < len(sorted) && i < mvi.config.MaxKeywordsPerFact; i++ {
		result = append(result, sorted[i].Key)
	}

	return result
}

// tokenize splits text into tokens.
func tokenize(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Split on non-alphanumeric characters
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
		} else if current.Len() > 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
}

// stem applies a simplified Porter stemmer.
func (mvi *MultiViewIndexer) stem(word string) string {
	// Check cache
	if stemmed, ok := mvi.stemCache[word]; ok {
		return stemmed
	}

	stemmed := simpleStem(word)
	mvi.stemCache[word] = stemmed
	return stemmed
}

// simpleStem is a simplified Porter stemmer for English.
func simpleStem(word string) string {
	if len(word) < 4 {
		return word
	}

	// Remove common suffixes
	suffixes := []struct {
		suffix      string
		replacement string
		minLen      int
	}{
		{"ization", "ize", 3},
		{"ational", "ate", 3},
		{"fulness", "ful", 3},
		{"ousness", "ous", 3},
		{"iveness", "ive", 3},
		{"tional", "tion", 3},
		{"biliti", "ble", 3},
		{"ement", "", 3},
		{"ness", "", 3},
		{"ment", "", 3},
		{"able", "", 3},
		{"ible", "", 3},
		{"ance", "", 3},
		{"ence", "", 3},
		{"ally", "al", 3},
		{"tion", "", 3},
		{"sion", "", 3},
		{"izer", "ize", 3},
		{"ator", "ate", 3},
		{"ling", "", 3},
		{"ing", "", 3},
		{"ies", "y", 2},
		{"ied", "y", 2},
		{"ion", "", 3},
		{"ity", "", 3},
		{"ful", "", 3},
		{"ous", "", 3},
		{"ive", "", 3},
		{"ess", "", 3},
		{"ist", "", 3},
		{"ism", "", 3},
		{"ial", "", 3},
		{"ual", "", 3},
		{"ly", "", 2},
		{"ed", "", 2},
		{"er", "", 2},
		{"es", "", 2},
		{"'s", "", 2},
		{"s", "", 2},
	}

	for _, s := range suffixes {
		if strings.HasSuffix(word, s.suffix) && len(word)-len(s.suffix) >= s.minLen {
			return word[:len(word)-len(s.suffix)] + s.replacement
		}
	}

	return word
}

// buildStopwordSet returns a set of English stopwords.
func buildStopwordSet() map[string]bool {
	words := []string{
		// Articles
		"a", "an", "the",
		// Pronouns
		"i", "me", "my", "myself", "we", "our", "ours", "ourselves",
		"you", "your", "yours", "yourself", "yourselves",
		"he", "him", "his", "himself", "she", "her", "hers", "herself",
		"it", "its", "itself", "they", "them", "their", "theirs", "themselves",
		"what", "which", "who", "whom", "this", "that", "these", "those",
		// Verbs
		"am", "is", "are", "was", "were", "be", "been", "being",
		"have", "has", "had", "having", "do", "does", "did", "doing",
		"would", "should", "could", "ought", "might", "must", "shall", "will", "can",
		// Prepositions
		"about", "above", "across", "after", "against", "along", "among", "around",
		"at", "before", "behind", "below", "beneath", "beside", "between", "beyond",
		"by", "down", "during", "except", "for", "from", "in", "inside", "into",
		"near", "of", "off", "on", "onto", "out", "outside", "over", "past",
		"through", "throughout", "to", "toward", "under", "underneath", "until",
		"up", "upon", "with", "within", "without",
		// Conjunctions
		"and", "but", "or", "nor", "for", "yet", "so", "because", "although",
		"while", "if", "when", "where", "whether", "though", "unless", "since",
		// Other common words
		"no", "not", "only", "own", "same", "than", "too", "very", "just",
		"also", "now", "here", "there", "then", "once", "both", "each", "few",
		"more", "most", "other", "some", "such", "any", "all", "every", "either",
		"neither", "much", "many", "how", "why", "again", "ever", "always",
		"never", "sometimes", "often", "usually", "really", "actually",
	}

	set := make(map[string]bool, len(words))
	for _, w := range words {
		set[w] = true
	}
	return set
}

// ============================================================================
// Metadata Extraction
// ============================================================================

// Metadata extraction patterns
var (
	// Date patterns
	patternDateYMD     = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}\b`)
	patternDateMDY     = regexp.MustCompile(`\b\d{1,2}/\d{1,2}/\d{2,4}\b`)
	patternDateWritten = regexp.MustCompile(`(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b`)
	patternYear        = regexp.MustCompile(`\b(19|20)\d{2}\b`)

	// Time patterns
	patternTime12h = regexp.MustCompile(`\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b`)
	patternTime24h = regexp.MustCompile(`\b\d{1,2}:\d{2}(?::\d{2})?\b`)

	// Location patterns
	patternCity    = regexp.MustCompile(`(?i)\b(in|at|from|to|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b`)
	patternCountry = regexp.MustCompile(`(?i)\b(USA|UK|US|Canada|Australia|Germany|France|Japan|China|India|Brazil|Mexico|Spain|Italy|Russia)\b`)
	patternState   = regexp.MustCompile(`(?i)\b(California|Texas|Florida|New York|Illinois|Pennsylvania|Ohio|Georgia|Michigan|North Carolina)\b`)

	// Person name patterns (capitalized words)
	patternProperNoun = regexp.MustCompile(`\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b`)

	// Number patterns
	patternNumber   = regexp.MustCompile(`\b\d+(?:\.\d+)?%?\b`)
	patternCurrency = regexp.MustCompile(`\$\d+(?:,\d{3})*(?:\.\d{2})?`)

	// Question pattern
	patternQuestion = regexp.MustCompile(`\?$`)

	// Negation patterns
	patternNegation = regexp.MustCompile(`(?i)\b(not|no|never|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't|isn't|aren't|wasn't|weren't)\b`)
)

// extractMetadata extracts structured metadata from text.
func (mvi *MultiViewIndexer) extractMetadata(text string) FactMetadata {
	meta := FactMetadata{}

	// Time expressions
	meta.TimeExpressions = append(meta.TimeExpressions, patternDateYMD.FindAllString(text, -1)...)
	meta.TimeExpressions = append(meta.TimeExpressions, patternDateMDY.FindAllString(text, -1)...)
	meta.TimeExpressions = append(meta.TimeExpressions, patternDateWritten.FindAllString(text, -1)...)

	// Try to parse first date as timestamp anchor
	if len(meta.TimeExpressions) > 0 {
		if t, err := parseFlexibleDate(meta.TimeExpressions[0]); err == nil {
			meta.TimestampAnchor = &t
		}
	}

	// Location extraction
	places := make(map[string]bool)
	for _, match := range patternCity.FindAllStringSubmatch(text, -1) {
		if len(match) >= 3 {
			places[match[2]] = true
		}
	}
	for _, match := range patternCountry.FindAllString(text, -1) {
		places[match] = true
	}
	for _, match := range patternState.FindAllString(text, -1) {
		places[match] = true
	}
	for p := range places {
		meta.Places = append(meta.Places, p)
	}
	if len(meta.Places) > 0 {
		meta.Location = meta.Places[0]
	}

	// Person/Org names (simple heuristic)
	properNouns := patternProperNoun.FindAllString(text, -1)
	for _, noun := range properNouns {
		if !isCommonWord(noun) && !contains(meta.Places, noun) {
			// Heuristic: single word = might be person, multi-word = might be org
			if strings.Contains(noun, " ") {
				meta.OrgNames = append(meta.OrgNames, noun)
			} else {
				meta.PersonNames = append(meta.PersonNames, noun)
			}
		}
	}

	// Quality signals
	meta.IsQuestion = patternQuestion.MatchString(strings.TrimSpace(text))
	meta.IsNegation = patternNegation.MatchString(text)
	meta.HasNumbers = patternNumber.MatchString(text) || patternCurrency.MatchString(text)

	// Specificity score (heuristic)
	meta.Specificity = mvi.calculateSpecificity(text, meta)

	return meta
}

// calculateSpecificity estimates how specific/concrete a fact is.
func (mvi *MultiViewIndexer) calculateSpecificity(text string, meta FactMetadata) float64 {
	score := 0.5 // Base score

	// More specific if has dates
	if len(meta.TimeExpressions) > 0 {
		score += 0.1
	}
	if meta.TimestampAnchor != nil {
		score += 0.1
	}

	// More specific if has locations
	if len(meta.Places) > 0 {
		score += 0.1
	}

	// More specific if has names
	if len(meta.PersonNames) > 0 || len(meta.OrgNames) > 0 {
		score += 0.1
	}

	// More specific if has numbers
	if meta.HasNumbers {
		score += 0.05
	}

	// Less specific if short
	wordCount := len(strings.Fields(text))
	if wordCount < 5 {
		score -= 0.1
	} else if wordCount > 15 {
		score += 0.05
	}

	return clampFloat(score, 0.0, 1.0)
}

// parseFlexibleDate attempts to parse various date formats.
func parseFlexibleDate(s string) (time.Time, error) {
	formats := []string{
		"2006-01-02",
		"01/02/2006",
		"1/2/2006",
		"01/02/06",
		"January 2, 2006",
		"Jan 2, 2006",
		"January 2 2006",
		"2 January 2006",
	}

	for _, format := range formats {
		if t, err := time.Parse(format, s); err == nil {
			return t, nil
		}
	}

	return time.Time{}, nil
}

// ============================================================================
// Entity Reference Extraction
// ============================================================================

// extractEntityRefs extracts entity references from text.
func (mvi *MultiViewIndexer) extractEntityRefs(text string) []EntityRef {
	var refs []EntityRef
	seen := make(map[string]bool)

	// Use patterns from atomic_encoder
	personMatches := patternPersonName.FindAllStringSubmatch(text, -1)
	for _, match := range personMatches {
		if len(match) >= 3 {
			name := match[2]
			if !seen[strings.ToLower(name)] {
				seen[strings.ToLower(name)] = true
				refs = append(refs, EntityRef{Name: name, Type: EntityPerson})
			}
		}
	}

	relativeMatches := patternMyRelative.FindAllStringSubmatch(text, -1)
	for _, match := range relativeMatches {
		if len(match) >= 3 {
			name := match[2]
			if !seen[strings.ToLower(name)] {
				seen[strings.ToLower(name)] = true
				refs = append(refs, EntityRef{Name: name, Type: EntityPerson})
			}
		}
	}

	companyMatches := patternCompany.FindAllStringSubmatch(text, -1)
	for _, match := range companyMatches {
		if len(match) >= 2 {
			name := strings.TrimSpace(match[0])
			if !seen[strings.ToLower(name)] {
				seen[strings.ToLower(name)] = true
				refs = append(refs, EntityRef{Name: name, Type: EntityOrganization})
			}
		}
	}

	return refs
}

// contains checks if a string slice contains a value.
func contains(slice []string, val string) bool {
	for _, s := range slice {
		if strings.EqualFold(s, val) {
			return true
		}
	}
	return false
}

// ============================================================================
// Scoring Helpers
// ============================================================================

// CalculateLexicalScore computes a simple lexical similarity score.
func (mvi *MultiViewIndexer) CalculateLexicalScore(queryKeywords, factKeywords []string) float64 {
	if len(queryKeywords) == 0 || len(factKeywords) == 0 {
		return 0.0
	}

	// Build keyword set for fact
	factSet := make(map[string]bool)
	for _, kw := range factKeywords {
		factSet[kw] = true
	}

	// Count matches
	matches := 0
	for _, qkw := range queryKeywords {
		if factSet[qkw] {
			matches++
		}
	}

	// Jaccard-like score
	union := len(queryKeywords) + len(factKeywords) - matches
	if union == 0 {
		return 0.0
	}

	return float64(matches) / float64(union)
}

// CalculateSymbolicScore computes metadata matching score.
func (mvi *MultiViewIndexer) CalculateSymbolicScore(queryMeta, factMeta FactMetadata) float64 {
	score := 0.0
	maxScore := 0.0

	// Time overlap
	if len(queryMeta.TimeExpressions) > 0 && len(factMeta.TimeExpressions) > 0 {
		maxScore += 0.25
		for _, qt := range queryMeta.TimeExpressions {
			for _, ft := range factMeta.TimeExpressions {
				if qt == ft {
					score += 0.25
					break
				}
			}
		}
	}

	// Location overlap
	if len(queryMeta.Places) > 0 && len(factMeta.Places) > 0 {
		maxScore += 0.25
		for _, qp := range queryMeta.Places {
			for _, fp := range factMeta.Places {
				if strings.EqualFold(qp, fp) {
					score += 0.25
					break
				}
			}
		}
	}

	// Person overlap
	if len(queryMeta.PersonNames) > 0 && len(factMeta.PersonNames) > 0 {
		maxScore += 0.25
		for _, qn := range queryMeta.PersonNames {
			for _, fn := range factMeta.PersonNames {
				if strings.EqualFold(qn, fn) {
					score += 0.25
					break
				}
			}
		}
	}

	// Org overlap
	if len(queryMeta.OrgNames) > 0 && len(factMeta.OrgNames) > 0 {
		maxScore += 0.25
		for _, qo := range queryMeta.OrgNames {
			for _, fo := range factMeta.OrgNames {
				if strings.EqualFold(qo, fo) {
					score += 0.25
					break
				}
			}
		}
	}

	if maxScore == 0 {
		return 0.0
	}

	return score / maxScore
}

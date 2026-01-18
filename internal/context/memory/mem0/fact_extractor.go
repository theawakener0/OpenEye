package mem0

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
)

// FactExtractor extracts facts, entities, and relationships from conversations.
// It uses the LLM for extraction and the embedding provider for vector generation.
type FactExtractor struct {
	config   ExtractionConfig
	manager  *runtime.Manager
	embedder embedding.Provider
	prompts  *PromptTemplates
	mu       sync.Mutex

	// Async extraction queue
	extractQueue chan extractionJob
	stopChan     chan struct{}
	wg           sync.WaitGroup
}

// extractionJob represents a pending extraction task.
type extractionJob struct {
	turns    []ConversationTurn
	callback func([]ExtractedFact, []ExtractedEntity, []ExtractedRelationship, error)
}

// ExtractionResult contains the complete extraction output.
type ExtractionResult struct {
	Facts         []ExtractedFact
	Entities      []ExtractedEntity
	Relationships []ExtractedRelationship
	ProcessedAt   time.Time
	TurnIDs       []string
}

// NewFactExtractor creates a new fact extractor.
func NewFactExtractor(cfg ExtractionConfig, manager *runtime.Manager, embedder embedding.Provider) (*FactExtractor, error) {
	if manager == nil {
		return nil, errors.New("runtime manager required")
	}

	cfg = applyExtractionDefaults(cfg)

	fe := &FactExtractor{
		config:   cfg,
		manager:  manager,
		embedder: embedder,
		prompts:  NewPromptTemplates(),
	}

	if cfg.Async {
		fe.extractQueue = make(chan extractionJob, 100)
		fe.stopChan = make(chan struct{})
		fe.wg.Add(1)
		go fe.asyncWorker()
	}

	return fe, nil
}

// applyExtractionDefaults fills in missing configuration values.
func applyExtractionDefaults(cfg ExtractionConfig) ExtractionConfig {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 3
	}
	if cfg.MinTextLength <= 0 {
		cfg.MinTextLength = 10
	}
	if cfg.MaxFactsPerExtraction <= 0 {
		cfg.MaxFactsPerExtraction = 10
	}
	return cfg
}

// Extract performs synchronous fact extraction from conversation turns.
func (fe *FactExtractor) Extract(ctx context.Context, turns []ConversationTurn) (*ExtractionResult, error) {
	if fe == nil {
		return nil, errors.New("fact extractor not initialized")
	}

	if !fe.config.Enabled {
		return &ExtractionResult{ProcessedAt: time.Now()}, nil
	}

	// Filter turns with sufficient content
	var validTurns []ConversationTurn
	for _, turn := range turns {
		if len(strings.TrimSpace(turn.Content)) >= fe.config.MinTextLength {
			validTurns = append(validTurns, turn)
		}
	}

	if len(validTurns) == 0 {
		return &ExtractionResult{ProcessedAt: time.Now()}, nil
	}

	// Format conversation for extraction
	conversation := fe.prompts.ConversationContextPrompt(validTurns)

	// Extract facts
	facts, err := fe.extractFacts(ctx, conversation)
	if err != nil {
		return nil, fmt.Errorf("fact extraction failed: %w", err)
	}

	result := &ExtractionResult{
		Facts:       facts,
		ProcessedAt: time.Now(),
	}

	// Collect turn IDs
	for _, turn := range validTurns {
		if turn.TurnID != "" {
			result.TurnIDs = append(result.TurnIDs, turn.TurnID)
		}
	}

	// Extract entities if enabled
	if fe.config.ExtractEntities && len(facts) > 0 {
		entities, err := fe.extractEntities(ctx, facts)
		if err != nil {
			log.Printf("warning: entity extraction failed: %v", err)
		} else {
			result.Entities = entities
		}
	}

	// Extract relationships if enabled
	if fe.config.ExtractRelationships && len(result.Entities) > 1 {
		relationships, err := fe.extractRelationships(ctx, facts, result.Entities)
		if err != nil {
			log.Printf("warning: relationship extraction failed: %v", err)
		} else {
			result.Relationships = relationships
		}
	}

	// Generate embeddings for facts
	if fe.embedder != nil {
		if err := fe.generateFactEmbeddings(ctx, facts); err != nil {
			log.Printf("warning: embedding generation failed: %v", err)
		}
	}

	return result, nil
}

// ExtractAsync queues an extraction job for async processing.
func (fe *FactExtractor) ExtractAsync(turns []ConversationTurn, callback func([]ExtractedFact, []ExtractedEntity, []ExtractedRelationship, error)) error {
	if fe == nil {
		return errors.New("fact extractor not initialized")
	}

	if !fe.config.Async {
		return errors.New("async extraction not enabled")
	}

	select {
	case fe.extractQueue <- extractionJob{turns: turns, callback: callback}:
		return nil
	default:
		return errors.New("extraction queue full")
	}
}

// extractFacts calls the LLM to extract facts from conversation.
func (fe *FactExtractor) extractFacts(ctx context.Context, conversation string) ([]ExtractedFact, error) {
	prompt := fe.prompts.FactExtractionPrompt(conversation, fe.config.MaxFactsPerExtraction)

	resp, err := fe.manager.Generate(ctx, runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:     512,
			Temperature:   0.3, // Lower temperature for more consistent extraction
			TopK:          40,
			TopP:          0.9,
			MinP:          0.05,
			RepeatPenalty: 1.1,
			RepeatLastN:   64,
		},
	})
	if err != nil {
		return nil, err
	}

	facts := ParseFactExtractionResponse(resp.Text)

	// Limit to max facts
	if len(facts) > fe.config.MaxFactsPerExtraction {
		facts = facts[:fe.config.MaxFactsPerExtraction]
	}

	return facts, nil
}

// extractEntities calls the LLM to extract entities from facts.
func (fe *FactExtractor) extractEntities(ctx context.Context, facts []ExtractedFact) ([]ExtractedEntity, error) {
	// Combine all fact texts
	var allText strings.Builder
	for _, f := range facts {
		allText.WriteString(f.Text)
		allText.WriteString(" ")
	}

	prompt := fe.prompts.EntityExtractionPrompt(allText.String())

	resp, err := fe.manager.Generate(ctx, runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:     256,
			Temperature:   0.2,
			TopK:          40,
			TopP:          0.9,
			MinP:          0.05,
			RepeatPenalty: 1.1,
			RepeatLastN:   64,
		},
	})
	if err != nil {
		return nil, err
	}

	entities := ParseEntityExtractionResponse(resp.Text)

	// Deduplicate entities by name (case-insensitive)
	seen := make(map[string]bool)
	var unique []ExtractedEntity
	for _, e := range entities {
		key := strings.ToLower(e.Name)
		if !seen[key] {
			seen[key] = true
			unique = append(unique, e)
		}
	}

	return unique, nil
}

// extractRelationships calls the LLM to extract relationships between entities.
func (fe *FactExtractor) extractRelationships(ctx context.Context, facts []ExtractedFact, entities []ExtractedEntity) ([]ExtractedRelationship, error) {
	if len(entities) < 2 {
		return nil, nil
	}

	// Get entity names
	entityNames := make([]string, len(entities))
	for i, e := range entities {
		entityNames[i] = e.Name
	}

	var allRelationships []ExtractedRelationship

	// Extract relationships for each fact
	for _, fact := range facts {
		prompt := fe.prompts.RelationshipExtractionPrompt(fact.Text, entityNames)

		resp, err := fe.manager.Generate(ctx, runtime.Request{
			Prompt: prompt,
			Options: runtime.GenerationOptions{
				MaxTokens:     128,
				Temperature:   0.2,
				TopK:          40,
				TopP:          0.9,
				MinP:          0.05,
				RepeatPenalty: 1.1,
				RepeatLastN:   64,
			},
		})
		if err != nil {
			continue
		}

		rels := ParseRelationshipExtractionResponse(resp.Text)
		allRelationships = append(allRelationships, rels...)
	}

	// Deduplicate relationships
	seen := make(map[string]bool)
	var unique []ExtractedRelationship
	for _, r := range allRelationships {
		key := fmt.Sprintf("%s|%s|%s", strings.ToLower(r.SourceName), r.RelationType, strings.ToLower(r.TargetName))
		if !seen[key] {
			seen[key] = true
			unique = append(unique, r)
		}
	}

	return unique, nil
}

// generateFactEmbeddings generates embedding vectors for facts.
// This pre-warms the embedding cache if a CachedProvider is used.
func (fe *FactExtractor) generateFactEmbeddings(ctx context.Context, facts []ExtractedFact) error {
	if fe.embedder == nil {
		return nil
	}

	// Generate embeddings individually (the Provider interface only has single Embed)
	for _, f := range facts {
		_, err := fe.embedder.Embed(ctx, f.Text)
		if err != nil {
			return err
		}
	}

	return nil
}

// EvaluateImportance uses the LLM to evaluate the importance of a fact.
func (fe *FactExtractor) EvaluateImportance(ctx context.Context, factText string) (float64, error) {
	prompt := fe.prompts.ImportanceEvaluationPrompt(factText)

	resp, err := fe.manager.Generate(ctx, runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:     16,
			Temperature:   0.1,
			TopK:          40,
			TopP:          0.9,
			MinP:          0.05,
			RepeatPenalty: 1.0,
			RepeatLastN:   0,
		},
	})
	if err != nil {
		return 0.5, err // Default to medium importance
	}

	importance := parseFloat(strings.TrimSpace(resp.Text), 0.5)
	return clampFloat(importance, 0.0, 1.0), nil
}

// ClassifyCategory uses the LLM to classify a fact into a category.
func (fe *FactExtractor) ClassifyCategory(ctx context.Context, factText string) (FactCategory, error) {
	prompt := fe.prompts.CategoryClassificationPrompt(factText)

	resp, err := fe.manager.Generate(ctx, runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:     16,
			Temperature:   0.1,
			TopK:          40,
			TopP:          0.9,
			MinP:          0.05,
			RepeatPenalty: 1.0,
			RepeatLastN:   0,
		},
	})
	if err != nil {
		return CategoryOther, err
	}

	return normalizeCategory(strings.TrimSpace(resp.Text)), nil
}

// asyncWorker processes extraction jobs from the queue.
func (fe *FactExtractor) asyncWorker() {
	defer fe.wg.Done()

	for {
		select {
		case <-fe.stopChan:
			return
		case job := <-fe.extractQueue:
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			result, err := fe.Extract(ctx, job.turns)
			cancel()

			if job.callback != nil {
				if err != nil {
					job.callback(nil, nil, nil, err)
				} else {
					job.callback(result.Facts, result.Entities, result.Relationships, nil)
				}
			}
		}
	}
}

// Close stops the async worker and cleans up resources.
func (fe *FactExtractor) Close() error {
	if fe == nil {
		return nil
	}

	if fe.config.Async && fe.stopChan != nil {
		close(fe.stopChan)
		fe.wg.Wait()
	}

	return nil
}

// ExtractedFactWithEmbedding extends ExtractedFact with an embedding vector.
type ExtractedFactWithEmbedding struct {
	ExtractedFact
	Embedding []float32
}

// ExtractWithEmbeddings performs extraction and generates embeddings for facts.
func (fe *FactExtractor) ExtractWithEmbeddings(ctx context.Context, turns []ConversationTurn) ([]ExtractedFactWithEmbedding, []ExtractedEntity, []ExtractedRelationship, error) {
	result, err := fe.Extract(ctx, turns)
	if err != nil {
		return nil, nil, nil, err
	}

	if len(result.Facts) == 0 || fe.embedder == nil {
		// Return facts without embeddings
		factsWithEmb := make([]ExtractedFactWithEmbedding, len(result.Facts))
		for i, f := range result.Facts {
			factsWithEmb[i] = ExtractedFactWithEmbedding{ExtractedFact: f}
		}
		return factsWithEmb, result.Entities, result.Relationships, nil
	}

	// Generate embeddings individually
	embeddings := make([][]float32, len(result.Facts))
	for i, f := range result.Facts {
		emb, err := fe.embedder.Embed(ctx, f.Text)
		if err != nil {
			// Return facts without embeddings if embedding fails
			factsWithEmb := make([]ExtractedFactWithEmbedding, len(result.Facts))
			for j, fact := range result.Facts {
				factsWithEmb[j] = ExtractedFactWithEmbedding{ExtractedFact: fact}
			}
			return factsWithEmb, result.Entities, result.Relationships, nil
		}
		embeddings[i] = emb
	}

	// Combine facts with embeddings
	factsWithEmb := make([]ExtractedFactWithEmbedding, len(result.Facts))
	for i, f := range result.Facts {
		factsWithEmb[i] = ExtractedFactWithEmbedding{
			ExtractedFact: f,
			Embedding:     embeddings[i],
		}
	}

	return factsWithEmb, result.Entities, result.Relationships, nil
}

// QuickExtract performs a lightweight extraction for simple use cases.
// It skips entity and relationship extraction.
func (fe *FactExtractor) QuickExtract(ctx context.Context, text string) ([]ExtractedFact, error) {
	if len(strings.TrimSpace(text)) < fe.config.MinTextLength {
		return nil, nil
	}

	turns := []ConversationTurn{{
		Role:    "user",
		Content: text,
	}}

	// Temporarily disable entity/relationship extraction
	origEntities := fe.config.ExtractEntities
	origRels := fe.config.ExtractRelationships
	fe.config.ExtractEntities = false
	fe.config.ExtractRelationships = false

	result, err := fe.Extract(ctx, turns)

	// Restore config
	fe.config.ExtractEntities = origEntities
	fe.config.ExtractRelationships = origRels

	if err != nil {
		return nil, err
	}

	return result.Facts, nil
}

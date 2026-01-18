package mem0

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
)

// MemoryUpdater handles intelligent memory updates using ADD/UPDATE/DELETE/NOOP operations.
// It compares new facts against existing memories and decides the appropriate action.
type MemoryUpdater struct {
	config   UpdateConfig
	store    *FactStore
	graph    *EntityGraph
	manager  *runtime.Manager
	embedder embedding.Provider
	prompts  *PromptTemplates
}

// UpdateResult represents the outcome of processing a new fact.
type UpdateResult struct {
	Operation      UpdateOperation
	NewFactID      int64   // For ADD operations
	UpdatedFactID  int64   // For UPDATE operations
	DeletedFactIDs []int64 // For DELETE operations
	Reason         string
	MergedText     string // For UPDATE operations with merging
}

// NewMemoryUpdater creates a new memory updater.
func NewMemoryUpdater(cfg UpdateConfig, store *FactStore, graph *EntityGraph, manager *runtime.Manager, embedder embedding.Provider) (*MemoryUpdater, error) {
	if store == nil {
		return nil, errors.New("fact store required")
	}
	if manager == nil {
		return nil, errors.New("runtime manager required")
	}

	cfg = applyUpdateDefaults(cfg)

	return &MemoryUpdater{
		config:   cfg,
		store:    store,
		graph:    graph,
		manager:  manager,
		embedder: embedder,
		prompts:  NewPromptTemplates(),
	}, nil
}

// applyUpdateDefaults fills in missing configuration values.
func applyUpdateDefaults(cfg UpdateConfig) UpdateConfig {
	if cfg.ConflictThreshold <= 0 {
		cfg.ConflictThreshold = 0.85
	}
	if cfg.TopSimilarCount <= 0 {
		cfg.TopSimilarCount = 5
	}
	return cfg
}

// ProcessFact decides how to handle a new extracted fact.
func (mu *MemoryUpdater) ProcessFact(ctx context.Context, fact ExtractedFactWithEmbedding, sourceTurnIDs []string) (*UpdateResult, error) {
	if mu == nil || mu.store == nil {
		return nil, errors.New("memory updater not initialized")
	}

	if !mu.config.Enabled {
		// If updates are disabled, just add the fact directly
		return mu.addFactDirect(ctx, fact, sourceTurnIDs)
	}

	// Find similar existing facts
	var similarFacts []Fact
	var similarities []float64

	if len(fact.Embedding) > 0 {
		var err error
		similarFacts, similarities, err = mu.store.SearchSimilarFacts(ctx, fact.Embedding, mu.config.TopSimilarCount, false)
		if err != nil {
			log.Printf("warning: similarity search failed: %v", err)
		}
	}

	// If no similar facts found, this is a simple ADD
	if len(similarFacts) == 0 {
		return mu.addFactDirect(ctx, fact, sourceTurnIDs)
	}

	// Check for very high similarity (potential duplicate)
	if len(similarities) > 0 && similarities[0] >= mu.config.ConflictThreshold {
		// Very similar fact exists - use LLM to decide
		return mu.decideWithLLM(ctx, fact, similarFacts, similarities, sourceTurnIDs)
	}

	// Moderate similarity - still use LLM but likely ADD
	if len(similarities) > 0 && similarities[0] >= 0.5 {
		return mu.decideWithLLM(ctx, fact, similarFacts, similarities, sourceTurnIDs)
	}

	// Low similarity - safe to ADD
	return mu.addFactDirect(ctx, fact, sourceTurnIDs)
}

// ProcessFactBatch processes multiple facts efficiently.
func (mu *MemoryUpdater) ProcessFactBatch(ctx context.Context, facts []ExtractedFactWithEmbedding, sourceTurnIDs []string) ([]UpdateResult, error) {
	results := make([]UpdateResult, 0, len(facts))

	for _, fact := range facts {
		result, err := mu.ProcessFact(ctx, fact, sourceTurnIDs)
		if err != nil {
			log.Printf("warning: failed to process fact: %v", err)
			continue
		}
		if result != nil {
			results = append(results, *result)
		}
	}

	return results, nil
}

// addFactDirect adds a fact without LLM comparison.
func (mu *MemoryUpdater) addFactDirect(ctx context.Context, fact ExtractedFactWithEmbedding, sourceTurnIDs []string) (*UpdateResult, error) {
	newFact := Fact{
		Text:          fact.Text,
		Category:      fact.Category,
		Importance:    fact.Importance,
		Embedding:     fact.Embedding,
		CreatedAt:     time.Now(),
		SourceTurnIDs: strings.Join(sourceTurnIDs, ","),
	}

	id, err := mu.store.InsertFact(ctx, newFact)
	if err != nil {
		return nil, fmt.Errorf("failed to insert fact: %w", err)
	}

	return &UpdateResult{
		Operation: OpAdd,
		NewFactID: id,
		Reason:    "new information",
	}, nil
}

// decideWithLLM uses the LLM to decide how to handle the new fact.
func (mu *MemoryUpdater) decideWithLLM(ctx context.Context, newFact ExtractedFactWithEmbedding, existingFacts []Fact, similarities []float64, sourceTurnIDs []string) (*UpdateResult, error) {
	// Prepare existing facts for the prompt
	existingTexts := make([]string, len(existingFacts))
	for i, f := range existingFacts {
		existingTexts[i] = f.Text
	}

	prompt := mu.prompts.MemoryUpdatePrompt(newFact.Text, existingTexts)

	resp, err := mu.manager.Generate(ctx, runtime.Request{
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
		// On LLM failure, default to ADD
		log.Printf("warning: LLM decision failed, defaulting to ADD: %v", err)
		return mu.addFactDirect(ctx, newFact, sourceTurnIDs)
	}

	operation, targetIdx, reason := ParseMemoryUpdateResponse(resp.Text)

	switch operation {
	case OpAdd:
		return mu.addFactDirect(ctx, newFact, sourceTurnIDs)

	case OpUpdate:
		return mu.handleUpdate(ctx, newFact, existingFacts, targetIdx, reason, sourceTurnIDs)

	case OpDelete:
		return mu.handleDelete(ctx, newFact, existingFacts, targetIdx, reason, sourceTurnIDs)

	case OpNoop:
		return &UpdateResult{
			Operation: OpNoop,
			Reason:    reason,
		}, nil

	default:
		// Unknown operation, default to ADD
		return mu.addFactDirect(ctx, newFact, sourceTurnIDs)
	}
}

// handleUpdate handles UPDATE operations.
func (mu *MemoryUpdater) handleUpdate(ctx context.Context, newFact ExtractedFactWithEmbedding, existingFacts []Fact, targetIdx int, reason string, sourceTurnIDs []string) (*UpdateResult, error) {
	// Validate target index (1-based from LLM)
	if targetIdx < 1 || targetIdx > len(existingFacts) {
		// Invalid index, default to ADD
		return mu.addFactDirect(ctx, newFact, sourceTurnIDs)
	}

	existingFact := existingFacts[targetIdx-1]

	// Determine how to update: merge or replace
	var mergedText string
	var err error

	if mu.config.AutoResolveConflicts {
		// Use LLM to merge facts
		mergedText, err = mu.mergeFacts(ctx, newFact.Text, existingFact.Text)
		if err != nil {
			// On merge failure, use new fact text
			mergedText = newFact.Text
		}
	} else {
		// Replace with new fact text
		mergedText = newFact.Text
	}

	// Update the existing fact
	updatedFact := existingFact
	updatedFact.Text = mergedText
	updatedFact.LastAccessed = time.Now()

	// Update embedding if text changed significantly
	if mergedText != existingFact.Text && mu.embedder != nil {
		emb, err := mu.embedder.Embed(ctx, mergedText)
		if err == nil {
			updatedFact.Embedding = emb
		}
	}

	if err := mu.store.UpdateFact(ctx, updatedFact); err != nil {
		return nil, fmt.Errorf("failed to update fact: %w", err)
	}

	// Track supersession if enabled
	if mu.config.TrackSupersession {
		// The existing fact is being updated in place, so we don't need to track supersession
		// Supersession is more relevant for DELETE operations
	}

	return &UpdateResult{
		Operation:     OpUpdate,
		UpdatedFactID: existingFact.ID,
		Reason:        reason,
		MergedText:    mergedText,
	}, nil
}

// handleDelete handles DELETE operations.
func (mu *MemoryUpdater) handleDelete(ctx context.Context, newFact ExtractedFactWithEmbedding, existingFacts []Fact, targetIdx int, reason string, sourceTurnIDs []string) (*UpdateResult, error) {
	// Validate target index (1-based from LLM)
	if targetIdx < 1 || targetIdx > len(existingFacts) {
		// Invalid index, just add the new fact
		return mu.addFactDirect(ctx, newFact, sourceTurnIDs)
	}

	existingFact := existingFacts[targetIdx-1]

	// First add the new fact
	addResult, err := mu.addFactDirect(ctx, newFact, sourceTurnIDs)
	if err != nil {
		return nil, err
	}

	// Mark the old fact as obsolete
	var supersededBy *int64
	if mu.config.TrackSupersession && addResult.NewFactID > 0 {
		supersededBy = &addResult.NewFactID
	}

	if err := mu.store.MarkObsolete(ctx, existingFact.ID, supersededBy); err != nil {
		log.Printf("warning: failed to mark fact obsolete: %v", err)
	}

	return &UpdateResult{
		Operation:      OpDelete,
		NewFactID:      addResult.NewFactID,
		DeletedFactIDs: []int64{existingFact.ID},
		Reason:         reason,
	}, nil
}

// mergeFacts uses the LLM to merge two facts into one.
func (mu *MemoryUpdater) mergeFacts(ctx context.Context, newFactText, existingFactText string) (string, error) {
	prompt := mu.prompts.MemoryUpdateWithMergePrompt(newFactText, existingFactText)

	resp, err := mu.manager.Generate(ctx, runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:     128,
			Temperature:   0.3,
			TopK:          40,
			TopP:          0.9,
			MinP:          0.05,
			RepeatPenalty: 1.1,
			RepeatLastN:   64,
		},
	})
	if err != nil {
		return "", err
	}

	merged := strings.TrimSpace(resp.Text)
	if merged == "" {
		return newFactText, nil
	}

	return merged, nil
}

// ProcessEntities stores extracted entities in the graph.
func (mu *MemoryUpdater) ProcessEntities(ctx context.Context, entities []ExtractedEntity, factID int64) error {
	if mu.graph == nil || len(entities) == 0 {
		return nil
	}

	for _, entity := range entities {
		entStruct := Entity{
			Name:       entity.Name,
			EntityType: entity.EntityType,
			CreatedAt:  time.Now(),
		}

		// Generate embedding for entity name if possible
		if mu.embedder != nil {
			emb, err := mu.embedder.Embed(ctx, entity.Name)
			if err == nil {
				entStruct.Embedding = emb
			}
		}

		entityID, err := mu.graph.UpsertEntity(ctx, entStruct)
		if err != nil {
			log.Printf("warning: failed to upsert entity %q: %v", entity.Name, err)
			continue
		}

		// Link entity to fact
		if factID > 0 {
			if err := mu.graph.LinkFactToEntity(ctx, entityID, factID); err != nil {
				log.Printf("warning: failed to link entity to fact: %v", err)
			}
		}
	}

	return nil
}

// ProcessRelationships stores extracted relationships in the graph.
func (mu *MemoryUpdater) ProcessRelationships(ctx context.Context, relationships []ExtractedRelationship, factID int64) error {
	if mu.graph == nil || len(relationships) == 0 {
		return nil
	}

	for _, rel := range relationships {
		// Find or create source entity
		sourceEntity, err := mu.graph.FindEntityByName(ctx, rel.SourceName)
		if err != nil || sourceEntity == nil {
			// Create the entity
			sourceID, err := mu.graph.UpsertEntity(ctx, Entity{
				Name:       rel.SourceName,
				EntityType: EntityOther,
				CreatedAt:  time.Now(),
			})
			if err != nil {
				log.Printf("warning: failed to create source entity %q: %v", rel.SourceName, err)
				continue
			}
			sourceEntity = &Entity{ID: sourceID, Name: rel.SourceName}
		}

		// Find or create target entity
		targetEntity, err := mu.graph.FindEntityByName(ctx, rel.TargetName)
		if err != nil || targetEntity == nil {
			// Create the entity
			targetID, err := mu.graph.UpsertEntity(ctx, Entity{
				Name:       rel.TargetName,
				EntityType: EntityOther,
				CreatedAt:  time.Now(),
			})
			if err != nil {
				log.Printf("warning: failed to create target entity %q: %v", rel.TargetName, err)
				continue
			}
			targetEntity = &Entity{ID: targetID, Name: rel.TargetName}
		}

		// Create the relationship
		_, err = mu.graph.CreateRelationship(ctx, Relationship{
			SourceEntityID: sourceEntity.ID,
			TargetEntityID: targetEntity.ID,
			RelationType:   rel.RelationType,
			FactID:         factID,
			Confidence:     rel.Confidence,
			CreatedAt:      time.Now(),
		})
		if err != nil {
			log.Printf("warning: failed to create relationship: %v", err)
		}
	}

	return nil
}

// QuickAdd adds a fact without any LLM-based comparison.
// Useful for bootstrapping or when speed is critical.
func (mu *MemoryUpdater) QuickAdd(ctx context.Context, text string, category FactCategory, importance float64, sourceTurnIDs []string) (int64, error) {
	if mu == nil || mu.store == nil {
		return 0, errors.New("memory updater not initialized")
	}

	fact := Fact{
		Text:          text,
		Category:      category,
		Importance:    clampFloat(importance, 0.0, 1.0),
		CreatedAt:     time.Now(),
		SourceTurnIDs: strings.Join(sourceTurnIDs, ","),
	}

	// Generate embedding
	if mu.embedder != nil {
		emb, err := mu.embedder.Embed(ctx, text)
		if err == nil {
			fact.Embedding = emb
		}
	}

	return mu.store.InsertFact(ctx, fact)
}

// BulkAdd adds multiple facts without comparison (for bootstrapping).
func (mu *MemoryUpdater) BulkAdd(ctx context.Context, facts []Fact) ([]int64, error) {
	if mu == nil || mu.store == nil {
		return nil, errors.New("memory updater not initialized")
	}

	ids := make([]int64, 0, len(facts))
	for _, fact := range facts {
		// Generate embedding if missing
		if len(fact.Embedding) == 0 && mu.embedder != nil {
			emb, err := mu.embedder.Embed(ctx, fact.Text)
			if err == nil {
				fact.Embedding = emb
			}
		}

		id, err := mu.store.InsertFact(ctx, fact)
		if err != nil {
			log.Printf("warning: failed to add fact: %v", err)
			continue
		}
		ids = append(ids, id)
	}

	return ids, nil
}

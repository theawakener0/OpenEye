package omem

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// EntityGraphLite provides a lightweight entity-relationship graph (HippoRAG-inspired).
// Key design decisions for SLM optimization:
// - Regex + heuristics for entity extraction (no LLM required)
// - 1-hop traversal only (vs multi-hop PageRank)
// - Simple adjacency scoring (vs expensive graph algorithms)
// - Works with existing omem_entities and omem_relations tables
type EntityGraphLite struct {
	db     *sql.DB
	config EntityGraphConfig
	mu     sync.RWMutex

	// Prepared statements
	insertEntityStmt     *sql.Stmt
	updateEntityStmt     *sql.Stmt
	findEntityStmt       *sql.Stmt
	insertRelationStmt   *sql.Stmt
	findRelationStmt     *sql.Stmt
	getNeighborsStmt     *sql.Stmt
	getEntityFactsStmt   *sql.Stmt
	linkFactToEntityStmt *sql.Stmt
}

// Entity represents a node in the lightweight knowledge graph.
type Entity struct {
	ID             int64      `json:"id"`
	Name           string     `json:"name"`
	NormalizedName string     `json:"normalized_name"`
	Type           EntityType `json:"type"`
	Embedding      []float32  `json:"-"`
	CreatedAt      time.Time  `json:"created_at"`
	FactIDs        []int64    `json:"fact_ids,omitempty"`
	MentionCount   int        `json:"mention_count"`
}

// Relation represents a directed edge in the lightweight knowledge graph.
type Relation struct {
	ID             int64     `json:"id"`
	SourceEntityID int64     `json:"source_entity_id"`
	TargetEntityID int64     `json:"target_entity_id"`
	RelationType   string    `json:"relation_type"`
	FactID         int64     `json:"fact_id,omitempty"`
	Confidence     float64   `json:"confidence"`
	CreatedAt      time.Time `json:"created_at"`
	IsObsolete     bool      `json:"is_obsolete"`

	// Populated on retrieval
	SourceEntity *Entity `json:"source_entity,omitempty"`
	TargetEntity *Entity `json:"target_entity,omitempty"`
}

// NeighborResult represents an entity connected to the query entity.
type NeighborResult struct {
	Entity       Entity
	Relation     Relation
	Direction    string  // "outgoing" or "incoming"
	RelatedFacts []int64 // Fact IDs associated with this neighbor
}

// GraphScoreResult contains scoring information for retrieval boosting.
type GraphScoreResult struct {
	FactID        int64
	GraphScore    float64 // Combined graph-based score
	EntityBoost   float64 // Boost from entity mentions
	RelationBoost float64 // Boost from relationship matches
}

// NewEntityGraphLite creates a new lightweight entity graph using the existing database.
func NewEntityGraphLite(db *sql.DB, cfg EntityGraphConfig) (*EntityGraphLite, error) {
	if db == nil {
		return nil, errors.New("database connection required")
	}

	cfg = applyEntityGraphDefaults(cfg)

	graph := &EntityGraphLite{
		db:     db,
		config: cfg,
	}

	if err := graph.prepareStatements(); err != nil {
		return nil, err
	}

	return graph, nil
}

// applyEntityGraphDefaults fills in missing configuration values.
func applyEntityGraphDefaults(cfg EntityGraphConfig) EntityGraphConfig {
	if cfg.MaxHops <= 0 {
		cfg.MaxHops = 1 // Keep it simple for SLMs
	}
	if cfg.MaxHops > 2 {
		cfg.MaxHops = 2 // Safety limit
	}
	if cfg.SimilarityThreshold <= 0 {
		cfg.SimilarityThreshold = 0.85
	}
	if cfg.GraphBoostWeight <= 0 {
		cfg.GraphBoostWeight = 0.2
	}
	return cfg
}

// prepareStatements pre-compiles frequently used SQL statements.
func (g *EntityGraphLite) prepareStatements() error {
	var err error

	g.insertEntityStmt, err = g.db.Prepare(`
		INSERT INTO omem_entities (id, name, normalized_name, entity_type, embedding, created_at, fact_ids, mention_count)
		VALUES (nextval('omem_entities_id_seq'), ?, ?, ?, ?, ?, ?, 1)
		RETURNING id
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare insert entity statement: %w", err)
	}

	g.updateEntityStmt, err = g.db.Prepare(`
		UPDATE omem_entities 
		SET mention_count = mention_count + 1, fact_ids = ?
		WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare update entity statement: %w", err)
	}

	g.findEntityStmt, err = g.db.Prepare(`
		SELECT id, name, normalized_name, entity_type, embedding, created_at, fact_ids, mention_count
		FROM omem_entities
		WHERE normalized_name = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare find entity statement: %w", err)
	}

	g.insertRelationStmt, err = g.db.Prepare(`
		INSERT INTO omem_relations (id, source_entity_id, target_entity_id, relation_type, fact_id, confidence, created_at)
		VALUES (nextval('omem_relations_id_seq'), ?, ?, ?, ?, ?, ?)
		RETURNING id
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare insert relation statement: %w", err)
	}

	g.findRelationStmt, err = g.db.Prepare(`
		SELECT id, source_entity_id, target_entity_id, relation_type, fact_id, confidence, created_at, is_obsolete
		FROM omem_relations
		WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ? AND is_obsolete = FALSE
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare find relation statement: %w", err)
	}

	// Get 1-hop neighbors (both directions)
	g.getNeighborsStmt, err = g.db.Prepare(`
		SELECT e.id, e.name, e.normalized_name, e.entity_type, e.embedding, e.created_at, e.fact_ids, e.mention_count,
		       r.id, r.source_entity_id, r.target_entity_id, r.relation_type, r.fact_id, r.confidence, r.created_at,
		       CASE WHEN r.source_entity_id = ? THEN 'outgoing' ELSE 'incoming' END as direction
		FROM omem_relations r
		JOIN omem_entities e ON (
			(r.source_entity_id = ? AND e.id = r.target_entity_id) OR
			(r.target_entity_id = ? AND e.id = r.source_entity_id)
		)
		WHERE (r.source_entity_id = ? OR r.target_entity_id = ?) AND r.is_obsolete = FALSE
		ORDER BY r.confidence DESC
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare get neighbors statement: %w", err)
	}

	g.getEntityFactsStmt, err = g.db.Prepare(`
		SELECT fact_ids FROM omem_entities WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare get entity facts statement: %w", err)
	}

	return nil
}

// ============================================================================
// Entity Operations
// ============================================================================

// UpsertEntity creates a new entity or updates an existing one.
func (g *EntityGraphLite) UpsertEntity(ctx context.Context, entity ExtractedEntity, factID int64) (int64, error) {
	if g == nil || g.db == nil {
		return 0, errors.New("entity graph not initialized")
	}

	entity.Name = strings.TrimSpace(entity.Name)
	if entity.Name == "" {
		return 0, errors.New("entity name cannot be empty")
	}

	normalized := normalizeEntityName(entity.Name)

	g.mu.Lock()
	defer g.mu.Unlock()

	// Check for existing entity
	existing, err := g.findEntityLocked(ctx, normalized)
	if err == nil && existing != nil {
		// Update existing entity
		if err := g.addFactToEntityLocked(ctx, existing.ID, factID); err != nil {
			return existing.ID, err // Return ID even if fact link fails
		}
		return existing.ID, nil
	}

	// Create new entity
	now := time.Now()
	factIDsJSON := "[]"
	if factID > 0 {
		factIDsJSON = fmt.Sprintf("[%d]", factID)
	}

	var id int64
	err = g.insertEntityStmt.QueryRowContext(ctx,
		entity.Name,
		normalized,
		string(entity.EntityType),
		nil, // No embedding for now
		now,
		factIDsJSON,
	).Scan(&id)
	if err != nil {
		// Handle unique constraint (race condition)
		if strings.Contains(err.Error(), "UNIQUE") || strings.Contains(err.Error(), "unique") {
			existing, _ := g.findEntityLocked(ctx, normalized)
			if existing != nil {
				return existing.ID, nil
			}
		}
		return 0, fmt.Errorf("failed to insert entity: %w", err)
	}

	return id, nil
}

// findEntityLocked finds an entity by normalized name (caller must hold lock).
func (g *EntityGraphLite) findEntityLocked(ctx context.Context, normalizedName string) (*Entity, error) {
	row := g.findEntityStmt.QueryRowContext(ctx, normalizedName)
	return g.scanEntity(row)
}

// FindEntity finds an entity by name.
func (g *EntityGraphLite) FindEntity(ctx context.Context, name string) (*Entity, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	normalized := normalizeEntityName(name)

	g.mu.RLock()
	defer g.mu.RUnlock()

	row := g.findEntityStmt.QueryRowContext(ctx, normalized)
	return g.scanEntity(row)
}

// FindEntitiesByNames finds multiple entities by their names.
func (g *EntityGraphLite) FindEntitiesByNames(ctx context.Context, names []string) ([]Entity, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	if len(names) == 0 {
		return nil, nil
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	// Build query with normalized names
	normalized := make([]string, len(names))
	for i, name := range names {
		normalized[i] = normalizeEntityName(name)
	}

	placeholders := strings.Repeat("?,", len(normalized))
	placeholders = placeholders[:len(placeholders)-1]

	query := fmt.Sprintf(`
		SELECT id, name, normalized_name, entity_type, embedding, created_at, fact_ids, mention_count
		FROM omem_entities
		WHERE normalized_name IN (%s)
	`, placeholders)

	args := make([]interface{}, len(normalized))
	for i, n := range normalized {
		args[i] = n
	}

	rows, err := g.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query entities: %w", err)
	}
	defer rows.Close()

	var entities []Entity
	for rows.Next() {
		entity, err := g.scanEntityFromRows(rows)
		if err != nil {
			continue
		}
		entities = append(entities, *entity)
	}

	return entities, nil
}

// GetEntityByID retrieves an entity by its ID.
func (g *EntityGraphLite) GetEntityByID(ctx context.Context, id int64) (*Entity, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	row := g.db.QueryRowContext(ctx, `
		SELECT id, name, normalized_name, entity_type, embedding, created_at, fact_ids, mention_count
		FROM omem_entities WHERE id = ?
	`, id)

	return g.scanEntity(row)
}

// addFactToEntityLocked adds a fact ID to an entity's fact list.
func (g *EntityGraphLite) addFactToEntityLocked(ctx context.Context, entityID, factID int64) error {
	if factID <= 0 {
		return nil
	}

	// Get current fact IDs
	var factIDsJSON sql.NullString
	err := g.getEntityFactsStmt.QueryRowContext(ctx, entityID).Scan(&factIDsJSON)
	if err != nil {
		return err
	}

	var factIDs []int64
	if factIDsJSON.Valid && factIDsJSON.String != "" {
		_ = json.Unmarshal([]byte(factIDsJSON.String), &factIDs)
	}

	// Check if already linked
	for _, fid := range factIDs {
		if fid == factID {
			// Already linked, just increment mention count
			_, err := g.updateEntityStmt.ExecContext(ctx, factIDsJSON.String, entityID)
			return err
		}
	}

	// Add new fact ID
	factIDs = append(factIDs, factID)
	newJSON, _ := json.Marshal(factIDs)

	_, err = g.updateEntityStmt.ExecContext(ctx, string(newJSON), entityID)
	return err
}

// scanEntity scans an entity from a row.
func (g *EntityGraphLite) scanEntity(row *sql.Row) (*Entity, error) {
	var entity Entity
	var embeddingBlob []byte
	var entType string
	var factIDsJSON sql.NullString

	err := row.Scan(
		&entity.ID,
		&entity.Name,
		&entity.NormalizedName,
		&entType,
		&embeddingBlob,
		&entity.CreatedAt,
		&factIDsJSON,
		&entity.MentionCount,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}

	entity.Type = EntityType(entType)
	if len(embeddingBlob) > 0 {
		entity.Embedding = bytesToFloat32Slice(embeddingBlob)
	}
	if factIDsJSON.Valid && factIDsJSON.String != "" {
		_ = json.Unmarshal([]byte(factIDsJSON.String), &entity.FactIDs)
	}

	return &entity, nil
}

// scanEntityFromRows scans an entity from rows iterator.
func (g *EntityGraphLite) scanEntityFromRows(rows *sql.Rows) (*Entity, error) {
	var entity Entity
	var embeddingBlob []byte
	var entType string
	var factIDsJSON sql.NullString

	err := rows.Scan(
		&entity.ID,
		&entity.Name,
		&entity.NormalizedName,
		&entType,
		&embeddingBlob,
		&entity.CreatedAt,
		&factIDsJSON,
		&entity.MentionCount,
	)
	if err != nil {
		return nil, err
	}

	entity.Type = EntityType(entType)
	if len(embeddingBlob) > 0 {
		entity.Embedding = bytesToFloat32Slice(embeddingBlob)
	}
	if factIDsJSON.Valid && factIDsJSON.String != "" {
		_ = json.Unmarshal([]byte(factIDsJSON.String), &entity.FactIDs)
	}

	return &entity, nil
}

// ============================================================================
// Relation Operations
// ============================================================================

// AddRelation creates a new relationship between entities.
func (g *EntityGraphLite) AddRelation(ctx context.Context, rel ExtractedRelationship, factID int64) (int64, error) {
	if g == nil || g.db == nil {
		return 0, errors.New("entity graph not initialized")
	}

	if rel.SourceName == "" || rel.TargetName == "" {
		return 0, errors.New("source and target entities required")
	}
	if rel.RelationType == "" {
		rel.RelationType = "related_to"
	}

	// First ensure both entities exist
	sourceID, err := g.UpsertEntity(ctx, ExtractedEntity{Name: rel.SourceName, EntityType: EntityOther}, factID)
	if err != nil {
		return 0, fmt.Errorf("failed to upsert source entity: %w", err)
	}

	targetID, err := g.UpsertEntity(ctx, ExtractedEntity{Name: rel.TargetName, EntityType: EntityOther}, factID)
	if err != nil {
		return 0, fmt.Errorf("failed to upsert target entity: %w", err)
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	// Check for existing relation
	var existingID int64
	err = g.findRelationStmt.QueryRowContext(ctx, sourceID, targetID, rel.RelationType).Scan(
		&existingID, new(int64), new(int64), new(string), new(sql.NullInt64), new(float64), new(time.Time), new(bool),
	)
	if err == nil {
		// Relation exists, update confidence if higher
		if rel.Confidence > 0 {
			_, _ = g.db.ExecContext(ctx, `
				UPDATE omem_relations 
				SET confidence = CASE WHEN confidence < ? THEN ? ELSE confidence END
				WHERE id = ?
			`, rel.Confidence, rel.Confidence, existingID)
		}
		return existingID, nil
	}

	// Create new relation
	now := time.Now()
	confidence := rel.Confidence
	if confidence <= 0 {
		confidence = 0.5
	}

	var id int64
	err = g.insertRelationStmt.QueryRowContext(ctx,
		sourceID,
		targetID,
		rel.RelationType,
		factID,
		confidence,
		now,
	).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to insert relation: %w", err)
	}

	return id, nil
}

// GetNeighbors returns 1-hop neighbors of an entity.
func (g *EntityGraphLite) GetNeighbors(ctx context.Context, entityID int64) ([]NeighborResult, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	rows, err := g.getNeighborsStmt.QueryContext(ctx, entityID, entityID, entityID, entityID, entityID)
	if err != nil {
		return nil, fmt.Errorf("failed to query neighbors: %w", err)
	}
	defer rows.Close()

	var results []NeighborResult
	for rows.Next() {
		var entity Entity
		var rel Relation
		var direction string
		var embeddingBlob []byte
		var entType string
		var factIDsJSON sql.NullString
		var relFactID sql.NullInt64

		err := rows.Scan(
			&entity.ID, &entity.Name, &entity.NormalizedName, &entType,
			&embeddingBlob, &entity.CreatedAt, &factIDsJSON, &entity.MentionCount,
			&rel.ID, &rel.SourceEntityID, &rel.TargetEntityID, &rel.RelationType,
			&relFactID, &rel.Confidence, &rel.CreatedAt,
			&direction,
		)
		if err != nil {
			continue
		}

		entity.Type = EntityType(entType)
		if len(embeddingBlob) > 0 {
			entity.Embedding = bytesToFloat32Slice(embeddingBlob)
		}
		if factIDsJSON.Valid && factIDsJSON.String != "" {
			_ = json.Unmarshal([]byte(factIDsJSON.String), &entity.FactIDs)
		}
		if relFactID.Valid {
			rel.FactID = relFactID.Int64
		}

		results = append(results, NeighborResult{
			Entity:       entity,
			Relation:     rel,
			Direction:    direction,
			RelatedFacts: entity.FactIDs,
		})
	}

	return results, nil
}

// ============================================================================
// Entity Extraction (Regex + Heuristics)
// ============================================================================

// Common relation patterns for extraction
var (
	// "X works at Y", "X is employed by Y"
	patternWorksAt = regexp.MustCompile(`(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:works at|works for|is employed by|joined)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)`)
	// "X lives in Y", "X is from Y"
	patternLivesIn = regexp.MustCompile(`(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:lives in|is from|moved to|resides in)\s+([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)`)
	// "X knows Y", "X met Y"
	patternKnows = regexp.MustCompile(`(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:knows|met|befriended)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)`)
	// "X is Y's [relation]" (e.g., "John is Mary's husband")
	patternIsRelation = regexp.MustCompile(`(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\s+(wife|husband|brother|sister|mother|father|friend|colleague|boss|manager)`)
	// "X uses Y", "X prefers Y"
	patternUses = regexp.MustCompile(`(?i)(?:user|[A-Z][a-z]+)\s+(?:uses|prefers|likes|loves|enjoys|owns)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)`)
	// "X is a member of Y"
	patternMemberOf = regexp.MustCompile(`(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+(?:a\s+)?(?:member|part)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)`)
)

// ExtractEntitiesFromFact extracts entities from a fact using regex + heuristics.
// This is the lightweight alternative to LLM-based extraction.
func (g *EntityGraphLite) ExtractEntitiesFromFact(fact string) []ExtractedEntity {
	entities := make(map[string]ExtractedEntity)

	// Use patterns from atomic_encoder.go
	// Capitalized word sequences (potential proper nouns)
	capitalizedPattern := regexp.MustCompile(`\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b`)
	matches := capitalizedPattern.FindAllString(fact, -1)
	for _, match := range matches {
		normalized := normalizeEntityName(match)
		if len(normalized) > 2 && !isStopEntity(match) {
			entType := inferEntityType(match)
			entities[normalized] = ExtractedEntity{
				Name:       match,
				EntityType: entType,
			}
		}
	}

	// Filter out common false positives
	filtered := make([]ExtractedEntity, 0, len(entities))
	for _, e := range entities {
		filtered = append(filtered, e)
	}

	return filtered
}

// ExtractRelationsFromFact extracts relationships from a fact using regex patterns.
func (g *EntityGraphLite) ExtractRelationsFromFact(fact string) []ExtractedRelationship {
	var relations []ExtractedRelationship

	// works_at pattern
	if matches := patternWorksAt.FindAllStringSubmatch(fact, -1); matches != nil {
		for _, match := range matches {
			if len(match) >= 3 {
				relations = append(relations, ExtractedRelationship{
					SourceName:   match[1],
					TargetName:   match[2],
					RelationType: "works_at",
					Confidence:   0.8,
				})
			}
		}
	}

	// lives_in pattern
	if matches := patternLivesIn.FindAllStringSubmatch(fact, -1); matches != nil {
		for _, match := range matches {
			if len(match) >= 3 {
				relations = append(relations, ExtractedRelationship{
					SourceName:   match[1],
					TargetName:   match[2],
					RelationType: "lives_in",
					Confidence:   0.8,
				})
			}
		}
	}

	// knows pattern
	if matches := patternKnows.FindAllStringSubmatch(fact, -1); matches != nil {
		for _, match := range matches {
			if len(match) >= 3 {
				relations = append(relations, ExtractedRelationship{
					SourceName:   match[1],
					TargetName:   match[2],
					RelationType: "knows",
					Confidence:   0.7,
				})
			}
		}
	}

	// is_relation pattern (e.g., "John is Mary's husband")
	if matches := patternIsRelation.FindAllStringSubmatch(fact, -1); matches != nil {
		for _, match := range matches {
			if len(match) >= 4 {
				relations = append(relations, ExtractedRelationship{
					SourceName:   match[1],
					TargetName:   match[2],
					RelationType: match[3], // The relation type (husband, wife, etc.)
					Confidence:   0.9,
				})
			}
		}
	}

	// member_of pattern
	if matches := patternMemberOf.FindAllStringSubmatch(fact, -1); matches != nil {
		for _, match := range matches {
			if len(match) >= 3 {
				relations = append(relations, ExtractedRelationship{
					SourceName:   match[1],
					TargetName:   match[2],
					RelationType: "member_of",
					Confidence:   0.75,
				})
			}
		}
	}

	return relations
}

// ============================================================================
// Graph Scoring (Simple Adjacency-Based)
// ============================================================================

// ScoreByGraph calculates graph-based relevance scores for facts.
// Uses simple adjacency scoring (no PageRank) for SLM efficiency.
func (g *EntityGraphLite) ScoreByGraph(ctx context.Context, queryEntities []string, candidateFactIDs []int64) (map[int64]GraphScoreResult, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	if len(queryEntities) == 0 || len(candidateFactIDs) == 0 {
		return make(map[int64]GraphScoreResult), nil
	}

	// Find entities mentioned in query
	queryEntityObjs, err := g.FindEntitiesByNames(ctx, queryEntities)
	if err != nil {
		return nil, err
	}

	if len(queryEntityObjs) == 0 {
		return make(map[int64]GraphScoreResult), nil
	}

	// Collect all related fact IDs from query entities and their neighbors
	relatedFacts := make(map[int64]float64) // fact_id -> accumulated score

	for _, qEntity := range queryEntityObjs {
		// Direct facts from this entity
		for _, fid := range qEntity.FactIDs {
			relatedFacts[fid] += 1.0 // Direct match
		}

		// 1-hop neighbors
		if g.config.MaxHops >= 1 {
			neighbors, err := g.GetNeighbors(ctx, qEntity.ID)
			if err != nil {
				continue
			}

			for _, neighbor := range neighbors {
				// Facts from neighbor (weighted by relation confidence)
				for _, fid := range neighbor.RelatedFacts {
					relatedFacts[fid] += 0.5 * neighbor.Relation.Confidence
				}
				// Fact that established the relation
				if neighbor.Relation.FactID > 0 {
					relatedFacts[neighbor.Relation.FactID] += 0.3 * neighbor.Relation.Confidence
				}
			}
		}
	}

	// Build result for candidate facts
	results := make(map[int64]GraphScoreResult)
	for _, factID := range candidateFactIDs {
		score := relatedFacts[factID]
		if score > 0 {
			// Normalize score (cap at 1.0)
			if score > 1.0 {
				score = 1.0
			}
			results[factID] = GraphScoreResult{
				FactID:        factID,
				GraphScore:    score * g.config.GraphBoostWeight,
				EntityBoost:   score,
				RelationBoost: score * 0.5,
			}
		}
	}

	return results, nil
}

// GetFactsForEntities returns all fact IDs associated with given entity names.
func (g *EntityGraphLite) GetFactsForEntities(ctx context.Context, entityNames []string) ([]int64, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	entities, err := g.FindEntitiesByNames(ctx, entityNames)
	if err != nil {
		return nil, err
	}

	factIDSet := make(map[int64]bool)
	for _, e := range entities {
		for _, fid := range e.FactIDs {
			factIDSet[fid] = true
		}
	}

	factIDs := make([]int64, 0, len(factIDSet))
	for fid := range factIDSet {
		factIDs = append(factIDs, fid)
	}

	// Sort for deterministic results
	sort.Slice(factIDs, func(i, j int) bool { return factIDs[i] < factIDs[j] })

	return factIDs, nil
}

// ============================================================================
// Entity Resolution
// ============================================================================

// ResolveEntity attempts to find an existing entity that matches the given name.
// Uses edit distance for fuzzy matching when enabled.
func (g *EntityGraphLite) ResolveEntity(ctx context.Context, name string) (*Entity, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	// Exact match first
	entity, err := g.FindEntity(ctx, name)
	if err == nil && entity != nil {
		return entity, nil
	}

	// Fuzzy matching if enabled
	if g.config.EntityResolution {
		normalized := normalizeEntityName(name)

		// Search for similar names
		g.mu.RLock()
		defer g.mu.RUnlock()

		rows, err := g.db.QueryContext(ctx, `
			SELECT id, name, normalized_name, entity_type, embedding, created_at, fact_ids, mention_count
			FROM omem_entities
			WHERE normalized_name LIKE ?
			LIMIT 10
		`, normalized[:minInt(3, len(normalized))]+"%")
		if err != nil {
			return nil, err
		}
		defer rows.Close()

		var bestMatch *Entity
		var bestScore float64

		for rows.Next() {
			candidate, err := g.scanEntityFromRows(rows)
			if err != nil {
				continue
			}

			score := similarityScore(normalized, candidate.NormalizedName)
			if score >= g.config.SimilarityThreshold && score > bestScore {
				bestMatch = candidate
				bestScore = score
			}
		}

		if bestMatch != nil {
			return bestMatch, nil
		}
	}

	return nil, nil
}

// ============================================================================
// Statistics
// ============================================================================

// GetStats returns statistics about the entity graph.
func (g *EntityGraphLite) GetStats(ctx context.Context) (map[string]interface{}, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	stats := make(map[string]interface{})

	// Total entities
	var totalEntities int
	row := g.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_entities`)
	if err := row.Scan(&totalEntities); err == nil {
		stats["total_entities"] = totalEntities
	}

	// Entities by type
	typeRows, err := g.db.QueryContext(ctx, `
		SELECT entity_type, COUNT(*) FROM omem_entities GROUP BY entity_type
	`)
	if err == nil {
		types := make(map[string]int)
		for typeRows.Next() {
			var entType string
			var count int
			if err := typeRows.Scan(&entType, &count); err == nil {
				types[entType] = count
			}
		}
		typeRows.Close()
		stats["entities_by_type"] = types
	}

	// Total relations
	var totalRels int
	row = g.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_relations`)
	if err := row.Scan(&totalRels); err == nil {
		stats["total_relations"] = totalRels
	}

	// Active relations
	var activeRels int
	row = g.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_relations WHERE is_obsolete = FALSE`)
	if err := row.Scan(&activeRels); err == nil {
		stats["active_relations"] = activeRels
	}

	return stats, nil
}

// Close releases prepared statements.
func (g *EntityGraphLite) Close() error {
	if g == nil {
		return nil
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	var firstErr error
	stmts := []*sql.Stmt{
		g.insertEntityStmt,
		g.updateEntityStmt,
		g.findEntityStmt,
		g.insertRelationStmt,
		g.findRelationStmt,
		g.getNeighborsStmt,
		g.getEntityFactsStmt,
	}

	for _, stmt := range stmts {
		if stmt != nil {
			if err := stmt.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}

	return firstErr
}

// ============================================================================
// Helper Functions
// ============================================================================

// normalizeEntityName normalizes an entity name for matching.
func normalizeEntityName(name string) string {
	// Lowercase and trim
	normalized := strings.ToLower(strings.TrimSpace(name))
	// Remove extra whitespace
	normalized = strings.Join(strings.Fields(normalized), " ")
	return normalized
}

// isStopEntity checks if an entity name is a common false positive.
func isStopEntity(name string) bool {
	stopEntities := map[string]bool{
		"The":       true,
		"A":         true,
		"An":        true,
		"This":      true,
		"That":      true,
		"I":         true,
		"You":       true,
		"We":        true,
		"They":      true,
		"It":        true,
		"User":      true,
		"Assistant": true,
		"Today":     true,
		"Yesterday": true,
		"Tomorrow":  true,
		"Monday":    true,
		"Tuesday":   true,
		"Wednesday": true,
		"Thursday":  true,
		"Friday":    true,
		"Saturday":  true,
		"Sunday":    true,
		"January":   true,
		"February":  true,
		"March":     true,
		"April":     true,
		"May":       true,
		"June":      true,
		"July":      true,
		"August":    true,
		"September": true,
		"October":   true,
		"November":  true,
		"December":  true,
	}
	return stopEntities[name]
}

// inferEntityType infers the entity type from name patterns.
func inferEntityType(name string) EntityType {
	lower := strings.ToLower(name)

	// Organization indicators
	orgSuffixes := []string{"inc", "corp", "llc", "ltd", "company", "co", "group", "foundation"}
	for _, suffix := range orgSuffixes {
		if strings.HasSuffix(lower, suffix) || strings.HasSuffix(lower, suffix+".") {
			return EntityOrganization
		}
	}

	// Place indicators (very simple)
	placePrefixes := []string{"new ", "san ", "los ", "las ", "saint ", "st. "}
	for _, prefix := range placePrefixes {
		if strings.HasPrefix(lower, prefix) {
			return EntityPlace
		}
	}

	// Default to person for capitalized names without indicators
	words := strings.Fields(name)
	if len(words) >= 1 && len(words) <= 3 {
		// Likely a person name
		return EntityPerson
	}

	return EntityOther
}

// similarityScore calculates a simple similarity score between two strings.
// Uses Jaccard similarity on character trigrams for efficiency.
func similarityScore(a, b string) float64 {
	if a == b {
		return 1.0
	}
	if len(a) < 3 || len(b) < 3 {
		if a == b {
			return 1.0
		}
		return 0.0
	}

	// Generate trigrams
	trigramsA := make(map[string]bool)
	for i := 0; i <= len(a)-3; i++ {
		trigramsA[a[i:i+3]] = true
	}

	trigramsB := make(map[string]bool)
	for i := 0; i <= len(b)-3; i++ {
		trigramsB[b[i:i+3]] = true
	}

	// Calculate Jaccard similarity
	intersection := 0
	for t := range trigramsA {
		if trigramsB[t] {
			intersection++
		}
	}

	union := len(trigramsA) + len(trigramsB) - intersection
	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// Note: bytesToFloat32Slice is defined in fact_store.go and reused here

// minInt returns the minimum of two integers.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

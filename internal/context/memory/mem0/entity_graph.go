package mem0

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Entity represents a node in the knowledge graph.
type Entity struct {
	ID         int64      `json:"id"`
	Name       string     `json:"name"`
	EntityType EntityType `json:"entity_type"`
	Embedding  []float32  `json:"-"`
	CreatedAt  time.Time  `json:"created_at"`
	FactIDs    []int64    `json:"fact_ids,omitempty"`
}

// Relationship represents a directed edge in the knowledge graph.
type Relationship struct {
	ID             int64     `json:"id"`
	SourceEntityID int64     `json:"source_entity_id"`
	TargetEntityID int64     `json:"target_entity_id"`
	RelationType   string    `json:"relation_type"`
	FactID         int64     `json:"fact_id,omitempty"`
	Confidence     float64   `json:"confidence"`
	CreatedAt      time.Time `json:"created_at"`
	IsObsolete     bool      `json:"is_obsolete"`
	SourceEntity   *Entity   `json:"source_entity,omitempty"`
	TargetEntity   *Entity   `json:"target_entity,omitempty"`
}

// GraphPath represents a path through the knowledge graph.
type GraphPath struct {
	Entities      []Entity       `json:"entities"`
	Relationships []Relationship `json:"relationships"`
	TotalHops     int            `json:"total_hops"`
}

// EntityGraph provides graph-based storage for entities and relationships.
type EntityGraph struct {
	db     *sql.DB
	config GraphConfig
	mu     sync.RWMutex

	// Prepared statements
	insertEntityStmt       *sql.Stmt
	insertRelationshipStmt *sql.Stmt
	findEntityStmt         *sql.Stmt
	linkFactToEntityStmt   *sql.Stmt
}

// NewEntityGraph creates a new entity graph using the existing database connection.
func NewEntityGraph(db *sql.DB, cfg GraphConfig) (*EntityGraph, error) {
	if db == nil {
		return nil, errors.New("database connection required")
	}

	cfg = applyGraphDefaults(cfg)

	graph := &EntityGraph{
		db:     db,
		config: cfg,
	}

	if err := graph.bootstrap(); err != nil {
		return nil, err
	}

	if err := graph.prepareStatements(); err != nil {
		return nil, err
	}

	return graph, nil
}

// applyGraphDefaults fills in missing configuration values.
func applyGraphDefaults(cfg GraphConfig) GraphConfig {
	if cfg.MaxHops <= 0 {
		cfg.MaxHops = 2
	}
	if cfg.EntitySimilarityThreshold <= 0 {
		cfg.EntitySimilarityThreshold = 0.9
	}
	return cfg
}

// bootstrap creates the required tables for the entity graph.
func (g *EntityGraph) bootstrap() error {
	// Create entities table
	if _, err := g.db.Exec(`
		CREATE TABLE IF NOT EXISTS entities (
			id INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			entity_type TEXT DEFAULT 'other',
			embedding BLOB,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(name, entity_type)
		)
	`); err != nil {
		return fmt.Errorf("failed to create entities table: %w", err)
	}

	// Create relationships table (directed edges)
	if _, err := g.db.Exec(`
		CREATE TABLE IF NOT EXISTS relationships (
			id INTEGER PRIMARY KEY,
			source_entity_id INTEGER NOT NULL,
			target_entity_id INTEGER NOT NULL,
			relation_type TEXT NOT NULL,
			fact_id INTEGER,
			confidence DOUBLE DEFAULT 1.0,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			is_obsolete BOOLEAN DEFAULT FALSE,
			FOREIGN KEY (source_entity_id) REFERENCES entities(id),
			FOREIGN KEY (target_entity_id) REFERENCES entities(id),
			FOREIGN KEY (fact_id) REFERENCES facts(id)
		)
	`); err != nil {
		return fmt.Errorf("failed to create relationships table: %w", err)
	}

	// Create entity-fact linking table
	if _, err := g.db.Exec(`
		CREATE TABLE IF NOT EXISTS entity_facts (
			entity_id INTEGER NOT NULL,
			fact_id INTEGER NOT NULL,
			PRIMARY KEY (entity_id, fact_id),
			FOREIGN KEY (entity_id) REFERENCES entities(id),
			FOREIGN KEY (fact_id) REFERENCES facts(id)
		)
	`); err != nil {
		return fmt.Errorf("failed to create entity_facts table: %w", err)
	}

	// Create indexes
	indexes := []string{
		`CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type)`,
		`CREATE INDEX IF NOT EXISTS idx_relationships_fact ON relationships(fact_id)`,
		`CREATE INDEX IF NOT EXISTS idx_entity_facts_entity ON entity_facts(entity_id)`,
		`CREATE INDEX IF NOT EXISTS idx_entity_facts_fact ON entity_facts(fact_id)`,
	}

	for _, idx := range indexes {
		if _, err := g.db.Exec(idx); err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}
	}

	// Create sequences
	if _, err := g.db.Exec(`CREATE SEQUENCE IF NOT EXISTS entities_id_seq START 1`); err != nil {
		return fmt.Errorf("failed to create entities sequence: %w", err)
	}
	if _, err := g.db.Exec(`CREATE SEQUENCE IF NOT EXISTS relationships_id_seq START 1`); err != nil {
		return fmt.Errorf("failed to create relationships sequence: %w", err)
	}

	return nil
}

// prepareStatements pre-compiles frequently used SQL statements.
func (g *EntityGraph) prepareStatements() error {
	var err error

	g.insertEntityStmt, err = g.db.Prepare(`
		INSERT INTO entities (id, name, entity_type, embedding, created_at)
		VALUES (nextval('entities_id_seq'), ?, ?, ?, ?)
		RETURNING id
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare insert entity statement: %w", err)
	}

	g.insertRelationshipStmt, err = g.db.Prepare(`
		INSERT INTO relationships (id, source_entity_id, target_entity_id, relation_type, fact_id, confidence, created_at)
		VALUES (nextval('relationships_id_seq'), ?, ?, ?, ?, ?, ?)
		RETURNING id
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare insert relationship statement: %w", err)
	}

	g.findEntityStmt, err = g.db.Prepare(`
		SELECT id, name, entity_type, embedding, created_at
		FROM entities
		WHERE LOWER(name) = LOWER(?) AND entity_type = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare find entity statement: %w", err)
	}

	g.linkFactToEntityStmt, err = g.db.Prepare(`
		INSERT INTO entity_facts (entity_id, fact_id)
		VALUES (?, ?)
		ON CONFLICT (entity_id, fact_id) DO NOTHING
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare link fact statement: %w", err)
	}

	return nil
}

// UpsertEntity creates or retrieves an existing entity.
func (g *EntityGraph) UpsertEntity(ctx context.Context, entity Entity) (int64, error) {
	if g == nil || g.db == nil {
		return 0, errors.New("entity graph not initialized")
	}

	entity.Name = strings.TrimSpace(entity.Name)
	if entity.Name == "" {
		return 0, errors.New("entity name cannot be empty")
	}
	if entity.EntityType == "" {
		entity.EntityType = EntityOther
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	// Try to find existing entity
	existing, err := g.findEntityLocked(ctx, entity.Name, entity.EntityType)
	if err == nil && existing != nil {
		return existing.ID, nil
	}

	// Create new entity
	if entity.CreatedAt.IsZero() {
		entity.CreatedAt = time.Now()
	}

	var embeddingBlob []byte
	if len(entity.Embedding) > 0 {
		embeddingBlob = float32SliceToBytes(entity.Embedding)
	}

	var id int64
	err = g.insertEntityStmt.QueryRowContext(ctx,
		entity.Name,
		string(entity.EntityType),
		embeddingBlob,
		entity.CreatedAt,
	).Scan(&id)
	if err != nil {
		// Check if it's a unique constraint violation (race condition)
		if strings.Contains(err.Error(), "UNIQUE") || strings.Contains(err.Error(), "unique") {
			existing, findErr := g.findEntityLocked(ctx, entity.Name, entity.EntityType)
			if findErr == nil && existing != nil {
				return existing.ID, nil
			}
		}
		return 0, fmt.Errorf("failed to insert entity: %w", err)
	}

	return id, nil
}

// findEntityLocked finds an entity by name and type (caller must hold lock).
func (g *EntityGraph) findEntityLocked(ctx context.Context, name string, entityType EntityType) (*Entity, error) {
	row := g.findEntityStmt.QueryRowContext(ctx, name, string(entityType))

	var entity Entity
	var embeddingBlob []byte
	var entType string
	var createdAt time.Time

	err := row.Scan(&entity.ID, &entity.Name, &entType, &embeddingBlob, &createdAt)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}

	entity.EntityType = EntityType(entType)
	entity.CreatedAt = createdAt
	if len(embeddingBlob) > 0 {
		entity.Embedding = bytesToFloat32Slice(embeddingBlob)
	}

	return &entity, nil
}

// FindEntity finds an entity by name and type.
func (g *EntityGraph) FindEntity(ctx context.Context, name string, entityType EntityType) (*Entity, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	return g.findEntityLocked(ctx, name, entityType)
}

// FindEntityByName finds an entity by name (any type).
func (g *EntityGraph) FindEntityByName(ctx context.Context, name string) (*Entity, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	row := g.db.QueryRowContext(ctx, `
		SELECT id, name, entity_type, embedding, created_at
		FROM entities
		WHERE LOWER(name) = LOWER(?)
		LIMIT 1
	`, name)

	var entity Entity
	var embeddingBlob []byte
	var entType string
	var createdAt time.Time

	err := row.Scan(&entity.ID, &entity.Name, &entType, &embeddingBlob, &createdAt)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}

	entity.EntityType = EntityType(entType)
	entity.CreatedAt = createdAt
	if len(embeddingBlob) > 0 {
		entity.Embedding = bytesToFloat32Slice(embeddingBlob)
	}

	return &entity, nil
}

// GetEntityByID retrieves an entity by ID.
func (g *EntityGraph) GetEntityByID(ctx context.Context, id int64) (*Entity, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	row := g.db.QueryRowContext(ctx, `
		SELECT id, name, entity_type, embedding, created_at
		FROM entities
		WHERE id = ?
	`, id)

	var entity Entity
	var embeddingBlob []byte
	var entType string
	var createdAt time.Time

	err := row.Scan(&entity.ID, &entity.Name, &entType, &embeddingBlob, &createdAt)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}

	entity.EntityType = EntityType(entType)
	entity.CreatedAt = createdAt
	if len(embeddingBlob) > 0 {
		entity.Embedding = bytesToFloat32Slice(embeddingBlob)
	}

	return &entity, nil
}

// CreateRelationship creates a new relationship between entities.
func (g *EntityGraph) CreateRelationship(ctx context.Context, rel Relationship) (int64, error) {
	if g == nil || g.db == nil {
		return 0, errors.New("entity graph not initialized")
	}

	if rel.SourceEntityID == 0 || rel.TargetEntityID == 0 {
		return 0, errors.New("source and target entity IDs required")
	}
	if rel.RelationType == "" {
		return 0, errors.New("relation type required")
	}
	if rel.Confidence <= 0 {
		rel.Confidence = 1.0
	}
	if rel.CreatedAt.IsZero() {
		rel.CreatedAt = time.Now()
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	// Check for existing similar relationship
	var existingID int64
	err := g.db.QueryRowContext(ctx, `
		SELECT id FROM relationships
		WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ? AND is_obsolete = FALSE
		LIMIT 1
	`, rel.SourceEntityID, rel.TargetEntityID, rel.RelationType).Scan(&existingID)

	if err == nil {
		// Relationship exists, update confidence if new is higher
		_, _ = g.db.ExecContext(ctx, `
			UPDATE relationships 
			SET confidence = CASE WHEN confidence < ? THEN ? ELSE confidence END,
			    fact_id = COALESCE(?, fact_id)
			WHERE id = ?
		`, rel.Confidence, rel.Confidence, rel.FactID, existingID)
		return existingID, nil
	}

	var factID interface{}
	if rel.FactID > 0 {
		factID = rel.FactID
	}

	var id int64
	err = g.insertRelationshipStmt.QueryRowContext(ctx,
		rel.SourceEntityID,
		rel.TargetEntityID,
		rel.RelationType,
		factID,
		rel.Confidence,
		rel.CreatedAt,
	).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to insert relationship: %w", err)
	}

	return id, nil
}

// LinkFactToEntity associates a fact with an entity.
func (g *EntityGraph) LinkFactToEntity(ctx context.Context, entityID, factID int64) error {
	if g == nil || g.db == nil {
		return errors.New("entity graph not initialized")
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	_, err := g.linkFactToEntityStmt.ExecContext(ctx, entityID, factID)
	if err != nil {
		return fmt.Errorf("failed to link fact to entity: %w", err)
	}

	return nil
}

// GetRelationshipsFrom gets all relationships originating from an entity.
func (g *EntityGraph) GetRelationshipsFrom(ctx context.Context, entityID int64, includeObsolete bool) ([]Relationship, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	query := `
		SELECT r.id, r.source_entity_id, r.target_entity_id, r.relation_type, 
		       r.fact_id, r.confidence, r.created_at, r.is_obsolete,
		       e.id, e.name, e.entity_type, e.created_at
		FROM relationships r
		JOIN entities e ON r.target_entity_id = e.id
		WHERE r.source_entity_id = ?
	`
	if !includeObsolete {
		query += ` AND r.is_obsolete = FALSE`
	}
	query += ` ORDER BY r.confidence DESC, r.created_at DESC`

	rows, err := g.db.QueryContext(ctx, query, entityID)
	if err != nil {
		return nil, fmt.Errorf("failed to query relationships: %w", err)
	}
	defer rows.Close()

	var relationships []Relationship
	for rows.Next() {
		var rel Relationship
		var factID sql.NullInt64
		var createdAt time.Time
		var targetEntity Entity
		var targetType string
		var targetCreatedAt time.Time

		err := rows.Scan(
			&rel.ID, &rel.SourceEntityID, &rel.TargetEntityID, &rel.RelationType,
			&factID, &rel.Confidence, &createdAt, &rel.IsObsolete,
			&targetEntity.ID, &targetEntity.Name, &targetType, &targetCreatedAt,
		)
		if err != nil {
			continue
		}

		rel.CreatedAt = createdAt
		if factID.Valid {
			rel.FactID = factID.Int64
		}
		targetEntity.EntityType = EntityType(targetType)
		targetEntity.CreatedAt = targetCreatedAt
		rel.TargetEntity = &targetEntity

		relationships = append(relationships, rel)
	}

	return relationships, nil
}

// GetRelationshipsTo gets all relationships pointing to an entity.
func (g *EntityGraph) GetRelationshipsTo(ctx context.Context, entityID int64, includeObsolete bool) ([]Relationship, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	query := `
		SELECT r.id, r.source_entity_id, r.target_entity_id, r.relation_type, 
		       r.fact_id, r.confidence, r.created_at, r.is_obsolete,
		       e.id, e.name, e.entity_type, e.created_at
		FROM relationships r
		JOIN entities e ON r.source_entity_id = e.id
		WHERE r.target_entity_id = ?
	`
	if !includeObsolete {
		query += ` AND r.is_obsolete = FALSE`
	}
	query += ` ORDER BY r.confidence DESC, r.created_at DESC`

	rows, err := g.db.QueryContext(ctx, query, entityID)
	if err != nil {
		return nil, fmt.Errorf("failed to query relationships: %w", err)
	}
	defer rows.Close()

	var relationships []Relationship
	for rows.Next() {
		var rel Relationship
		var factID sql.NullInt64
		var createdAt time.Time
		var sourceEntity Entity
		var sourceType string
		var sourceCreatedAt time.Time

		err := rows.Scan(
			&rel.ID, &rel.SourceEntityID, &rel.TargetEntityID, &rel.RelationType,
			&factID, &rel.Confidence, &createdAt, &rel.IsObsolete,
			&sourceEntity.ID, &sourceEntity.Name, &sourceType, &sourceCreatedAt,
		)
		if err != nil {
			continue
		}

		rel.CreatedAt = createdAt
		if factID.Valid {
			rel.FactID = factID.Int64
		}
		sourceEntity.EntityType = EntityType(sourceType)
		sourceEntity.CreatedAt = sourceCreatedAt
		rel.SourceEntity = &sourceEntity

		relationships = append(relationships, rel)
	}

	return relationships, nil
}

// GetFactsForEntity gets all facts associated with an entity.
func (g *EntityGraph) GetFactsForEntity(ctx context.Context, entityID int64) ([]int64, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	rows, err := g.db.QueryContext(ctx, `
		SELECT fact_id FROM entity_facts WHERE entity_id = ?
	`, entityID)
	if err != nil {
		return nil, fmt.Errorf("failed to query entity facts: %w", err)
	}
	defer rows.Close()

	var factIDs []int64
	for rows.Next() {
		var factID int64
		if err := rows.Scan(&factID); err != nil {
			continue
		}
		factIDs = append(factIDs, factID)
	}

	return factIDs, nil
}

// TraverseFrom performs a breadth-first traversal from an entity up to maxHops.
func (g *EntityGraph) TraverseFrom(ctx context.Context, entityID int64, maxHops int) ([]GraphPath, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	if maxHops <= 0 {
		maxHops = g.config.MaxHops
	}
	if maxHops > 5 {
		maxHops = 5 // Safety limit
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	// Get starting entity
	startEntity, err := g.GetEntityByID(ctx, entityID)
	if err != nil || startEntity == nil {
		return nil, fmt.Errorf("entity not found: %d", entityID)
	}

	var paths []GraphPath
	visited := make(map[int64]bool)
	visited[entityID] = true

	// BFS queue: each item is (current path, current entity ID)
	type queueItem struct {
		path     GraphPath
		entityID int64
	}

	queue := []queueItem{{
		path: GraphPath{
			Entities:  []Entity{*startEntity},
			TotalHops: 0,
		},
		entityID: entityID,
	}}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if current.path.TotalHops >= maxHops {
			paths = append(paths, current.path)
			continue
		}

		// Get outgoing relationships
		rels, err := g.GetRelationshipsFrom(ctx, current.entityID, false)
		if err != nil {
			continue
		}

		if len(rels) == 0 {
			// No more edges, this is a terminal path
			paths = append(paths, current.path)
			continue
		}

		for _, rel := range rels {
			if visited[rel.TargetEntityID] {
				continue
			}

			visited[rel.TargetEntityID] = true

			newPath := GraphPath{
				Entities:      append(append([]Entity{}, current.path.Entities...), *rel.TargetEntity),
				Relationships: append(append([]Relationship{}, current.path.Relationships...), rel),
				TotalHops:     current.path.TotalHops + 1,
			}

			queue = append(queue, queueItem{
				path:     newPath,
				entityID: rel.TargetEntityID,
			})
		}
	}

	return paths, nil
}

// SearchEntitiesByName searches for entities with similar names.
func (g *EntityGraph) SearchEntitiesByName(ctx context.Context, query string, limit int) ([]Entity, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	if limit <= 0 {
		limit = 10
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	// Use LIKE for simple text matching
	rows, err := g.db.QueryContext(ctx, `
		SELECT id, name, entity_type, embedding, created_at
		FROM entities
		WHERE LOWER(name) LIKE '%' || LOWER(?) || '%'
		ORDER BY 
			CASE WHEN LOWER(name) = LOWER(?) THEN 0 ELSE 1 END,
			created_at DESC
		LIMIT ?
	`, query, query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to search entities: %w", err)
	}
	defer rows.Close()

	var entities []Entity
	for rows.Next() {
		var entity Entity
		var embeddingBlob []byte
		var entType string
		var createdAt time.Time

		if err := rows.Scan(&entity.ID, &entity.Name, &entType, &embeddingBlob, &createdAt); err != nil {
			continue
		}

		entity.EntityType = EntityType(entType)
		entity.CreatedAt = createdAt
		if len(embeddingBlob) > 0 {
			entity.Embedding = bytesToFloat32Slice(embeddingBlob)
		}

		entities = append(entities, entity)
	}

	return entities, nil
}

// MarkRelationshipObsolete marks a relationship as obsolete.
func (g *EntityGraph) MarkRelationshipObsolete(ctx context.Context, relationshipID int64) error {
	if g == nil || g.db == nil {
		return errors.New("entity graph not initialized")
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	_, err := g.db.ExecContext(ctx, `
		UPDATE relationships SET is_obsolete = TRUE WHERE id = ?
	`, relationshipID)
	if err != nil {
		return fmt.Errorf("failed to mark relationship obsolete: %w", err)
	}

	return nil
}

// GetStats returns statistics about the entity graph.
func (g *EntityGraph) GetStats(ctx context.Context) (map[string]interface{}, error) {
	if g == nil || g.db == nil {
		return nil, errors.New("entity graph not initialized")
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	stats := make(map[string]interface{})

	// Total entities
	var totalEntities int
	row := g.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM entities`)
	if err := row.Scan(&totalEntities); err == nil {
		stats["total_entities"] = totalEntities
	}

	// Entities by type
	typeRows, err := g.db.QueryContext(ctx, `
		SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type
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

	// Total relationships
	var totalRels int
	row = g.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM relationships`)
	if err := row.Scan(&totalRels); err == nil {
		stats["total_relationships"] = totalRels
	}

	// Active relationships
	var activeRels int
	row = g.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM relationships WHERE is_obsolete = FALSE`)
	if err := row.Scan(&activeRels); err == nil {
		stats["active_relationships"] = activeRels
	}

	// Relationships by type
	relRows, err := g.db.QueryContext(ctx, `
		SELECT relation_type, COUNT(*) 
		FROM relationships 
		WHERE is_obsolete = FALSE 
		GROUP BY relation_type
	`)
	if err == nil {
		relTypes := make(map[string]int)
		for relRows.Next() {
			var relType string
			var count int
			if err := relRows.Scan(&relType, &count); err == nil {
				relTypes[relType] = count
			}
		}
		relRows.Close()
		stats["relationships_by_type"] = relTypes
	}

	return stats, nil
}

// Close releases prepared statements (db is shared, not closed here).
func (g *EntityGraph) Close() error {
	if g == nil {
		return nil
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	var firstErr error
	stmts := []*sql.Stmt{
		g.insertEntityStmt,
		g.insertRelationshipStmt,
		g.findEntityStmt,
		g.linkFactToEntityStmt,
	}

	for _, stmt := range stmts {
		if stmt != nil {
			if err := stmt.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}

	g.insertEntityStmt = nil
	g.insertRelationshipStmt = nil
	g.findEntityStmt = nil
	g.linkFactToEntityStmt = nil

	return firstErr
}

package omem

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	_ "github.com/marcboeker/go-duckdb"
)

// Fact represents an atomic piece of knowledge with multi-view indexing.
type Fact struct {
	ID int64 `json:"id"`

	// Core content
	Text       string `json:"text"`        // Original text
	AtomicText string `json:"atomic_text"` // Disambiguated text (no pronouns, absolute timestamps)

	// Classification
	Category   FactCategory `json:"category"`
	Importance float64      `json:"importance"`

	// Multi-view index fields
	Embedding []float32 `json:"-"`        // Dense vector for semantic search
	Keywords  []string  `json:"keywords"` // Extracted keywords for BM25

	// Symbolic metadata
	TimestampAnchor *time.Time  `json:"timestamp_anchor,omitempty"` // Absolute time of fact
	Location        string      `json:"location,omitempty"`         // Location if mentioned
	Entities        []EntityRef `json:"entities,omitempty"`         // Entity references

	// Episode tracking
	EpisodeID int64  `json:"episode_id,omitempty"`
	TurnID    string `json:"turn_id,omitempty"`

	// Lifecycle
	CreatedAt    time.Time `json:"created_at"`
	LastAccessed time.Time `json:"last_accessed"`
	AccessCount  int       `json:"access_count"`
	IsObsolete   bool      `json:"is_obsolete"`
	SupersededBy *int64    `json:"superseded_by,omitempty"`
}

// EntityRef represents a reference to an entity in a fact.
type EntityRef struct {
	Name string     `json:"name"`
	Type EntityType `json:"type"`
}

// Episode represents a conversation session/episode.
type Episode struct {
	ID             int64      `json:"id"`
	SessionID      string     `json:"session_id"`
	StartedAt      time.Time  `json:"started_at"`
	EndedAt        *time.Time `json:"ended_at,omitempty"`
	Summary        string     `json:"summary,omitempty"`
	FactCount      int        `json:"fact_count"`
	EntityMentions []string   `json:"entity_mentions,omitempty"`
}

// RollingSummary represents the cached rolling summary of facts.
type RollingSummary struct {
	ID             int64     `json:"id"`
	Summary        string    `json:"summary"`
	Embedding      []float32 `json:"-"`
	UpdatedAt      time.Time `json:"updated_at"`
	SourceFactIDs  []int64   `json:"source_fact_ids"`
	FactCount      int       `json:"fact_count"`
	PendingFactIDs []int64   `json:"pending_fact_ids"` // Facts added since last summary
}

// ScoredFact represents a fact with retrieval scores.
type ScoredFact struct {
	Fact          Fact
	Score         float64
	SemanticScore float64
	LexicalScore  float64
	SymbolicScore float64
	GraphBoost    float64
	RecencyScore  float64
}

// FactStore provides DuckDB-backed storage with multi-view indexing.
type FactStore struct {
	db     *sql.DB
	config StorageConfig
	mu     sync.RWMutex

	// Prepared statements for performance
	insertFactStmt    *sql.Stmt
	updateFactStmt    *sql.Stmt
	markObsoleteStmt  *sql.Stmt
	updateAccessStmt  *sql.Stmt
	insertEpisodeStmt *sql.Stmt
	updateEpisodeStmt *sql.Stmt
}

// NewFactStore creates a new fact store with the given configuration.
func NewFactStore(cfg StorageConfig) (*FactStore, error) {
	cfg = applyStorageDefaults(cfg)

	// Ensure directory exists
	if dir := filepath.Dir(filepath.Clean(cfg.DBPath)); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("failed to create omem directory: %w", err)
		}
	}

	db, err := sql.Open("duckdb", cfg.DBPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB for omem: %w", err)
	}

	store := &FactStore{
		db:     db,
		config: cfg,
	}

	if err := store.bootstrap(); err != nil {
		db.Close()
		return nil, err
	}

	if err := store.prepareStatements(); err != nil {
		db.Close()
		return nil, err
	}

	return store, nil
}

// applyStorageDefaults fills in missing configuration values.
func applyStorageDefaults(cfg StorageConfig) StorageConfig {
	if cfg.DBPath == "" {
		cfg.DBPath = "openeye_omem.duckdb"
	}
	if cfg.MaxFacts <= 0 {
		cfg.MaxFacts = 10000
	}
	if cfg.PruneThreshold <= 0 {
		cfg.PruneThreshold = cfg.MaxFacts + 2000
	}
	if cfg.PruneKeepRecent <= 0 {
		cfg.PruneKeepRecent = cfg.MaxFacts / 2
	}
	return cfg
}

// bootstrap creates the required tables and indexes for multi-view storage.
func (s *FactStore) bootstrap() error {
	// Create facts table with multi-view schema
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS omem_facts (
			id INTEGER PRIMARY KEY,
			
			-- Core content
			fact_text TEXT NOT NULL,
			atomic_text TEXT NOT NULL,
			
			-- Classification
			category TEXT DEFAULT 'other',
			importance DOUBLE DEFAULT 0.5,
			
			-- Multi-view index fields
			embedding BLOB,
			keywords TEXT,
			
			-- Symbolic metadata
			timestamp_anchor TIMESTAMP,
			location TEXT,
			entities_json TEXT,
			
			-- Episode tracking
			episode_id INTEGER,
			turn_id TEXT,
			
			-- Lifecycle
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			access_count INTEGER DEFAULT 0,
			is_obsolete BOOLEAN DEFAULT FALSE,
			superseded_by INTEGER
		)
	`); err != nil {
		return fmt.Errorf("failed to create omem_facts table: %w", err)
	}

	// Create episodes table
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS omem_episodes (
			id INTEGER PRIMARY KEY,
			session_id TEXT UNIQUE,
			started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			ended_at TIMESTAMP,
			summary TEXT,
			fact_count INTEGER DEFAULT 0,
			entity_mentions TEXT
		)
	`); err != nil {
		return fmt.Errorf("failed to create omem_episodes table: %w", err)
	}

	// Create entities table for lightweight graph
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS omem_entities (
			id INTEGER PRIMARY KEY,
			name TEXT NOT NULL,
			normalized_name TEXT NOT NULL,
			entity_type TEXT DEFAULT 'other',
			embedding BLOB,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			fact_ids TEXT,
			mention_count INTEGER DEFAULT 1
		)
	`); err != nil {
		return fmt.Errorf("failed to create omem_entities table: %w", err)
	}

	// Create relations table for entity graph
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS omem_relations (
			id INTEGER PRIMARY KEY,
			source_entity_id INTEGER,
			target_entity_id INTEGER,
			relation_type TEXT,
			fact_id INTEGER,
			confidence DOUBLE DEFAULT 0.5,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			is_obsolete BOOLEAN DEFAULT FALSE
		)
	`); err != nil {
		return fmt.Errorf("failed to create omem_relations table: %w", err)
	}

	// Create rolling summary table
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS omem_rolling_summary (
			id INTEGER PRIMARY KEY DEFAULT 1,
			summary TEXT,
			embedding BLOB,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			source_fact_ids TEXT,
			fact_count INTEGER DEFAULT 0,
			pending_fact_ids TEXT
		)
	`); err != nil {
		return fmt.Errorf("failed to create omem_rolling_summary table: %w", err)
	}

	// Create indexes for efficient queries
	indexes := []string{
		// Fact indexes
		`CREATE INDEX IF NOT EXISTS idx_omem_facts_category ON omem_facts(category)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_facts_importance ON omem_facts(importance DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_facts_created ON omem_facts(created_at DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_facts_accessed ON omem_facts(last_accessed DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_facts_obsolete ON omem_facts(is_obsolete)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_facts_episode ON omem_facts(episode_id)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_facts_timestamp ON omem_facts(timestamp_anchor)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_facts_location ON omem_facts(location)`,
		// Episode indexes
		`CREATE INDEX IF NOT EXISTS idx_omem_episodes_session ON omem_episodes(session_id)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_episodes_started ON omem_episodes(started_at DESC)`,
		// Entity indexes
		`CREATE INDEX IF NOT EXISTS idx_omem_entities_name ON omem_entities(normalized_name)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_entities_type ON omem_entities(entity_type)`,
		// Relation indexes
		`CREATE INDEX IF NOT EXISTS idx_omem_relations_source ON omem_relations(source_entity_id)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_relations_target ON omem_relations(target_entity_id)`,
		`CREATE INDEX IF NOT EXISTS idx_omem_relations_fact ON omem_relations(fact_id)`,
	}

	for _, idx := range indexes {
		if _, err := s.db.Exec(idx); err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}
	}

	// Create sequences
	if _, err := s.db.Exec(`CREATE SEQUENCE IF NOT EXISTS omem_facts_id_seq START 1`); err != nil {
		return fmt.Errorf("failed to create facts sequence: %w", err)
	}
	if _, err := s.db.Exec(`CREATE SEQUENCE IF NOT EXISTS omem_episodes_id_seq START 1`); err != nil {
		return fmt.Errorf("failed to create episodes sequence: %w", err)
	}
	if _, err := s.db.Exec(`CREATE SEQUENCE IF NOT EXISTS omem_entities_id_seq START 1`); err != nil {
		return fmt.Errorf("failed to create entities sequence: %w", err)
	}
	if _, err := s.db.Exec(`CREATE SEQUENCE IF NOT EXISTS omem_relations_id_seq START 1`); err != nil {
		return fmt.Errorf("failed to create relations sequence: %w", err)
	}

	// Initialize rolling summary if not exists
	if _, err := s.db.Exec(`
		INSERT INTO omem_rolling_summary (id, summary, fact_count, pending_fact_ids) 
		SELECT 1, '', 0, ''
		WHERE NOT EXISTS (SELECT 1 FROM omem_rolling_summary WHERE id = 1)
	`); err != nil {
		_ = err // Ignore if already exists
	}

	// Enable FTS extension if configured
	if s.config.EnableFTS {
		if err := s.enableFTS(); err != nil {
			// Log warning but don't fail - FTS is optional
			fmt.Printf("warning: failed to enable FTS: %v\n", err)
		}
	}

	return nil
}

// enableFTS enables DuckDB Full-Text Search extension.
func (s *FactStore) enableFTS() error {
	// Install and load FTS extension
	if _, err := s.db.Exec(`INSTALL fts`); err != nil {
		return fmt.Errorf("failed to install FTS: %w", err)
	}
	if _, err := s.db.Exec(`LOAD fts`); err != nil {
		return fmt.Errorf("failed to load FTS: %w", err)
	}

	// Create FTS index on keywords column
	// Note: This creates a stemmed, lowercased index
	if _, err := s.db.Exec(`
		PRAGMA create_fts_index('omem_facts', 'id', 'keywords', 'atomic_text', 
			stemmer = 'english', 
			lower = true, 
			overwrite = true
		)
	`); err != nil {
		// FTS index creation might fail if already exists or keywords are empty
		// This is non-fatal
		return fmt.Errorf("failed to create FTS index: %w", err)
	}

	return nil
}

// prepareStatements pre-compiles frequently used SQL statements.
func (s *FactStore) prepareStatements() error {
	var err error

	s.insertFactStmt, err = s.db.Prepare(`
		INSERT INTO omem_facts (
			id, fact_text, atomic_text, category, importance, embedding, keywords,
			timestamp_anchor, location, entities_json, episode_id, turn_id, created_at
		)
		VALUES (nextval('omem_facts_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		RETURNING id
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare insert fact statement: %w", err)
	}

	s.updateFactStmt, err = s.db.Prepare(`
		UPDATE omem_facts 
		SET fact_text = ?, atomic_text = ?, category = ?, importance = ?, 
		    embedding = ?, keywords = ?, last_accessed = ?
		WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare update fact statement: %w", err)
	}

	s.markObsoleteStmt, err = s.db.Prepare(`
		UPDATE omem_facts 
		SET is_obsolete = TRUE, superseded_by = ?
		WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare mark obsolete statement: %w", err)
	}

	s.updateAccessStmt, err = s.db.Prepare(`
		UPDATE omem_facts 
		SET last_accessed = ?, access_count = access_count + 1
		WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare update access statement: %w", err)
	}

	s.insertEpisodeStmt, err = s.db.Prepare(`
		INSERT INTO omem_episodes (id, session_id, started_at)
		VALUES (nextval('omem_episodes_id_seq'), ?, ?)
		RETURNING id
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare insert episode statement: %w", err)
	}

	s.updateEpisodeStmt, err = s.db.Prepare(`
		UPDATE omem_episodes 
		SET ended_at = ?, summary = ?, fact_count = ?, entity_mentions = ?
		WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare update episode statement: %w", err)
	}

	return nil
}

// InsertFact stores a new fact and returns its ID.
func (s *FactStore) InsertFact(ctx context.Context, fact Fact) (int64, error) {
	if s == nil || s.db == nil {
		return 0, errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	fact.Text = strings.TrimSpace(fact.Text)
	if fact.Text == "" {
		return 0, errors.New("fact text cannot be empty")
	}

	// Default atomic text to original if not set
	if fact.AtomicText == "" {
		fact.AtomicText = fact.Text
	}

	if fact.Category == "" {
		fact.Category = CategoryOther
	}
	if fact.Importance <= 0 {
		fact.Importance = 0.5
	}
	if fact.Importance > 1.0 {
		fact.Importance = 1.0
	}
	if fact.CreatedAt.IsZero() {
		fact.CreatedAt = time.Now()
	}

	var embeddingBlob []byte
	if len(fact.Embedding) > 0 {
		embeddingBlob = float32SliceToBytes(fact.Embedding)
	}

	keywordsStr := strings.Join(fact.Keywords, " ")

	var entitiesJSON []byte
	if len(fact.Entities) > 0 {
		entitiesJSON, _ = json.Marshal(fact.Entities)
	}

	var id int64
	err := s.insertFactStmt.QueryRowContext(ctx,
		fact.Text,
		fact.AtomicText,
		string(fact.Category),
		fact.Importance,
		embeddingBlob,
		keywordsStr,
		fact.TimestampAnchor,
		fact.Location,
		string(entitiesJSON),
		fact.EpisodeID,
		fact.TurnID,
		fact.CreatedAt,
	).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to insert fact: %w", err)
	}

	return id, nil
}

// UpdateFact updates an existing fact.
func (s *FactStore) UpdateFact(ctx context.Context, fact Fact) error {
	if s == nil || s.db == nil {
		return errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var embeddingBlob []byte
	if len(fact.Embedding) > 0 {
		embeddingBlob = float32SliceToBytes(fact.Embedding)
	}

	keywordsStr := strings.Join(fact.Keywords, " ")

	_, err := s.updateFactStmt.ExecContext(ctx,
		fact.Text,
		fact.AtomicText,
		string(fact.Category),
		fact.Importance,
		embeddingBlob,
		keywordsStr,
		time.Now(),
		fact.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update fact: %w", err)
	}

	return nil
}

// MarkObsolete marks a fact as obsolete.
func (s *FactStore) MarkObsolete(ctx context.Context, factID int64, supersededBy *int64) error {
	if s == nil || s.db == nil {
		return errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var supersededByVal interface{}
	if supersededBy != nil {
		supersededByVal = *supersededBy
	}

	_, err := s.markObsoleteStmt.ExecContext(ctx, supersededByVal, factID)
	if err != nil {
		return fmt.Errorf("failed to mark fact obsolete: %w", err)
	}

	return nil
}

// GetFactByID retrieves a single fact by ID.
func (s *FactStore) GetFactByID(ctx context.Context, id int64) (*Fact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	row := s.db.QueryRowContext(ctx, `
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by
		FROM omem_facts
		WHERE id = ?
	`, id)

	return s.scanFact(row)
}

// GetFactsByIDs retrieves multiple facts by their IDs.
func (s *FactStore) GetFactsByIDs(ctx context.Context, ids []int64) ([]Fact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	if len(ids) == 0 {
		return nil, nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Build placeholders
	placeholders := make([]string, len(ids))
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		placeholders[i] = "?"
		args[i] = id
	}

	query := fmt.Sprintf(`
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by
		FROM omem_facts
		WHERE id IN (%s)
	`, strings.Join(placeholders, ","))

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query facts by IDs: %w", err)
	}
	defer rows.Close()

	var facts []Fact
	for rows.Next() {
		fact, err := s.scanFactFromRows(rows)
		if err != nil {
			continue
		}
		facts = append(facts, *fact)
	}

	return facts, nil
}

// SemanticSearch finds facts similar to the query embedding.
func (s *FactStore) SemanticSearch(ctx context.Context, queryEmbedding []float32, limit int) ([]ScoredFact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	if len(queryEmbedding) == 0 {
		return nil, errors.New("query embedding cannot be empty")
	}

	if limit <= 0 {
		limit = 10
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Normalize query vector
	queryVec := normalizeVectorF32(queryEmbedding)

	// Fetch candidates (more than limit for filtering)
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by
		FROM omem_facts
		WHERE embedding IS NOT NULL AND is_obsolete = FALSE
		ORDER BY created_at DESC
		LIMIT ?
	`, 100)
	if err != nil {
		return nil, fmt.Errorf("failed to query facts: %w", err)
	}
	defer rows.Close()

	var results []ScoredFact
	for rows.Next() {
		fact, err := s.scanFactFromRows(rows)
		if err != nil {
			continue
		}

		if len(fact.Embedding) > 0 {
			score := cosineSimilarityOptimized(queryVec, fact.Embedding)
			results = append(results, ScoredFact{
				Fact:          *fact,
				SemanticScore: score,
				Score:         score,
			})
		}
	}

	// Sort by score descending
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// FTSSearch performs full-text search using DuckDB FTS (BM25).
func (s *FactStore) FTSSearch(ctx context.Context, queryText string, limit int) ([]ScoredFact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	if queryText == "" {
		return nil, nil
	}

	if limit <= 0 {
		limit = 10
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Use DuckDB FTS with match_bm25 function
	rows, err := s.db.QueryContext(ctx, `
		WITH fts_results AS (
			SELECT *, fts_main_omem_facts.match_bm25(id, ?, fields := 'keywords,atomic_text') AS score
			FROM omem_facts
			WHERE score IS NOT NULL AND is_obsolete = FALSE
		)
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by, score
		FROM fts_results
		ORDER BY score DESC
		LIMIT ?
	`, queryText, limit)
	if err != nil {
		// FTS might not be available, fall back to LIKE search
		return s.fallbackTextSearch(ctx, queryText, limit)
	}
	defer rows.Close()

	var results []ScoredFact
	for rows.Next() {
		var fact Fact
		var embeddingBlob []byte
		var category string
		var keywordsStr sql.NullString
		var timestampAnchor sql.NullTime
		var location sql.NullString
		var entitiesJSON sql.NullString
		var episodeID sql.NullInt64
		var turnID sql.NullString
		var supersededBy sql.NullInt64
		var score float64

		err := rows.Scan(
			&fact.ID, &fact.Text, &fact.AtomicText, &category, &fact.Importance,
			&embeddingBlob, &keywordsStr, &timestampAnchor, &location, &entitiesJSON,
			&episodeID, &turnID, &fact.CreatedAt, &fact.LastAccessed, &fact.AccessCount,
			&fact.IsObsolete, &supersededBy, &score,
		)
		if err != nil {
			continue
		}

		fact.Category = FactCategory(category)
		if keywordsStr.Valid {
			fact.Keywords = strings.Fields(keywordsStr.String)
		}
		if timestampAnchor.Valid {
			fact.TimestampAnchor = &timestampAnchor.Time
		}
		if location.Valid {
			fact.Location = location.String
		}
		if entitiesJSON.Valid {
			json.Unmarshal([]byte(entitiesJSON.String), &fact.Entities)
		}
		if episodeID.Valid {
			fact.EpisodeID = episodeID.Int64
		}
		if turnID.Valid {
			fact.TurnID = turnID.String
		}
		if supersededBy.Valid {
			fact.SupersededBy = &supersededBy.Int64
		}
		if len(embeddingBlob) > 0 {
			fact.Embedding = bytesToFloat32Slice(embeddingBlob)
		}

		results = append(results, ScoredFact{
			Fact:         fact,
			LexicalScore: score,
			Score:        score,
		})
	}

	return results, nil
}

// fallbackTextSearch uses LIKE for text search when FTS is unavailable.
func (s *FactStore) fallbackTextSearch(ctx context.Context, queryText string, limit int) ([]ScoredFact, error) {
	// Split query into words for LIKE matching
	words := strings.Fields(strings.ToLower(queryText))
	if len(words) == 0 {
		return nil, nil
	}

	// Build WHERE clause with OR conditions
	conditions := make([]string, len(words))
	for i := range words {
		conditions[i] = "(LOWER(atomic_text) LIKE ? OR LOWER(keywords) LIKE ?)"
	}

	query := fmt.Sprintf(`
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by
		FROM omem_facts
		WHERE (%s) AND is_obsolete = FALSE
		ORDER BY importance DESC, created_at DESC
		LIMIT ?
	`, strings.Join(conditions, " OR "))

	// Flatten args for LIKE queries
	flatArgs := make([]interface{}, 0, len(words)*2+1)
	for _, word := range words {
		flatArgs = append(flatArgs, "%"+word+"%", "%"+word+"%")
	}
	flatArgs = append(flatArgs, limit)

	rows, err := s.db.QueryContext(ctx, query, flatArgs...)
	if err != nil {
		return nil, fmt.Errorf("failed to query facts with LIKE: %w", err)
	}
	defer rows.Close()

	var results []ScoredFact
	for rows.Next() {
		fact, err := s.scanFactFromRows(rows)
		if err != nil {
			continue
		}

		// Calculate a simple lexical score based on word matches
		score := s.calculateSimpleLexicalScore(fact.AtomicText, fact.Keywords, words)
		results = append(results, ScoredFact{
			Fact:         *fact,
			LexicalScore: score,
			Score:        score,
		})
	}

	return results, nil
}

// calculateSimpleLexicalScore computes a simple TF-based score.
func (s *FactStore) calculateSimpleLexicalScore(text string, keywords []string, queryWords []string) float64 {
	textLower := strings.ToLower(text)
	keywordsLower := strings.ToLower(strings.Join(keywords, " "))

	score := 0.0
	for _, word := range queryWords {
		if strings.Contains(textLower, word) {
			score += 1.0
		}
		if strings.Contains(keywordsLower, word) {
			score += 0.5
		}
	}

	// Normalize by query length
	if len(queryWords) > 0 {
		score /= float64(len(queryWords))
	}

	return score
}

// GetRecentFacts retrieves the most recent facts.
func (s *FactStore) GetRecentFacts(ctx context.Context, limit int) ([]Fact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	if limit <= 0 {
		limit = 20
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by
		FROM omem_facts
		WHERE is_obsolete = FALSE
		ORDER BY created_at DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent facts: %w", err)
	}
	defer rows.Close()

	var facts []Fact
	for rows.Next() {
		fact, err := s.scanFactFromRows(rows)
		if err != nil {
			continue
		}
		facts = append(facts, *fact)
	}

	return facts, nil
}

// GetFactsByCategory retrieves facts of a specific category.
func (s *FactStore) GetFactsByCategory(ctx context.Context, category FactCategory, limit int) ([]Fact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	if limit <= 0 {
		limit = 20
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by
		FROM omem_facts
		WHERE category = ? AND is_obsolete = FALSE
		ORDER BY importance DESC, created_at DESC
		LIMIT ?
	`, string(category), limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query facts by category: %w", err)
	}
	defer rows.Close()

	var facts []Fact
	for rows.Next() {
		fact, err := s.scanFactFromRows(rows)
		if err != nil {
			continue
		}
		facts = append(facts, *fact)
	}

	return facts, nil
}

// GetFactsByTimeRange retrieves facts within a time range.
func (s *FactStore) GetFactsByTimeRange(ctx context.Context, start, end time.Time, limit int) ([]Fact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	if limit <= 0 {
		limit = 50
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by
		FROM omem_facts
		WHERE is_obsolete = FALSE 
		  AND timestamp_anchor >= ? AND timestamp_anchor <= ?
		ORDER BY timestamp_anchor DESC
		LIMIT ?
	`, start, end, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query facts by time range: %w", err)
	}
	defer rows.Close()

	var facts []Fact
	for rows.Next() {
		fact, err := s.scanFactFromRows(rows)
		if err != nil {
			continue
		}
		facts = append(facts, *fact)
	}

	return facts, nil
}

// GetFactsByEpisode retrieves facts from a specific episode.
func (s *FactStore) GetFactsByEpisode(ctx context.Context, episodeID int64) ([]Fact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, fact_text, atomic_text, category, importance, embedding, keywords,
		       timestamp_anchor, location, entities_json, episode_id, turn_id,
		       created_at, last_accessed, access_count, is_obsolete, superseded_by
		FROM omem_facts
		WHERE episode_id = ? AND is_obsolete = FALSE
		ORDER BY created_at ASC
	`, episodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to query facts by episode: %w", err)
	}
	defer rows.Close()

	var facts []Fact
	for rows.Next() {
		fact, err := s.scanFactFromRows(rows)
		if err != nil {
			continue
		}
		facts = append(facts, *fact)
	}

	return facts, nil
}

// UpdateAccess records that a fact was accessed.
func (s *FactStore) UpdateAccess(ctx context.Context, factID int64) error {
	if s == nil || s.db == nil {
		return errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.updateAccessStmt.ExecContext(ctx, time.Now(), factID)
	if err != nil {
		return fmt.Errorf("failed to update access: %w", err)
	}

	return nil
}

// Episode methods

// InsertEpisode creates a new episode.
func (s *FactStore) InsertEpisode(ctx context.Context, sessionID string) (int64, error) {
	if s == nil || s.db == nil {
		return 0, errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var id int64
	err := s.insertEpisodeStmt.QueryRowContext(ctx, sessionID, time.Now()).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to insert episode: %w", err)
	}

	return id, nil
}

// UpdateEpisode updates an existing episode.
func (s *FactStore) UpdateEpisode(ctx context.Context, episode Episode) error {
	if s == nil || s.db == nil {
		return errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	entityMentionsStr := strings.Join(episode.EntityMentions, ",")

	_, err := s.updateEpisodeStmt.ExecContext(ctx,
		episode.EndedAt,
		episode.Summary,
		episode.FactCount,
		entityMentionsStr,
		episode.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update episode: %w", err)
	}

	return nil
}

// GetEpisode retrieves an episode by ID.
func (s *FactStore) GetEpisode(ctx context.Context, id int64) (*Episode, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	row := s.db.QueryRowContext(ctx, `
		SELECT id, session_id, started_at, ended_at, summary, fact_count, entity_mentions
		FROM omem_episodes
		WHERE id = ?
	`, id)

	var episode Episode
	var endedAt sql.NullTime
	var summary sql.NullString
	var entityMentions sql.NullString

	err := row.Scan(
		&episode.ID, &episode.SessionID, &episode.StartedAt,
		&endedAt, &summary, &episode.FactCount, &entityMentions,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get episode: %w", err)
	}

	if endedAt.Valid {
		episode.EndedAt = &endedAt.Time
	}
	if summary.Valid {
		episode.Summary = summary.String
	}
	if entityMentions.Valid && entityMentions.String != "" {
		episode.EntityMentions = strings.Split(entityMentions.String, ",")
	}

	return &episode, nil
}

// GetRecentEpisodes retrieves the most recent episodes.
func (s *FactStore) GetRecentEpisodes(ctx context.Context, limit int) ([]Episode, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	if limit <= 0 {
		limit = 10
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, session_id, started_at, ended_at, summary, fact_count, entity_mentions
		FROM omem_episodes
		ORDER BY started_at DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent episodes: %w", err)
	}
	defer rows.Close()

	var episodes []Episode
	for rows.Next() {
		var episode Episode
		var endedAt sql.NullTime
		var summary sql.NullString
		var entityMentions sql.NullString

		err := rows.Scan(
			&episode.ID, &episode.SessionID, &episode.StartedAt,
			&endedAt, &summary, &episode.FactCount, &entityMentions,
		)
		if err != nil {
			continue
		}

		if endedAt.Valid {
			episode.EndedAt = &endedAt.Time
		}
		if summary.Valid {
			episode.Summary = summary.String
		}
		if entityMentions.Valid && entityMentions.String != "" {
			episode.EntityMentions = strings.Split(entityMentions.String, ",")
		}

		episodes = append(episodes, episode)
	}

	return episodes, nil
}

// Rolling summary methods

// GetRollingSummary retrieves the current rolling summary.
func (s *FactStore) GetRollingSummary(ctx context.Context) (*RollingSummary, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	row := s.db.QueryRowContext(ctx, `
		SELECT id, summary, embedding, updated_at, source_fact_ids, fact_count, pending_fact_ids
		FROM omem_rolling_summary
		WHERE id = 1
	`)

	var summary RollingSummary
	var embeddingBlob []byte
	var sourceFactIDs sql.NullString
	var pendingFactIDs sql.NullString

	err := row.Scan(
		&summary.ID, &summary.Summary, &embeddingBlob, &summary.UpdatedAt,
		&sourceFactIDs, &summary.FactCount, &pendingFactIDs,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return &RollingSummary{ID: 1}, nil
		}
		return nil, fmt.Errorf("failed to get rolling summary: %w", err)
	}

	if len(embeddingBlob) > 0 {
		summary.Embedding = bytesToFloat32Slice(embeddingBlob)
	}
	if sourceFactIDs.Valid && sourceFactIDs.String != "" {
		summary.SourceFactIDs = parseIntList(sourceFactIDs.String)
	}
	if pendingFactIDs.Valid && pendingFactIDs.String != "" {
		summary.PendingFactIDs = parseIntList(pendingFactIDs.String)
	}

	return &summary, nil
}

// UpdateRollingSummary updates the rolling summary.
func (s *FactStore) UpdateRollingSummary(ctx context.Context, summary RollingSummary) error {
	if s == nil || s.db == nil {
		return errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var embeddingBlob []byte
	if len(summary.Embedding) > 0 {
		embeddingBlob = float32SliceToBytes(summary.Embedding)
	}

	sourceFactIDsStr := formatIntList(summary.SourceFactIDs)
	pendingFactIDsStr := formatIntList(summary.PendingFactIDs)

	_, err := s.db.ExecContext(ctx, `
		UPDATE omem_rolling_summary 
		SET summary = ?, embedding = ?, updated_at = ?, source_fact_ids = ?, 
		    fact_count = ?, pending_fact_ids = ?
		WHERE id = 1
	`, summary.Summary, embeddingBlob, time.Now(), sourceFactIDsStr,
		summary.FactCount, pendingFactIDsStr)
	if err != nil {
		return fmt.Errorf("failed to update rolling summary: %w", err)
	}

	return nil
}

// AddPendingFactID adds a fact ID to the pending list for incremental summary.
func (s *FactStore) AddPendingFactID(ctx context.Context, factID int64) error {
	if s == nil || s.db == nil {
		return errors.New("fact store not initialized")
	}

	summary, err := s.GetRollingSummary(ctx)
	if err != nil {
		return err
	}

	summary.PendingFactIDs = append(summary.PendingFactIDs, factID)
	return s.UpdateRollingSummary(ctx, *summary)
}

// GetStats returns statistics about the fact store.
func (s *FactStore) GetStats(ctx context.Context) (map[string]interface{}, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := make(map[string]interface{})

	// Total facts
	var totalFacts int
	row := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_facts`)
	if err := row.Scan(&totalFacts); err == nil {
		stats["total_facts"] = totalFacts
	}

	// Active facts
	var activeFacts int
	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_facts WHERE is_obsolete = FALSE`)
	if err := row.Scan(&activeFacts); err == nil {
		stats["active_facts"] = activeFacts
	}

	stats["obsolete_facts"] = totalFacts - activeFacts

	// Episodes count
	var episodeCount int
	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_episodes`)
	if err := row.Scan(&episodeCount); err == nil {
		stats["total_episodes"] = episodeCount
	}

	// Entities count
	var entityCount int
	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_entities`)
	if err := row.Scan(&entityCount); err == nil {
		stats["total_entities"] = entityCount
	}

	// Relations count
	var relationCount int
	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_relations WHERE is_obsolete = FALSE`)
	if err := row.Scan(&relationCount); err == nil {
		stats["total_relations"] = relationCount
	}

	// Facts by category
	categoryRows, err := s.db.QueryContext(ctx, `
		SELECT category, COUNT(*) 
		FROM omem_facts 
		WHERE is_obsolete = FALSE 
		GROUP BY category
	`)
	if err == nil {
		categories := make(map[string]int)
		for categoryRows.Next() {
			var cat string
			var count int
			if err := categoryRows.Scan(&cat, &count); err == nil {
				categories[cat] = count
			}
		}
		categoryRows.Close()
		stats["facts_by_category"] = categories
	}

	// Average importance
	var avgImportance float64
	row = s.db.QueryRowContext(ctx, `SELECT AVG(importance) FROM omem_facts WHERE is_obsolete = FALSE`)
	if err := row.Scan(&avgImportance); err == nil {
		stats["avg_importance"] = avgImportance
	}

	return stats, nil
}

// PruneOldFacts removes old, low-importance, obsolete facts.
func (s *FactStore) PruneOldFacts(ctx context.Context) (int, error) {
	if s == nil || s.db == nil {
		return 0, errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Count current facts
	var count int
	row := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_facts`)
	if err := row.Scan(&count); err != nil {
		return 0, err
	}

	if count <= s.config.PruneThreshold {
		return 0, nil
	}

	// Delete old obsolete facts first
	result, err := s.db.ExecContext(ctx, `
		DELETE FROM omem_facts 
		WHERE is_obsolete = TRUE
		AND id NOT IN (
			SELECT id FROM omem_facts 
			WHERE is_obsolete = TRUE 
			ORDER BY created_at DESC 
			LIMIT ?
		)
	`, s.config.PruneKeepRecent/4)
	if err != nil {
		return 0, fmt.Errorf("failed to prune obsolete facts: %w", err)
	}

	deleted, _ := result.RowsAffected()

	// If still over threshold, delete low-importance old facts
	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM omem_facts`)
	if err := row.Scan(&count); err != nil {
		return int(deleted), nil
	}

	if count > s.config.MaxFacts {
		toDelete := count - s.config.MaxFacts
		result, err = s.db.ExecContext(ctx, `
			DELETE FROM omem_facts 
			WHERE id IN (
				SELECT id FROM omem_facts 
				WHERE is_obsolete = FALSE
				ORDER BY importance ASC, access_count ASC, created_at ASC
				LIMIT ?
			)
		`, toDelete)
		if err == nil {
			d, _ := result.RowsAffected()
			deleted += d
		}
	}

	return int(deleted), nil
}

// Close releases database resources.
func (s *FactStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var firstErr error
	stmts := []*sql.Stmt{
		s.insertFactStmt,
		s.updateFactStmt,
		s.markObsoleteStmt,
		s.updateAccessStmt,
		s.insertEpisodeStmt,
		s.updateEpisodeStmt,
	}

	for _, stmt := range stmts {
		if stmt != nil {
			if err := stmt.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}

	if s.db != nil {
		if err := s.db.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

// Helper functions

func (s *FactStore) scanFact(row *sql.Row) (*Fact, error) {
	var fact Fact
	var embeddingBlob []byte
	var category string
	var keywordsStr sql.NullString
	var timestampAnchor sql.NullTime
	var location sql.NullString
	var entitiesJSON sql.NullString
	var episodeID sql.NullInt64
	var turnID sql.NullString
	var supersededBy sql.NullInt64

	err := row.Scan(
		&fact.ID, &fact.Text, &fact.AtomicText, &category, &fact.Importance,
		&embeddingBlob, &keywordsStr, &timestampAnchor, &location, &entitiesJSON,
		&episodeID, &turnID, &fact.CreatedAt, &fact.LastAccessed, &fact.AccessCount,
		&fact.IsObsolete, &supersededBy,
	)
	if err != nil {
		return nil, err
	}

	fact.Category = FactCategory(category)
	if keywordsStr.Valid {
		fact.Keywords = strings.Fields(keywordsStr.String)
	}
	if timestampAnchor.Valid {
		fact.TimestampAnchor = &timestampAnchor.Time
	}
	if location.Valid {
		fact.Location = location.String
	}
	if entitiesJSON.Valid {
		json.Unmarshal([]byte(entitiesJSON.String), &fact.Entities)
	}
	if episodeID.Valid {
		fact.EpisodeID = episodeID.Int64
	}
	if turnID.Valid {
		fact.TurnID = turnID.String
	}
	if supersededBy.Valid {
		fact.SupersededBy = &supersededBy.Int64
	}
	if len(embeddingBlob) > 0 {
		fact.Embedding = bytesToFloat32Slice(embeddingBlob)
	}

	return &fact, nil
}

func (s *FactStore) scanFactFromRows(rows *sql.Rows) (*Fact, error) {
	var fact Fact
	var embeddingBlob []byte
	var category string
	var keywordsStr sql.NullString
	var timestampAnchor sql.NullTime
	var location sql.NullString
	var entitiesJSON sql.NullString
	var episodeID sql.NullInt64
	var turnID sql.NullString
	var supersededBy sql.NullInt64

	err := rows.Scan(
		&fact.ID, &fact.Text, &fact.AtomicText, &category, &fact.Importance,
		&embeddingBlob, &keywordsStr, &timestampAnchor, &location, &entitiesJSON,
		&episodeID, &turnID, &fact.CreatedAt, &fact.LastAccessed, &fact.AccessCount,
		&fact.IsObsolete, &supersededBy,
	)
	if err != nil {
		return nil, err
	}

	fact.Category = FactCategory(category)
	if keywordsStr.Valid {
		fact.Keywords = strings.Fields(keywordsStr.String)
	}
	if timestampAnchor.Valid {
		fact.TimestampAnchor = &timestampAnchor.Time
	}
	if location.Valid {
		fact.Location = location.String
	}
	if entitiesJSON.Valid {
		json.Unmarshal([]byte(entitiesJSON.String), &fact.Entities)
	}
	if episodeID.Valid {
		fact.EpisodeID = episodeID.Int64
	}
	if turnID.Valid {
		fact.TurnID = turnID.String
	}
	if supersededBy.Valid {
		fact.SupersededBy = &supersededBy.Int64
	}
	if len(embeddingBlob) > 0 {
		fact.Embedding = bytesToFloat32Slice(embeddingBlob)
	}

	return &fact, nil
}

// Vector helper functions

func float32SliceToBytes(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(buf[i*4:], bits)
	}
	return buf
}

func bytesToFloat32Slice(buf []byte) []float32 {
	if len(buf)%4 != 0 {
		return nil
	}
	vec := make([]float32, len(buf)/4)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(buf[i*4:])
		vec[i] = math.Float32frombits(bits)
	}
	return vec
}

func normalizeVectorF32(vec []float32) []float32 {
	var norm float64
	for _, v := range vec {
		norm += float64(v * v)
	}
	if norm == 0 {
		return vec
	}
	factor := float32(1.0 / math.Sqrt(norm))
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = v * factor
	}
	return result
}

func cosineSimilarityOptimized(queryNormalized, doc []float32) float64 {
	if len(queryNormalized) != len(doc) || len(queryNormalized) == 0 {
		return 0
	}

	var dot, normSq float64
	n := len(queryNormalized)

	// Process 4 elements at a time
	i := 0
	for ; i <= n-4; i += 4 {
		dot += float64(queryNormalized[i])*float64(doc[i]) +
			float64(queryNormalized[i+1])*float64(doc[i+1]) +
			float64(queryNormalized[i+2])*float64(doc[i+2]) +
			float64(queryNormalized[i+3])*float64(doc[i+3])

		normSq += float64(doc[i])*float64(doc[i]) +
			float64(doc[i+1])*float64(doc[i+1]) +
			float64(doc[i+2])*float64(doc[i+2]) +
			float64(doc[i+3])*float64(doc[i+3])
	}

	for ; i < n; i++ {
		dot += float64(queryNormalized[i]) * float64(doc[i])
		normSq += float64(doc[i]) * float64(doc[i])
	}

	if normSq == 0 {
		return 0
	}

	return dot / math.Sqrt(normSq)
}

// Int list helpers

func parseIntList(s string) []int64 {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	result := make([]int64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var v int64
		if _, err := fmt.Sscanf(p, "%d", &v); err == nil {
			result = append(result, v)
		}
	}
	return result
}

func formatIntList(ids []int64) string {
	if len(ids) == 0 {
		return ""
	}
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = fmt.Sprintf("%d", id)
	}
	return strings.Join(parts, ",")
}

// GetDB returns the underlying database connection for sharing with other components.
func (s *FactStore) GetDB() *sql.DB {
	if s == nil {
		return nil
	}
	return s.db
}

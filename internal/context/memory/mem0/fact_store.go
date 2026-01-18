package mem0

import (
	"context"
	"database/sql"
	"encoding/binary"
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

// Fact represents an atomic piece of knowledge extracted from conversations.
type Fact struct {
	ID            int64        `json:"id"`
	Text          string       `json:"text"`
	Category      FactCategory `json:"category"`
	Importance    float64      `json:"importance"`
	Embedding     []float32    `json:"-"`
	CreatedAt     time.Time    `json:"created_at"`
	LastAccessed  time.Time    `json:"last_accessed"`
	AccessCount   int          `json:"access_count"`
	IsObsolete    bool         `json:"is_obsolete"`
	SupersededBy  *int64       `json:"superseded_by,omitempty"`
	SourceTurnIDs string       `json:"source_turn_ids,omitempty"`
}

// RollingSummary represents the cached rolling summary of facts.
type RollingSummary struct {
	ID          int64     `json:"id"`
	Summary     string    `json:"summary"`
	Embedding   []float32 `json:"-"`
	UpdatedAt   time.Time `json:"updated_at"`
	SourceFacts string    `json:"source_facts"`
	FactCount   int       `json:"fact_count"`
}

// FactStore provides DuckDB-backed storage for facts with full CRUD operations.
type FactStore struct {
	db     *sql.DB
	config StorageConfig
	mu     sync.RWMutex

	// Prepared statements for performance
	insertFactStmt   *sql.Stmt
	updateFactStmt   *sql.Stmt
	markObsoleteStmt *sql.Stmt
	searchStmt       *sql.Stmt
	updateAccessStmt *sql.Stmt
}

// NewFactStore creates a new fact store with the given configuration.
func NewFactStore(cfg StorageConfig) (*FactStore, error) {
	cfg = applyStorageDefaults(cfg)

	// Ensure directory exists
	if dir := filepath.Dir(filepath.Clean(cfg.DBPath)); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("failed to create mem0 directory: %w", err)
		}
	}

	db, err := sql.Open("duckdb", cfg.DBPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB for mem0: %w", err)
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
		cfg.DBPath = "openeye_mem0.duckdb"
	}
	if cfg.EmbeddingDim <= 0 {
		cfg.EmbeddingDim = 384
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

// bootstrap creates the required tables and indexes.
func (s *FactStore) bootstrap() error {
	// Create facts table - the core storage for atomic facts
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS facts (
			id INTEGER PRIMARY KEY,
			fact_text TEXT NOT NULL,
			category TEXT DEFAULT 'other',
			importance DOUBLE DEFAULT 0.5,
			embedding BLOB,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			access_count INTEGER DEFAULT 0,
			is_obsolete BOOLEAN DEFAULT FALSE,
			superseded_by INTEGER,
			source_turn_ids TEXT
		)
	`); err != nil {
		return fmt.Errorf("failed to create facts table: %w", err)
	}

	// Create rolling summary table
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS rolling_summary (
			id INTEGER PRIMARY KEY DEFAULT 1,
			summary TEXT,
			embedding BLOB,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			source_fact_ids TEXT,
			fact_count INTEGER DEFAULT 0
		)
	`); err != nil {
		return fmt.Errorf("failed to create rolling_summary table: %w", err)
	}

	// Create indexes for efficient queries
	indexes := []string{
		`CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)`,
		`CREATE INDEX IF NOT EXISTS idx_facts_importance ON facts(importance DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_facts_accessed ON facts(last_accessed DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_facts_obsolete ON facts(is_obsolete)`,
		`CREATE INDEX IF NOT EXISTS idx_facts_superseded ON facts(superseded_by)`,
	}

	for _, idx := range indexes {
		if _, err := s.db.Exec(idx); err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}
	}

	// Create sequence for fact IDs
	if _, err := s.db.Exec(`CREATE SEQUENCE IF NOT EXISTS facts_id_seq START 1`); err != nil {
		return fmt.Errorf("failed to create facts sequence: %w", err)
	}

	// Initialize rolling summary if not exists
	if _, err := s.db.Exec(`
		INSERT INTO rolling_summary (id, summary, fact_count) 
		SELECT 1, '', 0 
		WHERE NOT EXISTS (SELECT 1 FROM rolling_summary WHERE id = 1)
	`); err != nil {
		// Ignore error if row already exists
		_ = err
	}

	return nil
}

// prepareStatements pre-compiles frequently used SQL statements.
func (s *FactStore) prepareStatements() error {
	var err error

	s.insertFactStmt, err = s.db.Prepare(`
		INSERT INTO facts (id, fact_text, category, importance, embedding, created_at, source_turn_ids)
		VALUES (nextval('facts_id_seq'), ?, ?, ?, ?, ?, ?)
		RETURNING id
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare insert fact statement: %w", err)
	}

	s.updateFactStmt, err = s.db.Prepare(`
		UPDATE facts 
		SET fact_text = ?, category = ?, importance = ?, embedding = ?, last_accessed = ?
		WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare update fact statement: %w", err)
	}

	s.markObsoleteStmt, err = s.db.Prepare(`
		UPDATE facts 
		SET is_obsolete = TRUE, superseded_by = ?
		WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare mark obsolete statement: %w", err)
	}

	s.updateAccessStmt, err = s.db.Prepare(`
		UPDATE facts 
		SET last_accessed = ?, access_count = access_count + 1
		WHERE id = ?
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare update access statement: %w", err)
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

	var id int64
	err := s.insertFactStmt.QueryRowContext(ctx,
		fact.Text,
		string(fact.Category),
		fact.Importance,
		embeddingBlob,
		fact.CreatedAt,
		fact.SourceTurnIDs,
	).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to insert fact: %w", err)
	}

	return id, nil
}

// UpdateFact updates an existing fact (for UPDATE operation).
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

	_, err := s.updateFactStmt.ExecContext(ctx,
		fact.Text,
		string(fact.Category),
		fact.Importance,
		embeddingBlob,
		time.Now(),
		fact.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update fact: %w", err)
	}

	return nil
}

// MarkObsolete marks a fact as obsolete (for DELETE operation).
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
		SELECT id, fact_text, category, importance, embedding, 
		       created_at, last_accessed, access_count, is_obsolete, 
		       superseded_by, source_turn_ids
		FROM facts
		WHERE id = ?
	`, id)

	return s.scanFact(row)
}

// SearchSimilarFacts finds facts similar to the query embedding.
func (s *FactStore) SearchSimilarFacts(ctx context.Context, queryEmbedding []float32, limit int, includeObsolete bool) ([]Fact, []float64, error) {
	if s == nil || s.db == nil {
		return nil, nil, errors.New("fact store not initialized")
	}

	if len(queryEmbedding) == 0 {
		return nil, nil, errors.New("query embedding cannot be empty")
	}

	if limit <= 0 {
		limit = 10
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Normalize query vector
	queryVec := normalizeVectorF32(queryEmbedding)

	// Build query based on whether to include obsolete facts
	query := `
		SELECT id, fact_text, category, importance, embedding, 
		       created_at, last_accessed, access_count, is_obsolete, 
		       superseded_by, source_turn_ids
		FROM facts
		WHERE embedding IS NOT NULL
	`
	if !includeObsolete {
		query += ` AND is_obsolete = FALSE`
	}
	query += ` ORDER BY importance DESC, created_at DESC LIMIT ?`

	rows, err := s.db.QueryContext(ctx, query, limit*5) // Fetch more for scoring
	if err != nil {
		return nil, nil, fmt.Errorf("failed to query facts: %w", err)
	}
	defer rows.Close()

	type scoredFact struct {
		fact  Fact
		score float64
	}

	var results []scoredFact
	for rows.Next() {
		fact, err := s.scanFactFromRows(rows)
		if err != nil {
			continue
		}

		if len(fact.Embedding) > 0 {
			score := cosineSimilarityOptimized(queryVec, fact.Embedding)
			results = append(results, scoredFact{fact: *fact, score: score})
		}
	}

	// Sort by score descending
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Limit results
	if len(results) > limit {
		results = results[:limit]
	}

	facts := make([]Fact, len(results))
	scores := make([]float64, len(results))
	for i, r := range results {
		facts[i] = r.fact
		scores[i] = r.score
	}

	return facts, scores, nil
}

// GetRecentFacts retrieves the most recent facts.
func (s *FactStore) GetRecentFacts(ctx context.Context, limit int, includeObsolete bool) ([]Fact, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	if limit <= 0 {
		limit = 20
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	query := `
		SELECT id, fact_text, category, importance, embedding, 
		       created_at, last_accessed, access_count, is_obsolete, 
		       superseded_by, source_turn_ids
		FROM facts
	`
	if !includeObsolete {
		query += ` WHERE is_obsolete = FALSE`
	}
	query += ` ORDER BY created_at DESC LIMIT ?`

	rows, err := s.db.QueryContext(ctx, query, limit)
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
		SELECT id, fact_text, category, importance, embedding, 
		       created_at, last_accessed, access_count, is_obsolete, 
		       superseded_by, source_turn_ids
		FROM facts
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

// UpdateAccess records that a fact was accessed (for frequency scoring).
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
	row := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM facts`)
	if err := row.Scan(&totalFacts); err == nil {
		stats["total_facts"] = totalFacts
	}

	// Active facts
	var activeFacts int
	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM facts WHERE is_obsolete = FALSE`)
	if err := row.Scan(&activeFacts); err == nil {
		stats["active_facts"] = activeFacts
	}

	// Obsolete facts
	stats["obsolete_facts"] = totalFacts - activeFacts

	// Facts by category
	categoryRows, err := s.db.QueryContext(ctx, `
		SELECT category, COUNT(*) 
		FROM facts 
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
	row = s.db.QueryRowContext(ctx, `SELECT AVG(importance) FROM facts WHERE is_obsolete = FALSE`)
	if err := row.Scan(&avgImportance); err == nil {
		stats["avg_importance"] = avgImportance
	}

	return stats, nil
}

// GetRollingSummary retrieves the current rolling summary.
func (s *FactStore) GetRollingSummary(ctx context.Context) (*RollingSummary, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fact store not initialized")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	row := s.db.QueryRowContext(ctx, `
		SELECT id, summary, embedding, updated_at, source_fact_ids, fact_count
		FROM rolling_summary
		WHERE id = 1
	`)

	var summary RollingSummary
	var embeddingBlob []byte
	var updatedAt time.Time
	var sourceFacts sql.NullString

	err := row.Scan(
		&summary.ID,
		&summary.Summary,
		&embeddingBlob,
		&updatedAt,
		&sourceFacts,
		&summary.FactCount,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return &RollingSummary{ID: 1}, nil
		}
		return nil, fmt.Errorf("failed to get rolling summary: %w", err)
	}

	summary.UpdatedAt = updatedAt
	if sourceFacts.Valid {
		summary.SourceFacts = sourceFacts.String
	}
	if len(embeddingBlob) > 0 {
		summary.Embedding = bytesToFloat32Slice(embeddingBlob)
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

	_, err := s.db.ExecContext(ctx, `
		UPDATE rolling_summary 
		SET summary = ?, embedding = ?, updated_at = ?, source_fact_ids = ?, fact_count = ?
		WHERE id = 1
	`, summary.Summary, embeddingBlob, time.Now(), summary.SourceFacts, summary.FactCount)
	if err != nil {
		return fmt.Errorf("failed to update rolling summary: %w", err)
	}

	return nil
}

// PruneOldFacts removes old, low-importance, obsolete facts to stay within limits.
func (s *FactStore) PruneOldFacts(ctx context.Context) (int, error) {
	if s == nil || s.db == nil {
		return 0, errors.New("fact store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Count current facts
	var count int
	row := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM facts`)
	if err := row.Scan(&count); err != nil {
		return 0, err
	}

	if count <= s.config.PruneThreshold {
		return 0, nil // No pruning needed
	}

	// Delete old obsolete facts first
	result, err := s.db.ExecContext(ctx, `
		DELETE FROM facts 
		WHERE is_obsolete = TRUE
		AND id NOT IN (
			SELECT id FROM facts 
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
	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM facts`)
	if err := row.Scan(&count); err != nil {
		return int(deleted), nil
	}

	if count > s.config.MaxFacts {
		toDelete := count - s.config.MaxFacts
		result, err = s.db.ExecContext(ctx, `
			DELETE FROM facts 
			WHERE id IN (
				SELECT id FROM facts 
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

	s.insertFactStmt = nil
	s.updateFactStmt = nil
	s.markObsoleteStmt = nil
	s.updateAccessStmt = nil
	s.db = nil

	return firstErr
}

// Helper functions

func (s *FactStore) scanFact(row *sql.Row) (*Fact, error) {
	var fact Fact
	var embeddingBlob []byte
	var category string
	var createdAt, lastAccessed time.Time
	var supersededBy sql.NullInt64
	var sourceTurnIDs sql.NullString

	err := row.Scan(
		&fact.ID,
		&fact.Text,
		&category,
		&fact.Importance,
		&embeddingBlob,
		&createdAt,
		&lastAccessed,
		&fact.AccessCount,
		&fact.IsObsolete,
		&supersededBy,
		&sourceTurnIDs,
	)
	if err != nil {
		return nil, err
	}

	fact.Category = FactCategory(category)
	fact.CreatedAt = createdAt
	fact.LastAccessed = lastAccessed
	if supersededBy.Valid {
		fact.SupersededBy = &supersededBy.Int64
	}
	if sourceTurnIDs.Valid {
		fact.SourceTurnIDs = sourceTurnIDs.String
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
	var createdAt, lastAccessed time.Time
	var supersededBy sql.NullInt64
	var sourceTurnIDs sql.NullString

	err := rows.Scan(
		&fact.ID,
		&fact.Text,
		&category,
		&fact.Importance,
		&embeddingBlob,
		&createdAt,
		&lastAccessed,
		&fact.AccessCount,
		&fact.IsObsolete,
		&supersededBy,
		&sourceTurnIDs,
	)
	if err != nil {
		return nil, err
	}

	fact.Category = FactCategory(category)
	fact.CreatedAt = createdAt
	fact.LastAccessed = lastAccessed
	if supersededBy.Valid {
		fact.SupersededBy = &supersededBy.Int64
	}
	if sourceTurnIDs.Valid {
		fact.SourceTurnIDs = sourceTurnIDs.String
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

// cosineSimilarityOptimized computes cosine similarity between a normalized query vector
// and an unnormalized document vector. Uses SIMD-friendly loop unrolling.
func cosineSimilarityOptimized(queryNormalized, doc []float32) float64 {
	if len(queryNormalized) != len(doc) || len(queryNormalized) == 0 {
		return 0
	}

	var dot, normSq float64
	n := len(queryNormalized)

	// Process 4 elements at a time for better CPU cache utilization
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

	// Handle remaining elements
	for ; i < n; i++ {
		dot += float64(queryNormalized[i]) * float64(doc[i])
		normSq += float64(doc[i]) * float64(doc[i])
	}

	if normSq == 0 {
		return 0
	}

	return dot / math.Sqrt(normSq)
}

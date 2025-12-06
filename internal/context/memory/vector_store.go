package memory

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	_ "github.com/marcboeker/go-duckdb"
)

type VectorEntry struct {
	ID        int64
	Text      string
	Summary   string
	Role      string
	Embedding []float32
	CreatedAt time.Time
	TokenCount int
}

type VectorStoreConfig struct {
	DBPath             string
	EmbeddingDim       int
	MaxContextTokens   int
	CompressionEnabled bool
	SlidingWindowSize  int
	MinSimilarity      float64
}

// VectorStore provides a DuckDB-backed vector memory engine for SLMs.
type VectorStore struct {
	db                 *sql.DB
	config             VectorStoreConfig
	mu                 sync.RWMutex
	insertStmt         *sql.Stmt
	searchStmt         *sql.Stmt
	compressionEnabled bool
}

// DefaultVectorStoreConfig returns sensible defaults for the vector store.
func DefaultVectorStoreConfig() VectorStoreConfig {
	return VectorStoreConfig{
		DBPath:             "openeye_vector.duckdb",
		EmbeddingDim:       384,
		MaxContextTokens:   2048,
		CompressionEnabled: true,
		SlidingWindowSize:  50,
		MinSimilarity:      0.3,
	}
}

// NewVectorStore creates a new DuckDB-backed vector memory store.
func NewVectorStore(cfg VectorStoreConfig) (*VectorStore, error) {
	if cfg.DBPath == "" {
		cfg.DBPath = "openeye_vector.duckdb"
	}
	if cfg.EmbeddingDim <= 0 {
		cfg.EmbeddingDim = 384
	}
	if cfg.MaxContextTokens <= 0 {
		cfg.MaxContextTokens = 2048
	}
	if cfg.SlidingWindowSize <= 0 {
		cfg.SlidingWindowSize = 50
	}
	if cfg.MinSimilarity <= 0 {
		cfg.MinSimilarity = 0.3
	}

	if dir := filepath.Dir(filepath.Clean(cfg.DBPath)); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("failed to create memory directory: %w", err)
		}
	}

	db, err := sql.Open("duckdb", cfg.DBPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	store := &VectorStore{
		db:                 db,
		config:             cfg,
		compressionEnabled: cfg.CompressionEnabled,
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

// bootstrap creates the required tables and indexes.
func (s *VectorStore) bootstrap() error {
	// Create main memory table
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS memory (
			id INTEGER PRIMARY KEY,
			text TEXT NOT NULL,
			summary TEXT,
			role TEXT NOT NULL,
			embedding BLOB,
			token_count INTEGER DEFAULT 0,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			compressed BOOLEAN DEFAULT FALSE,
			parent_id INTEGER,
			importance_score DOUBLE DEFAULT 0.0
		)
	`); err != nil {
		return fmt.Errorf("failed to create memory table: %w", err)
	}

	// Create compressed summaries table for long-term memory
	if _, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS memory_compressed (
			id INTEGER PRIMARY KEY,
			summary TEXT NOT NULL,
			embedding BLOB,
			source_ids TEXT,
			token_count INTEGER DEFAULT 0,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			time_range_start TIMESTAMP,
			time_range_end TIMESTAMP
		)
	`); err != nil {
		return fmt.Errorf("failed to create compressed memory table: %w", err)
	}

	// Create indexes for faster queries
	if _, err := s.db.Exec(`
		CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory(created_at DESC)
	`); err != nil {
		return fmt.Errorf("failed to create timestamp index: %w", err)
	}

	if _, err := s.db.Exec(`
		CREATE INDEX IF NOT EXISTS idx_memory_role ON memory(role)
	`); err != nil {
		return fmt.Errorf("failed to create role index: %w", err)
	}

	// Create sequence for IDs if not exists
	if _, err := s.db.Exec(`
		CREATE SEQUENCE IF NOT EXISTS memory_id_seq START 1
	`); err != nil {
		return fmt.Errorf("failed to create sequence: %w", err)
	}

	return nil
}

// prepareStatements pre-compiles frequently used SQL statements.
func (s *VectorStore) prepareStatements() error {
	var err error

	s.insertStmt, err = s.db.Prepare(`
		INSERT INTO memory (id, text, summary, role, embedding, token_count, created_at, importance_score)
		VALUES (nextval('memory_id_seq'), ?, ?, ?, ?, ?, ?, ?)
		RETURNING id
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare insert statement: %w", err)
	}

	return nil
}

// InsertMemory stores a new memory entry with optional embedding and summary.
func (s *VectorStore) InsertMemory(ctx context.Context, text, summary, role string, embedding []float32) (int64, error) {
	if s == nil || s.db == nil {
		return 0, errors.New("vector store not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	text = strings.TrimSpace(text)
	if text == "" {
		return 0, errors.New("text cannot be empty")
	}
	if role == "" {
		role = "user"
	}

	tokenCount := estimateTokens(text)
	importance := calculateImportance(text, role)

	var embeddingBlob []byte
	if len(embedding) > 0 {
		embeddingBlob = float32SliceToBytes(embedding)
	}

	var id int64
	err := s.insertStmt.QueryRowContext(ctx, text, summary, role, embeddingBlob, tokenCount, time.Now(), importance).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to insert memory: %w", err)
	}

	return id, nil
}

// SearchMemory performs vector similarity search and returns top-K relevant memories.
func (s *VectorStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]VectorEntry, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("vector store not initialized")
	}

	if len(queryEmbedding) == 0 {
		return nil, errors.New("query embedding cannot be empty")
	}

	if limit <= 0 {
		limit = 5
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Fetch all memories with embeddings
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, text, summary, role, embedding, token_count, created_at
		FROM memory
		WHERE embedding IS NOT NULL
		ORDER BY created_at DESC
		LIMIT 1000
	`)
	if err != nil {
		return nil, fmt.Errorf("failed to query memories: %w", err)
	}
	defer rows.Close()

	type scoredEntry struct {
		entry VectorEntry
		score float64
	}

	var results []scoredEntry
	queryVec := normalizeVectorF32(queryEmbedding)

	for rows.Next() {
		var entry VectorEntry
		var embeddingBlob []byte
		var createdAt time.Time

		if err := rows.Scan(&entry.ID, &entry.Text, &entry.Summary, &entry.Role, &embeddingBlob, &entry.TokenCount, &createdAt); err != nil {
			continue
		}
		entry.CreatedAt = createdAt

		if len(embeddingBlob) > 0 {
			entry.Embedding = bytesToFloat32Slice(embeddingBlob)
			entryVec := normalizeVectorF32(entry.Embedding)
			score := cosineSimilarityF32(queryVec, entryVec)

			if score >= s.config.MinSimilarity {
				results = append(results, scoredEntry{entry: entry, score: score})
			}
		}
	}

	// Sort by similarity score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if len(results) > limit {
		results = results[:limit]
	}

	entries := make([]VectorEntry, len(results))
	for i, r := range results {
		entries[i] = r.entry
	}

	return entries, nil
}

// RetrieveContext returns memory snippets optimized for context window limits.
func (s *VectorStore) RetrieveContext(ctx context.Context, queryEmbedding []float32, maxTokens int) ([]VectorEntry, int, error) {
	if maxTokens <= 0 {
		maxTokens = s.config.MaxContextTokens
	}

	// Get more candidates than we need
	candidates, err := s.SearchMemory(ctx, queryEmbedding, s.config.SlidingWindowSize)
	if err != nil {
		return nil, 0, err
	}

	// Apply sliding window context fitting
	var selected []VectorEntry
	totalTokens := 0

	for _, entry := range candidates {
		if totalTokens+entry.TokenCount > maxTokens {
			break
		}
		selected = append(selected, entry)
		totalTokens += entry.TokenCount
	}

	return selected, totalTokens, nil
}

// GetRecentMemories retrieves the most recent N memory entries.
func (s *VectorStore) GetRecentMemories(ctx context.Context, limit int) ([]VectorEntry, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("vector store not initialized")
	}

	if limit <= 0 {
		limit = 10
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, text, summary, role, embedding, token_count, created_at
		FROM memory
		ORDER BY created_at DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent memories: %w", err)
	}
	defer rows.Close()

	var entries []VectorEntry
	for rows.Next() {
		var entry VectorEntry
		var embeddingBlob []byte
		var createdAt time.Time

		if err := rows.Scan(&entry.ID, &entry.Text, &entry.Summary, &entry.Role, &embeddingBlob, &entry.TokenCount, &createdAt); err != nil {
			continue
		}
		entry.CreatedAt = createdAt
		if len(embeddingBlob) > 0 {
			entry.Embedding = bytesToFloat32Slice(embeddingBlob)
		}
		entries = append(entries, entry)
	}

	// Reverse to get chronological order
	for i, j := 0, len(entries)-1; i < j; i, j = i+1, j-1 {
		entries[i], entries[j] = entries[j], entries[i]
	}

	return entries, nil
}

// CompressOldMemories merges old memory entries into compressed summaries.
func (s *VectorStore) CompressOldMemories(ctx context.Context, olderThan time.Duration, summaryFn func([]string) (string, error), embedFn func(string) ([]float32, error)) error {
	if !s.compressionEnabled {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	cutoff := time.Now().Add(-olderThan)

	// Find old uncompressed memories
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, text, role, created_at
		FROM memory
		WHERE created_at < ? AND compressed = FALSE
		ORDER BY created_at ASC
		LIMIT 100
	`, cutoff)
	if err != nil {
		return fmt.Errorf("failed to query old memories: %w", err)
	}
	defer rows.Close()

	var texts []string
	var ids []int64
	var timeStart, timeEnd time.Time

	for rows.Next() {
		var id int64
		var text, role string
		var createdAt time.Time

		if err := rows.Scan(&id, &text, &role, &createdAt); err != nil {
			continue
		}

		ids = append(ids, id)
		texts = append(texts, fmt.Sprintf("%s: %s", role, text))

		if timeStart.IsZero() || createdAt.Before(timeStart) {
			timeStart = createdAt
		}
		if timeEnd.IsZero() || createdAt.After(timeEnd) {
			timeEnd = createdAt
		}
	}

	if len(texts) < 5 {
		return nil // Not enough to compress
	}

	// Generate summary
	summary, err := summaryFn(texts)
	if err != nil {
		return fmt.Errorf("failed to generate compression summary: %w", err)
	}

	// Generate embedding for summary
	var embeddingBlob []byte
	if embedFn != nil {
		embedding, err := embedFn(summary)
		if err == nil && len(embedding) > 0 {
			embeddingBlob = float32SliceToBytes(embedding)
		}
	}

	// Store compressed summary
	sourceIDs := formatIDList(ids)
	tokenCount := estimateTokens(summary)

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO memory_compressed (summary, embedding, source_ids, token_count, time_range_start, time_range_end)
		VALUES (?, ?, ?, ?, ?, ?)
	`, summary, embeddingBlob, sourceIDs, tokenCount, timeStart, timeEnd)
	if err != nil {
		return fmt.Errorf("failed to insert compressed memory: %w", err)
	}

	// Mark original memories as compressed
	for _, id := range ids {
		s.db.ExecContext(ctx, `UPDATE memory SET compressed = TRUE WHERE id = ?`, id)
	}

	return nil
}

// PruneMemories removes old compressed memories to free space.
func (s *VectorStore) PruneMemories(ctx context.Context, keepLast int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Delete old compressed source memories
	_, err := s.db.ExecContext(ctx, `
		DELETE FROM memory 
		WHERE compressed = TRUE 
		AND id NOT IN (
			SELECT id FROM memory WHERE compressed = TRUE ORDER BY created_at DESC LIMIT ?
		)
	`, keepLast)
	if err != nil {
		return fmt.Errorf("failed to prune memories: %w", err)
	}

	return nil
}

// GetMemoryStats returns statistics about memory usage.
func (s *VectorStore) GetMemoryStats(ctx context.Context) (map[string]interface{}, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := make(map[string]interface{})

	var totalCount, compressedCount int
	var totalTokens int64

	row := s.db.QueryRowContext(ctx, `SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM memory`)
	if err := row.Scan(&totalCount, &totalTokens); err != nil {
		return nil, err
	}
	stats["total_memories"] = totalCount
	stats["total_tokens"] = totalTokens

	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM memory WHERE compressed = TRUE`)
	if err := row.Scan(&compressedCount); err == nil {
		stats["compressed_memories"] = compressedCount
	}

	row = s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM memory_compressed`)
	var compSummaryCount int
	if err := row.Scan(&compSummaryCount); err == nil {
		stats["compressed_summaries"] = compSummaryCount
	}

	return stats, nil
}

// Close releases database resources.
func (s *VectorStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var firstErr error
	if s.insertStmt != nil {
		if err := s.insertStmt.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if s.searchStmt != nil {
		if err := s.searchStmt.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if s.db != nil {
		if err := s.db.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	s.insertStmt = nil
	s.searchStmt = nil
	s.db = nil

	return firstErr
}

// Helper functions

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

func cosineSimilarityF32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}

func estimateTokens(text string) int {
	// Rough estimation: ~4 chars per token for English
	return len(text) / 4
}

func calculateImportance(text string, role string) float64 {
	score := 0.5

	// Questions are often important
	if strings.Contains(text, "?") {
		score += 0.1
	}

	// Longer content might be more informative
	if len(text) > 200 {
		score += 0.1
	}

	// Assistant responses might contain key information
	if role == "assistant" {
		score += 0.1
	}

	// Look for importance indicators
	importanceWords := []string{"important", "remember", "key", "critical", "note", "must"}
	textLower := strings.ToLower(text)
	for _, word := range importanceWords {
		if strings.Contains(textLower, word) {
			score += 0.05
		}
	}

	if score > 1.0 {
		score = 1.0
	}
	return score
}

func formatIDList(ids []int64) string {
	strs := make([]string, len(ids))
	for i, id := range ids {
		strs[i] = fmt.Sprintf("%d", id)
	}
	return strings.Join(strs, ",")
}

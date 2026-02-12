package memory

import (
	"database/sql"
	"errors"
	"fmt"
	_ "modernc.org/sqlite"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Entry represents a single conversational turn persisted in the store.
type Entry struct {
	Role      string
	Content   string
	CreatedAt time.Time
}

// Store wraps a SQLite database connection for persisting conversation history.
type Store struct {
	db         *sql.DB
	insertStmt *sql.Stmt
	selectStmt *sql.Stmt
	mu         sync.RWMutex
}

// NewStore opens (and initializes) a SQLite database file for conversation memory.
func NewStore(path string) (*Store, error) {
	if path == "" {
		path = "openeye_memory.db"
	}

	if dir := filepath.Dir(filepath.Clean(path)); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("failed to ensure memory directory: %w", err)
		}
	}

	db, err := sql.Open("sqlite", fmt.Sprintf("file:%s?_busy_timeout=5000&_journal_mode=WAL", path))
	if err != nil {
		return nil, fmt.Errorf("failed to open memory store: %w", err)
	}

	if err := bootstrap(db); err != nil {
		db.Close()
		return nil, err
	}

	insertStmt, err := db.Prepare(`INSERT INTO interactions (role, content, created_at) VALUES (?, ?, ?)`)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to prepare insert statement: %w", err)
	}

	selectStmt, err := db.Prepare(`SELECT role, content, created_at FROM interactions ORDER BY created_at DESC LIMIT ?`)
	if err != nil {
		insertStmt.Close()
		db.Close()
		return nil, fmt.Errorf("failed to prepare select statement: %w", err)
	}

	return &Store{db: db, insertStmt: insertStmt, selectStmt: selectStmt}, nil
}

func bootstrap(db *sql.DB) error {
	if _, err := db.Exec(`
		PRAGMA journal_mode=WAL;
		PRAGMA synchronous=NORMAL;
	`); err != nil {
		return fmt.Errorf("failed to configure database: %w", err)
	}

	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS interactions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			role TEXT NOT NULL,
			content TEXT NOT NULL,
			created_at INTEGER NOT NULL
		);
	`); err != nil {
		return fmt.Errorf("failed to create interactions table: %w", err)
	}

	return nil
}

// Append persists a new conversational turn into the store.
func (s *Store) Append(role, content string) error {
	if s == nil || s.db == nil {
		return errors.New("memory store is not initialized")
	}

	if role == "" {
		return errors.New("role must not be empty")
	}
	if content == "" {
		return errors.New("content must not be empty")
	}

	s.mu.RLock()
	stmt := s.insertStmt
	s.mu.RUnlock()

	if stmt == nil {
		return errors.New("insert statement not prepared")
	}

	if _, err := stmt.Exec(role, content, time.Now().Unix()); err != nil {
		return fmt.Errorf("failed to append memory entry: %w", err)
	}

	return nil
}

// Recent retrieves up to the provided limit of the most recent entries.
func (s *Store) Recent(limit int) ([]Entry, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("memory store is not initialized")
	}
	if limit <= 0 {
		return nil, errors.New("limit must be greater than zero")
	}

	s.mu.RLock()
	stmt := s.selectStmt
	s.mu.RUnlock()
	if stmt == nil {
		return nil, errors.New("select statement not prepared")
	}

	rows, err := stmt.Query(limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent memory: %w", err)
	}
	defer rows.Close()

	entries := make([]Entry, 0, limit)
	for rows.Next() {
		var (
			role    string
			content string
			ts      int64
		)
		if err := rows.Scan(&role, &content, &ts); err != nil {
			return nil, fmt.Errorf("failed to scan memory row: %w", err)
		}
		entries = append(entries, Entry{Role: role, Content: content, CreatedAt: time.Unix(ts, 0)})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating memory rows: %w", err)
	}

	return entries, nil
}

// Search performs a text search on the stored content using LIKE pattern matching.
// It extracts key search terms from the query to improve recall.
func (s *Store) Search(query string, limit int) ([]Entry, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("memory store is not initialized")
	}
	if limit <= 0 {
		return nil, errors.New("limit must be greater than zero")
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Extract key terms from query
	searchTerms := extractSearchTerms(query)

	// Build OR query for all extracted terms
	if len(searchTerms) == 0 {
		return nil, nil
	}

	queryBuilder := `SELECT role, content, created_at FROM interactions WHERE `
	conditions := make([]string, len(searchTerms))
	args := make([]interface{}, len(searchTerms))

	for i, term := range searchTerms {
		conditions[i] = "content LIKE ?"
		args[i] = "%" + term + "%"
	}
	queryBuilder += strings.Join(conditions, " OR ")
	queryBuilder += " ORDER BY created_at DESC LIMIT ?"
	args = append(args, limit)

	rows, err := s.db.Query(queryBuilder, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to search memory: %w", err)
	}
	defer rows.Close()

	entries := make([]Entry, 0, limit)
	for rows.Next() {
		var (
			role    string
			content string
			ts      int64
		)
		if err := rows.Scan(&role, &content, &ts); err != nil {
			return nil, fmt.Errorf("failed to scan memory row: %w", err)
		}
		entries = append(entries, Entry{Role: role, Content: content, CreatedAt: time.Unix(ts, 0)})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating memory rows: %w", err)
	}

	return entries, nil
}

// extractSearchTerms extracts key terms from a query for searching
func extractSearchTerms(query string) []string {
	query = strings.ToLower(query)

	// Remove common question words and noise
	stopWords := []string{
		"what", "is", "are", "do", "does", "did", "when", "where", "who", "how",
		"my", "the", "a", "an", "can", "could", "would", "should", "will", "you",
		"i", "me", "your", "yours", "tell", "me", "about", "remember",
	}

	// Tokenize
	words := make([]string, 0)
	current := ""
	for _, c := range query {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			current += string(c)
		} else {
			if current != "" {
				words = append(words, current)
				current = ""
			}
		}
	}
	if current != "" {
		words = append(words, current)
	}

	// Filter stop words and short words
	terms := make([]string, 0)
	for _, w := range words {
		if len(w) <= 2 {
			continue
		}
		isStop := false
		for _, sw := range stopWords {
			if w == sw {
				isStop = true
				break
			}
		}
		if !isStop {
			terms = append(terms, w)
		}
	}

	return terms
}

// Close releases database resources held by the store.
func (s *Store) Close() error {
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
	if s.selectStmt != nil {
		if err := s.selectStmt.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if s.db != nil {
		if err := s.db.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	s.insertStmt = nil
	s.selectStmt = nil
	s.db = nil

	return firstErr
}

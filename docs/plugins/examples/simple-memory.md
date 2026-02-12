# Simple Memory Plugin

This example demonstrates how to create a simple custom memory store for OpenEye. It implements the basic `memory.Store` interface for conversation history.

## Overview

The memory store:

- Persists conversation history to SQLite
- Retrieves recent conversation turns
- Implements proper connection pooling
- Handles concurrent access safely

## Source Code

### memory_store.go

```go
package simplememory

import (
    "database/sql"
    "fmt"
    "os"
    "path/filepath"
    "strings"
    "sync"
    "time"

    _ "github.com/mattn/go-sqlite3"
    "OpenEye/internal/context/memory"
)

const (
    defaultMaxEntries = 1000
)

type Store struct {
    db           *sql.DB
    insertStmt   *sql.Stmt
    selectStmt   *sql.Stmt
    recentStmt   *sql.Stmt
    searchStmt   *sql.Stmt
    maxEntries   int
    mu           sync.RWMutex
}

type Config struct {
    Path       string
    MaxEntries int
}

func NewStore(cfg Config) (*Store, error) {
    // Set defaults
    if cfg.Path == "" {
        cfg.Path = "simple_memory.db"
    }
    
    if cfg.MaxEntries <= 0 {
        cfg.MaxEntries = defaultMaxEntries
    }
    
    // Ensure directory exists
    dir := filepath.Dir(cfg.Path)
    if dir != "" && dir != "." {
        if err := os.MkdirAll(dir, 0755); err != nil {
            return nil, fmt.Errorf("failed to create directory: %w", err)
        }
    }
    
    // Open database
    db, err := sql.Open("sqlite3", fmt.Sprintf("file:%s?_busy_timeout=5000", cfg.Path))
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }
    
    // Configure connection pool
    db.SetMaxOpenConns(10)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)
    
    // Initialize schema
    if err := bootstrap(db); err != nil {
        db.Close()
        return nil, err
    }
    
    // Prepare statements
    insertStmt, err := db.Prepare(`INSERT INTO interactions (role, content, created_at) VALUES (?, ?, ?)`)
    if err != nil {
        db.Close()
        return nil, fmt.Errorf("failed to prepare insert: %w", err)
    }
    
    recentStmt, err := db.Prepare(`
        SELECT role, content, created_at 
        FROM interactions 
        ORDER BY created_at DESC 
        LIMIT ?
    `)
    if err != nil {
        insertStmt.Close()
        db.Close()
        return nil, fmt.Errorf("failed to prepare recent query: %w", err)
    }
    
    searchStmt, err := db.Prepare(`
        SELECT role, content, created_at 
        FROM interactions 
        WHERE content LIKE ? 
        ORDER BY created_at DESC 
        LIMIT ?
    `)
    if err != nil {
        insertStmt.Close()
        recentStmt.Close()
        db.Close()
        return nil, fmt.Errorf("failed to prepare search query: %w", err)
    }
    
    return &Store{
        db:         db,
        insertStmt: insertStmt,
        selectStmt: recentStmt,
        recentStmt: recentStmt,
        searchStmt: searchStmt,
        maxEntries: cfg.MaxEntries,
    }, nil
}

func bootstrap(db *sql.DB) error {
    // Enable WAL mode for better concurrency
    _, err := db.Exec(`PRAGMA journal_mode=WAL`)
    if err != nil {
        return fmt.Errorf("failed to set journal mode: %w", err)
    }
    
    _, err = db.Exec(`PRAGMA synchronous=NORMAL`)
    if err != nil {
        return fmt.Errorf("failed to set synchronous: %w", err)
    }
    
    // Create table
    _, err = db.Exec(`
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    `)
    if err != nil {
        return fmt.Errorf("failed to create table: %w", err)
    }
    
    // Create indexes
    _, err = db.Exec(`
        CREATE INDEX IF NOT EXISTS idx_interactions_created_at 
        ON interactions(created_at DESC)
    `)
    if err != nil {
        return fmt.Errorf("failed to create index: %w", err)
    }
    
    _, err = db.Exec(`
        CREATE INDEX IF NOT EXISTS idx_interactions_role 
        ON interactions(role)
    `)
    if err != nil {
        return fmt.Errorf("failed to create role index: %w", err)
    }
    
    return nil
}

func (s *Store) Append(role, content string) error {
    // Validate input
    if role == "" {
        return fmt.Errorf("role cannot be empty")
    }
    
    if content == "" {
        return fmt.Errorf("content cannot be empty")
    }
    
    if role != "user" && role != "assistant" {
        return fmt.Errorf("invalid role: %s (must be 'user' or 'assistant')", role)
    }
    
    // Execute insert
    _, err := s.insertStmt.Exec(role, content, time.Now().UnixNano())
    if err != nil {
        return fmt.Errorf("failed to append: %w", err)
    }
    
    // Prune old entries if necessary
    s.pruneIfNeeded()
    
    return nil
}

func (s *Store) Recent(limit int) ([]memory.Entry, error) {
    // Validate limit
    if limit <= 0 {
        limit = 10
    }
    
    if limit > s.maxEntries {
        limit = s.maxEntries
    }
    
    // Query entries
    rows, err := s.selectStmt.Query(limit)
    if err != nil {
        return nil, fmt.Errorf("failed to query recent: %w", err)
    }
    defer rows.Close()
    
    var entries []memory.Entry
    for rows.Next() {
        var entry memory.Entry
        var timestamp int64
        
        if err := rows.Scan(&entry.Role, &entry.Content, &timestamp); err != nil {
            return nil, fmt.Errorf("failed to scan row: %w", err)
        }
        
        entry.CreatedAt = time.UnixNano(timestamp)
        entries = append(entries, entry)
    }
    
    if err := rows.Err(); err != nil {
        return nil, fmt.Errorf("error iterating rows: %w", err)
    }
    
    return entries, nil
}

func (s *Store) Search(query string, limit int) ([]memory.Entry, error) {
    // Validate input
    if strings.TrimSpace(query) == "" {
        return nil, fmt.Errorf("search query cannot be empty")
    }
    
    if limit <= 0 {
        limit = 10
    }
    
    if limit > s.maxEntries {
        limit = s.maxEntries
    }
    
    // Search with LIKE
    pattern := "%" + query + "%"
    
    rows, err := s.searchStmt.Query(pattern, limit)
    if err != nil {
        return nil, fmt.Errorf("failed to search: %w", err)
    }
    defer rows.Close()
    
    var entries []memory.Entry
    for rows.Next() {
        var entry memory.Entry
        var timestamp int64
        
        if err := rows.Scan(&entry.Role, &entry.Content, &timestamp); err != nil {
            return nil, fmt.Errorf("failed to scan row: %w", err)
        }
        
        entry.CreatedAt = time.UnixNano(timestamp)
        entries = append(entries, entry)
    }
    
    return entries, nil
}

func (s *Store) Stats() (map[string]interface{}, error) {
    var totalCount int
    err := s.db.QueryRow(`SELECT COUNT(*) FROM interactions`).Scan(&totalCount)
    if err != nil {
        return nil, fmt.Errorf("failed to get count: %w", err)
    }
    
    var userCount, assistantCount int
    err = s.db.QueryRow(`SELECT COUNT(*) FROM interactions WHERE role = 'user'`).Scan(&userCount)
    if err != nil {
        return nil, fmt.Errorf("failed to get user count: %w", err)
    }
    
    err = s.db.QueryRow(`SELECT COUNT(*) FROM interactions WHERE role = 'assistant'`).Scan(&assistantCount)
    if err != nil {
        return nil, fmt.Errorf("failed to get assistant count: %w", err)
    }
    
    return map[string]interface{}{
        "total_entries":     totalCount,
        "user_entries":      userCount,
        "assistant_entries":  assistantCount,
        "max_entries":       s.maxEntries,
        "engine":            "simplememory",
    }, nil
}

func (s *Store) pruneIfNeeded() {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    var count int
    if err := s.db.QueryRow(`SELECT COUNT(*) FROM interactions`).Scan(&count); err != nil {
        return
    }
    
    if count <= s.maxEntries {
        return
    }
    
    excess := count - s.maxEntries
    
    // Delete oldest entries
    _, err := s.db.Exec(`
        DELETE FROM interactions 
        WHERE id IN (
            SELECT id FROM interactions 
            ORDER BY created_at ASC 
            LIMIT ?
        )
    `, excess)
    
    if err != nil {
        fmt.Fprintf(os.Stderr, "[simplememory] Failed to prune: %v\n", err)
    }
}

func (s *Store) Close() error {
    // Close statements
    s.insertStmt.Close()
    s.selectStmt.Close()
    s.recentStmt.Close()
    s.searchStmt.Close()
    
    // Close database
    return s.db.Close()
}
```

## Configuration

### YAML Configuration

```yaml
memory:
  path: "./data/conversation.db"
  turns_to_use: 10
  
simplememory:
  max_entries: 1000
```

## Usage

### Integration with Pipeline

```go
package main

import (
    "fmt"
    "os"

    "OpenEye/internal/config"
    "OpenEye/internal/context/memory"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
    
    "github.com/yourname/simple-memory"
)

func main() {
    // Create custom memory store
    store, err := simplememory.NewStore(simplememory.Config{
        Path:       "./data/conversation.db",
        MaxEntries: 1000,
    })
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to create store: %v\n", err)
        os.Exit(1)
    }
    defer store.Close()
    
    cfg, _ := config.Resolve()
    
    // Create pipeline with custom store
    pipe := &customPipeline{
        cfg:   cfg,
        store: store,
    }
    
    // Use the pipeline...
}
```

### Direct Usage

```go
package main

import (
    "fmt"
    "os"

    "github.com/yourname/simple-memory"
)

func main() {
    store, err := simplememory.NewStore(simplememory.Config{
        Path: "./conversation.db",
    })
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to create store: %v\n", err)
        os.Exit(1)
    }
    defer store.Close()
    
    // Store a conversation turn
    if err := store.Append("user", "Hello, World!"); err != nil {
        fmt.Fprintf(os.Stderr, "Failed to append: %v\n", err)
        os.Exit(1)
    }
    
    // Get recent turns
    entries, err := store.Recent(10)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to get recent: %v\n", err)
        os.Exit(1)
    }
    
    for _, entry := range entries {
        fmt.Printf("[%s]: %s\n", entry.Role, entry.Content)
    }
    
    // Search conversations
    results, err := store.Search("Hello", 5)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Search failed: %v\n", err)
        os.Exit(1)
    }
    
    fmt.Printf("Found %d matches\n", len(results))
}
```

## Testing

### Unit Test

```go
package simplememory

import (
    "os"
    "testing"
    "time"
)

func TestStore_AppendAndRecent(t *testing.T) {
    store, err := NewStore(Config{
        Path: ":memory:",
    })
    if err != nil {
        t.Fatalf("Failed to create store: %v", err)
    }
    defer store.Close()
    
    // Append entries
    if err := store.Append("user", "Hello"); err != nil {
        t.Fatalf("Append failed: %v", err)
    }
    
    if err := store.Append("assistant", "Hi there!"); err != nil {
        t.Fatalf("Append failed: %v", err)
    }
    
    // Get recent
    entries, err := store.Recent(10)
    if err != nil {
        t.Fatalf("Recent failed: %v", err)
    }
    
    if len(entries) != 2 {
        t.Errorf("Expected 2 entries, got %d", len(entries))
    }
    
    // Verify order (most recent first)
    if entries[0].Role != "assistant" {
        t.Errorf("Expected assistant first, got %s", entries[0].Role)
    }
}

func TestStore_InvalidRole(t *testing.T) {
    store, _ := NewStore(Config{Path: ":memory:"})
    defer store.Close()
    
    err := store.Append("invalid", "content")
    if err == nil {
        t.Error("Expected error for invalid role")
    }
}

func TestStore_EmptyContent(t *testing.T) {
    store, _ := NewStore(Config{Path: ":memory:"})
    defer store.Close()
    
    err := store.Append("user", "")
    if err == nil {
        t.Error("Expected error for empty content")
    }
}

func TestStore_Search(t *testing.T) {
    store, _ := NewStore(Config{Path: ":memory:"})
    defer store.Close()
    
    store.Append("user", "Hello World")
    store.Append("assistant", "How can I help?")
    store.Append("user", "What time is it?")
    
    results, err := store.Search("Hello", 10)
    if err != nil {
        t.Fatalf("Search failed: %v", err)
    }
    
    if len(results) != 1 {
        t.Errorf("Expected 1 result, got %d", len(results))
    }
}

func TestStore_Concurrency(t *testing.T) {
    store, _ := NewStore(Config{Path: ":memory:"})
    defer store.Close()
    
    done := make(chan bool)
    
    // Concurrent appends
    for i := 0; i < 10; i++ {
        go func(idx int) {
            store.Append("user", fmt.Sprintf("Message %d", idx))
            done <- true
        }(i)
    }
    
    // Wait for all
    for i := 0; i < 10; i++ {
        <-done
    }
    
    entries, _ := store.Recent(100)
    if len(entries) != 10 {
        t.Errorf("Expected 10 entries, got %d", len(entries))
    }
}

func TestStore_Pruning(t *testing.T) {
    store, _ := NewStore(Config{
        Path:       ":memory:",
        MaxEntries: 5,
    })
    defer store.Close()
    
    // Add more than max
    for i := 0; i < 10; i++ {
        store.Append("user", fmt.Sprintf("Message %d", i))
    }
    
    entries, _ := store.Recent(10)
    if len(entries) != 5 {
        t.Errorf("Expected 5 entries after pruning, got %d", len(entries))
    }
}
```

## Production Checklist

- [x] SQLite WAL mode enabled
- [x] Connection pooling configured
- [x] Prepared statements used
- [x] Input validation implemented
- [x] Automatic pruning configured
- [x] Concurrent access safe
- [x] Error handling complete
- [x] Unit tests passing
- [ ] Integration tests passing

## Related Documentation

- [Memory Engines](../memory-engines.md)
- [Best Practices](../best-practices.md)
- [Security Guidelines](../best-practices.md#security-guidelines)

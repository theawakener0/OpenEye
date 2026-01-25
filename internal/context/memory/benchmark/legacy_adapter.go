package benchmark

import (
	"OpenEye/internal/context/memory"
	"context"
	"fmt"
)

// LegacyAdapter implements MemorySystemAdapter for the legacy SQLite system.
type LegacyAdapter struct {
	store  *memory.Store
	dbPath string
}

func NewLegacyAdapter(dbPath string) *LegacyAdapter {
	return &LegacyAdapter{
		dbPath: dbPath,
	}
}

func (a *LegacyAdapter) Name() string {
	return "legacy_sqlite"
}

func (a *LegacyAdapter) Initialize(ctx context.Context) error {
	var err error
	a.store, err = memory.NewStore(a.dbPath)
	return err
}

func (a *LegacyAdapter) Store(ctx context.Context, role, content string, turnIndex int) error {
	if a.store == nil {
		return fmt.Errorf("store not initialized")
	}
	return a.store.Append(role, content)
}

func (a *LegacyAdapter) Retrieve(ctx context.Context, query string, limit int) ([]string, error) {
	if a.store == nil {
		return nil, fmt.Errorf("store not initialized")
	}
	entries, err := a.store.Recent(limit)
	if err != nil {
		return nil, err
	}
	var results []string
	for _, e := range entries {
		results = append(results, e.Content)
	}
	return results, nil
}

func (a *LegacyAdapter) BuildContext(ctx context.Context, query string, maxTokens int) (string, int, error) {
	if a.store == nil {
		return "", 0, fmt.Errorf("store not initialized")
	}
	// Legacy doesn't have sophisticated context building, just takes recent
	entries, err := a.store.Recent(10)
	if err != nil {
		return "", 0, err
	}
	var context string
	for _, e := range entries {
		context += fmt.Sprintf("%s: %s\n", e.Role, e.Content)
	}
	return context, len(context) / 4, nil // Rough estimate
}

func (a *LegacyAdapter) GetStats(ctx context.Context) (map[string]interface{}, error) {
	return map[string]interface{}{"type": "sqlite"}, nil
}

func (a *LegacyAdapter) Close() error {
	if a.store != nil {
		return a.store.Close()
	}
	return nil
}

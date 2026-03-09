package omem

import (
	"context"
	"path/filepath"
	"testing"
)

func TestIVFPQIndexPersistsVectorsAndAssignments(t *testing.T) {
	ctx := context.Background()
	indexPath := filepath.Join(t.TempDir(), "test.ivfpq")
	cfg := ANNConfig{
		Enabled:          true,
		Backend:          "ivfpq",
		IndexPath:        indexPath,
		FallbackToScan:   true,
		MinFactsToEnable: 2,
		OversampleFactor: 4,
		ExactRerankLimit: 4,
		NList:            2,
		NProbe:           1,
		PQSubvectors:     2,
		PQBits:           2,
		TrainMinFacts:    2,
	}

	index, err := newIVFPQIndex(cfg, 4, true, "test-model")
	if err != nil {
		t.Fatalf("newIVFPQIndex failed: %v", err)
	}
	if err := index.Upsert(ctx, 1, []float32{1, 0, 0, 0}); err != nil {
		t.Fatalf("upsert 1 failed: %v", err)
	}
	if err := index.Upsert(ctx, 2, []float32{0, 1, 0, 0}); err != nil {
		t.Fatalf("upsert 2 failed: %v", err)
	}
	if err := index.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	reloaded, err := newIVFPQIndex(cfg, 4, true, "test-model")
	if err != nil {
		t.Fatalf("reload failed: %v", err)
	}
	stats, err := reloaded.Stats(ctx)
	if err != nil {
		t.Fatalf("stats failed: %v", err)
	}
	if got := stats["fact_count"].(int); got != 2 {
		t.Fatalf("fact_count = %d, want 2", got)
	}
	results, err := reloaded.Search(ctx, []float32{1, 0, 0, 0}, 1, 4, 4)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatalf("expected ANN results after reload")
	}
}

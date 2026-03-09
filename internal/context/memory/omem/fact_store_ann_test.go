package omem

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestFactStoreSemanticSearchUsesANNAndReranksExactly(t *testing.T) {
	ctx := context.Background()
	store, err := NewFactStore(StorageConfig{DBPath: filepath.Join(t.TempDir(), "omem.duckdb")})
	if err != nil {
		t.Fatalf("NewFactStore failed: %v", err)
	}
	defer store.Close()

	annCfg := ANNConfig{
		Enabled:          true,
		Backend:          "ivfpq",
		IndexPath:        filepath.Join(t.TempDir(), "omem.ivfpq"),
		FallbackToScan:   true,
		MinFactsToEnable: 2,
		OversampleFactor: 4,
		ExactRerankLimit: 4,
		NList:            2,
		NProbe:           2,
		PQSubvectors:     2,
		PQBits:           2,
		TrainMinFacts:    2,
	}
	annIndex, err := newIVFPQIndex(annCfg, 4, true, "test-model")
	if err != nil {
		t.Fatalf("newIVFPQIndex failed: %v", err)
	}
	store.SetANNIndex(annIndex)
	store.SetANNConfig(annCfg)

	now := time.Now()
	factA := Fact{Text: "A", AtomicText: "A", Importance: 0.9, CreatedAt: now, Embedding: []float32{1, 0, 0, 0}}
	factB := Fact{Text: "B", AtomicText: "B", Importance: 0.9, CreatedAt: now, Embedding: []float32{0.9, 0.1, 0, 0}}
	if _, err := store.InsertFact(ctx, factA); err != nil {
		t.Fatalf("insert A failed: %v", err)
	}
	if _, err := store.InsertFact(ctx, factB); err != nil {
		t.Fatalf("insert B failed: %v", err)
	}

	results, err := store.SemanticSearch(ctx, []float32{1, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("SemanticSearch failed: %v", err)
	}
	if len(results) < 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Fact.AtomicText != "A" {
		t.Fatalf("top result = %q, want A", results[0].Fact.AtomicText)
	}
	if results[0].Score < results[1].Score {
		t.Fatalf("results not exact-reranked: %f < %f", results[0].Score, results[1].Score)
	}
}

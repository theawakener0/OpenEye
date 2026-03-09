package omem

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type semanticRecallSnapshot struct {
	Facts            int       `json:"facts"`
	Limit            int       `json:"limit"`
	BruteForceRecall float64   `json:"brute_force_recall"`
	ANNRecall        float64   `json:"ann_recall"`
	RecallAtK        float64   `json:"recall_at_k"`
	AverageOverlap   float64   `json:"average_overlap"`
	ANNConfig        ANNConfig `json:"ann_config"`
	GeneratedAt      time.Time `json:"generated_at"`
}

func BenchmarkSemanticSearchBruteForce(b *testing.B) {
	for _, size := range []int{1000, 5000} {
		b.Run(fmt.Sprintf("facts=%d", size), func(b *testing.B) {
			ctx := context.Background()
			store, _, queries := buildBenchmarkStore(b, size, false, ANNConfig{})
			defer store.Close()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				query := queries[i%len(queries)]
				results, err := store.SemanticSearch(ctx, query, 10)
				if err != nil {
					b.Fatalf("SemanticSearch failed: %v", err)
				}
				if len(results) == 0 {
					b.Fatal("expected results")
				}
			}
		})
	}
}

func BenchmarkSemanticSearchANN(b *testing.B) {
	configs := []struct {
		name string
		cfg  ANNConfig
	}{
		{
			name: "nlist=32_nprobe=4_pq=16x4",
			cfg:  benchmarkANNConfig(32, 4, 16, 4),
		},
		{
			name: "nlist=64_nprobe=6_pq=24x4",
			cfg:  benchmarkANNConfig(64, 6, 24, 4),
		},
	}

	for _, size := range []int{5000, 10000} {
		for _, config := range configs {
			b.Run(fmt.Sprintf("facts=%d/%s", size, config.name), func(b *testing.B) {
				ctx := context.Background()
				store, _, queries := buildBenchmarkStore(b, size, true, config.cfg)
				defer store.Close()

				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					query := queries[i%len(queries)]
					results, err := store.SemanticSearch(ctx, query, 10)
					if err != nil {
						b.Fatalf("SemanticSearch failed: %v", err)
					}
					if len(results) == 0 {
						b.Fatal("expected results")
					}
				}
			})
		}
	}
}

func BenchmarkSemanticSearchANNTuning(b *testing.B) {
	for _, nprobe := range []int{2, 4, 6, 8} {
		cfg := benchmarkANNConfig(64, nprobe, 24, 4)
		b.Run(fmt.Sprintf("nprobe=%d", nprobe), func(b *testing.B) {
			ctx := context.Background()
			store, baselineQueries, queries := buildBenchmarkStore(b, 10000, true, cfg)
			defer store.Close()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				query := queries[i%len(queries)]
				results, err := store.SemanticSearch(ctx, query, 10)
				if err != nil {
					b.Fatalf("SemanticSearch failed: %v", err)
				}
				if len(results) == 0 {
					b.Fatal("expected results")
				}
				if len(baselineQueries) > 0 && len(results) > 0 {
					_ = baselineQueries[i%len(baselineQueries)]
				}
			}
		})
	}
}

func TestSemanticSearchANNRecallAgainstBruteForce(t *testing.T) {
	ctx := context.Background()
	annCfg := benchmarkANNConfig(64, 6, 24, 4)
	store, facts, queries := buildBenchmarkStoreForTest(t, 5000, true, annCfg)
	defer store.Close()

	limit := 10
	totalOverlap := 0.0
	totalRecall := 0.0
	for _, query := range queries {
		annResults, err := store.SemanticSearch(ctx, query, limit)
		if err != nil {
			t.Fatalf("ANN SemanticSearch failed: %v", err)
		}
		bruteResults := bruteForceResults(query, facts, limit)
		overlap, recall := compareResultSets(annResults, bruteResults, limit)
		totalOverlap += overlap
		totalRecall += recall
	}

	avgOverlap := totalOverlap / float64(len(queries))
	avgRecall := totalRecall / float64(len(queries))
	if avgRecall < 0.70 {
		t.Fatalf("ANN recall too low: %.3f", avgRecall)
	}

	writeSemanticRecallSnapshot(t, semanticRecallSnapshot{
		Facts:            len(facts),
		Limit:            limit,
		BruteForceRecall: 1.0,
		ANNRecall:        avgRecall,
		RecallAtK:        avgRecall,
		AverageOverlap:   avgOverlap,
		ANNConfig:        annCfg,
		GeneratedAt:      time.Now(),
	})
}

func benchmarkANNConfig(nlist, nprobe, pqSubvectors, pqBits int) ANNConfig {
	return ANNConfig{
		Enabled:          true,
		Backend:          "ivfpq",
		FallbackToScan:   true,
		MinFactsToEnable: 2000,
		OversampleFactor: 4,
		ExactRerankLimit: 64,
		NList:            nlist,
		NProbe:           nprobe,
		PQSubvectors:     pqSubvectors,
		PQBits:           pqBits,
		TrainMinFacts:    1000,
	}
}

func buildBenchmarkStore(b *testing.B, factCount int, enableANN bool, annCfg ANNConfig) (*FactStore, []Fact, [][]float32) {
	b.Helper()
	factsStore, facts, queries := buildBenchmarkStoreCore(b.TempDir(), factCount, enableANN, annCfg, func(format string, args ...interface{}) {
		b.Fatalf(format, args...)
	})
	return factsStore, facts, queries
}

func buildBenchmarkStoreForTest(t *testing.T, factCount int, enableANN bool, annCfg ANNConfig) (*FactStore, []Fact, [][]float32) {
	t.Helper()
	return buildBenchmarkStoreCore(t.TempDir(), factCount, enableANN, annCfg, func(format string, args ...interface{}) {
		t.Fatalf(format, args...)
	})
}

func buildBenchmarkStoreCore(tempDir string, factCount int, enableANN bool, annCfg ANNConfig, failf func(format string, args ...interface{})) (*FactStore, []Fact, [][]float32) {
	ctx := context.Background()
	store, err := NewFactStore(StorageConfig{DBPath: filepath.Join(tempDir, "omem_bench.duckdb")})
	if err != nil {
		failf("NewFactStore failed: %v", err)
	}

	facts := make([]Fact, 0, factCount)
	queries := make([][]float32, 0, 32)
	now := time.Now()
	for i := 0; i < factCount; i++ {
		emb := syntheticBenchmarkEmbedding(i, 384)
		fact := Fact{
			Text:       fmt.Sprintf("fact-%d", i),
			AtomicText: fmt.Sprintf("fact-%d", i),
			Category:   CategoryKnowledge,
			Importance: 0.5 + float64(i%5)*0.1,
			Embedding:  emb,
			CreatedAt:  now.Add(-time.Duration(i) * time.Minute),
		}
		id, err := store.InsertFact(ctx, fact)
		if err != nil {
			store.Close()
			failf("InsertFact failed: %v", err)
		}
		fact.ID = id
		facts = append(facts, fact)
		if i < 32 {
			queries = append(queries, syntheticBenchmarkQuery(i, 384))
		}
	}

	if enableANN {
		annCfg.IndexPath = filepath.Join(tempDir, "omem_bench.ivfpq")
		annIndex, err := newIVFPQIndex(annCfg, 384, true, "benchmark-model")
		if err != nil {
			store.Close()
			failf("newIVFPQIndex failed: %v", err)
		}
		if err := annIndex.Rebuild(ctx, facts); err != nil {
			store.Close()
			failf("ANN rebuild failed: %v", err)
		}
		store.SetANNIndex(annIndex)
		store.SetANNConfig(annCfg)
	}

	return store, facts, queries
}

func syntheticBenchmarkEmbedding(seed int, dim int) []float32 {
	vec := make([]float32, dim)
	base := seed % 64
	for i := 0; i < dim; i++ {
		angle := float64((base + i) % 32)
		vec[i] = float32(math.Sin(angle*0.17) + math.Cos(float64(seed%17+i)*0.11))
	}
	return normalizeVectorF32(vec)
}

func syntheticBenchmarkQuery(seed int, dim int) []float32 {
	vec := syntheticBenchmarkEmbedding(seed, dim)
	for i := 0; i < len(vec); i += 31 {
		vec[i] += 0.03
	}
	return normalizeVectorF32(vec)
}

func bruteForceResults(query []float32, facts []Fact, limit int) []ScoredFact {
	queryVec := normalizeVectorF32(query)
	results := make([]ScoredFact, 0, len(facts))
	for _, fact := range facts {
		score := cosineSimilarityOptimized(queryVec, fact.Embedding)
		results = append(results, ScoredFact{Fact: fact, SemanticScore: score, Score: score})
	}
	sortScoredFacts(results)
	if len(results) > limit {
		results = results[:limit]
	}
	return results
}

func compareResultSets(actual, expected []ScoredFact, limit int) (overlap float64, recall float64) {
	if len(expected) == 0 {
		return 1.0, 1.0
	}
	if len(actual) > limit {
		actual = actual[:limit]
	}
	if len(expected) > limit {
		expected = expected[:limit]
	}
	expectedSet := make(map[int64]struct{}, len(expected))
	for _, item := range expected {
		expectedSet[item.Fact.ID] = struct{}{}
	}
	matches := 0
	for _, item := range actual {
		if _, ok := expectedSet[item.Fact.ID]; ok {
			matches++
		}
	}
	recall = float64(matches) / float64(len(expected))
	overlap = recall
	return overlap, recall
}

func sortScoredFacts(results []ScoredFact) {
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
}

func writeSemanticRecallSnapshot(t *testing.T, snapshot semanticRecallSnapshot) {
	t.Helper()
	outputDir := filepath.Join("testdata", "benchmarks")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir benchmark snapshot dir: %v", err)
	}
	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	filePath := filepath.Join(outputDir, "semantic_search_ann_recall.json")
	if err := os.WriteFile(filePath, data, 0o644); err != nil {
		t.Fatalf("write snapshot: %v", err)
	}
}

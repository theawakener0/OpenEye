package benchmark

import (
	"OpenEye/internal/config"
	"OpenEye/internal/context/memory/omem"
	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
	"context"
	"fmt"
	"os"
	"testing"
)

func TestGenerateConversation(t *testing.T) {
	gen := NewGenerator(DefaultGeneratorConfig())
	conv := gen.GenerateConversation()

	if len(conv.Turns) != DefaultGeneratorConfig().NumTurns*2 {
		t.Errorf("Expected %d turns, got %d", DefaultGeneratorConfig().NumTurns*2, len(conv.Turns))
	}

	if len(conv.PlantedFacts) != DefaultGeneratorConfig().PlantedFactsCount {
		t.Errorf("Expected %d planted facts, got %d", DefaultGeneratorConfig().PlantedFactsCount, len(conv.PlantedFacts))
	}

	t.Logf("Generated persona: %s (%s)", conv.Persona.Name, conv.Persona.Occupation)
	for _, fact := range conv.PlantedFacts {
		t.Logf("Planted fact: %s (Turn %d)", fact.Fact, fact.TurnIndex)
	}
}

func TestComplexityCorpus(t *testing.T) {
	stats := GetCorpusStats()
	if stats["total"] != 100 {
		t.Errorf("Expected 100 queries, got %d", stats["total"])
	}
	t.Logf("Corpus stats: %v", stats)
}

// TestBenchmarkSuite is the entry point for the full benchmark suite.
func TestBenchmarkSuite(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping benchmark suite in short mode")
	}

	ctx := context.Background()
	config := DefaultBenchmarkConfig()
	config.NumTurns = 10 // Reduced for testing
	config.NumRecallTests = 3

	// Setup systems
	legacy := NewLegacyAdapter("test_legacy.db")
	omemFull := NewOmemAdapter(omem.AblationFull, nil, nil)
	omemMin := NewOmemAdapter(omem.AblationMinimal, nil, nil)

	runner := NewBenchmarkRunner(config, legacy, omemFull, omemMin)
	results, err := runner.Run(ctx)
	if err != nil {
		t.Fatalf("Benchmark failed: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	for _, res := range results {
		t.Logf("System: %s, Recall: %.2f, Avg Write: %.2fus",
			res.SystemName, res.RecallMetrics.RecallAccuracy, res.WriteLatency.Mean)
	}
}

// TestHardwareBenchmark runs the benchmark with real models if configured.
// Run with: USE_REAL_MODELS=true go test -v -run TestHardwareBenchmark
func TestHardwareBenchmark(t *testing.T) {
	if os.Getenv("USE_REAL_MODELS") != "true" {
		t.Skip("Skipping hardware benchmark (USE_REAL_MODELS not set to true)")
	}

	ctx := context.Background()
	
	// Load real configuration
	cfg, err := config.Resolve()
	if err != nil {
		t.Fatalf("Failed to resolve config: %v", err)
	}

	// Initialize real models
	fmt.Println("Initializing real models for hardware benchmark...")
	manager := runtime.MustManager(cfg.Runtime)
	embedder, err := embedding.New(cfg.Embedding)
	if err != nil {
		t.Fatalf("Failed to initialize embedder: %v", err)
	}
	defer func() {
		if embedder != nil {
			embedder.Close()
		}
		manager.Close()
	}()

	benchmarkCfg := DefaultBenchmarkConfig()
	benchmarkCfg.NumTurns = 20
	benchmarkCfg.NumRecallTests = 5
	benchmarkCfg.SaveRawMeasurements = true

	// Setup systems
	legacy := NewLegacyAdapter("hardware_legacy.db")
	omemFull := NewOmemAdapter(omem.AblationFull, manager, embedder)
	omemMin := NewOmemAdapter(omem.AblationMinimal, manager, embedder)

	runner := NewBenchmarkRunner(benchmarkCfg, legacy, omemFull, omemMin)
	results, err := runner.Run(ctx)
	if err != nil {
		t.Fatalf("Hardware benchmark failed: %v", err)
	}

	for _, res := range results {
		fmt.Printf("System: %s, Recall: %.2f, Avg Write: %.2fms, Peak Memory: %.2fMB\n",
			res.SystemName, res.RecallMetrics.RecallAccuracy, res.WriteLatency.Mean/1000, 
			float64(res.MemoryMetrics.PeakHeapBytes)/(1024*1024))
	}
}

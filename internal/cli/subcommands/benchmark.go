package subcommands

import (
	"OpenEye/internal/config"
	"OpenEye/internal/context/memory/benchmark"
	"OpenEye/internal/context/memory/omem"
	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
	"context"
	"flag"
	"fmt"
	"os"
)

// RunBenchmark executes the memory benchmark suite from the CLI.
func RunBenchmark(cfg config.Config, args []string) int {
	fs := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	numTurns := fs.Int("turns", 20, "Number of conversation turns to simulate")
	numRecall := fs.Int("recall", 5, "Number of facts to plant and test for recall")
	realModels := fs.Bool("real", false, "Use real LLM and embedding models instead of mocks")
	outputDir := fs.String("output", "benchmark_results", "Directory to save results")
	
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse flags: %v\n", err)
		return 1
	}

	ctx := context.Background()
	
	benchmarkCfg := benchmark.DefaultBenchmarkConfig()
	benchmarkCfg.NumTurns = *numTurns
	benchmarkCfg.NumRecallTests = *numRecall
	benchmarkCfg.OutputDir = *outputDir

	var manager *runtime.Manager
	var embedder embedding.Provider
	var err error

	if *realModels {
		fmt.Printf("Initializing real models (Backend: %s, Embedding BaseURL: %s)...\n", 
			cfg.Runtime.Backend, cfg.Embedding.LlamaCpp.BaseURL)
		manager = runtime.MustManager(cfg.Runtime)
		embedder, err = embedding.New(cfg.Embedding)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to initialize embedder: %v\n", err)
			return 1
		}
	} else {
		fmt.Println("Using mock models (use --real for actual benchmarking on target hardware).")
	}

	// Setup systems
	systems := []benchmark.MemorySystemAdapter{
		benchmark.NewLegacyAdapter("benchmark_legacy.db"),
		benchmark.NewOmemAdapter(omem.AblationFull, manager, embedder),
		benchmark.NewOmemAdapter(omem.AblationMinimal, manager, embedder),
		benchmark.NewOmemAdapter(omem.AblationSemanticOnly, manager, embedder),
	}

	runner := benchmark.NewBenchmarkRunner(benchmarkCfg, systems...)
	
	fmt.Printf("Starting benchmark with %d turns and %d recall tests...\n", *numTurns, *numRecall)
	results, err := runner.Run(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Benchmark failed: %v\n", err)
		return 1
	}

	fmt.Println("\n\033[1;32mBenchmark Complete!\033[0m")
	for _, res := range results {
		fmt.Printf("\nSystem: %s\n", res.SystemName)
		fmt.Printf("  Recall Accuracy: %.1f%%\n", res.RecallMetrics.RecallAccuracy*100)
		fmt.Printf("  Avg Write Latency: %.2fms\n", res.WriteLatency.Mean/1000)
		fmt.Printf("  Avg Retrieve Latency: %.2fms\n", res.RetrieveLatency.Mean/1000)
		fmt.Printf("  Peak Heap: %.2f MB\n", float64(res.MemoryMetrics.PeakHeapBytes)/(1024*1024))
	}

	return 0
}

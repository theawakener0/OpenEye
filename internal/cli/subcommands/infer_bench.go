package subcommands

import (
	"context"
	"flag"
	"fmt"
	"os"

	"OpenEye/internal/config"
	"OpenEye/internal/inferbench"
	"OpenEye/internal/runtime"
)

// RunInferBench executes the inference benchmark suite from the CLI.
func RunInferBench(cfg config.Config, registry runtime.Registry, args []string) int {
	fs := flag.NewFlagSet("infer-bench", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)

	// Standard benchmark flags.
	iterations := fs.Int("iterations", 5, "Number of iterations per prompt")
	maxTokens := fs.Int("max-tokens", 128, "Maximum tokens to generate per request")
	warmup := fs.Int("warmup", 1, "Warmup iterations (not recorded)")
	output := fs.String("output", "", "Path to save JSON results (optional)")
	prompt := fs.String("prompt", "", "Custom prompt to benchmark (uses standard set if empty)")
	verbose := fs.Bool("verbose", false, "Print per-iteration details")

	// Optimization toggle flags for A/B testing.
	kvCacheType := fs.String("kv-cache-type", "", "Override KV cache type (f16, q8_0, q4_0)")
	noSpeculative := fs.Bool("no-speculative", false, "Disable speculative decoding for this run")
	speculativeN := fs.Int("speculative-n", 0, "Override number of draft tokens (0=use config)")
	streamChunkSize := fs.Int("stream-chunk-size", 0, "Override stream chunk size (0=use config)")
	noContextShift := fs.Bool("no-context-shift", false, "Disable context window shifting")
	stream := fs.Bool("stream", false, "Benchmark Stream() instead of Generate()")
	compare := fs.Bool("compare", false, "Run baseline (no optimizations) then optimized, print delta")

	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse flags: %v\n", err)
		return 1
	}

	// Apply optimization overrides to config before creating adapter.
	if *kvCacheType != "" {
		cfg.Runtime.Native.KVCacheType = *kvCacheType
	}
	if *noSpeculative {
		cfg.Runtime.Native.DraftModelPath = ""
		cfg.Runtime.Native.SpeculativeN = 0
	}
	if *speculativeN > 0 {
		cfg.Runtime.Native.SpeculativeN = *speculativeN
	}
	if *streamChunkSize > 0 {
		cfg.Runtime.Native.StreamChunkSize = *streamChunkSize
	}
	if *noContextShift {
		f := false
		cfg.Runtime.Native.ContextShift = &f
	}

	if *compare {
		return runComparison(cfg, registry, *iterations, *maxTokens, *warmup, *output, *prompt, *verbose, *stream)
	}

	return runSingle(cfg, registry, *iterations, *maxTokens, *warmup, *output, *prompt, *verbose, *stream)
}

// runSingle executes a single benchmark run with the (possibly overridden) config.
func runSingle(cfg config.Config, registry runtime.Registry, iterations, maxTokens, warmup int, output, prompt string, verbose, useStream bool) int {
	backend := cfg.Runtime.Backend
	if backend == "" {
		backend = "http"
	}
	factory, ok := registry[backend]
	if !ok {
		fmt.Fprintf(os.Stderr, "runtime backend %q not registered\n", backend)
		return 1
	}
	adapter, err := factory(cfg.Runtime)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create adapter: %v\n", err)
		return 1
	}
	defer adapter.Close()

	benchCfg := inferbench.DefaultConfig()
	benchCfg.Iterations = iterations
	benchCfg.MaxTokens = maxTokens
	benchCfg.WarmupIterations = warmup
	benchCfg.OutputPath = output
	benchCfg.Verbose = verbose
	benchCfg.UseStream = useStream

	if prompt != "" {
		benchCfg.Prompts = []inferbench.Prompt{
			{Name: "custom", Text: prompt},
		}
	}

	mode := "Generate"
	if useStream {
		mode = "Stream"
	}
	specInfo := ""
	if cfg.Runtime.Native.DraftModelPath != "" {
		specInfo = fmt.Sprintf("  speculative_n=%d", cfg.Runtime.Native.SpeculativeN)
	}

	fmt.Printf("OpenEye Inference Benchmark\n")
	fmt.Printf("Backend: %s  Mode: %s\n", backend, mode)
	fmt.Printf("Iterations: %d (warmup: %d)\n", benchCfg.Iterations, benchCfg.WarmupIterations)
	fmt.Printf("Max tokens: %d%s\n", benchCfg.MaxTokens, specInfo)
	if cfg.Runtime.Native.KVCacheType != "" {
		fmt.Printf("KV cache: %s\n", cfg.Runtime.Native.KVCacheType)
	}

	runner := inferbench.NewRunner(adapter, benchCfg)
	ctx := context.Background()

	report, err := runner.Run(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Benchmark failed: %v\n", err)
		return 1
	}

	printReport(report)
	return 0
}

// runComparison runs a baseline (no optimizations) and an optimized run, then
// prints a delta comparison table.
func runComparison(cfg config.Config, registry runtime.Registry, iterations, maxTokens, warmup int, output, prompt string, verbose, useStream bool) int {
	backend := cfg.Runtime.Backend
	if backend == "" {
		backend = "http"
	}
	factory, ok := registry[backend]
	if !ok {
		fmt.Fprintf(os.Stderr, "runtime backend %q not registered\n", backend)
		return 1
	}

	// --- Baseline run: strip optimizations ---
	fmt.Printf("=== BASELINE (no optimizations) ===\n")
	baselineCfg := cfg
	baselineCfg.Runtime.Native.DraftModelPath = ""
	baselineCfg.Runtime.Native.SpeculativeN = 0
	baselineCfg.Runtime.Native.KVCacheType = "f16"
	baselineCfg.Runtime.Native.StreamChunkSize = 1
	f := false
	baselineCfg.Runtime.Native.ContextShift = &f

	baseAdapter, err := factory(baselineCfg.Runtime)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create baseline adapter: %v\n", err)
		return 1
	}

	benchCfg := inferbench.DefaultConfig()
	benchCfg.Iterations = iterations
	benchCfg.MaxTokens = maxTokens
	benchCfg.WarmupIterations = warmup
	benchCfg.Verbose = verbose
	benchCfg.UseStream = useStream

	if prompt != "" {
		benchCfg.Prompts = []inferbench.Prompt{
			{Name: "custom", Text: prompt},
		}
	}

	baseRunner := inferbench.NewRunner(baseAdapter, benchCfg)
	ctx := context.Background()

	baseReport, err := baseRunner.Run(ctx)
	baseAdapter.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Baseline benchmark failed: %v\n", err)
		return 1
	}

	// --- Optimized run: use full config as-is ---
	fmt.Printf("\n=== OPTIMIZED ===\n")
	optAdapter, err := factory(cfg.Runtime)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create optimized adapter: %v\n", err)
		return 1
	}

	optRunner := inferbench.NewRunner(optAdapter, benchCfg)
	optReport, err := optRunner.Run(ctx)
	optAdapter.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Optimized benchmark failed: %v\n", err)
		return 1
	}

	// --- Print delta comparison ---
	fmt.Printf("\n=== COMPARISON (baseline vs optimized) ===\n")
	printComparison(baseReport, optReport)

	// Save optimized report if output path specified.
	if output != "" {
		benchCfg.OutputPath = output
		// Saved already by the runner, but we note it here.
		fmt.Printf("Optimized results saved to %s\n", output)
	}

	return 0
}

// printReport prints the final summary for a benchmark run.
func printReport(report *inferbench.BenchmarkReport) {
	fmt.Printf("\n=== Final Summary ===\n")
	for _, s := range report.Summaries {
		fmt.Printf("\n[%s]\n", s.Name)
		fmt.Printf("  TTFT:       avg=%v  p95=%v\n", s.TTFT.Mean, s.TTFT.P95)
		fmt.Printf("  Gen TPS:    avg=%.1f  p95=%.1f\n", s.GenerationTPS.Mean, s.GenerationTPS.P95)
		fmt.Printf("  Prompt TPS: avg=%.1f\n", s.PromptTPS.Mean)
		if s.PeakRSSBytes > 0 {
			fmt.Printf("  Peak RSS:   %.1f MB\n", float64(s.PeakRSSBytes)/(1024*1024))
		}
		if s.CacheHitImprove != 0 {
			fmt.Printf("  Cache TTFT: %.1f%% faster on repeat\n", s.CacheHitImprove)
		}
		if s.SpeculativeActive {
			fmt.Printf("  Spec:       acceptance=%.1f%%  drafted=%.0f/iter  accepted=%.0f/iter\n",
				s.SpecAcceptanceRate.Mean, s.AvgSpecDrafted, s.AvgSpecAccepted)
		}
	}
}

// printComparison prints a side-by-side delta table for two benchmark reports.
func printComparison(base, opt *inferbench.BenchmarkReport) {
	// Match summaries by name.
	optMap := make(map[string]inferbench.PromptSummary)
	for _, s := range opt.Summaries {
		optMap[s.Name] = s
	}

	fmt.Printf("%-15s  %12s  %12s  %10s\n", "Metric", "Baseline", "Optimized", "Delta")
	fmt.Printf("%-15s  %12s  %12s  %10s\n", "------", "--------", "---------", "-----")

	for _, bs := range base.Summaries {
		os, ok := optMap[bs.Name]
		if !ok {
			continue
		}

		fmt.Printf("\n[%s]\n", bs.Name)

		// TTFT
		ttftDelta := deltaPercent(float64(bs.TTFT.Mean), float64(os.TTFT.Mean))
		fmt.Printf("  %-13s  %12v  %12v  %+9.1f%%\n", "TTFT avg",
			bs.TTFT.Mean, os.TTFT.Mean, ttftDelta)

		// Gen TPS
		tpsDelta := deltaPercent(bs.GenerationTPS.Mean, os.GenerationTPS.Mean)
		fmt.Printf("  %-13s  %12.1f  %12.1f  %+9.1f%%\n", "Gen TPS avg",
			bs.GenerationTPS.Mean, os.GenerationTPS.Mean, tpsDelta)

		// Duration
		durDelta := deltaPercent(float64(bs.Duration.Mean), float64(os.Duration.Mean))
		fmt.Printf("  %-13s  %12v  %12v  %+9.1f%%\n", "Duration avg",
			bs.Duration.Mean, os.Duration.Mean, durDelta)

		// Peak RSS
		if bs.PeakRSSBytes > 0 || os.PeakRSSBytes > 0 {
			rssDelta := deltaPercent(float64(bs.PeakRSSBytes), float64(os.PeakRSSBytes))
			fmt.Printf("  %-13s  %10.1f MB  %10.1f MB  %+9.1f%%\n", "Peak RSS",
				float64(bs.PeakRSSBytes)/(1024*1024),
				float64(os.PeakRSSBytes)/(1024*1024), rssDelta)
		}

		// Speculative stats (only for optimized)
		if os.SpeculativeActive {
			fmt.Printf("  %-13s  %12s  %11.1f%%\n", "Spec accept", "n/a",
				os.SpecAcceptanceRate.Mean)
		}
	}
}

// deltaPercent computes the percentage change from baseline to optimized.
// Negative means faster/less, positive means slower/more.
func deltaPercent(baseline, optimized float64) float64 {
	if baseline == 0 {
		return 0
	}
	return (optimized - baseline) / baseline * 100.0
}

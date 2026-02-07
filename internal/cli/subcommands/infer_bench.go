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

	iterations := fs.Int("iterations", 5, "Number of iterations per prompt")
	maxTokens := fs.Int("max-tokens", 128, "Maximum tokens to generate per request")
	warmup := fs.Int("warmup", 1, "Warmup iterations (not recorded)")
	output := fs.String("output", "", "Path to save JSON results (optional)")
	prompt := fs.String("prompt", "", "Custom prompt to benchmark (uses standard set if empty)")
	verbose := fs.Bool("verbose", false, "Print per-iteration details")

	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse flags: %v\n", err)
		return 1
	}

	// Create runtime adapter from config via the registry.
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

	// Build benchmark config.
	benchCfg := inferbench.DefaultConfig()
	benchCfg.Iterations = *iterations
	benchCfg.MaxTokens = *maxTokens
	benchCfg.WarmupIterations = *warmup
	benchCfg.OutputPath = *output
	benchCfg.Verbose = *verbose

	if *prompt != "" {
		benchCfg.Prompts = []inferbench.Prompt{
			{Name: "custom", Text: *prompt},
		}
	}

	fmt.Printf("OpenEye Inference Benchmark\n")
	fmt.Printf("Backend: %s\n", backend)
	fmt.Printf("Iterations: %d (warmup: %d)\n", benchCfg.Iterations, benchCfg.WarmupIterations)
	fmt.Printf("Max tokens: %d\n", benchCfg.MaxTokens)

	runner := inferbench.NewRunner(adapter, benchCfg)
	ctx := context.Background()

	report, err := runner.Run(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Benchmark failed: %v\n", err)
		return 1
	}

	// Print final summary.
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
	}

	return 0
}

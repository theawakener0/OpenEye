// Package inferbench provides an inference benchmark runner for measuring
// SLM performance on edge devices. It works through the runtime.Adapter
// interface, making it backend-agnostic (native CGo or HTTP).
package inferbench

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"OpenEye/internal/runtime"
)

// Config controls the benchmark parameters.
type Config struct {
	// Iterations is how many times each prompt is run.
	Iterations int

	// MaxTokens caps generation length per request.
	MaxTokens int

	// Prompts to benchmark. If empty, StandardPrompts() is used.
	Prompts []Prompt

	// OutputPath is the optional JSON file to write results to.
	OutputPath string

	// Warmup runs N throw-away iterations before recording.
	WarmupIterations int

	// Verbose enables per-iteration logging.
	Verbose bool

	// UseStream benchmarks Stream() instead of Generate().
	// When true, runOnce uses the streaming interface and captures
	// inter-chunk latency and chunk count.
	UseStream bool
}

// DefaultConfig returns reasonable defaults for edge benchmarking.
func DefaultConfig() Config {
	return Config{
		Iterations:       5,
		MaxTokens:        128,
		WarmupIterations: 1,
		Verbose:          false,
	}
}

// Prompt is a single benchmark prompt with metadata.
type Prompt struct {
	Name   string // e.g. "short", "medium", "long", "cached"
	Text   string
	Repeat bool // if true, run the same prompt twice to test cache
}

// StandardPrompts returns a set of prompts that exercise different workloads.
func StandardPrompts() []Prompt {
	return []Prompt{
		{
			Name: "short",
			Text: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			Name: "medium",
			Text: "<|im_start|>system\nYou are a helpful assistant that answers questions clearly and concisely.<|im_end|>\n<|im_start|>user\nExplain the difference between a stack and a queue in computer science. Give a real-world analogy for each.<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			Name: "long",
			Text: "<|im_start|>system\nYou are a knowledgeable AI assistant specializing in science and technology. You provide detailed, accurate responses.<|im_end|>\n<|im_start|>user\nI'm building a small weather station with a Raspberry Pi 5. I want to measure temperature, humidity, barometric pressure, wind speed, and rainfall. Can you help me design the hardware setup? What sensors should I use, how should I wire them, and what software stack would you recommend for collecting and visualizing the data? I'd like to log data every 5 minutes and serve a web dashboard on my local network.<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			Name:   "cache-test",
			Text:   "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
			Repeat: true,
		},
	}
}

// IterationResult captures metrics from a single generation call.
type IterationResult struct {
	PromptName      string        `json:"prompt_name"`
	Iteration       int           `json:"iteration"`
	TTFT            time.Duration `json:"ttft_ns"`
	Duration        time.Duration `json:"duration_ns"`
	TokensEvaluated int           `json:"tokens_evaluated"`
	TokensGenerated int           `json:"tokens_generated"`
	TokensCached    int           `json:"tokens_cached"`
	PromptTPS       float64       `json:"prompt_tps"`
	GenerationTPS   float64       `json:"generation_tps"`
	RSSBytes        int64         `json:"rss_bytes"`
	Error           string        `json:"error,omitempty"`

	// Speculative decoding metrics (zero when speculative is off).
	SpecDrafted        int     `json:"spec_drafted,omitempty"`
	SpecAccepted       int     `json:"spec_accepted,omitempty"`
	SpecAcceptanceRate float64 `json:"spec_acceptance_rate,omitempty"`

	// Streaming metrics (zero when using Generate mode).
	ChunkCount      int           `json:"chunk_count,omitempty"`
	AvgChunkLatency time.Duration `json:"avg_chunk_latency_ns,omitempty"`
	MaxChunkLatency time.Duration `json:"max_chunk_latency_ns,omitempty"`
}

// PromptSummary aggregates results across iterations for a single prompt.
type PromptSummary struct {
	Name            string        `json:"name"`
	Iterations      int           `json:"iterations"`
	TTFT            DurationStats `json:"ttft"`
	Duration        DurationStats `json:"duration"`
	PromptTPS       FloatStats    `json:"prompt_tps"`
	GenerationTPS   FloatStats    `json:"generation_tps"`
	AvgTokensGen    float64       `json:"avg_tokens_generated"`
	AvgTokensCached float64       `json:"avg_tokens_cached"`
	CacheHitImprove float64       `json:"cache_hit_ttft_improvement_pct,omitempty"` // for Repeat prompts
	PeakRSSBytes    int64         `json:"peak_rss_bytes"`
	Errors          int           `json:"errors"`

	// Speculative decoding aggregates (omitted from JSON when zero).
	SpecAcceptanceRate FloatStats `json:"spec_acceptance_rate,omitempty"`
	AvgSpecDrafted     float64    `json:"avg_spec_drafted,omitempty"`
	AvgSpecAccepted    float64    `json:"avg_spec_accepted,omitempty"`
	SpeculativeActive  bool       `json:"speculative_active,omitempty"`

	// Streaming aggregates (omitted when using Generate mode).
	AvgChunkLatency DurationStats `json:"avg_chunk_latency,omitempty"`
	MaxChunkLatency DurationStats `json:"max_chunk_latency,omitempty"`
	AvgChunkCount   float64       `json:"avg_chunk_count,omitempty"`
	StreamMode      bool          `json:"stream_mode,omitempty"`
}

// DurationStats summarises a collection of time.Duration values.
type DurationStats struct {
	Min    time.Duration `json:"min_ns"`
	Max    time.Duration `json:"max_ns"`
	Mean   time.Duration `json:"mean_ns"`
	Median time.Duration `json:"median_ns"`
	P95    time.Duration `json:"p95_ns"`
}

// FloatStats summarises a collection of float64 values.
type FloatStats struct {
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	P95    float64 `json:"p95"`
}

// BenchmarkReport is the top-level result container.
type BenchmarkReport struct {
	Timestamp  time.Time         `json:"timestamp"`
	Config     Config            `json:"config"`
	SystemInfo string            `json:"system_info,omitempty"`
	Summaries  []PromptSummary   `json:"summaries"`
	Raw        []IterationResult `json:"raw_results,omitempty"`
}

// Runner executes inference benchmarks against a runtime.Adapter.
type Runner struct {
	adapter runtime.Adapter
	cfg     Config
}

// NewRunner creates a benchmark runner.
func NewRunner(adapter runtime.Adapter, cfg Config) *Runner {
	if len(cfg.Prompts) == 0 {
		cfg.Prompts = StandardPrompts()
	}
	return &Runner{adapter: adapter, cfg: cfg}
}

// Run executes the full benchmark suite and returns a report.
func (r *Runner) Run(ctx context.Context) (*BenchmarkReport, error) {
	report := &BenchmarkReport{
		Timestamp: time.Now(),
		Config:    r.cfg,
	}

	var allResults []IterationResult

	for _, prompt := range r.cfg.Prompts {
		if err := ctx.Err(); err != nil {
			return report, err
		}

		fmt.Printf("\n--- Benchmark: %s ---\n", prompt.Name)

		results, err := r.benchmarkPrompt(ctx, prompt)
		if err != nil {
			return report, fmt.Errorf("prompt %q: %w", prompt.Name, err)
		}

		allResults = append(allResults, results...)
		summary := summarize(prompt, results)
		report.Summaries = append(report.Summaries, summary)

		printSummary(summary)
	}

	report.Raw = allResults

	// Save to file if configured.
	if r.cfg.OutputPath != "" {
		if err := saveReport(report, r.cfg.OutputPath); err != nil {
			fmt.Printf("Warning: failed to save report: %v\n", err)
		} else {
			fmt.Printf("\nResults saved to %s\n", r.cfg.OutputPath)
		}
	}

	return report, nil
}

// benchmarkPrompt runs all iterations for a single prompt.
func (r *Runner) benchmarkPrompt(ctx context.Context, prompt Prompt) ([]IterationResult, error) {
	totalIter := r.cfg.Iterations
	if prompt.Repeat {
		// For cache tests, we need pairs: first run (cold) + second run (cached).
		// We'll run warmup + iterations of pairs.
		totalIter = r.cfg.Iterations
	}

	var results []IterationResult

	// Warmup iterations (not recorded).
	for i := 0; i < r.cfg.WarmupIterations; i++ {
		if r.cfg.Verbose {
			fmt.Printf("  warmup %d/%d...\n", i+1, r.cfg.WarmupIterations)
		}
		_, _ = r.runOnce(ctx, prompt, -1)
	}

	// Recorded iterations.
	for i := 0; i < totalIter; i++ {
		if r.cfg.Verbose {
			fmt.Printf("  iteration %d/%d...\n", i+1, totalIter)
		}

		res, err := r.runOnce(ctx, prompt, i)
		if err != nil {
			res.Error = err.Error()
		}
		results = append(results, res)

		// For cache-test prompts, immediately run the same prompt again.
		if prompt.Repeat {
			res2, err := r.runOnce(ctx, prompt, i)
			if err != nil {
				res2.Error = err.Error()
			}
			res2.PromptName = prompt.Name + "-cached"
			results = append(results, res2)
		}
	}

	return results, nil
}

// runOnce executes a single Generate call and captures metrics.
func (r *Runner) runOnce(ctx context.Context, prompt Prompt, iteration int) (IterationResult, error) {
	if r.cfg.UseStream {
		return r.runOnceStream(ctx, prompt, iteration)
	}

	rss := readRSS()

	req := runtime.Request{
		Prompt: prompt.Text,
		Options: runtime.GenerationOptions{
			MaxTokens: r.cfg.MaxTokens,
		},
	}

	resp, err := r.adapter.Generate(ctx, req)
	result := IterationResult{
		PromptName: prompt.Name,
		Iteration:  iteration,
		RSSBytes:   rss,
	}

	if err != nil {
		return result, err
	}

	result.TTFT = resp.Stats.TTFT
	result.Duration = resp.Stats.Duration
	result.TokensEvaluated = resp.Stats.TokensEvaluated
	result.TokensGenerated = resp.Stats.TokensGenerated
	result.TokensCached = resp.Stats.TokensCached
	result.PromptTPS = resp.Stats.PromptTPS
	result.GenerationTPS = resp.Stats.GenerationTPS
	result.SpecDrafted = resp.Stats.SpeculativeAttempted
	result.SpecAccepted = resp.Stats.SpeculativeAccepted
	result.SpecAcceptanceRate = resp.Stats.SpeculativeAcceptanceRate
	result.RSSBytes = readRSS()

	if r.cfg.Verbose {
		specInfo := ""
		if result.SpecDrafted > 0 {
			specInfo = fmt.Sprintf("  spec=%d/%d (%.0f%%)", result.SpecAccepted, result.SpecDrafted, result.SpecAcceptanceRate)
		}
		fmt.Printf("    TTFT=%v  gen=%d tok @ %.1f tok/s  cached=%d%s\n",
			result.TTFT.Round(time.Millisecond),
			result.TokensGenerated,
			result.GenerationTPS,
			result.TokensCached,
			specInfo)
	}

	return result, nil
}

// runOnceStream executes a single Stream call and captures metrics including
// inter-chunk latency.
func (r *Runner) runOnceStream(ctx context.Context, prompt Prompt, iteration int) (IterationResult, error) {
	rss := readRSS()

	req := runtime.Request{
		Prompt: prompt.Text,
		Options: runtime.GenerationOptions{
			MaxTokens: r.cfg.MaxTokens,
		},
	}

	result := IterationResult{
		PromptName: prompt.Name,
		Iteration:  iteration,
		RSSBytes:   rss,
	}

	var chunkCount int
	var maxChunkLatency time.Duration
	var totalChunkLatency time.Duration
	var lastChunkTime time.Time
	startTime := time.Now()

	err := r.adapter.Stream(ctx, req, func(ev runtime.StreamEvent) error {
		now := time.Now()

		if ev.Err != nil {
			return ev.Err
		}

		if ev.Final {
			// Capture final stats from the streaming event.
			if ev.Stats != nil {
				result.TTFT = ev.Stats.TTFT
				result.Duration = ev.Stats.Duration
				result.TokensEvaluated = ev.Stats.TokensEvaluated
				result.TokensGenerated = ev.Stats.TokensGenerated
				result.TokensCached = ev.Stats.TokensCached
				result.PromptTPS = ev.Stats.PromptTPS
				result.GenerationTPS = ev.Stats.GenerationTPS
				result.SpecDrafted = ev.Stats.SpeculativeAttempted
				result.SpecAccepted = ev.Stats.SpeculativeAccepted
				result.SpecAcceptanceRate = ev.Stats.SpeculativeAcceptanceRate
			}
			return nil
		}

		// Non-final event: track chunk latency.
		chunkCount++
		if !lastChunkTime.IsZero() {
			latency := now.Sub(lastChunkTime)
			totalChunkLatency += latency
			if latency > maxChunkLatency {
				maxChunkLatency = latency
			}
		}
		lastChunkTime = now
		return nil
	})

	if err != nil {
		result.Error = err.Error()
		return result, err
	}

	// If Duration wasn't set by final stats, compute it.
	if result.Duration == 0 {
		result.Duration = time.Since(startTime)
	}

	result.ChunkCount = chunkCount
	result.MaxChunkLatency = maxChunkLatency
	if chunkCount > 1 {
		result.AvgChunkLatency = totalChunkLatency / time.Duration(chunkCount-1)
	}
	result.RSSBytes = readRSS()

	if r.cfg.Verbose {
		specInfo := ""
		if result.SpecDrafted > 0 {
			specInfo = fmt.Sprintf("  spec=%d/%d (%.0f%%)", result.SpecAccepted, result.SpecDrafted, result.SpecAcceptanceRate)
		}
		fmt.Printf("    TTFT=%v  gen=%d tok  chunks=%d  avg_latency=%v  max_latency=%v%s\n",
			result.TTFT.Round(time.Millisecond),
			result.TokensGenerated,
			result.ChunkCount,
			result.AvgChunkLatency.Round(time.Microsecond),
			result.MaxChunkLatency.Round(time.Microsecond),
			specInfo)
	}

	return result, nil
}

// summarize computes aggregate statistics for a prompt's results.
func summarize(prompt Prompt, results []IterationResult) PromptSummary {
	summary := PromptSummary{Name: prompt.Name}

	// Split results by prompt name (for cache-test, separate cold vs cached).
	cold := filterByName(results, prompt.Name)
	cached := filterByName(results, prompt.Name+"-cached")

	// Summarize the cold (primary) runs.
	valid := filterValid(cold)
	summary.Iterations = len(valid)
	summary.Errors = len(cold) - len(valid)

	if len(valid) == 0 {
		return summary
	}

	summary.TTFT = computeDurationStats(extractDurations(valid, func(r IterationResult) time.Duration { return r.TTFT }))
	summary.Duration = computeDurationStats(extractDurations(valid, func(r IterationResult) time.Duration { return r.Duration }))
	summary.PromptTPS = computeFloatStats(extractFloats(valid, func(r IterationResult) float64 { return r.PromptTPS }))
	summary.GenerationTPS = computeFloatStats(extractFloats(valid, func(r IterationResult) float64 { return r.GenerationTPS }))

	var tokGenSum, tokCacheSum float64
	var peakRSS int64
	var specDraftedSum, specAcceptedSum float64
	hasSpec := false
	for _, r := range valid {
		tokGenSum += float64(r.TokensGenerated)
		tokCacheSum += float64(r.TokensCached)
		if r.RSSBytes > peakRSS {
			peakRSS = r.RSSBytes
		}
		if r.SpecDrafted > 0 {
			hasSpec = true
			specDraftedSum += float64(r.SpecDrafted)
			specAcceptedSum += float64(r.SpecAccepted)
		}
	}
	summary.AvgTokensGen = tokGenSum / float64(len(valid))
	summary.AvgTokensCached = tokCacheSum / float64(len(valid))
	summary.PeakRSSBytes = peakRSS

	// Speculative decoding aggregates.
	if hasSpec {
		summary.SpeculativeActive = true
		summary.AvgSpecDrafted = specDraftedSum / float64(len(valid))
		summary.AvgSpecAccepted = specAcceptedSum / float64(len(valid))
		summary.SpecAcceptanceRate = computeFloatStats(extractFloats(valid, func(r IterationResult) float64 { return r.SpecAcceptanceRate }))
	}

	// Streaming aggregates.
	hasStream := false
	for _, r := range valid {
		if r.ChunkCount > 0 {
			hasStream = true
			break
		}
	}
	if hasStream {
		summary.StreamMode = true
		summary.AvgChunkLatency = computeDurationStats(extractDurations(valid, func(r IterationResult) time.Duration { return r.AvgChunkLatency }))
		summary.MaxChunkLatency = computeDurationStats(extractDurations(valid, func(r IterationResult) time.Duration { return r.MaxChunkLatency }))
		var chunkSum float64
		for _, r := range valid {
			chunkSum += float64(r.ChunkCount)
		}
		summary.AvgChunkCount = chunkSum / float64(len(valid))
	}

	// Cache hit improvement (for Repeat prompts).
	if prompt.Repeat && len(cached) > 0 {
		validCached := filterValid(cached)
		if len(validCached) > 0 && summary.TTFT.Mean > 0 {
			cachedTTFT := computeDurationStats(extractDurations(validCached, func(r IterationResult) time.Duration { return r.TTFT }))
			improvement := float64(summary.TTFT.Mean-cachedTTFT.Mean) / float64(summary.TTFT.Mean) * 100
			summary.CacheHitImprove = math.Round(improvement*10) / 10
		}
	}

	return summary
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

func filterByName(results []IterationResult, name string) []IterationResult {
	var out []IterationResult
	for _, r := range results {
		if r.PromptName == name {
			out = append(out, r)
		}
	}
	return out
}

func filterValid(results []IterationResult) []IterationResult {
	var out []IterationResult
	for _, r := range results {
		if r.Error == "" {
			out = append(out, r)
		}
	}
	return out
}

func extractDurations(results []IterationResult, fn func(IterationResult) time.Duration) []time.Duration {
	out := make([]time.Duration, len(results))
	for i, r := range results {
		out[i] = fn(r)
	}
	return out
}

func extractFloats(results []IterationResult, fn func(IterationResult) float64) []float64 {
	out := make([]float64, len(results))
	for i, r := range results {
		out[i] = fn(r)
	}
	return out
}

func computeDurationStats(vals []time.Duration) DurationStats {
	if len(vals) == 0 {
		return DurationStats{}
	}
	sorted := make([]time.Duration, len(vals))
	copy(sorted, vals)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	var sum time.Duration
	for _, v := range sorted {
		sum += v
	}

	n := len(sorted)
	var median time.Duration
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	} else {
		median = sorted[n/2]
	}

	return DurationStats{
		Min:    sorted[0],
		Max:    sorted[n-1],
		Mean:   sum / time.Duration(n),
		Median: median,
		P95:    sorted[percentileIndex(n, 95)],
	}
}

func computeFloatStats(vals []float64) FloatStats {
	if len(vals) == 0 {
		return FloatStats{}
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)

	var sum float64
	for _, v := range sorted {
		sum += v
	}

	n := len(sorted)
	var median float64
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2
	} else {
		median = sorted[n/2]
	}

	return FloatStats{
		Min:    sorted[0],
		Max:    sorted[n-1],
		Mean:   sum / float64(n),
		Median: median,
		P95:    sorted[percentileIndex(n, 95)],
	}
}

// percentileIndex returns the index for the pct-th percentile using the
// nearest-rank method: index = ceil(n * pct / 100) - 1, clamped to [0, n-1].
func percentileIndex(n, pct int) int {
	if n <= 0 {
		return 0
	}
	// ceil(n * pct / 100) - 1
	idx := (n*pct+99)/100 - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	return idx
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

func printSummary(s PromptSummary) {
	fmt.Printf("  TTFT:      min=%v  avg=%v  p95=%v\n",
		s.TTFT.Min.Round(time.Millisecond),
		s.TTFT.Mean.Round(time.Millisecond),
		s.TTFT.P95.Round(time.Millisecond))
	fmt.Printf("  Duration:  min=%v  avg=%v  p95=%v\n",
		s.Duration.Min.Round(time.Millisecond),
		s.Duration.Mean.Round(time.Millisecond),
		s.Duration.P95.Round(time.Millisecond))
	fmt.Printf("  Gen TPS:   min=%.1f  avg=%.1f  p95=%.1f\n",
		s.GenerationTPS.Min, s.GenerationTPS.Mean, s.GenerationTPS.P95)
	fmt.Printf("  Prompt TPS: min=%.1f  avg=%.1f  p95=%.1f\n",
		s.PromptTPS.Min, s.PromptTPS.Mean, s.PromptTPS.P95)
	fmt.Printf("  Tokens:    avg_gen=%.0f  avg_cached=%.0f\n",
		s.AvgTokensGen, s.AvgTokensCached)
	if s.PeakRSSBytes > 0 {
		fmt.Printf("  RSS:       peak=%.1f MB\n", float64(s.PeakRSSBytes)/(1024*1024))
	}
	if s.CacheHitImprove != 0 {
		fmt.Printf("  Cache:     TTFT improvement=%.1f%%\n", s.CacheHitImprove)
	}
	if s.SpeculativeActive {
		fmt.Printf("  Spec:      acceptance=%.1f%% (avg=%.1f%%)  drafted=%.0f  accepted=%.0f\n",
			s.SpecAcceptanceRate.Median, s.SpecAcceptanceRate.Mean,
			s.AvgSpecDrafted, s.AvgSpecAccepted)
	}
	if s.StreamMode {
		fmt.Printf("  Chunks:    avg=%.0f  avg_latency=%v  max_latency=%v (p95=%v)\n",
			s.AvgChunkCount,
			s.AvgChunkLatency.Mean.Round(time.Microsecond),
			s.MaxChunkLatency.Mean.Round(time.Microsecond),
			s.MaxChunkLatency.P95.Round(time.Microsecond))
	}
	if s.Errors > 0 {
		fmt.Printf("  Errors:    %d/%d\n", s.Errors, s.Iterations+s.Errors)
	}
}

func saveReport(report *BenchmarkReport, path string) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}

	// Ensure parent directory exists.
	if idx := strings.LastIndex(path, "/"); idx > 0 {
		if err := os.MkdirAll(path[:idx], 0755); err != nil {
			return err
		}
	}

	return os.WriteFile(path, data, 0644)
}

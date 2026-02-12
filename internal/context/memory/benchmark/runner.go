package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"
)

// BenchmarkResult holds the results of a single benchmark run.
type BenchmarkResult struct {
	SystemName    string                 `json:"system_name"`
	Timestamp     time.Time              `json:"timestamp"`
	Configuration map[string]interface{} `json:"configuration"`

	// Latency metrics (in microseconds)
	WriteLatency    LatencyStats `json:"write_latency_us"`
	RetrieveLatency LatencyStats `json:"retrieve_latency_us"`

	// Token efficiency
	TokenMetrics TokenStats `json:"token_metrics"`

	// Memory usage
	MemoryMetrics MemoryStats `json:"memory_metrics"`

	// Recall accuracy (for planted facts)
	RecallMetrics RecallStats `json:"recall_metrics"`

	// Sub-metric timing for Omem (aggregated)
	SubMetricLatency SubMetricStats `json:"sub_metric_latency"`

	// Raw measurements for analysis
	RawMeasurements []Measurement `json:"raw_measurements,omitempty"`
}

// SubMetricStats holds aggregated sub-metric timing statistics.
type SubMetricStats struct {
	EmbeddingLatency  LatencyStats `json:"embedding_latency_us"`
	LLMLatency        LatencyStats `json:"llm_latency_us"`
	DBLatency         LatencyStats `json:"db_latency_us"`
	ProcessingLatency LatencyStats `json:"processing_latency_us"`
}

// LatencyStats holds statistical summary of latencies.
type LatencyStats struct {
	Count  int     `json:"count"`
	Mean   float64 `json:"mean_us"`
	Median float64 `json:"median_us"`
	P95    float64 `json:"p95_us"`
	P99    float64 `json:"p99_us"`
	Min    float64 `json:"min_us"`
	Max    float64 `json:"max_us"`
	StdDev float64 `json:"std_dev_us"`
}

// TokenStats holds token consumption metrics.
type TokenStats struct {
	TotalInputTokens  int     `json:"total_input_tokens"`
	TotalOutputTokens int     `json:"total_output_tokens"`
	AvgContextSize    float64 `json:"avg_context_size_tokens"`
	MaxContextSize    int     `json:"max_context_size_tokens"`
	ContextGrowthRate float64 `json:"context_growth_rate"` // Tokens per turn
	CompressionRatio  float64 `json:"compression_ratio"`   // If compression used
}

// MemoryStats holds memory usage metrics.
type MemoryStats struct {
	PeakHeapBytes   uint64 `json:"peak_heap_bytes"`
	AvgHeapBytes    uint64 `json:"avg_heap_bytes"`
	TotalAllocBytes uint64 `json:"total_alloc_bytes"`
	NumGC           uint32 `json:"num_gc"`
	GCPauseTotal    uint64 `json:"gc_pause_total_ns"`
	DBSizeBytes     int64  `json:"db_size_bytes"`
}

// RecallStats holds fact recall accuracy metrics.
type RecallStats struct {
	TotalPlanted     int     `json:"total_planted_facts"`
	CorrectRecalls   int     `json:"correct_recalls"`
	PartialRecalls   int     `json:"partial_recalls"`
	FailedRecalls    int     `json:"failed_recalls"`
	RecallAccuracy   float64 `json:"recall_accuracy"`
	AvgRecallLatency float64 `json:"avg_recall_latency_us"`

	// By distance (turns since fact was planted)
	RecallByDistance map[string]float64 `json:"recall_by_distance"` // "0-10", "11-25", "26-50", "50+"
}

// Measurement is a single timed operation.
type Measurement struct {
	Operation  string    `json:"operation"`
	DurationUS int64     `json:"duration_us"`
	Timestamp  time.Time `json:"timestamp"`
	TurnIndex  int       `json:"turn_index,omitempty"`
	TokenCount int       `json:"token_count,omitempty"`
	Success    bool      `json:"success"`
	Error      string    `json:"error,omitempty"`

	// Sub-metric timing for Omem (in microseconds)
	EmbeddingLatency  int64 `json:"embedding_latency_us,omitempty"`
	LLMLatency        int64 `json:"llm_latency_us,omitempty"`
	DBLatency         int64 `json:"db_latency_us,omitempty"`
	ProcessingLatency int64 `json:"processing_latency_us,omitempty"`
}

// MemorySystemAdapter is the interface that all memory systems must implement for benchmarking.
type MemorySystemAdapter interface {
	Name() string
	Initialize(ctx context.Context) error
	Store(ctx context.Context, role, content string, turnIndex int) error
	Retrieve(ctx context.Context, query string, limit int) ([]string, error)
	BuildContext(ctx context.Context, query string, maxTokens int) (string, int, error) // returns context and token count
	GetStats(ctx context.Context) (map[string]interface{}, error)
	Close() error
}

// BenchmarkRunner executes benchmarks against memory systems.
type BenchmarkRunner struct {
	systems      []MemorySystemAdapter
	config       BenchmarkConfig
	measurements []Measurement
	mu           sync.Mutex
}

// BenchmarkConfig configures the benchmark run.
type BenchmarkConfig struct {
	NumTurns            int           `json:"num_turns"`
	NumRecallTests      int           `json:"num_recall_tests"`
	RetrievalLimit      int           `json:"retrieval_limit"`
	MaxContextTokens    int           `json:"max_context_tokens"`
	WarmupTurns         int           `json:"warmup_turns"`
	CooldownBetweenOps  time.Duration `json:"cooldown_between_ops"`
	SaveRawMeasurements bool          `json:"save_raw_measurements"`
	OutputDir           string        `json:"output_dir"`
}

// DefaultBenchmarkConfig returns sensible defaults.
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		NumTurns:            50,
		NumRecallTests:      10,
		RetrievalLimit:      10,
		MaxContextTokens:    2048,
		WarmupTurns:         5,
		CooldownBetweenOps:  10 * time.Millisecond,
		SaveRawMeasurements: true,
		OutputDir:           "benchmark_results",
	}
}

// NewBenchmarkRunner creates a new benchmark runner.
func NewBenchmarkRunner(config BenchmarkConfig, systems ...MemorySystemAdapter) *BenchmarkRunner {
	return &BenchmarkRunner{
		systems:      systems,
		config:       config,
		measurements: make([]Measurement, 0),
	}
}

// Run executes the full benchmark suite.
func (r *BenchmarkRunner) Run(ctx context.Context) ([]BenchmarkResult, error) {
	// Generate test data
	gen := NewGenerator(GeneratorConfig{
		NumTurns:          r.config.NumTurns,
		PlantedFactsCount: r.config.NumRecallTests,
		Seed:              42, // Fixed seed for reproducibility
	})

	conversation := gen.GenerateConversation()
	recallTests := gen.GenerateRecallTests(conversation)

	results := make([]BenchmarkResult, 0, len(r.systems))

	for _, system := range r.systems {
		fmt.Printf("\n=== Benchmarking: %s ===\n", system.Name())

		result, err := r.benchmarkSystem(ctx, system, conversation, recallTests)
		if err != nil {
			fmt.Printf("Error benchmarking %s: %v\n", system.Name(), err)
			continue
		}

		results = append(results, result)

		// Save individual result
		if r.config.OutputDir != "" {
			r.saveResult(result)
		}
	}

	return results, nil
}

func (r *BenchmarkRunner) benchmarkSystem(
	ctx context.Context,
	system MemorySystemAdapter,
	conversation SyntheticConversation,
	recallTests []RecallTest,
) (BenchmarkResult, error) {
	result := BenchmarkResult{
		SystemName: system.Name(),
		Timestamp:  time.Now(),
		Configuration: map[string]interface{}{
			"num_turns":    r.config.NumTurns,
			"recall_tests": r.config.NumRecallTests,
			"max_tokens":   r.config.MaxContextTokens,
		},
		RecallMetrics: RecallStats{
			RecallByDistance: make(map[string]float64),
		},
	}

	// Initialize
	if err := system.Initialize(ctx); err != nil {
		return result, fmt.Errorf("failed to initialize: %w", err)
	}
	defer system.Close()

	// Clear measurements
	r.mu.Lock()
	r.measurements = make([]Measurement, 0, len(conversation.Turns)*2)
	r.mu.Unlock()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	initialAlloc := memStats.TotalAlloc
	initialGC := memStats.NumGC
	var peakHeap, totalHeap uint64
	heapSamples := 0

	writeTimes := make([]int64, 0, len(conversation.Turns))
	retrieveTimes := make([]int64, 0, len(conversation.Turns))
	contextSizes := make([]int, 0, len(conversation.Turns))

	// Process conversation turns
	for i, turn := range conversation.Turns {
		if (i/2)%5 == 0 && i%2 == 0 {
			fmt.Printf("  Processing turn %d/%d...\n", i/2, len(conversation.Turns)/2)
		}
		// Skip warmup in metrics
		isWarmup := i < r.config.WarmupTurns*2 // *2 because user+assistant

		// Store operation
		start := time.Now()
		err := system.Store(ctx, turn.Role, turn.Content, i)
		elapsed := time.Since(start).Microseconds()

		if !isWarmup {
			writeTimes = append(writeTimes, elapsed)
		}

		r.recordMeasurement(Measurement{
			Operation:  "store",
			DurationUS: elapsed,
			Timestamp:  time.Now(),
			TurnIndex:  i,
			Success:    err == nil,
			Error:      errorString(err),
		})

		// Periodically test retrieval (every 5th turn for user messages)
		if turn.Role == "user" && i%10 == 0 && !isWarmup {
			start = time.Now()
			ctxStr, tokenCount, err := system.BuildContext(ctx, turn.Content, r.config.MaxContextTokens)
			elapsed = time.Since(start).Microseconds()

			retrieveTimes = append(retrieveTimes, elapsed)
			if ctxStr != "" {
				contextSizes = append(contextSizes, tokenCount)
			}

			r.recordMeasurement(Measurement{
				Operation:  "retrieve",
				DurationUS: elapsed,
				Timestamp:  time.Now(),
				TurnIndex:  i,
				TokenCount: tokenCount,
				Success:    err == nil,
				Error:      errorString(err),
			})
		}

		// Sample memory
		runtime.ReadMemStats(&memStats)
		if memStats.HeapAlloc > peakHeap {
			peakHeap = memStats.HeapAlloc
		}
		totalHeap += memStats.HeapAlloc
		heapSamples++

		time.Sleep(r.config.CooldownBetweenOps)
	}

	// Run recall tests
	fmt.Printf("  Running %d recall tests...\n", len(recallTests))
	recallResults := r.runRecallTests(ctx, system, recallTests, len(conversation.Turns))
	result.RecallMetrics = recallResults

	// Calculate final statistics
	runtime.ReadMemStats(&memStats)

	result.WriteLatency = calculateLatencyStats(writeTimes)
	result.RetrieveLatency = calculateLatencyStats(retrieveTimes)

	result.TokenMetrics = TokenStats{
		AvgContextSize: average(contextSizes),
		MaxContextSize: max(contextSizes),
	}
	if len(contextSizes) > 1 {
		result.TokenMetrics.ContextGrowthRate = float64(contextSizes[len(contextSizes)-1]-contextSizes[0]) / float64(len(contextSizes))
	}

	result.MemoryMetrics = MemoryStats{
		PeakHeapBytes:   peakHeap,
		AvgHeapBytes:    totalHeap / uint64(heapSamples),
		TotalAllocBytes: memStats.TotalAlloc - initialAlloc,
		NumGC:           memStats.NumGC - initialGC,
		GCPauseTotal:    memStats.PauseTotalNs,
	}

	if r.config.SaveRawMeasurements {
		r.mu.Lock()
		result.RawMeasurements = r.measurements
		r.mu.Unlock()
	}

	// Calculate sub-metric statistics from raw measurements
	result.SubMetricLatency = calculateSubMetricStats(r.measurements)

	// Print summary
	r.printSummary(result)

	return result, nil
}

func (r *BenchmarkRunner) runRecallTests(
	ctx context.Context,
	system MemorySystemAdapter,
	tests []RecallTest,
	totalTurns int,
) RecallStats {
	stats := RecallStats{
		TotalPlanted:     len(tests),
		RecallByDistance: make(map[string]float64),
	}

	distanceBuckets := map[string][]bool{
		"0-10":  {},
		"11-25": {},
		"26-50": {},
		"50+":   {},
	}

	var totalLatency int64

	for _, test := range tests {
		start := time.Now()
		results, err := system.Retrieve(ctx, test.Query, 5)
		elapsed := time.Since(start).Microseconds()
		totalLatency += elapsed

		if err != nil {
			stats.FailedRecalls++
			continue
		}

		// Check if expected answer appears in results
		found := false
		partial := false
		fmt.Printf("    Test Query: %s\n", test.Query)
		fmt.Printf("    Expected Answer: %s\n", test.ExpectedAnswer)
		for i, result := range results {
			fmt.Printf("      Result %d: %s\n", i, result)
			if containsAnswer(result, test.ExpectedAnswer) {
				found = true
				break
			}
			if partialMatch(result, test.ExpectedAnswer) {
				partial = true
			}
		}
		if !found && !partial {
			fmt.Printf("    FAILED RECALL\n")
		} else if found {
			fmt.Printf("    SUCCESS RECALL\n")
		} else {
			fmt.Printf("    PARTIAL RECALL\n")
		}

		if found {
			stats.CorrectRecalls++
		} else if partial {
			stats.PartialRecalls++
		} else {
			stats.FailedRecalls++
		}

		// Categorize by distance
		distance := totalTurns - test.TurnPlanted*2 // Approximate
		bucket := distanceBucket(distance)
		distanceBuckets[bucket] = append(distanceBuckets[bucket], found)
	}

	stats.RecallAccuracy = float64(stats.CorrectRecalls) / float64(stats.TotalPlanted)
	stats.AvgRecallLatency = float64(totalLatency) / float64(len(tests))

	// Calculate per-bucket accuracy
	for bucket, results := range distanceBuckets {
		if len(results) > 0 {
			correct := 0
			for _, r := range results {
				if r {
					correct++
				}
			}
			stats.RecallByDistance[bucket] = float64(correct) / float64(len(results))
		}
	}

	return stats
}

func (r *BenchmarkRunner) recordMeasurement(m Measurement) {
	r.mu.Lock()
	r.measurements = append(r.measurements, m)
	r.mu.Unlock()
}

func (r *BenchmarkRunner) printSummary(result BenchmarkResult) {
	fmt.Printf("\n--- %s Results ---\n", result.SystemName)
	fmt.Printf("Write Latency:    P50=%.2fus  P95=%.2fus  P99=%.2fus\n",
		result.WriteLatency.Median, result.WriteLatency.P95, result.WriteLatency.P99)
	fmt.Printf("Retrieve Latency: P50=%.2fus  P95=%.2fus  P99=%.2fus\n",
		result.RetrieveLatency.Median, result.RetrieveLatency.P95, result.RetrieveLatency.P99)
	fmt.Printf("Context Size:     Avg=%.1f  Max=%d tokens\n",
		result.TokenMetrics.AvgContextSize, result.TokenMetrics.MaxContextSize)
	fmt.Printf("Recall Accuracy:  %.1f%% (%d/%d)\n",
		result.RecallMetrics.RecallAccuracy*100,
		result.RecallMetrics.CorrectRecalls,
		result.RecallMetrics.TotalPlanted)
	fmt.Printf("Peak Memory:      %.2f MB\n", float64(result.MemoryMetrics.PeakHeapBytes)/(1024*1024))

	// Print sub-metric timing if available
	if result.SubMetricLatency.EmbeddingLatency.Count > 0 {
		fmt.Printf("Sub-metric Latency:\n")
		fmt.Printf("  Embedding:   P50=%.2fus  P95=%.2fus\n",
			result.SubMetricLatency.EmbeddingLatency.Median,
			result.SubMetricLatency.EmbeddingLatency.P95)
		fmt.Printf("  LLM:         P50=%.2fus  P95=%.2fus\n",
			result.SubMetricLatency.LLMLatency.Median,
			result.SubMetricLatency.LLMLatency.P95)
		fmt.Printf("  DB:          P50=%.2fus  P95=%.2fus\n",
			result.SubMetricLatency.DBLatency.Median,
			result.SubMetricLatency.DBLatency.P95)
		fmt.Printf("  Processing:  P50=%.2fus  P95=%.2fus\n",
			result.SubMetricLatency.ProcessingLatency.Median,
			result.SubMetricLatency.ProcessingLatency.P95)
	}
}

func (r *BenchmarkRunner) saveResult(result BenchmarkResult) {
	if err := os.MkdirAll(r.config.OutputDir, 0755); err != nil {
		fmt.Printf("Warning: could not create output dir: %v\n", err)
		return
	}

	filename := fmt.Sprintf("%s/%s_%s.json",
		r.config.OutputDir,
		result.SystemName,
		result.Timestamp.Format("20060102_150405"))

	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		fmt.Printf("Warning: could not marshal result: %v\n", err)
		return
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		fmt.Printf("Warning: could not write result file: %v\n", err)
		return
	}

	fmt.Printf("  Results saved to: %s\n", filename)
}

// Helper functions

func calculateLatencyStats(values []int64) LatencyStats {
	if len(values) == 0 {
		return LatencyStats{}
	}

	// Sort for percentiles
	sorted := make([]int64, len(values))
	copy(sorted, values)
	sortInt64(sorted)

	stats := LatencyStats{
		Count:  len(values),
		Min:    float64(sorted[0]),
		Max:    float64(sorted[len(sorted)-1]),
		Median: float64(sorted[len(sorted)/2]),
		P95:    float64(sorted[int(float64(len(sorted))*0.95)]),
		P99:    float64(sorted[int(float64(len(sorted))*0.99)]),
	}

	// Calculate mean and std dev
	var sum int64
	for _, v := range values {
		sum += v
	}
	stats.Mean = float64(sum) / float64(len(values))

	var variance float64
	for _, v := range values {
		diff := float64(v) - stats.Mean
		variance += diff * diff
	}
	variance /= float64(len(values))
	stats.StdDev = sqrt(variance)

	return stats
}

func sortInt64(arr []int64) {
	// Simple insertion sort for small arrays
	for i := 1; i < len(arr); i++ {
		key := arr[i]
		j := i - 1
		for j >= 0 && arr[j] > key {
			arr[j+1] = arr[j]
			j--
		}
		arr[j+1] = key
	}
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x / 2
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

func average(values []int) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0
	for _, v := range values {
		sum += v
	}
	return float64(sum) / float64(len(values))
}

func max(values []int) int {
	if len(values) == 0 {
		return 0
	}
	m := values[0]
	for _, v := range values[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func containsAnswer(text, answer string) bool {
	// Case-insensitive substring check
	textLower := toLower(text)
	answerLower := toLower(answer)
	return contains(textLower, answerLower)
}

func partialMatch(text, answer string) bool {
	// Check if at least half the words match
	words := splitWords(answer)
	matches := 0
	textLower := toLower(text)
	for _, word := range words {
		if contains(textLower, toLower(word)) {
			matches++
		}
	}
	return float64(matches) >= float64(len(words))*0.5
}

func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func toLower(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		b[i] = c
	}
	return string(b)
}

func splitWords(s string) []string {
	words := make([]string, 0)
	start := -1
	for i, c := range s {
		isLetter := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
		if isLetter {
			if start == -1 {
				start = i
			}
		} else {
			if start != -1 {
				words = append(words, s[start:i])
				start = -1
			}
		}
	}
	if start != -1 {
		words = append(words, s[start:])
	}
	return words
}

func calculateSubMetricStats(measurements []Measurement) SubMetricStats {
	embeddingTimes := make([]int64, 0)
	llmTimes := make([]int64, 0)
	dbTimes := make([]int64, 0)
	processingTimes := make([]int64, 0)

	for _, m := range measurements {
		if m.EmbeddingLatency > 0 {
			embeddingTimes = append(embeddingTimes, m.EmbeddingLatency)
		}
		if m.LLMLatency > 0 {
			llmTimes = append(llmTimes, m.LLMLatency)
		}
		if m.DBLatency > 0 {
			dbTimes = append(dbTimes, m.DBLatency)
		}
		if m.ProcessingLatency > 0 {
			processingTimes = append(processingTimes, m.ProcessingLatency)
		}
	}

	return SubMetricStats{
		EmbeddingLatency:  calculateLatencyStats(embeddingTimes),
		LLMLatency:        calculateLatencyStats(llmTimes),
		DBLatency:         calculateLatencyStats(dbTimes),
		ProcessingLatency: calculateLatencyStats(processingTimes),
	}
}

func distanceBucket(distance int) string {
	switch {
	case distance <= 10:
		return "0-10"
	case distance <= 25:
		return "11-25"
	case distance <= 50:
		return "26-50"
	default:
		return "50+"
	}
}

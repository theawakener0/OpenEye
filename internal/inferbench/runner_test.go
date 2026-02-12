package inferbench

import (
	"testing"
	"time"
)

func TestComputeFloatStats(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		s := computeFloatStats(nil)
		if s.Min != 0 || s.Max != 0 || s.Mean != 0 {
			t.Errorf("expected zero stats for empty input, got %+v", s)
		}
	})

	t.Run("single value", func(t *testing.T) {
		s := computeFloatStats([]float64{42.0})
		if s.Min != 42 || s.Max != 42 || s.Mean != 42 || s.Median != 42 {
			t.Errorf("single value stats wrong: %+v", s)
		}
	})

	t.Run("multiple values", func(t *testing.T) {
		s := computeFloatStats([]float64{10, 20, 30, 40, 50})
		if s.Min != 10 {
			t.Errorf("Min = %f, want 10", s.Min)
		}
		if s.Max != 50 {
			t.Errorf("Max = %f, want 50", s.Max)
		}
		if s.Mean != 30 {
			t.Errorf("Mean = %f, want 30", s.Mean)
		}
		if s.Median != 30 {
			t.Errorf("Median = %f, want 30", s.Median)
		}
	})

	t.Run("even count median", func(t *testing.T) {
		s := computeFloatStats([]float64{10, 20, 30, 40})
		// Median of [10,20,30,40] = (20+30)/2 = 25
		if s.Median != 25 {
			t.Errorf("Median = %f, want 25", s.Median)
		}
	})

	t.Run("unsorted input", func(t *testing.T) {
		s := computeFloatStats([]float64{50, 10, 30, 20, 40})
		if s.Min != 10 || s.Max != 50 {
			t.Errorf("Min=%f Max=%f after unsorted input", s.Min, s.Max)
		}
	})
}

func TestComputeDurationStats(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		s := computeDurationStats(nil)
		if s.Min != 0 || s.Max != 0 || s.Mean != 0 {
			t.Errorf("expected zero stats for empty input, got %+v", s)
		}
	})

	t.Run("basic", func(t *testing.T) {
		vals := []time.Duration{
			100 * time.Millisecond,
			200 * time.Millisecond,
			300 * time.Millisecond,
		}
		s := computeDurationStats(vals)
		if s.Min != 100*time.Millisecond {
			t.Errorf("Min = %v, want 100ms", s.Min)
		}
		if s.Max != 300*time.Millisecond {
			t.Errorf("Max = %v, want 300ms", s.Max)
		}
		if s.Mean != 200*time.Millisecond {
			t.Errorf("Mean = %v, want 200ms", s.Mean)
		}
		if s.Median != 200*time.Millisecond {
			t.Errorf("Median = %v, want 200ms", s.Median)
		}
	})
}

func TestPercentileIndex(t *testing.T) {
	tests := []struct {
		n, pct, want int
	}{
		{0, 95, 0},    // edge: empty
		{1, 95, 0},    // single element
		{5, 95, 4},    // p95 of 5 items = index 4
		{10, 50, 4},   // p50 of 10 items = index 4
		{100, 95, 94}, // p95 of 100 items = index 94
		{100, 99, 98}, // p99 of 100 items = index 98
		{20, 95, 18},  // p95 of 20 items = index 18
	}

	for _, tt := range tests {
		got := percentileIndex(tt.n, tt.pct)
		if got != tt.want {
			t.Errorf("percentileIndex(%d, %d) = %d, want %d", tt.n, tt.pct, got, tt.want)
		}
	}
}

func TestFilterValid(t *testing.T) {
	results := []IterationResult{
		{PromptName: "a", Error: ""},
		{PromptName: "b", Error: "something broke"},
		{PromptName: "c", Error: ""},
	}
	valid := filterValid(results)
	if len(valid) != 2 {
		t.Errorf("filterValid returned %d results, want 2", len(valid))
	}
}

func TestFilterByName(t *testing.T) {
	results := []IterationResult{
		{PromptName: "short"},
		{PromptName: "long"},
		{PromptName: "short"},
		{PromptName: "short-cached"},
	}
	filtered := filterByName(results, "short")
	if len(filtered) != 2 {
		t.Errorf("filterByName(short) returned %d, want 2", len(filtered))
	}
	cached := filterByName(results, "short-cached")
	if len(cached) != 1 {
		t.Errorf("filterByName(short-cached) returned %d, want 1", len(cached))
	}
}

func TestSummarizeSpeculativeStats(t *testing.T) {
	prompt := Prompt{Name: "test", Text: "hello"}
	results := []IterationResult{
		{
			PromptName:         "test",
			TTFT:               100 * time.Millisecond,
			Duration:           500 * time.Millisecond,
			TokensGenerated:    50,
			GenerationTPS:      100.0,
			SpecDrafted:        20,
			SpecAccepted:       16,
			SpecAcceptanceRate: 80.0,
		},
		{
			PromptName:         "test",
			TTFT:               120 * time.Millisecond,
			Duration:           600 * time.Millisecond,
			TokensGenerated:    60,
			GenerationTPS:      100.0,
			SpecDrafted:        25,
			SpecAccepted:       22,
			SpecAcceptanceRate: 88.0,
		},
	}

	summary := summarize(prompt, results)

	if !summary.SpeculativeActive {
		t.Fatal("SpeculativeActive should be true")
	}
	if summary.AvgSpecDrafted != 22.5 {
		t.Errorf("AvgSpecDrafted = %f, want 22.5", summary.AvgSpecDrafted)
	}
	if summary.AvgSpecAccepted != 19.0 {
		t.Errorf("AvgSpecAccepted = %f, want 19.0", summary.AvgSpecAccepted)
	}
	// Mean acceptance rate should be (80+88)/2 = 84
	if summary.SpecAcceptanceRate.Mean != 84.0 {
		t.Errorf("SpecAcceptanceRate.Mean = %f, want 84.0", summary.SpecAcceptanceRate.Mean)
	}
}

func TestSummarizeNoSpeculative(t *testing.T) {
	prompt := Prompt{Name: "test", Text: "hello"}
	results := []IterationResult{
		{
			PromptName:      "test",
			TTFT:            100 * time.Millisecond,
			Duration:        500 * time.Millisecond,
			TokensGenerated: 50,
			GenerationTPS:   100.0,
		},
	}

	summary := summarize(prompt, results)
	if summary.SpeculativeActive {
		t.Error("SpeculativeActive should be false when no spec stats")
	}
}

func TestSummarizeStreamingStats(t *testing.T) {
	prompt := Prompt{Name: "test", Text: "hello"}
	results := []IterationResult{
		{
			PromptName:      "test",
			TTFT:            100 * time.Millisecond,
			Duration:        500 * time.Millisecond,
			TokensGenerated: 50,
			GenerationTPS:   100.0,
			ChunkCount:      20,
			AvgChunkLatency: 25 * time.Millisecond,
			MaxChunkLatency: 50 * time.Millisecond,
		},
		{
			PromptName:      "test",
			TTFT:            120 * time.Millisecond,
			Duration:        600 * time.Millisecond,
			TokensGenerated: 60,
			GenerationTPS:   100.0,
			ChunkCount:      30,
			AvgChunkLatency: 20 * time.Millisecond,
			MaxChunkLatency: 40 * time.Millisecond,
		},
	}

	summary := summarize(prompt, results)

	if !summary.StreamMode {
		t.Fatal("StreamMode should be true")
	}
	if summary.AvgChunkCount != 25.0 {
		t.Errorf("AvgChunkCount = %f, want 25.0", summary.AvgChunkCount)
	}
}

func TestSummarizeCacheTest(t *testing.T) {
	prompt := Prompt{Name: "cache", Text: "test", Repeat: true}
	results := []IterationResult{
		// Cold run
		{PromptName: "cache", TTFT: 200 * time.Millisecond, Duration: 500 * time.Millisecond, TokensGenerated: 50, GenerationTPS: 100},
		// Cached run
		{PromptName: "cache-cached", TTFT: 50 * time.Millisecond, Duration: 400 * time.Millisecond, TokensGenerated: 50, GenerationTPS: 125},
	}

	summary := summarize(prompt, results)
	// Cache improvement should be (200-50)/200 * 100 = 75%
	if summary.CacheHitImprove != 75.0 {
		t.Errorf("CacheHitImprove = %f, want 75.0", summary.CacheHitImprove)
	}
}

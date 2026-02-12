package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/context/memory"
	"OpenEye/internal/context/memory/omem"
	"OpenEye/internal/embedding"
	_ "OpenEye/internal/native"
)

// RetrievalTest represents a test case with fact and query
type RetrievalTest struct {
	Fact     string   // The fact to store
	Queries  []string // Different ways to ask about the fact
	Category string   // Category of the fact
}

// Test dataset with varied queries for better retrieval testing
var retrievalTests = []RetrievalTest{
	{
		Fact:     "My name is Alice Johnson",
		Queries:  []string{"What is my name?", "Who am I?", "What do people call me?"},
		Category: "personal",
	},
	{
		Fact:     "I live in San Francisco, California",
		Queries:  []string{"Where do I live?", "What city am I in?", "My location?"},
		Category: "location",
	},
	{
		Fact:     "I work as a software engineer at Google",
		Queries:  []string{"Where do I work?", "What is my job?", "My company?"},
		Category: "work",
	},
	{
		Fact:     "My favorite hobby is hiking in the mountains",
		Queries:  []string{"What is my favorite hobby?", "What do I do for fun?", "My hobby?"},
		Category: "hobby",
	},
	{
		Fact:     "I am allergic to peanuts",
		Queries:  []string{"What food am I allergic to?", "My allergies?", "What can't I eat?"},
		Category: "health",
	},
	{
		Fact:     "I prefer tea over coffee",
		Queries:  []string{"Do I prefer tea or coffee?", "What do I drink?", "My beverage preference?"},
		Category: "preference",
	},
	{
		Fact:     "I want to learn Spanish this year",
		Queries:  []string{"What is my goal for this year?", "What language do I want to learn?", "My learning goal?"},
		Category: "goal",
	},
	{
		Fact:     "I visited Japan in 2019 and loved Tokyo",
		Queries:  []string{"Where did I visit in 2019?", "What country did I travel to?", "My Japan trip?"},
		Category: "travel",
	},
	{
		Fact:     "My mother's name is Sarah",
		Queries:  []string{"What is my mother's name?", "Who is my mom?", "My parent's name?"},
		Category: "family",
	},
	{
		Fact:     "I graduated from MIT with a CS degree",
		Queries:  []string{"Where did I graduate from?", "What university did I attend?", "My education?"},
		Category: "education",
	},
}

// RetrievalResult holds results for one memory system
type RetrievalResult struct {
	SystemName      string        `json:"system_name"`
	TotalTests      int           `json:"total_tests"`
	TotalQueries    int           `json:"total_queries"`
	CorrectExact    int           `json:"correct_exact"`
	CorrectPartial  int           `json:"correct_partial"`
	Failed          int           `json:"failed"`
	ExactAccuracy   float64       `json:"exact_accuracy_pct"`
	PartialAccuracy float64       `json:"partial_accuracy_pct"`
	QueryResults    []QueryResult `json:"query_results"`
	AvgLatency      time.Duration `json:"avg_latency"`
}

// QueryResult holds result for individual query
type QueryResult struct {
	Fact      string        `json:"fact"`
	Query     string        `json:"query"`
	Retrieved string        `json:"retrieved"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
}

// createEmbedder creates embedding provider
func createEmbedder(modelPath string) (embedding.Provider, error) {
	cfg := config.EmbeddingConfig{
		Enabled: func() *bool { b := true; return &b }(),
		Backend: "native",
		Native: config.NativeEmbeddingConfig{
			ModelPath:   modelPath,
			ContextSize: 512,
			Threads:     4,
		},
	}
	return embedding.New(cfg)
}

// testLegacyRetrieval tests Legacy memory retrieval
func testLegacyRetrieval() (*RetrievalResult, error) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("TESTING: LEGACY MEMORY RETRIEVAL")
	fmt.Println(strings.Repeat("=", 80))

	result := &RetrievalResult{SystemName: "Legacy (SQLite)"}

	// Clean up
	os.Remove("retrieval_test_legacy.db")
	store, err := memory.NewStore("retrieval_test_legacy.db")
	if err != nil {
		return nil, err
	}
	defer store.Close()
	defer os.Remove("retrieval_test_legacy.db")

	// Store all facts
	fmt.Println("\n[STORAGE PHASE]")
	for i, test := range retrievalTests {
		err := store.Append("user", test.Fact)
		if err != nil {
			return nil, fmt.Errorf("failed to store fact %d: %w", i, err)
		}
		if (i+1)%3 == 0 {
			fmt.Printf("  Stored %d/%d facts...\n", i+1, len(retrievalTests))
		}
	}
	fmt.Printf("  ✓ Stored %d facts\n", len(retrievalTests))

	// Test retrieval
	fmt.Println("\n[RETRIEVAL PHASE]")
	result.QueryResults = make([]QueryResult, 0)
	var totalLatency time.Duration

	for _, test := range retrievalTests {
		for _, query := range test.Queries {
			start := time.Now()
			entries, err := store.Search(query, 3)
			latency := time.Since(start)
			totalLatency += latency

			qr := QueryResult{
				Fact:    test.Fact,
				Query:   query,
				Latency: latency,
			}

			if err != nil {
				qr.Retrieved = "ERROR: " + err.Error()
				result.Failed++
			} else if len(entries) > 0 {
				qr.Retrieved = entries[0].Content

				// Check exact match
				if strings.Contains(strings.ToLower(entries[0].Content), strings.ToLower(test.Fact)) {
					qr.Correct = true
					result.CorrectExact++
				} else if strings.Contains(strings.ToLower(test.Fact), strings.ToLower(query)) ||
					strings.Contains(strings.ToLower(query), strings.ToLower(test.Fact)) {
					// Partial match
					qr.Correct = true
					result.CorrectPartial++
				}
			} else {
				qr.Retrieved = "NO RESULTS"
			}

			result.QueryResults = append(result.QueryResults, qr)
			result.TotalQueries++
		}
	}

	result.TotalTests = len(retrievalTests)
	result.AvgLatency = totalLatency / time.Duration(result.TotalQueries)
	result.ExactAccuracy = float64(result.CorrectExact) / float64(result.TotalQueries) * 100
	result.PartialAccuracy = float64(result.CorrectExact+result.CorrectPartial) / float64(result.TotalQueries) * 100

	fmt.Printf("  Total queries: %d\n", result.TotalQueries)
	fmt.Printf("  Exact matches: %d (%.1f%%)\n", result.CorrectExact, result.ExactAccuracy)
	fmt.Printf("  Partial matches: %d (%.1f%%)\n", result.CorrectPartial, result.PartialAccuracy)
	fmt.Printf("  Failed: %d\n", result.Failed)
	fmt.Printf("  Avg latency: %v\n", result.AvgLatency)

	return result, nil
}

// testOmemRetrieval tests Omem retrieval with real embeddings
func testOmemRetrieval(modelPath string, embedder embedding.Provider) (*RetrievalResult, error) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Printf("TESTING: OMEM RETRIEVAL (%s)\n", modelPath)
	fmt.Println(strings.Repeat("=", 80))

	result := &RetrievalResult{SystemName: fmt.Sprintf("Omem (%s)", modelPath)}

	dbPath := fmt.Sprintf("retrieval_test_omem_%s.duckdb", strings.ReplaceAll(modelPath, "/", "_"))
	os.Remove(dbPath)

	// Create Omem engine
	cfg := omem.DefaultConfig()
	cfg.Enabled = true
	cfg.Storage.DBPath = dbPath
	cfg.AtomicEncoder.Enabled = true
	cfg.MultiViewIndex.Enabled = true
	cfg.MultiViewIndex.ExtractKeywords = true
	cfg.EntityGraph.Enabled = true

	engine, err := omem.NewEngine(cfg)
	if err != nil {
		return nil, err
	}

	// Initialize with real embedder
	llmFunc := func(ctx context.Context, prompt string) (string, error) { return "", nil }
	embedFunc := func(ctx context.Context, text string) ([]float32, error) {
		return embedder.Embed(ctx, text)
	}

	err = engine.Initialize(llmFunc, embedFunc)
	if err != nil {
		return nil, err
	}

	defer engine.Close()
	defer os.Remove(dbPath)

	ctx := context.Background()

	// Store all facts
	fmt.Println("\n[STORAGE PHASE]")
	for i, test := range retrievalTests {
		_, err := engine.ProcessText(ctx, test.Fact, "user")
		if err != nil {
			return nil, fmt.Errorf("failed to store fact %d: %w", i, err)
		}
		if (i+1)%3 == 0 {
			fmt.Printf("  Stored %d/%d facts with embeddings...\n", i+1, len(retrievalTests))
		}
	}
	fmt.Printf("  ✓ Stored %d facts with real embeddings\n", len(retrievalTests))

	// Test retrieval
	fmt.Println("\n[RETRIEVAL PHASE]")
	result.QueryResults = make([]QueryResult, 0)
	var totalLatency time.Duration

	for _, test := range retrievalTests {
		for _, query := range test.Queries {
			start := time.Now()
			facts, err := engine.GetFacts(ctx, query)
			latency := time.Since(start)
			totalLatency += latency

			qr := QueryResult{
				Fact:    test.Fact,
				Query:   query,
				Latency: latency,
			}

			if err != nil {
				qr.Retrieved = "ERROR: " + err.Error()
				result.Failed++
			} else if len(facts) > 0 {
				qr.Retrieved = facts[0].Fact.Text

				// Check for semantic match
				factLower := strings.ToLower(test.Fact)
				retrievedLower := strings.ToLower(facts[0].Fact.Text)
				queryLower := strings.ToLower(query)

				// Exact match
				if strings.Contains(retrievedLower, factLower) || strings.Contains(factLower, retrievedLower) {
					qr.Correct = true
					result.CorrectExact++
				} else if strings.Contains(retrievedLower, test.Category) ||
					containsKeywordMatch(retrievedLower, queryLower) {
					// Semantic/keyword match
					qr.Correct = true
					result.CorrectPartial++
				}
			} else {
				qr.Retrieved = "NO RESULTS"
			}

			result.QueryResults = append(result.QueryResults, qr)
			result.TotalQueries++
		}
	}

	result.TotalTests = len(retrievalTests)
	result.AvgLatency = totalLatency / time.Duration(result.TotalQueries)
	result.ExactAccuracy = float64(result.CorrectExact) / float64(result.TotalQueries) * 100
	result.PartialAccuracy = float64(result.CorrectExact+result.CorrectPartial) / float64(result.TotalQueries) * 100

	fmt.Printf("  Total queries: %d\n", result.TotalQueries)
	fmt.Printf("  Exact matches: %d (%.1f%%)\n", result.CorrectExact, result.ExactAccuracy)
	fmt.Printf("  Partial matches: %d (%.1f%%)\n", result.CorrectPartial, result.PartialAccuracy)
	fmt.Printf("  Failed: %d\n", result.Failed)
	fmt.Printf("  Avg latency: %v\n", result.AvgLatency)

	return result, nil
}

func containsKeywordMatch(text, query string) bool {
	// Extract keywords from query (words > 3 chars)
	queryWords := strings.Fields(query)
	matchCount := 0
	for _, word := range queryWords {
		if len(word) > 3 && strings.Contains(text, strings.ToLower(word)) {
			matchCount++
		}
	}
	return matchCount >= 2
}

func printComparisonTable(results []*RetrievalResult) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("RETRIEVAL ACCURACY COMPARISON")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println()

	fmt.Printf("%-30s", "Metric")
	for _, r := range results {
		name := r.SystemName
		if len(name) > 20 {
			name = name[:17] + "..."
		}
		fmt.Printf("%-22s", name)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 100))

	// Tests and queries
	fmt.Printf("%-30s", "Total Tests (Facts)")
	for _, r := range results {
		fmt.Printf("%-22d", r.TotalTests)
	}
	fmt.Println()

	fmt.Printf("%-30s", "Total Queries")
	for _, r := range results {
		fmt.Printf("%-22d", r.TotalQueries)
	}
	fmt.Println()

	fmt.Println(strings.Repeat("-", 100))

	// Accuracy scores
	fmt.Printf("%-30s", "EXACT MATCH ACCURACY")
	for _, r := range results {
		fmt.Printf("%-22.1f%%", r.ExactAccuracy)
	}
	fmt.Println()

	fmt.Printf("%-30s", "PARTIAL MATCH ACCURACY")
	for _, r := range results {
		fmt.Printf("%-22.1f%%", r.PartialAccuracy)
	}
	fmt.Println()

	fmt.Println(strings.Repeat("-", 100))

	// Breakdown
	fmt.Printf("%-30s", "Exact Matches")
	for _, r := range results {
		fmt.Printf("%-22d", r.CorrectExact)
	}
	fmt.Println()

	fmt.Printf("%-30s", "Partial Matches")
	for _, r := range results {
		fmt.Printf("%-22d", r.CorrectPartial)
	}
	fmt.Println()

	fmt.Printf("%-30s", "Failed Queries")
	for _, r := range results {
		fmt.Printf("%-22d", r.Failed)
	}
	fmt.Println()

	fmt.Println(strings.Repeat("-", 100))

	// Latency
	fmt.Printf("%-30s", "Avg Query Latency")
	for _, r := range results {
		fmt.Printf("%-22v", r.AvgLatency)
	}
	fmt.Println()

	fmt.Println()
}

func printDetailedAnalysis(results []*RetrievalResult) {
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println("DETAILED RETRIEVAL ANALYSIS")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println()

	// Find best performer
	var bestExact, bestPartial *RetrievalResult
	bestExactPct := 0.0
	bestPartialPct := 0.0

	for _, r := range results {
		if r.ExactAccuracy > bestExactPct {
			bestExactPct = r.ExactAccuracy
			bestExact = r
		}
		if r.PartialAccuracy > bestPartialPct {
			bestPartialPct = r.PartialAccuracy
			bestPartial = r
		}
	}

	fmt.Printf("BEST EXACT MATCH RETRIEVAL: %s (%.1f%%)\n", bestExact.SystemName, bestExact.ExactAccuracy)
	fmt.Printf("BEST OVERALL RETRIEVAL: %s (%.1f%% partial match)\n", bestPartial.SystemName, bestPartial.PartialAccuracy)
	fmt.Println()

	// Show sample queries
	fmt.Println("SAMPLE QUERY RESULTS:")
	fmt.Println(strings.Repeat("-", 100))

	if len(results) > 0 && len(results[0].QueryResults) > 0 {
		for i := 0; i < 5 && i < len(results[0].QueryResults); i++ {
			qr := results[0].QueryResults[i]
			fmt.Printf("\nQuery %d: %s\n", i+1, qr.Query)
			fmt.Printf("  Expected: %s\n", qr.Fact)

			for _, r := range results {
				for _, queryResult := range r.QueryResults {
					if queryResult.Query == qr.Query {
						status := "✗"
						if queryResult.Correct {
							status = "✓"
						}
						retrieved := queryResult.Retrieved
						if len(retrieved) > 50 {
							retrieved = retrieved[:47] + "..."
						}
						systemName := r.SystemName
						if len(systemName) > 15 {
							systemName = systemName[:12] + "..."
						}
						fmt.Printf("  %s %-15s: %s\n", status, systemName, retrieved)
						break
					}
				}
			}
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("RETRIEVAL PERFORMANCE SUMMARY")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println()

	for _, r := range results {
		fmt.Printf("%s:\n", r.SystemName)
		fmt.Printf("  • Exact Match Accuracy:  %.1f%% (%d/%d)\n", r.ExactAccuracy, r.CorrectExact, r.TotalQueries)
		fmt.Printf("  • Partial Match Accuracy: %.1f%% (%d/%d)\n", r.PartialAccuracy, r.CorrectExact+r.CorrectPartial, r.TotalQueries)
		fmt.Printf("  • Average Latency:       %v\n", r.AvgLatency)
		fmt.Println()
	}
}

func main() {
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println("OPENEYE MEMORY RETRIEVAL ACCURACY BENCHMARK")
	fmt.Println("Comparing retrieval performance with real embeddings")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println()
	fmt.Printf("Test Dataset: %d facts with %d total queries\n", len(retrievalTests), len(retrievalTests)*3)
	fmt.Println("Each fact tested with 3 different query formulations")
	fmt.Println()

	results := make([]*RetrievalResult, 0, 3)

	// Test Legacy
	legacyResult, err := testLegacyRetrieval()
	if err != nil {
		fmt.Printf("Legacy test failed: %v\n", err)
	} else {
		results = append(results, legacyResult)
	}

	// Test Omem with all-MiniLM
	fmt.Println("\nLoading embedding model: all-MiniLM-L6-v2-Q4_K_M.gguf...")
	embedder1, err := createEmbedder("models/all-MiniLM-L6-v2-Q4_K_M.gguf")
	if err != nil {
		fmt.Printf("Failed to load all-MiniLM: %v\n", err)
	} else {
		testVec, _ := embedder1.Embed(context.Background(), "test")
		fmt.Printf("✓ Model loaded (dimension: %d)\n", len(testVec))

		omemResult, err := testOmemRetrieval("all-MiniLM-L6-v2", embedder1)
		if err != nil {
			fmt.Printf("Omem test failed: %v\n", err)
		} else {
			results = append(results, omemResult)
		}
		embedder1.Close()
	}

	// Note about Mem0
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("NOTE: Mem0 requires full runtime manager with LLM")
	fmt.Println("Skipping Mem0 in standalone benchmark")
	fmt.Println("Use './openeye-native memory-benchmark' for complete Mem0 testing")
	fmt.Println(strings.Repeat("=", 80))

	// Print results
	if len(results) > 0 {
		printComparisonTable(results)
		printDetailedAnalysis(results)

		// Save results
		jsonData, _ := json.MarshalIndent(results, "", "  ")
		os.WriteFile("retrieval_accuracy_results.json", jsonData, 0644)
		fmt.Println("\n✓ Results saved to retrieval_accuracy_results.json")
	}

	fmt.Println()
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println("RETRIEVAL BENCHMARK COMPLETE")
	fmt.Println(strings.Repeat("=", 100))
}

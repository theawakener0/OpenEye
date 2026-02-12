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

// ProductionTest represents a test case
// Paper claims: 50-turn conversations, 10 planted facts per conversation
type ProductionTest struct {
	Fact     string
	Query    string
	Category string
}

// 50 facts like the paper's benchmark
var productionTests = []ProductionTest{
	// Personal (10 facts)
	{"My name is Alice Johnson", "What is my name?", "personal"},
	{"I was born on March 15, 1990", "When is my birthday?", "personal"},
	{"I grew up in Boston Massachusetts", "Where did I grow up?", "personal"},
	{"My favorite color is blue", "What is my favorite color?", "personal"},
	{"I am 5 feet 6 inches tall", "How tall am I?", "personal"},
	{"My blood type is O positive", "What is my blood type?", "personal"},
	{"I have brown eyes", "What color are my eyes?", "personal"},
	{"My middle name is Marie", "What is my middle name?", "personal"},
	{"I am left-handed", "Which hand do I write with?", "personal"},
	{"My zodiac sign is Pisces", "What is my zodiac sign?", "personal"},

	// Location/Work (10 facts)
	{"I live in San Francisco California", "Where do I live?", "location"},
	{"I work as a software engineer at Google", "Where do I work?", "work"},
	{"My office is in Mountain View", "Where is my office?", "work"},
	{"I have been working for 8 years", "How long have I been working?", "work"},
	{"My apartment is on the 3rd floor", "What floor do I live on?", "location"},
	{"I moved to California in 2018", "When did I move to California?", "location"},
	{"My commute takes 45 minutes", "How long is my commute?", "work"},
	{"I work from home on Fridays", "When do I work from home?", "work"},
	{"My team has 12 people", "How big is my team?", "work"},
	{"I have a standing desk at work", "What kind of desk do I have?", "work"},

	// Hobbies/Interests (10 facts)
	{"My favorite hobby is hiking in the mountains", "What is my favorite hobby?", "hobby"},
	{"I enjoy playing the guitar", "What instrument do I play?", "hobby"},
	{"I like to read science fiction books", "What genre do I read?", "hobby"},
	{"I run 5 kilometers every morning", "What is my exercise routine?", "hobby"},
	{"I collect vintage vinyl records", "What do I collect?", "hobby"},
	{"I love watching documentaries", "What type of movies do I watch?", "hobby"},
	{"I paint watercolor landscapes", "What kind of painting do I do?", "hobby"},
	{"I practice yoga on weekends", "What exercise do I do on weekends?", "hobby"},
	{"I play chess competitively", "What game do I play competitively?", "hobby"},
	{"I enjoy cooking Italian food", "What cuisine do I cook?", "hobby"},

	// Preferences/Food (10 facts)
	{"I prefer tea over coffee", "Do I prefer tea or coffee?", "preference"},
	{"I am allergic to peanuts", "What food am I allergic to?", "health"},
	{"My favorite food is Italian pasta", "What is my favorite food?", "food"},
	{"I like working from home better", "Where do I prefer to work?", "preference"},
	{"I prefer cats over dogs", "Do I like cats or dogs?", "preference"},
	{"I enjoy winter more than summer", "Which season do I prefer?", "preference"},
	{"I like spicy food", "Do I like spicy food?", "preference"},
	{"I prefer reading physical books", "Do I prefer ebooks or physical books?", "preference"},
	{"I like to wake up early", "Am I a morning person?", "preference"},
	{"I prefer Android over iPhone", "What phone do I prefer?", "preference"},

	// Goals/Memories (10 facts)
	{"My goal is to learn Spanish this year", "What is my goal for this year?", "goal"},
	{"I want to run a marathon", "What athletic goal do I have?", "goal"},
	{"I visited Japan in 2019", "Where did I visit in 2019?", "travel"},
	{"My first pet was a golden retriever", "What was my first pet?", "memory"},
	{"I graduated from MIT in 2015", "When did I graduate?", "education"},
	{"I have two siblings", "How many siblings do I have?", "family"},
	{"My mother's name is Sarah", "What is my mother's name?", "family"},
	{"I learned to drive at 16", "When did I learn to drive?", "memory"},
	{"I want to visit New Zealand", "Where do I want to visit?", "goal"},
	{"I have a master's degree in CS", "What degree do I have?", "education"},
}

type ProductionResult struct {
	SystemName   string  `json:"system_name"`
	TotalFacts   int     `json:"total_facts"`
	Correct      int     `json:"correct"`
	Accuracy     float64 `json:"accuracy_pct"`
	Top3Accuracy float64 `json:"top3_accuracy_pct"`
	AvgLatencyMs float64 `json:"avg_latency_ms"`
	MinScore     float64 `json:"min_score"`
	MaxScore     float64 `json:"max_score"`
	AvgScore     float64 `json:"avg_score"`
}

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

func testLegacyProduction() (*ProductionResult, error) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("PRODUCTION TEST: LEGACY (SQLite)")
	fmt.Println("Expected: ~0% recall (recent-only, no semantic search)")
	fmt.Println(strings.Repeat("=", 80))

	result := &ProductionResult{SystemName: "Legacy (SQLite)"}

	os.Remove("prod_legacy.db")
	store, err := memory.NewStore("prod_legacy.db")
	if err != nil {
		return nil, err
	}
	defer store.Close()
	defer os.Remove("prod_legacy.db")

	// Store all 50 facts
	fmt.Println("\n[STORAGE] Storing 50 facts...")
	for _, test := range productionTests {
		store.Append("user", test.Fact)
	}
	fmt.Println("✓ Stored 50 facts")

	// Test retrieval
	fmt.Println("\n[RETRIEVAL] Testing recall...")
	correct := 0
	var totalLatency time.Duration

	for _, test := range productionTests {
		start := time.Now()
		entries, _ := store.Search(test.Query, 5)
		latency := time.Since(start)
		totalLatency += latency

		for _, e := range entries {
			factPrefix := test.Fact
			if len(factPrefix) > 20 {
				factPrefix = factPrefix[:20]
			}
			if strings.Contains(strings.ToLower(e.Content), strings.ToLower(factPrefix)) {
				correct++
				break
			}
		}
	}

	result.TotalFacts = len(productionTests)
	result.Correct = correct
	result.Accuracy = float64(correct) / float64(len(productionTests)) * 100
	result.AvgLatencyMs = float64(totalLatency.Milliseconds()) / float64(len(productionTests))

	fmt.Printf("✓ Results: %d/%d (%.1f%%)\n", correct, len(productionTests), result.Accuracy)
	fmt.Printf("✓ Avg Latency: %.2f ms\n", result.AvgLatencyMs)

	return result, nil
}

func testOmemProduction(modelPath string, embedder embedding.Provider) (*ProductionResult, error) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Printf("PRODUCTION TEST: OMEM (%s)\n", modelPath)
	fmt.Println("Target: 82% recall (per paper)")
	fmt.Println(strings.Repeat("=", 80))

	result := &ProductionResult{SystemName: fmt.Sprintf("Omem (%s)", modelPath)}

	dbPath := "prod_omem.duckdb"
	os.Remove(dbPath)

	cfg := omem.DefaultConfig()
	cfg.Storage.DBPath = dbPath
	cfg.Retrieval.MinScore = 0.0 // No filtering, let us see all scores
	cfg.Retrieval.DefaultTopK = 10

	engine, err := omem.NewEngine(cfg)
	if err != nil {
		return nil, err
	}

	err = engine.Initialize(
		func(ctx context.Context, prompt string) (string, error) { return "", nil },
		func(ctx context.Context, text string) ([]float32, error) { return embedder.Embed(ctx, text) },
	)
	if err != nil {
		return nil, err
	}
	defer engine.Close()
	defer os.Remove(dbPath)

	ctx := context.Background()

	// Store all 50 facts
	fmt.Println("\n[STORAGE] Storing 50 facts with embeddings...")
	for i, test := range productionTests {
		_, err := engine.ProcessText(ctx, test.Fact, "user")
		if err != nil {
			fmt.Printf("  Warning: fact %d failed: %v\n", i+1, err)
		}
	}

	stats := engine.GetStats(ctx)
	if storeStats, ok := stats["store"].(map[string]interface{}); ok {
		fmt.Printf("✓ Stored facts: %v\n", storeStats["total_facts"])
	}

	// Test retrieval
	fmt.Println("\n[RETRIEVAL] Testing semantic recall...")
	correct := 0
	top3Correct := 0
	var totalLatency time.Duration
	minScore := 1.0
	maxScore := 0.0
	var totalScore float64
	scoreCount := 0

	for _, test := range productionTests {
		start := time.Now()
		facts, _ := engine.GetFacts(ctx, test.Query)
		latency := time.Since(start)
		totalLatency += latency

		if len(facts) > 0 {
			factPrefix := test.Fact
			if len(factPrefix) > 20 {
				factPrefix = factPrefix[:20]
			}

			// Check top result
			if strings.Contains(strings.ToLower(facts[0].Fact.Text), strings.ToLower(factPrefix)) {
				correct++
			}

			// Check top 3
			for i := 0; i < min(3, len(facts)); i++ {
				if strings.Contains(strings.ToLower(facts[i].Fact.Text), strings.ToLower(factPrefix)) {
					top3Correct++
					break
				}
			}

			// Track scores
			score := facts[0].Score
			totalScore += score
			scoreCount++
			if score < minScore {
				minScore = score
			}
			if score > maxScore {
				maxScore = score
			}
		}
	}

	result.TotalFacts = len(productionTests)
	result.Correct = correct
	result.Accuracy = float64(correct) / float64(len(productionTests)) * 100
	result.Top3Accuracy = float64(top3Correct) / float64(len(productionTests)) * 100
	result.AvgLatencyMs = float64(totalLatency.Milliseconds()) / float64(len(productionTests))

	if scoreCount > 0 {
		result.MinScore = minScore
		result.MaxScore = maxScore
		result.AvgScore = totalScore / float64(scoreCount)
	}

	fmt.Printf("✓ Results: %d/%d (%.1f%%)\n", correct, len(productionTests), result.Accuracy)
	fmt.Printf("✓ Top-3 Results: %d/%d (%.1f%%)\n", top3Correct, len(productionTests), result.Top3Accuracy)
	fmt.Printf("✓ Score Range: %.3f - %.3f (avg: %.3f)\n", minScore, maxScore, result.AvgScore)
	fmt.Printf("✓ Avg Latency: %.2f ms\n", result.AvgLatencyMs)

	return result, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func printProductionResults(results []*ProductionResult) {
	fmt.Println("\n" + strings.Repeat("=", 100))
	fmt.Println("PRODUCTION BENCHMARK RESULTS")
	fmt.Println("Paper Claims: Omem = 82% recall, Legacy = 0% recall")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println()

	fmt.Printf("%-30s", "Metric")
	for _, r := range results {
		name := r.SystemName
		if len(name) > 20 {
			name = name[:17] + "..."
		}
		fmt.Printf("%-25s", name)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 100))

	metrics := []struct {
		label string
		fn    func(*ProductionResult) string
	}{
		{"Facts Stored", func(r *ProductionResult) string { return fmt.Sprintf("%d", r.TotalFacts) }},
		{"Top-1 Accuracy", func(r *ProductionResult) string { return fmt.Sprintf("%.1f%%", r.Accuracy) }},
		{"Top-3 Accuracy", func(r *ProductionResult) string { return fmt.Sprintf("%.1f%%", r.Top3Accuracy) }},
		{"Correct / Total", func(r *ProductionResult) string { return fmt.Sprintf("%d/%d", r.Correct, r.TotalFacts) }},
		{"Avg Latency", func(r *ProductionResult) string { return fmt.Sprintf("%.1f ms", r.AvgLatencyMs) }},
		{"Min Score", func(r *ProductionResult) string { return fmt.Sprintf("%.3f", r.MinScore) }},
		{"Max Score", func(r *ProductionResult) string { return fmt.Sprintf("%.3f", r.MaxScore) }},
		{"Avg Score", func(r *ProductionResult) string { return fmt.Sprintf("%.3f", r.AvgScore) }},
	}

	for _, m := range metrics {
		fmt.Printf("%-30s", m.label)
		for _, r := range results {
			fmt.Printf("%-25s", m.fn(r))
		}
		fmt.Println()
	}

	fmt.Println()
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println("ANALYSIS")
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println()

	for _, r := range results {
		fmt.Printf("%s:\n", r.SystemName)
		fmt.Printf("  Recall Accuracy:    %.1f%% (target: %s)\n",
			r.Accuracy,
			map[bool]string{true: "82%", false: "0%"}[r.SystemName != "Legacy (SQLite)"])

		if r.Accuracy >= 80 {
			fmt.Printf("  Status:            ✓ PRODUCTION READY\n")
		} else if r.Accuracy >= 60 {
			fmt.Printf("  Status:            ⚠ NEEDS IMPROVEMENT\n")
		} else {
			fmt.Printf("  Status:            ✗ NOT READY\n")
		}

		if r.Accuracy < 10 && r.SystemName != "Legacy (SQLite)" {
			fmt.Printf("  Issue:             Very low recall - check similarity scores\n")
			fmt.Printf("  Score Range:       %.3f - %.3f\n", r.MinScore, r.MaxScore)
		}
		fmt.Println()
	}

	// Compare to paper
	fmt.Println("COMPARISON TO PAPER:")
	for _, r := range results {
		target := 0.0
		if r.SystemName == "Legacy (SQLite)" {
			target = 0.0
		} else {
			target = 82.0
		}
		diff := r.Accuracy - target
		fmt.Printf("  %s: %.1f%% vs %.1f%% target (%+.1f%%)\n",
			r.SystemName, r.Accuracy, target, diff)
	}
}

func main() {
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println("OPENEYE MEMORY - PRODUCTION READINESS TEST")
	fmt.Println("Testing with 50 facts (like paper's benchmark)")
	fmt.Println("Paper Target: Omem = 82% recall")
	fmt.Println(strings.Repeat("=", 100))

	results := make([]*ProductionResult, 0, 2)

	// Test Legacy
	legacyResult, err := testLegacyProduction()
	if err != nil {
		fmt.Printf("Legacy test failed: %v\n", err)
	} else {
		results = append(results, legacyResult)
	}

	// Test Omem
	fmt.Println("\nLoading embedding model...")
	embedder, err := createEmbedder("models/all-MiniLM-L6-v2-Q4_K_M.gguf")
	if err != nil {
		fmt.Printf("Failed to load embedder: %v\n", err)
	} else {
		defer embedder.Close()

		vec, _ := embedder.Embed(context.Background(), "test")
		fmt.Printf("✓ Model loaded (dimension: %d)\n\n", len(vec))

		omemResult, err := testOmemProduction("all-MiniLM-L6-v2", embedder)
		if err != nil {
			fmt.Printf("Omem test failed: %v\n", err)
		} else {
			results = append(results, omemResult)
		}
	}

	// Print results
	if len(results) > 0 {
		printProductionResults(results)

		// Save to JSON
		jsonData, _ := json.MarshalIndent(results, "", "  ")
		os.WriteFile("production_test_results.json", jsonData, 0644)
		fmt.Println("\n✓ Results saved to production_test_results.json")
	}

	fmt.Println()
	fmt.Println(strings.Repeat("=", 100))
	fmt.Println("PRODUCTION TEST COMPLETE")
	fmt.Println(strings.Repeat("=", 100))
}

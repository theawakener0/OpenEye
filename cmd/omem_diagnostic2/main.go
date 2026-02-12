package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"OpenEye/internal/config"
	"OpenEye/internal/context/memory/omem"
	"OpenEye/internal/embedding"
	_ "OpenEye/internal/native"
)

func main() {
	fmt.Println("OMEM SEMANTIC SEARCH DIAGNOSTIC")
	fmt.Println(strings.Repeat("=", 80))

	// Clean up
	os.Remove("diagnostic_omem2.duckdb")

	// Create embedder
	fmt.Println("\n1. Loading embedding model...")
	embedder, err := embedding.New(config.EmbeddingConfig{
		Enabled: func() *bool { b := true; return &b }(),
		Backend: "native",
		Native: config.NativeEmbeddingConfig{
			ModelPath:   "models/all-MiniLM-L6-v2-Q4_K_M.gguf",
			ContextSize: 512,
			Threads:     4,
		},
	})
	if err != nil {
		fmt.Printf("Failed to create embedder: %v\n", err)
		return
	}
	defer embedder.Close()

	testVec, err := embedder.Embed(context.Background(), "test")
	if err != nil {
		fmt.Printf("Failed to embed: %v\n", err)
		return
	}
	fmt.Printf("   ✓ Embedding model loaded (dim: %d)\n", len(testVec))

	// Create Omem engine
	fmt.Println("\n2. Creating Omem engine...")
	cfg := omem.DefaultConfig()
	cfg.Storage.DBPath = "diagnostic_omem2.duckdb"
	cfg.Retrieval.MinScore = 0.0

	engine, err := omem.NewEngine(cfg)
	if err != nil {
		fmt.Printf("Failed to create engine: %v\n", err)
		return
	}

	// Initialize with embedder
	err = engine.Initialize(
		func(ctx context.Context, prompt string) (string, error) { return "", nil },
		func(ctx context.Context, text string) ([]float32, error) { return embedder.Embed(ctx, text) },
	)
	if err != nil {
		fmt.Printf("Failed to initialize: %v\n", err)
		return
	}
	defer engine.Close()
	defer os.Remove("diagnostic_omem2.duckdb")

	ctx := context.Background()

	// Store test facts with explicit embedding check
	fmt.Println("\n3. Storing test facts with embedding verification...")
	facts := []string{
		"My name is Alice Johnson",
		"I live in San Francisco California",
		"I work at Google as a software engineer",
	}

	for i, factText := range facts {
		// Generate embedding first
		emb, err := embedder.Embed(ctx, factText)
		if err != nil {
			fmt.Printf("   ✗ Failed to embed fact %d: %v\n", i+1, err)
			continue
		}
		fmt.Printf("   Fact %d embedding: dim=%d, first 5 values=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
			i+1, len(emb), emb[0], emb[1], emb[2], emb[3], emb[4])

		_, err = engine.ProcessText(ctx, factText, "user")
		if err != nil {
			fmt.Printf("   ✗ Failed to store fact %d: %v\n", i+1, err)
		} else {
			fmt.Printf("   ✓ Stored: %s\n", factText)
		}
	}

	// Get stats
	fmt.Println("\n4. Checking stored facts...")
	stats := engine.GetStats(ctx)
	if storeStats, ok := stats["store"].(map[string]interface{}); ok {
		fmt.Printf("   Total facts: %v\n", storeStats["total_facts"])
		fmt.Printf("   Active facts: %v\n", storeStats["active_facts"])
	}

	// Test with exact match query (same text as stored)
	fmt.Println("\n5. Testing retrieval with EXACT text match...")
	for _, fact := range facts {
		fmt.Printf("\n   Query (exact): '%s'\n", fact)
		results, err := engine.GetFacts(ctx, fact)
		if err != nil {
			fmt.Printf("     Error: %v\n", err)
		} else if len(results) == 0 {
			fmt.Printf("     No results\n")
		} else {
			fmt.Printf("     ✓ Got %d results:\n", len(results))
			for i, r := range results {
				fmt.Printf("       %d. Score=%.4f, Fact='%s'\n", i+1, r.Score, r.Fact.Text)
			}
		}
	}

	// Test with semantic queries
	fmt.Println("\n6. Testing retrieval with SEMANTIC queries...")
	queries := []struct {
		query  string
		expect string
	}{
		{"What is my name?", "Alice Johnson"},
		{"Where do I live?", "San Francisco"},
		{"Where do I work?", "Google"},
	}

	for _, q := range queries {
		fmt.Printf("\n   Query: '%s' (expecting: %s)\n", q.query, q.expect)

		// Get query embedding
		queryEmb, err := embedder.Embed(ctx, q.query)
		if err != nil {
			fmt.Printf("     ✗ Failed to embed query: %v\n", err)
			continue
		}
		fmt.Printf("     Query embedding: dim=%d, first 5=[%.4f, %.4f, %.4f, %.4f, %.4f]\n",
			len(queryEmb), queryEmb[0], queryEmb[1], queryEmb[2], queryEmb[3], queryEmb[4])

		results, err := engine.GetFacts(ctx, q.query)
		if err != nil {
			fmt.Printf("     Error: %v\n", err)
		} else if len(results) == 0 {
			fmt.Printf("     No results\n")
		} else {
			fmt.Printf("     ✓ Got %d results:\n", len(results))
			for i, r := range results {
				fmt.Printf("       %d. Score=%.4f, Fact='%s'\n", i+1, r.Score, r.Fact.Text)
			}
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("DIAGNOSTIC COMPLETE")
	fmt.Println("\nIf no results are returned even for exact matches,")
	fmt.Println("the issue is likely in the retrieval pipeline (not storage).")
}

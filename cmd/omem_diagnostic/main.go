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
	fmt.Println("OMEM RETRIEVAL DIAGNOSTIC")
	fmt.Println(strings.Repeat("=", 80))

	// Clean up
	os.Remove("diagnostic_omem.duckdb")

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

	// Test embedding
	testVec, err := embedder.Embed(context.Background(), "test")
	if err != nil {
		fmt.Printf("Failed to embed: %v\n", err)
		return
	}
	fmt.Printf("   ✓ Embedding model loaded (dim: %d)\n", len(testVec))

	// Create Omem engine
	fmt.Println("\n2. Creating Omem engine...")
	cfg := omem.DefaultConfig()
	cfg.Storage.DBPath = "diagnostic_omem.duckdb"
	cfg.Retrieval.MinScore = 0.0

	engine, err := omem.NewEngine(cfg)
	if err != nil {
		fmt.Printf("Failed to create engine: %v\n", err)
		return
	}

	// Initialize
	err = engine.Initialize(
		func(ctx context.Context, prompt string) (string, error) { return "", nil },
		func(ctx context.Context, text string) ([]float32, error) { return embedder.Embed(ctx, text) },
	)
	if err != nil {
		fmt.Printf("Failed to initialize: %v\n", err)
		return
	}
	defer engine.Close()
	defer os.Remove("diagnostic_omem.duckdb")

	ctx := context.Background()

	// Store test facts
	fmt.Println("\n3. Storing test facts...")
	facts := []string{
		"My name is Alice Johnson",
		"I live in San Francisco",
		"I work at Google",
	}

	for i, fact := range facts {
		_, err := engine.ProcessText(ctx, fact, "user")
		if err != nil {
			fmt.Printf("   ✗ Failed to store fact %d: %v\n", i+1, err)
		} else {
			fmt.Printf("   ✓ Stored: %s\n", fact)
		}
	}

	// Get stats
	fmt.Println("\n4. Getting stats...")
	stats := engine.GetStats(ctx)
	fmt.Printf("   Store stats: %v\n", stats)

	// Test retrieval
	fmt.Println("\n5. Testing retrieval...")
	queries := []string{
		"What is my name?",
		"Where do I live?",
		"Where do I work?",
	}

	for _, query := range queries {
		fmt.Printf("\n   Query: %s\n", query)
		facts, err := engine.GetFacts(ctx, query)
		if err != nil {
			fmt.Printf("     ✗ Error: %v\n", err)
		} else if len(facts) == 0 {
			fmt.Printf("     ✗ No results\n")
		} else {
			fmt.Printf("     ✓ Got %d results:\n", len(facts))
			for i, f := range facts {
				fmt.Printf("       %d. [%.3f] %s\n", i+1, f.Score, f.Fact.Text)
			}
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("DIAGNOSTIC COMPLETE")
}

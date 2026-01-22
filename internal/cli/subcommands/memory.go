package subcommands

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/context/memory"
)

// RunMemory provides memory inspection and management commands.
func RunMemory(cfg config.Config, args []string) int {
	fs := flag.NewFlagSet("memory", flag.ContinueOnError)
	limit := fs.Int("n", cfg.Memory.TurnsToUse, "Number of turns to display")
	showStats := fs.Bool("stats", false, "Show memory statistics")
	compress := fs.Bool("compress", false, "Trigger memory compression")
	vectorSearch := fs.String("search", "", "Search vector memory for a query")
	searchLimit := fs.Int("search-limit", 5, "Number of search results to return")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse flags: %v\n", err)
		return 1
	}

	ctx := context.Background()

	// Handle vector memory operations
	if cfg.Memory.VectorEnabled {
		return handleVectorMemoryOperations(ctx, cfg, *showStats, *compress, *vectorSearch, *searchLimit, *limit)
	}

	// Fallback to basic memory store
	return handleBasicMemory(cfg, *limit)
}

func handleVectorMemoryOperations(ctx context.Context, cfg config.Config, showStats, compress bool, searchQuery string, searchLimit, recentLimit int) int {
	// Parse compression age
	compressionAge := 24 * time.Hour
	if cfg.Memory.CompressionAge != "" {
		if parsed, err := time.ParseDuration(cfg.Memory.CompressionAge); err == nil {
			compressionAge = parsed
		}
	}

	engineCfg := memory.EngineConfig{
		DBPath:             cfg.Memory.VectorDBPath,
		EmbeddingDim:       cfg.Memory.EmbeddingDim,
		MaxContextTokens:   cfg.Memory.MaxContextTokens,
		ReservedForPrompt:  cfg.Memory.ReservedForPrompt,
		ReservedForSummary: cfg.Memory.ReservedForSummary,
		MinSimilarity:      cfg.Memory.MinSimilarity,
		SlidingWindowSize:  cfg.Memory.SlidingWindowSize,
		RecencyWeight:      cfg.Memory.RecencyWeight,
		RelevanceWeight:    cfg.Memory.RelevanceWeight,
		CompressionEnabled: cfg.Memory.CompressionEnabled,
		CompressionAge:     compressionAge,
		CompressBatchSize:  cfg.Memory.CompressBatchSize,
		AutoCompress:       cfg.Memory.AutoCompress,
		CompressEveryN:     cfg.Memory.CompressEveryN,
	}

	// Create engine without embedding/summarizer for read-only operations
	engine, err := memory.NewEngine(engineCfg, nil, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open vector memory: %v\n", err)
		// Fallback to basic memory
		return handleBasicMemory(cfg, recentLimit)
	}
	defer engine.Close()

	// Show statistics
	if showStats {
		stats, err := engine.GetStats(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to get stats: %v\n", err)
			return 1
		}

		fmt.Println("\033[1m=== Vector Memory Statistics ===\033[0m")
		fmt.Printf("Database: \033[36m%s\033[0m\n", cfg.Memory.VectorDBPath)
		fmt.Println()

		for key, value := range stats {
			fmt.Printf("  \033[34m%-20s\033[0m %v\n", formatStatKey(key)+":", value)
		}
		fmt.Println()
		return 0
	}

	// Trigger compression
	if compress {
		fmt.Println("Triggering memory compression...")
		if err := engine.Compress(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "compression failed: %v\n", err)
			return 1
		}
		fmt.Println("Memory compression completed.")
		return 0
	}

	// Vector search
	if searchQuery != "" {
		fmt.Printf("Note: Vector search requires embedding model. Showing recent memories instead.\n\n")
	}

	// Show recent memories
	entries, err := engine.Retrieve(ctx, "", recentLimit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to retrieve memories: %v\n", err)
		return 1
	}

	if len(entries) == 0 {
		fmt.Println("No memory entries yet")
		return 0
	}

	fmt.Printf("=== Last %d Memory Entries ===\n\n", len(entries))
	for i, entry := range entries {
		roleLabel := formatRole(entry.Role)
		timestamp := entry.CreatedAt.Format("2006-01-02 15:04:05")
		content := entry.Text
		if len(content) > 200 {
			content = content[:200] + "..."
		}
		fmt.Printf("[%d] %s | %s\n", i+1, timestamp, roleLabel)
		fmt.Printf("    %s\n", content)
		if entry.Summary != "" {
			summary := entry.Summary
			if len(summary) > 100 {
				summary = summary[:100] + "..."
			}
			fmt.Printf("    Summary: %s\n", summary)
		}
		fmt.Println()
	}

	return 0
}

func handleBasicMemory(cfg config.Config, limit int) int {
	store, err := memory.NewStore(cfg.Memory.Path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open memory store: %v\n", err)
		return 1
	}
	defer store.Close()

	if limit <= 0 {
		limit = cfg.Memory.TurnsToUse
		if limit <= 0 {
			limit = 10
		}
	}

	entries, err := store.Recent(limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read memory: %v\n", err)
		return 1
	}

	if len(entries) == 0 {
		fmt.Println("No memory entries yet")
		return 0
	}

	fmt.Printf("=== Last %d Conversation Turns ===\n\n", len(entries))
	for i := len(entries) - 1; i >= 0; i-- {
		entry := entries[i]
		roleLabel := formatRole(entry.Role)
		timestamp := entry.CreatedAt.Format("2006-01-02 15:04:05")
		fmt.Printf("[%s] %s: %s\n", timestamp, roleLabel, entry.Content)
	}
	return 0
}

func formatRole(role string) string {
	switch role {
	case "user":
		return "User"
	case "assistant":
		return "Assistant"
	case "system":
		return "System"
	default:
		return role
	}
}

func formatStatKey(key string) string {
	// Convert snake_case to Title Case
	replacer := map[string]string{
		"total_memories":       "Total Memories",
		"total_tokens":         "Total Tokens",
		"compressed_memories":  "Compressed Memories",
		"compressed_summaries": "Compressed Summaries",
	}
	if formatted, ok := replacer[key]; ok {
		return formatted
	}
	return key
}

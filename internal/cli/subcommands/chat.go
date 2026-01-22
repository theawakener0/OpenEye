package subcommands

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/pipeline"
	"OpenEye/internal/runtime"
)

// RunChat executes the chat subcommand.
func RunChat(ctx context.Context, cfg config.Config, registry runtime.Registry, message string, image []string, opts ChatOptions) int {
	pipe, err := pipeline.New(cfg, registry)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize pipeline: %v\n", err)
		return 1
	}
	defer func() {
		if closeErr := pipe.Close(); closeErr != nil {
			log.Printf("warning: failed to close pipeline: %v", closeErr)
		}
	}()

	start := time.Now()
	options := pipeline.Options{}
	
	var spinnerDone chan struct{}
	if opts.Stream {
		options.Stream = true
		options.StreamCallback = func(evt runtime.StreamEvent) error {
			if spinnerDone != nil {
				close(spinnerDone)
				spinnerDone = nil
				fmt.Print("\r\033[K") // Clear spinner line
			}
			if evt.Err != nil {
				return evt.Err
			}
			if evt.Final {
				fmt.Println()
				return nil
			}
			fmt.Print(evt.Token)
			os.Stdout.Sync() // Flush immediately
			return nil
		}
	} else {
		spinnerDone = make(chan struct{})
		go runCLISpinner(spinnerDone, "Thinking")
	}

	options.DisableRAG = opts.DisableRAG
	options.DisableSummary = opts.DisableSummary
	options.DisableVectorMemory = opts.DisableVectorMemory
	options.RAGLimit = opts.RAGLimit
	options.MemoryLimit = opts.MemoryLimit

	result, err := pipe.Respond(ctx, message, image, options)
	
	if spinnerDone != nil {
		close(spinnerDone)
		fmt.Print("\r\033[K") // Clear spinner line
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "runtime error: %v\n", err)
		return 1
	}

	if !opts.Stream {
		fmt.Println(result.Text)
	}

	duration := time.Since(start)
	
	// Show detailed statistics if requested
	if opts.ShowStats {
		fmt.Println("\n\033[1;90m--- Statistics ---\033[0m")
		if result.Stats.TokensEvaluated > 0 || result.Stats.TokensGenerated > 0 || result.Stats.TokensCached > 0 {
			fmt.Printf("\033[90mTokens:\033[0m eval=%d gen=%d cached=%d\n", result.Stats.TokensEvaluated, result.Stats.TokensGenerated, result.Stats.TokensCached)
		}
		if result.Summary != "" {
			fmt.Printf("\033[90mMemory summary:\033[0m %s\n", truncateString(result.Summary, 100))
		}
		if len(result.Retrieved) > 0 {
			fmt.Printf("\033[90mRetrieved %d RAG chunks:\033[0m\n", len(result.Retrieved))
			for i, doc := range result.Retrieved {
				fmt.Printf("  [%d] \033[36m%s\033[0m (score: %.2f)\n", i+1, doc.Source, doc.Score)
			}
		}
		fmt.Printf("\033[90mDuration:\033[0m %s\n", duration.Truncate(time.Millisecond))
	} else if result.Stats.TokensEvaluated > 0 || result.Stats.TokensGenerated > 0 || result.Stats.TokensCached > 0 {
		log.Printf("stats: eval=%d gen=%d cached=%d", result.Stats.TokensEvaluated, result.Stats.TokensGenerated, result.Stats.TokensCached)
	}
	
	log.Printf("completed in %s", duration.Truncate(10*time.Millisecond))
	return 0
}

// ChatOptions capture per-invocation controls beyond CLI flags.
type ChatOptions struct {
	Stream              bool
	DisableRAG          bool
	DisableSummary      bool
	DisableVectorMemory bool
	RAGLimit            int
	MemoryLimit         int
	ShowStats           bool
}

// truncateString truncates a string to maxLen characters.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

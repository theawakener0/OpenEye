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
func RunChat(ctx context.Context, cfg config.Config, registry runtime.Registry, message string, opts ChatOptions) int {
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
	if opts.Stream {
		options.Stream = true
		options.StreamCallback = func(evt runtime.StreamEvent) error {
			if evt.Err != nil {
				return evt.Err
			}
			if evt.Final {
				fmt.Println()
				return nil
			}
			fmt.Print(evt.Token)
			return nil
		}
	}
	options.DisableRAG = opts.DisableRAG
	options.DisableSummary = opts.DisableSummary
	options.RAGLimit = opts.RAGLimit

	result, err := pipe.Respond(ctx, message, options)
	if err != nil {
		fmt.Fprintf(os.Stderr, "runtime error: %v\n", err)
		return 1
	}

	if !opts.Stream {
		fmt.Println(result.Text)
	}

	duration := time.Since(start)
	if result.Stats.TokensEvaluated > 0 || result.Stats.TokensGenerated > 0 || result.Stats.TokensCached > 0 {
		log.Printf("stats: eval=%d gen=%d cached=%d", result.Stats.TokensEvaluated, result.Stats.TokensGenerated, result.Stats.TokensCached)
	}
	log.Printf("completed in %s", duration.Truncate(10*time.Millisecond))
	return 0
}

// ChatOptions capture per-invocation controls beyond CLI flags.
type ChatOptions struct {
	Stream         bool
	DisableRAG     bool
	DisableSummary bool
	RAGLimit       int
}

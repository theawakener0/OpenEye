package cli

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"OpenEye/internal/cli/subcommands"
	"OpenEye/internal/config"
	"OpenEye/internal/runtime"

	_ "OpenEye/internal/llmclient"
)

// Execute is the entry point for the OpenEye CLI.
func Execute() int {
	ctx := context.Background()
	args := os.Args[1:]

	cfg, err := config.Resolve()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		return 1
	}

	registry := runtime.DefaultRegistry

	if len(args) == 0 {
		return runChat(ctx, cfg, registry, []string{})
	}

	subcommand := args[0]
	switch subcommand {
	case "chat":
		return runChat(ctx, cfg, registry, args[1:])
	case "serve":
		return subcommands.RunServe(ctx, cfg)
	case "memory":
		return subcommands.RunMemory(cfg, args[1:])
	case "help", "-h", "--help":
		printHelp()
		return 0
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", subcommand)
		printHelp()
		return 1
	}
}

func runChat(ctx context.Context, cfg config.Config, registry runtime.Registry, args []string) int {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	fs.SetOutput(os.Stdout)
	message := fs.String("message", "", "Prompt to send to the configured runtime")
	stream := fs.Bool("stream", false, "Stream tokens instead of waiting for the full response")
	disableRAG := fs.Bool("no-rag", false, "Disable retrieval augmented generation for this request")
	disableSummary := fs.Bool("no-summary", false, "Disable helper summarizer for this request")
	ragLimit := fs.Int("rag-limit", 0, "Override number of retrieved chunks (0 uses config default)")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse flags: %v\n", err)
		return 1
	}

	remaining := fs.Args()
	if *message == "" && len(remaining) > 0 {
		*message = strings.Join(remaining, " ")
	}

	if strings.TrimSpace(*message) == "" {
		fmt.Fprintln(os.Stderr, "chat requires a message (--message) or positional argument")
		return 1
	}

	options := subcommands.ChatOptions{
		Stream:         *stream,
		DisableRAG:     *disableRAG,
		DisableSummary: *disableSummary,
		RAGLimit:       *ragLimit,
	}

	return subcommands.RunChat(ctx, cfg, registry, strings.TrimSpace(*message), options)
}

func printHelp() {
	fmt.Println(`OpenEye CLI

Usage:
  OpenEye [command] [flags]

Commands:
  chat    Run a single prompt against the configured runtime
  serve   Start the TCP server using runtime + memory pipeline
  memory  Inspect recent conversation turns

Use "OpenEye [command] --help" for more information about a command.`)
}

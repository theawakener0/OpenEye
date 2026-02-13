package subcommands

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/pipeline"
	"OpenEye/internal/runtime"
	"OpenEye/server"
)

const maxRetries = 3

func respondWithRetry(pipe *pipeline.Pipeline, content string, images []string, opts pipeline.Options) (pipeline.Result, error) {
	ctx := context.Background()
	result, err := pipe.Respond(ctx, content, images, opts)
	if err == nil {
		return result, nil
	}

	if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		return result, err
	}

	for attempt := 1; attempt < maxRetries; attempt++ {
		log.Printf("retrying after context canceled (attempt %d/%d)", attempt+1, maxRetries)
		ctx := context.Background()
		result, err = pipe.Respond(ctx, content, images, opts)
		if err == nil {
			return result, nil
		}
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			return result, err
		}
	}

	return result, err
}

// RunServe starts the server and processes inbound prompts through the runtime.
// Supports both HTTP and TCP server types based on configuration.
func RunServe(ctx context.Context, cfg config.Config, args []string) int {
	if !cfg.ServerEnabled() {
		fmt.Fprintln(os.Stderr, "server disabled by configuration")
		return 1
	}

	// Parse CLI flags
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	hostFlag := fs.String("host", "", "Server host address (overrides config)")
	portFlag := fs.Int("port", 0, "Server port (overrides config)")
	typeFlag := fs.String("type", "", "Server type: http or tcp (overrides config)")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintf(os.Stderr, "failed to parse flags: %v\n", err)
		return 1
	}

	// Apply CLI overrides to config
	host := cfg.Server.Host
	if *hostFlag != "" {
		host = *hostFlag
	}
	if host == "" {
		host = "127.0.0.1"
	}

	port := cfg.Server.Port
	if *portFlag > 0 {
		port = *portFlag
	}

	serverType := cfg.ServerType()
	if *typeFlag != "" {
		serverType = strings.ToLower(*typeFlag)
	}
	if serverType != "http" && serverType != "tcp" {
		fmt.Fprintf(os.Stderr, "invalid server type: %s (must be http or tcp)\n", serverType)
		return 1
	}

	pipe, err := pipeline.New(cfg, runtime.DefaultRegistry)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize pipeline: %v\n", err)
		return 1
	}
	defer func() {
		if closeErr := pipe.Close(); closeErr != nil {
			log.Printf("warning: failed to close pipeline: %v", closeErr)
		}
	}()

	sigCtx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Route to appropriate server type
	if serverType == "http" {
		return runHTTPServer(sigCtx, host, port, cfg.Runtime.Backend, pipe)
	}
	return runTCPServer(sigCtx, host, port, pipe)
}

// runHTTPServer starts the HTTP server and handles requests
func runHTTPServer(ctx context.Context, host string, port int, backend string, pipe *pipeline.Pipeline) int {
	httpServer := server.NewHTTPServer(host, strconv.Itoa(port))
	if err := httpServer.Start(backend); err != nil {
		fmt.Fprintf(os.Stderr, "failed to start HTTP server: %v\n", err)
		return 1
	}
	defer func() {
		if stopErr := httpServer.Stop(); stopErr != nil {
			log.Printf("warning: failed to stop HTTP server: %v", stopErr)
		}
	}()

	fmt.Printf("OpenEye HTTP server listening on http://%s:%d\n", host, port)
	fmt.Printf("  Health:   http://%s:%d/health\n", host, port)
	fmt.Printf("  Chat API: http://%s:%d/v1/chat\n", host, port)

	for {
		select {
		case <-ctx.Done():
			fmt.Println("HTTP server shutting down")
			return 0
		default:
		}

		msg, err := httpServer.Receive()
		if err != nil {
			log.Printf("receive error: %v", err)
			time.Sleep(200 * time.Millisecond)
			continue
		}

		go func(inbound server.HTTPMessage) {
			images := inbound.Images
			if len(images) > 0 {
				log.Printf("Processing request with %d image(s)", len(images))
			}

			// Clear context to ensure clean state between HTTP requests
			// This prevents prompt caching issues that accumulate across requests
			if err := pipe.ClearContext(); err != nil {
				log.Printf("warning: failed to clear context: %v", err)
			}

			opts := pipeline.Options{
				Stream: inbound.Stream,
				StreamCallback: func(evt runtime.StreamEvent) error {
					if evt.Err != nil {
						return evt.Err
					}
					if !evt.Final {
						return inbound.StreamToken(evt.Token)
					}
					return nil
				},
			}

			// Apply custom options if provided (non-zero values override config defaults)
			if temp, ok := inbound.Options["temperature"].(float64); ok && temp != 0 {
				opts.GenerationHints.Temperature = temp
			}
			if maxTokens, ok := inbound.Options["max_tokens"].(float64); ok && maxTokens != 0 {
				opts.GenerationHints.MaxTokens = int(maxTokens)
			}
			if topK, ok := inbound.Options["top_k"].(float64); ok && topK != 0 {
				opts.GenerationHints.TopK = int(topK)
			}
			if topP, ok := inbound.Options["top_p"].(float64); ok && topP != 0 {
				opts.GenerationHints.TopP = topP
			}
			if minP, ok := inbound.Options["min_p"].(float64); ok && minP != 0 {
				opts.GenerationHints.MinP = minP
			}
			if repeatPenalty, ok := inbound.Options["repeat_penalty"].(float64); ok && repeatPenalty != 0 {
				opts.GenerationHints.RepeatPenalty = repeatPenalty
			}
			if repeatLastN, ok := inbound.Options["repeat_last_n"].(float64); ok && repeatLastN != 0 {
				opts.GenerationHints.RepeatLastN = int(repeatLastN)
			}

			result, runErr := respondWithRetry(pipe, inbound.Content, images, opts)
			if runErr != nil {
				log.Printf("runtime error: %v", runErr)
				if respErr := inbound.RespondError(runErr); respErr != nil {
					log.Printf("response error: %v", respErr)
				}
				return
			}
			if respErr := inbound.Respond(result.Text); respErr != nil {
				log.Printf("response error: %v", respErr)
			}
			log.Printf("response: %s", truncateResponse(result.Text, 100))
		}(msg)
	}
}

// runTCPServer starts the TCP server and handles requests
func runTCPServer(ctx context.Context, host string, port int, pipe *pipeline.Pipeline) int {
	tcpServer := server.NewTCPServer(host, strconv.Itoa(port))
	if err := tcpServer.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to start TCP server: %v\n", err)
		return 1
	}
	defer func() {
		if stopErr := tcpServer.Stop(); stopErr != nil {
			log.Printf("warning: failed to stop TCP server: %v", stopErr)
		}
	}()

	fmt.Printf("OpenEye TCP server listening on %s:%d\n", host, port)

	for {
		select {
		case <-ctx.Done():
			fmt.Println("TCP server shutting down")
			return 0
		default:
		}

		msg, err := tcpServer.Receive()
		if err != nil {
			log.Printf("receive error: %v", err)
			time.Sleep(200 * time.Millisecond)
			continue
		}

		go func(inbound server.Message) {
			images := inbound.Images
			if len(images) > 0 {
				log.Printf("Processing request with %d image(s)", len(images))
			}

			opts := pipeline.Options{
				Stream: true,
				StreamCallback: func(evt runtime.StreamEvent) error {
					if evt.Err != nil {
						return evt.Err
					}
					if !evt.Final {
						return inbound.StreamToken(evt.Token)
					}
					return nil
				},
			}

			result, runErr := respondWithRetry(pipe, inbound.Content, images, opts)
			if runErr != nil {
				log.Printf("runtime error: %v", runErr)
				if respErr := inbound.RespondError(runErr); respErr != nil {
					log.Printf("response error: %v", respErr)
				}
				return
			}
			if respErr := inbound.Respond(result.Text); respErr != nil {
				log.Printf("response error: %v", respErr)
			}
			log.Printf("response: %s", truncateResponse(result.Text, 100))
		}(msg)
	}
}

// truncateResponse shortens a response for logging.
func truncateResponse(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

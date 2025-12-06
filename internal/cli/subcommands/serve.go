package subcommands

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/pipeline"
	"OpenEye/internal/runtime"
	"OpenEye/server"
)

// RunServe starts the TCP server and processes inbound prompts through the runtime.
func RunServe(ctx context.Context, cfg config.Config) int {
	if !cfg.ServerEnabled() {
		fmt.Fprintln(os.Stderr, "server disabled by configuration")
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

	host := cfg.Server.Host
	if host == "" {
		host = "127.0.0.1"
	}

	tcpServer := server.NewTCPServer(host, strconv.Itoa(cfg.Server.Port))
	if err := tcpServer.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to start TCP server: %v\n", err)
		return 1
	}
	defer func() {
		if stopErr := tcpServer.Stop(); stopErr != nil {
			log.Printf("warning: failed to stop server: %v", stopErr)
		}
	}()

	fmt.Printf("OpenEye server listening on %s:%d\n", host, cfg.Server.Port)

	sigCtx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	for {
		select {
		case <-sigCtx.Done():
			fmt.Println("server shutting down")
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
			// Use images from the message if provided
			var images []string
			if len(inbound.Images) > 0 {
				images = inbound.Images
				log.Printf("Processing request with %d image(s)", len(images))
			}
			
			result, runErr := pipe.Respond(context.Background(), inbound.Content, images, pipeline.Options{})
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

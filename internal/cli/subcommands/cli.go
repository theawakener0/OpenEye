package subcommands

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/pipeline"
	"OpenEye/internal/runtime"
)

// RunCli executes the interactive CLI mode.
func RunCli(ctx context.Context, cfg config.Config, registry runtime.Registry, opts CliOptions) int {
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

	fmt.Println("OpenEye Interactive Mode")
	fmt.Println("Type 'exit' or 'quit' to end the session")
	fmt.Println("Type '/help' for available commands")
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)
	var attachedImages []string // Track attached images for the next message
	imageMode := cfg.Image.Enabled // Image support based on config

	for {
		fmt.Print("You: ")
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Fprintf(os.Stderr, "input error: %v\n", err)
			return 1
		}

		message := strings.TrimSpace(input)
		if message == "" {
			continue
		}

		// Handle special commands
		if strings.HasPrefix(message, "/") {
			handled, newImages, newImageMode := handleCommand(ctx, pipe, message, opts, attachedImages, imageMode, cfg.Image.Enabled)
			if handled {
				attachedImages = newImages
				imageMode = newImageMode
				continue
			}
		}

		// Check for exit commands
		lowerMsg := strings.ToLower(message)
		if lowerMsg == "exit" || lowerMsg == "quit" || lowerMsg == "/exit" || lowerMsg == "/quit" || lowerMsg == "/bye" {
			fmt.Println("Goodbye!")
			return 0
		}		

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
		options.DisableVectorMemory = opts.DisableVectorMemory
		options.RAGLimit = opts.RAGLimit
		options.MemoryLimit = opts.MemoryLimit

		// Only include images if image mode is enabled
		var imagesToSend []string
		if imageMode && len(attachedImages) > 0 {
			imagesToSend = attachedImages
			fmt.Printf("[Sending with %d image(s)]\n", len(attachedImages))
		}

		fmt.Print("OpenEye: ")
		result, err := pipe.Respond(ctx, message, imagesToSend, options)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nruntime error: %v\n", err)
			continue
		}

		// Clear images after sending (one-time use)
		attachedImages = nil

		if !opts.Stream {
			fmt.Println(result.Text)
		}

		duration := time.Since(start)
		
		if opts.ShowStats {
			printResponseStats(result, duration)
		} else if result.Stats.TokensEvaluated > 0 || result.Stats.TokensGenerated > 0 || result.Stats.TokensCached > 0 {
			log.Printf("stats: eval=%d gen=%d cached=%d time=%s", 
				result.Stats.TokensEvaluated, result.Stats.TokensGenerated, result.Stats.TokensCached,
				duration.Truncate(10*time.Millisecond))
		}
		fmt.Println()
	}
}

// handleCommand processes special CLI commands.
// Returns (handled bool, updatedImages []string, imageMode bool).
func handleCommand(ctx context.Context, pipe *pipeline.Pipeline, cmd string, opts CliOptions, images []string, imageMode bool, imageConfigEnabled bool) (bool, []string, bool) {
	lowerCmd := strings.ToLower(cmd)
	
	// Handle commands with arguments
	if strings.HasPrefix(lowerCmd, "/image ") {
		if !imageConfigEnabled {
			fmt.Println("Image processing is disabled in configuration.")
			fmt.Println("Enable it in openeye.yaml: image.enabled: true")
			return true, images, imageMode
		}
		if !imageMode {
			fmt.Println("Image mode is off. Use /image-on to enable it first.")
			return true, images, imageMode
		}
		imagePath := strings.TrimSpace(cmd[7:]) // Extract path after "/image "
		if imagePath == "" {
			fmt.Println("Usage: /image <path|base64>")
			return true, images, imageMode
		}
		// Check if it's a file path or base64 data
		if !strings.HasPrefix(imagePath, "data:") && !isLikelyBase64(imagePath) {
			// Validate file exists for file paths
			if _, err := os.Stat(imagePath); os.IsNotExist(err) {
				fmt.Printf("Image file not found: %s\n", imagePath)
				return true, images, imageMode
			}
		}
		images = append(images, imagePath)
		fmt.Printf("Image attached: %s (total: %d)\n", truncatePath(imagePath, 50), len(images))
		return true, images, imageMode
	}

	switch lowerCmd {
	case "/help":
		printCliHelp(imageConfigEnabled)
		return true, images, imageMode
	case "/stats":
		showMemoryStats(ctx, pipe)
		return true, images, imageMode
	case "/compress":
		triggerCompression(ctx, pipe)
		return true, images, imageMode
	case "/clear":
		fmt.Print("\033[H\033[2J") // Clear terminal
		return true, images, imageMode
	case "/toggle-stats":
		opts.ShowStats = !opts.ShowStats
		fmt.Printf("Stats display: %v\n", opts.ShowStats)
		return true, images, imageMode
	case "/images":
		if !imageConfigEnabled {
			fmt.Println("Image processing is disabled in configuration.")
			return true, images, imageMode
		}
		if len(images) == 0 {
			fmt.Println("No images attached.")
		} else {
			fmt.Printf("Attached images (%d):\n", len(images))
			for i, img := range images {
				fmt.Printf("  %d. %s\n", i+1, truncatePath(img, 60))
			}
		}
		fmt.Printf("Image mode: %v\n", imageMode)
		return true, images, imageMode
	case "/clear-images":
		count := len(images)
		images = nil
		fmt.Printf("Cleared %d image(s).\n", count)
		return true, images, imageMode
	case "/image":
		fmt.Println("Usage: /image <path|base64>")
		fmt.Println("Examples:")
		fmt.Println("  /image /path/to/photo.jpg")
		fmt.Println("  /image data:image/jpeg;base64,/9j/4AAQ...")
		return true, images, imageMode
	case "/image-on":
		if !imageConfigEnabled {
			fmt.Println("Image processing is disabled in configuration.")
			fmt.Println("Enable it in openeye.yaml: image.enabled: true")
			return true, images, imageMode
		}
		imageMode = true
		fmt.Println("Image mode enabled. Use /image <path> to attach images.")
		return true, images, imageMode
	case "/image-off":
		imageMode = false
		images = nil // Clear any attached images
		fmt.Println("Image mode disabled. Attached images cleared.")
		return true, images, imageMode
	case "/image-status":
		fmt.Printf("Image processing config: %v\n", imageConfigEnabled)
		fmt.Printf("Image mode: %v\n", imageMode)
		fmt.Printf("Attached images: %d\n", len(images))
		return true, images, imageMode
	default:
		if strings.HasPrefix(cmd, "/") {
			fmt.Printf("Unknown command: %s (type /help for available commands)\n", cmd)
			return true, images, imageMode
		}
		return false, images, imageMode
	}
}

func printCliHelp(imageEnabled bool) {
	fmt.Println(`
Available Commands:
  /help          Show this help message
  /stats         Show memory statistics
  /compress      Trigger memory compression
  /clear         Clear the terminal screen
  /toggle-stats  Toggle per-response statistics display
  /exit, /quit   Exit the CLI
  exit, quit     Exit the CLI`)

	if imageEnabled {
		fmt.Println(`
Image Commands:
  /image <path>    Attach an image file or base64 data
  /images          List all attached images
  /clear-images    Clear all attached images
  /image-on        Enable image mode
  /image-off       Disable image mode and clear images
  /image-status    Show image processing status

Image Usage:
  1. Enable mode:    /image-on (if not already on)
  2. Attach images:  /image /path/to/image1.jpg
                     /image /path/to/image2.png
                     /image data:image/jpeg;base64,/9j/4AAQ...
  3. Send message:   Describe what's in these images
  4. Images are automatically cleared after each message

Supported formats: JPEG, PNG, BMP
Images are automatically resized and optimized for the model.`)
	} else {
		fmt.Println(`
Image Commands: (disabled in config)
  Enable in openeye.yaml: image.enabled: true`)
	}

	fmt.Println(`
Flags (set at startup):
  --stream       Stream tokens as they're generated
  --no-rag       Disable RAG retrieval
  --no-summary   Disable summarization
  --no-vector    Disable vector memory retrieval
  --stats        Show detailed statistics after each response
  --rag-limit N  Override RAG chunk limit
  --memory-limit N  Override memory retrieval limit
`)
}

// isLikelyBase64 checks if a string appears to be base64 encoded image data.
func isLikelyBase64(s string) bool {
	// Check for data URI prefix
	if strings.HasPrefix(s, "data:image/") {
		return true
	}
	// Check if it's long enough and contains only base64 chars
	if len(s) < 100 {
		return false
	}
	for _, c := range s[:100] {
		if !((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
			(c >= '0' && c <= '9') || c == '+' || c == '/' || c == '=') {
			return false
		}
	}
	return true
}

// truncatePath shortens a path or base64 string for display.
func truncatePath(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if strings.HasPrefix(s, "data:image/") {
		// For data URIs, show format and truncate
		if idx := strings.Index(s, ","); idx != -1 && idx < 30 {
			return s[:idx+1] + "..." + fmt.Sprintf(" (%d bytes)", len(s))
		}
	}
	return s[:maxLen-3] + "..."
}

func showMemoryStats(ctx context.Context, pipe *pipeline.Pipeline) {
	stats, err := pipe.GetMemoryStats(ctx)
	if err != nil {
		fmt.Printf("Failed to get memory stats: %v\n", err)
		return
	}

	fmt.Println("\n--- Memory Statistics ---")
	for key, value := range stats {
		fmt.Printf("  %s: %v\n", key, value)
	}
	fmt.Println()
}

func triggerCompression(ctx context.Context, pipe *pipeline.Pipeline) {
	fmt.Println("Triggering memory compression...")
	if err := pipe.CompressMemory(ctx); err != nil {
		fmt.Printf("Compression failed: %v\n", err)
		return
	}
	fmt.Println("Memory compression completed.")
}

func printResponseStats(result pipeline.Result, duration time.Duration) {
	fmt.Println("\n  --- Response Stats ---")
	if result.Stats.TokensEvaluated > 0 || result.Stats.TokensGenerated > 0 {
		fmt.Printf("  Tokens: eval=%d gen=%d cached=%d\n", 
			result.Stats.TokensEvaluated, result.Stats.TokensGenerated, result.Stats.TokensCached)
	}
	if result.Summary != "" {
		summary := result.Summary
		if len(summary) > 80 {
			summary = summary[:80] + "..."
		}
		fmt.Printf("  Summary: %s\n", summary)
	}
	if len(result.Retrieved) > 0 {
		fmt.Printf("  Retrieved: %d chunks\n", len(result.Retrieved))
	}
	fmt.Printf("  Duration: %s\n", duration.Truncate(time.Millisecond))
}

// CliOptions capture per-invocation controls beyond CLI flags.
type CliOptions struct {
	Stream              bool
	DisableRAG          bool
	DisableSummary      bool
	DisableVectorMemory bool
	RAGLimit            int
	MemoryLimit         int
	ShowStats           bool
}

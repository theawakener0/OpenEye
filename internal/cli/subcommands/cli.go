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

const (
	colorReset  = "\033[0m"
	colorBold   = "\033[1m"
	colorBlue   = "\033[34m"
	colorGreen  = "\033[32m"
	colorCyan   = "\033[36m"
	colorGray   = "\033[90m"
	colorYellow = "\033[33m"
	colorRed    = "\033[31m"
)

const logo = `
 ▄██████▄     ▄███████▄    ▄████████ ███▄▄▄▄      ▄████████ ▄██   ▄      ▄████████ 
███    ███   ███    ███   ███    ███ ███▀▀▀██▄   ███    ███ ███   ██▄   ███    ███ 
███    ███   ███    ███   ███    █▀  ███   ███   ███    █▀  ███▄▄▄███   ███    █▀  
███    ███   ███    ███  ▄███▄▄▄     ███   ███  ▄███▄▄▄     ▀▀▀▀▀▀███  ▄███▄▄▄     
███    ███ ▀█████████▀  ▀▀███▀▀▀     ███   ███ ▀▀███▀▀▀     ▄██   ███ ▀▀███▀▀▀     
███    ███   ███          ███    █▄  ███   ███   ███    █▄  ███   ███   ███    █▄  
███    ███   ███          ███    ███ ███   ███   ███    ███ ███   ███   ███    ███ 
 ▀██████▀   ▄████▀        ██████████  ▀█   █▀    ██████████  ▀█████▀    ██████████ 
                                                                                   
`

// RunCli executes the interactive CLI mode (traditional ANSI interface).
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

	fmt.Print(colorCyan + logo + colorReset)
	fmt.Printf("%sThe New Revolution of SLMs%s\n\n", colorCyan, colorReset)
	fmt.Printf("%sOpenEye Interactive Mode%s\n", colorBold, colorReset)
	fmt.Printf("%sType 'exit' to quit | '/help' for commands%s\n", colorGray, colorReset)
	fmt.Printf("%sRuntime: %s%s\n", colorGray, cfg.Runtime.Backend, colorReset)
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)
	var attachedImages []string
	imageMode := cfg.Image.Enabled

	for {
		fmt.Printf("%sYou: %s", colorBlue+colorBold, colorReset)
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Fprintf(os.Stderr, "input error: %v\n", err)
			return 1
		}

		message := strings.TrimSpace(input)
		// Support multi-line input if ending with backslash
		for strings.HasSuffix(message, "\\") {
			message = strings.TrimSuffix(message, "\\")
			fmt.Printf("%s...  %s", colorGray, colorReset)
			nextPart, _ := reader.ReadString('\n')
			message += strings.TrimSpace(nextPart)
		}

		if message == "" {
			continue
		}

		// Handle special commands
		if strings.HasPrefix(message, "/") {
			handled, newImages, newImageMode := handleCommand(ctx, pipe, message, &opts, attachedImages, imageMode, cfg.Image.Enabled)
			if handled {
				attachedImages = newImages
				imageMode = newImageMode
				continue
			}
		}

		// Check for exit commands
		lowerMsg := strings.ToLower(message)
		if lowerMsg == "exit" || lowerMsg == "quit" || lowerMsg == "/exit" || lowerMsg == "/quit" || lowerMsg == "/bye" {
			fmt.Printf("\n%sGoodbye!%s\n", colorCyan, colorReset)
			return 0
		}

		// Run pipeline
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
					fmt.Printf("%sOpenEye: %s", colorGreen+colorBold, colorReset)
				}

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
			fmt.Printf("%s[Processing %d image(s)]%s\n", colorCyan, len(attachedImages), colorReset)
		}

		if opts.Stream {
			spinnerDone = make(chan struct{})
			go runCLISpinner(spinnerDone, "Thinking")
		} else {
			fmt.Printf("%sOpenEye: %s", colorGreen+colorBold, colorReset)
			spinnerDone = make(chan struct{})
			go runCLISpinner(spinnerDone, "Thinking")
		}

		result, err := pipe.Respond(ctx, message, imagesToSend, options)
		
		if spinnerDone != nil {
			close(spinnerDone)
			fmt.Print("\r\033[K") // Clear spinner line
		}

		if err != nil {
			fmt.Fprintf(os.Stderr, "\r%sruntime error: %v%s\n", colorRed, err, colorReset)
			continue
		}

		// Clear images after sending (one-time use)
		attachedImages = nil

		if !opts.Stream {
			fmt.Printf("%sOpenEye: %s", colorGreen+colorBold, colorReset)
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
func handleCommand(ctx context.Context, pipe *pipeline.Pipeline, cmd string, opts *CliOptions, images []string, imageMode bool, imageConfigEnabled bool) (bool, []string, bool) {
	lowerCmd := strings.ToLower(cmd)
	
	// Handle commands with arguments
	if strings.HasPrefix(lowerCmd, "/image ") {
		if !imageConfigEnabled {
			fmt.Println(colorYellow + "Image processing is disabled in configuration." + colorReset)
			fmt.Println("Enable it in openeye.yaml: image.enabled: true")
			return true, images, imageMode
		}
		if !imageMode {
			fmt.Println(colorYellow + "Image mode is off. Use /image-on to enable it first." + colorReset)
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
				fmt.Printf(colorRed+"Image file not found: %s\n"+colorReset, imagePath)
				return true, images, imageMode
			}
		}
		images = append(images, imagePath)
		fmt.Printf(colorCyan+"Image attached: %s (total: %d)"+colorReset+"\n", truncatePath(imagePath, 50), len(images))
		return true, images, imageMode
	}

	if strings.HasPrefix(lowerCmd, "/set ") {
		parts := strings.SplitN(cmd[5:], " ", 2)
		if len(parts) < 2 {
			fmt.Println("Usage: /set <param> <value>")
			return true, images, imageMode
		}
		param := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		handleSetParam(opts, param, value)
		return true, images, imageMode
	}

	switch lowerCmd {
	case "/help":
		printCliHelp(imageConfigEnabled)
		return true, images, imageMode
	case "/stats":
		showMemoryStats(ctx, pipe)
		return true, images, imageMode
	case "/config":
		fmt.Printf("\n%s--- Current Session Options ---%s\n", colorBold, colorReset)
		fmt.Printf("  %sStream:%s         %v\n", colorCyan, colorReset, opts.Stream)
		fmt.Printf("  %sShow Stats:%s     %v\n", colorCyan, colorReset, opts.ShowStats)
		fmt.Printf("  %sNo RAG:%s         %v\n", colorCyan, colorReset, opts.DisableRAG)
		fmt.Printf("  %sNo Summary:%s     %v\n", colorCyan, colorReset, opts.DisableSummary)
		fmt.Printf("  %sNo Vector:%s      %v\n", colorCyan, colorReset, opts.DisableVectorMemory)
		fmt.Printf("  %sRAG Limit:%s      %d\n", colorCyan, colorReset, opts.RAGLimit)
		fmt.Printf("  %sMemory Limit:%s   %d\n", colorCyan, colorReset, opts.MemoryLimit)
		return true, images, imageMode
	case "/compress":
		triggerCompression(ctx, pipe)
		return true, images, imageMode
	case "/clear":
		fmt.Print("\033[H\033[2J") // Clear terminal
		return true, images, imageMode
	case "/exit", "/quit":
		fmt.Printf("\n%sGoodbye!%s\n", colorCyan, colorReset)
		os.Exit(0)
		return true, images, imageMode
	case "/toggle-stats":
		opts.ShowStats = !opts.ShowStats
		fmt.Printf("Stats display: %s%v%s\n", colorCyan, opts.ShowStats, colorReset)
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
		fmt.Println(colorCyan + "Image mode enabled. Use /image <path> to attach images." + colorReset)
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
			fmt.Printf(colorYellow+"Unknown command: %s (type /help for available commands)"+colorReset+"\n", cmd)
			return true, images, imageMode
		}
		return false, images, imageMode
	}
}

func printCliHelp(imageEnabled bool) {
	fmt.Printf("\n%sAvailable Commands:%s\n", colorBold, colorReset)
	fmt.Printf("  %s/help%s          Show this help message\n", colorCyan, colorReset)
	fmt.Printf("  %s/config%s        Show session configuration\n", colorCyan, colorReset)
	fmt.Printf("  %s/set <p> <v>%s   Set session parameter (e.g. /set stream off)\n", colorCyan, colorReset)
	fmt.Printf("  %s/stats%s         Show memory statistics\n", colorCyan, colorReset)
	fmt.Printf("  %s/compress%s      Trigger memory compression\n", colorCyan, colorReset)
	fmt.Printf("  %s/clear%s         Clear the terminal screen\n", colorCyan, colorReset)
	fmt.Printf("  %s/toggle-stats%s  Toggle per-response statistics display\n", colorCyan, colorReset)
	fmt.Printf("  %s/exit%s, %s/quit%s   Exit the CLI\n", colorCyan, colorReset, colorCyan, colorReset)
	fmt.Printf("  %sexit%s, %squit%s     Exit the CLI\n", colorCyan, colorReset, colorCyan, colorReset)

	if imageEnabled {
		fmt.Printf("\n%sImage Commands:%s\n", colorBold, colorReset)
		fmt.Printf("  %s/image <path>%s    Attach an image file or base64 data\n", colorCyan, colorReset)
		fmt.Printf("  %s/images%s          List all attached images\n", colorCyan, colorReset)
		fmt.Printf("  %s/clear-images%s    Clear all attached images\n", colorCyan, colorReset)
		fmt.Printf("  %s/image-on%s        Enable image mode\n", colorCyan, colorReset)
		fmt.Printf("  %s/image-off%s       Disable image mode and clear images\n", colorCyan, colorReset)
		fmt.Printf("  %s/image-status%s    Show image processing status\n", colorCyan, colorReset)

		fmt.Printf("\n%sImage Usage:%s\n", colorBold, colorReset)
		fmt.Println("  1. Enable mode:    /image-on (if not already on)")
		fmt.Println("  2. Attach images:  /image /path/to/image1.jpg")
		fmt.Println("  3. Send message:   Describe what's in these images")
		fmt.Println("  4. Images are automatically cleared after each message")
	}

	fmt.Printf("\n%sFlags (set at startup):%s\n", colorBold, colorReset)
	fmt.Println("  --stream       Stream tokens as they're generated")
	fmt.Println("  --stats        Show detailed statistics after each response")
}

func showMemoryStats(ctx context.Context, pipe *pipeline.Pipeline) {
	stats, err := pipe.GetMemoryStats(ctx)
	if err != nil {
		fmt.Printf(colorRed+"Failed to get memory stats: %v"+colorReset+"\n", err)
		return
	}

	fmt.Printf("\n%s--- Memory Statistics ---%s\n", colorBold, colorReset)
	for key, value := range stats {
		fmt.Printf("  %s%s%s: %v\n", colorCyan, key, colorReset, value)
	}
	fmt.Println()
}

func triggerCompression(ctx context.Context, pipe *pipeline.Pipeline) {
	fmt.Printf("%sTriggering memory compression...%s\n", colorYellow, colorReset)
	if err := pipe.CompressMemory(ctx); err != nil {
		fmt.Printf(colorRed+"Compression failed: %v"+colorReset+"\n", err)
		return
	}
	fmt.Printf("%sMemory compression completed.%s\n", colorGreen, colorReset)
}

func printResponseStats(result pipeline.Result, duration time.Duration) {
	fmt.Printf("\n  %s--- Response Stats ---%s\n", colorGray+colorBold, colorReset)
	if result.Stats.TokensEvaluated > 0 || result.Stats.TokensGenerated > 0 {
		fmt.Printf("  %sTokens:%s eval=%d gen=%d cached=%d\n", 
			colorGray, colorReset, result.Stats.TokensEvaluated, result.Stats.TokensGenerated, result.Stats.TokensCached)
	}
	if result.Summary != "" {
		summary := result.Summary
		if len(summary) > 80 {
			summary = summary[:80] + "..."
		}
		fmt.Printf("  %sSummary:%s %s\n", colorGray, colorReset, summary)
	}
	if len(result.Retrieved) > 0 {
		fmt.Printf("  %sRetrieved:%s %d chunks\n", colorGray, colorReset, len(result.Retrieved))
	}
	fmt.Printf("  %sDuration:%s %s\n", colorGray, colorReset, duration.Truncate(time.Millisecond))
}

func handleSetParam(opts *CliOptions, param, value string) {
	lowerVal := strings.ToLower(value)
	isTrue := lowerVal == "true" || lowerVal == "on" || lowerVal == "1" || lowerVal == "yes"
	isFalse := lowerVal == "false" || lowerVal == "off" || lowerVal == "0" || lowerVal == "no"

	switch strings.ToLower(param) {
	case "stream":
		if isTrue {
			opts.Stream = true
		} else if isFalse {
			opts.Stream = false
		}
		fmt.Printf("Param %sStream%s set to %v\n", colorCyan, colorReset, opts.Stream)
	case "stats":
		if isTrue {
			opts.ShowStats = true
		} else if isFalse {
			opts.ShowStats = false
		}
		fmt.Printf("Param %sShowStats%s set to %v\n", colorCyan, colorReset, opts.ShowStats)
	case "rag":
		if isTrue {
			opts.DisableRAG = false
		} else if isFalse {
			opts.DisableRAG = true
		}
		fmt.Printf("Param %sRAG%s enabled: %v\n", colorCyan, colorReset, !opts.DisableRAG)
	case "summary":
		if isTrue {
			opts.DisableSummary = false
		} else if isFalse {
			opts.DisableSummary = true
		}
		fmt.Printf("Param %sSummary%s enabled: %v\n", colorCyan, colorReset, !opts.DisableSummary)
	case "vector":
		if isTrue {
			opts.DisableVectorMemory = false
		} else if isFalse {
			opts.DisableVectorMemory = true
		}
		fmt.Printf("Param %sVectorMemory%s enabled: %v\n", colorCyan, colorReset, !opts.DisableVectorMemory)
	case "rag-limit":
		var val int
		if _, err := fmt.Sscanf(value, "%d", &val); err == nil {
			opts.RAGLimit = val
			fmt.Printf("Param %sRAGLimit%s set to %d\n", colorCyan, colorReset, opts.RAGLimit)
		}
	case "memory-limit":
		var val int
		if _, err := fmt.Sscanf(value, "%d", &val); err == nil {
			opts.MemoryLimit = val
			fmt.Printf("Param %sMemoryLimit%s set to %d\n", colorCyan, colorReset, opts.MemoryLimit)
		}
	default:
		fmt.Printf(colorYellow+"Unknown parameter: %s"+colorReset+"\n", param)
	}
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

func isLikelyBase64(s string) bool {
	if len(s) < 64 {
		return false
	}
	// Check if it's strictly alphanumeric + plus/slash/equals
for _, r := range s {
if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '+' || r == '/' || r == '=') {
return false
}
}
return true
}

func truncatePath(path string, maxLen int) string {
	if len(path) <= maxLen {
		return path
	}
	return path[:maxLen-3] + "..."
}

func runCLISpinner(done chan struct{}, message string) {
	spinnerChars := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	i := 0
	for {
		select {
		case <-done:
			return
		default:
			fmt.Printf("\r%s%s %s...%s", colorCyan, spinnerChars[i], message, colorReset)
			i = (i + 1) % len(spinnerChars)
			time.Sleep(100 * time.Millisecond)
		}
	}
}

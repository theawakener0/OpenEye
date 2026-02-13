package pipeline

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"OpenEye/internal/config"
	conversation "OpenEye/internal/context"
	"OpenEye/internal/context/memory"
	"OpenEye/internal/context/memory/omem"
	"OpenEye/internal/embedding"
	"OpenEye/internal/image"
	"OpenEye/internal/rag"
	"OpenEye/internal/runtime"
)

// Options controls how the pipeline produces a response.
type Options struct {
	Stream              bool
	StreamCallback      runtime.StreamCallback
	GenerationHints     runtime.GenerationOptions
	DisableSummary      bool
	DisableRAG          bool
	DisableVectorMemory bool
	RAGLimit            int
	MemoryLimit         int
}

// Result captures the textual output and optional statistics.
type Result struct {
	Text      string
	Stats     runtime.Stats
	Summary   string
	Retrieved []rag.Document
}

// Pipeline orchestrates context assembly, runtime invocation, and memory storage.
type Pipeline struct {
	cfg            config.Config
	manager        *runtime.Manager
	store          *memory.Store
	vectorEngine   memory.MemoryEngine
	embedder       embedding.Provider
	retriever      rag.Retriever
	responseCache  sync.Map // Simple in-memory cache
	summarizer     summarizer
	imageProcessor image.Processor
	imageCache     sync.Map

	// Omem long-term memory integration
	omemAdapter *omem.Adapter
	omemHook    *omem.PipelineHook
}

// New constructs a Pipeline using the provided configuration and runtime registry.
func New(cfg config.Config, registry runtime.Registry) (*Pipeline, error) {
	mgr, err := runtime.NewManager(cfg.Runtime, registry)
	if err != nil {
		return nil, fmt.Errorf("pipeline: failed to initialise runtime: %w", err)
	}

	store, err := memory.NewStore(cfg.Memory.Path)
	if err != nil {
		mgr.Close()
		return nil, fmt.Errorf("pipeline: failed to open memory store: %w", err)
	}

	embedder, err := embedding.New(cfg.Embedding)
	if err != nil {
		store.Close()
		mgr.Close()
		return nil, fmt.Errorf("pipeline: failed to initialise embedding provider: %w", err)
	}

	retriever, err := initialiseRetriever(cfg, embedder)
	if err != nil {
		store.Close()
		mgr.Close()
		if embedder != nil {
			embedder.Close()
		}
		return nil, err
	}

	summarizer := newSummarizer(mgr, cfg.Assistants.Summarizer, embedder)

	// Initialize vector memory engine if enabled
	var vectorEngine memory.MemoryEngine
	if cfg.Memory.VectorEnabled && embedder != nil {
		vectorEngine, err = initializeVectorEngine(cfg, embedder, mgr)
		if err != nil {
			log.Printf("warning: failed to initialize vector memory engine: %v", err)
		}
	}

	// Initialize image processor if enabled
	var imageProcessor image.Processor
	if cfg.Image.Enabled {
		imageProcessor = initializeImageProcessor(cfg.Image)
	}

	// Initialize Omem long-term memory if enabled
	var omemAdapter *omem.Adapter
	var omemHook *omem.PipelineHook
	if cfg.Memory.Omem.Enabled != nil && *cfg.Memory.Omem.Enabled && embedder != nil {
		var err error
		omemAdapter, err = omem.NewAdapterFromConfig(cfg.Memory.Omem, mgr, embedder)
		if err != nil {
			log.Printf("warning: failed to initialize omem: %v", err)
			// Non-fatal, continue without omem
		} else {
			omemHook = omem.NewPipelineHook(omemAdapter)
			log.Println("omem long-term memory initialized")
		}
	}

	return &Pipeline{
		cfg:            cfg,
		manager:        mgr,
		store:          store,
		vectorEngine:   vectorEngine,
		embedder:       embedder,
		retriever:      retriever,
		summarizer:     summarizer,
		imageProcessor: imageProcessor,
		omemAdapter:    omemAdapter,
		omemHook:       omemHook,
	}, nil
}

// initializeVectorEngine creates the DuckDB-backed vector memory engine.
func initializeVectorEngine(cfg config.Config, embedder embedding.Provider, mgr *runtime.Manager) (memory.MemoryEngine, error) {
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

	// Create embedding wrapper
	embeddingWrapper := &embeddingProviderWrapper{provider: embedder}

	// Create summarization wrapper using the manager
	var summarizerWrapper *summarizerProviderWrapper
	if mgr != nil {
		summarizerWrapper = &summarizerProviderWrapper{manager: mgr}
	}

	return memory.NewEngine(engineCfg, embeddingWrapper, summarizerWrapper)
}

// initializeImageProcessor creates an image processor from config.
func initializeImageProcessor(cfg config.ImageConfig) image.Processor {
	format := image.FormatJPEG
	switch strings.ToLower(cfg.OutputFormat) {
	case "png":
		format = image.FormatPNG
	case "bmp":
		format = image.FormatBMP
	case "jpeg", "jpg":
		format = image.FormatJPEG
	}

	processorCfg := image.ProcessorConfig{
		MaxWidth:            cfg.MaxWidth,
		MaxHeight:           cfg.MaxHeight,
		OutputFormat:        format,
		Quality:             cfg.Quality,
		PreserveAspectRatio: cfg.PreserveAspectRatio,
		AutoDetectInput:     cfg.AutoDetectInput,
		OutputAsBase64:      cfg.OutputAsBase64,
	}

	return image.NewProcessor(processorCfg)
}

// embeddingProviderWrapper wraps embedding.Provider to satisfy memory.EmbeddingProvider.
type embeddingProviderWrapper struct {
	provider embedding.Provider
}

func (w *embeddingProviderWrapper) Embed(ctx context.Context, text string) ([]float32, error) {
	if w.provider == nil {
		return nil, fmt.Errorf("embedding provider not available")
	}
	return w.provider.Embed(ctx, text)
}

// summarizerProviderWrapper wraps runtime.Manager to satisfy memory.SummaryProvider.
type summarizerProviderWrapper struct {
	manager *runtime.Manager
}

func (w *summarizerProviderWrapper) Summarize(ctx context.Context, texts []string) (string, error) {
	if w.manager == nil {
		return "", fmt.Errorf("runtime manager not available")
	}

	// Build a summarization prompt
	var transcript strings.Builder
	for _, text := range texts {
		transcript.WriteString(text)
		transcript.WriteString("\n")
	}

	prompt := fmt.Sprintf(`Summarise the following conversation into keypoints:
		%s
		`, transcript.String())

	req := runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:   128,
			Temperature: 0.3,
			TopP:        0.9,
		},
	}

	resp, err := w.manager.Generate(ctx, req)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(resp.Text), nil
}

// Close releases underlying resources.
func (p *Pipeline) Close() error {
	if p == nil {
		return nil
	}

	var firstErr error

	// Close omem adapter first (flushes pending writes)
	if p.omemAdapter != nil {
		if err := p.omemAdapter.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if p.vectorEngine != nil {
		if err := p.vectorEngine.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if p.store != nil {
		if err := p.store.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if p.retriever != nil {
		if err := p.retriever.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if p.embedder != nil {
		if err := p.embedder.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if p.manager != nil {
		if err := p.manager.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

// GetMemoryStats returns statistics about the memory system.
func (p *Pipeline) GetMemoryStats(ctx context.Context) (map[string]interface{}, error) {
	if p == nil {
		return nil, fmt.Errorf("pipeline: not initialised")
	}

	stats := make(map[string]interface{})

	// Get vector memory stats if available
	if p.vectorEngine != nil {
		vectorStats, err := p.vectorEngine.GetStats(ctx)
		if err == nil {
			for k, v := range vectorStats {
				stats["vector_"+k] = v
			}
		}
	}

	// Get basic memory store count
	if p.store != nil {
		entries, err := p.store.Recent(1000)
		if err == nil {
			stats["basic_memory_entries"] = len(entries)
		}
	}

	// Get omem stats if available
	if p.omemAdapter != nil && p.omemAdapter.IsEnabled() {
		omemStats := p.omemAdapter.GetStats(ctx)
		for k, v := range omemStats {
			stats["omem_"+k] = v
		}
	}

	// Add config info
	stats["vector_enabled"] = p.cfg.Memory.VectorEnabled
	stats["rag_enabled"] = p.cfg.RAG.Enabled
	stats["compression_enabled"] = p.cfg.Memory.CompressionEnabled
	stats["omem_enabled"] = p.cfg.Memory.Omem.Enabled != nil && *p.cfg.Memory.Omem.Enabled

	return stats, nil
}

// CompressMemory triggers memory compression.
func (p *Pipeline) CompressMemory(ctx context.Context) error {
	if p == nil {
		return fmt.Errorf("pipeline: not initialised")
	}

	if p.vectorEngine == nil {
		return fmt.Errorf("pipeline: vector memory not enabled")
	}

	return p.vectorEngine.Compress(ctx)
}

// SetImageProcessor replaces the current image processor with a custom implementation.
// This allows developers to plug in their own image processing logic.
func (p *Pipeline) SetImageProcessor(processor image.Processor) {
	if p == nil {
		return
	}
	p.imageProcessor = processor
}

// GetImageProcessor returns the current image processor.
func (p *Pipeline) GetImageProcessor() image.Processor {
	if p == nil {
		return nil
	}
	return p.imageProcessor
}

// Respond generates a reply for the supplied user message.
func (p *Pipeline) Respond(ctx context.Context, message string, images []string, opts Options) (Result, error) {
	if p == nil {
		return Result{}, fmt.Errorf("pipeline: not initialised")
	}

	normalized := strings.TrimSpace(message)
	if normalized == "" {
		return Result{}, nil
	}

	// Process images (resize, format)
	log.Printf("pipeline: received %d image(s) from CLI", len(images))
	processedImages, err := p.processImages(ctx, images)
	if err != nil {
		log.Printf("warning: failed to process images: %v", err)
		processedImages = images // Fallback to original
	}
	if len(processedImages) > 0 {
		log.Printf("pipeline: processed %d image(s)", len(processedImages))
	}

	// Check cache
	cacheKey := generateCacheKey(normalized, processedImages)
	if cached, ok := p.responseCache.Load(cacheKey); ok {
		log.Println("returning cached response")
		return cached.(Result), nil
	}

	// 1. Fetch History (Fast, IO bound local DB)
	historyLimit := p.cfg.Memory.TurnsToUse
	if historyLimit <= 0 {
		historyLimit = 6
	}

	previousEntries, err := p.store.Recent(historyLimit)
	if err != nil {
		log.Printf("warning: pipeline failed to load memory history: %v", err)
		// Continue with empty history
		previousEntries = []memory.Entry{}
	}

	history := make([]conversation.HistoryItem, 0, len(previousEntries))
	for i := len(previousEntries) - 1; i >= 0; i-- {
		entry := previousEntries[i]
		history = append(history, conversation.HistoryItem{Role: entry.Role, Content: entry.Content})
	}

	// 2. Parallelize Heavy Context Tasks (Summarization, Vector Search, RAG, Omem)
	var (
		summary       string
		vectorContext string
		omemContext   string
		retrieved     []rag.Document
		wg            sync.WaitGroup
	)

	// Task A: Summarization
	if !opts.DisableSummary && p.summarizer != nil && len(history) > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var err error
			summary, err = p.summarizer.Summarize(ctx, history)
			if err != nil {
				log.Printf("warning: pipeline failed to summarise memory: %v", err)
			}
		}()
	}

	// Task B: Vector Search
	if !opts.DisableVectorMemory && p.vectorEngine != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var err error
			vectorContext, err = p.vectorEngine.BuildContext(ctx, normalized, p.cfg.Memory.MaxContextTokens)
			if err != nil {
				log.Printf("warning: pipeline failed to build vector context: %v", err)
			}
		}()
	}

	// Task C: RAG Retrieval
	limit := p.cfg.RAG.MaxChunks
	if opts.RAGLimit > 0 {
		limit = opts.RAGLimit
	}
	if !opts.DisableRAG && p.retriever != nil && limit > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var err error
			retrieved, err = p.retriever.Retrieve(ctx, normalized, limit)
			if err != nil {
				log.Printf("warning: pipeline failed to retrieve knowledge: %v", err)
			}
		}()
	}

	// Task D: Omem Long-term Memory Context
	if p.omemHook != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			omemContext = p.omemHook.OnBeforeGenerate(ctx, normalized, p.cfg.Memory.Omem.Retrieval.MaxContextTokens)
		}()
	}

	// Wait for all context to be gathered
	wg.Wait()

	// Assemble Context
	promptMessage := normalized
	if len(processedImages) > 0 {
		promptMessage = addImageMarkers(normalized, len(processedImages))
	}

	ctxBuilder := conversation.NewContext(p.cfg.Conversation.SystemMessage, promptMessage)
	ctxBuilder.SetHistory(history)
	ctxBuilder.SetTemplatePath(p.cfg.Conversation.TemplatePath)

	// Adaptive context budget: when using the native backend with a known
	// context size, compute a character budget for conversation history so we
	// don't overflow the model's context window.  Reserve ~25% of the context
	// for system prompt, summary, knowledge, and response generation; the
	// remaining 75% is available for history.  Rough approximation:
	// 1 token ≈ 3.5 characters for English text.
	if p.cfg.Runtime.Backend == "native" && p.cfg.Runtime.Native.ContextSize > 0 {
		ctxSize := p.cfg.Runtime.Native.ContextSize
		historyBudget := int(float64(ctxSize) * 0.75 * 3.5)
		if historyBudget < 500 {
			historyBudget = 500 // absolute minimum
		}
		ctxBuilder.MaxHistoryChars = historyBudget
	}

	// Build combined memory context: omem (long-term) + summary + vector (short-term)
	combinedContext := omem.BuildEnrichedContext(omemContext, summary, vectorContext)
	if combinedContext != "" {
		ctxBuilder.SetSummary(combinedContext)
	}

	if len(retrieved) > 0 {
		contextSnippets := make([]string, 0, len(retrieved))
		for _, doc := range retrieved {
			content := strings.TrimSpace(doc.Text)
			if len(content) > 800 {
				content = content[:800] + "..."
			}
			contextSnippets = append(contextSnippets, fmt.Sprintf("%s (score %.2f): %s", doc.Source, doc.Score, content))
		}
		ctxBuilder.SetKnowledge(contextSnippets)
	}

	prompt := ctxBuilder.Format()

	// Persist user turn
	if err := p.store.Append("user", normalized); err != nil {
		log.Printf("warning: pipeline failed to persist user turn: %v", err)
	}
	// Also store in vector memory engine for semantic retrieval
	if p.vectorEngine != nil {
		if _, err := p.vectorEngine.Store(ctx, normalized, "user"); err != nil {
			log.Printf("warning: pipeline failed to store user turn in vector memory: %v", err)
		}
	}

	// Log if images are being sent
	if len(processedImages) > 0 {
		log.Printf("pipeline: sending request with %d image(s) to runtime", len(processedImages))
	}

	req := runtime.Request{Prompt: prompt, Image: processedImages}
	req.Options = mergeOptions(p.cfg.Runtime.Defaults, opts.GenerationHints)

	if opts.Stream {
		var builder strings.Builder
		var streamStats runtime.Stats
		streamCallback := func(evt runtime.StreamEvent) error {
			if evt.Err != nil {
				return evt.Err
			}
			if !evt.Final {
				builder.WriteString(evt.Token)
			}
			// Capture stats from the final event so they propagate to the Result.
			if evt.Final && evt.Stats != nil {
				streamStats = *evt.Stats
			}
			if opts.StreamCallback != nil {
				return opts.StreamCallback(evt)
			}
			return nil
		}

		err := p.manager.Stream(ctx, req, streamCallback)
		if err != nil {
			return Result{}, err
		}

		text := builder.String()
		if err := p.store.Append("assistant", text); err != nil {
			log.Printf("warning: pipeline failed to persist assistant turn: %v", err)
		}
		// Also store in vector memory engine
		if p.vectorEngine != nil {
			if _, err := p.vectorEngine.Store(ctx, text, "assistant"); err != nil {
				log.Printf("warning: pipeline failed to store assistant turn in vector memory: %v", err)
			}
		}

		// Learn from this conversation turn (async, non-blocking)
		if p.omemHook != nil {
			turnID := fmt.Sprintf("%d", time.Now().UnixNano())
			p.omemHook.OnAfterGenerate(ctx, normalized, text, turnID)
		}

		result := Result{Text: text, Stats: streamStats, Summary: summary, Retrieved: retrieved}
		p.responseCache.Store(cacheKey, result)
		return result, nil
	}

	response, err := p.manager.Generate(ctx, req)
	if err != nil {
		return Result{}, err
	}

	if err := p.store.Append("assistant", response.Text); err != nil {
		log.Printf("warning: pipeline failed to persist assistant turn: %v", err)
	}
	// Also store in vector memory engine
	if p.vectorEngine != nil {
		if _, err := p.vectorEngine.Store(ctx, response.Text, "assistant"); err != nil {
			log.Printf("warning: pipeline failed to store assistant turn in vector memory: %v", err)
		}
	}

	// Learn from this conversation turn (async, non-blocking)
	if p.omemHook != nil {
		turnID := fmt.Sprintf("%d", time.Now().UnixNano())
		p.omemHook.OnAfterGenerate(ctx, normalized, response.Text, turnID)
	}

	result := Result{Text: response.Text, Stats: response.Stats, Summary: summary, Retrieved: retrieved}
	p.responseCache.Store(cacheKey, result)
	return result, nil
}

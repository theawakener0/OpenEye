package pipeline

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"OpenEye/internal/config"
	conversation "OpenEye/internal/context"
	"OpenEye/internal/context/memory"
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
	summarizer     summarizer
	imageProcessor image.Processor
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

	return &Pipeline{
		cfg:            cfg,
		manager:        mgr,
		store:          store,
		vectorEngine:   vectorEngine,
		embedder:       embedder,
		retriever:      retriever,
		summarizer:     summarizer,
		imageProcessor: imageProcessor,
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

	prompt := fmt.Sprintf(`Summarise this following conservation into keypoints:
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

	// Add config info
	stats["vector_enabled"] = p.cfg.Memory.VectorEnabled
	stats["rag_enabled"] = p.cfg.RAG.Enabled
	stats["compression_enabled"] = p.cfg.Memory.CompressionEnabled

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
		return Result{}, fmt.Errorf("pipeline: message cannot be empty")
	}

	// Process images if image processor is available
	processedImages, err := p.processImages(ctx, images)
	if err != nil {
		log.Printf("warning: pipeline failed to process some images: %v", err)
	}

	historyLimit := p.cfg.Memory.TurnsToUse
	if historyLimit <= 0 {
		historyLimit = 6
	}

	previousEntries, err := p.store.Recent(historyLimit)
	if err != nil {
		log.Printf("warning: pipeline failed to load memory history: %v", err)
	}

	history := make([]conversation.HistoryItem, 0, len(previousEntries))
	for i := len(previousEntries) - 1; i >= 0; i-- {
		entry := previousEntries[i]
		history = append(history, conversation.HistoryItem{Role: entry.Role, Content: entry.Content})
	}

	// If images are present, prepend image markers to the message
	// llama.cpp multimodal requires [img-N] markers in the prompt to match the images
	promptMessage := normalized
	if len(processedImages) > 0 {
		promptMessage = addImageMarkers(normalized, len(processedImages))
	}

	ctxBuilder := conversation.NewContext(p.cfg.Conversation.SystemMessage, promptMessage)
	ctxBuilder.SetHistory(history)
	ctxBuilder.SetTemplatePath(p.cfg.Conversation.TemplatePath)

	var summary string
	if !opts.DisableSummary && p.summarizer != nil && len(history) > 0 {
		summary, err = p.summarizer.Summarize(ctx, history)
		if err != nil {
			log.Printf("warning: pipeline failed to summarise memory: %v", err)
		} else if summary != "" {
			ctxBuilder.SetSummary(summary)
		}
	}

	// Retrieve semantically relevant memories from vector engine
	var vectorContext string
	if !opts.DisableVectorMemory && p.vectorEngine != nil {
		vectorContext, err = p.vectorEngine.BuildContext(ctx, normalized, p.cfg.Memory.MaxContextTokens)
		if err != nil {
			log.Printf("warning: pipeline failed to build vector context: %v", err)
		} else if vectorContext != "" {
			// Use vector-retrieved context as enhanced summary if no summary exists
			if summary == "" {
				ctxBuilder.SetSummary(vectorContext)
			}
		}
	}

	var retrieved []rag.Document
	limit := p.cfg.RAG.MaxChunks
	if opts.RAGLimit > 0 {
		limit = opts.RAGLimit
	}
	if !opts.DisableRAG && p.retriever != nil && limit > 0 {
		retrieved, err = p.retriever.Retrieve(ctx, normalized, limit)
		if err != nil {
			log.Printf("warning: pipeline failed to retrieve knowledge: %v", err)
		} else if len(retrieved) > 0 {
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
	}

	prompt := ctxBuilder.Format()
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
		log.Printf("sending request with %d image(s)", len(processedImages))
	}

	req := runtime.Request{Prompt: prompt, Image: processedImages}
	req.Options = mergeOptions(p.cfg.Runtime.Defaults, opts.GenerationHints)

	if opts.Stream {
		var builder strings.Builder
		streamCallback := func(evt runtime.StreamEvent) error {
			if evt.Err != nil {
				return evt.Err
			}
			if !evt.Final {
				builder.WriteString(evt.Token)
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
		return Result{Text: text, Summary: summary, Retrieved: retrieved}, nil
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

	return Result{Text: response.Text, Stats: response.Stats, Summary: summary, Retrieved: retrieved}, nil
}

func mergeOptions(defaults config.GenerationDefaults, hints runtime.GenerationOptions) runtime.GenerationOptions {
	result := runtime.GenerationOptions{
		MaxTokens:     defaults.MaxTokens,
		Temperature:   defaults.Temperature,
		TopK:          defaults.TopK,
		TopP:          defaults.TopP,
		MinP:          defaults.MinP,
		RepeatPenalty: defaults.RepeatPenalty,
		RepeatLastN:   defaults.RepeatLastN,
	}

	if len(hints.Stop) > 0 {
		result.Stop = append([]string(nil), hints.Stop...)
	}
	if hints.MaxTokens != 0 {
		result.MaxTokens = hints.MaxTokens
	}
	if hints.Temperature != 0 {
		result.Temperature = hints.Temperature
	}
	if hints.TopK != 0 {
		result.TopK = hints.TopK
	}
	if hints.TopP != 0 {
		result.TopP = hints.TopP
	}
	if hints.MinP != 0 {
		result.MinP = hints.MinP
	}
	if hints.RepeatPenalty != 0 {
		result.RepeatPenalty = hints.RepeatPenalty
	}
	if hints.RepeatLastN != 0 {
		result.RepeatLastN = hints.RepeatLastN
	}
	if hints.Stop == nil && defaults.Stop != nil {
		result.Stop = append([]string(nil), defaults.Stop...)
	}

	return result
}

// processImages processes input images using the configured image processor.
// Returns processed image data (base64 or file paths depending on config).
func (p *Pipeline) processImages(ctx context.Context, inputs []string) ([]string, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	// If no processor configured, return inputs as-is
	if p.imageProcessor == nil {
		return inputs, nil
	}

	processed, err := p.imageProcessor.ProcessMultiple(ctx, inputs)
	if err != nil {
		// Return original inputs on error, but log the warning
		return inputs, err
	}

	// Convert processed images to the format expected by the runtime
	// llama.cpp expects raw base64 data, not data URIs
	result := make([]string, 0, len(processed))
	for _, img := range processed {
		if img == nil {
			continue
		}
		// Return raw base64 data for llama.cpp
		if img.Base64 != "" {
			result = append(result, img.Base64)
		} else if img.OriginalPath != "" {
			// Fall back to original path if available
			result = append(result, img.OriginalPath)
		}
	}

	return result, nil
}

func initialiseRetriever(cfg config.Config, embedder embedding.Provider) (rag.Retriever, error) {
	if !cfg.RAG.Enabled {
		return nil, nil
	}

	// Use hybrid retriever if enabled
	if cfg.RAG.HybridEnabled {
		hybridCfg := rag.HybridRetrieverConfig{
			RAGConfig:            cfg.RAG,
			MaxCandidates:        cfg.RAG.MaxCandidates,
			DiversityThreshold:   cfg.RAG.DiversityThreshold,
			SemanticWeight:       cfg.RAG.SemanticWeight,
			KeywordWeight:        cfg.RAG.KeywordWeight,
			RecencyWeight:        cfg.RAG.RAGRecencyWeight,
			EnableQueryExpansion: cfg.RAG.EnableQueryExpansion,
			DedupeThreshold:      cfg.RAG.DedupeThreshold,
		}

		retriever, err := rag.NewHybridRetriever(hybridCfg, embedder)
		if err != nil {
			return nil, fmt.Errorf("pipeline: failed to initialise hybrid retriever: %w", err)
		}
		return retriever, nil
	}

	// Fall back to basic filesystem retriever
	retriever, err := rag.NewFilesystemRetriever(cfg.RAG, embedder)
	if err != nil {
		return nil, fmt.Errorf("pipeline: failed to initialise retriever: %w", err)
	}
	return retriever, nil
}

// addImageMarkers prepends image markers to the prompt for llama.cpp multimodal.
// The MTMD (multimodal) marker <__media__> must appear in the prompt for each
// image in multimodal_data. See llama.cpp mtmd_default_marker().
func addImageMarkers(message string, imageCount int) string {
	if imageCount == 0 {
		return message
	}
	var markers strings.Builder
	for i := 0; i < imageCount; i++ {
		markers.WriteString("<__media__>")
		if i < imageCount-1 {
			markers.WriteString(" ")
		}
	}
	markers.WriteString("\n")
	markers.WriteString(message)
	return markers.String()
}

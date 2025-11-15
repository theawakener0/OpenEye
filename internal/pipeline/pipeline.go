package pipeline

import (
	"context"
	"fmt"
	"log"
	"strings"

	"OpenEye/internal/config"
	conversation "OpenEye/internal/context"
	"OpenEye/internal/context/memory"
	"OpenEye/internal/embedding"
	"OpenEye/internal/rag"
	"OpenEye/internal/runtime"
)

// Options controls how the pipeline produces a response.
type Options struct {
	Stream          bool
	StreamCallback  runtime.StreamCallback
	GenerationHints runtime.GenerationOptions
	DisableSummary  bool
	DisableRAG      bool
	RAGLimit        int
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
	cfg        config.Config
	manager    *runtime.Manager
	store      *memory.Store
	embedder   embedding.Provider
	retriever  rag.Retriever
	summarizer summarizer
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

	return &Pipeline{cfg: cfg, manager: mgr, store: store, embedder: embedder, retriever: retriever, summarizer: summarizer}, nil
}

// Close releases underlying resources.
func (p *Pipeline) Close() error {
	if p == nil {
		return nil
	}

	var firstErr error
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

// Respond generates a reply for the supplied user message.
func (p *Pipeline) Respond(ctx context.Context, message string, opts Options) (Result, error) {
	if p == nil {
		return Result{}, fmt.Errorf("pipeline: not initialised")
	}

	normalized := strings.TrimSpace(message)
	if normalized == "" {
		return Result{}, fmt.Errorf("pipeline: message cannot be empty")
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

	ctxBuilder := conversation.NewContext(p.cfg.Conversation.SystemMessage, normalized)
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

	req := runtime.Request{Prompt: prompt}
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
		return Result{Text: text, Summary: summary, Retrieved: retrieved}, nil
	}

	response, err := p.manager.Generate(ctx, req)
	if err != nil {
		return Result{}, err
	}

	if err := p.store.Append("assistant", response.Text); err != nil {
		log.Printf("warning: pipeline failed to persist assistant turn: %v", err)
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

func initialiseRetriever(cfg config.Config, embedder embedding.Provider) (rag.Retriever, error) {
	if !cfg.RAG.Enabled {
		return nil, nil
	}
	retriever, err := rag.NewFilesystemRetriever(cfg.RAG, embedder)
	if err != nil {
		return nil, fmt.Errorf("pipeline: failed to initialise retriever: %w", err)
	}
	return retriever, nil
}

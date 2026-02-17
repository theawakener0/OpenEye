//go:build native

package native

import (
	"context"
	"fmt"
	"log"
	"math"
	"sync"

	"OpenEye/internal/config"
	"OpenEye/internal/embedding"
)

// EmbeddingProvider implements embedding.Provider using native llama.cpp
// bindings. It loads a GGUF embedding model and computes embeddings
// entirely in-process, eliminating the HTTP round-trip to a server.
type EmbeddingProvider struct {
	model      *Model
	ctx        *Context
	hasEncoder bool           // true for BERT/encoder-only models
	nuBatch    uint32         // microbatch size for embedding extraction
	ctxOpts    ContextOptions // saved for context recreation
	mu         sync.Mutex
}

func init() {
	embedding.RegisterProvider("native", newNativeEmbeddingProvider)
}

// newNativeEmbeddingProvider constructs a native embedding provider.
func newNativeEmbeddingProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
	nc := cfg.Native

	if nc.ModelPath == "" {
		return nil, fmt.Errorf("native embedding: model_path is required")
	}

	// Ensure backend is initialized.
	BackendInit()

	// Load model.
	modelOpts := DefaultModelOptions()
	modelOpts.NGPULayers = int32(nc.GPULayers)
	if nc.Mmap != nil {
		modelOpts.UseMmap = *nc.Mmap
	}
	if nc.Mlock != nil {
		modelOpts.UseMlock = *nc.Mlock
	}

	model, err := LoadModel(nc.ModelPath, modelOpts)
	if err != nil {
		return nil, fmt.Errorf("native embedding: %w", err)
	}

	// Auto-detect embedding model preset from GGUF metadata and filename.
	info := model.Info()
	preset, matched := matchPreset(info.Description, nc.ModelPath)
	if matched {
		log.Printf("native embedding: auto-detected preset %q", preset.Name)
	}

	// Create context with embeddings enabled, applying preset defaults.
	ctxOpts := DefaultContextOptions()
	ctxOpts.Embeddings = true
	if matched {
		applyPresetToContextOpts(&ctxOpts, preset)
	}
	if nc.ContextSize > 0 {
		ctxOpts.NCtx = uint32(nc.ContextSize)
	}
	if nc.BatchSize > 0 {
		ctxOpts.NBatch = uint32(nc.BatchSize)
	}
	if nc.UbatchSize > 0 {
		ctxOpts.NUbatch = uint32(nc.UbatchSize)
	} else if info.NCtxTrain > 0 {
		// Auto-detect ubatch from model's training context size
		ctxOpts.NUbatch = uint32(info.NCtxTrain)
		log.Printf("native embedding: auto-detected ubatch size from model: %d", ctxOpts.NUbatch)
	} else {
		ctxOpts.NUbatch = 512 // llama.cpp default
		log.Printf("native embedding: using default ubatch size: %d", ctxOpts.NUbatch)
	}
	if nc.Threads > 0 {
		ctxOpts.NThreads = int32(nc.Threads)
	}

	llCtx, err := NewContext(model, ctxOpts)
	if err != nil {
		model.Close()
		return nil, fmt.Errorf("native embedding: %w", err)
	}

	log.Printf("native embedding: model loaded — %s, %d dims, ctx=%d, ubatch=%d, encoder=%v",
		info.Description, info.NEmbedding, ctxOpts.NCtx, ctxOpts.NUbatch, info.HasEncoder)

	return &EmbeddingProvider{
		model:      model,
		ctx:        llCtx,
		hasEncoder: info.HasEncoder,
		nuBatch:    ctxOpts.NUbatch,
		ctxOpts:    ctxOpts,
	}, nil
}

// Embed tokenizes the input text and returns its embedding vector.
// The returned vector is L2-normalized for cosine similarity.
// It includes fallback retry logic to handle crashes due to batch size issues.
// All errors are logged but handled gracefully to prevent cascade failures.
func (p *EmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.model == nil {
		log.Printf("native embedding: provider is closed, skipping embed")
		return nil, nil // Return nil silently instead of error to prevent cascade failures
	}

	// Tokenize.
	tokens, err := p.model.Tokenize(text, true, true)
	if err != nil {
		return nil, fmt.Errorf("native embedding: tokenize: %w", err)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("native embedding: empty token sequence")
	}

	// Attempt embedding with fallback retry (max 3 attempts)
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		// Check if context is closed (may have been recreated in previous attempt)
		if p.ctx == nil {
			break
		}

		result, err := p.tryEmbed(tokens, attempt > 0)
		if err == nil {
			return result, nil
		}

		lastErr = err
		log.Printf("native embedding: attempt %d failed: %v", attempt+1, err)

		// Attempt recovery strategies
		if attempt == 0 {
			// First failure: clear KV cache and retry
			if !p.hasEncoder && p.ctx != nil {
				p.ctx.ClearKV()
			}
		} else if attempt == 1 {
			// Second failure: truncate tokens to ubatch size and retry
			if int(p.nuBatch) > 0 && len(tokens) > int(p.nuBatch) {
				tokens = tokens[len(tokens)-int(p.nuBatch):]
				log.Printf("native embedding: truncating to %d tokens", len(tokens))
			}
		}
		// Third attempt will recreate context if still failing
	}

	// Last resort: try to recreate context
	if p.ctx == nil {
		newCtx, err := NewContext(p.model, p.ctxOpts)
		if err != nil {
			return nil, fmt.Errorf("native embedding: context recreation failed: %w", err)
		}
		p.ctx = newCtx
		result, err := p.tryEmbed(tokens, false)
		if err != nil {
			return nil, fmt.Errorf("native embedding: fallback failed after context recreation: %w", err)
		}
		return result, nil
	}

	return nil, fmt.Errorf("native embedding: all fallback attempts failed: %w", lastErr)
}

// tryEmbed attempts to compute embeddings, returning error if it fails.
func (p *EmbeddingProvider) tryEmbed(tokens []int32, truncated bool) ([]float32, error) {
	if p.ctx == nil {
		return nil, fmt.Errorf("native embedding: context is nil")
	}

	// Clear KV cache for decoder models before evaluation
	if !p.hasEncoder {
		p.ctx.ClearKV()
	}

	// Encode/evaluate tokens
	var err error
	if p.hasEncoder {
		err = p.ctx.Encode(tokens)
	} else {
		err = p.ctx.Eval(tokens)
	}
	if err != nil {
		return nil, fmt.Errorf("native embedding: encode/eval: %w", err)
	}

	// Extract pooled sequence embeddings (preferred for embedding models).
	raw := p.ctx.GetEmbeddingsSeq()
	if len(raw) == 0 {
		// Fall back to last-token embeddings.
		raw = p.ctx.GetEmbeddings()
	}
	if len(raw) == 0 {
		return nil, fmt.Errorf("native embedding: no embeddings returned (is this an embedding model?)")
	}

	// Validate dimensionality matches model's expected embedding size.
	info := p.model.Info()
	if expected := int(info.NEmbedding); expected > 0 && len(raw) != expected {
		return nil, fmt.Errorf("native embedding: dimension mismatch: got %d, expected %d", len(raw), expected)
	}

	// L2-normalize for cosine similarity.
	l2Normalize(raw)

	return raw, nil
}

// Close frees all native resources.
func (p *EmbeddingProvider) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.ctx != nil {
		p.ctx.Close()
		p.ctx = nil
	}
	if p.model != nil {
		p.model.Close()
		p.model = nil
	}
	return nil
}

// l2Normalize normalizes a vector in-place to unit length.
func l2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	norm := float32(math.Sqrt(sum))
	for i := range v {
		v[i] /= norm
	}
}

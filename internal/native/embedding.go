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
	hasEncoder bool // true for BERT/encoder-only models
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
	if nc.Threads > 0 {
		ctxOpts.NThreads = int32(nc.Threads)
	}

	llCtx, err := NewContext(model, ctxOpts)
	if err != nil {
		model.Close()
		return nil, fmt.Errorf("native embedding: %w", err)
	}

	log.Printf("native embedding: model loaded — %s, %d dims, ctx=%d, encoder=%v",
		info.Description, info.NEmbedding, ctxOpts.NCtx, info.HasEncoder)

	return &EmbeddingProvider{
		model:      model,
		ctx:        llCtx,
		hasEncoder: info.HasEncoder,
	}, nil
}

// Embed tokenizes the input text and returns its embedding vector.
// The returned vector is L2-normalized for cosine similarity.
func (p *EmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.model == nil || p.ctx == nil {
		return nil, fmt.Errorf("native embedding: provider is closed")
	}

	// Tokenize.
	tokens, err := p.model.Tokenize(text, true, true)
	if err != nil {
		return nil, fmt.Errorf("native embedding: tokenize: %w", err)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("native embedding: empty token sequence")
	}

	// Clear KV cache (decoder models) and evaluate.
	if p.hasEncoder {
		// Encoder-only models (BERT): use encode path directly.
		// No KV cache to clear — encoder is stateless per call.
		if err := p.ctx.Encode(tokens); err != nil {
			return nil, fmt.Errorf("native embedding: encode: %w", err)
		}
	} else {
		// Decoder models used for embeddings: clear KV and decode.
		p.ctx.ClearKV()
		if err := p.ctx.Eval(tokens); err != nil {
			return nil, fmt.Errorf("native embedding: eval: %w", err)
		}
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

	// L2-normalize for cosine similarity. The raw slice is already a
	// Go-owned copy (allocated in cGetEmbeddings/cGetEmbeddingsSeq),
	// so we can normalize in place without an extra allocation.
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

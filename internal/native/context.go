//go:build native

package native

/*
#include "binding.h"
*/
import "C"
import (
	"fmt"
	"sync"
)

// Context wraps a llama.cpp inference context. It holds the KV cache and
// manages batch evaluation. NOT safe for concurrent use — callers must
// synchronize access externally (the adapter does this).
type Context struct {
	handle C.oe_context_t
	model  *Model
	modelH C.oe_model_t // immutable copy of model handle, captured at construction
	mu     sync.Mutex
	closed bool

	// Track the current position in the KV cache for auto-advancing decode.
	pos int32
}

// ContextOptions configures the inference context.
type ContextOptions struct {
	// NCtx is the context window size. 0 = use model's training context.
	NCtx uint32

	// NBatch is the max batch size for prompt processing.
	NBatch uint32

	// NThreads is the number of threads for single-token generation.
	// 0 = auto-detect.
	NThreads int32

	// NThreadsBatch is the number of threads for batch processing.
	// 0 = same as NThreads.
	NThreadsBatch int32

	// Embeddings enables embedding extraction mode.
	Embeddings bool

	// FlashAttn controls flash attention: -1=auto, 0=disabled, 1=enabled.
	FlashAttn int32

	// TypeK controls KV cache key quantization: 0=f16, 1=q8_0, 2=q4_0.
	TypeK int32

	// TypeV controls KV cache value quantization: 0=f16, 1=q8_0, 2=q4_0.
	TypeV int32
}

// DefaultContextOptions returns sensible defaults for edge inference.
func DefaultContextOptions() ContextOptions {
	return ContextOptions{
		NCtx:      2048, // reasonable for 1-3B models on edge
		NBatch:    512,  // process up to 512 tokens at once
		NThreads:  4,    // RPi 5 has 4 cores
		FlashAttn: -1,   // auto-detect
	}
}

// NewContext creates an inference context from a loaded model.
func NewContext(model *Model, opts ContextOptions) (*Context, error) {
	if model == nil || model.handle == nil {
		return nil, fmt.Errorf("native: model is nil or closed")
	}

	handle := cContextNew(model.handle, opts.NCtx, opts.NBatch,
		opts.NThreads, opts.NThreadsBatch, opts.Embeddings, opts.FlashAttn,
		opts.TypeK, opts.TypeV)
	if handle == nil {
		return nil, fmt.Errorf("native: failed to create context")
	}

	return &Context{
		handle: handle,
		model:  model,
		modelH: model.handle, // snapshot: immutable for this context's lifetime
		pos:    0,
	}, nil
}

// Eval processes a batch of tokens, appending them to the KV cache.
// Positions are managed automatically. Only the last token produces logits.
// Returns an error if the KV cache is full or another error occurs.
func (c *Context) Eval(tokens []int32) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("native: context is closed")
	}
	if len(tokens) == 0 {
		return nil
	}

	rc := cDecodeBatch(c.handle, tokens, c.pos)
	if rc != 0 {
		if rc == 1 {
			return fmt.Errorf("native: KV cache full, need larger context or shorter prompt")
		}
		return fmt.Errorf("native: decode failed with code %d", rc)
	}

	c.pos += int32(len(tokens))
	return nil
}

// EvalToken processes a single token, appending it to the KV cache.
func (c *Context) EvalToken(token int32) error {
	return c.Eval([]int32{token})
}

// Encode processes a batch of tokens through the model's encoder path.
// This is the correct method for encoder-only models (BERT, embedding models).
// Unlike Eval/Decode, positions always start at 0 and all tokens are marked
// as outputs for embedding extraction. Does NOT advance the KV cache position.
func (c *Context) Encode(tokens []int32) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("native: context is closed")
	}
	if len(tokens) == 0 {
		return nil
	}

	rc := cEncode(c.handle, tokens)
	if rc != 0 {
		return fmt.Errorf("native: encode failed with code %d", rc)
	}
	return nil
}

// SampleToken samples the next token using the provided sampler chain.
// This uses the logits from the last Eval call.
func (c *Context) SampleToken(sampler *SamplerChain) (int32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, fmt.Errorf("native: context is closed")
	}
	if sampler == nil {
		return 0, fmt.Errorf("native: sampler is nil")
	}

	token := cSamplerSample(sampler.handle, c.handle, -1)
	return token, nil
}

// GetEmbeddings returns the embedding vector for the last evaluated token.
// Requires that the context was created with Embeddings=true.
func (c *Context) GetEmbeddings() []float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	return cGetEmbeddings(c.handle, c.modelH, -1)
}

// GetEmbeddingsSeq returns pooled embeddings for sequence 0.
func (c *Context) GetEmbeddingsSeq() []float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	return cGetEmbeddingsSeq(c.handle, c.modelH, 0)
}

// Pos returns the current position in the KV cache.
func (c *Context) Pos() int32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.pos
}

// ClearKV clears the entire KV cache and resets the position counter.
func (c *Context) ClearKV() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	cMemoryClear(c.handle)
	c.pos = 0
}

// TruncateKV removes tokens from position p0 onwards from the KV cache
// and resets the position counter to p0. Useful for prompt caching:
// keep the shared prefix, discard the diverging suffix.
func (c *Context) TruncateKV(p0 int32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	cMemorySeqRm(c.handle, 0, p0, -1)
	c.pos = p0
}

// ShiftKV implements context window sliding. It removes the oldest nDiscard
// tokens from the KV cache and shifts the remaining positions down by
// nDiscard, keeping them contiguous. The position counter is updated
// accordingly. This enables infinite-length conversations by recycling
// context space instead of failing when the window fills up.
//
// The caller must ensure nDiscard < c.pos. Typically called when the
// context reaches ~90% capacity, discarding ~25% of the oldest entries.
func (c *Context) ShiftKV(nDiscard int32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || nDiscard <= 0 || nDiscard >= c.pos {
		return
	}
	// 1. Remove tokens in positions [0, nDiscard) for sequence 0.
	cMemorySeqRm(c.handle, 0, 0, nDiscard)
	// 2. Shift remaining positions [nDiscard, inf) down by nDiscard.
	cMemorySeqAdd(c.handle, 0, nDiscard, -1, -nDiscard)
	// 3. Update position counter.
	c.pos -= nDiscard
}

// SetThreads updates the thread counts for generation and batch processing.
func (c *Context) SetThreads(nThreads, nThreadsBatch int32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	cSetNThreads(c.handle, nThreads, nThreadsBatch)
}

// SetEmbeddings enables or disables embedding extraction at runtime.
func (c *Context) SetEmbeddings(enabled bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	cSetEmbeddings(c.handle, enabled)
}

// Warmup runs a warmup pass to preload tensor weights into CPU cache.
// This evaluates a single BOS token in warmup mode (skipping certain
// computations), then clears the KV cache.
func (c *Context) Warmup() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("native: context is closed")
	}

	cSetWarmup(c.handle, true)

	// Decode a single BOS token to trigger weight loading.
	bos := cTokenBOS(c.modelH)
	tokens := []int32{bos}
	rc := cDecodeBatch(c.handle, tokens, 0)

	cSetWarmup(c.handle, false)

	// Clear the KV cache after warmup — we don't want the BOS token lingering.
	cMemoryClear(c.handle)
	c.pos = 0

	if rc != 0 {
		return fmt.Errorf("native: warmup decode failed with code %d", rc)
	}
	return nil
}

// WarmupFull runs an extended warmup pass that processes a short batch of
// tokens. This better exercises the model's attention layers and memory
// subsystem compared to the single-token Warmup, resulting in more
// consistent first-inference latency on edge devices.
func (c *Context) WarmupFull(nTokens int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("native: context is closed")
	}
	if nTokens <= 0 {
		nTokens = 8 // sensible default: exercises batch path without being slow
	}

	cSetWarmup(c.handle, true)

	// Build a batch of BOS tokens. The actual token values don't matter
	// since warmup mode skips logit computation — we just want to
	// exercise the matrix multiplications and memory access patterns.
	bos := cTokenBOS(c.modelH)
	tokens := make([]int32, nTokens)
	for i := range tokens {
		tokens[i] = bos
	}

	rc := cDecodeBatch(c.handle, tokens, 0)

	cSetWarmup(c.handle, false)

	// Clear the KV cache after warmup.
	cMemoryClear(c.handle)
	c.pos = 0

	if rc != 0 {
		return fmt.Errorf("native: full warmup decode failed with code %d", rc)
	}
	return nil
}

// Perf returns performance counters since last reset.
func (c *Context) Perf() PerfData {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return PerfData{}
	}
	return cPerfContext(c.handle)
}

// PerfReset resets the performance counters.
func (c *Context) PerfReset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	cPerfContextReset(c.handle)
}

// Handle returns the raw C context handle for use by VisionContext and other
// low-level operations. Returns nil if the context has been closed.
func (c *Context) Handle() C.oe_context_t {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	return c.handle
}

// IsClosed returns true if the context has been freed.
func (c *Context) IsClosed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closed
}

// SetPos directly sets the KV cache position counter. This is used after
// vision eval, which manages KV positions internally via mtmd — the Go
// Context's pos field must be synchronized with where mtmd left off.
func (c *Context) SetPos(pos int32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.pos = pos
}

// EvalLogitsAll processes a batch of tokens like Eval, but computes logits
// for ALL tokens in the batch (not just the last). This is needed for
// speculative decoding verification where the target model must check each
// draft token's logits individually.
func (c *Context) EvalLogitsAll(tokens []int32) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("native: context is closed")
	}
	if len(tokens) == 0 {
		return nil
	}

	rc := cDecodeBatchLogitsAll(c.handle, tokens, c.pos)
	if rc != 0 {
		if rc == 1 {
			return fmt.Errorf("native: KV cache full during speculative verify")
		}
		return fmt.Errorf("native: decode logits-all failed with code %d", rc)
	}

	c.pos += int32(len(tokens))
	return nil
}

// GetLogitsAt returns the logits for the token at the given batch output
// index. The vocabSize must match the model's vocabulary size.
// The returned slice is context-owned and valid until the next decode call.
func (c *Context) GetLogitsAt(idx int32, vocabSize int32) []float32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	return cGetLogits(c.handle, idx, vocabSize)
}

// Close frees the context. The parent Model must outlive this Context.
func (c *Context) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	cContextFree(c.handle)
	c.handle = nil
	return nil
}

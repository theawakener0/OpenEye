//go:build native

package native

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"OpenEye/internal/config"
	"OpenEye/internal/runtime"
)

// Adapter implements runtime.Adapter using the native llama.cpp bindings.
// It manages the model lifecycle, context, and sampler chain, providing
// both blocking (Generate) and streaming (Stream) token generation.
type Adapter struct {
	model  *Model
	ctx    *Context
	vision *VisionContext // nil if no mmproj configured
	cfg    config.RuntimeConfig
	mu     sync.Mutex

	// Prompt caching: store the last prompt's token sequence so we can
	// reuse the KV cache prefix on the next request. On edge devices this
	// eliminates redundant re-evaluation of system prompt + shared history,
	// dramatically reducing time-to-first-token (TTFT).
	lastPromptTokens []int32
}

// newNativeAdapter constructs a native adapter from configuration.
// This is the factory registered with the runtime registry.
func newNativeAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
	nc := cfg.Native

	if nc.ModelPath == "" {
		return nil, fmt.Errorf("native: model_path is required")
	}

	// Initialize the backend (idempotent, but we call it here to be safe).
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
		return nil, fmt.Errorf("native: %w", err)
	}

	// Auto-detect model preset from GGUF metadata and filename.
	info := model.Info()
	preset, matched := matchPreset(info.Description, nc.ModelPath)
	if matched {
		logPreset(preset, info.Description)
	}

	// Create inference context, applying preset defaults for unset values.
	ctxOpts := DefaultContextOptions()
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
	if nc.ThreadsBatch > 0 {
		ctxOpts.NThreadsBatch = int32(nc.ThreadsBatch)
	}
	if nc.FlashAttention != nil {
		if *nc.FlashAttention {
			ctxOpts.FlashAttn = 1
		} else {
			ctxOpts.FlashAttn = 0
		}
	}

	llCtx, err := NewContext(model, ctxOpts)
	if err != nil {
		model.Close()
		return nil, fmt.Errorf("native: %w", err)
	}

	// Apply preset defaults to generation parameters.
	if matched {
		applyPresetToDefaults(&cfg.Defaults, preset)
	}

	// Determine whether to run warmup. Explicit config takes precedence,
	// otherwise use the preset's recommendation.
	doWarmup := nc.Warmup
	if !doWarmup && matched && preset.WarmupRecommended {
		doWarmup = true
		log.Printf("native: warmup recommended by %q preset", preset.Name)
	}

	// Optional warmup pass to preload weights into CPU cache.
	if doWarmup {
		warmupStart := time.Now()
		if nc.WarmupTokens > 0 {
			// Multi-token warmup: exercises batch decode path and attention layers.
			if err := llCtx.WarmupFull(nc.WarmupTokens); err != nil {
				llCtx.Close()
				model.Close()
				return nil, fmt.Errorf("native: warmup failed: %w", err)
			}
			log.Printf("native: full warmup (%d tokens) completed in %v", nc.WarmupTokens, time.Since(warmupStart))
		} else {
			// Single-token warmup: fast, just preloads weight tensors.
			if err := llCtx.Warmup(); err != nil {
				llCtx.Close()
				model.Close()
				return nil, fmt.Errorf("native: warmup failed: %w", err)
			}
			log.Printf("native: warmup completed in %v", time.Since(warmupStart))
		}
	}

	log.Printf("native: model loaded — %s, %d params, ctx=%d, batch=%d, threads=%d",
		info.Description, info.NParams, ctxOpts.NCtx, ctxOpts.NBatch, ctxOpts.NThreads)

	// Initialize vision (multimodal) if an mmproj path is configured.
	var vision *VisionContext
	if nc.MmprojPath != "" {
		var visionErr error
		vision, visionErr = NewVisionContext(nc.MmprojPath, model, int(ctxOpts.NThreads), nc.GPULayers > 0)
		if visionErr != nil {
			llCtx.Close()
			model.Close()
			return nil, fmt.Errorf("native: %w", visionErr)
		}
		log.Printf("native: vision enabled via mmproj")
	}

	return &Adapter{
		model:  model,
		ctx:    llCtx,
		vision: vision,
		cfg:    cfg,
	}, nil
}

// Name returns the adapter identifier.
func (a *Adapter) Name() string { return "native" }

// Generate performs a blocking completion, returning the full response.
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	opts := mergeNativeOptions(a.cfg.Defaults, req.Options)

	// Build sampler chain for this request.
	sampler := NewSamplerChain(SamplerOptions{
		Temperature:      float32(opts.Temperature),
		TopK:             int32(opts.TopK),
		TopP:             float32(opts.TopP),
		MinP:             float32(opts.MinP),
		RepeatPenalty:    float32(opts.RepeatPenalty),
		RepeatLastN:      int32(opts.RepeatLastN),
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		Seed:             0xFFFFFFFF,
	})
	defer sampler.Close()

	// Reset performance counters.
	a.ctx.PerfReset()

	// Track prompt token count for stats (vision path estimates differently).
	var promptTokenCount int
	var prefixLen int

	// --- Vision path: images present and vision context available ---
	if len(req.Image) > 0 && a.vision != nil {
		// Clear KV cache — prompt caching doesn't work with vision requests
		// because the KV layout includes image embeddings that differ each time.
		a.ctx.ClearKV()
		a.lastPromptTokens = nil

		batchSize := int32(a.cfg.Native.BatchSize)
		if batchSize <= 0 {
			batchSize = 512
		}

		newPos, visionErr := a.vision.EvalWithImages(a.ctx, req.Prompt, req.Image, batchSize)
		if visionErr != nil {
			return runtime.Response{}, fmt.Errorf("native: vision eval: %w", visionErr)
		}

		// Synchronize the context's position with where vision eval left off.
		a.ctx.SetPos(newPos)
		promptTokenCount = int(newPos) // approximate: vision tokens + text tokens

	} else {
		// --- Standard text-only path ---

		// Tokenize the prompt.
		tokens, err := a.model.Tokenize(req.Prompt, true, true)
		if err != nil {
			return runtime.Response{}, fmt.Errorf("native: tokenize: %w", err)
		}
		promptTokenCount = len(tokens)

		// Prompt caching: find the longest common prefix with the last prompt.
		// Reuse cached KV entries and only evaluate the new suffix tokens.
		prefixLen = commonPrefixLen(a.lastPromptTokens, tokens)
		if prefixLen > 0 {
			// Keep the shared prefix in KV, discard everything after.
			a.ctx.TruncateKV(int32(prefixLen))
		} else {
			a.ctx.ClearKV()
		}

		// Evaluate only the new tokens (from prefixLen onwards).
		newTokens := tokens[prefixLen:]
		if len(newTokens) > 0 {
			if err := a.ctx.Eval(newTokens); err != nil {
				// Generation failed before starting; invalidate prompt cache.
				a.lastPromptTokens = nil
				return runtime.Response{}, fmt.Errorf("native: eval prompt: %w", err)
			}
		}

		// Store tokens for prompt caching on next call.
		a.lastPromptTokens = make([]int32, len(tokens))
		copy(a.lastPromptTokens, tokens)
	}

	// Resolve max tokens for generation.
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	// Context window overflow protection: ensure there's room for generation.
	ctxSize := int(a.cfg.Native.ContextSize)
	if ctxSize <= 0 {
		ctxSize = 2048 // default
	}
	currentPos := int(a.ctx.Pos())
	if currentPos+maxTokens > ctxSize {
		available := ctxSize - currentPos
		if available <= 0 {
			a.lastPromptTokens = nil
			return runtime.Response{}, fmt.Errorf("native: context window full (%d/%d tokens used by prompt), no room for generation", currentPos, ctxSize)
		}
		maxTokens = available
	}

	// Generate tokens.
	var result strings.Builder
	tokensGenerated := 0
	startTime := time.Now()
	var ttft time.Duration
	finishReason := "length" // default: loop exhausted max tokens

	for i := 0; i < maxTokens; i++ {
		// Check for cancellation.
		select {
		case <-ctx.Done():
			// Partial generation; invalidate prompt cache since KV state is
			// indeterminate relative to what we'd expect on next call.
			a.lastPromptTokens = nil
			return runtime.Response{
				Text: result.String(),
				Stats: runtime.Stats{
					TokensEvaluated: promptTokenCount,
					TokensGenerated: tokensGenerated,
					TokensCached:    prefixLen,
					Duration:        time.Since(startTime),
					TTFT:            ttft,
				},
				Finish: "cancelled",
			}, ctx.Err()
		default:
		}

		// Sample next token.
		token, err := a.ctx.SampleToken(sampler)
		if err != nil {
			a.lastPromptTokens = nil
			return runtime.Response{}, fmt.Errorf("native: sample: %w", err)
		}

		// Record TTFT on the first generated token.
		if tokensGenerated == 0 {
			ttft = time.Since(startTime)
		}

		// Check for end-of-generation.
		if a.model.TokenIsEOG(token) {
			finishReason = "stop"
			break
		}

		// Convert token to text.
		piece := a.model.TokenToPiece(token)
		result.WriteString(piece)
		tokensGenerated++

		// Check for stop sequences.
		if shouldStop(result.String(), opts.Stop) {
			trimmed := trimAtStop(result.String(), opts.Stop)
			result.Reset()
			result.WriteString(trimmed)
			finishReason = "stop"
			break
		}

		// Evaluate the token to advance the KV cache.
		if err := a.ctx.EvalToken(token); err != nil {
			a.lastPromptTokens = nil
			return runtime.Response{}, fmt.Errorf("native: eval token: %w", err)
		}
	}

	// Generation succeeded — prompt cache was already updated in the
	// text-only path above. For vision path, prompt cache stays nil
	// (vision requests don't benefit from text prompt caching).

	perf := a.ctx.Perf()
	duration := time.Since(startTime)

	// Compute throughput from llama.cpp perf counters.
	var promptTPS, genTPS float64
	if perf.PromptMs > 0 && perf.PromptCount > 0 {
		promptTPS = float64(perf.PromptCount) / (perf.PromptMs / 1000.0)
	}
	if perf.EvalMs > 0 && perf.EvalCount > 0 {
		genTPS = float64(perf.EvalCount) / (perf.EvalMs / 1000.0)
	}

	return runtime.Response{
		Text: result.String(),
		Stats: runtime.Stats{
			TokensEvaluated: promptTokenCount,
			TokensGenerated: tokensGenerated,
			TokensCached:    prefixLen,
			Duration:        duration,
			TTFT:            ttft,
			PromptTPS:       promptTPS,
			GenerationTPS:   genTPS,
		},
		Raw:    perf,
		Finish: finishReason,
	}, nil
}

// Stream performs token-by-token streaming generation.
func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	opts := mergeNativeOptions(a.cfg.Defaults, req.Options)

	// Build sampler chain.
	sampler := NewSamplerChain(SamplerOptions{
		Temperature:      float32(opts.Temperature),
		TopK:             int32(opts.TopK),
		TopP:             float32(opts.TopP),
		MinP:             float32(opts.MinP),
		RepeatPenalty:    float32(opts.RepeatPenalty),
		RepeatLastN:      int32(opts.RepeatLastN),
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		Seed:             0xFFFFFFFF,
	})
	defer sampler.Close()

	// Reset performance counters.
	a.ctx.PerfReset()

	// Track prompt token count for stats (vision path estimates differently).
	var promptTokenCount int
	var prefixLen int

	// --- Vision path: images present and vision context available ---
	if len(req.Image) > 0 && a.vision != nil {
		// Clear KV cache — prompt caching doesn't work with vision requests.
		a.ctx.ClearKV()
		a.lastPromptTokens = nil

		batchSize := int32(a.cfg.Native.BatchSize)
		if batchSize <= 0 {
			batchSize = 512
		}

		newPos, visionErr := a.vision.EvalWithImages(a.ctx, req.Prompt, req.Image, batchSize)
		if visionErr != nil {
			return fmt.Errorf("native: vision eval: %w", visionErr)
		}

		// Synchronize the context's position with where vision eval left off.
		a.ctx.SetPos(newPos)
		promptTokenCount = int(newPos)

	} else {
		// --- Standard text-only path ---

		// Tokenize prompt.
		tokens, err := a.model.Tokenize(req.Prompt, true, true)
		if err != nil {
			return fmt.Errorf("native: tokenize: %w", err)
		}
		promptTokenCount = len(tokens)

		// Prompt caching: reuse KV cache prefix.
		prefixLen = commonPrefixLen(a.lastPromptTokens, tokens)
		if prefixLen > 0 {
			a.ctx.TruncateKV(int32(prefixLen))
		} else {
			a.ctx.ClearKV()
		}

		newTokens := tokens[prefixLen:]
		if len(newTokens) > 0 {
			if err := a.ctx.Eval(newTokens); err != nil {
				a.lastPromptTokens = nil
				return fmt.Errorf("native: eval prompt: %w", err)
			}
		}

		// Store tokens for prompt caching on next call.
		a.lastPromptTokens = make([]int32, len(tokens))
		copy(a.lastPromptTokens, tokens)
	}

	// Resolve max tokens for generation.
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	// Context window overflow protection.
	ctxSize := int(a.cfg.Native.ContextSize)
	if ctxSize <= 0 {
		ctxSize = 2048
	}
	currentPos := int(a.ctx.Pos())
	if currentPos+maxTokens > ctxSize {
		available := ctxSize - currentPos
		if available <= 0 {
			a.lastPromptTokens = nil
			return fmt.Errorf("native: context window full (%d/%d tokens used by prompt), no room for generation", currentPos, ctxSize)
		}
		maxTokens = available
	}

	// Generate tokens with streaming callbacks.
	var accumulated strings.Builder
	idx := 0
	startTime := time.Now()
	var ttft time.Duration

	for i := 0; i < maxTokens; i++ {
		// Check cancellation.
		select {
		case <-ctx.Done():
			a.lastPromptTokens = nil
			_ = cb(runtime.StreamEvent{Final: true, Err: ctx.Err()})
			return ctx.Err()
		default:
		}

		// Sample.
		token, err := a.ctx.SampleToken(sampler)
		if err != nil {
			a.lastPromptTokens = nil
			return fmt.Errorf("native: sample: %w", err)
		}

		// Record TTFT on the first generated token.
		if idx == 0 {
			ttft = time.Since(startTime)
		}

		// EOG check.
		if a.model.TokenIsEOG(token) {
			break
		}

		piece := a.model.TokenToPiece(token)
		accumulated.WriteString(piece)

		// Check stop sequences.
		if shouldStop(accumulated.String(), opts.Stop) {
			break
		}

		// Emit token.
		if err := cb(runtime.StreamEvent{Token: piece, Index: idx}); err != nil {
			a.lastPromptTokens = nil
			return err
		}
		idx++

		// Advance KV cache.
		if err := a.ctx.EvalToken(token); err != nil {
			a.lastPromptTokens = nil
			return fmt.Errorf("native: eval token: %w", err)
		}
	}

	// Streaming generation succeeded — prompt cache was already updated
	// in the text-only path above. For vision path, it stays nil.

	// Collect perf counters and build final stats.
	perf := a.ctx.Perf()
	duration := time.Since(startTime)

	var promptTPS, genTPS float64
	if perf.PromptMs > 0 && perf.PromptCount > 0 {
		promptTPS = float64(perf.PromptCount) / (perf.PromptMs / 1000.0)
	}
	if perf.EvalMs > 0 && perf.EvalCount > 0 {
		genTPS = float64(perf.EvalCount) / (perf.EvalMs / 1000.0)
	}

	finalStats := &runtime.Stats{
		TokensEvaluated: promptTokenCount,
		TokensGenerated: idx,
		TokensCached:    prefixLen,
		Duration:        duration,
		TTFT:            ttft,
		PromptTPS:       promptTPS,
		GenerationTPS:   genTPS,
	}

	// Signal completion with stats.
	return cb(runtime.StreamEvent{Final: true, Stats: finalStats})
}

// Close frees all native resources.
func (a *Adapter) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.vision != nil {
		a.vision.Close()
		a.vision = nil
	}
	if a.ctx != nil {
		a.ctx.Close()
		a.ctx = nil
	}
	if a.model != nil {
		a.model.Close()
		a.model = nil
	}
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func mergeNativeOptions(base config.GenerationDefaults, override runtime.GenerationOptions) runtime.GenerationOptions {
	result := runtime.GenerationOptions{
		MaxTokens:     base.MaxTokens,
		Temperature:   base.Temperature,
		TopK:          base.TopK,
		TopP:          base.TopP,
		MinP:          base.MinP,
		RepeatPenalty: base.RepeatPenalty,
		RepeatLastN:   base.RepeatLastN,
		Stop:          base.Stop,
	}

	if override.MaxTokens != 0 {
		result.MaxTokens = override.MaxTokens
	}
	if override.Temperature != 0 {
		result.Temperature = override.Temperature
	}
	if override.TopK != 0 {
		result.TopK = override.TopK
	}
	if override.TopP != 0 {
		result.TopP = override.TopP
	}
	if override.MinP != 0 {
		result.MinP = override.MinP
	}
	if override.RepeatPenalty != 0 {
		result.RepeatPenalty = override.RepeatPenalty
	}
	if override.RepeatLastN != 0 {
		result.RepeatLastN = override.RepeatLastN
	}
	if len(override.Stop) > 0 {
		result.Stop = append([]string(nil), override.Stop...)
	}

	return result
}

// shouldStop checks if the accumulated text ends with any stop sequence.
func shouldStop(text string, stops []string) bool {
	for _, s := range stops {
		if strings.HasSuffix(text, s) {
			return true
		}
	}
	return false
}

// trimAtStop removes text from the first occurrence of any stop sequence.
func trimAtStop(text string, stops []string) string {
	earliest := len(text)
	for _, s := range stops {
		if idx := strings.Index(text, s); idx >= 0 && idx < earliest {
			earliest = idx
		}
	}
	return text[:earliest]
}

// commonPrefixLen returns the length of the longest common prefix between
// two token sequences. Used for KV cache reuse across conversation turns.
func commonPrefixLen(a, b []int32) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

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

	// Sampler reuse: avoid per-request allocation of CGo sampler chains.
	// The sampler is rebuilt only when sampling parameters change; between
	// identical requests we just Reset() the penalty history.
	sampler     *SamplerChain
	samplerOpts SamplerOptions

	// Speculative decoding: a smaller "draft" model generates candidate tokens
	// that the target model verifies in batch. When verification passes, we
	// accept multiple tokens per decode step, yielding 1.5-2.5x throughput.
	draftModel   *Model
	draftCtx     *Context
	speculativeN int // number of draft tokens to generate before verification

	// Batch size limits for safety checks
	draftBatchSize  uint32
	targetBatchSize uint32
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

	// KV cache quantization: map config string to type enum.
	if nc.KVCacheType != "" {
		kvType := kvCacheTypeFromString(nc.KVCacheType)
		ctxOpts.TypeK = kvType
		ctxOpts.TypeV = kvType
		if kvType > 0 {
			log.Printf("native: KV cache quantization set to %s (saves ~%.0f%% KV memory)",
				nc.KVCacheType, float64(kvType)*37.5) // q8_0≈37%, q4_0≈75%
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

	// Speculative decoding: load a smaller draft model if configured.
	// The draft model generates candidate tokens cheaply; the target model
	// verifies them in batch. Net effect: 1.5-2.5x faster generation on CPU.
	var draftModel *Model
	var draftCtx *Context
	specN := nc.SpeculativeN
	if specN <= 0 {
		specN = 5 // default: 5 draft tokens before verification
	}

	// Track batch sizes for safety checks
	targetBatchSize := ctxOpts.NBatch
	if targetBatchSize <= 0 {
		targetBatchSize = 512
	}
	var draftBatchSize uint32 = 512

	if nc.DraftModelPath != "" {
		draftOpts := DefaultModelOptions()
		draftOpts.NGPULayers = int32(nc.GPULayers)
		if nc.Mmap != nil {
			draftOpts.UseMmap = *nc.Mmap
		}
		if nc.Mlock != nil {
			draftOpts.UseMlock = *nc.Mlock
		}

		var draftErr error
		draftModel, draftErr = LoadModel(nc.DraftModelPath, draftOpts)
		if draftErr != nil {
			if vision != nil {
				vision.Close()
			}
			llCtx.Close()
			model.Close()
			return nil, fmt.Errorf("native: draft model: %w", draftErr)
		}

		// Create a context for the draft model with same context size but
		// reasonable batch size to handle prompts efficiently.
		draftCtxOpts := DefaultContextOptions()
		draftCtxOpts.NCtx = ctxOpts.NCtx
		draftCtxOpts.NBatch = 512 // increased from 1 to handle larger prompts while maintaining efficiency
		draftCtxOpts.NThreads = ctxOpts.NThreads
		draftCtxOpts.NThreadsBatch = ctxOpts.NThreadsBatch
		// Draft model doesn't need flash attention or KV quantization —
		// it's tiny enough that the overhead isn't worth it.
		draftCtxOpts.FlashAttn = -1

		draftCtx, draftErr = NewContext(draftModel, draftCtxOpts)
		if draftErr != nil {
			draftModel.Close()
			if vision != nil {
				vision.Close()
			}
			llCtx.Close()
			model.Close()
			return nil, fmt.Errorf("native: draft context: %w", draftErr)
		}

		draftInfo := draftModel.Info()
		log.Printf("native: speculative decoding enabled — draft=%s (%d params), N=%d",
			draftInfo.Description, draftInfo.NParams, specN)
	}

	return &Adapter{
		model:           model,
		ctx:             llCtx,
		vision:          vision,
		cfg:             cfg,
		draftModel:      draftModel,
		draftCtx:        draftCtx,
		speculativeN:    specN,
		draftBatchSize:  draftBatchSize,
		targetBatchSize: targetBatchSize,
	}, nil
}

// Name returns the adapter identifier.
func (a *Adapter) Name() string { return "native" }

// Generate performs a blocking completion, returning the full response.
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	opts := mergeNativeOptions(a.cfg.Defaults, req.Options)

	// Get or reuse sampler chain for this request's parameters.
	sampler := a.getOrBuildSampler(SamplerOptions{
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
		if prefixLen > 0 && prefixLen < len(tokens) {
			// Partial cache hit: keep the shared prefix, discard diverging suffix
			// and any generated tokens beyond the prompt.
			a.ctx.TruncateKV(int32(prefixLen))
			// Also truncate draft model to maintain sync during speculative decoding
			if a.draftCtx != nil {
				a.draftCtx.TruncateKV(int32(prefixLen))
			}
		} else if prefixLen == len(tokens) {
			// Full cache hit: identical prompt. The KV cache contains the prompt
			// tokens plus generated tokens from the previous call. We cannot
			// safely truncate and re-use because the generated tokens have
			// altered the KV state at positions beyond the prompt. Clear
			// everything and re-evaluate from scratch.
			a.ctx.ClearKV()
			// Also clear draft model to maintain sync
			if a.draftCtx != nil {
				a.draftCtx.ClearKV()
			}
			prefixLen = 0
		} else {
			a.ctx.ClearKV()
			// Also clear draft model to maintain sync
			if a.draftCtx != nil {
				a.draftCtx.ClearKV()
			}
		}

		// Evaluate only the new tokens (from prefixLen onwards).
		newTokens := tokens[prefixLen:]
		if len(newTokens) > 0 {
			// Check batch size limits and process in chunks if needed
			if err := a.evalTokensInChunks(newTokens); err != nil {
				// Generation failed before starting; invalidate prompt cache.
				a.lastPromptTokens = nil
				return runtime.Response{}, fmt.Errorf("native: eval prompt: %w", err)
			}
		}

		// Store tokens for prompt caching on next call.
		a.lastPromptTokens = make([]int32, len(tokens))
		copy(a.lastPromptTokens, tokens)

		// Sync draft model with prompt tokens for speculative decoding.
		if a.draftModel != nil {
			if err := a.syncDraftPrompt(tokens); err != nil {
				log.Printf("native: draft model sync failed, falling back to standard generation: %v", err)
			}
		}
	}

	// Resolve max tokens for generation.
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	// Context window management: shift if near-full, then cap max tokens.
	a.maybeShiftContext()
	ctxSize := a.contextSize()
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
	ring := newStopRing(opts.Stop)

	// Speculative decoding: if a draft model is loaded and not in vision mode,
	// use the speculative path for faster generation.
	useSpeculative := a.draftModel != nil && a.draftCtx != nil && len(req.Image) == 0
	var specDrafted, specAccepted int // accumulators for speculative stats

	for i := 0; i < maxTokens; {
		// Check for cancellation.
		select {
		case <-ctx.Done():
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

		if useSpeculative {
			// --- Speculative decoding path ---
			specResult, err := a.speculativeGenerate(sampler)
			if err != nil {
				// Silent fallback: if speculative fails, try standard generation once
				log.Printf("native: speculative generation failed, falling back: %v", err)
				useSpeculative = false
				continue
			}

			// Accumulate speculative decoding stats.
			specDrafted += specResult.drafted
			specAccepted += specResult.accepted

			// Record TTFT on the first generated token.
			if tokensGenerated == 0 && len(specResult.tokens) > 0 {
				ttft = time.Since(startTime)
			}

			// Process accepted tokens.
			hitStop := false
			for j, piece := range specResult.pieces {
				if a.model.TokenIsEOG(specResult.tokens[j]) {
					finishReason = "stop"
					hitStop = true
					break
				}
				result.WriteString(piece)
				tokensGenerated++
				i++

				if ring != nil {
					ring.write(piece)
					if ring.check() {
						trimmed := trimAtStop(result.String(), opts.Stop)
						result.Reset()
						result.WriteString(trimmed)
						finishReason = "stop"
						hitStop = true
						break
					}
				}
			}

			if hitStop || specResult.hitEOG {
				if finishReason != "stop" {
					finishReason = "stop"
				}
				break
			}

			// Mid-generation context shift.
			a.maybeShiftContext()

		} else {
			// --- Standard single-token path ---
			token, err := a.ctx.SampleToken(sampler)
			if err != nil {
				a.lastPromptTokens = nil
				return runtime.Response{}, fmt.Errorf("native: sample: %w", err)
			}

			if tokensGenerated == 0 {
				ttft = time.Since(startTime)
			}

			if a.model.TokenIsEOG(token) {
				finishReason = "stop"
				break
			}

			piece := a.model.TokenToPiece(token)
			result.WriteString(piece)
			tokensGenerated++
			i++

			if ring != nil {
				ring.write(piece)
				if ring.check() {
					trimmed := trimAtStop(result.String(), opts.Stop)
					result.Reset()
					result.WriteString(trimmed)
					finishReason = "stop"
					break
				}
			}

			if err := a.ctx.EvalToken(token); err != nil {
				a.lastPromptTokens = nil
				return runtime.Response{}, fmt.Errorf("native: eval token: %w", err)
			}

			a.maybeShiftContext()
		}
	}

	// Generation succeeded — prompt cache was already updated in the
	// text-only path above. For vision path, prompt cache stays nil
	// (vision requests don't benefit from text prompt caching).

	perf := a.ctx.Perf()
	duration := time.Since(startTime)

	// Compute throughput. Prefer llama.cpp perf counters, but fall back to
	// Go-side wall-clock measurement if the C counters return zero timing
	// (which happens in some newer llama.cpp versions).
	var promptTPS, genTPS float64
	if perf.PromptMs > 0 && perf.PromptCount > 0 {
		promptTPS = float64(perf.PromptCount) / (perf.PromptMs / 1000.0)
	} else if promptTokenCount > 0 && ttft > 0 {
		// Fallback: TTFT includes prompt eval; use it to estimate prompt TPS.
		promptTPS = float64(promptTokenCount) / ttft.Seconds()
	}
	if perf.EvalMs > 0 && perf.EvalCount > 0 {
		genTPS = float64(perf.EvalCount) / (perf.EvalMs / 1000.0)
	} else if tokensGenerated > 0 && ttft > 0 && duration > ttft {
		// Fallback: generation time = total duration minus TTFT.
		genDuration := duration - ttft
		if genDuration > 0 {
			genTPS = float64(tokensGenerated) / genDuration.Seconds()
		}
	}

	// Compute speculative acceptance rate.
	var specRate float64
	if specDrafted > 0 {
		specRate = float64(specAccepted) / float64(specDrafted) * 100.0
	}

	return runtime.Response{
		Text: result.String(),
		Stats: runtime.Stats{
			TokensEvaluated:           promptTokenCount,
			TokensGenerated:           tokensGenerated,
			TokensCached:              prefixLen,
			Duration:                  duration,
			TTFT:                      ttft,
			PromptTPS:                 promptTPS,
			GenerationTPS:             genTPS,
			SpeculativeAttempted:      specDrafted,
			SpeculativeAccepted:       specAccepted,
			SpeculativeAcceptanceRate: specRate,
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

	// Get or reuse sampler chain.
	sampler := a.getOrBuildSampler(SamplerOptions{
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
		if prefixLen > 0 && prefixLen < len(tokens) {
			// Partial cache hit: keep the shared prefix, discard diverging suffix
			// and any generated tokens beyond the prompt.
			a.ctx.TruncateKV(int32(prefixLen))
			// Also truncate draft model to maintain sync during speculative decoding
			if a.draftCtx != nil {
				a.draftCtx.TruncateKV(int32(prefixLen))
			}
		} else if prefixLen == len(tokens) {
			// Full cache hit: identical prompt. The KV cache contains generated
			// tokens from the previous call beyond the prompt. Clear and re-eval.
			a.ctx.ClearKV()
			// Also clear draft model to maintain sync
			if a.draftCtx != nil {
				a.draftCtx.ClearKV()
			}
			prefixLen = 0
		} else {
			a.ctx.ClearKV()
			// Also clear draft model to maintain sync
			if a.draftCtx != nil {
				a.draftCtx.ClearKV()
			}
		}

		newTokens := tokens[prefixLen:]
		if len(newTokens) > 0 {
			// Check batch size limits and process in chunks if needed
			if err := a.evalTokensInChunks(newTokens); err != nil {
				a.lastPromptTokens = nil
				return fmt.Errorf("native: eval prompt: %w", err)
			}
		}

		// Store tokens for prompt caching on next call.
		a.lastPromptTokens = make([]int32, len(tokens))
		copy(a.lastPromptTokens, tokens)

		// Sync draft model with prompt tokens for speculative decoding.
		if a.draftModel != nil {
			if err := a.syncDraftPrompt(tokens); err != nil {
				log.Printf("native: draft model sync failed, falling back to standard generation: %v", err)
			}
		}
	}

	// Resolve max tokens for generation.
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}

	// Context window management: shift if near-full, then cap max tokens.
	a.maybeShiftContext()
	ctxSize := a.contextSize()
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
	streamRing := newStopRing(opts.Stop)

	// Token chunking: buffer tokens and emit in word-sized chunks for
	// smoother streaming. chunkSize=0 or 1 disables chunking.
	chunkSize := a.cfg.Native.StreamChunkSize
	if chunkSize <= 0 {
		chunkSize = 1 // default: emit every token (no chunking)
	}
	var chunkBuf strings.Builder
	chunkCount := 0

	// Speculative decoding: if a draft model is loaded and not in vision mode,
	// use the speculative path for faster generation.
	useSpeculative := a.draftModel != nil && a.draftCtx != nil && len(req.Image) == 0
	var specDrafted, specAccepted int // accumulators for speculative stats

	for i := 0; i < maxTokens; {
		// Check cancellation.
		select {
		case <-ctx.Done():
			a.lastPromptTokens = nil
			_ = cb(runtime.StreamEvent{Final: true, Err: ctx.Err()})
			return ctx.Err()
		default:
		}

		if useSpeculative {
			// --- Speculative decoding path ---
			specResult, err := a.speculativeGenerate(sampler)
			if err != nil {
				// Silent fallback: if speculative fails, try standard generation once
				log.Printf("native: speculative streaming failed, falling back: %v", err)
				useSpeculative = false
				continue
			}

			// Accumulate speculative decoding stats.
			specDrafted += specResult.drafted
			specAccepted += specResult.accepted

			// Record TTFT on the first generated token.
			if idx == 0 && len(specResult.tokens) > 0 {
				ttft = time.Since(startTime)
			}

			// Process accepted tokens through the chunk buffer.
			hitStop := false
			for j, piece := range specResult.pieces {
				if a.model.TokenIsEOG(specResult.tokens[j]) {
					// Flush any buffered chunk before finishing.
					if chunkBuf.Len() > 0 {
						if err := cb(runtime.StreamEvent{Token: chunkBuf.String(), Index: idx}); err != nil {
							a.lastPromptTokens = nil
							return err
						}
						idx++
						chunkBuf.Reset()
						chunkCount = 0
					}
					hitStop = true
					break
				}

				accumulated.WriteString(piece)

				// Check stop sequences using ring buffer.
				if streamRing != nil {
					streamRing.write(piece)
					if streamRing.check() {
						// Stop sequence detected — don't flush chunk.
						hitStop = true
						break
					}
				}

				// Buffer the token for chunked emission.
				chunkBuf.WriteString(piece)
				chunkCount++
				i++

				// Flush the chunk when: buffer is full, or piece ends with
				// whitespace (natural word boundary for readable streaming).
				flushChunk := chunkCount >= chunkSize
				if !flushChunk && len(piece) > 0 {
					lastByte := piece[len(piece)-1]
					if lastByte == ' ' || lastByte == '\n' || lastByte == '\t' {
						flushChunk = true
					}
				}

				if flushChunk && chunkBuf.Len() > 0 {
					if err := cb(runtime.StreamEvent{Token: chunkBuf.String(), Index: idx}); err != nil {
						a.lastPromptTokens = nil
						return err
					}
					idx++
					chunkBuf.Reset()
					chunkCount = 0
				}
			}

			if hitStop || specResult.hitEOG {
				break
			}

			// Mid-generation context shift.
			a.maybeShiftContext()

		} else {
			// --- Standard single-token path ---

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
				// Flush any buffered chunk before finishing.
				if chunkBuf.Len() > 0 {
					if err := cb(runtime.StreamEvent{Token: chunkBuf.String(), Index: idx}); err != nil {
						a.lastPromptTokens = nil
						return err
					}
					idx++
				}
				break
			}

			piece := a.model.TokenToPiece(token)
			accumulated.WriteString(piece)

			// Check stop sequences using ring buffer.
			if streamRing != nil {
				streamRing.write(piece)
				if streamRing.check() {
					// Don't flush the chunk — it may contain the stop sequence.
					break
				}
			}

			// Buffer the token for chunked emission.
			chunkBuf.WriteString(piece)
			chunkCount++
			i++

			// Flush the chunk when: buffer is full, or piece ends with whitespace
			// (natural word boundary for readable streaming).
			flushChunk := chunkCount >= chunkSize
			if !flushChunk && len(piece) > 0 {
				lastByte := piece[len(piece)-1]
				if lastByte == ' ' || lastByte == '\n' || lastByte == '\t' {
					flushChunk = true
				}
			}

			if flushChunk && chunkBuf.Len() > 0 {
				if err := cb(runtime.StreamEvent{Token: chunkBuf.String(), Index: idx}); err != nil {
					a.lastPromptTokens = nil
					return err
				}
				idx++
				chunkBuf.Reset()
				chunkCount = 0
			}

			// Advance KV cache.
			if err := a.ctx.EvalToken(token); err != nil {
				a.lastPromptTokens = nil
				return fmt.Errorf("native: eval token: %w", err)
			}

			// Mid-generation context shift.
			a.maybeShiftContext()
		}
	}

	// Streaming generation succeeded — prompt cache was already updated
	// in the text-only path above. For vision path, it stays nil.

	// Collect perf counters and build final stats.
	perf := a.ctx.Perf()
	duration := time.Since(startTime)

	// Compute throughput. Prefer llama.cpp perf counters, but fall back to
	// Go-side wall-clock measurement if the C counters return zero timing.
	var promptTPS, genTPS float64
	if perf.PromptMs > 0 && perf.PromptCount > 0 {
		promptTPS = float64(perf.PromptCount) / (perf.PromptMs / 1000.0)
	} else if promptTokenCount > 0 && ttft > 0 {
		promptTPS = float64(promptTokenCount) / ttft.Seconds()
	}
	if perf.EvalMs > 0 && perf.EvalCount > 0 {
		genTPS = float64(perf.EvalCount) / (perf.EvalMs / 1000.0)
	} else if idx > 0 && ttft > 0 && duration > ttft {
		genDuration := duration - ttft
		if genDuration > 0 {
			genTPS = float64(idx) / genDuration.Seconds()
		}
	}

	// Compute speculative acceptance rate.
	var specRate float64
	if specDrafted > 0 {
		specRate = float64(specAccepted) / float64(specDrafted) * 100.0
	}

	finalStats := &runtime.Stats{
		TokensEvaluated:           promptTokenCount,
		TokensGenerated:           idx,
		TokensCached:              prefixLen,
		Duration:                  duration,
		TTFT:                      ttft,
		PromptTPS:                 promptTPS,
		GenerationTPS:             genTPS,
		SpeculativeAttempted:      specDrafted,
		SpeculativeAccepted:       specAccepted,
		SpeculativeAcceptanceRate: specRate,
	}

	// Signal completion with stats.
	return cb(runtime.StreamEvent{Final: true, Stats: finalStats})
}

// Close frees all native resources.
func (a *Adapter) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.sampler != nil {
		a.sampler.Close()
		a.sampler = nil
	}
	if a.vision != nil {
		a.vision.Close()
		a.vision = nil
	}
	// Free draft model resources (speculative decoding).
	if a.draftCtx != nil {
		a.draftCtx.Close()
		a.draftCtx = nil
	}
	if a.draftModel != nil {
		a.draftModel.Close()
		a.draftModel = nil
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

// contextShiftEnabled returns whether automatic context window sliding is on.
// Defaults to true unless explicitly disabled.
func (a *Adapter) contextShiftEnabled() bool {
	if a.cfg.Native.ContextShift != nil {
		return *a.cfg.Native.ContextShift
	}
	return true // default: enabled
}

// contextSize returns the configured context window size.
func (a *Adapter) contextSize() int {
	if a.cfg.Native.ContextSize > 0 {
		return a.cfg.Native.ContextSize
	}
	return 2048
}

// batchSize returns the configured batch size for the target model.
func (a *Adapter) batchSize() int {
	if a.cfg.Native.BatchSize > 0 {
		return a.cfg.Native.BatchSize
	}
	return 512
}

// maybeShiftContext checks if the KV cache is at or above the shift threshold
// (75% of context size). If context shifting is enabled, it discards the
// oldest 25% of the KV cache and shifts remaining positions down, freeing
// space for continued generation. Returns the number of tokens discarded.
//
// After shifting, the prompt cache is invalidated since positions have changed.
// When speculative decoding is enabled, both target and draft models are shifted
// to maintain synchronization.
func (a *Adapter) maybeShiftContext() int {
	if !a.contextShiftEnabled() {
		return 0
	}
	ctxSize := a.contextSize()
	// Lowered threshold from 90% to 75% to provide more headroom for prompts
	// and prevent context window overflow during multi-turn conversations
	threshold := int(float64(ctxSize) * 0.75)
	currentPos := int(a.ctx.Pos())
	if currentPos < threshold {
		return 0
	}
	// Discard oldest 25% of the context.
	nDiscard := int32(float64(ctxSize) * 0.25)
	if nDiscard <= 0 {
		nDiscard = 1
	}
	if nDiscard >= a.ctx.Pos() {
		// Shouldn't happen, but guard against it.
		return 0
	}

	// Shift target model
	a.ctx.ShiftKV(nDiscard)
	a.lastPromptTokens = nil // positions shifted — prompt cache invalid

	// Also shift draft model to maintain synchronization during speculative decoding
	if a.draftCtx != nil && a.draftCtx.Pos() >= nDiscard {
		a.draftCtx.ShiftKV(nDiscard)
	}

	log.Printf("native: context shift — discarded %d oldest tokens, pos %d → %d",
		nDiscard, currentPos, a.ctx.Pos())
	return int(nDiscard)
}

// getOrBuildSampler returns a sampler chain for the given options, reusing
// the cached one if the parameters haven't changed. This avoids CGo
// allocation overhead on every request — significant on edge devices where
// requests often share the same temperature/top-k/top-p settings.
func (a *Adapter) getOrBuildSampler(opts SamplerOptions) *SamplerChain {
	if a.sampler != nil && a.samplerOpts == opts {
		// Same parameters — just reset penalty history for the new request.
		a.sampler.Reset()
		return a.sampler
	}
	// Parameters changed — rebuild the chain.
	if a.sampler != nil {
		a.sampler.Close()
	}
	a.sampler = NewSamplerChain(opts)
	a.samplerOpts = opts
	return a.sampler
}

// kvCacheTypeFromString maps a KV cache type string to the integer enum
// used by the C binding: 0=f16 (default), 1=q8_0, 2=q4_0.
func kvCacheTypeFromString(s string) int32 {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "q8_0", "q8":
		return 1
	case "q4_0", "q4":
		return 2
	default:
		return 0 // f16
	}
}

// ---------------------------------------------------------------------------
// Speculative decoding
// ---------------------------------------------------------------------------
//
// evalTokensInChunks evaluates tokens in batches that fit within the model's
// batch size limit. This prevents "n_tokens_all <= cparams.n_batch" assertion
// failures when processing large prompts.
func (a *Adapter) evalTokensInChunks(tokens []int32) error {
	batchSize := int32(a.batchSize())
	if batchSize <= 0 {
		batchSize = 512
	}

	// Process tokens in chunks to respect batch size limits
	for len(tokens) > 0 {
		chunk := tokens
		if int32(len(chunk)) > batchSize {
			chunk = tokens[:batchSize]
		}

		if err := a.ctx.Eval(chunk); err != nil {
			return fmt.Errorf("eval chunk of %d tokens: %w", len(chunk), err)
		}

		tokens = tokens[len(chunk):]
	}

	return nil
}

// speculativeResult holds the output of one speculative decoding step.
// It contains the accepted tokens (possibly more than one) and metadata
// about the verification outcome.
type speculativeResult struct {
	// tokens contains the accepted token IDs (1 to N+1 tokens).
	tokens []int32
	// pieces contains the text representation of each accepted token.
	pieces []string
	// hitEOG is true if any accepted token was an end-of-generation token.
	hitEOG bool
	// drafted is the number of tokens the draft model proposed this round.
	drafted int
	// accepted is the number of draft tokens that matched the target model.
	// This excludes the bonus token and divergence-replacement tokens.
	accepted int
}

// speculativeGenerate performs one round of speculative decoding:
//
//  1. Draft phase: the small draft model greedily generates N candidate tokens.
//  2. Verify phase: the target model evaluates all N draft tokens in a single
//     batch with logits computed for every position.
//  3. Accept phase: compare each draft token against the target model's argmax.
//     Accept all matching tokens; at the first mismatch, use the target's token.
//
// Returns the accepted tokens (1 to N+1 per call). The caller should append
// these to the output and advance both KV caches accordingly.
//
// Both the target and draft model KV caches must be in sync at the position
// where the last token was evaluated (i.e., the prompt has already been
// processed on both models).
func (a *Adapter) speculativeGenerate(sampler *SamplerChain) (speculativeResult, error) {
	// Calculate how many draft tokens we can safely generate
	// We need: currentPos + n <= batchSize
	currentPos := int(a.ctx.Pos())
	batchSize := a.batchSize()

	// Determine effective speculative N based on available batch capacity
	n := a.speculativeN
	if currentPos+n > batchSize {
		// Reduce speculative tokens to fit within batch limit
		n = batchSize - currentPos - 1 // leave 1 token margin for safety
		if n <= 0 {
			// No room for speculative decoding, fall back to single token
			token, err := a.ctx.SampleToken(sampler)
			if err != nil {
				return speculativeResult{}, fmt.Errorf("native: sample fallback: %w", err)
			}
			piece := a.model.TokenToPiece(token)
			return speculativeResult{
				tokens:   []int32{token},
				pieces:   []string{piece},
				hitEOG:   a.model.TokenIsEOG(token),
				drafted:  0,
				accepted: 0,
			}, nil
		}
	}

	vocabSize := a.model.VocabSize()

	// --- 1. Draft phase: generate N candidate tokens with the draft model ---
	draftTokens := make([]int32, 0, n)
	draftSampler := NewSamplerChain(SamplerOptions{
		// Use greedy for drafting — fast and deterministic.
		Temperature: 0.0,
		Seed:        0xFFFFFFFF,
	})
	defer draftSampler.Close()

	for i := 0; i < n; i++ {
		token, err := a.draftCtx.SampleToken(draftSampler)
		if err != nil {
			return speculativeResult{}, fmt.Errorf("native: draft sample: %w", err)
		}

		draftTokens = append(draftTokens, token)

		// Check if the draft model hit EOG. If so, we still need to verify
		// what we have so far with the target model.
		if a.draftModel.TokenIsEOG(token) {
			break
		}

		// Advance draft model KV cache for next token.
		if i < n-1 { // don't eval after the last draft token
			if err := a.draftCtx.EvalToken(token); err != nil {
				return speculativeResult{}, fmt.Errorf("native: draft eval: %w", err)
			}
		}
	}

	if len(draftTokens) == 0 {
		// Shouldn't happen, but fall back to single-token generation.
		token, err := a.ctx.SampleToken(sampler)
		if err != nil {
			return speculativeResult{}, fmt.Errorf("native: sample fallback: %w", err)
		}
		piece := a.model.TokenToPiece(token)
		return speculativeResult{
			tokens: []int32{token},
			pieces: []string{piece},
			hitEOG: a.model.TokenIsEOG(token),
		}, nil
	}

	// --- 2. Verify phase: evaluate draft tokens on target model in one batch ---
	// Use EvalLogitsAll to get logits at every position.
	targetPos := a.ctx.Pos() // save position before eval
	if err := a.ctx.EvalLogitsAll(draftTokens); err != nil {
		// Verification failed; rollback target model to pre-verify state.
		a.ctx.TruncateKV(targetPos)
		return speculativeResult{}, fmt.Errorf("native: verify batch: %w", err)
	}

	// --- 3. Accept phase: compare draft vs target at each position ---
	accepted := make([]int32, 0, len(draftTokens)+1)
	acceptedPieces := make([]string, 0, len(draftTokens)+1)
	hitEOG := false
	draftMatches := 0 // count of draft tokens that matched target

	for i, draftToken := range draftTokens {
		// Get target model's logits at position i in the batch.
		logits := a.ctx.GetLogitsAt(int32(i), vocabSize)
		if logits == nil {
			// Can't verify this position — accept what we have.
			break
		}

		targetToken := argmaxToken(logits)

		if draftToken == targetToken {
			// Draft agrees with target — accept this token.
			accepted = append(accepted, draftToken)
			acceptedPieces = append(acceptedPieces, a.model.TokenToPiece(draftToken))
			draftMatches++
			if a.model.TokenIsEOG(draftToken) {
				hitEOG = true
				break
			}
		} else {
			// Divergence — reject draft token, use target's choice instead.
			accepted = append(accepted, targetToken)
			acceptedPieces = append(acceptedPieces, a.model.TokenToPiece(targetToken))
			if a.model.TokenIsEOG(targetToken) {
				hitEOG = true
			}
			break
		}
	}

	// If all N draft tokens matched, we can also accept one more token from
	// the target's logits at the last position (bonus token).
	if !hitEOG && len(accepted) == len(draftTokens) && !a.model.TokenIsEOG(draftTokens[len(draftTokens)-1]) {
		// The target model evaluated the draft tokens, so the logits at
		// position N-1 actually predict the token AFTER the last draft token.
		// We already read those logits above when we verified the last draft
		// token. To get a true bonus token, we'd need to sample from the
		// target model's last output. Use the provided sampler for this.
		bonusToken, err := a.ctx.SampleToken(sampler)
		if err == nil {
			accepted = append(accepted, bonusToken)
			acceptedPieces = append(acceptedPieces, a.model.TokenToPiece(bonusToken))
			if a.model.TokenIsEOG(bonusToken) {
				hitEOG = true
			}
		}
	}

	// --- 4. KV cache cleanup ---
	// Truncate target model's KV cache to only include accepted tokens.
	// The target model evaluated all N draft tokens, but we may have
	// rejected some. Truncate to: original position + accepted count.
	acceptedCount := int32(len(accepted))
	finalTargetPos := targetPos + acceptedCount
	if finalTargetPos < a.ctx.Pos() {
		a.ctx.TruncateKV(finalTargetPos)
	}

	// Sync draft model's KV cache with what was accepted.
	// The draft model may have evaluated more tokens than were accepted.
	// Truncate it back to the same position as the target.
	draftFinalPos := targetPos + acceptedCount
	if draftFinalPos < a.draftCtx.Pos() {
		a.draftCtx.TruncateKV(draftFinalPos)
	} else if draftFinalPos > a.draftCtx.Pos() {
		// Draft model is behind target (e.g., bonus token was accepted).
		// Eval the extra accepted tokens on the draft model to keep in sync.
		draftBehind := a.draftCtx.Pos()
		for j := draftBehind - targetPos; j < acceptedCount; j++ {
			if err := a.draftCtx.EvalToken(accepted[j]); err != nil {
				// Non-fatal: draft model desync means next round will
				// produce worse drafts, but generation still works.
				log.Printf("native: draft model sync eval failed: %v", err)
				break
			}
		}
	}

	return speculativeResult{
		tokens:   accepted,
		pieces:   acceptedPieces,
		hitEOG:   hitEOG,
		drafted:  len(draftTokens),
		accepted: draftMatches,
	}, nil
}

// syncDraftPrompt evaluates the prompt tokens on the draft model so its
// KV cache is in sync with the target model. This must be called after
// the target model processes the prompt and before speculative generation.
//
// This function now handles prompts larger than the draft model's batch size
// by processing them in chunks.
func (a *Adapter) syncDraftPrompt(tokens []int32) error {
	if a.draftModel == nil || a.draftCtx == nil {
		return nil
	}

	// Clear draft model's KV cache first
	a.draftCtx.ClearKV()

	if len(tokens) == 0 {
		return nil
	}

	// Process tokens in chunks to respect draft model's batch size
	draftBatchSize := int32(a.draftBatchSize)
	if draftBatchSize <= 0 {
		draftBatchSize = 512
	}

	for len(tokens) > 0 {
		chunk := tokens
		if int32(len(chunk)) > draftBatchSize {
			chunk = tokens[:draftBatchSize]
		}

		if err := a.draftCtx.Eval(chunk); err != nil {
			return fmt.Errorf("native: draft prompt eval chunk of %d tokens: %w", len(chunk), err)
		}

		tokens = tokens[len(chunk):]
	}

	return nil
}

// argmaxToken returns the token ID with the highest logit value.
func argmaxToken(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}
	best := int32(0)
	bestVal := logits[0]
	for i := int32(1); i < int32(len(logits)); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			best = i
		}
	}
	return best
}

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

// ---------------------------------------------------------------------------
// stopRing: optimized stop sequence detector using a ring buffer.
// Instead of calling strings.HasSuffix on the entire accumulated text for
// every token, we maintain a small ring buffer of the last N characters
// (where N = max stop sequence length). This makes stop detection O(1) in
// accumulated text length — significant when generating hundreds of tokens.
// ---------------------------------------------------------------------------

type stopRing struct {
	buf    []byte // circular buffer
	size   int    // capacity = max stop sequence length
	pos    int    // write position
	filled int    // how many bytes written total (capped at size for indexing)
	stops  []string
}

// newStopRing creates a stop detector from the given stop sequences.
// Returns nil if there are no stop sequences (caller should skip checks).
func newStopRing(stops []string) *stopRing {
	if len(stops) == 0 {
		return nil
	}
	maxLen := 0
	for _, s := range stops {
		if len(s) > maxLen {
			maxLen = len(s)
		}
	}
	if maxLen == 0 {
		return nil
	}
	return &stopRing{
		buf:   make([]byte, maxLen),
		size:  maxLen,
		stops: stops,
	}
}

// write appends text to the ring buffer.
func (r *stopRing) write(s string) {
	for i := 0; i < len(s); i++ {
		r.buf[r.pos%r.size] = s[i]
		r.pos++
	}
	r.filled += len(s)
	if r.filled > r.size {
		r.filled = r.size
	}
}

// check returns true if the ring buffer ends with any stop sequence.
func (r *stopRing) check() bool {
	avail := r.filled
	if avail == 0 {
		return false
	}
	for _, stop := range r.stops {
		slen := len(stop)
		if slen == 0 || slen > avail {
			continue
		}
		match := true
		for j := 0; j < slen; j++ {
			// Read from ring buffer: the tail byte at offset (pos - slen + j)
			idx := ((r.pos-slen+j)%r.size + r.size) % r.size
			if r.buf[idx] != stop[j] {
				match = false
				break
			}
		}
		if match {
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

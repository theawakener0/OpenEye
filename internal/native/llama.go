//go:build native

// Package native provides direct CGo bindings to llama.cpp for in-process
// SLM inference on edge devices. It wraps a thin C binding layer (binding.h/c)
// that calls the llama.h API, exposing Go-safe types.
//
// Build with: go build -tags native
// Requires: pre-built libllama.a and libggml*.a from the vendored llama.cpp.
package native

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/include -I${SRCDIR}/../../llama.cpp/ggml/include -I${SRCDIR}/../../llama.cpp/tools/mtmd -O2
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build/src -L${SRCDIR}/../../llama.cpp/build/ggml/src -L${SRCDIR}/../../llama.cpp/build/tools/mtmd -lmtmd -lllama -lggml -lggml-cpu -lggml-base -lm -lstdc++ -lpthread -lgomp
#include "binding.h"
#include "binding_vision.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"sync"
	"unsafe"

	"OpenEye/internal/logging"
)

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------

var logOnce sync.Once

// BackendInit initializes the llama.cpp backend. Call once at startup.
func BackendInit() {
	C.oe_backend_init()

	// Redirect C-level llama.cpp logs on first init. When file logging
	// is active (chat/cli/tui modes), send C logs to the same file.
	// Otherwise suppress them so they don't pollute stderr.
	logOnce.Do(func() {
		if logging.IsFileLogging() {
			if p := logging.GetLogFilePath(); p != "" {
				LogToFile(p)
				VisionLogToFile(p)
			}
		} else {
			LogDisable()
			VisionLogDisable()
		}
	})
}

// BackendFree releases backend resources. Call once at shutdown.
func BackendFree() {
	C.oe_backend_free()
}

// LogToFile redirects all llama.cpp C-level log output to the given file
// (append mode). This prevents C-level messages from appearing on stderr.
func LogToFile(path string) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	C.oe_log_to_file(cpath)
}

// LogDisable suppresses all llama.cpp C-level log output.
func LogDisable() {
	C.oe_log_disable()
}

// ---------------------------------------------------------------------------
// Model — low-level C wrappers
// ---------------------------------------------------------------------------

// cModelLoad loads a GGUF model and returns the opaque C handle.
func cModelLoad(path string, nGPULayers int32, useMmap, useMlock bool) C.oe_model_t {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	return C.oe_model_load(cpath, C.int32_t(nGPULayers),
		C.bool(useMmap), C.bool(useMlock))
}

// cModelFree frees a loaded model.
func cModelFree(m C.oe_model_t) {
	C.oe_model_free(m)
}

// ModelInfo holds model metadata returned from the C layer.
type ModelInfo struct {
	NEmbedding   int32
	NCtxTrain    int32
	NLayer       int32
	NHead        int32
	ModelSize    uint64
	NParams      uint64
	Description  string
	ChatTemplate string
	HasEncoder   bool
}

// cModelGetInfo retrieves model metadata.
func cModelGetInfo(m C.oe_model_t) ModelInfo {
	ci := C.oe_model_get_info(m)
	return ModelInfo{
		NEmbedding:   int32(ci.n_embd),
		NCtxTrain:    int32(ci.n_ctx_train),
		NLayer:       int32(ci.n_layer),
		NHead:        int32(ci.n_head),
		ModelSize:    uint64(ci.model_size),
		NParams:      uint64(ci.n_params),
		Description:  C.GoString(&ci.desc[0]),
		ChatTemplate: C.GoString(&ci.chat_template[0]),
		HasEncoder:   bool(ci.has_encoder),
	}
}

// ---------------------------------------------------------------------------
// Context — low-level C wrappers
// ---------------------------------------------------------------------------

// cContextNew creates an inference context.
func cContextNew(m C.oe_model_t, nCtx, nBatch, nUbatch uint32,
	nThreads, nThreadsBatch int32, embeddings bool, flashAttn int32,
	typeK, typeV int32) C.oe_context_t {
	return C.oe_context_new(m, C.uint32_t(nCtx), C.uint32_t(nBatch), C.uint32_t(nUbatch),
		C.int32_t(nThreads), C.int32_t(nThreadsBatch),
		C.bool(embeddings), C.int32_t(flashAttn),
		C.int32_t(typeK), C.int32_t(typeV))
}

// cContextFree frees an inference context.
func cContextFree(ctx C.oe_context_t) {
	C.oe_context_free(ctx)
}

// ---------------------------------------------------------------------------
// Tokenization — low-level C wrappers
// ---------------------------------------------------------------------------

// cTokenize tokenizes text into the provided slice.
// Returns the number of tokens, or negative if buf is too small.
func cTokenize(m C.oe_model_t, text string, tokens []int32,
	addSpecial, parseSpecial bool) int32 {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))

	var tokPtr *C.int32_t
	if len(tokens) > 0 {
		tokPtr = (*C.int32_t)(unsafe.Pointer(&tokens[0]))
	}

	n := int32(C.oe_tokenize(m, ctext, C.int32_t(len(text)),
		tokPtr, C.int32_t(len(tokens)),
		C.bool(addSpecial), C.bool(parseSpecial)))
	runtime.KeepAlive(tokens) // prevent GC of backing array while C uses it
	return n
}

// cTokenToPiece converts a token ID to its text piece.
func cTokenToPiece(m C.oe_model_t, token int32) string {
	buf := make([]byte, 128)
	n := C.oe_token_to_piece(m, C.int32_t(token),
		(*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	if n < 0 {
		// Buffer too small, retry with required size.
		buf = make([]byte, -n)
		n = C.oe_token_to_piece(m, C.int32_t(token),
			(*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	}
	if n <= 0 {
		return ""
	}
	return string(buf[:n])
}

// cTokenIsEOG checks if a token signals end-of-generation.
func cTokenIsEOG(m C.oe_model_t, token int32) bool {
	return bool(C.oe_token_is_eog(m, C.int32_t(token)))
}

// cTokenBOS returns the beginning-of-sentence token ID.
func cTokenBOS(m C.oe_model_t) int32 {
	return int32(C.oe_token_bos(m))
}

// cTokenEOS returns the end-of-sentence token ID.
func cTokenEOS(m C.oe_model_t) int32 {
	return int32(C.oe_token_eos(m))
}

// cVocabNTokens returns the total number of tokens in the model's vocabulary.
func cVocabNTokens(m C.oe_model_t) int32 {
	return int32(C.oe_vocab_n_tokens(m))
}

// ---------------------------------------------------------------------------
// Decode / Evaluate — low-level C wrappers
// ---------------------------------------------------------------------------

// cDecode evaluates a batch of tokens with automatic position tracking.
func cDecode(ctx C.oe_context_t, tokens []int32) int32 {
	if len(tokens) == 0 {
		return 0
	}
	rc := int32(C.oe_decode(ctx,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int32_t(len(tokens))))
	runtime.KeepAlive(tokens)
	return rc
}

// cDecodeBatch evaluates tokens with explicit position control.
func cDecodeBatch(ctx C.oe_context_t, tokens []int32, posStart int32) int32 {
	if len(tokens) == 0 {
		return 0
	}
	rc := int32(C.oe_decode_batch(ctx,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int32_t(len(tokens)), C.int32_t(posStart)))
	runtime.KeepAlive(tokens)
	return rc
}

// cEncode runs the encoder path (for BERT/encoder-only models).
// All tokens are marked as outputs for embedding extraction.
func cEncode(ctx C.oe_context_t, tokens []int32) int32 {
	if len(tokens) == 0 {
		return 0
	}
	rc := int32(C.oe_encode(ctx,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int32_t(len(tokens))))
	runtime.KeepAlive(tokens)
	return rc
}

// ---------------------------------------------------------------------------
// Embeddings — low-level C wrappers
// ---------------------------------------------------------------------------

// cGetEmbeddings returns the embedding vector for the given output index.
// The returned slice is a Go-owned copy; the caller may hold it indefinitely.
func cGetEmbeddings(ctx C.oe_context_t, m C.oe_model_t, idx int32) []float32 {
	ptr := C.oe_get_embeddings(ctx, C.int32_t(idx))
	if ptr == nil {
		return nil
	}
	info := cModelGetInfo(m)
	nEmbd := info.NEmbedding
	if nEmbd <= 0 {
		return nil
	}
	// Copy from C-owned memory into a Go-allocated slice so the caller
	// is not left holding a dangling pointer after the next Eval/Close.
	cSlice := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
	result := make([]float32, nEmbd)
	copy(result, cSlice)
	return result
}

// cGetEmbeddingsSeq returns pooled embeddings for a sequence.
// The returned slice is a Go-owned copy; the caller may hold it indefinitely.
func cGetEmbeddingsSeq(ctx C.oe_context_t, m C.oe_model_t, seqID int32) []float32 {
	ptr := C.oe_get_embeddings_seq(ctx, C.int32_t(seqID))
	if ptr == nil {
		return nil
	}
	info := cModelGetInfo(m)
	nEmbd := info.NEmbedding
	if nEmbd <= 0 {
		return nil
	}
	// Copy from C-owned memory into a Go-allocated slice.
	cSlice := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
	result := make([]float32, nEmbd)
	copy(result, cSlice)
	return result
}

// ---------------------------------------------------------------------------
// KV / Memory management — low-level C wrappers
// ---------------------------------------------------------------------------

// cMemoryClear clears all KV cache contents.
func cMemoryClear(ctx C.oe_context_t) {
	C.oe_memory_clear(ctx)
}

// cMemorySeqRm removes tokens in [p0, p1) for seq_id.
func cMemorySeqRm(ctx C.oe_context_t, seqID, p0, p1 int32) bool {
	return bool(C.oe_memory_seq_rm(ctx,
		C.int32_t(seqID), C.int32_t(p0), C.int32_t(p1)))
}

// cMemorySeqPosMax returns the max position for a sequence, or -1 if empty.
func cMemorySeqPosMax(ctx C.oe_context_t, seqID int32) int32 {
	return int32(C.oe_memory_seq_pos_max(ctx, C.int32_t(seqID)))
}

// cMemorySeqAdd shifts KV cache positions in [p0, p1) for seqID by delta.
// Used for context window sliding: after removing old tokens, shift the
// remaining positions down to keep them contiguous.
func cMemorySeqAdd(ctx C.oe_context_t, seqID, p0, p1, delta int32) {
	C.oe_memory_seq_add(ctx, C.int32_t(seqID),
		C.int32_t(p0), C.int32_t(p1), C.int32_t(delta))
}

// ---------------------------------------------------------------------------
// Speculative decoding — low-level C wrappers
// ---------------------------------------------------------------------------

// cDecodeBatchLogitsAll evaluates tokens with logits computed for ALL tokens.
// This is used for speculative decoding verification where the target model
// needs to check each draft token against its own distribution.
func cDecodeBatchLogitsAll(ctx C.oe_context_t, tokens []int32, posStart int32) int32 {
	if len(tokens) == 0 {
		return 0
	}
	rc := int32(C.oe_decode_batch_logits_all(ctx,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int32_t(len(tokens)), C.int32_t(posStart)))
	runtime.KeepAlive(tokens)
	return rc
}

// cGetLogits returns the logits for the token at output index idx.
// Returns nil if idx is invalid. The returned pointer is context-owned
// and valid until the next decode call.
func cGetLogits(ctx C.oe_context_t, idx int32, vocabSize int32) []float32 {
	ptr := C.oe_get_logits(ctx, C.int32_t(idx))
	if ptr == nil {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), vocabSize)
}

// ---------------------------------------------------------------------------
// Sampler — low-level C wrappers
// ---------------------------------------------------------------------------

// cSamplerChainNew creates a new sampler chain.
func cSamplerChainNew() C.oe_sampler_t {
	return C.oe_sampler_chain_new()
}

func cSamplerChainAddTemp(chain C.oe_sampler_t, temp float32) {
	C.oe_sampler_chain_add_temp(chain, C.float(temp))
}

func cSamplerChainAddTopK(chain C.oe_sampler_t, k int32) {
	C.oe_sampler_chain_add_top_k(chain, C.int32_t(k))
}

func cSamplerChainAddTopP(chain C.oe_sampler_t, p float32) {
	C.oe_sampler_chain_add_top_p(chain, C.float(p))
}

func cSamplerChainAddMinP(chain C.oe_sampler_t, p float32) {
	C.oe_sampler_chain_add_min_p(chain, C.float(p))
}

func cSamplerChainAddPenalties(chain C.oe_sampler_t, lastN int32, repeat, freq, present float32) {
	C.oe_sampler_chain_add_penalties(chain,
		C.int32_t(lastN), C.float(repeat), C.float(freq), C.float(present))
}

func cSamplerChainAddDist(chain C.oe_sampler_t, seed uint32) {
	C.oe_sampler_chain_add_dist(chain, C.uint32_t(seed))
}

func cSamplerChainAddGreedy(chain C.oe_sampler_t) {
	C.oe_sampler_chain_add_greedy(chain)
}

// cSamplerSample samples a token from context at the given output index.
func cSamplerSample(chain C.oe_sampler_t, ctx C.oe_context_t, idx int32) int32 {
	return int32(C.oe_sampler_sample(chain, ctx, C.int32_t(idx)))
}

// cSamplerReset resets sampler chain state.
func cSamplerReset(chain C.oe_sampler_t) {
	C.oe_sampler_reset(chain)
}

// cSamplerFree frees the sampler chain and all owned samplers.
func cSamplerFree(chain C.oe_sampler_t) {
	C.oe_sampler_free(chain)
}

// ---------------------------------------------------------------------------
// Context control — low-level C wrappers
// ---------------------------------------------------------------------------

func cSetEmbeddings(ctx C.oe_context_t, enabled bool) {
	C.oe_set_embeddings(ctx, C.bool(enabled))
}

func cSetCausalAttn(ctx C.oe_context_t, causal bool) {
	C.oe_set_causal_attn(ctx, C.bool(causal))
}

func cSetWarmup(ctx C.oe_context_t, warmup bool) {
	C.oe_set_warmup(ctx, C.bool(warmup))
}

func cSetNThreads(ctx C.oe_context_t, nThreads, nThreadsBatch int32) {
	C.oe_set_n_threads(ctx, C.int32_t(nThreads), C.int32_t(nThreadsBatch))
}

// ---------------------------------------------------------------------------
// Performance — low-level C wrappers
// ---------------------------------------------------------------------------

// PerfData holds performance counters from the inference context.
type PerfData struct {
	LoadMs      float64
	PromptMs    float64
	EvalMs      float64
	PromptCount int32
	EvalCount   int32
}

// cPerfContext returns performance counters.
func cPerfContext(ctx C.oe_context_t) PerfData {
	p := C.oe_perf_context(ctx)
	return PerfData{
		LoadMs:      float64(p.t_load_ms),
		PromptMs:    float64(p.t_p_eval_ms),
		EvalMs:      float64(p.t_eval_ms),
		PromptCount: int32(p.n_p_eval),
		EvalCount:   int32(p.n_eval),
	}
}

// cPerfContextReset resets performance counters.
func cPerfContextReset(ctx C.oe_context_t) {
	C.oe_perf_context_reset(ctx)
}

// SystemInfo returns a string describing CPU features and build info.
func SystemInfo() string {
	return C.GoString(C.oe_system_info())
}

// ---------------------------------------------------------------------------
// Vision (multimodal) — low-level C wrappers
// ---------------------------------------------------------------------------

// cVisionInit creates a vision context from an mmproj GGUF file.
func cVisionInit(mmprojPath string, model C.oe_model_t, nThreads int, useGPU bool) C.oe_vision_t {
	cpath := C.CString(mmprojPath)
	defer C.free(unsafe.Pointer(cpath))
	return C.oe_vision_init(cpath, model, C.int(nThreads), C.bool(useGPU))
}

// cVisionFree frees a vision context.
func cVisionFree(vctx C.oe_vision_t) {
	C.oe_vision_free(vctx)
}

// cVisionSupported checks if the vision context supports vision input.
func cVisionSupported(vctx C.oe_vision_t) bool {
	return bool(C.oe_vision_supported(vctx))
}

// cVisionEval tokenizes a prompt with image markers and evaluates all chunks.
// Returns 0 on success, negative on error. Writes the new KV position to newNPast.
func cVisionEval(vctx C.oe_vision_t, lctx C.oe_context_t,
	prompt string, imagePaths []string, nPast, nBatch int32) (newNPast int32, rc int32) {

	cprompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cprompt))

	var cPaths **C.char
	nImages := len(imagePaths)
	if nImages > 0 {
		// Allocate array of C strings for image paths.
		pathPtrs := make([]*C.char, nImages)
		for i, p := range imagePaths {
			pathPtrs[i] = C.CString(p)
		}
		defer func() {
			for _, cp := range pathPtrs {
				C.free(unsafe.Pointer(cp))
			}
		}()
		// cPaths points into pathPtrs which holds Go-allocated *C.char pointers.
		// The elements themselves are C-allocated strings (safe to pass to C),
		// but we must keep pathPtrs alive until C returns.
		cPaths = (**C.char)(unsafe.Pointer(&pathPtrs[0]))
	}

	var outNPast C.int32_t
	ret := C.oe_vision_eval(vctx, lctx, cprompt, cPaths,
		C.int(nImages), C.int32_t(nPast), C.int32_t(nBatch), &outNPast)
	runtime.KeepAlive(imagePaths) // keep Go strings alive while C holds derived pointers
	runtime.KeepAlive(cPaths)     // keep pathPtrs array alive through the C call

	return int32(outNPast), int32(ret)
}

// cVisionDefaultMarker returns the default media marker ("<__media__>").
func cVisionDefaultMarker() string {
	return C.GoString(C.oe_vision_default_marker())
}

// VisionLogToFile redirects mtmd log output to a file.
func VisionLogToFile(path string) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	C.oe_vision_log_to_file(cpath)
}

// VisionLogDisable suppresses all mtmd log output.
func VisionLogDisable() {
	C.oe_vision_log_disable()
}

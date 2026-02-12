// OpenEye native binding — thin C wrapper around llama.h
// This header exposes only the functions OpenEye needs, with opaque handles
// to avoid leaking llama.cpp internals into Go via CGo.

#ifndef OPENEYE_BINDING_H
#define OPENEYE_BINDING_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------
typedef void* oe_model_t;
typedef void* oe_context_t;
typedef void* oe_sampler_t;
// oe_memory_t removed — not used in binding API

// ---------------------------------------------------------------------------
// Model information returned after loading
// ---------------------------------------------------------------------------
typedef struct {
    int32_t  n_embd;
    int32_t  n_ctx_train;
    int32_t  n_layer;
    int32_t  n_head;
    uint64_t model_size;
    uint64_t n_params;
    char     desc[256];
    char     chat_template[4096];
    bool     has_encoder;
} oe_model_info_t;

// ---------------------------------------------------------------------------
// Performance data returned after generation
// ---------------------------------------------------------------------------
typedef struct {
    double  t_load_ms;
    double  t_p_eval_ms;
    double  t_eval_ms;
    int32_t n_p_eval;
    int32_t n_eval;
} oe_perf_data_t;

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------
void oe_backend_init(void);
void oe_backend_free(void);

// ---------------------------------------------------------------------------
// Model operations
// ---------------------------------------------------------------------------

// Load a GGUF model from path. Returns NULL on failure.
// n_gpu_layers: number of layers to offload to GPU (-1 = all, 0 = none)
// use_mmap:     use memory-mapped I/O for model loading
// use_mlock:    lock model memory to prevent swapping
oe_model_t oe_model_load(const char *path, int32_t n_gpu_layers,
                          bool use_mmap, bool use_mlock);

// Free a loaded model.
void oe_model_free(oe_model_t model);

// Retrieve model metadata.
oe_model_info_t oe_model_get_info(oe_model_t model);

// ---------------------------------------------------------------------------
// Context operations
// ---------------------------------------------------------------------------

// Create an inference context from a loaded model.
// n_ctx:        context size (0 = use model default)
// n_batch:      maximum batch size for prompt processing
// n_threads:    threads for single-token generation (0 = auto)
// n_threads_batch: threads for batch processing (0 = same as n_threads)
// embeddings:   enable embedding extraction
// flash_attn:   enable flash attention (-1=auto, 0=off, 1=on)
// type_k:       KV cache key type (0=f16, 1=q8_0, 2=q4_0)
// type_v:       KV cache value type (0=f16, 1=q8_0, 2=q4_0)
oe_context_t oe_context_new(oe_model_t model, uint32_t n_ctx, uint32_t n_batch,
                             int32_t n_threads, int32_t n_threads_batch,
                             bool embeddings, int32_t flash_attn,
                             int32_t type_k, int32_t type_v);

// Free an inference context.
void oe_context_free(oe_context_t ctx);

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

// Tokenize text into the provided token buffer.
// Returns the number of tokens produced, or a negative value if the buffer
// is too small (the absolute value is the required size).
int32_t oe_tokenize(oe_model_t model, const char *text, int32_t text_len,
                     int32_t *tokens, int32_t n_max_tokens,
                     bool add_special, bool parse_special);

// Convert a single token to its text piece.
// Returns the number of bytes written, or negative if buffer too small.
int32_t oe_token_to_piece(oe_model_t model, int32_t token,
                           char *buf, int32_t buf_len);

// Check if a token signals end-of-generation.
bool oe_token_is_eog(oe_model_t model, int32_t token);

// Get special token IDs.
int32_t oe_token_bos(oe_model_t model);
int32_t oe_token_eos(oe_model_t model);

// Get the vocabulary size (number of tokens in the model's vocabulary).
int32_t oe_vocab_n_tokens(oe_model_t model);

// ---------------------------------------------------------------------------
// Decode / Evaluate
// ---------------------------------------------------------------------------

// Evaluate a batch of tokens. Positions are tracked automatically.
// Returns 0 on success, 1 if KV cache is full, negative on error.
int32_t oe_decode(oe_context_t ctx, int32_t *tokens, int32_t n_tokens);

// Evaluate a single batch with explicit position and logits flag control.
// tokens[i] is placed at pos_start + i, and logits are only computed for the
// last token (the one at index n_tokens-1).
// Returns 0 on success, 1 if KV cache is full, negative on error.
int32_t oe_decode_batch(oe_context_t ctx, int32_t *tokens, int32_t n_tokens,
                         int32_t pos_start);

// Encode a batch of tokens using the model's encoder (for BERT/encoder models).
// All tokens are marked as outputs so embeddings can be extracted for each.
// Returns 0 on success, negative on error.
int32_t oe_encode(oe_context_t ctx, int32_t *tokens, int32_t n_tokens);

// ---------------------------------------------------------------------------
// Logits & Embeddings
// ---------------------------------------------------------------------------

// Get logits for the token at position idx in the last decoded batch.
// Returns NULL if idx is invalid. The returned pointer points into
// context-owned memory and remains valid until the next decode call.
float *oe_get_logits(oe_context_t ctx, int32_t idx);

// Get the embedding vector for the token at position idx.
// Returns NULL if embeddings are not enabled or idx is invalid.
float *oe_get_embeddings(oe_context_t ctx, int32_t idx);

// Get pooled embeddings for a sequence. Returns NULL if pooling is disabled.
float *oe_get_embeddings_seq(oe_context_t ctx, int32_t seq_id);

// ---------------------------------------------------------------------------
// KV / Memory management
// ---------------------------------------------------------------------------

// Clear all KV cache contents.
void oe_memory_clear(oe_context_t ctx);

// Remove tokens in [p0, p1) for a given sequence from the KV cache.
// seq_id < 0 matches all sequences. p0 < 0 means 0. p1 < 0 means inf.
bool oe_memory_seq_rm(oe_context_t ctx, int32_t seq_id,
                       int32_t p0, int32_t p1);

// Get the maximum position present in the KV cache for the given sequence.
// Returns -1 if the sequence is empty.
int32_t oe_memory_seq_pos_max(oe_context_t ctx, int32_t seq_id);

// Shift positions in the KV cache for a sequence by delta.
// All positions in [p0, p1) for seq_id are shifted by delta.
// Used for context window sliding: after removing old tokens,
// shift remaining positions down to keep them contiguous.
void oe_memory_seq_add(oe_context_t ctx, int32_t seq_id,
                       int32_t p0, int32_t p1, int32_t delta);

// ---------------------------------------------------------------------------
// Speculative decoding
// ---------------------------------------------------------------------------

// Evaluate a batch of tokens with logits computed for ALL tokens (not just
// the last one). This is needed for speculative decoding verification: the
// target model evaluates N draft tokens and checks each one against its own
// distribution. Returns 0 on success, 1 if KV cache full, negative on error.
int32_t oe_decode_batch_logits_all(oe_context_t ctx, int32_t *tokens,
                                    int32_t n_tokens, int32_t pos_start);

// ---------------------------------------------------------------------------
// Sampler chain
// ---------------------------------------------------------------------------

// Create a sampler chain. The chain takes ownership of added samplers.
oe_sampler_t oe_sampler_chain_new(void);

// Add a temperature sampler to the chain.
void oe_sampler_chain_add_temp(oe_sampler_t chain, float temp);

// Add a top-K sampler to the chain.
void oe_sampler_chain_add_top_k(oe_sampler_t chain, int32_t k);

// Add a top-P (nucleus) sampler to the chain.
void oe_sampler_chain_add_top_p(oe_sampler_t chain, float p);

// Add a min-P sampler to the chain.
void oe_sampler_chain_add_min_p(oe_sampler_t chain, float p);

// Add a repetition/frequency/presence penalty sampler to the chain.
void oe_sampler_chain_add_penalties(oe_sampler_t chain, int32_t last_n,
                                     float repeat, float freq, float present);

// Add a distribution sampler (random sampling with seed) to the chain.
// seed == 0xFFFFFFFF uses a random seed.
void oe_sampler_chain_add_dist(oe_sampler_t chain, uint32_t seed);

// Add a greedy (argmax) sampler to the chain.
void oe_sampler_chain_add_greedy(oe_sampler_t chain);

// Sample a token from the given context at the specified output index.
// idx = -1 means the last token in the batch.
int32_t oe_sampler_sample(oe_sampler_t chain, oe_context_t ctx, int32_t idx);

// Reset the sampler chain state (e.g., penalty tracking).
void oe_sampler_reset(oe_sampler_t chain);

// Free the sampler chain and all samplers it owns.
void oe_sampler_free(oe_sampler_t chain);

// ---------------------------------------------------------------------------
// Context control
// ---------------------------------------------------------------------------

// Enable or disable embedding extraction mode.
void oe_set_embeddings(oe_context_t ctx, bool enabled);

// Enable or disable causal attention.
void oe_set_causal_attn(oe_context_t ctx, bool causal);

// Set warmup mode (pre-loads tensor weights into cache).
void oe_set_warmup(oe_context_t ctx, bool warmup);

// Set thread counts for generation and batch processing.
void oe_set_n_threads(oe_context_t ctx, int32_t n_threads,
                       int32_t n_threads_batch);

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

// Get performance counters for the context.
oe_perf_data_t oe_perf_context(oe_context_t ctx);

// Reset performance counters.
void oe_perf_context_reset(oe_context_t ctx);

// Get system info string (CPU features, etc.)
const char *oe_system_info(void);

// ---------------------------------------------------------------------------
// Log control
// ---------------------------------------------------------------------------

// Redirect all llama.cpp log output to a file (append mode).
// Pass NULL to restore default stderr logging.
void oe_log_to_file(const char *path);

// Suppress all llama.cpp log output.
void oe_log_disable(void);

#ifdef __cplusplus
}
#endif

#endif // OPENEYE_BINDING_H

// OpenEye native binding — C implementation
// Thin wrapper around llama.h, translating opaque handles to/from llama types.

#include "binding.h"
#include "llama.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------

void oe_backend_init(void) {
    llama_backend_init();
}

void oe_backend_free(void) {
    llama_backend_free();
}

// ---------------------------------------------------------------------------
// Model operations
// ---------------------------------------------------------------------------

oe_model_t oe_model_load(const char *path, int32_t n_gpu_layers,
                          bool use_mmap, bool use_mlock) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.use_mmap     = use_mmap;
    params.use_mlock    = use_mlock;

    struct llama_model *model = llama_model_load_from_file(path, params);
    return (oe_model_t)model;
}

void oe_model_free(oe_model_t model) {
    if (model) {
        llama_model_free((struct llama_model *)model);
    }
}

oe_model_info_t oe_model_get_info(oe_model_t model) {
    oe_model_info_t info;
    memset(&info, 0, sizeof(info));

    struct llama_model *m = (struct llama_model *)model;
    if (!m) return info;

    info.n_embd      = llama_model_n_embd(m);
    info.n_ctx_train  = llama_model_n_ctx_train(m);
    info.n_layer      = llama_model_n_layer(m);
    info.n_head       = llama_model_n_head(m);
    info.model_size   = llama_model_size(m);
    info.n_params     = llama_model_n_params(m);
    info.has_encoder   = llama_model_has_encoder(m);

    llama_model_desc(m, info.desc, sizeof(info.desc));

    const char *tmpl = llama_model_chat_template(m, NULL);
    if (tmpl) {
        size_t len = strlen(tmpl);
        if (len >= sizeof(info.chat_template)) {
            len = sizeof(info.chat_template) - 1;
        }
        memcpy(info.chat_template, tmpl, len);
        info.chat_template[len] = '\0';
    }

    return info;
}

// ---------------------------------------------------------------------------
// Context operations
// ---------------------------------------------------------------------------

oe_context_t oe_context_new(oe_model_t model, uint32_t n_ctx, uint32_t n_batch,
                             int32_t n_threads, int32_t n_threads_batch,
                             bool embeddings, int32_t flash_attn) {
    struct llama_context_params params = llama_context_default_params();

    if (n_ctx > 0)        params.n_ctx          = n_ctx;
    if (n_batch > 0)      params.n_batch        = n_batch;
    if (n_threads > 0)    params.n_threads       = n_threads;
    if (n_threads_batch > 0) params.n_threads_batch = n_threads_batch;
    else if (n_threads > 0)  params.n_threads_batch = n_threads;

    params.embeddings = embeddings;

    if (flash_attn >= 0) {
        params.flash_attn_type = (enum llama_flash_attn_type)flash_attn;
    }

    struct llama_context *ctx = llama_init_from_model(
        (struct llama_model *)model, params);
    return (oe_context_t)ctx;
}

void oe_context_free(oe_context_t ctx) {
    if (ctx) {
        llama_free((struct llama_context *)ctx);
    }
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

int32_t oe_tokenize(oe_model_t model, const char *text, int32_t text_len,
                     int32_t *tokens, int32_t n_max_tokens,
                     bool add_special, bool parse_special) {
    if (!model) return 0;
    struct llama_model *m = (struct llama_model *)model;
    const struct llama_vocab *vocab = llama_model_get_vocab(m);
    return llama_tokenize(vocab, text, text_len,
                          (llama_token *)tokens, n_max_tokens,
                          add_special, parse_special);
}

int32_t oe_token_to_piece(oe_model_t model, int32_t token,
                           char *buf, int32_t buf_len) {
    if (!model) return 0;
    struct llama_model *m = (struct llama_model *)model;
    const struct llama_vocab *vocab = llama_model_get_vocab(m);
    return llama_token_to_piece(vocab, (llama_token)token, buf, buf_len, 0, false);
}

bool oe_token_is_eog(oe_model_t model, int32_t token) {
    if (!model) return true; // treat as EOG if model is NULL (safe default)
    struct llama_model *m = (struct llama_model *)model;
    const struct llama_vocab *vocab = llama_model_get_vocab(m);
    return llama_vocab_is_eog(vocab, (llama_token)token);
}

int32_t oe_token_bos(oe_model_t model) {
    if (!model) return 0;
    struct llama_model *m = (struct llama_model *)model;
    const struct llama_vocab *vocab = llama_model_get_vocab(m);
    return (int32_t)llama_vocab_bos(vocab);
}

int32_t oe_token_eos(oe_model_t model) {
    if (!model) return 0;
    struct llama_model *m = (struct llama_model *)model;
    const struct llama_vocab *vocab = llama_model_get_vocab(m);
    return (int32_t)llama_vocab_eos(vocab);
}

// ---------------------------------------------------------------------------
// Decode / Evaluate
// ---------------------------------------------------------------------------

int32_t oe_decode(oe_context_t ctx, int32_t *tokens, int32_t n_tokens) {
    if (!ctx || !tokens || n_tokens <= 0) return -1;
    struct llama_batch batch = llama_batch_get_one(
        (llama_token *)tokens, n_tokens);
    return llama_decode((struct llama_context *)ctx, batch);
}

int32_t oe_decode_batch(oe_context_t ctx, int32_t *tokens, int32_t n_tokens,
                         int32_t pos_start) {
    if (!ctx || !tokens || n_tokens <= 0) return -1;
    struct llama_context *c = (struct llama_context *)ctx;

    // Allocate a full batch so we can set positions and logits flags explicitly.
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]    = (llama_token)tokens[i];
        batch.pos[i]      = (llama_pos)(pos_start + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        // Only request logits for the last token.
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }

    int32_t rc = llama_decode(c, batch);
    llama_batch_free(batch);
    return rc;
}

int32_t oe_encode(oe_context_t ctx, int32_t *tokens, int32_t n_tokens) {
    if (!ctx || !tokens || n_tokens <= 0) return -1;
    struct llama_context *c = (struct llama_context *)ctx;

    // Build a batch with all tokens marked as outputs (for embedding extraction).
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]     = (llama_token)tokens[i];
        batch.pos[i]       = (llama_pos)i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1; // mark all as outputs for embeddings
    }

    int32_t rc = llama_encode(c, batch);
    llama_batch_free(batch);
    return rc;
}

// ---------------------------------------------------------------------------
// Logits & Embeddings
// ---------------------------------------------------------------------------

float *oe_get_logits(oe_context_t ctx, int32_t idx) {
    if (!ctx) return NULL;
    return llama_get_logits_ith((struct llama_context *)ctx, idx);
}

float *oe_get_embeddings(oe_context_t ctx, int32_t idx) {
    if (!ctx) return NULL;
    return llama_get_embeddings_ith((struct llama_context *)ctx, idx);
}

float *oe_get_embeddings_seq(oe_context_t ctx, int32_t seq_id) {
    if (!ctx) return NULL;
    return llama_get_embeddings_seq(
        (struct llama_context *)ctx, (llama_seq_id)seq_id);
}

// ---------------------------------------------------------------------------
// KV / Memory management
// ---------------------------------------------------------------------------

void oe_memory_clear(oe_context_t ctx) {
    if (!ctx) return;
    llama_memory_t mem = llama_get_memory((struct llama_context *)ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }
}

bool oe_memory_seq_rm(oe_context_t ctx, int32_t seq_id,
                       int32_t p0, int32_t p1) {
    if (!ctx) return false;
    llama_memory_t mem = llama_get_memory((struct llama_context *)ctx);
    if (!mem) return false;
    return llama_memory_seq_rm(mem, (llama_seq_id)seq_id,
                                (llama_pos)p0, (llama_pos)p1);
}

int32_t oe_memory_seq_pos_max(oe_context_t ctx, int32_t seq_id) {
    if (!ctx) return -1;
    llama_memory_t mem = llama_get_memory((struct llama_context *)ctx);
    if (!mem) return -1;
    return (int32_t)llama_memory_seq_pos_max(mem, (llama_seq_id)seq_id);
}

// ---------------------------------------------------------------------------
// Sampler chain
// ---------------------------------------------------------------------------

oe_sampler_t oe_sampler_chain_new(void) {
    struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
    struct llama_sampler *chain = llama_sampler_chain_init(params);
    return (oe_sampler_t)chain;
}

void oe_sampler_chain_add_temp(oe_sampler_t chain, float temp) {
    if (!chain) return;
    llama_sampler_chain_add(
        (struct llama_sampler *)chain,
        llama_sampler_init_temp(temp));
}

void oe_sampler_chain_add_top_k(oe_sampler_t chain, int32_t k) {
    if (!chain) return;
    llama_sampler_chain_add(
        (struct llama_sampler *)chain,
        llama_sampler_init_top_k(k));
}

void oe_sampler_chain_add_top_p(oe_sampler_t chain, float p) {
    if (!chain) return;
    llama_sampler_chain_add(
        (struct llama_sampler *)chain,
        llama_sampler_init_top_p(p, 1));
}

void oe_sampler_chain_add_min_p(oe_sampler_t chain, float p) {
    if (!chain) return;
    llama_sampler_chain_add(
        (struct llama_sampler *)chain,
        llama_sampler_init_min_p(p, 1));
}

void oe_sampler_chain_add_penalties(oe_sampler_t chain, int32_t last_n,
                                     float repeat, float freq, float present) {
    if (!chain) return;
    llama_sampler_chain_add(
        (struct llama_sampler *)chain,
        llama_sampler_init_penalties(last_n, repeat, freq, present));
}

void oe_sampler_chain_add_dist(oe_sampler_t chain, uint32_t seed) {
    if (!chain) return;
    llama_sampler_chain_add(
        (struct llama_sampler *)chain,
        llama_sampler_init_dist(seed));
}

void oe_sampler_chain_add_greedy(oe_sampler_t chain) {
    if (!chain) return;
    llama_sampler_chain_add(
        (struct llama_sampler *)chain,
        llama_sampler_init_greedy());
}

int32_t oe_sampler_sample(oe_sampler_t chain, oe_context_t ctx, int32_t idx) {
    if (!chain || !ctx) return 0;
    return (int32_t)llama_sampler_sample(
        (struct llama_sampler *)chain,
        (struct llama_context *)ctx, idx);
}

void oe_sampler_reset(oe_sampler_t chain) {
    if (!chain) return;
    llama_sampler_reset((struct llama_sampler *)chain);
}

void oe_sampler_free(oe_sampler_t chain) {
    if (chain) {
        llama_sampler_free((struct llama_sampler *)chain);
    }
}

// ---------------------------------------------------------------------------
// Context control
// ---------------------------------------------------------------------------

void oe_set_embeddings(oe_context_t ctx, bool enabled) {
    if (!ctx) return;
    llama_set_embeddings((struct llama_context *)ctx, enabled);
}

void oe_set_causal_attn(oe_context_t ctx, bool causal) {
    if (!ctx) return;
    llama_set_causal_attn((struct llama_context *)ctx, causal);
}

void oe_set_warmup(oe_context_t ctx, bool warmup) {
    if (!ctx) return;
    llama_set_warmup((struct llama_context *)ctx, warmup);
}

void oe_set_n_threads(oe_context_t ctx, int32_t n_threads,
                       int32_t n_threads_batch) {
    if (!ctx) return;
    llama_set_n_threads((struct llama_context *)ctx,
                         n_threads, n_threads_batch);
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

oe_perf_data_t oe_perf_context(oe_context_t ctx) {
    oe_perf_data_t data;
    memset(&data, 0, sizeof(data));

    if (!ctx) return data;

    struct llama_perf_context_data perf =
        llama_perf_context((struct llama_context *)ctx);

    data.t_load_ms   = perf.t_load_ms;
    data.t_p_eval_ms = perf.t_p_eval_ms;
    data.t_eval_ms   = perf.t_eval_ms;
    data.n_p_eval    = perf.n_p_eval;
    data.n_eval      = perf.n_eval;

    return data;
}

void oe_perf_context_reset(oe_context_t ctx) {
    if (!ctx) return;
    llama_perf_context_reset((struct llama_context *)ctx);
}

const char *oe_system_info(void) {
    return llama_print_system_info();
}

// ---------------------------------------------------------------------------
// Log control
// ---------------------------------------------------------------------------

static FILE *oe_log_fp = NULL;

static void oe_log_file_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)user_data;
    if (oe_log_fp) {
        fputs(text, oe_log_fp);
        fflush(oe_log_fp);
    }
}

static void oe_log_noop_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

void oe_log_to_file(const char *path) {
    // Close previous file if any (but not stderr).
    if (oe_log_fp && oe_log_fp != stderr) {
        fclose(oe_log_fp);
        oe_log_fp = NULL;
    }
    if (!path) {
        // Restore default stderr logging.
        llama_log_set(NULL, NULL);
        return;
    }
    oe_log_fp = fopen(path, "a");
    if (oe_log_fp) {
        llama_log_set(oe_log_file_callback, NULL);
    }
}

void oe_log_disable(void) {
    if (oe_log_fp && oe_log_fp != stderr) {
        fclose(oe_log_fp);
        oe_log_fp = NULL;
    }
    llama_log_set(oe_log_noop_callback, NULL);
}

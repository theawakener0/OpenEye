// OpenEye vision binding — C implementation
// Wraps the mtmd (multimodal) API for image+text evaluation.

#include "binding_vision.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Vision context lifecycle
// ---------------------------------------------------------------------------

oe_vision_t oe_vision_init(const char *mmproj_path, oe_model_t text_model,
                            int n_threads, bool use_gpu) {
    if (!mmproj_path || !text_model) return NULL;

    struct mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu       = use_gpu;
    params.n_threads     = n_threads;
    params.print_timings = false;
    params.warmup        = false; // we handle warmup separately

    mtmd_context *ctx = mtmd_init_from_file(
        mmproj_path,
        (const struct llama_model *)text_model,
        params);

    return (oe_vision_t)ctx;
}

void oe_vision_free(oe_vision_t vctx) {
    if (vctx) {
        mtmd_free((mtmd_context *)vctx);
    }
}

bool oe_vision_supported(oe_vision_t vctx) {
    if (!vctx) return false;
    return mtmd_support_vision((mtmd_context *)vctx);
}

// ---------------------------------------------------------------------------
// Bitmap loading
// ---------------------------------------------------------------------------

oe_bitmap_t oe_vision_load_image(oe_vision_t vctx, const char *path) {
    if (!vctx || !path) return NULL;
    mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_file(
        (mtmd_context *)vctx, path);
    return (oe_bitmap_t)bmp;
}

void oe_vision_bitmap_free(oe_bitmap_t bmp) {
    if (bmp) {
        mtmd_bitmap_free((mtmd_bitmap *)bmp);
    }
}

// ---------------------------------------------------------------------------
// Tokenize + Evaluate
// ---------------------------------------------------------------------------

int32_t oe_vision_eval(oe_vision_t vctx, oe_context_t lctx,
                        const char *prompt, const char **image_paths,
                        int n_images, int32_t n_past, int32_t n_batch,
                        int32_t *new_n_past) {
    if (!vctx || !lctx || !prompt) return -1;
    if (n_images > 0 && !image_paths) return -1;

    mtmd_context *mctx = (mtmd_context *)vctx;
    struct llama_context *llctx = (struct llama_context *)lctx;

    // Load all bitmaps from file paths.
    mtmd_bitmap **bitmaps = NULL;
    if (n_images > 0) {
        bitmaps = (mtmd_bitmap **)calloc(n_images, sizeof(mtmd_bitmap *));
        if (!bitmaps) return -2;

        for (int i = 0; i < n_images; i++) {
            bitmaps[i] = mtmd_helper_bitmap_init_from_file(mctx, image_paths[i]);
            if (!bitmaps[i]) {
                // Clean up already loaded bitmaps.
                for (int j = 0; j < i; j++) {
                    mtmd_bitmap_free(bitmaps[j]);
                }
                free(bitmaps);
                return -3; // image load failed
            }
        }
    }

    // Tokenize prompt with image markers.
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (!chunks) {
        for (int i = 0; i < n_images; i++) {
            mtmd_bitmap_free(bitmaps[i]);
        }
        free(bitmaps);
        return -4;
    }

    mtmd_input_text input_text;
    input_text.text          = prompt;
    input_text.add_special   = true;
    input_text.parse_special = true;

    int32_t tok_rc = mtmd_tokenize(mctx, chunks, &input_text,
                                    (const mtmd_bitmap **)bitmaps,
                                    (size_t)n_images);

    // Free bitmaps after tokenization — they're no longer needed.
    for (int i = 0; i < n_images; i++) {
        mtmd_bitmap_free(bitmaps[i]);
    }
    free(bitmaps);

    if (tok_rc != 0) {
        mtmd_input_chunks_free(chunks);
        return -5; // tokenization failed (marker/bitmap mismatch)
    }

    // Evaluate all chunks (text + encoded images) into the llama context.
    llama_pos out_n_past = (llama_pos)n_past;
    int32_t eval_rc = mtmd_helper_eval_chunks(
        mctx, llctx, chunks,
        (llama_pos)n_past,      // n_past
        0,                       // seq_id
        (int32_t)n_batch,        // n_batch
        true,                    // logits_last
        &out_n_past);

    mtmd_input_chunks_free(chunks);

    if (eval_rc != 0) {
        return -6; // eval failed
    }

    if (new_n_past) {
        *new_n_past = (int32_t)out_n_past;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Log control
// ---------------------------------------------------------------------------

// Reuse the log callbacks from binding.c (file and noop).
// We declare them extern here because they are static in binding.c.
// Instead, we use mtmd's own log set function.

static FILE *oe_vision_log_fp = NULL;

static void oe_vision_log_file_cb(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)user_data;
    if (oe_vision_log_fp) {
        fputs(text, oe_vision_log_fp);
        fflush(oe_vision_log_fp);
    }
}

static void oe_vision_log_noop_cb(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

void oe_vision_log_to_file(const char *path) {
    if (oe_vision_log_fp && oe_vision_log_fp != stderr) {
        fclose(oe_vision_log_fp);
        oe_vision_log_fp = NULL;
    }
    if (!path) {
        mtmd_helper_log_set(NULL, NULL);
        return;
    }
    oe_vision_log_fp = fopen(path, "a");
    if (oe_vision_log_fp) {
        mtmd_helper_log_set(oe_vision_log_file_cb, NULL);
    }
}

void oe_vision_log_disable(void) {
    if (oe_vision_log_fp && oe_vision_log_fp != stderr) {
        fclose(oe_vision_log_fp);
        oe_vision_log_fp = NULL;
    }
    mtmd_helper_log_set(oe_vision_log_noop_cb, NULL);
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

const char *oe_vision_default_marker(void) {
    return mtmd_default_marker();
}

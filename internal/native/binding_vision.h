// OpenEye vision binding — thin C wrapper around mtmd (multimodal) API.
// Provides opaque handles for vision context, bitmaps, and chunk evaluation
// without leaking mtmd internals into Go via CGo.

#ifndef OPENEYE_BINDING_VISION_H
#define OPENEYE_BINDING_VISION_H

#include "binding.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------
typedef void* oe_vision_t;
typedef void* oe_bitmap_t;
typedef void* oe_chunks_t;

// ---------------------------------------------------------------------------
// Vision context lifecycle
// ---------------------------------------------------------------------------

// Initialize a vision (multimodal) context from a mmproj GGUF file.
// The text_model must already be loaded. Returns NULL on failure.
oe_vision_t oe_vision_init(const char *mmproj_path, oe_model_t text_model,
                            int n_threads, bool use_gpu);

// Free the vision context.
void oe_vision_free(oe_vision_t vctx);

// Check if the vision context supports vision input.
bool oe_vision_supported(oe_vision_t vctx);

// ---------------------------------------------------------------------------
// Bitmap (image) loading
// ---------------------------------------------------------------------------

// Load an image from a file path into a bitmap. Returns NULL on failure.
oe_bitmap_t oe_vision_load_image(oe_vision_t vctx, const char *path);

// Free a loaded bitmap.
void oe_vision_bitmap_free(oe_bitmap_t bmp);

// ---------------------------------------------------------------------------
// Tokenize + Evaluate (high-level)
// ---------------------------------------------------------------------------

// Tokenize a prompt containing <__media__> markers with the given images,
// then evaluate all chunks (text + image) into the llama context.
// On success, returns 0 and writes the new KV cache position to *new_n_past.
// On failure, returns a negative error code.
//
// image_paths: array of file paths, one per <__media__> marker in prompt.
// n_images:    number of image paths (must match marker count in prompt).
// n_past:      current KV cache position (typically 0 after clear).
// n_batch:     batch size for evaluation (should match context's n_batch).
int32_t oe_vision_eval(oe_vision_t vctx, oe_context_t lctx,
                        const char *prompt, const char **image_paths,
                        int n_images, int32_t n_past, int32_t n_batch,
                        int32_t *new_n_past);

// ---------------------------------------------------------------------------
// Log control (redirects mtmd logs along with llama logs)
// ---------------------------------------------------------------------------

// Redirect mtmd log output to a file (append mode).
void oe_vision_log_to_file(const char *path);

// Suppress all mtmd log output.
void oe_vision_log_disable(void);

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

// Return the default media marker string ("<__media__>").
const char *oe_vision_default_marker(void);

#ifdef __cplusplus
}
#endif

#endif // OPENEYE_BINDING_VISION_H

//go:build native

package native

/*
#include "binding_vision.h"
*/
import "C"

import (
	"fmt"
	"log"
	"sync"
)

// VisionContext manages a multimodal (vision) context backed by an mmproj model.
// It provides the ability to evaluate prompts containing image references.
type VisionContext struct {
	vctx   C.oe_vision_t
	closed bool
	mu     sync.Mutex
}

// NewVisionContext initializes the vision (multimodal) context from an mmproj
// GGUF file. The text model must already be loaded. The vision context shares
// the text model's weights but loads its own vision projector.
//
// Safety: model.Handle() is called once here to obtain the raw C pointer.
// This is safe because the model outlives the vision context (adapter.Close
// frees vision before model), and the adapter's mutex serializes all
// operations so the model cannot be closed while this is running.
func NewVisionContext(mmprojPath string, model *Model, nThreads int, useGPU bool) (*VisionContext, error) {
	if model == nil || model.IsClosed() {
		return nil, fmt.Errorf("vision: text model is nil or closed")
	}

	vctx := cVisionInit(mmprojPath, model.Handle(), nThreads, useGPU)
	if vctx == nil {
		return nil, fmt.Errorf("vision: failed to load mmproj from %q", mmprojPath)
	}

	if !cVisionSupported(vctx) {
		cVisionFree(vctx)
		return nil, fmt.Errorf("vision: mmproj at %q does not support vision input", mmprojPath)
	}

	log.Printf("native/vision: mmproj loaded from %s (vision supported)", mmprojPath)
	return &VisionContext{vctx: vctx}, nil
}

// EvalWithImages tokenizes a prompt containing <__media__> markers, loads the
// referenced images, runs the vision encoder on image chunks, and evaluates
// all chunks into the llama context's KV cache.
//
// Returns the new KV cache position after evaluation, or an error.
// The caller is responsible for clearing/managing KV cache before calling this.
//
// Safety: ctx.Handle() is called under this method's lock. The adapter's mutex
// ensures the llama context cannot be closed concurrently.
func (v *VisionContext) EvalWithImages(ctx *Context, prompt string, imagePaths []string, nBatch int32) (newPos int32, err error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.closed {
		return 0, fmt.Errorf("vision: context is closed")
	}
	if ctx == nil || ctx.IsClosed() {
		return 0, fmt.Errorf("vision: llama context is nil or closed")
	}

	newNPast, rc := cVisionEval(v.vctx, ctx.Handle(), prompt, imagePaths, 0, nBatch)
	if rc != 0 {
		return 0, fmt.Errorf("vision: eval failed (rc=%d) — check image paths and <__media__> marker count matches image count", rc)
	}

	return newNPast, nil
}

// Close frees the vision context and its resources.
func (v *VisionContext) Close() {
	v.mu.Lock()
	defer v.mu.Unlock()

	if !v.closed {
		cVisionFree(v.vctx)
		v.closed = true
	}
}

// IsClosed returns true if the vision context has been freed.
func (v *VisionContext) IsClosed() bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.closed
}

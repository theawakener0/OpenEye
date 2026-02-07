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

// Model wraps a loaded GGUF model. It is safe for concurrent use (the
// underlying llama_model is thread-safe for read operations).
type Model struct {
	handle C.oe_model_t
	info   ModelInfo
	mu     sync.RWMutex
	closed bool
}

// ModelOptions configures model loading behavior.
type ModelOptions struct {
	// NGPULayers is the number of layers to offload to GPU.
	// 0 = CPU-only, -1 = offload all layers.
	NGPULayers int32

	// UseMmap enables memory-mapped model loading (recommended for most cases).
	UseMmap bool

	// UseMlock locks model memory to prevent the OS from swapping it out.
	UseMlock bool
}

// DefaultModelOptions returns sensible defaults for edge deployment.
func DefaultModelOptions() ModelOptions {
	return ModelOptions{
		NGPULayers: 0,     // CPU-only for edge devices
		UseMmap:    true,  // mmap is efficient on Linux
		UseMlock:   false, // don't lock by default (limited RAM on edge)
	}
}

// LoadModel loads a GGUF model from the given path.
func LoadModel(path string, opts ModelOptions) (*Model, error) {
	handle := cModelLoad(path, opts.NGPULayers, opts.UseMmap, opts.UseMlock)
	if handle == nil {
		return nil, fmt.Errorf("native: failed to load model from %q", path)
	}

	info := cModelGetInfo(handle)

	return &Model{
		handle: handle,
		info:   info,
	}, nil
}

// Info returns cached model metadata.
func (m *Model) Info() ModelInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.info
}

// Handle returns the raw C model handle for use by Context and other
// low-level operations. Callers must not free the handle.
// Returns nil if the model has been closed.
func (m *Model) Handle() C.oe_model_t {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil
	}
	return m.handle
}

// IsClosed returns true if the model has been freed.
func (m *Model) IsClosed() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.closed
}

// Tokenize converts text to token IDs using the model's vocabulary.
// If addSpecial is true, BOS/EOS tokens are added per model config.
// If parseSpecial is true, special tokens in the text are parsed.
func (m *Model) Tokenize(text string, addSpecial, parseSpecial bool) ([]int32, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return nil, fmt.Errorf("native: model is closed")
	}

	// First pass: determine required size.
	n := cTokenize(m.handle, text, nil, addSpecial, parseSpecial)
	if n == 0 {
		return nil, nil
	}

	// If n is negative, its absolute value is the required buffer size.
	size := n
	if size < 0 {
		size = -size
	}

	tokens := make([]int32, size)
	n = cTokenize(m.handle, text, tokens, addSpecial, parseSpecial)
	if n < 0 {
		return nil, fmt.Errorf("native: tokenization failed, need %d tokens", -n)
	}

	return tokens[:n], nil
}

// TokenToPiece converts a token ID to its string representation.
func (m *Model) TokenToPiece(token int32) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return ""
	}
	return cTokenToPiece(m.handle, token)
}

// TokenIsEOG returns true if the token signals end-of-generation.
func (m *Model) TokenIsEOG(token int32) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return true // safe default: treat as EOG when model is closed
	}
	return cTokenIsEOG(m.handle, token)
}

// TokenBOS returns the beginning-of-sentence token ID.
func (m *Model) TokenBOS() int32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return -1
	}
	return cTokenBOS(m.handle)
}

// TokenEOS returns the end-of-sentence token ID.
func (m *Model) TokenEOS() int32 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed {
		return -1
	}
	return cTokenEOS(m.handle)
}

// Close frees the model and all associated resources.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	cModelFree(m.handle)
	m.handle = nil
	return nil
}

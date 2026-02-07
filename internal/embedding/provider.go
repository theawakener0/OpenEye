package embedding

import (
	"context"
	"fmt"
	"sync"

	"OpenEye/internal/config"
)

// Provider exposes semantic embedding capabilities.
type Provider interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Close() error
}

// ProviderFactory constructs a Provider from the embedding configuration.
type ProviderFactory func(config.EmbeddingConfig) (Provider, error)

var (
	providersMu sync.RWMutex
	providers   = map[string]ProviderFactory{}
)

// RegisterProvider registers an embedding provider factory under the given
// backend name. Typically called from an init() function.
func RegisterProvider(name string, factory ProviderFactory) {
	providersMu.Lock()
	defer providersMu.Unlock()
	providers[name] = factory
}

// New constructs an embedding provider based on configuration.
func New(cfg config.EmbeddingConfig) (Provider, error) {
	if cfg.Enabled == nil || !*cfg.Enabled {
		return nil, nil
	}

	backend := cfg.Backend
	if backend == "" {
		backend = "llamacpp"
	}

	// Check built-in backends first, then the registry.
	switch backend {
	case "llamacpp":
		return newLlamaCppProvider(cfg.LlamaCpp)
	default:
		providersMu.RLock()
		factory, ok := providers[backend]
		providersMu.RUnlock()
		if ok {
			return factory(cfg)
		}
		return nil, fmt.Errorf("embedding: unsupported backend %q", backend)
	}
}

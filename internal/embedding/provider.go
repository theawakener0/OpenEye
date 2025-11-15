package embedding

import (
	"context"
	"fmt"

	"OpenEye/internal/config"
)

// Provider exposes semantic embedding capabilities.
type Provider interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Close() error
}

// New constructs an embedding provider based on configuration.
func New(cfg config.EmbeddingConfig) (Provider, error) {
	if !cfg.Enabled {
		return nil, nil
	}

	backend := cfg.Backend
	if backend == "" {
		backend = "llamacpp"
	}

	switch backend {
	case "llamacpp":
		return newLlamaCppProvider(cfg.LlamaCpp)
	default:
		return nil, fmt.Errorf("embedding: unsupported backend %q", backend)
	}
}

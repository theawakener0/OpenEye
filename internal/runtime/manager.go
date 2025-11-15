package runtime

import (
	"context"
	"fmt"
	"strings"

	"OpenEye/internal/config"
)

// Manager routes generation requests to configured runtime adapters.
type Manager struct {
	adapter Adapter
}

// NewManager constructs the runtime manager using the provided configuration.
func NewManager(cfg config.RuntimeConfig, registry Registry) (*Manager, error) {
	backend := strings.TrimSpace(strings.ToLower(cfg.Backend))
	if backend == "" {
		backend = "http"
	}

	adapterFactory, ok := registry[backend]
	if !ok {
		return nil, fmt.Errorf("runtime: backend %q not registered", backend)
	}

	adapter, err := adapterFactory(cfg)
	if err != nil {
		return nil, err
	}

	return &Manager{adapter: adapter}, nil
}

// Close frees adapter resources.
func (m *Manager) Close() error {
	if m == nil || m.adapter == nil {
		return nil
	}
	return m.adapter.Close()
}

// Generate runs a single-shot completion request.
func (m *Manager) Generate(ctx context.Context, req Request) (Response, error) {
	if m == nil || m.adapter == nil {
		return Response{}, fmt.Errorf("runtime: no adapter configured")
	}
	return m.adapter.Generate(ctx, req)
}

// Stream requests a streaming generation.
func (m *Manager) Stream(ctx context.Context, req Request, cb StreamCallback) error {
	if m == nil || m.adapter == nil {
		return fmt.Errorf("runtime: no adapter configured")
	}
	return m.adapter.Stream(ctx, req, cb)
}

// Registry maps backend keys to factories initialising adapters.
type Registry map[string]AdapterFactory

// AdapterFactory constructs a new adapter instance from configuration.
type AdapterFactory func(config.RuntimeConfig) (Adapter, error)

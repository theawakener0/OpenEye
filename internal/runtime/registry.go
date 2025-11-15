package runtime

import "OpenEye/internal/config"

// DefaultRegistry provides built-in runtime adapters.
var DefaultRegistry = Registry{}

// Register adds a new adapter factory to the default registry.
func Register(name string, factory AdapterFactory) {
	DefaultRegistry[name] = factory
}

// MustManager builds a manager from the default registry and panics on error.
func MustManager(cfg config.RuntimeConfig) *Manager {
	mgr, err := NewManager(cfg, DefaultRegistry)
	if err != nil {
		panic(err)
	}
	return mgr
}

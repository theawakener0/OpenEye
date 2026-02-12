# OpenEye Plugin System

OpenEye is designed with a modular, extensible architecture that allows developers to customize and extend virtually every component of the system. This guide provides comprehensive documentation for building plugins and extensions for OpenEye.

## What Are OpenEye Plugins?

OpenEye plugins are Go modules that integrate with the OpenEye framework to provide additional functionality. Unlike traditional plugin systems that use dynamic loading, OpenEye uses a **registry pattern** where plugins register themselves at initialization time. This approach provides:

- **Type safety**: Plugins implement well-defined Go interfaces
- **Clean integration**: Seamless configuration and lifecycle management
- **No runtime overhead**: Plugins are compiled into the main application
- **Familiar Go patterns**: Uses standard Go initialization and registration

## Extension Points Overview

OpenEye provides several extension points for customization:

| Extension Point | Interface | Purpose | Common Use Cases |
|----------------|-----------|---------|-----------------|
| Runtime Adapters | `runtime.Adapter` | Add new LLM backends | Custom HTTP endpoints, local models, APIs |
| Embedding Providers | `embedding.Provider` | Add embedding services | Vector databases, custom embeddings |
| Memory Engines | `memory.Store` | Customize storage | Alternative databases, distributed stores |
| RAG Retrievers | `rag.Retriever` | Customize retrieval | Domain-specific search, hybrid search |
| CLI Subcommands | `subcommands` | Add CLI commands | Custom workflows, administration tools |
| Image Processors | `image.Processor` | Customize image handling | Preprocessing, format conversion |

## Quick Start: Hello World Plugin

Here's a minimal example of a custom runtime adapter:

```go
package myadapter

import (
    "context"
    "fmt"
    "time"

    "OpenEye/internal/config"
    "OpenEye/internal/runtime"
)

func init() {
    runtime.Register("myadapter", newAdapter)
}

func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    return &Adapter{
        baseURL: cfg.HTTP.BaseURL,
        timeout: 30 * time.Second,
    }, nil
}

type Adapter struct {
    baseURL string
    timeout time.Duration
}

func (a *Adapter) Name() string { return "myadapter" }

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Implementation here
    return runtime.Response{Text: "Hello from my adapter!"}, nil
}

func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
    // Streaming implementation
    return nil
}

func (a *Adapter) Close() error {
    return nil
}
```

To use this adapter, configure it in your `openeye.yaml`:

```yaml
runtime:
  backend: "myadapter"
  http:
    base_url: "http://localhost:8080"
    timeout: "30s"
```

## Prerequisites

Before building OpenEye plugins, ensure you have:

- **Go 1.21 or later**: OpenEye requires Go 1.21+
- **Understanding of Go interfaces**: Plugin development relies heavily on interface implementation
- **Familiarity with OpenEye architecture**: Review the [Architecture Guide](architecture.md)
- **Go module setup**: Plugins should be Go modules

```bash
# Initialize a new plugin module
mkdir my-openeye-plugin
cd my-openeye-plugin
go mod init github.com/yourname/my-openeye-plugin

# Add OpenEye dependency
go get github.com/theawakener0/OpenEye
```

## Plugin Directory Structure

A well-organized plugin follows this structure:

```
my-openeye-plugin/
├── go.mod                    # Module definition
├── README.md                 # Plugin documentation
├── plugin.yaml              # Plugin manifest (optional)
├── internal/
│   └── adapter.go          # Plugin implementation
├── config/
│   └── config.go           # Configuration structures
└── examples/
    └── basic_usage.go      # Usage examples
```

## Using Your Plugin

There are two ways to use custom plugins:

### Option 1: Compile-Time Integration (Recommended for Core Plugins)

Create a wrapper package that imports both OpenEye and your plugin:

```go
package main

import (
    _ "github.com/yourname/my-openeye-plugin" // Registers the plugin
    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

func main() {
    cfg, _ := config.Resolve()
    pipe, _ := pipeline.New(cfg, runtime.DefaultRegistry)
    defer pipe.Close()
    // Use the pipeline...
}
```

### Option 2: Import and Register Explicitly

```go
package main

import (
    "github.com/yourname/my-openeye-plugin"
    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

func main() {
    // Plugin's init() already registered itself
    cfg, _ := config.Resolve()
    pipe, _ := pipeline.New(cfg, runtime.DefaultRegistry)
    defer pipe.Close()
}
```

## SDKs for Other Languages

Currently, OpenEye plugins must be written in Go due to the type-safe interface requirements. However, we are working on SDKs for other languages:

| Language | Status | Notes |
|----------|--------|-------|
| Python | Planned | Using cbindgen/CGO bindings |
| JavaScript/TypeScript | Planned | Node.js addon |
| Rust | Planned | Native Rust SDK |
| Java | Consideration | Requires evaluation |

For language SDKs, plugins would run as external services that OpenEye communicates with via HTTP or stdin/stdout protocol.

## Plugin Manifest

For more complex plugins, you can include a `plugin.yaml` manifest:

```yaml
name: my-openeye-plugin
version: 1.0.0
author: Your Name <your@email.com>
description: A custom OpenEye plugin for XYZ functionality

# OpenEye version compatibility
requires:
  openeye: ">=1.0.0,<2.0.0"

# Plugin capabilities
capabilities:
  - runtime-adapter

# Entry point (for future dynamic loading)
entry: ./plugin.so

# Configuration defaults
defaults:
  timeout: "30s"
  max_tokens: 512
```

## Next Steps

1. **[Architecture Guide](architecture.md)**: Understand the plugin system internals
2. **[Runtime Adapters](runtime-adapters.md)**: Build custom LLM backends (most common)
3. **[Embedding Providers](embedding-providers.md)**: Add custom embedding services
4. **[Best Practices](best-practices.md)**: Security, performance, and testing guidelines
5. **[Examples](examples/)**: Working code examples to learn from

## Common Patterns

All OpenEye plugins follow these patterns:

### Registration Pattern

Plugins register themselves during Go's `init()` function:

```go
func init() {
    runtime.Register("mybackend", adapterFactory)
}
```

### Configuration Pattern

Plugins define configuration structures that integrate with OpenEye's config system:

```go
type MyConfig struct {
    APIKey string `yaml:"api_key"`
    Timeout string `yaml:"timeout"`
}
```

### Error Handling Pattern

Plugins return structured errors that OpenEye can handle:

```go
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    if err := validateRequest(req); err != nil {
        return runtime.Response{}, fmt.Errorf("validation failed: %w", err)
    }
    // ... implementation
}
```

## Troubleshooting

### Plugin Not Found

If your plugin isn't being recognized:

1. Ensure the import statement is present in your main package
2. Check that `init()` is being called (use `fmt.Println("init called")` for debugging)
3. Verify the registry key matches your configuration

### Configuration Not Loading

Common issues:

1. YAML field names must match the config struct tags
2. Environment variables use `APP_*` prefix (see [Configuration](../overview.md#configuration))
3. Check for typos in configuration keys

### Initialization Failures

If your adapter fails to initialize:

1. Check that required fields are not empty
2. Validate network connectivity if applicable
3. Review logs for specific error messages

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share patterns
- **Examples**: See the [examples directory](examples/) for working code

## Related Documentation

- [Architecture Guide](architecture.md)
- [Runtime Adapters](runtime-adapters.md)
- [Embedding Providers](embedding-providers.md)
- [Memory Engines](memory-engines.md)
- [Best Practices](best-practices.md)
- [Distribution Guide](distribution.md)
- [Plugin Manifest](manifest.md)

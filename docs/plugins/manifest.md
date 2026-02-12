# Plugin Manifest

The plugin manifest (`plugin.yaml`) defines metadata, dependencies, and capabilities for OpenEye plugins. This document describes the manifest format and its usage.

## Overview

The plugin manifest is a YAML file that provides:

- Plugin identification (name, version, author)
- OpenEye version compatibility
- Plugin capabilities
- Dependencies declaration
- Security permissions
- Configuration defaults

## Manifest Format

```yaml
# Plugin manifest for OpenEye
apiVersion: openeye.io/v1
kind: Plugin

# Identification
name: my-openeye-plugin
version: 1.0.0
description: A custom runtime adapter for the Example LLM API

# Author information
author:
  name: Your Name
  email: your@email.com
  url: https://github.com/yourname

# OpenEye version requirements
requires:
  openeye: ">=1.0.0,<2.0.0"  # SemVer range

# Plugin capabilities
capabilities:
  - runtime-adapter
  - embedding-provider

# Optional dependencies on other plugins
dependencies:
  other-plugin: ">=1.0.0,<2.0.0"

# Security permissions model
permissions:
  - network
  - filesystem
  - environment

# Configuration defaults
defaults:
  timeout: "60s"
  max_tokens: 1024

# Entry point (for future dynamic loading)
entry: ./plugin.so

# Plugin metadata
keywords:
  - llm
  - custom
  - example
license: MIT
```

## Field Reference

### apiVersion

The manifest API version:

```yaml
apiVersion: openeye.io/v1
```

### kind

Must be `Plugin`:

```yaml
kind: Plugin
```

### name

Unique identifier for the plugin (lowercase, hyphenated):

```yaml
name: my-custom-adapter
```

### version

Semantic version of the plugin:

```yaml
version: 1.2.0
```

### description

Brief description of the plugin:

```yaml
description: A custom runtime adapter for the Example LLM API
```

### author

Author information:

```yaml
author:
  name: Your Name
  email: your@email.com
  url: https://github.com/yourname
```

### requires

OpenEye version requirements (SemVer range):

```yaml
requires:
  openeye: ">=1.0.0,<2.0.0"
```

Valid SemVer range formats:
- `"1.0.0"` - Exact version
- `">=1.0.0"` - Minimum version
- `">=1.0.0,<2.0.0"` - Range
- `">=1.0.0,!=1.5.0"` - Exclude version

### capabilities

List of plugin capabilities:

```yaml
capabilities:
  - runtime-adapter
  - embedding-provider
  - memory-engine
  - retriever
  - cli-extension
  - image-processor
```

Available capabilities:

| Capability | Description |
|------------|-------------|
| `runtime-adapter` | Provides a custom LLM backend |
| `embedding-provider` | Provides custom embeddings |
| `memory-engine` | Provides custom memory storage |
| `retriever` | Provides custom RAG retrieval |
| `cli-extension` | Provides CLI subcommands |
| `image-processor` | Provides custom image processing |

### dependencies

Optional plugin dependencies:

```yaml
dependencies:
  base-plugin: ">=1.0.0,<2.0.0"
  utilities: ">=1.1.0"
```

### permissions

Security permissions requested by the plugin:

```yaml
permissions:
  - network           # Make HTTP requests
  - filesystem       # Read/write files
  - environment     # Read environment variables
  - logging         # Write logs
```

Available permissions:

| Permission | Description |
|------------|-------------|
| `network` | Allow outbound HTTP requests |
| `filesystem` | Allow file system access |
| `environment` | Allow reading environment variables |
| `logging` | Allow writing logs |
| `metrics` | Allow exposing metrics |

### defaults

Configuration default values:

```yaml
defaults:
  timeout: "60s"
  max_tokens: 1024
  temperature: 0.7
```

### entry

Plugin entry point (for future dynamic loading):

```yaml
entry: ./plugin.so
```

### keywords

Searchable keywords:

```yaml
keywords:
  - llm
  - custom
  - api
  - adapter
```

### license

License identifier (SPDX format):

```yaml
license: MIT
```

## Example Manifests

### Runtime Adapter Plugin

```yaml
apiVersion: openeye.io/v1
kind: Plugin

name: example-llm-adapter
version: 1.0.0
description: Runtime adapter for the Example LLM API

author:
  name: Example Corp
  email: support@example.com
  url: https://example.com

requires:
  openeye: ">=1.0.0,<2.0.0"

capabilities:
  - runtime-adapter

permissions:
  - network
  - environment

defaults:
  timeout: "60s"
  max_tokens: 512
  temperature: 0.7

keywords:
  - llm
  - api
  - inference

license: Apache-2.0
```

### Embedding Provider Plugin

```yaml
apiVersion: openeye.io/v1
kind: Plugin

name: custom-embedding
version: 1.0.0
description: Custom embedding provider using Sentence Transformers

author:
  name: AI Research Lab
  email: hello@airesearch.dev

requires:
  openeye: ">=1.0.0,<2.0.0"

capabilities:
  - embedding-provider

dependencies:
  transformers-utils: ">=2.0.0"

permissions:
  - network
  - filesystem

defaults:
  dimension: 768
  timeout: "30s"

keywords:
  - embedding
  - vector
  - transformers

license: MIT
```

### CLI Extension Plugin

```yaml
apiVersion: openeye.io/v1
kind: Plugin

name: openeye-admin
version: 1.0.0
description: Administrative CLI tools for OpenEye

author:
  name: DevOps Team

requires:
  openeye: ">=1.0.0,<2.0.0"

capabilities:
  - cli-extension

permissions:
  - filesystem
  - logging

defaults:
  output: "text"

keywords:
  - admin
  - cli
  - management

license: MIT
```

## Manifest Validation

### Validation Rules

1. **Required fields:**
   - `apiVersion`
   - `kind`
   - `name`
   - `version`

2. **Format rules:**
   - `name`: lowercase, hyphenated, alphanumeric
   - `version`: valid SemVer
   - `license`: valid SPDX identifier

3. **Capability rules:**
   - Must reference valid capabilities
   - Entry point must exist (if specified)

### Validation Example

```go
package manifest "fmt"
    "os"

    "g

import (
   opkg.in/yaml.v3"
)

type Manifest struct {
    APIVersion  string            `yaml:"apiVersion"`
    Kind        string           `yaml:"kind"`
    Name        string           `yaml:"name"`
    Version     string           `yaml:"version"`
    Description string           `yaml:"description"`
    Author      Author           `yaml:"author"`
    Requires    Requires         `yaml:"requires"`
    Capabilities []string        `yaml:"capabilities"`
    Dependencies map[string]string `yaml:"dependencies"`
    Permissions  []string        `yaml:"permissions"`
    Defaults     map[string]string `yaml:"defaults"`
    Entry        string           `yaml:"entry"`
    Keywords     []string        `yaml:"keywords"`
    License      string           `yaml:"license"`
}

func Load(path string) (*Manifest, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("failed to read manifest: %w", err)
    }
    
    var manifest Manifest
    if err := yaml.Unmarshal(data, &manifest); err != nil {
        return nil, fmt.Errorf("failed to parse manifest: %w", err)
    }
    
    if err := manifest.Validate(); err != nil {
        return nil, fmt.Errorf("invalid manifest: %w", err)
    }
    
    return &manifest, nil
}

func (m *Manifest) Validate() error {
    // Check required fields
    if m.APIVersion == "" {
        return fmt.Errorf("apiVersion is required")
    }
    if m.Kind != "Plugin" {
        return fmt.Errorf("kind must be 'Plugin'")
    }
    if m.Name == "" {
        return fmt.Errorf("name is required")
    }
    if m.Version == "" {
        return fmt.Errorf("version is required")
    }
    
    // Validate capabilities
    validCapabilities := map[string]bool{
        "runtime-adapter": true,
        "embedding-provider": true,
        "memory-engine": true,
        "retriever": true,
        "cli-extension": true,
        "image-processor": true,
    }
    
    for _, cap := range m.Capabilities {
        if !validCapabilities[cap] {
            return fmt.Errorf("invalid capability: %s", cap)
        }
    }
    
    return nil
}
```

## Future Features

The manifest format supports future enhancements:

### Dynamic Loading

```yaml
# Future: Dynamic plugin loading
entry: ./plugin.so
load: lazy  # lazy | eager
```

### Version Pinning

```yaml
# Future: Pin specific OpenEye commit
requires:
  openeye: ">=1.0.0"
  commit: abc123def456
```

### Conditional Capabilities

```yaml
# Future: Platform-specific capabilities
capabilities:
  - runtime-adapter
  - embedding-provider:
      when: platform == "linux" && arch == "amd64"
```

## Related Documentation

- [Quick Start Guide](index.md)
- [Architecture Guide](architecture.md)
- [Runtime Adapters](runtime-adapters.md)
- [Distribution Guide](distribution.md)
- [Best Practices](best-practices.md)

# Plugin Distribution

This guide covers how to package, distribute, and version your OpenEye plugins for use by others.

## Distribution Methods

There are several ways to distribute OpenEye plugins:

1. **Go Modules** - Publish on GitHub/GitLab as a Go module
2. **Vendoring** - Include plugin source in your application
3. **Monorepo** - Keep plugins in the same repository as your application

## Go Module Distribution

### Publishing Your Plugin

#### Step 1: Create a Go Module

```bash
# Initialize your plugin as a Go module
mkdir my-openeye-plugin
cd my-openeye-plugin
go mod init github.com/yourname/my-openeye-plugin
```

#### Step 2: Configure go.mod

```go
module github.com/yourname/my-openeye-plugin

go 1.21

require (
    OpenEye v1.0.0 // Replace with actual version
)

replace OpenEye => ../OpenEye  // For local development
```

#### Step 3: Tag and Release

```bash
# Create a version tag
git tag v1.0.0
git push origin v1.0.0

# Create GitHub release
gh release create v1.0.0 --title "My OpenEye Plugin v1.0.0"
```

#### Step 4: Verify Publication

```bash
# List available versions
go list -m -versions github.com/yourname/my-openeye-plugin

# Download a specific version
go get github.com/yourname/my-openeye-plugin@v1.0.0
```

### Semantic Versioning

Follow Semantic Versioning (SemVer) for your plugins:

```
MAJOR.MINOR.PATCH

- MAJOR: Breaking changes to the interface
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)
```

Examples:
- `v1.0.0` - Initial release
- `v1.1.0` - Added new feature
- `v1.1.1` - Fixed bug
- `v2.0.0` - Breaking changes

### Version Compatibility

Declare OpenEye version compatibility:

```go
// go.mod
require (
    OpenEye v1.0.0 // Compatible with OpenEye 1.x
)

// For major version compatibility
require (
    OpenEye v2.0.0 // Use OpenEye v2.x only
)
```

## Vendoring Plugin

### Including Plugin Source

For applications that want to bundle plugins:

```
your-app/
├── go.mod
├── go.sum
├── main.go
├── internal/
│   └── plugins/
│       └── custom-adapter/
│           ├── adapter.go
│           └── config.go
└── vendor/
    └── github.com/
        └── yourname/
            └── my-openeye-plugin/
```

### Importing Vendored Plugin

```go
package main

import (
    _ "your-app/internal/plugins/custom-adapter"
    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

func main() {
    cfg, _ := config.Resolve()
    pipe, _ := pipeline.New(cfg, runtime.DefaultRegistry)
    defer pipe.Close()
}
```

## Monorepo Structure

For organizations with multiple plugins:

```
openeye-plugins/
├── plugins/
│   ├── custom-llm/
│   │   ├── go.mod
│   │   └── adapter.go
│   ├── custom-embedding/
│   │   ├── go.mod
│   │   └── provider.go
│   └── custom-memory/
│       ├── go.mod
│       └── store.go
├── go.mod  # Workspace root
└── go.work  # Go workspace
```

### Go Workspace Configuration

```go
// go.work
go 1.21

use (
    ./plugins/custom-llm
    ./plugins/custom-embedding
    ./plugins/custom-memory
)

replace OpenEye => ../OpenEye
```

## Distribution Best Practices

### 1. Clear Documentation

```markdown
# My OpenEye Plugin

A custom runtime adapter for the Example LLM API.

## Installation

```bash
go get github.com/yourname/my-plugin
```

## Usage

```yaml
runtime:
  backend: "my-plugin"
  http:
    base_url: "https://api.example.com"
    timeout: "60s"
```

## Configuration

| Field | Required | Description |
|-------|----------|-------------|
| base_url | Yes | API endpoint URL |
| timeout | No | Request timeout (default: 60s) |

## Environment Variables

- `MY_PLUGIN_API_KEY`: API authentication key

## Requirements

- Go 1.21+
- OpenEye v1.0.0+

## License

MIT
```

### 2. Automated Testing

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: '1.21'
          
      - name: Download dependencies
        run: go mod download
        
      - name: Run tests
        run: go test ./... -v
        
      - name: Run linter
        run: go vet ./...
```

### 3. Version Compatibility Matrix

| Plugin Version | OpenEye Version | Status |
|----------------|-----------------|--------|
| v1.0.x | v1.0.x | Supported |
| v1.1.x | v1.1.x | Supported |
| v2.0.x | v2.0.x | Supported |

### 4. Changelog

```markdown
# Changelog

## v1.1.0 (2024-01-15)

### Added
- Support for streaming responses
- New configuration option: `max_tokens`

### Changed
- Updated error handling for better diagnostics

## v1.0.0 (2024-01-01)

- Initial release
```

## Plugin Registry (Future)

A plugin registry allows discovery and distribution:

```yaml
# plugin.yaml (future format)
name: my-openeye-plugin
version: 1.1.0
description: Custom adapter for Example LLM

requires:
  openeye: ">=1.0.0,<2.0.0"

capabilities:
  - runtime-adapter

repository: https://github.com/yourname/my-openeye-plugin
license: MIT
author: Your Name <your@email.com>
```

## Installation Methods

### Method 1: Direct Import

```go
package main

import _ "github.com/yourname/my-plugin"  // Registers the plugin
```

### Method 2: Wrapper Package

```go
package yourapp

import (
    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
    "github.com/yourname/my-plugin"
)

func NewApp() (*pipeline.Pipeline, error) {
    cfg, _ := config.Resolve()
    return pipeline.New(cfg, runtime.DefaultRegistry)
}
```

### Method 3: Build Tags

```go
// +build myplugin

package main

import (
    _ "github.com/yourname/my-plugin"
)
```

```bash
# Build with plugin
go build -tags myplugin -o OpenEye .

# Build without plugin
go build -o OpenEye .
```

## Distribution Checklist

- [ ] Clear README with installation instructions
- [ ] Semantic versioning implemented
- [ ] Changelog maintained
- [ ] Automated tests passing
- [ ] Code linted
- [ ] Dependencies documented
- [ ] Version compatibility declared
- [ ] License included
- [ ] Security contact defined
- [ ] Automated releases configured

## Security Considerations

### 1. Supply Chain Security

```yaml
# .github/workflows/signed-release.yml
name: Signed Release

on:
  release:
    types: [created]

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build
        run: go build -o my-plugin .
        
      - name: Sign binary
        run: |
          echo "${{ secrets.GPG_PRIVATE_KEY }}" | gpg --import
          gpg --armor --detach-sign my-plugin
```

### 2. Dependency Management

```bash
# Use go.mod tidy to clean dependencies
go mod tidy

# Verify dependencies
go mod verify
```

### 3. Vulnerability Scanning

```bash
# Check for known vulnerabilities
go run golang.org/x/vuln/cmd/govulncheck@latest ./...
```

## Related Documentation

- [Quick Start Guide](index.md)
- [Architecture Guide](architecture.md)
- [Runtime Adapters](runtime-adapters.md)
- [Best Practices](best-practices.md)
- [Plugin Manifest](manifest.md)

# Hello World Adapter

This example demonstrates a minimal runtime adapter for OpenEye. It's a simple "Hello World" implementation that shows the essential structure.

## Overview

The adapter responds with a static message, demonstrating:
- Basic interface implementation
- Registration pattern
- Configuration handling
- Error handling

## Source Code

### hello_adapter.go

```go
package helloadapter

import (
    "context"
    "errors"
    "fmt"
    "time"

    "OpenEye/internal/config"
    "OpenEye/internal/runtime"
)

func init() {
    // Register the adapter with OpenEye
    runtime.Register("hello", newAdapter)
}

type Adapter struct {
    greeting string
    timeout time.Duration
}

func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    // Validate configuration
    if cfg.HTTP.BaseURL == "" {
        // Use default greeting if not specified
        cfg.HTTP.BaseURL = "World"
    }
    
    // Set default timeout
    timeout := 30 * time.Second
    if cfg.HTTP.Timeout != "" {
        var err error
        timeout, err = time.ParseDuration(cfg.HTTP.Timeout)
        if err != nil {
            return nil, fmt.Errorf("invalid timeout: %w", err)
        }
    }
    
    return &Adapter{
        greeting: cfg.HTTP.BaseURL,
        timeout:  timeout,
    }, nil
}

func (a *Adapter) Name() string {
    return "hello"
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Validate input
    if req.Prompt == "" {
        return runtime.Response{}, errors.New("prompt cannot be empty")
    }
    
    // Check context
    select {
    case <-ctx.Done():
        return runtime.Response{}, ctx.Err()
    default:
    }
    
    // Generate response
    response := fmt.Sprintf("Hello, %s! Your prompt was: %q", a.greeting, req.Prompt)
    
    return runtime.Response{
        Text: response,
        Stats: runtime.Stats{
            TokensGenerated: countTokens(response),
        },
    }, nil
}

func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
    // Validate input
    if req.Prompt == "" {
        return errors.New("prompt cannot be empty")
    }
    
    // For this simple adapter, stream the response word by word
    response := fmt.Sprintf("Hello, %s! Your prompt was: %q", a.greeting, req.Prompt)
    
    words := []string{"Hello,", a.greeting + "!", "Your", "prompt", "was:", req.Prompt}
    
    for i, word := range words {
        // Check context
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }
        
        // Emit token
        if err := cb(runtime.StreamEvent{
            Token: word + " ",
            Index: i,
        }); err != nil {
            return err
        }
    }
    
    // Signal completion
    return cb(runtime.StreamEvent{Final: true})
}

func (a *Adapter) Close() error {
    // No resources to clean up
    return nil
}

// Helper function to count tokens (approximate)
func countTokens(text string) int {
    if text == "" {
        return 0
    }
    // Rough approximation: 4 characters per token
    return (len(text) + 3) / 4
}
```

## Usage

### Step 1: Create Your Plugin

```bash
mkdir hello-adapter
cd hello-adapter
go mod init github.com/yourname/hello-adapter
```

Save the code above as `hello_adapter.go`.

### Step 2: Add Dependencies

```bash
go mod edit -require=OpenEye@v1.0.0
go mod tidy
```

### Step 3: Create Main Application

Create `main.go`:

```go
package main

import (
    "context"
    "fmt"
    "os"

    _ "github.com/yourname/hello-adapter"  // Import to register adapter
    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

func main() {
    // Load configuration
    cfg, err := config.Resolve()
    if err != nil {
        fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
        os.Exit(1)
    }
    
    // Create pipeline
    pipe, err := pipeline.New(cfg, runtime.DefaultRegistry)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Pipeline error: %v\n", err)
        os.Exit(1)
    }
    defer pipe.Close()
    
    // Create context
    ctx := context.Background()
    
    // Generate response
    result, err := pipe.Respond(ctx, "Test message", nil, pipeline.Options{})
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
    
    fmt.Println(result.Text)
}
```

### Step 4: Configure the Adapter

Create `openeye.yaml`:

```yaml
runtime:
  backend: "hello"
  http:
    base_url: "OpenEye User"  # Greeting name
    timeout: "30s"
  defaults:
    max_tokens: 100
    temperature: 0.7
```

### Step 5: Run

```bash
go run main.go
```

Expected output:
```
Hello, OpenEye User! Your prompt was: "Test message"
```

## Configuration Options

The hello adapter supports the following configuration:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `runtime.backend` | Yes | - | Must be `"hello"` |
| `runtime.http.base_url` | No | `"World"` | Name in greeting |
| `runtime.http.timeout` | No | `"30s"` | Request timeout |
| `runtime.defaults.max_tokens` | No | `100` | Max output tokens |

## Testing

### Unit Test

Create `hello_adapter_test.go`:

```go
package helloadapter

import (
    "context"
    "testing"

    "OpenEye/internal/runtime"
)

func TestAdapter_Generate(t *testing.T) {
    adapter, err := newAdapter(config.RuntimeConfig{
        Backend: "hello",
        HTTP: config.HTTPBackendConfig{
            BaseURL: "Test",
        },
    })
    if err != nil {
        t.Fatalf("Failed to create adapter: %v", err)
    }
    
    resp, err := adapter.Generate(context.Background(), runtime.Request{
        Prompt: "Hello",
    })
    if err != nil {
        t.Fatalf("Generate failed: %v", err)
    }
    
    expected := "Hello, Test! Your prompt was: \"Hello\""
    if resp.Text != expected {
        t.Errorf("Expected %q, got %q", expected, resp.Text)
    }
}

func TestAdapter_EmptyPrompt(t *testing.T) {
    adapter, _ := newAdapter(config.RuntimeConfig{})
    
    _, err := adapter.Generate(context.Background(), runtime.Request{
        Prompt: "",
    })
    if err == nil {
        t.Error("Expected error for empty prompt")
    }
}
```

### Run Tests

```bash
go test -v ./...
```

## Streaming Example

```go
package main

import (
    "context"
    "fmt"
    "os"

    _ "github.com/yourname/hello-adapter"
    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

func main() {
    cfg, _ := config.Resolve()
    pipe, _ := pipeline.New(cfg, runtime.DefaultRegistry)
    defer pipe.Close()
    
    ctx := context.Background()
    
    fmt.Println("Streaming response:")
    
    _, err := pipe.Respond(ctx, "Tell me a story", nil, pipeline.Options{
        Stream: true,
        StreamCallback: func(evt runtime.StreamEvent) error {
            if evt.Err != nil {
                return evt.Err
            }
            if evt.Final {
                fmt.Println("\n[End of stream]")
            } else {
                fmt.Print(evt.Token)
            }
            return nil
        },
    })
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
    }
}
```

## Extension Ideas

This adapter can be extended to:

1. **Custom greetings**: Use configuration to set different greetings
2. **Random responses**: Return different responses for variety
3. **Prompt analysis**: Analyze and categorize prompts
4. **Caching**: Cache responses for repeated prompts

```go
// Example: Add caching
type Adapter struct {
    greeting  string
    timeout   time.Duration
    cache     map[string]string
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Check cache
    if cached, ok := a.cache[req.Prompt]; ok {
        return runtime.Response{Text: cached}, nil
    }
    
    // Generate and cache
    response := fmt.Sprintf("Hello, %s!", a.greeting)
    a.cache[req.Prompt] = response
    
    return runtime.Response{Text: response}, nil
}
```

## Full Project Structure

```
hello-adapter/
├── go.mod
├── hello_adapter.go
├── hello_adapter_test.go
├── main.go
├── openeye.yaml
└── README.md
```

## Related Documentation

- [Runtime Adapters](../runtime-adapters.md)
- [Architecture Guide](../architecture.md)
- [Best Practices](../best-practices.md)

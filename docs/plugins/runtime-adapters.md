# Runtime Adapters

Runtime adapters are the most common extension point for OpenEye. They allow you to add support for new LLM backends, including custom HTTP APIs, local model servers, and specialized inference engines.

## Overview

A runtime adapter is a Go implementation of the `runtime.Adapter` interface that handles communication with an LLM backend. OpenEye includes two built-in adapters:

- **HTTP Adapter** (`internal/llmclient`): For HTTP-based LLM APIs
- **Native Adapter** (`internal/native`): For in-process llama.cpp models

This guide shows you how to create your own adapter.

## Interface Contract

All runtime adapters must implement the `runtime.Adapter` interface:

```go
package runtime

type Adapter interface {
    // Name returns a human-readable identifier for this adapter
    Name() string
    
    // Generate performs a single-shot completion request
    Generate(ctx context.Context, req Request) (Response, error)
    
    // Stream performs real-time token streaming
    Stream(ctx context.Context, req Request, cb StreamCallback) error
    
    // Close releases any resources held by the adapter
    Close() error
}
```

### Method Details

#### Name()

Returns a unique identifier for the adapter. This is used in configuration to select the adapter:

```go
func (a *Adapter) Name() string {
    return "my-custom-backend"
}
```

#### Generate()

Handles blocking completion requests:

```go
func (a *Adapter) Generate(ctx context.Context, req Request) (Response, error) {
    // Validate request
    if req.Prompt == "" {
        return Response{}, errors.New("prompt cannot be empty")
    }
    
    // Convert to backend-specific format
    apiReq := toAPIRequest(req)
    
    // Call the backend
    apiResp, err := a.client.Complete(ctx, apiReq)
    if err != nil {
        return Response{}, fmt.Errorf("API request failed: %w", err)
    }
    
    // Convert response
    return Response{
        Text:  apiResp.Text,
        Stats: Stats{TokensGenerated: apiResp.Usage.TotalTokens},
    }, nil
}
```

#### Stream()

Handles streaming responses. Must call the callback for each token:

```go
func (a *Adapter) Stream(ctx context.Context, req Request, cb StreamCallback) error {
    stream, err := a.client.StartStream(ctx, req.Prompt)
    if err != nil {
        return fmt.Errorf("failed to start stream: %w", err)
    }
    defer stream.Close()
    
    idx := 0
    for token := range stream.Tokens() {
        // Check for context cancellation
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }
        
        // Emit token
        if err := cb(StreamEvent{
            Token: token.Text,
            Index: idx,
        }); err != nil {
            return err
        }
        idx++
    }
    
    // Signal completion
    return cb(StreamEvent{Final: true})
}
```

#### Close()

Releases resources when the adapter is no longer needed:

```go
func (a *Adapter) Close() error {
    if a.client != nil {
        return a.client.Close()
    }
    return nil
}
```

## Complete Example: Custom HTTP Adapter

Here's a complete, production-ready example of a custom HTTP LLM adapter:

### Project Structure

```
custom-http-adapter/
├── go.mod
├── adapter.go
└── README.md
```

### adapter.go

```go
package customhttp

import (
    "bytes"
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "net/http"
    "strings"
    "sync"
    "time"

    "OpenEye/internal/config"
    "OpenEye/internal/runtime"
)

func init() {
    runtime.Register("custom-http", newAdapter)
}

type Adapter struct {
    baseURL    string
    apiKey     string
    model      string
    timeout    time.Duration
    client     *http.Client
    defaults   runtime.GenerationOptions
   mu         sync.RWMutex
}

func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    baseURL := strings.TrimSpace(cfg.HTTP.BaseURL)
    if baseURL == "" {
        return nil, errors.New("custom-http: base_url is required")
    }
    
    if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
        baseURL = "https://" + baseURL
    }
    
    timeout := 60 * time.Second
    if cfg.HTTP.Timeout != "" {
        var err error
        timeout, err = time.ParseDuration(cfg.HTTP.Timeout)
        if err != nil {
            return nil, fmt.Errorf("custom-http: invalid timeout %q: %w", cfg.HTTP.Timeout, err)
        }
    }
    
    return &Adapter{
        baseURL:  baseURL,
        apiKey:   os.Getenv("CUSTOM_HTTP_API_KEY"),
        model:    cfg.HTTP.Model,
        timeout:  timeout,
        client: &http.Client{
            Timeout: timeout,
            Transport: &http.Transport{
                MaxIdleConns:        10,
                MaxIdleConnsPerHost: 5,
                IdleConnTimeout:     90 * time.Second,
            },
        },
        defaults: runtime.GenerationOptions{
            MaxTokens:     cfg.Defaults.MaxTokens,
            Temperature:   cfg.Defaults.Temperature,
            TopK:          cfg.Defaults.TopK,
            TopP:          cfg.Defaults.TopP,
            MinP:          cfg.Defaults.MinP,
            RepeatPenalty: cfg.Defaults.RepeatPenalty,
            RepeatLastN:   cfg.Defaults.RepeatLastN,
            Stop:          cfg.Defaults.Stop,
        },
    }, nil
}

func (a *Adapter) Name() string {
    return "custom-http"
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    opts := mergeOptions(a.defaults, req.Options)
    
    apiReq := APIRequest{
        Model:       a.model,
        Prompt:      req.Prompt,
        MaxTokens:   opts.MaxTokens,
        Temperature: opts.Temperature,
        TopP:        opts.TopP,
        Stop:        opts.Stop,
    }
    
    body, err := json.Marshal(apiReq)
    if err != nil {
        return runtime.Response{}, fmt.Errorf("failed to marshal request: %w", err)
    }
    
    httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
        fmt.Sprintf("%s/v1/completions", a.baseURL), bytes.NewReader(body))
    if err != nil {
        return runtime.Response{}, fmt.Errorf("failed to create request: %w", err)
    }
    
    a.setHeaders(httpReq)
    
    httpResp, err := a.client.Do(httpReq)
    if err != nil {
        return runtime.Response{}, fmt.Errorf("request failed: %w", err)
    }
    defer httpResp.Body.Close()
    
    if httpResp.StatusCode != http.StatusOK {
        respBody, _ := io.ReadAll(httpResp.Body)
        return runtime.Response{}, fmt.Errorf("API error (status %d): %s", httpResp.StatusCode, respBody)
    }
    
    var apiResp APIResponse
    if err := json.NewDecoder(httpResp.Body).Decode(&apiResp); err != nil {
        return runtime.Response{}, fmt.Errorf("failed to decode response: %w", err)
    }
    
    if len(apiResp.Choices) == 0 {
        return runtime.Response{}, errors.New("no choices returned")
    }
    
    return runtime.Response{
        Text:  strings.TrimSpace(apiResp.Choices[0].Text),
        Stats: runtime.Stats{TokensGenerated: apiResp.Usage.TotalTokens},
    }, nil
}

func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
    opts := mergeOptions(a.defaults, req.Options)
    
    apiReq := APIRequest{
        Model:       a.model,
        Prompt:      req.Prompt,
        MaxTokens:   opts.MaxTokens,
        Temperature: opts.Temperature,
        TopP:        opts.TopP,
        Stop:        opts.Stop,
        Stream:      true,
    }
    
    body, err := json.Marshal(apiReq)
    if err != nil {
        return fmt.Errorf("failed to marshal request: %w", err)
    }
    
    httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
        fmt.Sprintf("%s/v1/completions", a.baseURL), bytes.NewReader(body))
    if err != nil {
        return fmt.Errorf("failed to create request: %w", err)
    }
    
    a.setHeaders(httpReq)
    
    httpResp, err := a.client.Do(httpReq)
    if err != nil {
        return fmt.Errorf("request failed: %w", err)
    }
    defer httpResp.Body.Close()
    
    if httpResp.StatusCode != http.StatusOK {
        respBody, _ := io.ReadAll(httpResp.Body)
        return fmt.Errorf("API error (status %d): %s", httpResp.StatusCode, respBody)
    }
    
    decoder := json.NewDecoder(httpResp)
    idx := 0
    for {
        var chunk APIChunk
        if err := decoder.Decode(&chunk); err != nil {
            if err == io.EOF {
                break
            }
            return fmt.Errorf("failed to decode chunk: %w", err)
        }
        
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }
        
        if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
            if err := cb(runtime.StreamEvent{
                Token: chunk.Choices[0].Delta.Content,
                Index: idx,
            }); err != nil {
                return err
            }
            idx++
        }
    }
    
    return cb(runtime.StreamEvent{Final: true})
}

func (a *Adapter) Close() error {
    a.client.CloseIdleConnections()
    return nil
}

func (a *Adapter) setHeaders(req *http.Request) {
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")
    if a.apiKey != "" {
        req.Header.Set("Authorization", "Bearer "+a.apiKey)
    }
}

func mergeOptions(base, override runtime.GenerationOptions) runtime.GenerationOptions {
    result := base
    
    if override.MaxTokens != 0 {
        result.MaxTokens = override.MaxTokens
    }
    if override.Temperature != 0 {
        result.Temperature = override.Temperature
    }
    if override.TopP != 0 {
        result.TopP = override.TopP
    }
    if override.TopK != 0 {
        result.TopK = override.TopK
    }
    if override.MinP != 0 {
        result.MinP = override.MinP
    }
    if override.RepeatPenalty != 0 {
        result.RepeatPenalty = override.RepeatPenalty
    }
    if override.RepeatLastN != 0 {
        result.RepeatLastN = override.RepeatLastN
    }
    if len(override.Stop) > 0 {
        result.Stop = append([]string(nil), override.Stop...)
    }
    
    return result
}

// API types for the custom HTTP backend
type APIRequest struct {
    Model       string    `json:"model"`
    Prompt      string    `json:"prompt"`
    MaxTokens   int       `json:"max_tokens,omitempty"`
    Temperature float64   `json:"temperature,omitempty"`
    TopP        float64   `json:"top_p,omitempty"`
    Stop        []string  `json:"stop,omitempty"`
    Stream      bool      `json:"stream,omitempty"`
}

type APIResponse struct {
    Choices []struct {
        Text string `json:"text"`
    } `json:"choices"`
    Usage struct {
        TotalTokens int `json:"total_tokens"`
    } `json:"usage"`
}

type APIChunk struct {
    Choices []struct {
        Delta struct {
            Content string `json:"content"`
        } `json:"delta"`
    } `json:"choices"`
}
```

## Configuration Integration

Your adapter integrates with OpenEye's configuration system through the `config.RuntimeConfig` structure:

```yaml
runtime:
  backend: "custom-http"
  http:
    base_url: "https://api.example.com"
    timeout: "60s"
  defaults:
    max_tokens: 1024
    temperature: 0.7
    top_p: 0.95
```

### Accessing Configuration

```go
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    // HTTP backend configuration
    baseURL := cfg.HTTP.BaseURL
    timeout := cfg.HTTP.Timeout
    
    // Generation defaults
    maxTokens := cfg.Defaults.MaxTokens
    temperature := cfg.Defaults.Temperature
    
    // ... implementation
}
```

### Environment Variable Overrides

Users can override configuration via environment variables:

```bash
export APP_LLM_BASEURL="https://api.example.com"
export APP_LLM_TIMEOUT="60s"
export APP_LLM_MAXTOKENS=1024
```

## Security Considerations

Security is critical when building runtime adapters. Follow these guidelines:

### 1. Input Validation

Always validate user inputs:

```go
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Validate prompt length
    if len(req.Prompt) > a.maxPromptLength {
        return runtime.Response{}, fmt.Errorf("prompt exceeds maximum length of %d", a.maxPromptLength)
    }
    
    // Check for empty or whitespace-only prompts
    if strings.TrimSpace(req.Prompt) == "" {
        return runtime.Response{}, errors.New("prompt cannot be empty or whitespace only")
    }
    
    // Validate temperature range
    if req.Options.Temperature < 0 || req.Options.Temperature > 2 {
        return runtime.Response{}, errors.New("temperature must be between 0 and 2")
    }
    
    // ... continue
}
```

### 2. Mandatory Timeouts

Always enforce timeouts to prevent hanging requests:

```go
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    timeout := 30 * time.Second
    if cfg.HTTP.Timeout != "" {
        var err error
        timeout, err = time.ParseDuration(cfg.HTTP.Timeout)
        if err != nil {
            return nil, fmt.Errorf("invalid timeout: %w", err)
        }
    }
    
    // Enforce maximum timeout
    if timeout > 5*time.Minute {
        timeout = 5 * time.Minute
    }
    
    return &Adapter{
        timeout: timeout,
        client: &http.Client{
            Timeout: timeout,
        },
    }, nil
}
```

### 3. SSRF Prevention

Prevent server-side request forgery attacks:

```go
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Parse and validate the base URL
    parsedURL, err := url.Parse(a.baseURL)
    if err != nil {
        return runtime.Response{}, fmt.Errorf("invalid base URL: %w", err)
    }
    
    // Only allow HTTPS in production
    if parsedURL.Scheme != "https" && parsedURL.Scheme != "http" {
        return runtime.Response{}, fmt.Errorf("unsupported URL scheme: %s", parsedURL.Scheme)
    }
    
    // Block private IP ranges (basic protection)
    if parsedURL.Hostname() != "" {
        host := parsedURL.Hostname()
        if ip := net.ParseIP(host); ip != nil {
            if isPrivateIP(ip) {
                return runtime.Response{}, errors.New("private IP addresses are not allowed")
            }
        }
    }
    
    // ... continue
}

func isPrivateIP(ip net.IP) bool {
    privateRanges := []net.IPNet{
        {IP: net.ParseIP("10.0.0.0"), Mask: net.CIDRMask(8, 32)},
        {IP: net.ParseIP("172.16.0.0"), Mask: net.CIDRMask(12, 32)},
        {IP: net.ParseIP("192.168.0.0"), Mask: net.CIDRMask(16, 32)},
        {IP: net.ParseIP("127.0.0.0"), Mask: net.CIDRMask(8, 32)},
    }
    
    for _, r := range privateRanges {
        if r.Contains(ip) {
            return true
        }
    }
    return false
}
```

### 4. TLS Certificate Validation

Always use secure TLS connections:

```go
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    return &Adapter{
        client: &http.Client{
            Timeout: timeout,
            Transport: &http.Transport{
                TLSClientConfig: &tls.Config{
                    MinVersion: tls.VersionTLS12,
                    // DO NOT disable certificate validation in production
                    // InsecureSkipVerify: true,  // NEVER DO THIS
                },
            },
        },
    }, nil
}
```

### 5. API Key Security

Never hardcode API keys:

```go
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    // Prefer environment variable
    apiKey := os.Getenv("CUSTOM_HTTP_API_KEY")
    
    // Check for configuration (less secure)
    if apiKey == "" {
        apiKey = cfg.HTTP.APIKey
    }
    
    // Warn if no API key is set
    if apiKey == "" {
        log.Printf("[custom-http] WARNING: No API key configured")
    }
    
    return &Adapter{apiKey: apiKey}, nil
}
```

### 6. Rate Limiting

Implement rate limiting to prevent abuse:

```go
type Adapter struct {
    // ... other fields
    rateLimiter   *rate.Limiter
    requests      []time.Time
}

func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    // Allow 10 requests per second with burst of 20
    limiter := rate.NewLimiter(10, 20)
    
    return &Adapter{
        rateLimiter: limiter,
        requests:    make([]time.Time, 0, 100),
    }, nil
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Check rate limit
    if err := a.rateLimiter.Wait(ctx); err != nil {
        return runtime.Response{}, fmt.Errorf("rate limit exceeded: %w", err)
    }
    
    // Track request time
    a.mu.Lock()
    a.requests = append(a.requests, time.Now())
    // Clean up old requests
    cutoff := time.Now().Add(-time.Minute)
    for len(a.requests) > 0 && a.requests[0].Before(cutoff) {
        a.requests = a.requests[1:]
    }
    a.mu.Unlock()
    
    // ... continue with request
}
```

### 7. Request Size Limits

Prevent large request attacks:

```go
const (
    maxPromptLength    = 100000  // 100KB
    maxRequestBodySize  = 1024 * 1024  // 1MB
)

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Check prompt size
    if len(req.Prompt) > maxPromptLength {
        return runtime.Response{}, fmt.Errorf("prompt exceeds maximum size of %d bytes", maxPromptLength)
    }
    
    // ... continue
}
```

## Streaming Implementation

Here's a more detailed streaming implementation:

```go
func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
    // Build request
    body := buildStreamRequest(req)
    
    // Create SSE request
    httpReq, err := http.NewRequestWithContext(ctx, "POST",
        fmt.Sprintf("%s/v1/completions", a.baseURL), bytes.NewReader(body))
    if err != nil {
        return fmt.Errorf("failed to create request: %w", err)
    }
    
    // Make request
    httpResp, err := a.client.Do(httpReq)
    if err != nil {
        return fmt.Errorf("request failed: %w", err)
    }
    defer httpResp.Body.Close()
    
    if httpResp.StatusCode != http.StatusOK {
        respBody, _ := io.ReadAll(httpResp.Body)
        return fmt.Errorf("API error (status %d): %s", httpResp.StatusCode, respBody)
    }
    
    // Parse SSE stream
    reader := bufio.NewReader(httpResp.Body)
    idx := 0
    
    for {
        line, err := reader.ReadString('\n')
        if err != nil {
            if err == io.EOF {
                break
            }
            return fmt.Errorf("stream read error: %w", err)
        }
        
        // Parse SSE format
        line = strings.TrimSpace(line)
        if !strings.HasPrefix(line, "data: ") {
            continue
        }
        
        data := strings.TrimPrefix(line, "data: ")
        if data == "[DONE]" {
            break
        }
        
        // Parse JSON chunk
        var chunk CompletionChunk
        if err := json.Unmarshal([]byte(data), &chunk); err != nil {
            log.Printf("[custom-http] Failed to parse chunk: %v", err)
            continue
        }
        
        // Extract token
        if len(chunk.Choices) > 0 {
            token := chunk.Choices[0].Delta.Content
            
            // Emit via callback
            if err := cb(runtime.StreamEvent{
                Token: token,
                Index: idx,
            }); err != nil {
                return err
            }
            idx++
        }
    }
    
    // Signal completion
    return cb(runtime.StreamEvent{Final: true})
}
```

## Testing Your Adapter

### Unit Testing

```go
package customhttp

import (
    "context"
    "net/http"
    "net/http/httptest"
    "testing"

    "OpenEye/internal/runtime"
)

func TestAdapter_Generate(t *testing.T) {
    // Create mock server
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{"choices":[{"text":"Hello!"}],"usage":{"total_tokens":2}}`))
    }))
    defer server.Close()
    
    // Create adapter
    adapter, err := newAdapter(config.RuntimeConfig{
        Backend: "custom-http",
        HTTP: config.HTTPBackendConfig{
            BaseURL: server.URL,
        },
    })
    if err != nil {
        t.Fatalf("Failed to create adapter: %v", err)
    }
    
    // Test generation
    resp, err := adapter.Generate(context.Background(), runtime.Request{
        Prompt: "Say hello",
    })
    if err != nil {
        t.Fatalf("Generate failed: %v", err)
    }
    
    if resp.Text != "Hello!" {
        t.Errorf("Expected 'Hello!', got '%s'", resp.Text)
    }
}
```

### Integration Testing

```go
func TestAdapter_Integration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    adapter, err := newAdapter(config.RuntimeConfig{
        Backend: "custom-http",
        HTTP: config.HTTPBackendConfig{
            BaseURL: "https://api.example.com",
        },
    })
    if err != nil {
        t.Fatalf("Failed to create adapter: %v", err)
    }
    defer adapter.Close()
    
    resp, err := adapter.Generate(context.Background(), runtime.Request{
        Prompt: "What is 2+2?",
    })
    if err != nil {
        t.Fatalf("Generate failed: %v", err)
    }
    
    if !strings.Contains(resp.Text, "4") {
        t.Errorf("Expected answer to contain '4', got '%s'", resp.Text)
    }
}
```

## Using Your Adapter

### Step 1: Import the Plugin

In your main application:

```go
package main

import (
    _ "github.com/yourname/custom-http-adapter" // Registers the adapter
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

### Step 2: Configure the Adapter

In your `openeye.yaml`:

```yaml
runtime:
  backend: "custom-http"
  http:
    base_url: "https://api.example.com/v1"
    timeout: "60s"
  defaults:
    max_tokens: 1024
    temperature: 0.7
```

### Step 3: Set API Key

```bash
export CUSTOM_HTTP_API_KEY="your-api-key"
openeye chat --message "Hello, world!"
```

## Complete Working Example

See [Custom HTTP Adapter Example](examples/custom-http-llm.md) for a full implementation.

## Common Issues

### Context Cancellation

Always respect context cancellation:

```go
func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
    stream, err := a.client.StartStream(req.Prompt)
    if err != nil {
        return err
    }
    
    for token := range stream.Tokens() {
        // Critical: Check context
        select {
        case <-ctx.Done():
            stream.Close()
            return ctx.Err()
        default:
        }
        
        if err := cb(runtime.StreamEvent{Token: token}); err != nil {
            return err
        }
    }
    
    return nil
}
```

### Memory Leaks

Always clean up resources:

```go
func (a *Adapter) Close() error {
    if a.client != nil {
        a.client.CloseIdleConnections()
    }
    if a.cache != nil {
        a.cache.Close()
    }
    return nil
}
```

## Checklist for Production

- [ ] Input validation implemented
- [ ] Mandatory timeouts configured
- [ ] SSRF protection enabled
- [ ] TLS certificate validation enabled
- [ ] API keys stored securely (env vars)
- [ ] Rate limiting implemented
- [ ] Request size limits enforced
- [ ] Context cancellation respected
- [ ] Proper error handling
- [ ] Resource cleanup in Close()
- [ ] Unit tests passing
- [ ] Integration tests passing

## Related Documentation

- [Quick Start Guide](index.md)
- [Architecture Guide](architecture.md)
- [Embedding Providers](embedding-providers.md)
- [Best Practices](best-practices.md)
- [Security Considerations](best-practices.md#security-guidelines)

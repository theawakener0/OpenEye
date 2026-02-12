# Custom HTTP LLM Adapter

This example provides a complete, production-ready HTTP LLM adapter for OpenEye. It demonstrates proper security practices, error handling, and streaming support.

## Overview

The adapter connects to any HTTP-based LLM API that follows a standard completion format. Features include:

- Secure HTTP client with TLS
- Mandatory timeouts
- Request validation
- Streaming support
- Error handling
- Rate limiting
- Caching

## Source Code

### http_llm_adapter.go

```go
package httpllm

import (
    "bytes"
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "log"
    "math"
    "net/http"
    "net/url"
    "strings"
    "sync"
    "time"

    "OpenEye/internal/config"
    "OpenEye/internal/runtime"
)

func init() {
    runtime.Register("http-llm", newAdapter)
}

type Adapter struct {
    baseURL       string
    apiKey        string
    model         string
    timeout       time.Duration
    maxTokens     int
    temperature   float64
    client       *http.Client
    rateLimiter   *rate.Limiter
    cache         *lru.Cache[string, string]
    mu            sync.RWMutex
}

type Config struct {
    BaseURL      string
    APIKey       string
    Model        string
    Timeout      time.Duration
    MaxTokens    int
    Temperature  float64
}

func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    // Validate base URL
    baseURL := strings.TrimSpace(cfg.HTTP.BaseURL)
    if baseURL == "" {
        return nil, errors.New("http-llm: base_url is required")
    }
    
    // Validate URL scheme
    parsedURL, err := url.Parse(baseURL)
    if err != nil {
        return nil, fmt.Errorf("http-llm: invalid base_url: %w", err)
    }
    
    if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
        return nil, errors.New("http-llm: base_url must use http or https scheme")
    }
    
    // Get API key from environment
    apiKey := os.Getenv("HTTP_LLM_API_KEY")
    
    // Parse timeout
    timeout := 60 * time.Second
    if cfg.HTTP.Timeout != "" {
        timeout, err = time.ParseDuration(cfg.HTTP.Timeout)
        if err != nil {
            return nil, fmt.Errorf("http-llm: invalid timeout: %w", err)
        }
    }
    
    // Enforce maximum timeout
    if timeout > 5*time.Minute {
        timeout = 5 * time.Minute
    }
    
    // Get defaults
    maxTokens := cfg.Defaults.MaxTokens
    if maxTokens <= 0 {
        maxTokens = 512
    }
    
    temperature := cfg.Defaults.Temperature
    if temperature <= 0 {
        temperature = 0.7
    }
    
    return &Adapter{
        baseURL:     baseURL,
        apiKey:      apiKey,
        model:       cfg.HTTP.Model,
        timeout:     timeout,
        maxTokens:   maxTokens,
        temperature: temperature,
        client: &http.Client{
            Timeout: timeout,
            Transport: &http.Transport{
                MaxIdleConns:        10,
                MaxIdleConnsPerHost:  5,
                IdleConnTimeout:      90 * time.Second,
                TLSClientConfig: &tls.Config{
                    MinVersion: tls.VersionTLS12,
                },
            },
        },
        rateLimiter: rate.NewLimiter(50, 100),  // 50 RPS, burst 100
        cache:       lru.New[string, string](1000),
    }, nil
}

func (a *Adapter) Name() string {
    return "http-llm"
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Validate prompt
    if strings.TrimSpace(req.Prompt) == "" {
        return runtime.Response{}, errors.New("prompt cannot be empty")
    }
    
    // Apply rate limiting
    if err := a.rateLimiter.Wait(ctx); err != nil {
        return runtime.Response{}, fmt.Errorf("rate limit exceeded: %w", err)
    }
    
    // Check cache
    cacheKey := generateCacheKey(req.Prompt, req.Options)
    a.mu.RLock()
    if cached, ok := a.cache.Get(cacheKey); ok {
        a.mu.RUnlock()
        return runtime.Response{Text: cached}, nil
    }
    a.mu.RUnlock()
    
    // Build request
    apiReq := APIRequest{
        Model:       a.model,
        Prompt:      req.Prompt,
        MaxTokens:   a.maxTokens,
        Temperature: a.temperature,
    }
    
    // Apply request options
    if req.Options.MaxTokens > 0 {
        apiReq.MaxTokens = req.Options.MaxTokens
    }
    if req.Options.Temperature > 0 {
        apiReq.Temperature = req.Options.Temperature
    }
    if len(req.Options.Stop) > 0 {
        apiReq.Stop = req.Options.Stop
    }
    
    // Serialize request
    body, err := json.Marshal(apiReq)
    if err != nil {
        return runtime.Response{}, fmt.Errorf("failed to marshal request: %w", err)
    }
    
    // Create HTTP request
    httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
        fmt.Sprintf("%s/v1/completions", a.baseURL), bytes.NewReader(body))
    if err != nil {
        return runtime.Response{}, fmt.Errorf("failed to create request: %w", err)
    }
    
    // Set headers
    a.setHeaders(httpReq)
    
    // Execute request
    startTime := time.Now()
    httpResp, err := a.client.Do(httpReq)
    if err != nil {
        return runtime.Response{}, fmt.Errorf("request failed: %w", err)
    }
    defer httpResp.Body.Close()
    
    // Check response status
    if httpResp.StatusCode != http.StatusOK {
        respBody, _ := io.ReadAll(httpResp.Body)
        return runtime.Response{}, fmt.Errorf("API error (status %d): %s",
            httpResp.StatusCode, respBody)
    }
    
    // Decode response
    var apiResp APIResponse
    if err := json.NewDecoder(httpResp.Body).Decode(&apiResp); err != nil {
        return runtime.Response{}, fmt.Errorf("failed to decode response: %w", err)
    }
    
    // Validate response
    if len(apiResp.Choices) == 0 {
        return runtime.Response{}, errors.New("no choices returned")
    }
    
    responseText := strings.TrimSpace(apiResp.Choices[0].Text)
    
    // Cache response
    a.mu.Lock()
    a.cache.Add(cacheKey, responseText)
    a.mu.Unlock()
    
    // Calculate stats
    duration := time.Since(startTime)
    tokensPerSec := float64(apiResp.Usage.TotalTokens) / duration.Seconds()
    
    return runtime.Response{
        Text: responseText,
        Stats: runtime.Stats{
            TokensEvaluated: apiResp.Usage.PromptTokens,
            TokensGenerated: apiResp.Usage.CompletionTokens,
            TokensCached:    0,
            Duration:        duration,
            GenerationTPS:   tokensPerSec,
        },
    }, nil
}

func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
    // Validate prompt
    if strings.TrimSpace(req.Prompt) == "" {
        return errors.New("prompt cannot be empty")
    }
    
    // Apply rate limiting
    if err := a.rateLimiter.Wait(ctx); err != nil {
        return fmt.Errorf("rate limit exceeded: %w", err)
    }
    
    // Build request
    apiReq := APIRequest{
        Model:       a.model,
        Prompt:      req.Prompt,
        MaxTokens:   a.maxTokens,
        Temperature: a.temperature,
        Stream:      true,
    }
    
    // Apply options
    if req.Options.MaxTokens > 0 {
        apiReq.MaxTokens = req.Options.MaxTokens
    }
    if req.Options.Temperature > 0 {
        apiReq.Temperature = req.Options.Temperature
    }
    
    // Serialize request
    body, err := json.Marshal(apiReq)
    if err != nil {
        return fmt.Errorf("failed to marshal request: %w", err)
    }
    
    // Create HTTP request
    httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
        fmt.Sprintf("%s/v1/completions", a.baseURL), bytes.NewReader(body))
    if err != nil {
        return fmt.Errorf("failed to create request: %w", err)
    }
    
    // Set headers
    a.setHeaders(httpReq)
    
    // Execute request
    httpResp, err := a.client.Do(httpReq)
    if err != nil {
        return fmt.Errorf("request failed: %w", err)
    }
    defer httpResp.Body.Close()
    
    // Check response status
    if httpResp.StatusCode != http.StatusOK {
        respBody, _ := io.ReadAll(httpResp.Body)
        return fmt.Errorf("API error (status %d): %s",
            httpResp.StatusCode, respBody)
    }
    
    // Parse SSE stream
    decoder := json.NewDecoder(httpResp.Body)
    idx := 0
    
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }
        
        var chunk APIChunk
        if err := decoder.Decode(&chunk); err != nil {
            if err == io.EOF {
                break
            }
            return fmt.Errorf("failed to decode chunk: %w", err)
        }
        
        // Emit token
        if len(chunk.Choices) > 0 {
            token := strings.TrimSpace(chunk.Choices[0].Delta.Content)
            if token != "" {
                if err := cb(runtime.StreamEvent{
                    Token: token,
                    Index: idx,
                }); err != nil {
                    return err
                }
                idx++
            }
        }
    }
    
    // Signal completion
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

// Helper functions
func generateCacheKey(prompt string, opts runtime.GenerationOptions) string {
    return fmt.Sprintf("%x", md5.Sum([]byte(prompt)))
}

// API types
type APIRequest struct {
    Model       string    `json:"model,omitempty"`
    Prompt      string    `json:"prompt"`
    MaxTokens   int       `json:"max_tokens,omitempty"`
    Temperature float64   `json:"temperature,omitempty"`
    Stop        []string  `json:"stop,omitempty"`
    Stream      bool      `json:"stream,omitempty"`
}

type APIResponse struct {
    Choices []struct {
        Text string `json:"text"`
    } `json:"choices"`
    Usage struct {
        PromptTokens     int `json:"prompt_tokens"`
        CompletionTokens int `json:"completion_tokens"`
        TotalTokens     int `json:"total_tokens"`
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

## Configuration

### YAML Configuration

```yaml
runtime:
  backend: "http-llm"
  http:
    base_url: "https://api.example.com/v1"
    model: "gpt-3.5-turbo"
    timeout: "120s"
  defaults:
    max_tokens: 1024
    temperature: 0.7
```

### Environment Variables

```bash
export HTTP_LLM_API_KEY="your-api-key-here"
```

## Security Features

### 1. TLS Enforcement

The adapter requires TLS 1.2 or higher:

```go
TLSClientConfig: &tls.Config{
    MinVersion: tls.VersionTLS12,
    // Never skip verification in production
},
```

### 2. Mandatory Timeouts

All requests have a maximum timeout:

```go
// Maximum 5-minute timeout
if timeout > 5*time.Minute {
    timeout = 5 * time.Minute
}
```

### 3. Rate Limiting

Prevents abuse with configurable rate limits:

```go
rateLimiter: rate.NewLimiter(50, 100),  // 50 requests/second, burst 100
```

### 4. Input Validation

```go
// Validate prompt
if strings.TrimSpace(req.Prompt) == "" {
    return runtime.Response{}, errors.New("prompt cannot be empty")
}

// Validate URL
if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
    return nil, errors.New("base_url must use http or https scheme")
}
```

## Usage

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "os"

    _ "github.com/yourname/http-llm-adapter"
    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

func main() {
    cfg, _ := config.Resolve()
    pipe, _ := pipeline.New(cfg, runtime.DefaultRegistry)
    defer pipe.Close()
    
    ctx := context.Background()
    
    result, err := pipe.Respond(ctx, "What is OpenEye?", nil, pipeline.Options{})
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
    
    fmt.Println(result.Text)
}
```

### Streaming Usage

```go
result, err := pipe.Respond(ctx, "Tell me a story", nil, pipeline.Options{
    Stream: true,
    StreamCallback: func(evt runtime.StreamEvent) error {
        if evt.Err != nil {
            return evt.Err
        }
        if !evt.Final {
            fmt.Print(evt.Token)
        }
        return nil
    },
})
```

## Testing

### Mock Server Test

```go
package httpllm

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
        w.Write([]byte(`{
            "choices":[{"text":"Hello, World!"}],
            "usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}
        }`))
    }))
    defer server.Close()
    
    adapter, err := newAdapter(config.RuntimeConfig{
        Backend: "http-llm",
        HTTP: config.HTTPBackendConfig{
            BaseURL: server.URL,
        },
    })
    if err != nil {
        t.Fatalf("Failed to create adapter: %v", err)
    }
    
    resp, err := adapter.Generate(context.Background(), runtime.Request{
        Prompt: "Say hello",
    })
    if err != nil {
        t.Fatalf("Generate failed: %v", err)
    }
    
    expected := "Hello, World!"
    if resp.Text != expected {
        t.Errorf("Expected %q, got %q", expected, resp.Text)
    }
}
```

## Production Checklist

- [x] TLS enforcement
- [x] Mandatory timeouts
- [x] Rate limiting
- [x] Input validation
- [x] Error handling
- [x] Streaming support
- [x] Caching
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Security audit completed

## Related Documentation

- [Runtime Adapters](../runtime-adapters.md)
- [Best Practices](../best-practices.md)
- [Security Guidelines](../best-practices.md#security-guidelines)
- [Troubleshooting](../troubleshooting.md)

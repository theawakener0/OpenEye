# Embedding Providers

Embedding providers enable semantic search and similarity capabilities in OpenEye. They convert text into vector representations used by RAG, memory systems, and retrieval components.

## Overview

An embedding provider is a Go implementation of the `embedding.Provider` interface. OpenEye uses embeddings for:

- **RAG Retrieval**: Finding semantically relevant documents
- **Vector Memory**: Storing and retrieving conversation context
- **Semantic Search**: Finding similar text passages
- **Fact Retrieval**: Matching queries to stored facts

The built-in embedding provider uses llama.cpp's embedding endpoint. This guide shows you how to create custom providers.

## Interface Contract

All embedding providers must implement the `embedding.Provider` interface:

```go
package embedding

// Provider exposes semantic embedding capabilities
type Provider interface {
    // Embed converts text to a vector embedding
    Embed(ctx context.Context, text string) ([]float32, error)
    
    // Close releases resources held by the provider
    Close() error
}
```

### Method Details

#### Embed()

Converts text to a vector representation:

```go
func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    // Validate input
    if strings.TrimSpace(text) == "" {
        return nil, errors.New("text cannot be empty")
    }
    
    // Truncate if necessary
    if len(text) > maxTextLength {
        text = text[:maxTextLength]
    }
    
    // Call the embedding API
    response, err := p.client.GetEmbedding(text)
    if err != nil {
        return nil, fmt.Errorf("embedding request failed: %w", err)
    }
    
    // Normalize vector
    return normalize(response.Vector), nil
}
```

#### Close()

Releases resources when the provider is no longer needed:

```go
func (p *Provider) Close() error {
    if p.client != nil {
        return p.client.Close()
    }
    return nil
}
```

## Complete Example: Custom Embedding Provider

Here's a complete example of a custom HTTP-based embedding provider:

### Project Structure

```
custom-embedding/
├── go.mod
├── provider.go
└── README.md
```

### provider.go

```go
package customembedding

import (
    "bytes"
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "math"
    "net/http"
    "os"
    "strings"
    "sync"
    "time"

    "OpenEye/internal/config"
    "OpenEye/internal/embedding"
)

const (
    defaultDimension = 768
    maxTextLength    = 8192
)

func init() {
    embedding.RegisterProvider("custom", newProvider)
}

type Provider struct {
    baseURL     string
    apiKey      string
    model       string
    dimension   int
    timeout     time.Duration
    client      *http.Client
    mu          sync.RWMutex
}

func newProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
    if cfg.Enabled == nil || !*cfg.Enabled {
        return nil, nil
    }
    
    baseURL := strings.TrimSpace(cfg.LlamaCpp.BaseURL)
    if baseURL == "" {
        return nil, errors.New("custom: base_url is required")
    }
    
    if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
        baseURL = "https://" + baseURL
    }
    
    timeout := 30 * time.Second
    if cfg.LlamaCpp.Timeout != "" {
        var err error
        timeout, err = time.ParseDuration(cfg.LlamaCpp.Timeout)
        if err != nil {
            return nil, fmt.Errorf("custom: invalid timeout %q: %w", cfg.LlamaCpp.Timeout, err)
        }
    }
    
    dimension := defaultDimension
    if cfg.LlamaCpp.Dimension > 0 {
        dimension = cfg.LlamaCpp.Dimension
    }
    
    return &Provider{
        baseURL:   baseURL,
        apiKey:    os.Getenv("CUSTOM_EMBEDDING_API_KEY"),
        model:     cfg.LlamaCpp.Model,
        dimension: dimension,
        timeout:   timeout,
        client: &http.Client{
            Timeout: timeout,
            Transport: &http.Transport{
                MaxIdleConns:        10,
                MaxIdleConnsPerHost: 5,
                IdleConnTimeout:     90 * time.Second,
            },
        },
    }, nil
}

func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    if strings.TrimSpace(text) == "" {
        return make([]float32, p.dimension), nil
    }
    
    if len(text) > maxTextLength {
        text = text[:maxTextLength]
    }
    
    request := EmbeddingRequest{
        Model: p.model,
        Input: text,
    }
    
    body, err := json.Marshal(request)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }
    
    httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
        fmt.Sprintf("%s/v1/embeddings", p.baseURL), bytes.NewReader(body))
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }
    
    p.setHeaders(httpReq)
    
    httpResp, err := p.client.Do(httpReq)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer httpResp.Body.Close()
    
    if httpResp.StatusCode != http.StatusOK {
        respBody, _ := io.ReadAll(httpResp.Body)
        return nil, fmt.Errorf("API error (status %d): %s", httpResp.StatusCode, respBody)
    }
    
    var resp EmbeddingResponse
    if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }
    
    if len(resp.Data) == 0 {
        return nil, errors.New("no embeddings returned")
    }
    
    vector := resp.Data[0].Embedding
    
    if len(vector) != p.dimension {
        if len(vector) > p.dimension {
            vector = vector[:p.dimension]
        } else {
            padded := make([]float32, p.dimension)
            copy(padded, vector)
            vector = padded
        }
    }
    
    return vector, nil
}

func (p *Provider) Close() error {
    p.client.CloseIdleConnections()
    return nil
}

func (p *Provider) setHeaders(req *http.Request) {
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")
    if p.apiKey != "" {
        req.Header.Set("Authorization", "Bearer "+p.apiKey)
    }
}

// Helper function to normalize vectors
func normalize(v []float32) []float32 {
    var norm float32
    for _, val := range v {
        norm += val * val
    }
    norm = float32(math.Sqrt(float64(norm)))
    if norm == 0 {
        return v
    }
    for i := range v {
        v[i] /= norm
    }
    return v
}

// API types for the embedding backend
type EmbeddingRequest struct {
    Model string `json:"model"`
    Input string `json:"input"`
}

type EmbeddingResponse struct {
    Data []struct {
        Embedding []float32 `json:"embedding"`
    } `json:"data"`
    Usage struct {
        TotalTokens int `json:"total_tokens"`
    } `json:"usage"`
}
```

## Configuration Integration

Your embedding provider integrates with OpenEye's configuration system:

```yaml
embedding:
  enabled: true
  backend: "custom"
  llamacpp:
    base_url: "https://api.example.com"
    model: "text-embedding-ada-002"
    dimension: 768
    timeout: "30s"
```

### Accessing Configuration

```go
func newProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
    // Check if enabled
    if cfg.Enabled == nil || !*cfg.Enabled {
        return nil, nil
    }
    
    // Access configuration
    baseURL := cfg.LlamaCpp.BaseURL
    model := cfg.LlamaCpp.Model
    dimension := cfg.LlamaCpp.Dimension
    timeout := cfg.LlamaCpp.Timeout
    
    // ... implementation
}
```

### Environment Variable Overrides

Users can override configuration via environment variables:

```bash
export APP_EMBEDDING_ENABLED=true
export APP_EMBEDDING_BACKEND=custom
export APP_EMBEDDING_BASEURL="https://api.example.com"
export APP_EMBEDDING_MODEL="text-embedding-ada-002"
export APP_EMBEDDING_TIMEOUT="30s"
```

## Vector Requirements

OpenEye has specific requirements for embedding vectors:

### Dimensionality

- **Default dimension**: 768 (from `all-MiniLM-L6-v2`)
- **Must be consistent**: All embeddings from a provider must have the same dimension
- **Padding/Padding**: Providers should pad or truncate vectors to match configured dimension

```go
func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    vector, err := p.getEmbedding(text)
    if err != nil {
        return nil, err
    }
    
    // Ensure correct dimension
    if len(vector) > p.dimension {
        vector = vector[:p.dimension]
    } else if len(vector) < p.dimension {
        padded := make([]float32, p.dimension)
        copy(padded, vector)
        vector = padded
    }
    
    return vector, nil
}
```

### Normalization

Vectors should be L2-normalized for cosine similarity calculations:

```go
func normalize(v []float32) []float32 {
    var norm float32
    for _, val := range v {
        norm += val * val
    }
    norm = float32(math.Sqrt(float64(norm)))
    if norm == 0 {
        return v
    }
    for i := range v {
        v[i] /= norm
    }
    return v
}
```

## Batch Embedding

While the interface only supports single text embedding, providers can batch internally:

```go
func (p *Provider) Embed(ctx context.Context, texts []string) ([][]float32, error) {
    // Batch multiple texts into a single request if supported
    request := BatchEmbeddingRequest{
        Model: p.model,
        Inputs: texts,
    }
    
    // Make single API call
    resp, err := p.client.GetBatchEmbeddings(request)
    if err != nil {
        return nil, err
    }
    
    vectors := make([][]float32, len(resp.Data))
    for i, data := range resp.Data {
        vectors[i] = normalize(data.Embedding)
    }
    
    return vectors, nil
}
```

## Security Considerations

Security is critical for embedding providers. Follow these guidelines:

### 1. Input Validation

```go
func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    // Check for empty input
    if strings.TrimSpace(text) == "" {
        return make([]float32, p.dimension), nil
    }
    
    // Validate text length
    if len(text) > maxTextLength {
        text = text[:maxTextLength]
    }
    
    // Check for null bytes and other suspicious characters
    if strings.Contains(text, "\x00") {
        return nil, errors.New("invalid input: contains null bytes")
    }
    
    return p.getEmbedding(text)
}
```

### 2. Mandatory Timeouts

```go
func newProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
    timeout := 30 * time.Second
    if cfg.LlamaCpp.Timeout != "" {
        var err error
        timeout, err = time.ParseDuration(cfg.LlamaCpp.Timeout)
        if err != nil {
            return nil, fmt.Errorf("invalid timeout: %w", err)
        }
    }
    
    // Enforce maximum timeout
    if timeout > 5*time.Minute {
        timeout = 5 * time.Minute
    }
    
    return &Provider{
        client: &http.Client{
            Timeout: timeout,
        },
    }, nil
}
```

### 3. Network Security

```go
func newProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
    // Validate HTTPS for production
    if !strings.HasPrefix(cfg.LlamaCpp.BaseURL, "https://") {
        log.Printf("[custom-embedding] WARNING: Using HTTP instead of HTTPS")
    }
    
    return &Provider{
        client: &http.Client{
            Timeout: timeout,
            Transport: &http.Transport{
                TLSClientConfig: &tls.Config{
                    MinVersion: tls.VersionTLS12,
                },
            },
        },
    }, nil
}
```

### 4. API Key Security

```go
func newProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
    // Prefer environment variable
    apiKey := os.Getenv("CUSTOM_EMBEDDING_API_KEY")
    
    // Check for configuration (less secure)
    if apiKey == "" {
        apiKey = cfg.LlamaCpp.APIKey
    }
    
    // Warn if no API key is set
    if apiKey == "" {
        log.Printf("[custom-embedding] WARNING: No API key configured")
    }
    
    return &Provider{apiKey: apiKey}, nil
}
```

### 5. Rate Limiting

```go
type Provider struct {
    rateLimiter *rate.Limiter
}

func newProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
    // Allow 50 requests per second with burst of 100
    limiter := rate.NewLimiter(50, 100)
    
    return &Provider{rateLimiter: limiter}, nil
}

func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    if err := p.rateLimiter.Wait(ctx); err != nil {
        return nil, fmt.Errorf("rate limit exceeded: %w", err)
    }
    
    // ... continue
}
```

## Caching Strategy

Embedding computation can be expensive. Implement caching:

```go
type Provider struct {
    cache     *lru.Cache[string, []float32]
    mu        sync.RWMutex
}

func newProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
    cache, _ := lru.New[string, []float32](1000)  // Cache 1000 embeddings
    
    return &Provider{
        cache: cache,
    }, nil
}

func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    // Check cache first
    p.mu.RLock()
    if cached, ok := p.cache.Get(text); ok {
        p.mu.RUnlock()
        return cached, nil
    }
    p.mu.RUnlock()
    
    // Generate embedding
    vector, err := p.getEmbedding(text)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    p.mu.Lock()
    p.cache.Add(text, vector)
    p.mu.Unlock()
    
    return vector, nil
}
```

## Error Handling

Proper error handling is essential:

```go
func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    if text == "" {
        return make([]float32, p.dimension), nil
    }
    
    response, err := p.client.GetEmbedding(text)
    if err != nil {
        // Check for specific error types
        if strings.Contains(err.Error(), "rate limit") {
            return nil, fmt.Errorf("embedding rate limit exceeded: %w", err)
        }
        if strings.Contains(err.Error(), "timeout") {
            return nil, fmt.Errorf("embedding request timed out: %w", err)
        }
        return nil, fmt.Errorf("embedding request failed: %w", err)
    }
    
    if len(response.Vector) == 0 {
        return nil, errors.New("empty embedding returned")
    }
    
    return normalize(response.Vector), nil
}
```

## Using Your Provider

### Step 1: Import the Provider

```go
package main

import (
    _ "github.com/yourname/custom-embedding"  // Registers the provider
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

### Step 2: Configure the Provider

In your `openeye.yaml`:

```yaml
embedding:
  enabled: true
  backend: "custom"
  llamacpp:
    base_url: "https://api.example.com/v1"
    model: "text-embedding-ada-002"
    dimension: 768
    timeout: "30s"

rag:
  enabled: true
  max_chunks: 4
  chunk_size: 512
```

### Step 3: Set API Key

```bash
export CUSTOM_EMBEDDING_API_KEY="your-api-key"
openeye chat --message "Search for similar documents"
```

## Testing Your Provider

### Unit Testing

```go
package customembedding

import (
    "context"
    "net/http"
    "net/http/httptest"
    "testing"

    "OpenEye/internal/config"
    "OpenEye/internal/embedding"
)

func TestProvider_Embed(t *testing.T) {
    // Create mock server
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3]}]}`))
    }))
    defer server.Close()
    
    enabled := true
    provider, err := newProvider(config.EmbeddingConfig{
        Enabled: &enabled,
        LlamaCpp: config.LlamaCppEmbeddingConfig{
            BaseURL: server.URL,
        },
    })
    if err != nil {
        t.Fatalf("Failed to create provider: %v", err)
    }
    
    vector, err := provider.Embed(context.Background(), "test text")
    if err != nil {
        t.Fatalf("Embed failed: %v", err)
    }
    
    if len(vector) != 3 {
        t.Errorf("Expected vector of length 3, got %d", len(vector))
    }
}

func TestProvider_EmptyText(t *testing.T) {
    enabled := true
    provider, _ := newProvider(config.EmbeddingConfig{
        Enabled: &enabled,
        LlamaCpp: config.LlamaCppEmbeddingConfig{
            Dimension: 768,
        },
    })
    
    vector, err := provider.Embed(context.Background(), "")
    if err != nil {
        t.Fatalf("Embed failed for empty text: %v", err)
    }
    
    if len(vector) != 768 {
        t.Errorf("Expected zero vector of length 768, got %d", len(vector))
    }
}
```

## Complete Working Example

See [Custom Embedding Provider Example](examples/custom-embedding.md) for a full implementation.

## Common Issues

### Vector Dimension Mismatch

Ensure all vectors have the same dimension:

```go
func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    vector, err := p.getEmbedding(text)
    if err != nil {
        return nil, err
    }
    
    // Verify and adjust dimension
    if len(vector) != p.dimension {
        return nil, fmt.Errorf("embedding dimension mismatch: expected %d, got %d",
            p.dimension, len(vector))
    }
    
    return vector, nil
}
```

### Slow Embeddings

Use caching for frequently embedded text:

```go
type Provider struct {
    cache *lru.Cache[string, []float32]
}

func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    // Check cache
    if cached, ok := p.cache.Get(text); ok {
        return cached, nil
    }
    
    // Generate and cache
    vector, err := p.getEmbedding(text)
    if err != nil {
        return nil, err
    }
    
    p.cache.Add(text, vector)
    return vector, nil
}
```

## Checklist for Production

- [ ] Vector dimension is consistent
- [ ] Vectors are L2-normalized
- [ ] Empty text handling implemented
- [ ] Input validation implemented
- [ ] Mandatory timeouts configured
- [ ] Network security (HTTPS, TLS)
- [ ] API key security (env vars)
- [ ] Rate limiting implemented
- [ ] Error handling complete
- [ ] Resource cleanup in Close()
- [ ] Unit tests passing
- [ ] Integration tests passing

## Related Documentation

- [Quick Start Guide](index.md)
- [Architecture Guide](architecture.md)
- [Runtime Adapters](runtime-adapters.md)
- [RAG Retrieval](custom-retrievers.md)
- [Best Practices](best-practices.md)

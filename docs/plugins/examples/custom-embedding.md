# Custom Embedding Provider

This example demonstrates how to create a custom embedding provider for OpenEye. It connects to any HTTP-based embedding API and provides vector embeddings for text.

## Overview

The embedding provider:

- Converts text to vector embeddings
- Validates input and handles errors
- Supports configurable dimensions
- Implements caching for performance
- Normalizes vectors for cosine similarity

## Source Code

### embedding_provider.go

```go
package customembedding

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
    "os"
    "strings"
    "sync"
    "time"

    "OpenEye/internal/config"
    "OpenEye/internal/embedding"
)

const (
    defaultDimension = 768
    maxTextLength   = 8192
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
    cache       *lru.Cache[string, []float32]
    mu          sync.RWMutex
}

type Config struct {
    BaseURL    string
    APIKey     string
    Model      string
    Dimension  int
    Timeout    time.Duration
}

func newProvider(cfg config.EmbeddingConfig) (embedding.Provider, error) {
    // Check if enabled
    if cfg.Enabled == nil || !*cfg.Enabled {
        return nil, nil
    }
    
    // Validate base URL
    baseURL := strings.TrimSpace(cfg.LlamaCpp.BaseURL)
    if baseURL == "" {
        return nil, errors.New("custom: base_url is required")
    }
    
    // Validate URL scheme
    parsedURL, err := url.Parse(baseURL)
    if err != nil {
        return nil, fmt.Errorf("custom: invalid base_url: %w", err)
    }
    
    if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
        return nil, errors.New("custom: base_url must use http or https scheme")
    }
    
    // Get API key from environment
    apiKey := os.Getenv("CUSTOM_EMBEDDING_API_KEY")
    
    // Parse timeout
    timeout := 30 * time.Second
    if cfg.LlamaCpp.Timeout != "" {
        timeout, err = time.ParseDuration(cfg.LlamaCpp.Timeout)
        if err != nil {
            return nil, fmt.Errorf("custom: invalid timeout: %w", err)
        }
    }
    
    // Set dimension
    dimension := defaultDimension
    if cfg.LlamaCpp.Dimension > 0 {
        dimension = cfg.LlamaCpp.Dimension
    }
    
    // Warn if no API key
    if apiKey == "" {
        log.Printf("[custom-embedding] WARNING: No API key configured")
    }
    
    return &Provider{
        baseURL:   baseURL,
        apiKey:    apiKey,
        model:     cfg.LlamaCpp.Model,
        dimension: dimension,
        timeout:   timeout,
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
        cache: lru.New[string, []float32](10000),
    }, nil
}

func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    // Handle empty text
    if strings.TrimSpace(text) == "" {
        return make([]float32, p.dimension), nil
    }
    
    // Truncate long text
    if len(text) > maxTextLength {
        text = text[:maxTextLength]
    }
    
    // Check cache
    cacheKey := generateCacheKey(text)
    p.mu.RLock()
    if cached, ok := p.cache.Get(cacheKey); ok {
        p.mu.RUnlock()
        return cached, nil
    }
    p.mu.RUnlock()
    
    // Build request
    apiReq := EmbeddingRequest{
        Model: p.model,
        Input: text,
    }
    
    // Serialize request
    body, err := json.Marshal(apiReq)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }
    
    // Create HTTP request
    httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
        fmt.Sprintf("%s/v1/embeddings", p.baseURL), bytes.NewReader(body))
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }
    
    // Set headers
    p.setHeaders(httpReq)
    
    // Execute request
    httpResp, err := p.client.Do(httpReq)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer httpResp.Body.Close()
    
    // Check response status
    if httpResp.StatusCode != http.StatusOK {
        respBody, _ := io.ReadAll(httpResp.Body)
        return nil, fmt.Errorf("API error (status %d): %s",
            httpResp.StatusCode, respBody)
    }
    
    // Decode response
    var apiResp EmbeddingResponse
    if err := json.NewDecoder(httpResp.Body).Decode(&apiResp); err != nil {
        return nil, fmt.Errorf("failed to decode response: %w", err)
    }
    
    // Validate response
    if len(apiResp.Data) == 0 {
        return nil, errors.New("no embeddings returned")
    }
    
    vector := apiResp.Data[0].Embedding
    
    // Adjust dimension
    vector = p.adjustDimension(vector)
    
    // Normalize vector
    vector = normalize(vector)
    
    // Cache result
    p.mu.Lock()
    p.cache.Add(cacheKey, vector)
    p.mu.Unlock()
    
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

func (p *Provider) adjustDimension(vector []float32) []float32 {
    if len(vector) == p.dimension {
        return vector
    }
    
    if len(vector) > p.dimension {
        return vector[:p.dimension]
    }
    
    // Pad with zeros
    padded := make([]float32, p.dimension)
    copy(padded, vector)
    return padded
}

// Normalize vector to unit length (L2 normalization)
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

func generateCacheKey(text string) string {
    return fmt.Sprintf("%x", md5.Sum([]byte(text)))
}

// API types
type EmbeddingRequest struct {
    Model string `json:"model,omitempty"`
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

## Configuration

### YAML Configuration

```yaml
embedding:
  enabled: true
  backend: "custom"
  llamacpp:
    base_url: "https://api.example.com/v1"
    model: "text-embedding-ada-002"
    dimension: 768
    timeout: "30s"
```

### Environment Variables

```bash
export CUSTOM_EMBEDDING_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "os"

    _ "github.com/yourname/custom-embedding"
    "OpenEye/internal/config"
    "OpenEye/internal/pipeline"
    "OpenEye/internal/runtime"
)

func main() {
    cfg, _ := config.Resolve()
    
    // Create pipeline (which uses the embedding provider)
    pipe, _ := pipeline.New(cfg, runtime.DefaultRegistry)
    defer pipe.Close()
    
    // The embedding provider is used automatically by:
    // - RAG retrieval
    // - Vector memory
    // - Omem/Mem0
    
    ctx := context.Background()
    
    // Enable RAG to use embeddings
    cfg.RAG.Enabled = true
    
    result, err := pipe.Respond(ctx, "Search for information", nil, pipeline.Options{})
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
    
    fmt.Println(result.Text)
}
```

### Direct Embedding Usage

```go
package main

import (
    "context"
    "fmt"

    _ "github.com/yourname/custom-embedding"
    "OpenEye/internal/config"
    "OpenEye/internal/embedding"
)

func main() {
    cfg := config.Default()
    enabled := true
    cfg.Embedding.Enabled = &enabled
    
    provider, err := embedding.New(cfg.Embedding)
    if err != nil {
        fmt.Printf("Failed to create provider: %v\n", err)
        return
    }
    defer provider.Close()
    
    ctx := context.Background()
    
    // Get embedding
    vector, err := provider.Embed(ctx, "Hello, World!")
    if err != nil {
        fmt.Printf("Failed to embed: %v\n", err)
        return
    }
    
    fmt.Printf("Embedding dimension: %d\n", len(vector))
    fmt.Printf("First 5 values: %v\n", vector[:5])
}
```

## Performance Optimization

### Cache Configuration

```go
// LRU cache with 10,000 entries
cache: lru.New[string, []float32](10000)
```

### Batch Embedding (if supported by API)

```go
func (p *Provider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
    // Batch texts into groups
    batchSize := 100
    var allVectors [][]float32
    
    for i := 0; i < len(texts); i += batchSize {
        end := min(i+batchSize, len(texts))
        batch := texts[i:end]
        
        vectors, err := p.embedBatch(ctx, batch)
        if err != nil {
            return nil, err
        }
        
        allVectors = append(allVectors, vectors...)
    }
    
    return allVectors, nil
}
```

## Testing

### Unit Test

```go
package customembedding

import (
    "context"
    "net/http"
    "net/http/httptest"
    "testing"

    "OpenEye/internal/config"
)

func TestProvider_Embed(t *testing.T) {
    // Create mock server
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        w.Write([]byte(`{
            "data":[{"embedding":[0.1,0.2,0.3,0.4]}]
        }`))
    }))
    defer server.Close()
    
    enabled := true
    provider, err := newProvider(config.EmbeddingConfig{
        Enabled: &enabled,
        LlamaCpp: config.LlamaCppEmbeddingConfig{
            BaseURL:    server.URL,
            Dimension: 4,
        },
    })
    if err != nil {
        t.Fatalf("Failed to create provider: %v", err)
    }
    
    vector, err := provider.Embed(context.Background(), "test text")
    if err != nil {
        t.Fatalf("Embed failed: %v", err)
    }
    
    if len(vector) != 4 {
        t.Errorf("Expected vector length 4, got %d", len(vector))
    }
}

func TestProvider_EmptyText(t *testing.T) {
    enabled := true
    provider, _ := newProvider(config.EmbeddingConfig{
        Enabled:   &enabled,
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

## Production Checklist

- [x] TLS enforcement
- [x] Mandatory timeouts
- [x] Input validation
- [x] Error handling
- [x] Caching
- [x] Vector normalization
- [x] Dimension adjustment
- [ ] Unit tests passing
- [ ] Integration tests passing

## Related Documentation

- [Embedding Providers](../embedding-providers.md)
- [Best Practices](../best-practices.md)
- [Security Guidelines](../best-practices.md#security-guidelines)
- [RAG Retrieval](../custom-retrievers.md)

# Best Practices

This document provides comprehensive guidelines for building secure, performant, and maintainable OpenEye plugins.

## Security Guidelines

Security is critical when building plugins that may handle sensitive data, make network requests, or interact with external systems.

### Input Validation

Always validate all inputs from users or external sources:

```go
// GOOD: Validate all inputs
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Validate prompt length
    if len(req.Prompt) > maxPromptLength {
        return runtime.Response{}, fmt.Errorf("prompt exceeds maximum length of %d", maxPromptLength)
    }
    
    // Check for empty or whitespace-only prompts
    if strings.TrimSpace(req.Prompt) == "" {
        return runtime.Response{}, errors.New("prompt cannot be empty")
    }
    
    // Validate generation options
    if req.Options.Temperature < 0 || req.Options.Temperature > 2 {
        return runtime.Response{}, errors.New("temperature must be between 0 and 2")
    }
    
    return a.client.Generate(req.Prompt)
}

// BAD: No validation
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    return a.client.Generate(req.Prompt)  // Blindly accepts any input
}
```

### Network Security

#### TLS Configuration

```go
// GOOD: Enforce TLS 1.2+
func newClient() *http.Client {
    return &http.Client{
        Timeout: 30 * time.Second,
        Transport: &http.Transport{
            TLSClientConfig: &tls.Config{
                MinVersion: tls.VersionTLS12,
                // Never disable certificate verification in production
                // InsecureSkipVerify: true,  // NEVER DO THIS
            },
        },
    }
}
```

#### SSRF Prevention

Prevent server-side request forgery attacks:

```go
func (a *Adapter) makeRequest(targetURL string) error {
    parsedURL, err := url.Parse(targetURL)
    if err != nil {
        return fmt.Errorf("invalid URL: %w", err)
    }
    
    // Only allow HTTPS
    if parsedURL.Scheme != "https" && parsedURL.Scheme != "http" {
        return errors.New("only HTTP and HTTPS schemes are allowed")
    }
    
    // Block private IP ranges
    if host := parsedURL.Hostname(); host != "" {
        if ip := net.ParseIP(host); ip != nil {
            if isPrivateIP(ip) {
                return errors.New("private IP addresses are not allowed")
            }
        }
    }
    
    return nil
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

### API Key Security

Never hardcode credentials:

```go
// GOOD: Prefer environment variables
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    apiKey := os.Getenv("MY_ADAPTER_API_KEY")
    
    // Check config as fallback (less secure)
    if apiKey == "" {
        apiKey = cfg.HTTP.APIKey
    }
    
    if apiKey == "" {
        return nil, errors.New("API key is required")
    }
    
    return &Adapter{apiKey: apiKey}, nil
}

// BAD: Hardcoded credentials
var apiKey = "sk-secret-key-12345"  // NEVER DO THIS
```

### Rate Limiting

Implement rate limiting to prevent abuse:

```go
type Adapter struct {
    rateLimiter *rate.Limiter
}

func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    // Allow 100 requests per second with burst of 200
    limiter := rate.NewLimiter(100, 200)
    
    return &Adapter{rateLimiter: limiter}, nil
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Check rate limit
    if err := a.rateLimiter.Wait(ctx); err != nil {
        return runtime.Response{}, fmt.Errorf("rate limit exceeded: %w", err)
    }
    
    return a.client.Generate(req.Prompt)
}
```

### Request Size Limits

Prevent large request attacks:

```go
const (
    maxPromptLength   = 100000  // 100KB
    maxRequestSize   = 1024 * 1024  // 1MB
    maxOutputTokens  = 4096
)

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Check prompt size
    if len(req.Prompt) > maxPromptLength {
        return runtime.Response{}, fmt.Errorf("prompt exceeds maximum size of %d bytes", maxPromptLength)
    }
    
    // Check output token limit
    if req.Options.MaxTokens > maxOutputTokens {
        return runtime.Response{}, fmt.Errorf("max_tokens exceeds maximum of %d", maxOutputTokens)
    }
    
    return a.client.Generate(req.Prompt)
}
```

## Performance Guidelines

### Connection Pooling

```go
func newHTTPClient() *http.Client {
    return &http.Client{
        Timeout: 30 * time.Second,
        Transport: &http.Transport{
            MaxIdleConns:        100,
            MaxIdleConnsPerHost:  10,
            IdleConnTimeout:      90 * time.Second,
            MaxConnsPerHost:      100,
        },
    }
}
```

### Caching

```go
type CachedProvider struct {
    provider embedding.Provider
    cache   *lru.Cache[string, []float32]
}

func (p *CachedProvider) Embed(ctx context.Context, text string) ([]float32, error) {
    // Check cache
    if cached, ok := p.cache.Get(text); ok {
        return cached, nil
    }
    
    // Generate
    vector, err := p.provider.Embed(ctx, text)
    if err != nil {
        return nil, err
    }
    
    // Cache result
    p.cache.Add(text, vector)
    
    return vector, nil
}
```

### Batch Processing

```go
func (p *Provider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
    // Batch into groups to avoid overwhelming the API
    batchSize := 100
    var allVectors [][]float32
    
    for i := 0; i < len(texts); i += batchSize {
        end := min(i+batchSize, len(texts))
        batch := texts[i:end]
        
        vectors, err := p.embedAPI(batch)
        if err != nil {
            return nil, fmt.Errorf("batch %d-%d failed: %w", i, end, err)
        }
        
        allVectors = append(allVectors, vectors...)
    }
    
    return allVectors, nil
}
```

### Context Cancellation

Always respect context cancellation:

```go
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Check for cancellation
    select {
    case <-ctx.Done():
        return runtime.Response{}, ctx.Err()
    default:
        // Continue
    }
    
    // Use context in all operations
    resp, err := a.client.GenerateWithContext(ctx, req.Prompt)
    return resp, err
}
```

## Error Handling

### Never Panic

```go
// GOOD: Return errors properly
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    if req.Prompt == "" {
        return runtime.Response{}, errors.New("prompt cannot be empty")
    }
    
    resp, err := a.client.Generate(req.Prompt)
    if err != nil {
        return runtime.Response{}, fmt.Errorf("generation failed: %w", err)
    }
    
    return resp, nil
}

// BAD: Never panic
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    if req.Prompt == "" {
        panic("empty prompt")  // NEVER DO THIS
    }
    return a.client.Generate(req.Prompt)
}
```

### Error Wrapping

```go
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    resp, err := a.client.Generate(req.Prompt)
    if err != nil {
        // Wrap with context
        return runtime.Response{}, fmt.Errorf("generation failed (backend=%s): %w", a.Name(), err)
    }
    
    return resp, nil
}
```

### Resource Cleanup

```go
func (a *Adapter) Close() error {
    var errs []error
    
    // Close in reverse order of creation
    if a.client != nil {
        if err := a.client.Close(); err != nil {
            errs = append(errs, err)
        }
    }
    
    if a.cache != nil {
        if err := a.cache.Close(); err != nil {
            errs = append(errs, err)
        }
    }
    
    // Return first error or nil
    if len(errs) > 0 {
        return fmt.Errorf("errors during shutdown: %v", errs)
    }
    return nil
}
```

## Logging Guidelines

### Structured Logging

```go
import "log"

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    log.Printf("[adapter=%s] Generating response (prompt_len=%d)", a.Name(), len(req.Prompt))
    
    start := time.Now()
    resp, err := a.client.Generate(req.Prompt)
    duration := time.Since(start)
    
    if err != nil {
        log.Printf("[adapter=%s] Generation failed after %v: %v", a.Name(), duration, err)
        return runtime.Response{}, err
    }
    
    log.Printf("[adapter=%s] Generated %d tokens in %v", a.Name(), len(resp.Text), duration)
    return resp, nil
}
```

### Sensitive Data

Never log sensitive information:

```go
// BAD: Logging API key
log.Printf("API request with key: %s", apiKey)  // NEVER DO THIS

// GOOD: Log without sensitive data
log.Printf("API request to %s (auth: present)", apiURL)
```

## Testing Strategies

### Mock Interfaces

```go
// Define interface for external dependency
type EmbeddingClient interface {
    Embed(ctx context.Context, text string) ([]float32, error)
}

// Mock implementation
type MockClient struct {
    EmbedFunc func(ctx context.Context, text string) ([]float32, error)
}

func (m *MockClient) Embed(ctx context.Context, text string) ([]float32, error) {
    if m.EmbedFunc != nil {
        return m.EmbedFunc(ctx, text)
    }
    return make([]float32, 384), nil
}

// Use in tests
func TestProvider_Embed(t *testing.T) {
    mock := &MockClient{
        EmbedFunc: func(ctx context.Context, text string) ([]float32, error) {
            return []float32{0.1, 0.2, 0.3}, nil
        },
    }
    
    provider := &Provider{client: mock}
    
    vector, err := provider.Embed(context.Background(), "test")
    if err != nil {
        t.Fatalf("Embed failed: %v", err)
    }
    
    if len(vector) != 3 {
        t.Errorf("Expected vector length 3, got %d", len(vector))
    }
}
```

### Integration Testing

```go
func TestAdapter_Integration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    // Create adapter with real configuration
    adapter, err := newAdapter(config.RuntimeConfig{
        Backend: "http",
        HTTP: config.HTTPBackendConfig{
            BaseURL: "http://localhost:8080",
        },
    })
    if err != nil {
        t.Fatalf("Failed to create adapter: %v", err)
    }
    defer adapter.Close()
    
    // Test generation
    resp, err := adapter.Generate(context.Background(), runtime.Request{
        Prompt: "Say hello",
    })
    if err != nil {
        t.Fatalf("Generation failed: %v", err)
    }
    
    if resp.Text == "" {
        t.Error("Expected non-empty response")
    }
}
```

### Contract Testing

```go
// Test that adapter satisfies the runtime.Adapter interface
var _ runtime.Adapter = (*Adapter)(nil)
```

## Anti-Patterns

### 1. Hardcoded Configuration

```go
// BAD
var baseURL = "https://api.example.com"

// GOOD
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    baseURL := cfg.HTTP.BaseURL
    if baseURL == "" {
        return nil, errors.New("base_url is required")
    }
}
```

### 2. Blocking Operations

```go
// BAD: Blocking operation without timeout
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    return a.client.Generate(req.Prompt)  // May block forever
}

// GOOD: With timeout
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()
    
    return a.client.GenerateWithContext(ctx, req.Prompt)
}
```

### 3. Global State

```go
// BAD: Global state
var globalClient *Client

func init() {
    globalClient, _ = NewClient()
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    return globalClient.Generate(req.Prompt)  // Shared state
}

// GOOD: Instance state
type Adapter struct {
    client *Client
}

func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    client, err := NewClient()
    if err != nil {
        return nil, err
    }
    
    return &Adapter{client: client}, nil
}
```

### 4. Ignoring Errors

```go
// BAD: Ignoring errors
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    resp, _ := a.client.Generate(req.Prompt)  // Ignoring error
    return resp, nil
}

// GOOD: Handling errors
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    resp, err := a.client.Generate(req.Prompt)
    if err != nil {
        return runtime.Response{}, err
    }
    return resp, nil
}
```

## Code Style Guidelines

### Package Organization

```
your-plugin/
├── adapter.go          // Main adapter implementation
├── config.go           // Configuration parsing
├── client.go          // HTTP client or external communication
├── types.go           // Type definitions
├── test/              // Test files
│   └── adapter_test.go
└── README.md
```

### Interface Design

```go
// Keep interfaces small and focused
type Generator interface {
    Generate(ctx context.Context, prompt string) (string, error)
    Stream(ctx context.Context, prompt string, cb func(token string)) error
}

// Split large interfaces
type Generator interface {
    Generate(ctx context.Context, prompt string) (string, error)
}

type StreamingGenerator interface {
    Generate(ctx context.Context, prompt string) (string, error)
    Stream(ctx context.Context, prompt string, cb func(token string)) error
}
```

### Documentation

```go
// Adapter provides access to the custom LLM backend.
//
// The adapter handles all communication with the backend,
// including authentication, request formatting, and response parsing.
//
// # Example Usage
//
//	cfg, _ := config.Resolve()
//	adapter, _ := newAdapter(cfg)
//	defer adapter.Close()
//
//	resp, _ := adapter.Generate(ctx, runtime.Request{Prompt: "Hello"})
//	fmt.Println(resp.Text)
type Adapter struct {
    client *Client
    config Config
}

// Generate creates a completion for the given prompt.
//
// The context is checked for cancellation, and the request
// will be aborted if the context is cancelled or times out.
//
// Returns an error if the request fails or the context is cancelled.
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Implementation
}
```

## Checklist for Production

### Security
- [ ] Input validation implemented
- [ ] Network security (TLS, certificate validation)
- [ ] SSRF prevention
- [ ] API key security (environment variables)
- [ ] Rate limiting implemented
- [ ] Request size limits enforced
- [ ] Path traversal prevention
- [ ] SQL injection prevention (parameterized queries)

### Performance
- [ ] Connection pooling configured
- [ ] Caching implemented
- [ ] Batch processing for bulk operations
- [ ] Context cancellation respected
- [ ] Resource cleanup in Close()

### Error Handling
- [ ] No panics in plugin code
- [ ] Errors wrapped with context
- [ ] Resource cleanup on error
- [ ] Error logging without sensitive data

### Testing
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Mock interfaces defined
- [ ] Contract tests (interface satisfaction)

### Code Quality
- [ ] Go fmt applied
- [ ] Go vet clean
- [ ] No hardcoded credentials
- [ ] Documentation for public APIs
- [ ] README with usage examples

## Related Documentation

- [Quick Start Guide](index.md)
- [Architecture Guide](architecture.md)
- [Runtime Adapters](runtime-adapters.md)
- [Embedding Providers](embedding-providers.md)
- [Security Considerations](troubleshooting.md#security-issues)

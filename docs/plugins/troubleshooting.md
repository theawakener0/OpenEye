# Troubleshooting

This guide covers common issues encountered when developing and using OpenEye plugins, along with solutions and debugging techniques.

## Common Issues

### Plugin Registration Issues

#### Problem: Plugin not found

**Symptoms:**
```
Error: runtime: backend "mybackend" not registered
```

**Causes:**
1. Plugin package not imported
2. init() function not called
3. Registration call in wrong package

**Solutions:**

```go
// GOOD: Register in init() of the imported package
package myadapter

import "OpenEye/internal/runtime"

func init() {
    runtime.Register("myadapter", newAdapter)
}

// In main.go:
import _ "github.com/yourname/myadapter"  // Package must be imported
```

```bash
# Verify plugin is registered
go build -v ./...
```

#### Problem: Duplicate registration

**Symptoms:**
```
panic: runtime: backend "mybackend" already registered
```

**Solutions:**
```go
// Check if already registered
func init() {
    if _, exists := runtime.DefaultRegistry["mybackend"]; exists {
        return  // Already registered
    }
    runtime.Register("myadapter", newAdapter)
}
```

### Configuration Issues

#### Problem: Configuration not loaded

**Symptoms:**
```
Error: base_url is required
```

**Causes:**
1. YAML configuration not found
2. Field names don't match
3. Environment variables not set

**Solutions:**

```yaml
# openeye.yaml
runtime:
  backend: "myadapter"
  http:
    base_url: "https://api.example.com"  # Must match config struct
    timeout: "60s"
```

```go
// Verify configuration loading
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    fmt.Printf("Config loaded: %+v\n", cfg)  // Debug
    return &Adapter{baseURL: cfg.HTTP.BaseURL}, nil
}
```

#### Problem: Environment variables not working

**Symptoms:**
Environment variables not being applied.

**Solutions:**

```go
// Environment variables use APP_* prefix
export APP_LLM_BASEURL="https://api.example.com"
export APP_LLM_TIMEOUT="60s"

// Verify in code
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    fmt.Printf("BaseURL from config: %s\n", cfg.HTTP.BaseURL)
    fmt.Printf("APP_LLM_BASEURL: %s\n", os.Getenv("APP_LLM_BASEURL"))
    
    // Apply env overrides manually if needed
    if url := os.Getenv("APP_LLM_BASEURL"); url != "" {
        cfg.HTTP.BaseURL = url
    }
    
    return &Adapter{baseURL: cfg.HTTP.BaseURL}, nil
}
```

### Runtime Issues

#### Problem: Context cancellation

**Symptoms:**
```
Context cancelled during request
```

**Solutions:**
```go
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Always check context
    select {
    case <-ctx.Done():
        return runtime.Response{}, ctx.Err()
    default:
        // Continue
    }
    
    // Pass context to all operations
    return a.client.GenerateWithContext(ctx, req.Prompt)
}
```

#### Problem: Timeout not working

**Symptoms:**
Requests hang indefinitely.

**Solutions:**
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

#### Problem: Memory leaks

**Symptoms:**
Memory usage grows over time.

**Solutions:**
```go
func (a *Adapter) Close() error {
    // Clean up all resources
    if a.client != nil {
        a.client.CloseIdleConnections()
    }
    if a.cache != nil {
        a.cache.Clear()
    }
    if a.connections != nil {
        a.connections.Close()
    }
    return nil
}

// Use defer for cleanup
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    conn, err := a.dial()
    if err != nil {
        return nil, err
    }
    defer conn.Close()  // Always clean up
    
    return conn.Generate(req.Prompt)
}
```

### Network Issues

#### Problem: Connection failures

**Symptoms:**
```
Connection refused
Connection timeout
```

**Solutions:**
```go
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Add retry logic
    maxRetries := 3
    var lastErr error
    
    for attempt := 0; attempt < maxRetries; attempt++ {
        resp, err := a.client.Generate(req.Prompt)
        if err == nil {
            return resp, nil
        }
        lastErr = err
        
        // Check if retryable
        if !isRetryable(err) {
            return nil, err
        }
        
        // Exponential backoff
        backoff := time.Duration(attempt+1) * time.Second
        time.Sleep(backoff)
    }
    
    return nil, fmt.Errorf("failed after %d attempts: %w", maxRetries, lastErr)
}

func isRetryable(err error) bool {
    // Retry on connection errors and timeouts
    if strings.Contains(err.Error(), "connection") ||
       strings.Contains(err.Error(), "timeout") ||
       strings.Contains(err.Error(), "no route") {
        return true
    }
    return false
}
```

#### Problem: SSL/TLS errors

**Symptoms:**
```
x509: certificate validation failed
```

**Solutions:**
```go
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    return &Adapter{
        client: &http.Client{
            Timeout: 30 * time.Second,
            Transport: &http.Transport{
                TLSClientConfig: &tls.Config{
                    MinVersion: tls.VersionTLS12,
                    // In production, NEVER skip verification
                    // InsecureSkipVerify: false,  // Default
                },
            },
        },
    }, nil
}
```

### Embedding Issues

#### Problem: Vector dimension mismatch

**Symptoms:**
```
embedding dimension mismatch: expected 768, got 1024
```

**Solutions:**
```go
func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
    vector, err := p.client.GetEmbedding(text)
    if err != nil {
        return nil, err
    }
    
    // Adjust dimension
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
```

#### Problem: Slow embeddings

**Symptoms:**
High latency on embedding requests.

**Solutions:**
```go
type CachedProvider struct {
    provider embedding.Provider
    cache   *lru.Cache[string, []float32]
}

func (p *CachedProvider) Embed(ctx context.Context, text string) ([]float32, error) {
    // Check cache first
    if cached, ok := p.cache.Get(text); ok {
        return cached, nil
    }
    
    vector, err := p.provider.Embed(ctx, text)
    if err != nil {
        return nil, err
    }
    
    p.cache.Add(text, vector)
    return vector, nil
}
```

### Memory Issues

#### Problem: Database locking

**Symptoms:**
```
database is locked
```

**Solutions:**
```go
func bootstrap(db *sql.DB) error {
    // Enable WAL mode for better concurrency
    _, err := db.Exec(`PRAGMA journal_mode=WAL`)
    if err != nil {
        return err
    }
    
    _, err = db.Exec(`PRAGMA synchronous=NORMAL`)
    return err
}

func (s *Store) recent(limit int) error {
    // Use prepared statements
    stmt, err := s.db.Prepare(`SELECT * FROM interactions LIMIT ?`)
    if err != nil {
        return err
    }
    defer stmt.Close()
    
    _, err = stmt.Query(limit)
    return err
}
```

### Security Issues

#### Problem: Path traversal

**Symptoms:**
Access to unauthorized files.

**Solutions:**
```go
func (r *Retriever) loadDocument(path string) (string, error) {
    // Resolve to absolute path
    absPath, err := filepath.Abs(path)
    if err != nil {
        return "", fmt.Errorf("invalid path: %w", err)
    }
    
    // Ensure path is within allowed directory
    allowedDir := r.config.CorpusPath
    if !strings.HasPrefix(absPath, allowedDir) {
        return "", errors.New("path outside allowed corpus directory")
    }
    
    // Read file
    content, err := os.ReadFile(absPath)
    if err != nil {
        return "", fmt.Errorf("failed to read file: %w", err)
    }
    
    return string(content), nil
}
```

#### Problem: API key exposure

**Symptoms:**
API keys in logs or error messages.

**Solutions:**
```go
// GOOD: Mask API keys in logs
log.Printf("API request to %s (auth: %s)", url, maskKey(apiKey))

// BAD: Log full API key
log.Printf("API request with key: %s", apiKey)  // NEVER

func maskKey(key string) string {
    if len(key) <= 8 {
        return "****"
    }
    return key[:4] + "****" + key[len(key)-4:]
}
```

## Debugging Techniques

### Enable Debug Logging

```go
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    if os.Getenv("DEBUG") != "" {
        log.Printf("[adapter] Generate called with prompt_len=%d", len(req.Prompt))
        defer log.Printf("[adapter] Generate completed")
    }
    
    // Implementation
}
```

```bash
DEBUG=1 OpenEye chat --message "Hello"
```

### Verbose HTTP Logging

```go
import "net/http/httputil"

func (a *Adapter) RoundTrip(req *http.Request) (*http.Response, error) {
    // Log request (without sensitive headers)
    dump, _ := httputil.DumpRequestOut(req, false)
    log.Printf("Request: %s", dump)
    
    resp, err := a.client.RoundTrip(req)
    
    // Log response
    dump, _ = httputil.DumpResponse(resp, false)
    log.Printf("Response: %s", dump)
    
    return resp, err
}
```

### Profiling

```go
import "runtime/pprof"

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Start CPU profiling
    pprof.StartCPUProfile(os.Stdout)
    defer pprof.StopCPUProfile()
    
    return a.client.Generate(req.Prompt)
}
```

```bash
# Profile CPU usage
go tool pprof cpu.prof

# Profile memory usage
go tool pprof heap.prof
```

## Error Codes Reference

| Error Code | Description | Solution |
|------------|-------------|----------|
| E001 | Backend not registered | Import plugin package |
| E002 | Configuration required | Set required config fields |
| E003 | Invalid timeout format | Use Go duration format (e.g., "30s") |
| E004 | Context cancelled | Check context in implementation |
| E005 | Connection refused | Check backend URL and availability |
| E006 | Vector dimension mismatch | Configure correct dimension |
| E007 | Database locked | Use WAL mode, reduce concurrency |
| E008 | Rate limit exceeded | Implement rate limiting |
| E009 | SSL/TLS error | Check certificates |
| E010 | Path traversal attempt | Validate file paths |

## Frequently Asked Questions

### Q: How do I debug init() functions?

```go
// Use log package - it works in init()
func init() {
    log.Println("[myadapter] Initializing...")
    
    // Or use fmt to stderr
    fmt.Fprintf(os.Stderr, "[myadapter] Registered as 'myadapter'\n")
}
```

### Q: Why is my plugin not loading?

1. Check if the package is imported
2. Verify init() is called
3. Check for panics in init()
4. Ensure Go module is valid

```bash
# Build with verbose output
go build -v ./...

# Check dependencies
go list -m all
```

### Q: How do I handle optional dependencies?

```go
func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    if cfg.HTTP.BaseURL == "" {
        // Return a no-op adapter or error
        return nil, errors.New("base_url is required")
    }
    
    // Check for optional API key
    apiKey := os.Getenv("MY_ADAPTER_API_KEY")
    if apiKey == "" {
        log.Printf("[myadapter] WARNING: No API key configured")
    }
    
    return &Adapter{
        baseURL: cfg.HTTP.BaseURL,
        apiKey:  apiKey,
    }, nil
}
```

### Q: How do I test with mocks?

```go
type MockAdapter struct {
    GenerateFunc func(ctx context.Context, req runtime.Request) (runtime.Response, error)
}

func (m *MockAdapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    if m.GenerateFunc != nil {
        return m.GenerateFunc(ctx, req)
    }
    return runtime.Response{Text: "mock response"}, nil
}

// In tests
adapter := &MockAdapter{
    GenerateFunc: func(ctx context.Context, req runtime.Request) (runtime.Response, error) {
        return runtime.Response{Text: "test response"}, nil
    },
}
```

### Q: How do I handle graceful degradation?

```go
type Adapter struct {
    primary   *PrimaryClient
    fallback  *FallbackClient
    usePrimary atomic.Bool
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    if a.usePrimary.Load() {
        resp, err := a.primary.Generate(ctx, req.Prompt)
        if err == nil {
            return resp, nil
        }
        
        // Fall back to secondary
        log.Printf("[adapter] Primary failed, using fallback: %v", err)
        a.usePrimary.Store(false)
    }
    
    return a.fallback.Generate(ctx, req.Prompt)
}
```

## Related Documentation

- [Best Practices](best-practices.md)
- [Security Guidelines](best-practices.md#security-guidelines)
- [Runtime Adapters](runtime-adapters.md)
- [Embedding Providers](embedding-providers.md)

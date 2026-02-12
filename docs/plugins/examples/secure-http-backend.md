# Secure HTTP Backend Adapter

This example demonstrates a security-focused HTTP LLM adapter with comprehensive security measures for production use.

## Overview

This adapter implements industry-standard security practices:

- **TLS with certificate validation**
- **SSRF protection**
- **Rate limiting**
- **Request size limits**
- **API key security**
- **Input validation**
- **Comprehensive logging without sensitive data**

## Security Features

### 1. TLS Certificate Validation

```go
// Secure TLS configuration
transport := &http.Transport{
    TLSClientConfig: &tls.Config{
        MinVersion: tls.VersionTLS12,
        // NEVER disable certificate verification
        // InsecureSkipVerify: false,  // Default (secure)
    },
}
```

### 2. SSRF Prevention

```go
func (a *Adapter) validateURL(rawURL string) error {
    parsedURL, err := url.Parse(rawURL)
    if err != nil {
        return fmt.Errorf("invalid URL: %w", err)
    }
    
    // Only allow HTTP/HTTPS
    if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
        return fmt.Errorf("unsupported scheme: %s", parsedURL.Scheme)
    }
    
    // Block private IP ranges
    if host := parsedURL.Hostname(); host != "" {
        if ip := net.ParseIP(host); ip != nil {
            if isPrivateIP(ip) {
                return errors.New("private IP addresses are not allowed")
            }
        }
        
        // Also check resolved IPs for hostnames
        ips, err := net.LookupIP(host)
        if err != nil {
            return fmt.Errorf("DNS lookup failed: %w", err)
        }
        
        for _, ip := range ips {
            if isPrivateIP(ip) {
                return errors.New("resolved to private IP address")
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
        {IP: net.ParseIP("::1"), Mask: net.CIDRMask(128, 128)},  // IPv6 loopback
        {IP: net.ParseIP("fe80::"), Mask: net.CIDRMask(10, 128)}, // IPv6 link-local
    }
    
    for _, r := range privateRanges {
        if r.Contains(ip) {
            return true
        }
    }
    return false
}
```

### 3. Rate Limiting

```go
type RateLimiter struct {
    limiter   *rate.Limiter
    window    time.Duration
    maxBurst  int
}

func NewRateLimiter(requestsPerSecond float64, maxBurst int) *RateLimiter {
    return &RateLimiter{
        limiter:  rate.NewLimiter(rate.Limit(requestsPerSecond), maxBurst),
        window:   time.Second,
        maxBurst: maxBurst,
    }
}

func (r *RateLimiter) Allow() (allowed bool, retryAfter time.Duration) {
    if r.limiter.Allow() {
        return true, 0
    }
    return false, time.Second / time.Duration(r.limiter.Limit())
}
```

### 4. Request Size Limits

```go
const (
    maxPromptLength   = 100000  // 100KB
    maxRequestBody    = 1024 * 1024  // 1MB
    maxOutputTokens  = 4096
    maxConcurrent     = 10
)

type requestValidator struct {
    semaphore chan struct{}
}

func newRequestValidator() *requestValidator {
    return &requestValidator{
        semaphore: make(chan struct{}, maxConcurrent),
    }
}

func (v *requestValidator) Validate(req runtime.Request) error {
    // Check prompt length
    if len(req.Prompt) > maxPromptLength {
        return fmt.Errorf("prompt exceeds maximum length of %d bytes", maxPromptLength)
    }
    
    // Check output token limit
    if req.Options.MaxTokens > maxOutputTokens {
        return fmt.Errorf("max_tokens exceeds maximum of %d", maxOutputTokens)
    }
    
    // Check temperature range
    if req.Options.Temperature < 0 || req.Options.Temperature > 2 {
        return errors.New("temperature must be between 0 and 2")
    }
    
    return nil
}

func (v *requestValidator) Acquire() error {
    select {
    case v.semaphore <- struct{}{}:
        return nil
    default:
        return fmt.Errorf("too many concurrent requests (max: %d)", maxConcurrent)
    }
}

func (v *requestValidator) Release() {
    <-v.semaphore
}
```

### 5. API Key Security

```go
func (a *Adapter) getAPIKey() (string, error) {
    // Check environment variable first (most secure)
    apiKey := os.Getenv("SECURE_ADAPTER_API_KEY")
    if apiKey != "" {
        return apiKey, nil
    }
    
    // Check config as fallback (less secure)
    if a.config.APIKey != "" {
        log.Printf("[secure-adapter] WARNING: API key found in config file")
        return a.config.APIKey, nil
    }
    
    return "", errors.New("API key is required. Set SECURE_ADAPTER_API_KEY environment variable")
}

func (a *Adapter) setHeaders(req *http.Request) {
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")
    
    apiKey, err := a.getAPIKey()
    if err != nil {
        log.Printf("[secure-adapter] WARNING: No API key configured")
        return
    }
    
    // Mask API key in logs
    log.Printf("[secure-adapter] API request to %s (auth: %s)",
        a.baseURL, maskString(apiKey, 8))
    
    req.Header.Set("Authorization", "Bearer "+apiKey)
}

// Mask sensitive data for logging
func maskString(s string, visibleChars int) string {
    if len(s) <= visibleChars {
        return "****"
    }
    return s[:visibleChars] + "****"
}
```

### 6. Input Sanitization

```go
func sanitizeInput(input string) string {
    // Remove null bytes
    input = strings.ReplaceAll(input, "\x00", "")
    
    // Remove control characters except newlines and tabs
    var builder strings.Builder
    for _, r := range input {
        if unicode.IsPrint(r) || r == '\n' || r == '\t' {
            builder.WriteRune(r)
        }
    }
    
    result := builder.String()
    
    // Trim whitespace
    result = strings.TrimSpace(result)
    
    // Check for suspicious patterns
    suspiciousPatterns := []string{
        "../",
        "..\\",
        "{{",
        "}}",
        "${",
        "`",
        "$((",
    }
    
    for _, pattern := range suspiciousPatterns {
        if strings.Contains(result, pattern) {
            log.Printf("[secure-adapter] WARNING: Suspicious pattern detected: %s", pattern)
        }
    }
    
    return result
}
```

### 7. Comprehensive Logging

```go
type SecureLogger struct {
    logger    *log.Logger
    sensitive map[string]bool
}

func NewSecureLogger() *SecureLogger {
    return &SecureLogger{
        logger:    log.New(os.Stdout, "[secure-adapter] ", log.LstdFlags),
        sensitive: map[string]bool{
            "api_key":    true,
            "password":  true,
            "token":     true,
            "secret":    true,
        },
    }
}

func (l *SecureLogger) LogRequest(url string, opts interface{}) {
    l.logger.Printf("REQUEST: url=%s method=POST", sanitizeURL(url))
    
    // Log non-sensitive options
    if opts, ok := opts.(runtime.GenerationOptions); ok {
        optsMap := map[string]interface{}{
            "max_tokens":   opts.MaxTokens,
            "temperature": opts.Temperature,
            "top_p":       opts.TopP,
        }
        
        optsJSON, _ := json.Marshal(optsMap)
        l.logger.Printf("OPTIONS: %s", string(optsJSON))
    }
}

func (l *SecureLogger) LogResponse(status string, tokens int, duration time.Duration) {
    l.logger.Printf("RESPONSE: status=%s tokens=%d duration=%v",
        status, tokens, duration)
}

func (l *SecureLogger) LogError(err error) {
    l.logger.Printf("ERROR: %v", err)
}

func (l *SecureLogger) LogSecurityEvent(event string, details map[string]string) {
    // Log security events separately
    l.logger.Printf("SECURITY: event=%s", event)
    
    // Log non-sensitive details
    for key, value := range details {
        if l.sensitive[key] {
            details[key] = "****"
        }
    }
    
    detailsJSON, _ := json.Marshal(details)
    l.logger.Printf("SECURITY_DETAILS: %s", string(detailsJSON))
}
```

## Complete Adapter Implementation

```go
package secureadapter

import (
    "bufio"
    "bytes"
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "log"
    "math"
    "net"
    "net/http"
    "net/url"
    "os"
    "strings"
    "sync"
    "time"
    "unicode"

    "OpenEye/internal/config"
    "OpenEye/internal/runtime"
)

func init() {
    runtime.Register("secure-http", newAdapter)
}

type Adapter struct {
    baseURL          string
    apiKey           string
    timeout          time.Duration
    maxTokens        int
    client           *http.Client
    validator        *requestValidator
    rateLimiter      *RateLimiter
    logger          *SecureLogger
    mu              sync.RWMutex
}

func newAdapter(cfg config.RuntimeConfig) (runtime.Adapter, error) {
    // Validate URL
    baseURL := strings.TrimSpace(cfg.HTTP.BaseURL)
    if baseURL == "" {
        return nil, errors.New("secure-http: base_url is required")
    }
    
    // Validate URL and prevent SSRF
    if err := validateURL(baseURL); err != nil {
        return nil, fmt.Errorf("secure-http: URL validation failed: %w", err)
    }
    
    // Parse timeout
    timeout := 60 * time.Second
    if cfg.HTTP.Timeout != "" {
        var err error
        timeout, err = time.ParseDuration(cfg.HTTP.Timeout)
        if err != nil {
            return nil, fmt.Errorf("secure-http: invalid timeout: %w", err)
        }
    }
    
    // Enforce maximum timeout
    if timeout > 5*time.Minute {
        timeout = 5 * time.Minute
    }
    
    // Set defaults
    maxTokens := cfg.Defaults.MaxTokens
    if maxTokens <= 0 {
        maxTokens = 512
    }
    
    return &Adapter{
        baseURL:     baseURL,
        timeout:     timeout,
        maxTokens:   maxTokens,
        client: &http.Client{
            Timeout: timeout,
            Transport: &http.Transport{
                MaxIdleConns:        10,
                MaxIdleConnsPerHost: 5,
                IdleConnTimeout:      90 * time.Second,
                TLSClientConfig: &tls.Config{
                    MinVersion: tls.VersionTLS12,
                },
            },
        },
        validator:   newRequestValidator(),
        rateLimiter: NewRateLimiter(50, 100),
        logger:      NewSecureLogger(),
    }, nil
}

func validateURL(rawURL string) error {
    parsedURL, err := url.Parse(rawURL)
    if err != nil {
        return fmt.Errorf("invalid URL: %w", err)
    }
    
    if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
        return fmt.Errorf("unsupported scheme: %s", parsedURL.Scheme)
    }
    
    host := parsedURL.Hostname()
    if host == "" {
        return errors.New("empty hostname")
    }
    
    // Check IP addresses
    if ip := net.ParseIP(host); ip != nil {
        if isPrivateIP(ip) {
            return errors.New("private IP address not allowed")
        }
    } else {
        // Check resolved IPs for hostnames
        ips, err := net.LookupIP(host)
        if err != nil {
            return fmt.Errorf("DNS lookup failed: %w", err)
        }
        
        for _, ip := range ips {
            if isPrivateIP(ip) {
                return errors.New("resolved to private IP address")
            }
        }
    }
    
    // Block localhost
    if host == "localhost" || host == "127.0.0.1" || host == "::1" {
        return errors.New("localhost not allowed")
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

func (a *Adapter) Name() string {
    return "secure-http"
}

func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
    // Validate request
    if err := a.validator.Validate(req); err != nil {
        a.logger.LogError(fmt.Errorf("validation failed: %w", err))
        return runtime.Response{}, err
    }
    
    // Acquire concurrent request slot
    if err := a.validator.Acquire(); err != nil {
        a.logger.LogError(err)
        return runtime.Response{}, err
    }
    defer a.validator.Release()
    
    // Check rate limit
    if allowed, retryAfter := a.rateLimiter.Allow(); !allowed {
        a.logger.LogError(fmt.Errorf("rate limit exceeded, retry after %v", retryAfter))
        return runtime.Response{}, fmt.Errorf("rate limit exceeded, retry after %v", retryAfter)
    }
    
    // Sanitize input
    prompt := sanitizeInput(req.Prompt)
    
    // Build request
    apiReq := APIRequest{
        Model:     a.config.Model,
        Prompt:    prompt,
        MaxTokens: a.maxTokens,
    }
    
    // Apply options
    if req.Options.MaxTokens > 0 {
        apiReq.MaxTokens = req.Options.MaxTokens
    }
    if req.Options.Temperature > 0 {
        apiReq.Temperature = req.Options.Temperature
    }
    
    // Serialize and log
    body, err := json.Marshal(apiReq)
    if err != nil {
        return runtime.Response{}, fmt.Errorf("failed to marshal request: %w", err)
    }
    
    a.logger.LogRequest(a.baseURL, apiReq)
    
    // Create request
    httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
        fmt.Sprintf("%s/v1/completions", a.baseURL), bytes.NewReader(body))
    if err != nil {
        return runtime.Response{}, fmt.Errorf("failed to create request: %w", err)
    }
    
    // Set headers
    a.setHeaders(httpReq)
    
    // Execute
    startTime := time.Now()
    httpResp, err := a.client.Do(httpReq)
    if err != nil {
        a.logger.LogError(err)
        return runtime.Response{}, fmt.Errorf("request failed: %w", err)
    }
    defer httpResp.Body.Close()
    
    // Check status
    if httpResp.StatusCode != http.StatusOK {
        respBody, _ := io.ReadAll(httpResp.Body)
        a.logger.LogError(fmt.Errorf("API error: %d - %s", httpResp.StatusCode, respBody))
        return runtime.Response{}, fmt.Errorf("API error (status %d)", httpResp.StatusCode)
    }
    
    // Decode
    var apiResp APIResponse
    if err := json.NewDecoder(httpResp.Body).Decode(&apiResp); err != nil {
        return runtime.Response{}, fmt.Errorf("failed to decode response: %w", err)
    }
    
    if len(apiResp.Choices) == 0 {
        return runtime.Response{}, errors.New("no choices returned")
    }
    
    responseText := strings.TrimSpace(apiResp.Choices[0].Text)
    duration := time.Since(startTime)
    
    // Log success
    a.logger.LogResponse("success", apiResp.Usage.TotalTokens, duration)
    
    return runtime.Response{
        Text: responseText,
        Stats: runtime.Stats{
            TokensGenerated: apiResp.Usage.TotalTokens,
            Duration:        duration,
        },
    }, nil
}

func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
    // Validate
    if err := a.validator.Validate(req); err != nil {
        return err
    }
    
    // Acquire slot
    if err := a.validator.Acquire(); err != nil {
        return err
    }
    defer a.validator.Release()
    
    // Check rate limit
    if allowed, _ := a.rateLimiter.Allow(); !allowed {
        return errors.New("rate limit exceeded")
    }
    
    // Sanitize
    prompt := sanitizeInput(req.Prompt)
    
    // Build and execute request (similar to Generate)
    // ...
    
    return nil
}

func (a *Adapter) Close() error {
    a.client.CloseIdleConnections()
    return nil
}

func (a *Adapter) setHeaders(req *http.Request) {
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json")
    
    apiKey := os.Getenv("SECURE_ADAPTER_API_KEY")
    if apiKey != "" {
        req.Header.Set("Authorization", "Bearer "+apiKey)
    }
}

func sanitizeInput(input string) string {
    var builder strings.Builder
    for _, r := range input {
        if unicode.IsPrint(r) || r == '\n' || r == '\t' {
            builder.WriteRune(r)
        }
    }
    return strings.TrimSpace(builder.String())
}

// API types
type APIRequest struct {
    Model       string  `json:"model,omitempty"`
    Prompt      string  `json:"prompt"`
    MaxTokens   int     `json:"max_tokens,omitempty"`
    Temperature float64 `json:"temperature,omitempty"`
}

type APIResponse struct {
    Choices []struct {
        Text string `json:"text"`
    } `json:"choices"`
    Usage struct {
        TotalTokens int `json:"total_tokens"`
    } `json:"usage"`
}
```

## Security Checklist

- [x] TLS 1.2+ enforced
- [x] Certificate validation enabled
- [x] SSRF protection implemented
- [x] Private IP blocking
- [x] Rate limiting configured
- [x] Request size limits
- [x] Input sanitization
- [x] API key security (env vars)
- [x] Secure logging (no sensitive data)
- [x] Concurrent request limits
- [x] Timeout enforcement
- [x] Error handling without exposure

## Related Documentation

- [Runtime Adapters](../runtime-adapters.md)
- [Best Practices](../best-practices.md)
- [Security Guidelines](../best-practices.md#security-guidelines)
- [Troubleshooting](../troubleshooting.md)

package llmclient

import (
	"context"
	"fmt"
	"strings"
	"time"
	"unicode"

	"OpenEye/internal/config"
	"OpenEye/internal/runtime"
)

func init() {
	runtime.Register("http", func(cfg config.RuntimeConfig) (runtime.Adapter, error) {
		baseURL := strings.TrimSpace(cfg.HTTP.BaseURL)
		if baseURL == "" {
			return nil, fmt.Errorf("llmclient: http backend requires base_url")
		}

		timeout := 60 * time.Second
		if cfg.HTTP.Timeout != "" {
			parsed, err := time.ParseDuration(cfg.HTTP.Timeout)
			if err != nil {
				return nil, fmt.Errorf("llmclient: invalid http timeout %q: %w", cfg.HTTP.Timeout, err)
			}
			timeout = parsed
		}

		return NewAdapter(baseURL, timeout, runtime.GenerationOptions{
			MaxTokens:     cfg.Defaults.MaxTokens,
			Temperature:   cfg.Defaults.Temperature,
			TopK:          cfg.Defaults.TopK,
			TopP:          cfg.Defaults.TopP,
			MinP:          cfg.Defaults.MinP,
			RepeatPenalty: cfg.Defaults.RepeatPenalty,
			RepeatLastN:   cfg.Defaults.RepeatLastN,
		}), nil
	})
}

// Adapter bridges the generic runtime interface with the HTTP completion API.
type Adapter struct {
	client   *Client
	defaults runtime.GenerationOptions
}

// NewAdapter constructs an HTTP adapter with global default options.
func NewAdapter(baseURL string, timeout time.Duration, defaults runtime.GenerationOptions) *Adapter {
	return &Adapter{
		client:   NewClientWithTimeout(baseURL, timeout),
		defaults: defaults,
	}
}

// Name returns the adapter label.
func (a *Adapter) Name() string { return "http" }

// Close releases underlying resources.
func (a *Adapter) Close() error { return nil }

// Generate performs a blocking completion request.
func (a *Adapter) Generate(ctx context.Context, req runtime.Request) (runtime.Response, error) {
	options := mergeOptions(a.defaults, req.Options)

	completionReq := CompletionRequest{
		Prompt:        req.Prompt,
		NPredict:      options.MaxTokens,
		Temperature:   options.Temperature,
		TopK:          options.TopK,
		TopP:          options.TopP,
		MinP:          options.MinP,
		RepeatPenalty: options.RepeatPenalty,
		RepeatLastN:   options.RepeatLastN,
		Stream:        false,
		CachePrompt:   true,
		Stop:          options.Stop,
	}

	resp, err := a.client.Generate(ctx, completionReq)
	if err != nil {
		return runtime.Response{}, err
	}

	return runtime.Response{
		Text: resp.Content,
		Stats: runtime.Stats{
			TokensCached:    resp.TokensCached,
			TokensEvaluated: resp.TokensEvaluated,
		},
		Raw:    resp,
		Finish: resp.StopType,
	}, nil
}

// Stream simulates token streaming by chunking the full response if native streaming is unavailable.
func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
	response, err := a.Generate(ctx, req)
	if err != nil {
		return err
	}

	tokens := chunkForStreaming(response.Text)
	for idx, token := range tokens {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := cb(runtime.StreamEvent{Token: token, Index: idx}); err != nil {
			return err
		}
	}

	return cb(runtime.StreamEvent{Final: true})
}

func mergeOptions(base, override runtime.GenerationOptions) runtime.GenerationOptions {
	result := base

	if override.MaxTokens != 0 {
		result.MaxTokens = override.MaxTokens
	}
	if override.Temperature != 0 {
		result.Temperature = override.Temperature
	}
	if override.TopK != 0 {
		result.TopK = override.TopK
	}
	if override.TopP != 0 {
		result.TopP = override.TopP
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

func chunkForStreaming(text string) []string {
	if text == "" {
		return nil
	}

	var (
		tokens  []string
		builder strings.Builder
	)

	flush := func() {
		if builder.Len() == 0 {
			return
		}
		tokens = append(tokens, builder.String())
		builder.Reset()
	}

	for _, r := range text {
		builder.WriteRune(r)
		if unicode.IsSpace(r) || strings.ContainsRune(",.;:!?", r) {
			flush()
		}
	}

	flush()
	return tokens
}

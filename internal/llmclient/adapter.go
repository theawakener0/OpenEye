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

	// If images are provided, use the multimodal prompt format
	if len(req.Image) > 0 {
		completionReq.Prompt = MultimodalPrompt{
			PromptString:   req.Prompt,
			MultimodalData: req.Image,
		}
	} else {
		completionReq.Prompt = req.Prompt
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

// Stream performs real token streaming via SSE, falling back to chunking if SSE fails.
func (a *Adapter) Stream(ctx context.Context, req runtime.Request, cb runtime.StreamCallback) error {
	options := mergeOptions(a.defaults, req.Options)

	completionReq := CompletionRequest{
		NPredict:      options.MaxTokens,
		Temperature:   options.Temperature,
		TopK:          options.TopK,
		TopP:          options.TopP,
		MinP:          options.MinP,
		RepeatPenalty: options.RepeatPenalty,
		RepeatLastN:   options.RepeatLastN,
		Stream:        true, // Request streaming
		CachePrompt:   true,
		Stop:          options.Stop,
	}

	// Handle multimodal prompts
	if len(req.Image) > 0 {
		completionReq.Prompt = MultimodalPrompt{
			PromptString:   req.Prompt,
			MultimodalData: req.Image,
		}
	} else {
		completionReq.Prompt = req.Prompt
	}

	// Try real SSE streaming first
	idx := 0
	err := a.client.GenerateStream(ctx, completionReq, func(token StreamToken) error {
		if token.Stop {
			return cb(runtime.StreamEvent{Final: true})
		}
		if token.Content != "" {
			if err := cb(runtime.StreamEvent{Token: token.Content, Index: idx}); err != nil {
				return err
			}
			idx++
		}
		return nil
	})

	// If streaming succeeded, we're done
	if err == nil {
		return nil
	}

	// Fallback: if streaming fails, use blocking generate + chunking
	response, genErr := a.Generate(ctx, req)
	if genErr != nil {
		return genErr
	}

	if response.Text == "" {
		return cb(runtime.StreamEvent{Final: true})
	}

	tokens := chunkForStreaming(response.Text)
	for i, token := range tokens {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := cb(runtime.StreamEvent{Token: token, Index: i}); err != nil {
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

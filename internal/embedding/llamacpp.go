package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"OpenEye/internal/config"
)

type llamaCppProvider struct {
	baseURL string
	model   string
	client  *http.Client
}

func newLlamaCppProvider(cfg config.LlamaCppEmbeddingConfig) (Provider, error) {
	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		return nil, fmt.Errorf("embedding: llamacpp base_url is required")
	}

	timeout := 30 * time.Second
	if cfg.Timeout != "" {
		if parsed, err := time.ParseDuration(cfg.Timeout); err == nil {
			timeout = parsed
		}
	}

	return &llamaCppProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   strings.TrimSpace(cfg.Model),
		client:  &http.Client{Timeout: timeout},
	}, nil
}

func (p *llamaCppProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	if p == nil {
		return nil, fmt.Errorf("embedding: provider not initialised")
	}

	payload := map[string]any{
		"content": text,
	}
	if p.model != "" {
		payload["model"] = p.model
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("embedding: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/embedding", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("embedding: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedding: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding: server returned %d", resp.StatusCode)
	}

	var decoded llamaCppEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("embedding: decode response: %w", err)
	}

	vector := decoded.Flatten()
	if len(vector) == 0 {
		return nil, fmt.Errorf("embedding: empty vector returned")
	}

	result := make([]float32, len(vector))
	for i, v := range vector {
		result[i] = float32(v)
	}

	return result, nil
}

func (p *llamaCppProvider) Close() error {
	// No persistent resources to release for HTTP client.
	return nil
}

type llamaCppEmbeddingResponse struct {
	Embedding []float64                    `json:"embedding"`
	Data      []llamaCppEmbeddingDataEntry `json:"data"`
}

type llamaCppEmbeddingDataEntry struct {
	Embedding []float64 `json:"embedding"`
}

func (r llamaCppEmbeddingResponse) Flatten() []float64 {
	if len(r.Embedding) != 0 {
		return r.Embedding
	}
	if len(r.Data) > 0 && len(r.Data[0].Embedding) > 0 {
		return r.Data[0].Embedding
	}
	return nil
}

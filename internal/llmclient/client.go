package llmclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type CompletionRequest struct {
	Prompt        string   `json:"prompt"`
	NPredict      int      `json:"n_predict,omitempty"`
	Temperature   float64  `json:"temperature,omitempty"`
	TopK          int      `json:"top_k,omitempty"`
	TopP          float64  `json:"top_p,omitempty"`
	MinP          float64  `json:"min_p,omitempty"`
	RepeatPenalty float64  `json:"repeat_penalty,omitempty"`
	RepeatLastN   int      `json:"repeat_last_n,omitempty"`
	Stream        bool     `json:"stream"`
	Stop          []string `json:"stop,omitempty"`
	CachePrompt   bool     `json:"cache_prompt"`
}

type CompletionResponse struct {
	Content            string         `json:"content"`
	Stop               bool           `json:"stop"`
	Model              string         `json:"model"`
	Tokens             []int          `json:"tokens,omitempty"`
	StopType           string         `json:"stop_type"`
	StoppingWord       string         `json:"stopping_word"`
	TokensCached       int            `json:"tokens_cached"`
	TokensEvaluated    int            `json:"tokens_evaluated"`
	GenerationSettings map[string]any `json:"generation_settings,omitempty"`
	Truncated          bool           `json:"truncated"`
	Timings            map[string]any `json:"timings,omitempty"`
}

type Client struct {
	BaseURL    string
	httpClient *http.Client
}

func NewClient(baseURL string) *Client {
	return NewClientWithTimeout(baseURL, 60*time.Second)
}

// NewClientWithTimeout constructs a client using the provided timeout for HTTP requests.
func NewClientWithTimeout(baseURL string, timeout time.Duration) *Client {
	return &Client{
		BaseURL:    baseURL,
		httpClient: &http.Client{Timeout: timeout},
	}
}

func (c *Client) Generate(ctx context.Context, req CompletionRequest) (CompletionResponse, error) {
	url := fmt.Sprintf("%s/completion", c.BaseURL)
	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errorBody map[string]any
		if json.NewDecoder(resp.Body).Decode(&errorBody) == nil {
			return CompletionResponse{}, fmt.Errorf("server returned %d: %v", resp.StatusCode, errorBody)
		}
		return CompletionResponse{}, fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	var respBody CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&respBody); err != nil {
		return CompletionResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}
	return respBody, nil
}

func (c *Client) GetResponse(prompt string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	req := CompletionRequest{
		Prompt:        prompt,
		NPredict:      512,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		MinP:          0.05,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
		Stream:        false,
		CachePrompt:   true,
	}

	resp, err := c.Generate(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	if resp.Content == "" {
		return "", fmt.Errorf("no response content generated")
	}

	return resp.Content, nil
}

package llmclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// MultimodalPrompt represents a prompt with optional multimodal data for llama.cpp
type MultimodalPrompt struct {
	PromptString   string   `json:"prompt_string"`
	MultimodalData []string `json:"multimodal_data,omitempty"`
}

// CompletionRequest represents the request structure for llama.cpp /completion endpoint
type CompletionRequest struct {
	Prompt        any      `json:"prompt"` // Can be string or MultimodalPrompt
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

// StreamToken represents a single token event from SSE stream
type StreamToken struct {
	Content  string `json:"content"`
	Stop     bool   `json:"stop"`
	StopType string `json:"stop_type,omitempty"`
}

// StreamCallback is called for each token received during streaming
type StreamCallback func(StreamToken) error

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

	// DEBUG: Log if images are included in request
	/*if len(req.Image) > 0 {
		log.Printf("debug: sending request with %d image(s), first image length=%d", len(req.Image), len(req.Image[0]))
	} else {
		log.Printf("debug: sending request WITHOUT images")
	}*/

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

	// Read the full response body for debugging
	respBodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return CompletionResponse{}, fmt.Errorf("failed to read response body: %w", err)
	}

	//log.Printf("debug: response status=%d, body length=%d", resp.StatusCode, len(respBodyBytes))

	// Print first 500 chars of response for debugging
	/*preview := string(respBodyBytes)
	if len(preview) > 500 {
		preview = preview[:500] + "..."
	}
	log.Printf("debug: raw response: %s", preview)
	*/

	if resp.StatusCode != http.StatusOK {
		var errorBody map[string]any
		if json.Unmarshal(respBodyBytes, &errorBody) == nil {
			return CompletionResponse{}, fmt.Errorf("server returned %d: %v", resp.StatusCode, errorBody)
		}
		return CompletionResponse{}, fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(respBodyBytes))
	}

	var respBody CompletionResponse
	if err := json.Unmarshal(respBodyBytes, &respBody); err != nil {
		log.Printf("debug: failed to parse response: %s", string(respBodyBytes))
		return CompletionResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}

	//log.Printf("debug: response content length=%d, stop=%v", len(respBody.Content), respBody.Stop)

	return respBody, nil
}

func (c *Client) GetResponse(prompt string) (string, error) {
	return c.GetResponseWithImages(prompt, nil)
}

// GetResponseWithImages generates a response with optional images.
func (c *Client) GetResponseWithImages(prompt string, images []string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	req := CompletionRequest{
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

	// Use multimodal prompt format if images are provided
	if len(images) > 0 {
		req.Prompt = MultimodalPrompt{
			PromptString:   prompt,
			MultimodalData: images,
		}
	} else {
		req.Prompt = prompt
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

// GenerateStream performs a streaming completion request, calling cb for each token.
// It parses SSE (Server-Sent Events) from the llama.cpp server.
func (c *Client) GenerateStream(ctx context.Context, req CompletionRequest, cb StreamCallback) error {
	req.Stream = true // Force streaming mode

	url := fmt.Sprintf("%s/completion", c.BaseURL)
	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	// Use a client without timeout for streaming since generation can take a while
	streamClient := &http.Client{}
	resp, err := streamClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server returned status %d: %s", resp.StatusCode, string(body))
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()

		// Skip empty lines (SSE event separators)
		if line == "" {
			continue
		}

		// Parse SSE data lines
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")

			var token StreamToken
			if err := json.Unmarshal([]byte(jsonData), &token); err != nil {
				// Skip malformed events
				continue
			}

			if err := cb(token); err != nil {
				return err
			}

			if token.Stop {
				return nil // Generation complete
			}
		}
	}

	return scanner.Err()
}

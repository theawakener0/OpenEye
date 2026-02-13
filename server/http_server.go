package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// ChatRequest represents a chat inference request
type ChatRequest struct {
	Message string                 `json:"message"`
	Images  []string               `json:"images,omitempty"`
	Stream  bool                   `json:"stream,omitempty"`
	Options map[string]interface{} `json:"options,omitempty"`
}

// ChatResponse represents a chat inference response
type ChatResponse struct {
	Message string `json:"message,omitempty"`
	Token   string `json:"token,omitempty"`
	Done    bool   `json:"done"`
	Error   string `json:"error,omitempty"`
}

// HealthResponse represents the health check response
type HealthResponse struct {
	Status  string `json:"status"`
	Backend string `json:"backend,omitempty"`
	Uptime  string `json:"uptime"`
}

// HTTPServer represents an HTTP server that handles chat requests
type HTTPServer struct {
	Address        string
	Port           string
	httpServer     *http.Server
	messageChannel chan HTTPMessage
	mu             sync.RWMutex
	shutdown       chan struct{}
	startTime      time.Time
}

// HTTPMessage represents a single HTTP request with facilities to respond
type HTTPMessage struct {
	Content         string
	Images          []string
	Stream          bool
	Options         map[string]interface{}
	IsNativeBackend bool
	respond         func(ChatResponse) error
	stream          func(string) error
}

// Respond sends a response back to the client
func (m HTTPMessage) Respond(response string) error {
	return m.respond(ChatResponse{Message: response, Done: true})
}

// StreamToken sends a partial token back to the client
func (m HTTPMessage) StreamToken(token string) error {
	if m.stream != nil {
		return m.stream(token)
	}
	return nil
}

// RespondError sends an error back to the client
func (m HTTPMessage) RespondError(err error) error {
	if err == nil {
		return nil
	}
	return m.respond(ChatResponse{Error: err.Error(), Done: true})
}

// NewHTTPServer creates a new HTTP server instance
func NewHTTPServer(address, port string) *HTTPServer {
	return &HTTPServer{
		Address:        address,
		Port:           port,
		messageChannel: make(chan HTTPMessage, 100),
		shutdown:       make(chan struct{}),
		startTime:      time.Now(),
	}
}

// Start begins listening for HTTP requests
func (s *HTTPServer) Start(backend string) error {
	mux := http.NewServeMux()

	// Health endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		resp := HealthResponse{
			Status:  "ok",
			Backend: backend,
			Uptime:  time.Since(s.startTime).Round(time.Second).String(),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// Chat endpoint
	mux.HandleFunc("/v1/chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf(`{"error": "invalid JSON: %s"}`, err.Error()), http.StatusBadRequest)
			return
		}

		// Check if native backend (for image handling)
		isNative := strings.ToLower(backend) == "native"

		// Process images if needed for native backend
		images := req.Images
		if isNative && len(images) > 0 {
			processedImages := make([]string, 0, len(images))
			for _, img := range images {
				// Check if already a file path
				if _, err := os.Stat(img); err == nil {
					processedImages = append(processedImages, img)
					continue
				}

				// Assume base64, decode and save to temp file
				path, err := s.saveBase64ToTemp(img)
				if err != nil {
					log.Printf("Failed to save image: %v", err)
					continue
				}
				processedImages = append(processedImages, path)
			}
			images = processedImages
		}

		if req.Stream {
			s.handleStreaming(w, r, req, images, isNative)
		} else {
			s.handleNonStreaming(w, r, req, images, isNative)
		}

		// Cleanup temp files after processing
		if isNative {
			for _, path := range images {
				if strings.Contains(path, "openeye-vision-") {
					os.Remove(path)
				}
			}
		}
	})

	s.httpServer = &http.Server{
		Addr:    fmt.Sprintf("%s:%s", s.Address, s.Port),
		Handler: mux,
	}

	go func() {
		log.Printf("HTTP server starting on %s:%s", s.Address, s.Port)
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("HTTP server error: %v", err)
		}
	}()

	return nil
}

// handleNonStreaming processes non-streaming chat requests
func (s *HTTPServer) handleNonStreaming(w http.ResponseWriter, r *http.Request, req ChatRequest, images []string, isNative bool) {
	replyCh := make(chan ChatResponse, 1)

	msg := HTTPMessage{
		Content:         req.Message,
		Images:          images,
		Stream:          false,
		Options:         req.Options,
		IsNativeBackend: isNative,
		respond: func(resp ChatResponse) error {
			replyCh <- resp
			return nil
		},
	}

	select {
	case s.messageChannel <- msg:
		resp := <-replyCh
		w.Header().Set("Content-Type", "application/json")
		if resp.Error != "" {
			w.WriteHeader(http.StatusInternalServerError)
		}
		json.NewEncoder(w).Encode(resp)
	case <-s.shutdown:
		http.Error(w, `{"error": "server shutting down"}`, http.StatusServiceUnavailable)
	}
}

// handleStreaming processes streaming chat requests with SSE
func (s *HTTPServer) handleStreaming(w http.ResponseWriter, r *http.Request, req ChatRequest, images []string, isNative bool) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, `{"error": "streaming not supported"}`, http.StatusInternalServerError)
		return
	}

	replyCh := make(chan ChatResponse, 1)
	streamCh := make(chan string, 100)

	msg := HTTPMessage{
		Content:         req.Message,
		Images:          images,
		Stream:          true,
		Options:         req.Options,
		IsNativeBackend: isNative,
		respond: func(resp ChatResponse) error {
			replyCh <- resp
			return nil
		},
		stream: func(token string) error {
			select {
			case streamCh <- token:
				return nil
			case <-r.Context().Done():
				return r.Context().Err()
			}
		},
	}

	select {
	case s.messageChannel <- msg:
		// Stream tokens until final response
		var finalResponse ChatResponse
		streaming := true

		for streaming {
			select {
			case token := <-streamCh:
				event := ChatResponse{Token: token, Done: false}
				data, _ := json.Marshal(event)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()

			case resp := <-replyCh:
				finalResponse = resp
				streaming = false

			case <-r.Context().Done():
				return

			case <-s.shutdown:
				event := ChatResponse{Error: "server shutting down", Done: true}
				data, _ := json.Marshal(event)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
				return
			}
		}

		// Send final response
		data, _ := json.Marshal(finalResponse)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

	case <-s.shutdown:
		fmt.Fprintf(w, "data: %s\n\n", `{"error": "server shutting down", "done": true}`)
		flusher.Flush()
	}
}

// saveBase64ToTemp decodes base64 image data and saves it to a temporary file
func (s *HTTPServer) saveBase64ToTemp(base64Data string) (string, error) {
	// Remove data URI prefix if present
	if idx := strings.Index(base64Data, ","); idx != -1 {
		base64Data = base64Data[idx+1:]
	}

	// Decode base64
	data, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		return "", fmt.Errorf("failed to decode base64: %w", err)
	}

	// Create temp file
	tmpFile, err := os.CreateTemp("", "openeye-vision-*.jpg")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer tmpFile.Close()

	// Write data
	if _, err := tmpFile.Write(data); err != nil {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("failed to write temp file: %w", err)
	}

	return tmpFile.Name(), nil
}

// Stop gracefully shuts down the HTTP server
func (s *HTTPServer) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	close(s.shutdown)

	if s.httpServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := s.httpServer.Shutdown(ctx); err != nil {
			return fmt.Errorf("failed to shutdown HTTP server: %w", err)
		}
		log.Printf("HTTP server on %s:%s stopped", s.Address, s.Port)
	}

	close(s.messageChannel)
	return nil
}

// Receive retrieves the next message from the message channel
func (s *HTTPServer) Receive() (HTTPMessage, error) {
	select {
	case msg, ok := <-s.messageChannel:
		if !ok {
			return HTTPMessage{}, fmt.Errorf("message channel closed")
		}
		return msg, nil
	case <-s.shutdown:
		return HTTPMessage{}, fmt.Errorf("server is shutting down")
	}
}

// ReceiveMessage retrieves just the content from the next message
func (s *HTTPServer) ReceiveMessage() (string, error) {
	msg, err := s.Receive()
	if err != nil {
		return "", err
	}
	return msg.Content, nil
}

// GetAddress returns the server address
func (s *HTTPServer) GetAddress() string {
	return s.Address
}

// GetPort returns the server port
func (s *HTTPServer) GetPort() string {
	return s.Port
}

// IsRunning returns true if the server is running
func (s *HTTPServer) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.httpServer != nil
}

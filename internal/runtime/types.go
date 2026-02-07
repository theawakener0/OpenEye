package runtime

import (
	"context"
	"errors"
	"time"
)

// ErrStreamingUnsupported is returned when an adapter cannot stream tokens.
var ErrStreamingUnsupported = errors.New("runtime: streaming not supported by adapter")

// Request captures a model prompt along with tunable generation options.
type Request struct {
	Prompt  string
	Image   []string
	Options GenerationOptions
}

// GenerationOptions maps to the most common inference controls for SLMs.
type GenerationOptions struct {
	MaxTokens     int
	Temperature   float64
	TopK          int
	TopP          float64
	MinP          float64
	RepeatPenalty float64
	RepeatLastN   int
	Stop          []string
}

// Response contains the final text plus optional statistics.
type Response struct {
	Text   string
	Stats  Stats
	Raw    any
	Finish string
}

// Stats summarises runtime execution characteristics.
type Stats struct {
	TokensEvaluated int
	TokensGenerated int
	TokensCached    int
	Duration        time.Duration

	// TTFT is the time-to-first-token: how long from request start until
	// the first generated token was produced. Critical for edge UX.
	TTFT time.Duration

	// PromptTPS is the prompt processing throughput (tokens/second).
	PromptTPS float64

	// GenerationTPS is the token generation throughput (tokens/second).
	GenerationTPS float64
}

// StreamEvent is emitted for each token or checkpoint during streaming.
type StreamEvent struct {
	Token string
	Index int
	Final bool
	Err   error

	// Stats is populated on the final event to report performance metrics.
	Stats *Stats
}

// StreamCallback is invoked for each StreamEvent while streaming results.
type StreamCallback func(StreamEvent) error

// Adapter is the contract runtime backends must implement.
type Adapter interface {
	Name() string
	Generate(ctx context.Context, req Request) (Response, error)
	Stream(ctx context.Context, req Request, cb StreamCallback) error
	Close() error
}

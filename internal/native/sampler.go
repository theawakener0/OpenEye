//go:build native

package native

/*
#include "binding.h"
*/
import "C"

// SamplerChain wraps a llama.cpp sampler chain that applies a sequence of
// sampling operations (temperature, top-k, top-p, penalties, etc.) to logits
// before selecting a token.
//
// The chain takes ownership of all added samplers and frees them on Close.
// Build the chain before the generation loop, then call Context.SampleToken.
type SamplerChain struct {
	handle C.oe_sampler_t
}

// SamplerOptions configures the sampling strategy.
type SamplerOptions struct {
	// Temperature for logit scaling. 0 = greedy (argmax).
	Temperature float32

	// TopK limits candidates to the top K tokens. 0 = disabled.
	TopK int32

	// TopP (nucleus) keeps tokens whose cumulative prob >= P. 1.0 = disabled.
	TopP float32

	// MinP discards tokens with prob < P * max_prob. 0.0 = disabled.
	MinP float32

	// RepeatPenalty penalizes recently used tokens. 1.0 = disabled.
	RepeatPenalty float32

	// RepeatLastN is how many recent tokens to consider for penalty. 0 = disabled.
	RepeatLastN int32

	// FrequencyPenalty penalizes tokens by their frequency. 0.0 = disabled.
	FrequencyPenalty float32

	// PresencePenalty penalizes tokens that appeared at all. 0.0 = disabled.
	PresencePenalty float32

	// Seed for random sampling. 0xFFFFFFFF = random seed.
	Seed uint32
}

// DefaultSamplerOptions returns balanced defaults suitable for chat.
func DefaultSamplerOptions() SamplerOptions {
	return SamplerOptions{
		Temperature:      0.7,
		TopK:             40,
		TopP:             0.95,
		MinP:             0.05,
		RepeatPenalty:    1.1,
		RepeatLastN:      64,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		Seed:             0xFFFFFFFF, // random
	}
}

// NewSamplerChain creates a sampler chain from the given options.
// The chain is built in the canonical order:
//
//	penalties -> top-k -> top-p -> min-p -> temperature -> dist/greedy
//
// This ordering follows llama.cpp's recommended pipeline: filter first,
// scale second, sample last.
func NewSamplerChain(opts SamplerOptions) *SamplerChain {
	chain := cSamplerChainNew()

	// 1. Repetition/frequency/presence penalties (before any filtering).
	if opts.RepeatLastN > 0 && (opts.RepeatPenalty != 1.0 ||
		opts.FrequencyPenalty != 0.0 || opts.PresencePenalty != 0.0) {
		cSamplerChainAddPenalties(chain, opts.RepeatLastN,
			opts.RepeatPenalty, opts.FrequencyPenalty, opts.PresencePenalty)
	}

	// 2. Top-K filtering.
	if opts.TopK > 0 {
		cSamplerChainAddTopK(chain, opts.TopK)
	}

	// 3. Top-P (nucleus) filtering.
	if opts.TopP > 0.0 && opts.TopP < 1.0 {
		cSamplerChainAddTopP(chain, opts.TopP)
	}

	// 4. Min-P filtering.
	if opts.MinP > 0.0 {
		cSamplerChainAddMinP(chain, opts.MinP)
	}

	// 5. Temperature scaling.
	if opts.Temperature > 0.0 {
		cSamplerChainAddTemp(chain, opts.Temperature)
	}

	// 6. Final selection: greedy if temp=0, otherwise random distribution.
	if opts.Temperature <= 0.0 {
		cSamplerChainAddGreedy(chain)
	} else {
		cSamplerChainAddDist(chain, opts.Seed)
	}

	return &SamplerChain{handle: chain}
}

// Reset clears the sampler's internal state (e.g., penalty token history).
// Call between independent generation requests.
func (s *SamplerChain) Reset() {
	if s.handle != nil {
		cSamplerReset(s.handle)
	}
}

// Close frees the sampler chain and all samplers it owns.
func (s *SamplerChain) Close() {
	if s.handle != nil {
		cSamplerFree(s.handle)
		s.handle = nil
	}
}

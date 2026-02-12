//go:build native

package native

import (
	"log"
	"strings"

	"OpenEye/internal/config"
)

// ModelPreset holds optimized default settings for a known model family.
// These are applied as fallback values — explicit user configuration always
// takes precedence. The goal is to let users point at a GGUF file and get
// sensible edge-optimized inference without manual tuning.
type ModelPreset struct {
	// Display name for logging.
	Name string

	// Context/inference defaults (applied if user value is 0).
	ContextSize    uint32
	BatchSize      uint32
	Threads        int32
	ThreadsBatch   int32
	FlashAttention int32 // -1=auto, 0=off, 1=on

	// Generation defaults (applied if user value is 0).
	MaxTokens     int
	Temperature   float64
	TopK          int
	TopP          float64
	MinP          float64
	RepeatPenalty float64
	RepeatLastN   int

	// Stop sequences specific to this model's chat template.
	Stop []string

	// Whether warmup is recommended for this model.
	WarmupRecommended bool
}

// presetEntry pairs a match key with its preset for ordered iteration.
type presetEntry struct {
	key    string
	preset ModelPreset
}

// knownPresets lists model presets in match priority order. Matching is done
// by checking if any key is a case-insensitive substring of the model
// description or filename. Ordering matters: more specific patterns (e.g.
// "qwen2.5 0.5b") should appear before less specific ones so they match
// first. Using a slice instead of a map ensures deterministic iteration.
//
// These presets are based on empirical testing on Raspberry Pi 5 (4GB)
// and similar edge devices. Values target the sweet spot between quality
// and speed for each model family.
var knownPresets = []presetEntry{
	// --- Qwen2.5 family (Alibaba) ---
	{"qwen2.5 0.5b", ModelPreset{
		Name:              "Qwen2.5-0.5B",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         512,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.1,
		RepeatLastN:       64,
		Stop:              []string{"<|im_end|>", "<|endoftext|>"},
		WarmupRecommended: true,
	}},
	{"qwen2.5 1.5b", ModelPreset{
		Name:              "Qwen2.5-1.5B",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         512,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.1,
		RepeatLastN:       64,
		Stop:              []string{"<|im_end|>", "<|endoftext|>"},
		WarmupRecommended: true,
	}},
	{"qwen2.5 3b", ModelPreset{
		Name:              "Qwen2.5-3B",
		ContextSize:       2048,
		BatchSize:         256,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         384,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.1,
		RepeatLastN:       64,
		Stop:              []string{"<|im_end|>", "<|endoftext|>"},
		WarmupRecommended: true,
	}},

	// --- Llama 3.2 family (Meta) ---
	{"llama 3.2 1b", ModelPreset{
		Name:              "Llama-3.2-1B",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         512,
		Temperature:       0.6,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.1,
		RepeatLastN:       64,
		Stop:              []string{"<|eot_id|>", "<|end_of_text|>"},
		WarmupRecommended: true,
	}},
	{"llama 3.2 3b", ModelPreset{
		Name:              "Llama-3.2-3B",
		ContextSize:       2048,
		BatchSize:         256,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         384,
		Temperature:       0.6,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.1,
		RepeatLastN:       64,
		Stop:              []string{"<|eot_id|>", "<|end_of_text|>"},
		WarmupRecommended: true,
	}},

	// --- SmolLM2 family (Hugging Face) ---
	{"smollm2 135m", ModelPreset{
		Name:              "SmolLM2-135M",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         512,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.15,
		RepeatLastN:       64,
		Stop:              []string{"<|im_end|>", "<|endoftext|>"},
		WarmupRecommended: true,
	}},
	{"smollm2 360m", ModelPreset{
		Name:              "SmolLM2-360M",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         512,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.15,
		RepeatLastN:       64,
		Stop:              []string{"<|im_end|>", "<|endoftext|>"},
		WarmupRecommended: true,
	}},
	{"smollm2 1.7b", ModelPreset{
		Name:              "SmolLM2-1.7B",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         512,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.15,
		RepeatLastN:       64,
		Stop:              []string{"<|im_end|>", "<|endoftext|>"},
		WarmupRecommended: true,
	}},

	// --- Gemma 2 family (Google) ---
	{"gemma 2 2b", ModelPreset{
		Name:              "Gemma-2-2B",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         512,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.1,
		RepeatLastN:       64,
		Stop:              []string{"<end_of_turn>", "<eos>"},
		WarmupRecommended: true,
	}},

	// --- Phi family (Microsoft) ---
	{"phi-3.5-mini", ModelPreset{
		Name:              "Phi-3.5-Mini",
		ContextSize:       2048,
		BatchSize:         256,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    1,
		MaxTokens:         384,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.1,
		RepeatLastN:       64,
		Stop:              []string{"<|end|>", "<|endoftext|>"},
		WarmupRecommended: true,
	}},

	// --- TinyLlama (community) ---
	{"tinyllama", ModelPreset{
		Name:              "TinyLlama-1.1B",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      4,
		FlashAttention:    -1,
		MaxTokens:         512,
		Temperature:       0.7,
		TopK:              40,
		TopP:              0.9,
		MinP:              0.05,
		RepeatPenalty:     1.1,
		RepeatLastN:       64,
		Stop:              []string{"</s>"},
		WarmupRecommended: true,
	}},

	// --- Embedding models ---
	// Embedding models use fewer batch threads since they only do
	// forward passes without autoregressive generation.
	{"nomic-embed", ModelPreset{
		Name:              "Nomic-Embed",
		ContextSize:       2048,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      2,
		FlashAttention:    -1,
		WarmupRecommended: false,
	}},
	{"all-minilm", ModelPreset{
		Name:              "All-MiniLM",
		ContextSize:       512,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      2,
		FlashAttention:    -1,
		WarmupRecommended: false,
	}},
	{"bge-small", ModelPreset{
		Name:              "BGE-Small",
		ContextSize:       512,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      2,
		FlashAttention:    -1,
		WarmupRecommended: false,
	}},
	{"bge-micro", ModelPreset{
		Name:              "BGE-Micro",
		ContextSize:       512,
		BatchSize:         512,
		Threads:           4,
		ThreadsBatch:      2,
		FlashAttention:    -1,
		WarmupRecommended: false,
	}},
}

// matchPreset tries to find a preset matching the model description or
// filename. Returns the preset and true if found, or zero preset and false.
// Matching is case-insensitive substring search.
func matchPreset(description, filePath string) (ModelPreset, bool) {
	// Normalize: lowercase, replace common separators with spaces.
	norm := func(s string) string {
		s = strings.ToLower(s)
		s = strings.ReplaceAll(s, "-", " ")
		s = strings.ReplaceAll(s, "_", " ")
		return s
	}

	desc := norm(description)
	file := norm(filePath)

	for _, entry := range knownPresets {
		if strings.Contains(desc, entry.key) || strings.Contains(file, entry.key) {
			return entry.preset, true
		}
	}
	return ModelPreset{}, false
}

// applyPresetToContextOpts fills in zero-valued context options from a preset.
func applyPresetToContextOpts(opts *ContextOptions, preset ModelPreset) {
	if opts.NCtx == 0 && preset.ContextSize > 0 {
		opts.NCtx = preset.ContextSize
	}
	if opts.NBatch == 0 && preset.BatchSize > 0 {
		opts.NBatch = preset.BatchSize
	}
	if opts.NThreads == 0 && preset.Threads > 0 {
		opts.NThreads = preset.Threads
	}
	if opts.NThreadsBatch == 0 && preset.ThreadsBatch > 0 {
		opts.NThreadsBatch = preset.ThreadsBatch
	}
	if opts.FlashAttn == -1 && preset.FlashAttention != -1 {
		opts.FlashAttn = preset.FlashAttention
	}
}

// applyPresetToDefaults fills in zero-valued generation defaults from a preset.
func applyPresetToDefaults(defaults *config.GenerationDefaults, preset ModelPreset) {
	if defaults.MaxTokens == 0 && preset.MaxTokens > 0 {
		defaults.MaxTokens = preset.MaxTokens
	}
	if defaults.Temperature == 0 && preset.Temperature > 0 {
		defaults.Temperature = preset.Temperature
	}
	if defaults.TopK == 0 && preset.TopK > 0 {
		defaults.TopK = preset.TopK
	}
	if defaults.TopP == 0 && preset.TopP > 0 {
		defaults.TopP = preset.TopP
	}
	if defaults.MinP == 0 && preset.MinP > 0 {
		defaults.MinP = preset.MinP
	}
	if defaults.RepeatPenalty == 0 && preset.RepeatPenalty > 0 {
		defaults.RepeatPenalty = preset.RepeatPenalty
	}
	if defaults.RepeatLastN == 0 && preset.RepeatLastN > 0 {
		defaults.RepeatLastN = preset.RepeatLastN
	}
	if len(defaults.Stop) == 0 && len(preset.Stop) > 0 {
		defaults.Stop = append([]string(nil), preset.Stop...)
	}
}

// logPreset logs which preset was auto-detected for the loaded model.
func logPreset(preset ModelPreset, source string) {
	log.Printf("native: auto-detected model preset %q (matched from %s)", preset.Name, source)
}

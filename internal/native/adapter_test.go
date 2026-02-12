//go:build native

package native

import (
	"math/rand"
	"strings"
	"testing"

	"OpenEye/internal/config"
	"OpenEye/internal/runtime"
)

// ---------------------------------------------------------------------------
// kvCacheTypeFromString
// ---------------------------------------------------------------------------

func TestKVCacheTypeFromString(t *testing.T) {
	tests := []struct {
		input string
		want  int32
	}{
		{"f16", 0},
		{"F16", 0},
		{"", 0},
		{"unknown", 0},
		{"q8_0", 1},
		{"Q8_0", 1},
		{"q8", 1},
		{"q4_0", 2},
		{"Q4_0", 2},
		{"q4", 2},
		{"  q8_0  ", 1}, // whitespace
	}
	for _, tt := range tests {
		got := kvCacheTypeFromString(tt.input)
		if got != tt.want {
			t.Errorf("kvCacheTypeFromString(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// argmaxToken
// ---------------------------------------------------------------------------

func TestArgmaxToken(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		got := argmaxToken(nil)
		if got != 0 {
			t.Errorf("argmaxToken(nil) = %d, want 0", got)
		}
	})

	t.Run("single element", func(t *testing.T) {
		got := argmaxToken([]float32{1.0})
		if got != 0 {
			t.Errorf("argmaxToken([1.0]) = %d, want 0", got)
		}
	})

	t.Run("max at end", func(t *testing.T) {
		got := argmaxToken([]float32{1.0, 2.0, 3.0, 4.0, 5.0})
		if got != 4 {
			t.Errorf("argmaxToken = %d, want 4", got)
		}
	})

	t.Run("max at start", func(t *testing.T) {
		got := argmaxToken([]float32{9.0, 2.0, 3.0, 4.0, 5.0})
		if got != 0 {
			t.Errorf("argmaxToken = %d, want 0", got)
		}
	})

	t.Run("max in middle", func(t *testing.T) {
		got := argmaxToken([]float32{1.0, 2.0, 99.0, 4.0, 5.0})
		if got != 2 {
			t.Errorf("argmaxToken = %d, want 2", got)
		}
	})

	t.Run("negative values", func(t *testing.T) {
		got := argmaxToken([]float32{-5.0, -3.0, -10.0, -1.0})
		if got != 3 {
			t.Errorf("argmaxToken = %d, want 3 (largest negative)", got)
		}
	})

	t.Run("duplicate max", func(t *testing.T) {
		// Should return first occurrence.
		got := argmaxToken([]float32{1.0, 5.0, 5.0, 3.0})
		if got != 1 {
			t.Errorf("argmaxToken = %d, want 1 (first occurrence)", got)
		}
	})
}

// ---------------------------------------------------------------------------
// commonPrefixLen
// ---------------------------------------------------------------------------

func TestCommonPrefixLen(t *testing.T) {
	tests := []struct {
		name string
		a, b []int32
		want int
	}{
		{"both empty", nil, nil, 0},
		{"a empty", nil, []int32{1, 2}, 0},
		{"b empty", []int32{1, 2}, nil, 0},
		{"identical", []int32{1, 2, 3}, []int32{1, 2, 3}, 3},
		{"partial match", []int32{1, 2, 3, 4}, []int32{1, 2, 5, 6}, 2},
		{"no match", []int32{1, 2}, []int32{3, 4}, 0},
		{"a shorter", []int32{1, 2}, []int32{1, 2, 3, 4}, 2},
		{"b shorter", []int32{1, 2, 3, 4}, []int32{1, 2}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := commonPrefixLen(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("commonPrefixLen = %d, want %d", got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// shouldStop
// ---------------------------------------------------------------------------

func TestShouldStop(t *testing.T) {
	tests := []struct {
		text  string
		stops []string
		want  bool
	}{
		{"hello world<|end|>", []string{"<|end|>"}, true},
		{"hello world", []string{"<|end|>"}, false},
		{"hello world\n", []string{"\n", "<|end|>"}, true},
		{"hello world", nil, false},
		{"hello world", []string{}, false},
		{"", []string{"x"}, false},
	}
	for _, tt := range tests {
		got := shouldStop(tt.text, tt.stops)
		if got != tt.want {
			t.Errorf("shouldStop(%q, %v) = %v, want %v", tt.text, tt.stops, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// trimAtStop
// ---------------------------------------------------------------------------

func TestTrimAtStop(t *testing.T) {
	tests := []struct {
		text  string
		stops []string
		want  string
	}{
		{"hello<|end|>world", []string{"<|end|>"}, "hello"},
		{"hello world", []string{"<|end|>"}, "hello world"},
		{"hello\nworld\n", []string{"\n"}, "hello"},
		{"abc", nil, "abc"},
		{"hello<|a|>world<|b|>", []string{"<|a|>", "<|b|>"}, "hello"},
	}
	for _, tt := range tests {
		got := trimAtStop(tt.text, tt.stops)
		if got != tt.want {
			t.Errorf("trimAtStop(%q, %v) = %q, want %q", tt.text, tt.stops, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// stopRing
// ---------------------------------------------------------------------------

func TestStopRingNil(t *testing.T) {
	r := newStopRing(nil)
	if r != nil {
		t.Error("newStopRing(nil) should return nil")
	}
	r = newStopRing([]string{})
	if r != nil {
		t.Error("newStopRing([]) should return nil")
	}
	r = newStopRing([]string{""})
	if r != nil {
		t.Error("newStopRing(['']) should return nil")
	}
}

func TestStopRingBasic(t *testing.T) {
	r := newStopRing([]string{"<|end|>"})
	if r == nil {
		t.Fatal("newStopRing returned nil")
	}

	// Write text that doesn't contain stop.
	r.write("hello world")
	if r.check() {
		t.Error("check() returned true before stop sequence")
	}

	// Now write the stop sequence character by character.
	r.write("<|end|>")
	if !r.check() {
		t.Error("check() returned false after writing stop sequence")
	}
}

func TestStopRingTokenByToken(t *testing.T) {
	r := newStopRing([]string{"<|end|>"})

	tokens := []string{"hello", " ", "world", "<|", "end", "|>"}
	for i, tok := range tokens {
		r.write(tok)
		shouldMatch := (i == len(tokens)-1)
		if r.check() != shouldMatch {
			t.Errorf("after writing %q (token %d), check()=%v, want %v",
				tok, i, r.check(), shouldMatch)
		}
	}
}

func TestStopRingMultipleStops(t *testing.T) {
	r := newStopRing([]string{"</s>", "<|end|>"})

	r.write("hello</s>")
	if !r.check() {
		t.Error("should match </s>")
	}

	// New ring for the other stop.
	r2 := newStopRing([]string{"</s>", "<|end|>"})
	r2.write("hello<|end|>")
	if !r2.check() {
		t.Error("should match <|end|>")
	}
}

func TestStopRingBufferOverflow(t *testing.T) {
	// Stop sequence is 7 bytes, write 1000 bytes of content first.
	r := newStopRing([]string{"<|end|>"})
	r.write(strings.Repeat("abcdefghij", 100)) // 1000 bytes
	if r.check() {
		t.Error("shouldn't match after 1000 bytes of non-stop content")
	}

	r.write("<|end|>")
	if !r.check() {
		t.Error("should match even after 1000 bytes of prior content")
	}
}

// ---------------------------------------------------------------------------
// mergeNativeOptions
// ---------------------------------------------------------------------------

func TestMergeNativeOptions(t *testing.T) {
	base := config.GenerationDefaults{
		MaxTokens:   512,
		Temperature: 0.2,
		TopK:        40,
		Stop:        []string{"<|end|>"},
	}

	t.Run("override temperature", func(t *testing.T) {
		override := runtime.GenerationOptions{Temperature: 0.8}
		result := mergeNativeOptions(base, override)
		if result.Temperature != 0.8 {
			t.Errorf("Temperature = %f, want 0.8", result.Temperature)
		}
		if result.MaxTokens != 512 {
			t.Errorf("MaxTokens = %d, want 512", result.MaxTokens)
		}
	})

	t.Run("override stop sequences", func(t *testing.T) {
		override := runtime.GenerationOptions{Stop: []string{"</s>"}}
		result := mergeNativeOptions(base, override)
		if len(result.Stop) != 1 || result.Stop[0] != "</s>" {
			t.Errorf("Stop = %v, want [</s>]", result.Stop)
		}
	})

	t.Run("zero override preserves base", func(t *testing.T) {
		override := runtime.GenerationOptions{}
		result := mergeNativeOptions(base, override)
		if result.MaxTokens != 512 {
			t.Errorf("MaxTokens = %d, want 512", result.MaxTokens)
		}
		if result.Temperature != 0.2 {
			t.Errorf("Temperature = %f, want 0.2", result.Temperature)
		}
		if result.TopK != 40 {
			t.Errorf("TopK = %d, want 40", result.TopK)
		}
	})
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkArgmaxToken(b *testing.B) {
	// Simulate a vocab of 32000 tokens (typical for small LLMs).
	logits := make([]float32, 32000)
	r := rand.New(rand.NewSource(42))
	for i := range logits {
		logits[i] = r.Float32()*20 - 10 // range [-10, 10]
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		argmaxToken(logits)
	}
}

func BenchmarkArgmaxTokenLargeVocab(b *testing.B) {
	// 128k vocab (some newer models).
	logits := make([]float32, 128000)
	r := rand.New(rand.NewSource(42))
	for i := range logits {
		logits[i] = r.Float32()*20 - 10
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		argmaxToken(logits)
	}
}

func BenchmarkStopRingWrite(b *testing.B) {
	ring := newStopRing([]string{"<|im_end|>", "<|endoftext|>", "</s>"})
	token := "Hello" // typical token

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ring.write(token)
	}
}

func BenchmarkStopRingCheck(b *testing.B) {
	ring := newStopRing([]string{"<|im_end|>", "<|endoftext|>", "</s>"})
	// Fill ring with some content.
	ring.write("The quick brown fox jumps over the lazy dog. This is a test. ")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ring.check()
	}
}

func BenchmarkStopRingWriteAndCheck(b *testing.B) {
	stops := []string{"<|im_end|>", "<|endoftext|>", "</s>"}
	ring := newStopRing(stops)
	token := "Hello"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ring.write(token)
		ring.check()
	}
}

func BenchmarkShouldStopNaive(b *testing.B) {
	// Benchmark the naive O(n) HasSuffix approach for comparison.
	stops := []string{"<|im_end|>", "<|endoftext|>", "</s>"}
	text := strings.Repeat("Hello world this is a test. ", 100) // ~2800 chars

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		shouldStop(text, stops)
	}
}

func BenchmarkCommonPrefixLen(b *testing.B) {
	// Typical prompt: 500 tokens, 450 in common (conversational reuse).
	a := make([]int32, 500)
	b2 := make([]int32, 500)
	for i := range a {
		a[i] = int32(i)
		b2[i] = int32(i)
	}
	// Diverge at position 450.
	for i := 450; i < 500; i++ {
		b2[i] = int32(i + 1000)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		commonPrefixLen(a, b2)
	}
}

package config

import (
	"testing"
)

// TestMergeNativeOptimizationFields verifies that the new inference
// optimization fields (DraftModelPath, SpeculativeN, KVCacheType,
// StreamChunkSize, ContextShift) are correctly merged from overrides.
func TestMergeNativeOptimizationFields(t *testing.T) {
	base := Config{}
	base.Runtime.Native.ModelPath = "base-model.gguf"
	base.Runtime.Native.ContextSize = 2048
	base.Runtime.Native.Threads = 4

	t.Run("DraftModelPath override", func(t *testing.T) {
		override := Config{}
		override.Runtime.Native.DraftModelPath = "draft.gguf"
		result := merge(base, override)
		if result.Runtime.Native.DraftModelPath != "draft.gguf" {
			t.Errorf("DraftModelPath = %q, want %q", result.Runtime.Native.DraftModelPath, "draft.gguf")
		}
		// Base fields preserved.
		if result.Runtime.Native.ModelPath != "base-model.gguf" {
			t.Errorf("ModelPath lost: got %q", result.Runtime.Native.ModelPath)
		}
	})

	t.Run("DraftModelPath not overridden when empty", func(t *testing.T) {
		baseCfg := Config{}
		baseCfg.Runtime.Native.DraftModelPath = "original-draft.gguf"
		override := Config{} // empty override
		result := merge(baseCfg, override)
		if result.Runtime.Native.DraftModelPath != "original-draft.gguf" {
			t.Errorf("DraftModelPath = %q, want %q", result.Runtime.Native.DraftModelPath, "original-draft.gguf")
		}
	})

	t.Run("SpeculativeN override", func(t *testing.T) {
		override := Config{}
		override.Runtime.Native.SpeculativeN = 8
		result := merge(base, override)
		if result.Runtime.Native.SpeculativeN != 8 {
			t.Errorf("SpeculativeN = %d, want 8", result.Runtime.Native.SpeculativeN)
		}
	})

	t.Run("SpeculativeN not overridden when zero", func(t *testing.T) {
		baseCfg := Config{}
		baseCfg.Runtime.Native.SpeculativeN = 5
		override := Config{}
		result := merge(baseCfg, override)
		if result.Runtime.Native.SpeculativeN != 5 {
			t.Errorf("SpeculativeN = %d, want 5", result.Runtime.Native.SpeculativeN)
		}
	})

	t.Run("KVCacheType override", func(t *testing.T) {
		override := Config{}
		override.Runtime.Native.KVCacheType = "q8_0"
		result := merge(base, override)
		if result.Runtime.Native.KVCacheType != "q8_0" {
			t.Errorf("KVCacheType = %q, want %q", result.Runtime.Native.KVCacheType, "q8_0")
		}
	})

	t.Run("KVCacheType not overridden when empty", func(t *testing.T) {
		baseCfg := Config{}
		baseCfg.Runtime.Native.KVCacheType = "q4_0"
		override := Config{}
		result := merge(baseCfg, override)
		if result.Runtime.Native.KVCacheType != "q4_0" {
			t.Errorf("KVCacheType = %q, want %q", result.Runtime.Native.KVCacheType, "q4_0")
		}
	})

	t.Run("StreamChunkSize override", func(t *testing.T) {
		override := Config{}
		override.Runtime.Native.StreamChunkSize = 5
		result := merge(base, override)
		if result.Runtime.Native.StreamChunkSize != 5 {
			t.Errorf("StreamChunkSize = %d, want 5", result.Runtime.Native.StreamChunkSize)
		}
	})

	t.Run("StreamChunkSize not overridden when zero", func(t *testing.T) {
		baseCfg := Config{}
		baseCfg.Runtime.Native.StreamChunkSize = 3
		override := Config{}
		result := merge(baseCfg, override)
		if result.Runtime.Native.StreamChunkSize != 3 {
			t.Errorf("StreamChunkSize = %d, want 3", result.Runtime.Native.StreamChunkSize)
		}
	})

	t.Run("ContextShift override to false", func(t *testing.T) {
		f := false
		override := Config{}
		override.Runtime.Native.ContextShift = &f
		result := merge(base, override)
		if result.Runtime.Native.ContextShift == nil {
			t.Fatal("ContextShift is nil, want *false")
		}
		if *result.Runtime.Native.ContextShift != false {
			t.Errorf("ContextShift = %v, want false", *result.Runtime.Native.ContextShift)
		}
	})

	t.Run("ContextShift override to true", func(t *testing.T) {
		tr := true
		override := Config{}
		override.Runtime.Native.ContextShift = &tr
		result := merge(base, override)
		if result.Runtime.Native.ContextShift == nil {
			t.Fatal("ContextShift is nil, want *true")
		}
		if *result.Runtime.Native.ContextShift != true {
			t.Errorf("ContextShift = %v, want true", *result.Runtime.Native.ContextShift)
		}
	})

	t.Run("ContextShift not overridden when nil", func(t *testing.T) {
		tr := true
		baseCfg := Config{}
		baseCfg.Runtime.Native.ContextShift = &tr
		override := Config{}
		result := merge(baseCfg, override)
		if result.Runtime.Native.ContextShift == nil {
			t.Fatal("ContextShift lost on merge")
		}
		if *result.Runtime.Native.ContextShift != true {
			t.Errorf("ContextShift = %v, want true", *result.Runtime.Native.ContextShift)
		}
	})

	t.Run("all optimization fields together", func(t *testing.T) {
		f := false
		override := Config{}
		override.Runtime.Native.DraftModelPath = "small.gguf"
		override.Runtime.Native.SpeculativeN = 6
		override.Runtime.Native.KVCacheType = "q4_0"
		override.Runtime.Native.StreamChunkSize = 4
		override.Runtime.Native.ContextShift = &f

		result := merge(base, override)
		if result.Runtime.Native.DraftModelPath != "small.gguf" {
			t.Errorf("DraftModelPath = %q", result.Runtime.Native.DraftModelPath)
		}
		if result.Runtime.Native.SpeculativeN != 6 {
			t.Errorf("SpeculativeN = %d", result.Runtime.Native.SpeculativeN)
		}
		if result.Runtime.Native.KVCacheType != "q4_0" {
			t.Errorf("KVCacheType = %q", result.Runtime.Native.KVCacheType)
		}
		if result.Runtime.Native.StreamChunkSize != 4 {
			t.Errorf("StreamChunkSize = %d", result.Runtime.Native.StreamChunkSize)
		}
		if result.Runtime.Native.ContextShift == nil || *result.Runtime.Native.ContextShift != false {
			t.Errorf("ContextShift not false")
		}
		// Verify base fields survived.
		if result.Runtime.Native.ModelPath != "base-model.gguf" {
			t.Errorf("ModelPath = %q", result.Runtime.Native.ModelPath)
		}
		if result.Runtime.Native.ContextSize != 2048 {
			t.Errorf("ContextSize = %d", result.Runtime.Native.ContextSize)
		}
		if result.Runtime.Native.Threads != 4 {
			t.Errorf("Threads = %d", result.Runtime.Native.Threads)
		}
	})
}

// TestMergePreservesExistingNativeFields ensures the merge function
// still correctly handles pre-existing Native fields.
func TestMergePreservesExistingNativeFields(t *testing.T) {
	base := Config{}
	base.Runtime.Native.ModelPath = "model.gguf"
	base.Runtime.Native.ContextSize = 4096
	base.Runtime.Native.BatchSize = 512
	base.Runtime.Native.Threads = 4
	base.Runtime.Native.ThreadsBatch = 4

	override := Config{}
	override.Runtime.Native.ContextSize = 2048

	result := merge(base, override)
	if result.Runtime.Native.ModelPath != "model.gguf" {
		t.Errorf("ModelPath = %q, want model.gguf", result.Runtime.Native.ModelPath)
	}
	if result.Runtime.Native.ContextSize != 2048 {
		t.Errorf("ContextSize = %d, want 2048", result.Runtime.Native.ContextSize)
	}
	if result.Runtime.Native.BatchSize != 512 {
		t.Errorf("BatchSize = %d, want 512", result.Runtime.Native.BatchSize)
	}
	if result.Runtime.Native.Threads != 4 {
		t.Errorf("Threads = %d, want 4", result.Runtime.Native.Threads)
	}
	if result.Runtime.Native.ThreadsBatch != 4 {
		t.Errorf("ThreadsBatch = %d, want 4", result.Runtime.Native.ThreadsBatch)
	}
}

// TestMergeGenerationDefaults checks that generation defaults merge correctly.
func TestMergeGenerationDefaults(t *testing.T) {
	base := Config{}
	base.Runtime.Defaults.MaxTokens = 512
	base.Runtime.Defaults.Temperature = 0.2
	base.Runtime.Defaults.TopK = 40
	base.Runtime.Defaults.Stop = []string{"<|end|>"}

	override := Config{}
	override.Runtime.Defaults.Temperature = 0.8

	result := merge(base, override)
	if result.Runtime.Defaults.MaxTokens != 512 {
		t.Errorf("MaxTokens = %d, want 512", result.Runtime.Defaults.MaxTokens)
	}
	if result.Runtime.Defaults.Temperature != 0.8 {
		t.Errorf("Temperature = %f, want 0.8", result.Runtime.Defaults.Temperature)
	}
	if result.Runtime.Defaults.TopK != 40 {
		t.Errorf("TopK = %d, want 40", result.Runtime.Defaults.TopK)
	}
	if len(result.Runtime.Defaults.Stop) != 1 || result.Runtime.Defaults.Stop[0] != "<|end|>" {
		t.Errorf("Stop = %v, want [<|end|>]", result.Runtime.Defaults.Stop)
	}
}

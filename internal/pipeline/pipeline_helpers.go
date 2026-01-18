package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"OpenEye/internal/config"
	"OpenEye/internal/embedding"
	"OpenEye/internal/rag"
	"OpenEye/internal/runtime"
)

func mergeOptions(defaults config.GenerationDefaults, hints runtime.GenerationOptions) runtime.GenerationOptions {
	result := runtime.GenerationOptions{
		MaxTokens:     defaults.MaxTokens,
		Temperature:   defaults.Temperature,
		TopK:          defaults.TopK,
		TopP:          defaults.TopP,
		MinP:          defaults.MinP,
		RepeatPenalty: defaults.RepeatPenalty,
		RepeatLastN:   defaults.RepeatLastN,
	}

	if len(hints.Stop) > 0 {
		result.Stop = append([]string(nil), hints.Stop...)
	}
	if hints.MaxTokens != 0 {
		result.MaxTokens = hints.MaxTokens
	}
	if hints.Temperature != 0 {
		result.Temperature = hints.Temperature
	}
	if hints.TopK != 0 {
		result.TopK = hints.TopK
	}
	if hints.TopP != 0 {
		result.TopP = hints.TopP
	}
	if hints.MinP != 0 {
		result.MinP = hints.MinP
	}
	if hints.RepeatPenalty != 0 {
		result.RepeatPenalty = hints.RepeatPenalty
	}
	if hints.RepeatLastN != 0 {
		result.RepeatLastN = hints.RepeatLastN
	}
	if hints.Stop == nil && defaults.Stop != nil {
		result.Stop = append([]string(nil), defaults.Stop...)
	}

	return result
}

// processImages processes input images using the configured image processor.
// Returns processed image data (base64 or file paths depending on config).
func (p *Pipeline) processImages(ctx context.Context, inputs []string) ([]string, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	// If no processor configured, return inputs as-is
	if p.imageProcessor == nil {
		return inputs, nil
	}

	processed, err := p.imageProcessor.ProcessMultiple(ctx, inputs)
	if err != nil {
		// Return original inputs on error, but log the warning
		return inputs, err
	}

	// Convert processed images to the format expected by the runtime
	// llama.cpp expects raw base64 data, not data URIs
	result := make([]string, 0, len(processed))
	for _, img := range processed {
		if img == nil {
			continue
		}
		// Return raw base64 data for llama.cpp
		if img.Base64 != "" {
			result = append(result, img.Base64)
		} else if img.OriginalPath != "" {
			// Fall back to original path if available
			result = append(result, img.OriginalPath)
		}
	}

	return result, nil
}

func initialiseRetriever(cfg config.Config, embedder embedding.Provider) (rag.Retriever, error) {
	if !cfg.RAG.Enabled {
		return nil, nil
	}

	// Use hybrid retriever if enabled
	if cfg.RAG.HybridEnabled {
		hybridCfg := rag.HybridRetrieverConfig{
			RAGConfig:            cfg.RAG,
			MaxCandidates:        cfg.RAG.MaxCandidates,
			DiversityThreshold:   cfg.RAG.DiversityThreshold,
			SemanticWeight:       cfg.RAG.SemanticWeight,
			KeywordWeight:        cfg.RAG.KeywordWeight,
			RecencyWeight:        cfg.RAG.RAGRecencyWeight,
			EnableQueryExpansion: cfg.RAG.EnableQueryExpansion,
			DedupeThreshold:      cfg.RAG.DedupeThreshold,
		}

		retriever, err := rag.NewHybridRetriever(hybridCfg, embedder)
		if err != nil {
			return nil, fmt.Errorf("pipeline: failed to initialise hybrid retriever: %w", err)
		}
		return retriever, nil
	}

	// Fall back to basic filesystem retriever
	retriever, err := rag.NewFilesystemRetriever(cfg.RAG, embedder)
	if err != nil {
		return nil, fmt.Errorf("pipeline: failed to initialise retriever: %w", err)
	}
	return retriever, nil
}

// addImageMarkers prepends image markers to the prompt for llama.cpp multimodal.
// The MTMD (multimodal) marker <__media__> must appear in the prompt for each
// image in multimodal_data. See llama.cpp mtmd_default_marker().
func addImageMarkers(message string, imageCount int) string {
	if imageCount == 0 {
		return message
	}
	var markers strings.Builder
	for i := 0; i < imageCount; i++ {
		markers.WriteString("<__media__>")
		if i < imageCount-1 {
			markers.WriteString(" ")
		}
	}
	markers.WriteString("\n")
	markers.WriteString(message)
	return markers.String()
}

func generateCacheKey(text string, images []string) string {
	hash := sha256.New()
	hash.Write([]byte(text))
	for _, img := range images {
		hash.Write([]byte(img))
	}
	return hex.EncodeToString(hash.Sum(nil))
}

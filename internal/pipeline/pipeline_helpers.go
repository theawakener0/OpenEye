package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"os"
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

	// Double check cache for each input before hitting the processor
	results := make([]string, 0, len(inputs))
	toProcess := make([]string, 0)
	processIndices := make([]int, 0)

	for i, input := range inputs {
		// Use a simple hash of the input (path or base64) to avoid long keys
		hasher := sha256.New()
		hasher.Write([]byte(input))
		key := hex.EncodeToString(hasher.Sum(nil))

		if cached, ok := p.imageCache.Load(key); ok {
			results = append(results, cached.(string))
		} else {
			results = append(results, "") // Placeholder
			toProcess = append(toProcess, input)
			processIndices = append(processIndices, i)
		}
	}

	if len(toProcess) == 0 {
		return results, nil
	}

	// If no processor configured, return inputs as-is
	if p.imageProcessor == nil {
		return inputs, nil
	}

	processed, err := p.imageProcessor.ProcessMultiple(ctx, toProcess)
	if err != nil {
		// Fallback for failed items
		for i, idx := range processIndices {
			results[idx] = toProcess[i]
		}
		return results, err
	}

	for i, img := range processed {
		idx := processIndices[i]
		if img == nil {
			results[idx] = toProcess[i]
			continue
		}

		var finalVal string
		if img.Base64 != "" {
			finalVal = img.Base64
		} else if img.Data != nil {
			tmpFile, err := os.CreateTemp("", "openeye-vision-*.jpg")
			if err != nil {
				log.Printf("warning: failed to create temp image file: %v", err)
				finalVal = toProcess[i]
			} else {
				if _, err := tmpFile.Write(img.Data); err != nil {
					log.Printf("warning: failed to write temp image: %v", err)
					os.Remove(tmpFile.Name())
					finalVal = toProcess[i]
				} else {
					tmpFile.Close()
					finalVal = tmpFile.Name()
				}
			}
		} else if img.OriginalPath != "" {
			finalVal = img.OriginalPath
		}

		if finalVal != "" {
			results[idx] = finalVal
			hasher := sha256.New()
			hasher.Write([]byte(toProcess[i]))
			key := hex.EncodeToString(hasher.Sum(nil))
			p.imageCache.Store(key, finalVal)
		} else {
			results[idx] = toProcess[i]
		}
	}

	return results, nil
}

func initialiseRetriever(cfg config.Config, embedder embedding.Provider) (rag.Retriever, error) {
	if !cfg.RAG.Enabled {
		log.Println("RAG disabled in config, skipping retriever initialization")
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

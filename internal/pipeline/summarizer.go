package pipeline

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"OpenEye/internal/config"
	conversation "OpenEye/internal/context"
	"OpenEye/internal/embedding"
	"OpenEye/internal/runtime"
)

type summarizer interface {
	Summarize(ctx context.Context, history []conversation.HistoryItem) (string, error)
}

type llmSummarizer struct {
	manager             *runtime.Manager
	prompt              string
	maxTokens           int
	embedder            embedding.Provider
	minTurns            int
	maxReferences       int
	similarityFloor     float64
	maxTranscriptTokens int
	cache               sync.Map
}

func newSummarizer(manager *runtime.Manager, cfg config.SummarizerConfig, embedder embedding.Provider) summarizer {
	if manager == nil || !cfg.Enabled {
		return nil
	}
	prompt := strings.TrimSpace(cfg.Prompt)
	if prompt == "" {
		prompt = defaultSummaryPrompt
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 128
	}
	minTurns := cfg.MinTurns
	if minTurns < 0 {
		minTurns = 0
	}
	maxReferences := cfg.MaxReferences
	if maxReferences <= 0 {
		maxReferences = 8
	}
	floor := cfg.SimilarityThreshold
	if floor <= 0 {
		floor = 0.1
	}
	return &llmSummarizer{
		manager:             manager,
		prompt:              prompt,
		maxTokens:           maxTokens,
		embedder:            embedder,
		minTurns:            minTurns,
		maxReferences:       maxReferences,
		similarityFloor:     floor,
		maxTranscriptTokens: cfg.MaxTranscriptTokens,
	}
}

func (s *llmSummarizer) Summarize(ctx context.Context, history []conversation.HistoryItem) (string, error) {
	if s == nil || s.manager == nil {
		return "", nil
	}
	if len(history) == 0 {
		return "", nil
	}
	if s.minTurns > 0 && len(history) < s.minTurns {
		return "", nil
	}

	reference := history
	if s.embedder != nil {
		if subset, err := s.selectRelevant(ctx, history); err != nil {
			log.Printf("warning: summarizer selection failed: %v", err)
		} else if len(subset) != 0 {
			reference = subset
		}
	}

	var transcript strings.Builder
	for _, item := range reference {
		if item.Content == "" {
			continue
		}
		role := item.Role
		if role == "" {
			role = "user"
		}
		transcript.WriteString(role)
		transcript.WriteString(": ")
		transcript.WriteString(item.Content)
		transcript.WriteString("\n")
	}

	transcriptStr := transcript.String()
	if s.maxTranscriptTokens > 0 {
		runes := []rune(transcriptStr)
		maxRunes := s.maxTranscriptTokens * 4
		if maxRunes > 0 && len(runes) > maxRunes {
			runes = runes[len(runes)-maxRunes:]
			transcriptStr = string(runes)
		}
	}

	prompt := fmt.Sprintf("%s\n\n=== Conversation Transcript ===\n%s\n===============================\nSummary:", s.prompt, transcriptStr)

	req := runtime.Request{
		Prompt: prompt,
		Options: runtime.GenerationOptions{
			MaxTokens:   s.maxTokens,
			Temperature: 0.2,
			TopP:        0.8,
		},
	}

	resp, err := s.manager.Generate(ctx, req)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(resp.Text), nil
}

func (s *llmSummarizer) selectRelevant(ctx context.Context, history []conversation.HistoryItem) ([]conversation.HistoryItem, error) {
	if s.embedder == nil {
		return nil, nil
	}
	queryContent := lastUserMessage(history)
	if strings.TrimSpace(queryContent) == "" {
		return nil, nil
	}

	queryVec, err := s.embeddingWithCache(ctx, queryContent)
	if err != nil {
		return nil, err
	}

	maxReferences := s.maxReferences
	if maxReferences <= 0 {
		maxReferences = 8
	}
	threshold := s.similarityFloor
	if threshold <= 0 {
		threshold = 0.1
	}

	type scored struct {
		idx   int
		item  conversation.HistoryItem
		score float64
	}

	results := make([]scored, 0, len(history))
	for idx, item := range history {
		content := strings.TrimSpace(item.Content)
		if content == "" {
			continue
		}
		vec, embedErr := s.embeddingWithCache(ctx, content)
		if embedErr != nil {
			return nil, embedErr
		}
		score := cosine(queryVec, vec)
		if score < threshold && idx < len(history)-2 {
			continue
		}
		results = append(results, scored{idx: idx, item: item, score: score})
	}

	if len(results) == 0 {
		return nil, nil
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].score == results[j].score {
			return results[i].idx < results[j].idx
		}
		return results[i].score > results[j].score
	})

	if len(results) > maxReferences {
		results = results[:maxReferences]
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].idx < results[j].idx
	})

	selected := make([]conversation.HistoryItem, 0, len(results))
	for _, res := range results {
		selected = append(selected, res.item)
	}

	return selected, nil
}

func lastUserMessage(history []conversation.HistoryItem) string {
	for i := len(history) - 1; i >= 0; i-- {
		content := strings.TrimSpace(history[i].Content)
		if content == "" {
			continue
		}
		return content
	}
	return ""
}

func (s *llmSummarizer) embeddingWithCache(ctx context.Context, text string) ([]float32, error) {
	if s.embedder == nil {
		return nil, fmt.Errorf("summarizer: embedding provider not configured")
	}
	key := cacheKey(text)
	if cached, ok := s.cache.Load(key); ok {
		stored := cached.([]float32)
		vec := make([]float32, len(stored))
		copy(vec, stored)
		return vec, nil
	}
	subCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	vec, err := s.embedder.Embed(subCtx, text)
	if err != nil {
		return nil, err
	}
	normalize(vec)
	s.cache.Store(key, append([]float32(nil), vec...))
	return vec, nil
}

func normalize(vec []float32) {
	var norm float64
	for _, v := range vec {
		norm += float64(v * v)
	}
	if norm == 0 {
		return
	}
	length := math.Sqrt(norm)
	if length == 0 {
		return
	}
	factor := 1 / float32(length)
	for i := range vec {
		vec[i] *= factor
	}
}

func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}

func cacheKey(text string) string {
	trimmed := strings.TrimSpace(text)
	sum := sha1.Sum([]byte(trimmed))
	return hex.EncodeToString(sum[:])
}

const defaultSummaryPrompt = "Summarize the conversation. Focus on intent, commitments, and unresolved items."
